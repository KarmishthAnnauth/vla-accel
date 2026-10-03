#!/usr/bin/env python3
"""Does TensorRT 10.3 on this Orin run a Q/DQ-quantised MatMul in INT8, and is
it faster than fp16?  A stack of ViT-MLP-shaped linears (1024 -> 4096 -> 1024),
9225 rows, with and without Q/DQ nodes.  Also torch's own int8 GEMM."""
import json, os, time
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import tensorrt as trt
import torch

ROWS, D, H, BLOCKS = 9225, 1024, 4096, 6
rng = np.random.default_rng(0)

def make(qdq: bool, per_channel: bool = True):
    nodes, inits = [], []
    cur = "x"
    def qdq_act(name, scale=0.05):
        inits.append(numpy_helper.from_array(np.array(scale, np.float32), name + "_s"))
        inits.append(numpy_helper.from_array(np.array(0, np.int8), name + "_z"))
        nodes.append(helper.make_node("QuantizeLinear", [name, name + "_s", name + "_z"], [name + "_q"]))
        nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name + "_dq"]))
        return name + "_dq"
    for b in range(BLOCKS):
        for j, (i_, o_) in enumerate(((D, H), (H, D))):
            w = (rng.standard_normal((i_, o_)) * 0.02).astype(np.float32)
            wn = f"w{b}_{j}"
            inits.append(numpy_helper.from_array(w, wn))
            inits.append(numpy_helper.from_array(np.zeros(o_, np.float32), wn + "_b"))
            a = cur
            if qdq:
                a = qdq_act(cur)
                if per_channel:
                    s = (np.abs(w).max(axis=0) / 127).astype(np.float32)
                    inits.append(numpy_helper.from_array(s, wn + "_s")); inits.append(numpy_helper.from_array(np.zeros(o_, np.int8), wn + "_z"))
                    nodes.append(helper.make_node("QuantizeLinear", [wn, wn + "_s", wn + "_z"], [wn + "_q"], axis=1))
                    nodes.append(helper.make_node("DequantizeLinear", [wn + "_q", wn + "_s", wn + "_z"], [wn + "_dq"], axis=1))
                else:
                    inits.append(numpy_helper.from_array(np.array(np.abs(w).max() / 127, np.float32), wn + "_s")); inits.append(numpy_helper.from_array(np.array(0, np.int8), wn + "_z"))
                    nodes.append(helper.make_node("QuantizeLinear", [wn, wn + "_s", wn + "_z"], [wn + "_q"]))
                    nodes.append(helper.make_node("DequantizeLinear", [wn + "_q", wn + "_s", wn + "_z"], [wn + "_dq"]))
                wn_use = wn + "_dq"
            else:
                wn_use = wn
            nodes.append(helper.make_node("MatMul", [a, wn_use], [f"m{b}_{j}"]))
            nodes.append(helper.make_node("Add", [f"m{b}_{j}", wn + "_b"], [f"a{b}_{j}"]))
            cur = f"a{b}_{j}"
            if j == 0:
                nodes.append(helper.make_node("Relu", [cur], [f"r{b}"])); cur = f"r{b}"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ROWS, D])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, ROWS, D])], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]).SerializeToString()

logger = trt.Logger(trt.Logger.ERROR)
def build_and_time(name, blob, int8):
    builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
    if not parser.parse(blob):
        print(name, "parse failed:", parser.get_error(0)); return
    cfg = builder.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if int8: cfg.set_flag(trt.BuilderFlag.INT8)
    eng_blob = builder.build_serialized_network(net, cfg)
    if eng_blob is None:
        print(name, "build failed"); return
    eng = trt.Runtime(logger).deserialize_cuda_engine(eng_blob); ctx = eng.create_execution_context()
    x = torch.randn(1, ROWS, D, device="cuda"); y = torch.zeros(1, ROWS, D, device="cuda")
    ctx.set_tensor_address("x", x.data_ptr()); ctx.set_tensor_address("y", y.data_ptr())
    s = torch.cuda.Stream()
    for _ in range(5): ctx.execute_async_v3(s.cuda_stream)
    s.synchronize(); t0 = time.perf_counter()
    for _ in range(20): ctx.execute_async_v3(s.cuda_stream)
    s.synchronize(); ms = (time.perf_counter() - t0) / 20 * 1e3
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    kinds = {}
    for l in info["Layers"]:
        k = (l.get("LayerType"), (l.get("Outputs") or [{}])[0].get("Format/Datatype", "?"))
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{name:34s} {ms:6.2f} ms for {2 * BLOCKS} linears = {ms / (2 * BLOCKS):.2f} ms each | " + ", ".join(f"{k[0]}:{k[1]} x{v}" for k, v in kinds.items()))

build_and_time("fp16, no Q/DQ", make(False), False)
build_and_time("Q/DQ per-channel weights, INT8", make(True, True), True)
build_and_time("Q/DQ per-tensor weights, INT8", make(True, False), True)

bf = torch.bfloat16
x = torch.randn(ROWS, D, device="cuda", dtype=bf); w1 = torch.randn(D, H, device="cuda", dtype=bf); h = torch.randn(ROWS, H, device="cuda", dtype=bf); w2 = torch.randn(H, D, device="cuda", dtype=bf)
xi = torch.randint(-127, 127, (ROWS, D), device="cuda", dtype=torch.int8); w1i = torch.randint(-127, 127, (D, H), device="cuda", dtype=torch.int8)
hi = torch.randint(-127, 127, (ROWS, H), device="cuda", dtype=torch.int8); w2i = torch.randint(-127, 127, (H, D), device="cuda", dtype=torch.int8)
w1i_t = w1i.t().contiguous(); w2i_t = w2i.t().contiguous()
def t(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
print(f"torch bf16 mm          1024->4096 {t(lambda: x @ w1):5.2f} ms   4096->1024 {t(lambda: h @ w2):5.2f} ms")
for name, a, b in (("int_mm, weight [K,N]", w1i, w2i), ("int_mm, weight [N,K].t()", w1i_t.t(), w2i_t.t())):
    try:
        print(f"torch._{name:26s} 1024->4096 {t(lambda: torch._int_mm(xi, a)):5.2f} ms   4096->1024 {t(lambda: torch._int_mm(hi, b)):5.2f} ms")
    except Exception as e:
        print(name, "failed:", str(e)[:150])
