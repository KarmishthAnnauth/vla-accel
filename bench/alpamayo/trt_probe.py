"""Would TensorRT (fp16) run Alpamayo's transformers faster than the PyTorch fast path?

Builds three engines at the fast path's shapes and times them:

  vit      the whole vision encoder, 4 x 720 patches            (fast path: 155 ms)
  layer    ONE decoder layer of the VLM, 860-row prefill window (fast path: 14.6 ms = 528 / 36)
  expert   ONE expert layer, 64 rows over a 972-row KV buffer   (fast path: 1.04 ms = 188 / 5 / 36)

One layer stands in for the 36: they are identical in shape, and a single
layer builds in minutes where the 8B decoder needs a 14 GiB engine.

    ./run.sh trt_probe.py vit layer expert [--level 3] [--rebuild]
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

import common

OUT = os.environ.get("ALPAMAYO_TRT_DIR", "/media/USER/EXTSSD/alpamayo-1.5-trt")


def build_engine(onnx_path: str, engine_path: str, level: int, half_inputs=()) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    for i in range(network.num_inputs):
        if network.get_input(i).name in half_inputs:
            network.get_input(i).dtype = trt.float16
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.builder_optimization_level = level
    config.set_flag(trt.BuilderFlag.FP16)
    cache_path = os.path.join(OUT, "timing.cache")
    cache = config.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(blob)
    with open(cache_path, "wb") as f:
        f.write(config.get_timing_cache().serialize())
    print(f"built {engine_path} ({os.path.getsize(engine_path) >> 20} MiB) in {time.time() - t0:.0f} s",
          flush=True)


class TrtRunner:
    """An engine with static shapes, its inputs and outputs bound to torch tensors."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                  trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32,
                  trt.DataType.BOOL: torch.bool}
        self.inputs, self.outputs = {}, {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            t = torch.zeros(tuple(self._engine.get_tensor_shape(name)), device="cuda",
                            dtype=dtypes[self._engine.get_tensor_dtype(name)])
            is_in = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_in else self.outputs)[name] = t
            self._context.set_tensor_address(name, t.data_ptr())
        self._stream = torch.cuda.Stream()

    def __call__(self):
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        current.wait_stream(self._stream)
        return self.outputs


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def repeat_kv(x: torch.Tensor, rep: int) -> torch.Tensor:
    b, g, s, d = x.shape
    return x[:, :, None].expand(b, g, rep, s, d).reshape(b, g * rep, s, d)


class Vit(torch.nn.Module):
    """FastAlpamayo._vit_fn in fp32: pixels [N, 1536] -> merged + deepstack features."""

    def __init__(self, visual, grid):
        super().__init__()
        self.vis = copy.deepcopy(visual).float()
        sizes = (grid[:, 1] * grid[:, 2]).tolist()
        self.n, self.s = len(sizes), sizes[0]
        self.register_buffer("pos", self.vis.fast_pos_embed_interpolate(grid).float())
        rot = self.vis.rot_pos_emb(grid)
        rot = torch.cat((rot, rot), dim=-1)
        self.register_buffer("cos", rot.cos().float())
        self.register_buffer("sin", rot.sin().float())

    def forward(self, pixels):
        vis, n, s = self.vis, self.n, self.s
        x = vis.patch_embed(pixels) + self.pos
        heads = vis.blocks[0].attn.num_heads
        deep = []
        for i, blk in enumerate(vis.blocks):
            qkv = blk.attn.qkv(blk.norm1(x)).reshape(n * s, 3, heads, -1).permute(1, 0, 2, 3)
            q, k = apply_rotary_pos_emb_vision(qkv[0], qkv[1], self.cos, self.sin)
            q, k, v = (t.reshape(n, s, heads, -1).transpose(1, 2) for t in (q, k, qkv[2]))
            a = F.scaled_dot_product_attention(q, k, v)
            x = x + blk.attn.proj(a.transpose(1, 2).reshape(n * s, -1))
            x = x + blk.mlp(blk.norm2(x))
            if i in vis.deepstack_visual_indexes:
                deep.append(vis.deepstack_merger_list[vis.deepstack_visual_indexes.index(i)](x))
        return (vis.merger(x), *deep)


class DecoderLayer(torch.nn.Module):
    """One Qwen3VLTextDecoderLayer over a causal window: h -> (h, keys, values)."""

    def __init__(self, layer, rotary, rows):
        super().__init__()
        self.layer = copy.deepcopy(layer).float()
        pos = torch.arange(rows, device="cuda").view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rotary(torch.zeros(1, 1, 1, device="cuda"), pos)
        self.register_buffer("cos", cos.float())
        self.register_buffer("sin", sin.float())

    def forward(self, h):
        layer, at = self.layer, self.layer.self_attn
        b, s, _ = h.shape
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        v = at.v_proj(x).view(b, s, -1, at.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        rep = q.shape[1] // k.shape[1]
        a = F.scaled_dot_product_attention(
            q, repeat_kv(k, rep), repeat_kv(v, rep), is_causal=True)
        h = h + at.o_proj(a.transpose(1, 2).reshape(b, s, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v


class ExpertLayer(torch.nn.Module):
    """One expert layer: 64 action rows attending to a KV buffer plus themselves."""

    def __init__(self, layer, rotary, rows):
        super().__init__()
        self.layer = copy.deepcopy(layer).float()
        pos = (torch.arange(rows, device="cuda") + 900).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rotary(torch.zeros(1, 1, 1, device="cuda"), pos)
        self.register_buffer("cos", cos.float())
        self.register_buffer("sin", sin.float())

    def forward(self, h, kb, vb, mask):
        layer, at = self.layer, self.layer.self_attn
        b, s, _ = h.shape
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        v = at.v_proj(x).view(b, s, -1, at.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        rep = q.shape[1] // k.shape[1]
        keys = repeat_kv(torch.cat([kb, k], dim=2), rep)
        values = repeat_kv(torch.cat([vb, v], dim=2), rep)
        a = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        h = h + at.o_proj(a.transpose(1, 2).reshape(b, s, -1))
        return h + layer.mlp(layer.post_attention_layernorm(h))


class ExpertLayerGrouped(ExpertLayer):
    """ExpertLayer without expanding the KV buffer to all heads (as the
    PyTorch fast path attends), meant for fp16 KV inputs."""

    def forward(self, h, kb, vb, mask):
        layer, at = self.layer, self.layer.self_attn
        b, s, _ = h.shape
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        v = at.v_proj(x).view(b, s, -1, at.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        g = k.shape[1]
        keys, values = torch.cat([kb, k], dim=2), torch.cat([vb, v], dim=2)
        scores = torch.matmul(q.reshape(b, g, -1, at.head_dim), keys.transpose(-1, -2)) * at.scaling
        probs = torch.softmax(scores + mask, dim=-1)
        a = torch.matmul(probs, values).view(b, -1, s, at.head_dim)
        h = h + at.o_proj(a.transpose(1, 2).reshape(b, s, -1))
        return h + layer.mlp(layer.post_attention_layernorm(h))


def probe(name, module, example, level, rebuild, fast_ms, bf16_ref=None, half_inputs=()):
    """Export -> engine -> time and compare with the fp32 module."""
    work = os.path.join(OUT, name)
    os.makedirs(work, exist_ok=True)
    onnx_path, engine_path = os.path.join(work, f"{name}.onnx"), os.path.join(work, f"{name}_fp16.engine")
    module = module.cuda().eval()
    names = list(example)
    args = tuple(example.values())
    with torch.no_grad():
        ref = module(*args)
        ref = ref if isinstance(ref, tuple) else (ref,)
        if rebuild or not os.path.exists(engine_path):
            torch.onnx.export(module, args, onnx_path, opset_version=17, dynamo=False,
                              input_names=names, output_names=[f"out{i}" for i in range(len(ref))],
                              do_constant_folding=True)
            size = sum(os.path.getsize(os.path.join(work, f)) for f in os.listdir(work)
                       if not f.endswith(".engine"))
            print(f"[{name}] exported ONNX, {size >> 20} MiB", flush=True)
            build_engine(onnx_path, engine_path, level, half_inputs)
        torch_ms = bench(lambda: module(*args), n=5, warm=2)
    del module
    torch.cuda.empty_cache()
    runner = TrtRunner(engine_path)
    for key, value in example.items():
        runner.inputs[key].copy_(value)
    ms = bench(runner)
    torch.cuda.synchronize()
    errs = [rel(runner.outputs[f"out{i}"], r) for i, r in enumerate(ref)]
    line = (f"[{name}] TensorRT fp16 {ms:.2f} ms | PyTorch fast path {fast_ms:.2f} ms | fp32 eager "
            f"{torch_ms:.2f} ms | rel-L2 vs fp32 {max(errs):.4f}")
    if bf16_ref is not None:
        line += f" (the bf16 fast path differs from fp32 by {bf16_ref:.4f})"
    print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["vit", "layer", "expert", "expert2"])
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    model, processor = common.load_model()
    lm, ex = model.vlm.model.language_model, model.expert
    g = torch.Generator(device="cuda").manual_seed(0)

    if "vit" in args.what:
        tok = common.make_inputs(processor, common.make_payload(1))["tokenized_data"]
        vit = Vit(model.vlm.model.visual, tok["image_grid_thw"])
        pixels = tok["pixel_values"].float()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf = model.vlm.model.visual(tok["pixel_values"], grid_thw=tok["image_grid_thw"])[0]
            bf_err = rel(bf, vit.cuda()(pixels)[0])
        probe("vit", vit, {"pixels": pixels}, args.level, args.rebuild, 155.0, bf_err)

    if "layer" in args.what:
        rows = 860
        h = torch.randn(1, rows, lm.config.hidden_size, device="cuda", generator=g)
        probe("layer", DecoderLayer(lm.layers[5], lm.rotary_emb, rows), {"h": h},
              args.level, args.rebuild, 528.0 / 36)

    if "expert2" in args.what:
        rows, buf = 64, 908
        h = torch.randn(1, rows, ex.config.hidden_size, device="cuda", generator=g)
        kb = torch.randn(1, 8, buf, 128, device="cuda", generator=g)
        vb = torch.randn(1, 8, buf, 128, device="cuda", generator=g)
        mask = torch.zeros(1, 1, 1, buf + rows, device="cuda")
        mask[..., 850:buf] = float("-inf")
        probe("expert2", ExpertLayerGrouped(ex.layers[5], ex.rotary_emb, rows),
              {"h": h, "kb": kb, "vb": vb, "mask": mask}, args.level, args.rebuild,
              188.0 / 5 / 36, half_inputs=("kb", "vb"))
        probe("expert2_level5", ExpertLayerGrouped(ex.layers[5], ex.rotary_emb, rows),
              {"h": h, "kb": kb, "vb": vb, "mask": mask}, 5, args.rebuild,
              188.0 / 5 / 36, half_inputs=("kb", "vb"))

    if "expert" in args.what:
        rows, buf = 64, 908
        h = torch.randn(1, rows, ex.config.hidden_size, device="cuda", generator=g)
        kb = torch.randn(1, 8, buf, 128, device="cuda", generator=g)
        vb = torch.randn(1, 8, buf, 128, device="cuda", generator=g)
        mask = torch.zeros(1, 1, 1, buf + rows, device="cuda")
        mask[..., 850:buf] = float("-inf")
        probe("expert", ExpertLayer(ex.layers[5], ex.rotary_emb, rows),
              {"h": h, "kb": kb, "vb": vb, "mask": mask}, args.level, args.rebuild, 188.0 / 5 / 36)


if __name__ == "__main__":
    main()
