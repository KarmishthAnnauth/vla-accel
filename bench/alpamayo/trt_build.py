"""Build the TensorRT engines FastAlpamayo can use for the vision encoder and the prefill.

  vit_fp16.engine           the whole vision encoder for the live image layout
  llm/layer_XX_fp16.engine  one engine per VLM decoder layer over the prefill
                            window: (h, cos, sin) -> (h, keys, values)

The decoder is built layer by layer rather than as one 14 GiB engine: every
layer builds in the memory of one, the timing cache carries over, and each is
checked on the hidden states the real model produces for a driving prompt
before the next is built.  The RMSNorms are pinned to fp32 (their x^2 overflows
fp16 on real activations).

    ./run.sh trt_build.py [--layers 0 36] [--level 3] [--rebuild] [--skip-vit]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

import common
from alpamayo_speedups import DRAFT_CAP
from trt_probe import OUT, TrtRunner, Vit, bench, rel, repeat_kv


class DecoderLayer(torch.nn.Module):
    """One Qwen3VLTextDecoderLayer over a causal window: (h, cos, sin) -> (h, keys, values)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = copy.deepcopy(layer).float()

    def forward(self, h, cos, sin):
        layer, at = self.layer, self.layer.self_attn
        b, s, _ = h.shape
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        v = at.v_proj(x).view(b, s, -1, at.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rep = q.shape[1] // k.shape[1]
        a = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep), is_causal=True)
        h = h + at.o_proj(a.transpose(1, 2).reshape(b, s, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v


def build_engine(onnx_path: str, engine_path: str, level: int, fp32_names=(),
                 int8: bool = False) -> float:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.builder_optimization_level = level
    config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    pinned = 0
    if fp32_names:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if (any(s in layer.name for s in fp32_names) and layer.num_outputs
                    and layer.get_output(0).dtype == trt.float32
                    and layer.type not in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT)):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                pinned += 1
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
    return time.time() - t0


@torch.no_grad()
def real_activations(model, data, rows):
    """Input and output of every decoder layer, and the rotary tables, for the
    prompt padded to the prefill window, from the unmodified model."""
    lm = model.vlm.model.language_model
    tok = data["tokenized_data"]
    ids = model.fuse_traj_tokens(tok["input_ids"], {
        "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]})
    pad = torch.full((1, rows - ids.shape[1]), model.tokenizer.pad_token_id, device=ids.device)
    ids = torch.cat([ids, pad], dim=1)
    rec = {"in": [], "out": []}
    hooks = []
    for layer in lm.layers:
        def pre(_m, args, kwargs, rec=rec):
            rec["in"].append(args[0].clone())
            rec["pos"] = kwargs["position_embeddings"]

        def post(_m, _a, _k, out, rec=rec):
            rec["out"].append((out[0] if isinstance(out, tuple) else out).clone())

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.register_forward_hook(post, with_kwargs=True))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        pixel_values=tok["pixel_values"], image_grid_thw=tok["image_grid_thw"],
                        use_cache=False)
    for h in hooks:
        h.remove()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=None)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-vit", action="store_true")
    args = ap.parse_args()

    model, processor = common.load_model()
    lm = model.vlm.model.language_model
    data = common.make_inputs(processor, common.make_payload(1))
    tok = data["tokenized_data"]
    rows = tok["input_ids"].shape[1] + DRAFT_CAP
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest.update({"rows": rows, "grid": tok["image_grid_thw"].flatten().tolist()})

    vit_engine = os.path.join(OUT, "vit_fp16.engine")
    if not args.skip_vit:
        if args.rebuild or not os.path.exists(vit_engine):
            probe_engine = os.path.join(OUT, "vit", "vit_fp16.engine")
            if os.path.exists(probe_engine) and not args.rebuild:
                shutil.copy(probe_engine, vit_engine)
            else:
                work = os.path.join(OUT, "vit")
                os.makedirs(work, exist_ok=True)
                vit = Vit(model.vlm.model.visual, tok["image_grid_thw"]).cuda().eval()
                with torch.no_grad():
                    torch.onnx.export(vit, (tok["pixel_values"].float(),), os.path.join(work, "vit.onnx"),
                                      opset_version=17, dynamo=False, input_names=["pixels"],
                                      output_names=[f"out{i}" for i in range(4)])
                del vit
                took = build_engine(os.path.join(work, "vit.onnx"), vit_engine, args.level)
                print(f"vit engine built in {took:.0f} s", flush=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = model.vlm.model.visual(tok["pixel_values"], grid_thw=tok["image_grid_thw"])[0]
        runner = TrtRunner(vit_engine)
        runner.inputs["pixels"].copy_(tok["pixel_values"])
        ms = bench(runner)
        torch.cuda.synchronize()
        print(f"vit: {ms:.1f} ms, rel-L2 vs the bf16 model {rel(runner.outputs['out0'], ref):.4f}",
              flush=True)
        manifest["vit"] = {"ms": ms}
        del runner

    rec = real_activations(model, data, rows)
    cos, sin = (t.float() for t in rec["pos"])
    lo, hi = args.layers or (0, len(lm.layers))
    os.makedirs(os.path.join(OUT, "llm"), exist_ok=True)
    work = os.path.join(OUT, "llm", "onnx")
    total = 0.0
    for i in range(lo, hi):
        engine = os.path.join(OUT, "llm", f"layer_{i:02d}_fp16.engine")
        h_in, h_out = rec["in"][i].float(), rec["out"][i].float()
        took = 0.0
        if args.rebuild or not os.path.exists(engine):
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work)
            module = DecoderLayer(lm.layers[i]).cuda().eval()
            with torch.no_grad():
                torch.onnx.export(module, (h_in, cos, sin), os.path.join(work, "layer.onnx"),
                                  opset_version=17, dynamo=False, input_names=["h", "cos", "sin"],
                                  output_names=["h_out", "k", "v"])
            del module
            torch.cuda.empty_cache()
            took = build_engine(os.path.join(work, "layer.onnx"), engine, args.level,
                                fp32_names=("norm",))
            shutil.rmtree(work, ignore_errors=True)
        runner = TrtRunner(engine)
        runner.inputs["h"].copy_(h_in)
        runner.inputs["cos"].copy_(cos)
        runner.inputs["sin"].copy_(sin)
        ms = bench(runner, n=10, warm=3)
        torch.cuda.synchronize()
        out = runner.outputs["h_out"]
        finite = bool(torch.isfinite(out).all())
        err = rel(out, h_out)
        total += ms
        print(f"layer {i:02d}: {ms:6.2f} ms, rel-L2 vs the bf16 model's layer output {err:.4f}, "
              f"|h| max {float(h_out.abs().max()):.0f}, finite {finite}"
              + (f", built in {took:.0f} s" if took else ""), flush=True)
        manifest.setdefault("layers", {})[str(i)] = {"ms": ms, "rel": err, "finite": finite}
        del runner
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"layers {lo}..{hi - 1}: {total:.0f} ms in total")


if __name__ == "__main__":
    main()
