"""Build the fp16 TensorRT engines FastAlpamayo loads from ``trt_dir``: the ViT and one engine per decoder layer.

Each layer engine is checked against the bf16 model's activations on a Bench2Drive prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from vla_accel.alpamayo import common
from vla_accel.alpamayo.speedups import DRAFT_CAP
from vla_accel.alpamayo.trt_common import (OUT, DecoderLayer, TrtRunner, Vit, bench, build_engine,
                                           real_activations, rel)


def main() -> None:
    """Build and check the ViT and the layer engines."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=None)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-vit", action="store_true")
    common.add_b2d_args(ap, frames=1)
    args = ap.parse_args()

    model, processor = common.load_model()
    lm = model.vlm.model.language_model
    data = common.scenes(processor, args)[0]
    tok = data["tokenized_data"]
    rows = tok["input_ids"].shape[1] + DRAFT_CAP
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest.update({"rows": rows, "grid": tok["image_grid_thw"].flatten().tolist()})

    vit_engine = os.path.join(OUT, "vit_fp16.engine")
    if not args.skip_vit:
        if args.rebuild or not os.path.exists(vit_engine):
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
