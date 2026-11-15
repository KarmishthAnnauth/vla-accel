"""ORION default and opt-in setups (INT8 LLM, lite vision, both) on Bench2Drive clip sequences.

One setup per process; planner noise is seeded per frame identically, so `--compare`
reports per-frame trajectory deviation from the default setup.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from vla_accel.common import jetson
from vla_accel.orion import inputs, model as om, speedups as sp

SETUPS = ["default", "noise", "int8", "lite", "both"]
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped log line."""
    print(f"[b2d +{time.time() - T0:7.1f}s] {msg}", flush=True)


def run(args) -> None:
    """Build one setup, run every clip as a sequence and save trajectories and timings."""
    import torch

    jetson.warn_if_unpinned()
    plan = inputs.frame_plan(args)
    out = os.path.abspath(args.out)
    int8 = args.setup in ("int8", "both")
    lite = args.setup in ("lite", "both")
    vit_size = 512 if lite else 640

    log(f"setup={args.setup} building model ...")
    model, cfg = om.build_model(args.config, args.checkpoint, "fp16")
    log("model loaded")

    sp.merge_lora(model, log)
    sp.patch_llm_flash_attention(model, log)
    sp.patch_vit_blocks(model, log)
    if vit_size != 640:
        sp.set_vit_input_size(model, vit_size, log)
    if int8:
        sp.apply_llm_int8(model, args.stats, 0.8, [0, 1, 30, 31], log, ["gate_proj", "up_proj", "down_proj"])
    else:
        sp.transpose_llm_down_proj(model, log)
    sp.compile_submodules(model, ["heads", "llm", "vit"], "default", log)
    sp.slice_map_head_one2one(model, log)
    stagger = sp.install_staggered_views(model, 2, log) if lite else None
    om.custom_wrap_fp16_model(model)
    pipeline = om.build_pipeline(cfg, vit_size, workers=6)
    log(f"{len(plan)} clips, {sum(len(i) for _, i in plan)} frames, stride {args.stride}")

    def forward(batch, seed):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            o = model(batch, return_loss=False)
        torch.cuda.synchronize()
        return o[0]["pts_bbox"]["ego_fut_preds"].float().cpu().numpy(), (time.perf_counter() - t) * 1e3

    c0, i0 = plan[0]
    model.test_flag = False
    if stagger is not None:
        stagger.reset()
    for k in range(4):
        imgs, anno = inputs.load_frame(c0, i0[min(k, len(i0) - 1)])
        _, ms = forward(inputs.build_batch(pipeline, imgs, anno, k)[0], 0)
        log(f"warm-up {k}: {ms:.0f} ms")

    trajs, fwd, prep, keys = [], [], [], []
    for c, idxs in plan:
        model.test_flag = False
        if stagger is not None:
            stagger.reset()
        name = os.path.basename(c)
        for idx in idxs:
            t = time.perf_counter()
            imgs, anno = inputs.load_frame(c, idx)
            batch, _ = inputs.build_batch(pipeline, imgs, anno, idx)
            p = (time.perf_counter() - t) * 1e3
            traj, ms = forward(batch, 1234 + args.seed_offset + len(keys))
            trajs.append(traj); fwd.append(ms); prep.append(p); keys.append(f"{name}:{idx}")
        log(f"{name}: {len(idxs)} frames, forward mean {np.mean(fwd[-len(idxs):]):.0f} ms")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, trajs=np.stack(trajs), forward_ms=np.array(fwd), prep_ms=np.array(prep),
             keys=np.array(keys), setup=args.setup)
    log(f"setup={args.setup}: forward {np.mean(fwd):.0f} ms (sd {np.std(fwd):.0f}), "
        f"prep {np.mean(prep):.0f} ms, saved {out}")


def compare(folder: str) -> None:
    """Print time and deviation from default.npz for every <setup>.npz in folder."""
    ref = np.load(os.path.join(folder, "default.npz"))
    print(f"{'setup':10s} {'forward ms':>10s} {'prep ms':>8s} {'mean dev m':>11s} {'median':>8s} {'p95':>7s} {'max m':>7s}")
    for s in SETUPS:
        p = os.path.join(folder, f"{s}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        assert list(d["keys"]) == list(ref["keys"]), s
        dev = np.linalg.norm(d["trajs"] - ref["trajs"], axis=-1).reshape(len(d["keys"]), -1).max(axis=1)
        print(f"{s:10s} {d['forward_ms'].mean():10.0f} {d['prep_ms'].mean():8.0f} {dev.mean():11.3f} "
              f"{np.median(dev):8.3f} {np.percentile(dev, 95):7.3f} {dev.max():7.3f}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", choices=["default", "int8", "lite", "both"], default="default")
    ap.add_argument("--config", default=om.CONFIG)
    ap.add_argument("--checkpoint", default=om.CHECKPOINT)
    ap.add_argument("--stats", default=om.INT8_STATS, help="SmoothQuant stats from int8_sweep.py --save-stats")
    inputs.add_b2d_args(ap, frames=15)
    ap.add_argument("--seed-offset", type=int, default=0, help="e.g. 1000 for the 'noise' run")
    ap.add_argument("--out", default="", help="output .npz, e.g. results/default.npz")
    ap.add_argument("--compare", default="", help="folder of <setup>.npz files to compare")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    else:
        if not a.out:
            ap.error("--out is required")
        run(a)


if __name__ == "__main__":
    main()
