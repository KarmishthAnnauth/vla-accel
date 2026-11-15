"""W8A8 INT8 sweep for the ORION LLM and the SmoothQuant calibration stats file.

`--phase 0 --save-stats engines/llm_act_stats.pt` only writes the stats used by bench_b2d.py.
Phases 1/2 report waypoint-feature error and max trajectory deviation per recipe vs fp16 (eager).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from vla_accel.common import b2d, jetson
from vla_accel.orion import inputs, model as om, speedups as sp

T0 = time.time()


def log(msg):
    """Timestamped log line."""
    print(f"[sweep +{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    """Calibrate, optionally save stats, and evaluate INT8 recipes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=om.CONFIG)
    ap.add_argument("--checkpoint", default=om.CHECKPOINT)
    inputs.add_b2d_args(ap, frames=3, split="calib")
    ap.add_argument("--alphas", default="0.5,0.65,0.8")
    ap.add_argument("--phase", type=int, default=1, help="0: stats only; 1: alpha/group/layer sweep; 2: high-alpha follow-up")
    ap.add_argument("--save-stats", default="", help=f"write calibration stats here (bench_b2d default {om.INT8_STATS})")
    args = ap.parse_args()
    args.b2d_dir = os.path.abspath(args.b2d_dir)
    save_stats = os.path.abspath(args.save_stats) if args.save_stats else ""
    jetson.warn_if_unpinned()

    clip = inputs.select_clips(args.b2d_dir, args.clips, args.split)[0]
    idxs = b2d.frame_indices(clip, args.frames, args.stride, args.start)
    frames = [(i, *inputs.load_frame(clip, i)) for i in idxs]
    log(f"calibration clip {os.path.basename(clip)}: frames {idxs}")

    model, cfg = om.build_model(args.config, args.checkpoint, "fp16")
    pipeline = om.build_pipeline(cfg)
    sp.merge_lora(model, log); sp.patch_llm_flash_attention(model, log); sp.patch_vit_blocks(model, log)
    sp.slice_map_head_one2one(model, log)
    om.custom_wrap_fp16_model(model)
    log("model ready")

    feats = []
    orig_inf = model.lm_head.inference_ego

    def inf_hook(*a, **kw):
        out = orig_inf(*a, **kw); feats.append(out.detach().float().clone()); return out
    model.lm_head.inference_ego = inf_hook

    def run(seed=1234):
        model.test_flag = False; feats.clear(); trajs = []
        for f, (idx, imgs, anno) in enumerate(frames):
            b, _ = inputs.build_batch(pipeline, imgs, anno, idx); torch.manual_seed(seed + f)
            with torch.inference_mode(): out = model(b, return_loss=False)
            trajs.append(out[0]["pts_bbox"]["ego_fut_preds"].float().cpu().numpy())
        return list(feats), trajs

    ref_f, ref_t = run()
    seed_spread = max(float(np.abs(a - b).max()) for s in (4321, 999) for a, b in zip(run(s)[1], ref_t))
    log(f"fp16 reference: seed-to-seed trajectory spread max {seed_spread:.3f} m")

    llama = model.lm_head.get_model()
    layers = llama.layers
    saved = [{k: v.detach().to("cpu", copy=True) for k, v in l.state_dict().items()} for l in layers]
    orig_linears = [{n: getattr(p, n) for p in (l.self_attn, l.mlp) for n in sp.LLM_INT8_TARGETS if hasattr(p, n)} for l in layers]

    def restore():
        for l, sd, lins in zip(layers, saved, orig_linears):
            for n, lin in lins.items():
                parent = l.self_attn if n in ("q_proj", "k_proj", "v_proj", "o_proj") else l.mlp
                setattr(parent, n, lin)
            l.load_state_dict(sd)
        torch.cuda.empty_cache()

    stats = sp.collect_llm_act_stats(model, lambda: run(), sp.LLM_INT8_TARGETS)
    if save_stats:
        os.makedirs(os.path.dirname(save_stats), exist_ok=True)
        sp.save_llm_act_stats(stats, save_stats); log(f"saved calibration stats -> {save_stats}")
    if args.phase == 0:
        return
    ratios = [float(stats[(i, "q_proj")].max() / stats[(i, "q_proj")].median()) for i in range(len(layers))]
    log(f"outlier ratio (max/median |x| into q_proj) per layer: " + " ".join(f"{r:.0f}" for r in ratios))

    def quantize(targets, skip_layers=()):
        n = 0
        for i, l in enumerate(layers):
            if i in skip_layers: continue
            for parent in (l.self_attn, l.mlp):
                for name in targets:
                    lin = getattr(parent, name, None)
                    if isinstance(lin, torch.nn.Linear):
                        setattr(parent, name, sp.Int8Linear(lin)); n += 1
        return n

    def evaluate(tag, alpha, targets, skip=()):
        restore()
        if alpha > 0: sp.smooth_llm(model, stats, alpha)
        n = quantize(targets, skip)
        f, t = run()
        ferr = max(float((a - b).norm() / b.norm()) for a, b in zip(f, ref_f))
        dtraj = max(float(np.abs(a - b).max()) for a, b in zip(t, ref_t))
        log(f"{tag:44s} n={n:3d}  feat rel err {ferr:.4f}  max|dtraj| {dtraj:.3f} m")

    ALL = sp.LLM_INT8_TARGETS; ATTN = ("q_proj", "k_proj", "v_proj", "o_proj"); MLP = ("gate_proj", "up_proj", "down_proj")
    NL = len(layers)
    evaluate("fp16 restore sanity (no quant)", 0.0, ())
    if args.phase == 2:
        skip4 = (0, 1, NL - 2, NL - 1)
        for a in (0.85, 0.9, 0.95, 1.0):
            evaluate(f"all, alpha={a}, skip 0-1 + last 2", a, ALL, skip=skip4)
        for a in (0.8, 0.9):
            evaluate(f"mlp only, alpha={a}, skip 0-1 + last 2", a, MLP, skip=skip4)
            evaluate(f"attn only, alpha={a}, skip 0-1 + last 2", a, ATTN, skip=skip4)
        evaluate("all, alpha=0.9, skip first 4 + last 4", 0.9, ALL, skip=(0, 1, 2, 3, NL - 4, NL - 3, NL - 2, NL - 1))
        return
    for a in [float(x) for x in args.alphas.split(",")]:
        evaluate(f"all, alpha={a}", a, ALL)
    evaluate("attn only, alpha=0.65", 0.65, ATTN)
    evaluate("mlp only, alpha=0.65", 0.65, MLP)
    evaluate("all but down_proj, alpha=0.65", 0.65, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"))
    evaluate("all, alpha=0.65, skip layers 0-1", 0.65, ALL, skip=(0, 1))
    evaluate("all, alpha=0.65, skip last 2", 0.65, ALL, skip=(NL - 2, NL - 1))
    evaluate("all, alpha=0.65, skip 0-1 + last 2", 0.65, ALL, skip=(0, 1, NL - 2, NL - 1))
    evaluate("all, alpha=0.8, skip 0-1 + last 2", 0.8, ALL, skip=(0, 1, NL - 2, NL - 1))
    worst = sorted(range(NL), key=lambda i: -ratios[i])[:4]
    evaluate(f"all, alpha=0.65, skip worst-outlier {worst}", 0.65, ALL, skip=tuple(worst))


if __name__ == "__main__":
    main()
