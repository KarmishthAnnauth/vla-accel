#!/usr/bin/env python3
"""Sweep W8A8 INT8 configurations for the ORION LLM in ONE process.

Loads the model once, keeps fp16 copies of the decoder weights, and for each
configuration: restores fp16, optionally SmoothQuant-rescales with calibration
stats, quantises the chosen projections / layers, and measures on 3 CARLA
frames (a) the relative error of the LLM's waypoint feature (the vector the
planner consumes) and (b) the max trajectory deviation, both against the fp16
model. Also reports the fp16 model's own seed-to-seed trajectory spread as the
yardstick. Eager mode (accuracy only; speed is known from bench_orion.py).

  docker exec orion_ros bash -c 'source /opt/ros/humble/setup.bash;
      source /opt/orion_ws/install/setup.bash; cd /root/Orion;
      python3 /benchmarking/alpamayo-autoware/src/orion_ros/tools/int8_sweep.py'
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T0 = time.time()


def log(msg):
    print(f"[sweep +{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/benchmarking/imgdiag/frame_0.png,/benchmarking/imgdiag/frame_1.png,/benchmarking/imgdiag/frame_2.png")
    ap.add_argument("--alphas", default="0.5,0.65,0.8")
    ap.add_argument("--phase", type=int, default=1, help="1: alpha/group/layer sweep; 2: high-alpha follow-up")
    ap.add_argument("--save-stats", default="", help="write calibration stats here (for the node's llm_int8_stats)")
    args = ap.parse_args()

    from bench_orion import build_model, load_test_image
    from orion_ros.orion_withpid_node import (LIDAR2IMG, LIDAR2CAM, LIDAR2EGO, ORION_CAMERA_ORDER, ORION_AGENT_HZ,
                                              command2hot, command2nohot, invert_matrix_egopose_numpy, custom_wrap_fp16_model)
    from orion_ros.camera_input import decode_image
    from orion_ros import orion_speedups as sp
    from mmcv.parallel.collate import collate
    from mmcv.core.bbox import get_box_type

    model, pipeline = build_model("/root/Orion/adzoo/orion/configs/orion_stage3_agent.py", "/models/Orion/Orion.pth", "fp16")
    sp.merge_lora(model, log); sp.patch_llm_flash_attention(model, log); sp.patch_vit_blocks(model, log)
    sp.slice_map_head_one2one(model, log)
    custom_wrap_fp16_model(model)
    log("model ready")

    imgs = [load_test_image(p) for p in args.images.split(",")]
    par = sp.ParallelDecoder(decode_image, 6)
    frames = [par([(0.0, im.shape[0], im.shape[1], "bgr8", im.tobytes())] * 6, 20) for im in imgs]

    def build_batch(frame_idx, ims):
        l2g = np.eye(4) @ LIDAR2EGO
        r = {"lidar2img": np.stack([LIDAR2IMG[c] for c in ORION_CAMERA_ORDER]), "lidar2cam": np.stack([LIDAR2CAM[c] for c in ORION_CAMERA_ORDER]),
             "cam_intrinsic": [LIDAR2IMG[c] @ np.linalg.inv(LIDAR2CAM[c]) for c in ORION_CAMERA_ORDER], "img": list(ims),
             "folder": " ", "scene_token": "sweep", "frame_idx": frame_idx, "timestamp": frame_idx / ORION_AGENT_HZ,
             "box_type_3d": get_box_type("LiDAR")[0], "can_bus": np.zeros(18), "command": command2nohot(4), "ego_fut_cmd": command2hot(4),
             "ego_pose": l2g, "ego_pose_inv": invert_matrix_egopose_numpy(l2g), "lidar2ego": LIDAR2EGO, "l2g_r_mat": l2g[:3, :3], "l2g_t": l2g[:3, 3]}
        st = np.stack(r["img"], axis=-1); r["img_shape"] = r["ori_shape"] = r["pad_shape"] = st.shape
        r = pipeline(r); b = collate([r], samples_per_gpu=1)
        for k, d in b.items():
            if k != "img_metas" and torch.is_tensor(d[0]): d[0] = d[0].to("cuda")
            if k == "input_ids":
                for i in range(len(d[0])):
                    for kk in range(len(d[0][i])): d[0][i][kk] = d[0][i][kk].to("cuda")
        return b

    feats = []
    orig_inf = model.lm_head.inference_ego
    def inf_hook(*a, **kw):
        out = orig_inf(*a, **kw); feats.append(out.detach().float().clone()); return out
    model.lm_head.inference_ego = inf_hook

    def run(seed=1234):
        model.test_flag = False; feats.clear(); trajs = []
        for f, ims in enumerate(frames):
            b = build_batch(f, ims); torch.manual_seed(seed + f)
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
    if args.save_stats:
        sp.save_llm_act_stats(stats, args.save_stats); log(f"saved calibration stats -> {args.save_stats}")
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
