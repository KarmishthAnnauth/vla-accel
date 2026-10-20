#!/usr/bin/env python3
"""ORION opt-in options (INT8 LLM, lite vision, both) on real six-camera
Bench2Drive clips.

One configuration per process (INT8 rewrites weights, lite vision changes the
compiled ViT shape), set up in the same order as orion_withpid_node._setup_model.
Each clip is fed as a sequence -- frames `--stride` apart, temporal memory
reset at the clip start -- with the inputs the reference agent builds
(team_code/orion_b2d_agent.py::run_step) from the clip's annotations. The
planner's VAE noise is seeded per frame identically in every configuration, so
trajectories can be compared frame by frame afterwards (--compare).

  python3 bench_orion_b2d.py --setup default --out /benchmarking/orion_b2d/default.npz
  python3 bench_orion_b2d.py --setup int8    --out ...
  python3 bench_orion_b2d.py --setup lite    --out ...
  python3 bench_orion_b2d.py --setup both    --out ...
  python3 bench_orion_b2d.py --setup default --seed-offset 1000 --out .../noise.npz
  python3 bench_orion_b2d.py --compare /benchmarking/orion_b2d
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)


def _stub_missing_ros_msgs() -> None:
    """The bare image has no colcon workspace (carla_msgs, autoware msgs).
    Only constants and helpers are imported from the node, so message
    packages that are missing are replaced by empty stand-ins."""
    import importlib
    import types
    from unittest import mock
    for pkg in ("carla_msgs", "autoware_planning_msgs", "autoware_vehicle_msgs",
                "ackermann_msgs", "visualization_msgs", "nav_msgs", "sensor_msgs",
                "std_msgs", "geometry_msgs", "builtin_interfaces"):
        try:
            importlib.import_module(pkg + ".msg")
        except ImportError:
            root = types.ModuleType(pkg)
            root.msg = mock.MagicMock()
            sys.modules[pkg] = root
            sys.modules[pkg + ".msg"] = root.msg

CAMS = [("CAM_FRONT", "rgb_front"), ("CAM_FRONT_LEFT", "rgb_front_left"),
        ("CAM_FRONT_RIGHT", "rgb_front_right"), ("CAM_BACK", "rgb_back"),
        ("CAM_BACK_LEFT", "rgb_back_left"), ("CAM_BACK_RIGHT", "rgb_back_right")]
STATS = os.path.join(SRC_DIR, "engines", "llm_act_stats.pt")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[b2d +{time.time() - T0:7.1f}s] {msg}", flush=True)


def load_frame(clip: str, idx: int):
    imgs = []
    for _, d in CAMS:
        img = cv2.imread(os.path.join(clip, "camera", d, f"{idx:05d}.jpg"), cv2.IMREAD_COLOR)
        assert img is not None, (clip, d, idx)
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        imgs.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    with gzip.open(os.path.join(clip, "anno", f"{idx:05d}.json.gz")) as f:
        anno = json.load(f)
    return imgs, anno


def run(args) -> None:
    import torch
    from pyquaternion import Quaternion
    from mmcv import Config
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint
    from mmcv.datasets.pipelines import Compose
    from mmcv.parallel.collate import collate
    from mmcv.core.bbox import get_box_type
    from orion_ros import orion_speedups as sp
    _stub_missing_ros_msgs()
    from orion_ros.orion_withpid_node import (
        LIDAR2IMG, LIDAR2CAM, LIDAR2EGO, command2hot, command2nohot,
        invert_matrix_egopose_numpy, custom_wrap_fp16_model)

    int8 = args.setup in ("int8", "both")
    lite = args.setup in ("lite", "both")
    vit_size = 512 if lite else 640

    cfg = Config.fromfile(args.config)
    cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = True, False, False
    log(f"setup={args.setup} building model ...")
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()
    log("model loaded")

    sp.merge_lora(model, log)
    sp.patch_llm_flash_attention(model, log)
    sp.patch_vit_blocks(model, log)
    if vit_size != 640:
        sp.set_vit_input_size(model, vit_size, log)
    if int8:
        targets = ["gate_proj", "up_proj", "down_proj"]
        sp.apply_llm_int8(model, STATS, 0.8, [0, 1, 30, 31], log, targets)
    else:
        sp.transpose_llm_down_proj(model, log)
    sp.compile_submodules(model, ["heads", "llm", "vit"], "default", log)
    sp.slice_map_head_one2one(model, log)
    stagger = sp.install_staggered_views(model, 2, log) if lite else None
    custom_wrap_fp16_model(model)

    pipe_cfg = [t for t in cfg.inference_only_pipeline if t["type"] != "LoadMultiViewImageFromFilesInCeph"]
    if vit_size != 640:
        sp.set_pipeline_input_size(pipe_cfg, vit_size)
    pipeline = Compose(pipe_cfg)
    sp.parallelize_pipeline(pipeline, 6)

    def build_batch(imgs, anno, frame_idx):
        raw_theta = anno["theta"] if not np.isnan(anno["theta"]) else 0.0
        ego_theta = -raw_theta + np.pi / 2
        can_bus = np.zeros(18)
        can_bus[0], can_bus[1] = anno["x"], -anno["y"]
        can_bus[3:7] = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus[7] = anno["speed"]
        can_bus[10:13] = anno["acceleration"]
        can_bus[11] *= -1
        can_bus[13:16] = -np.asarray(anno["angular_velocity"])
        can_bus[16], can_bus[17] = ego_theta, ego_theta / np.pi * 180
        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]
        l2g = ego2world @ LIDAR2EGO
        cmd = int(anno["command_near"])
        res = {
            "lidar2img": np.stack([LIDAR2IMG[c] for c, _ in CAMS]),
            "lidar2cam": np.stack([LIDAR2CAM[c] for c, _ in CAMS]),
            "cam_intrinsic": [LIDAR2IMG[c] @ np.linalg.inv(LIDAR2CAM[c]) for c, _ in CAMS],
            "img": imgs, "folder": " ", "scene_token": " ", "frame_idx": frame_idx,
            "timestamp": frame_idx / 10.0, "box_type_3d": get_box_type("LiDAR")[0],
            "can_bus": can_bus, "command": command2nohot(cmd), "ego_fut_cmd": command2hot(cmd),
            "ego_pose": l2g, "ego_pose_inv": invert_matrix_egopose_numpy(l2g), "lidar2ego": LIDAR2EGO,
            "l2g_r_mat": l2g[0:3, 0:3], "l2g_t": l2g[0:3, 3],
        }
        res["img_shape"] = res["ori_shape"] = res["pad_shape"] = np.stack(imgs, axis=-1).shape
        res = pipeline(res)
        batch = collate([res], samples_per_gpu=1)
        for key, data in batch.items():
            if key != "img_metas" and torch.is_tensor(data[0]):
                data[0] = data[0].to("cuda")
            if key == "input_ids":
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        data[0][i][k] = data[0][i][k].to("cuda")
        return batch

    clips = sorted(glob.glob(os.path.join(args.clips, "*_Route*")))
    plan = []
    for c in clips:
        n = len(glob.glob(os.path.join(c, "anno", "*.json.gz")))
        idxs = list(range(args.start, n, args.stride))[: args.frames]
        plan.append((c, idxs))
    log(f"{len(clips)} clips, {sum(len(i) for _, i in plan)} frames, stride {args.stride}")

    def forward(batch, seed):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model(batch, return_loss=False)
        torch.cuda.synchronize()
        return out[0]["pts_bbox"]["ego_fut_preds"].float().cpu().numpy(), (time.perf_counter() - t) * 1e3

    c0, i0 = plan[0]
    model.test_flag = False
    if stagger is not None:
        stagger.reset()
    for k in range(4):
        imgs, anno = load_frame(c0, i0[min(k, len(i0) - 1)])
        _, ms = forward(build_batch(imgs, anno, k), 0)
        log(f"warm-up {k}: {ms:.0f} ms")

    trajs, fwd, prep, keys = [], [], [], []
    for c, idxs in plan:
        model.test_flag = False
        if stagger is not None:
            stagger.reset()
        name = os.path.basename(c)
        for j, idx in enumerate(idxs):
            t = time.perf_counter()
            imgs, anno = load_frame(c, idx)
            batch = build_batch(imgs, anno, idx)
            torch.cuda.synchronize()
            p = (time.perf_counter() - t) * 1e3
            traj, ms = forward(batch, 1234 + args.seed_offset + len(keys))
            trajs.append(traj); fwd.append(ms); prep.append(p); keys.append(f"{name}:{idx}")
        log(f"{name}: {len(idxs)} frames, forward mean {np.mean(fwd[-len(idxs):]):.0f} ms")

    np.savez(args.out, trajs=np.stack(trajs), forward_ms=np.array(fwd), prep_ms=np.array(prep),
             keys=np.array(keys), setup=args.setup)
    log(f"setup={args.setup}: forward {np.mean(fwd):.0f} ms (sd {np.std(fwd):.0f}), "
        f"prep {np.mean(prep):.0f} ms, saved {args.out}")


def compare(folder: str) -> None:
    ref = np.load(os.path.join(folder, "default.npz"))
    print(f"{'setup':10s} {'forward ms':>10s} {'prep ms':>8s} {'mean dev m':>11s} {'median':>8s} {'p95':>7s} {'max m':>7s}")
    for s in ["default", "noise", "int8", "lite", "both"]:
        p = os.path.join(folder, f"{s}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        assert list(d["keys"]) == list(ref["keys"]), s
        dev = np.linalg.norm(d["trajs"] - ref["trajs"], axis=-1).reshape(len(d["keys"]), -1).max(axis=1)
        print(f"{s:10s} {d['forward_ms'].mean():10.0f} {d['prep_ms'].mean():8.0f} {dev.mean():11.3f} "
              f"{np.median(dev):8.3f} {np.percentile(dev, 95):7.3f} {dev.max():7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["default", "int8", "lite", "both"], default="default")
    ap.add_argument("--config", default="/root/Orion/adzoo/orion/configs/orion_stage3_agent.py")
    ap.add_argument("--checkpoint", default="/models/Orion/Orion.pth")
    ap.add_argument("--clips", default="/b2d")
    ap.add_argument("--frames", type=int, default=15, help="frames per clip")
    ap.add_argument("--stride", type=int, default=10, help="annotation frames apart (B2D is 10 Hz)")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    else:
        run(a)
