#!/usr/bin/env python3
"""Standalone ORION inference bench: per-stage timing + output parity across
the speedup variants in orion_ros/orion_speedups.py. Loads the model ONCE
(~200 s) and applies the variants cumulatively, so one run answers
"where does the time go" and "does each speedup change the trajectory".

Run inside the orion_ros container, from the pre-built repo:

  docker exec orion_ros bash -c 'source /opt/ros/humble/setup.bash;
      source /opt/orion_ws/install/setup.bash; cd /root/Orion;
      python3 /benchmarking/alpamayo-autoware/src/orion_ros/tools/bench_orion.py --frames 4'

The batch is built exactly like the node's warm-up (same calibration, same
pipeline, same collate), with a real image re-encoded at JPEG q20 in all six
slots. Timing is input-independent; the parity check is not, hence real pixels.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

ALL_VARIANTS = ["baseline", "merge_lora", "flash_attn", "compile_heads", "compile_llm", "vit_glue", "compile_vit", "map_slice", "overlap_heads", "compile_posembed", "int8_llm", "down_t", "vision_lite", "trt_vit"]


def log(msg: str) -> None:
    print(f"[bench +{time.time() - T0:7.1f}s] {msg}", flush=True)


T0 = time.time()


def build_model(cfg_path: str, ckpt_path: str, precision: str, vit_input_size: int = 640):
    from mmcv import Config
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint
    from mmcv.datasets.pipelines import Compose

    cfg = Config.fromfile(cfg_path)
    if precision == "fp16":
        cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = True, False, False
    else:
        cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = False, False, True
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.cuda().eval()
    pipe_cfg = [t for t in cfg.inference_only_pipeline if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)]
    if vit_input_size != 640:
        from orion_ros import orion_speedups as _sp
        _sp.set_pipeline_input_size(pipe_cfg, vit_input_size)
    pipeline = Compose(pipe_cfg)
    return model, pipeline


def load_test_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8)
    return cv2.resize(img, (1600, 900), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/Orion/adzoo/orion/configs/orion_stage3_agent.py")
    ap.add_argument("--checkpoint", default="/models/Orion/Orion.pth")
    ap.add_argument("--image", default="/benchmarking/imgdiag/frame_0.png")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--compile-mode", default="reduce-overhead")
    ap.add_argument("--profile", action="store_true", help="torch.profiler one baseline frame")
    ap.add_argument("--jpeg-quality", type=int, default=20)
    ap.add_argument("--engine", default="", help="TensorRT plan for the ViT (variant trt_vit)")
    ap.add_argument("--int8-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--int8-smooth-alpha", type=float, default=0.0, help="SmoothQuant alpha (0 = off)")
    ap.add_argument("--int8-calib-frames", type=int, default=3)
    ap.add_argument("--int8-skip-layers", default="", help="decoder layers kept fp16, e.g. 0,1,30,31")
    ap.add_argument("--vit-input-size", type=int, default=640, help="variant vision_lite: ViT input (multiple of 256)")
    ap.add_argument("--rear-refresh-every", type=int, default=2, help="variant vision_lite: rear views refreshed every N frames")
    ap.add_argument("--calib-images", default="/benchmarking/imgdiag/frame_0.png,/benchmarking/imgdiag/frame_1.png,/benchmarking/imgdiag/frame_2.png")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]

    from orion_ros.orion_withpid_node import (
        LIDAR2IMG, LIDAR2CAM, LIDAR2EGO, ORION_CAMERA_ORDER, ORION_AGENT_HZ,
        command2hot, command2nohot, invert_matrix_egopose_numpy, custom_wrap_fp16_model,
    )
    from orion_ros.camera_input import decode_image
    from orion_ros import orion_speedups as sp
    from mmcv.parallel.collate import collate
    from mmcv.core.bbox import get_box_type

    log(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}")
    log("loading model ...")
    model, pipeline = build_model(args.config, args.checkpoint, args.precision)
    pipe_holder = {"pipe": pipeline}
    torch.cuda.synchronize()
    log(f"model loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    img = load_test_image(args.image)
    record = (0.0, img.shape[0], img.shape[1], "bgr8", img.tobytes())
    records = [record] * 6
    records6 = [(0.0, img.shape[0], img.shape[1], "bgr8", np.ascontiguousarray(np.roll(img, 7 * i, axis=1)).tobytes())
                for i in range(6)]
    par = sp.ParallelDecoder(decode_image, workers=6)

    def build_batch(frame_idx: int, imgs: List[np.ndarray]):
        lidar2global = np.eye(4) @ LIDAR2EGO
        results = {
            "lidar2img": np.stack([LIDAR2IMG[c] for c in ORION_CAMERA_ORDER]),
            "lidar2cam": np.stack([LIDAR2CAM[c] for c in ORION_CAMERA_ORDER]),
            "cam_intrinsic": [np.matmul(LIDAR2IMG[c], np.linalg.inv(LIDAR2CAM[c])) for c in ORION_CAMERA_ORDER],
            "img": list(imgs),
            "folder": " ", "scene_token": "bench", "frame_idx": frame_idx,
            "timestamp": frame_idx / ORION_AGENT_HZ,
            "box_type_3d": get_box_type("LiDAR")[0],
            "can_bus": np.zeros(18), "command": command2nohot(4), "ego_fut_cmd": command2hot(4),
            "ego_pose": lidar2global, "ego_pose_inv": invert_matrix_egopose_numpy(lidar2global),
            "lidar2ego": LIDAR2EGO, "l2g_r_mat": lidar2global[0:3, 0:3], "l2g_t": lidar2global[0:3, 3],
        }
        stacked = np.stack(results["img"], axis=-1)
        results["img_shape"] = results["ori_shape"] = results["pad_shape"] = stacked.shape
        t0 = time.perf_counter()
        results = pipe_holder["pipe"](results)
        t1 = time.perf_counter()
        batch = collate([results], samples_per_gpu=1)
        t2 = time.perf_counter()
        for key, data in batch.items():
            if key != "img_metas" and torch.is_tensor(data[0]):
                data[0] = data[0].to("cuda")
            if key == "input_ids":
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        data[0][i][k] = data[0][i][k].to("cuda")
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return batch, {"pipeline": (t1 - t0) * 1e3, "collate": (t2 - t1) * 1e3, "to_device": (t3 - t2) * 1e3}

    log("prep timing (3 reps each) ...")
    for name, fn in [("decode x6 sequential", lambda: [decode_image(r, args.jpeg_quality) for r in records6]),
                     ("decode x6 parallel", lambda: par(records6, args.jpeg_quality))]:
        fn()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        log(f"  {name:24s} {np.mean(ts):6.1f} ms")
    imgs = [decode_image(r, args.jpeg_quality) for r in records]
    for _ in range(2):
        batch, pt = build_batch(0, imgs)
    log("  " + "  ".join(f"{k}={v:.1f}ms" for k, v in pt.items()))
    n_tok = sum(int(t.numel()) for t in batch["input_ids"][0][0]) if torch.is_tensor(batch["input_ids"][0][0][0]) else -1
    log(f"  prompt token ids in batch: {n_tok}  img tensor {tuple(batch['img'][0].shape)} {batch['img'][0].dtype}")

    timer = sp.StageTimer()
    timer.wrap_orion(model)
    custom_wrap_fp16_model(model)

    def run_frames(tag: str, n: int) -> Dict:
        """Reset temporal memory, run n+1 frames (first = warm-up, untimed)."""
        model.test_flag = False
        if getattr(model.img_backbone, "_orion_stagger", None) is not None:
            model.img_backbone._orion_stagger.reset()
        trajs, totals, stages_acc = [], [], {}
        for f in range(n + 1):
            batch, _ = build_batch(f, imgs)
            torch.manual_seed(1234 + f)
            timer.enabled = f > 0
            timer._pending.clear()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                out = model(batch, return_loss=False)
            torch.cuda.synchronize()
            total = (time.perf_counter() - t) * 1e3
            traj = out[0]["pts_bbox"]["ego_fut_preds"].float().cpu().numpy()
            if f == 0:
                log(f"  [{tag}] warm-up forward {total:.0f} ms")
                continue
            st = timer.report()
            for k, v in st.items():
                stages_acc[k] = stages_acc.get(k, 0.0) + v / n
            totals.append(total)
            trajs.append(traj)
            log(f"  [{tag}] frame {f}: {total:.0f} ms  {sp.StageTimer.format(st, total)}")
        return {"totals": totals, "stages": stages_acc, "trajs": trajs}

    results: Dict[str, Dict] = {}
    baseline_trajs = None
    for v in variants:
        log(f"=== variant: {v} ===")
        t_apply = time.perf_counter()
        try:
            if v == "merge_lora":
                sp.merge_lora(model, log)
                timer.wrap(model.lm_head, "inference_ego", "llm")
            elif v == "flash_attn":
                sp.patch_llm_flash_attention(model, log)
            elif v == "compile_llm":
                sp.compile_submodules(model, ["llm"], args.compile_mode, log)
            elif v == "compile_heads":
                sp.compile_submodules(model, ["heads"], args.compile_mode, log)
            elif v == "vit_glue":
                sp.patch_vit_blocks(model, log)
            elif v == "compile_vit":
                sp.compile_submodules(model, ["vit"], args.compile_mode, log)
            elif v == "int8_llm":
                targets = tuple(t for t in args.int8_targets.split(",") if t)
                if args.int8_smooth_alpha > 0:
                    was_compiled = sp.uncompile_submodules(model) > 0
                    cal_imgs = [load_test_image(pth) for pth in args.calib_images.split(",") if pth]
                    cal_par = sp.ParallelDecoder(decode_image, workers=6)

                    def run_cal():
                        model.test_flag = False
                        for f in range(args.int8_calib_frames):
                            ci = cal_imgs[f % len(cal_imgs)]
                            rec = (0.0, ci.shape[0], ci.shape[1], "bgr8", ci.tobytes())
                            cb, _ = build_batch(1000 + f, cal_par([rec] * 6, args.jpeg_quality))
                            with torch.inference_mode():
                                model(cb, return_loss=False)
                    stats = sp.collect_llm_act_stats(model, run_cal, targets)
                    k0 = next(iter(stats))
                    log(f"calibration stats: {len(stats)} projections, e.g. {k0} "
                        f"max|x| {stats[k0].max():.1f} / median {stats[k0].median():.2f}")
                    sp.smooth_llm(model, stats, args.int8_smooth_alpha, log)
                    skip = [int(i) for i in args.int8_skip_layers.split(",") if i]
                    sp.quantize_llm_int8(model, targets, log, skip_layers=skip)
                    if was_compiled:
                        sp.compile_submodules(model, ["heads", "llm", "vit"], args.compile_mode, log)
                else:
                    skip = [int(i) for i in args.int8_skip_layers.split(",") if i]
                    sp.quantize_llm_int8(model, targets, log, skip_layers=skip)
            elif v == "vision_lite":
                if args.vit_input_size != 640:
                    from mmcv import Config
                    from mmcv.datasets.pipelines import Compose
                    cfg = Config.fromfile(args.config)
                    pc = [t for t in cfg.inference_only_pipeline if t["type"] != "LoadMultiViewImageFromFilesInCeph"]
                    sp.set_pipeline_input_size(pc, args.vit_input_size)
                    pipe_holder["pipe"] = Compose(pc)
                    sp.parallelize_pipeline(pipe_holder["pipe"], 6)
                    sp.set_vit_input_size(model, args.vit_input_size, log)
                sv = sp.install_staggered_views(model, args.rear_refresh_every, log)
                if sv is not None:
                    sv.reset()
            elif v == "down_t":
                sp.transpose_llm_down_proj(model, log)
            elif v == "map_slice":
                sp.slice_map_head_one2one(model, log)
            elif v == "overlap_heads":
                sp.overlap_heads(model, log)
                timer.wrap(model.pts_bbox_head, "forward", "det_head")
                timer.wrap(model.map_head, "forward", "map_head")
            elif v == "compile_posembed":
                sp.compile_submodules(model, ["posembed"], args.compile_mode, log)
                timer.wrap(model, "position_embeding", "pos_embed")
            elif v == "trt_vit":
                if not args.engine:
                    log("trt_vit needs --engine"); continue
                if not sp.install_trt_backbone(model, args.engine, log):
                    continue
            elif v != "baseline":
                log(f"unknown variant {v}, skipping"); continue
        except Exception as e:
            log(f"apply failed: {type(e).__name__}: {e}"); continue
        try:
            r = run_frames(v, args.frames)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"variant {v} FAILED: {type(e).__name__}: {e}")
            if v.startswith("compile"):
                log(f"  restored {sp.uncompile_submodules(model)} eager forwards")
            continue
        r["apply_s"] = time.perf_counter() - t_apply - sum(r["totals"]) / 1e3
        if baseline_trajs is None:
            baseline_trajs = r["trajs"]
        r["max_dtraj"] = max(float(np.abs(a - b).max()) for a, b in zip(r["trajs"], baseline_trajs))
        results[v] = r
        log(f"  [{v}] mean forward {np.mean(r['totals']):.0f} ms  max|dtraj| vs baseline {r['max_dtraj']:.4f} m")
        if args.profile and v == "baseline":
            log("=== torch.profiler (one frame, baseline) ===")
            from torch.profiler import ProfilerActivity, profile
            batch, _ = build_batch(99, imgs)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with torch.inference_mode():
                    model(batch, return_loss=False)
                torch.cuda.synchronize()
            try:
                ka = prof.key_averages()
                attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
                cuda_total = sum(getattr(e, attr) for e in ka) / 1e3
                n_kernels = sum(e.count for e in ka if getattr(e, attr) > 0)
                log(f"sum of self GPU time {cuda_total:.0f} ms over ~{n_kernels} kernel launches")
                print(ka.table(sort_by=attr, row_limit=30), flush=True)
            except Exception as e:
                log(f"profiler table failed: {type(e).__name__}: {e}")

    print("\n==== SUMMARY (mean per frame, ms) ====")
    keys = sorted({k for r in results.values() for k in r["stages"]})
    print(f"{'variant':12s} {'total':>7s} " + " ".join(f"{k:>11s}" for k in keys) + f" {'|dtraj|m':>9s} {'apply s':>8s}")
    for v, r in results.items():
        print(f"{v:12s} {np.mean(r['totals']):7.0f} " + " ".join(f"{r['stages'].get(k, 0.0):11.0f}" for k in keys)
              + f" {r['max_dtraj']:9.4f} {r['apply_s']:8.0f}")
    par.shutdown()


if __name__ == "__main__":
    main()
