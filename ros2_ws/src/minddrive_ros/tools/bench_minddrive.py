#!/usr/bin/env python3
"""Standalone MindDrive inference bench: per-stage timing + output parity
across the speedups in minddrive_ros/minddrive_speedups.py.  Loads the model
ONCE and applies the variants cumulatively, so one run answers "where does
the time go" and "does each speedup change the trajectory or the decision".

Run inside the minddrive_ros container:

  docker exec minddrive_ros bash -c 'source /opt/ros/humble/setup.bash;
      cd /benchmarking/MindDrive; PYTHONPATH=/benchmarking/MindDrive
      python3 /benchmarking/alpamayo-autoware/src/minddrive_ros/tools/bench_minddrive.py \
          --precision fp16 --frames 4 --variants baseline,merge_lora,...'

Precision is a LOAD-time property (fp16_infer halves the ViT and loads the
LLM in fp16), so fp32 and fp16 are separate invocations; --ref-out / --ref-in
carry the fp32 reference outputs across so the fp16 chain is compared against
the true reference, not against fp16 eager.

The batch is built exactly like the node's (same calibration, same pipeline,
same collate), from real CARLA frames re-encoded at JPEG q20, rolled by a few
pixels per camera so the six slots differ.  Timing is input-independent; the
parity check is not, hence real pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

ALL_VARIANTS = ["baseline", "merge_lora", "flash_attn", "vit_glue", "map_slice", "down_t", "vit_t",
                "vit_win", "int8_llm", "int8_vit", "vision512", "compile_heads", "compile_llm", "compile_vit", "graph_vit",
                "graph_heads", "logits_slice", "overlap_experts", "stagger2"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[bench +{time.time() - T0:7.1f}s] {msg}", flush=True)


def build_model(cfg_path: str, ckpt_path: str, precision: str):
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
    return model, Compose(pipe_cfg)


def load_test_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8)
    return cv2.resize(img, (1600, 900), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py")
    ap.add_argument("--checkpoint", default="/models/MindDrive/minddrive_3b_rltrain.pth")
    ap.add_argument("--images", default="/benchmarking/imgdiag/frame_0.png,/benchmarking/imgdiag/frame_1.png,/benchmarking/imgdiag/frame_2.png")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--llm-attn", default="flash_attention_2", help="variant flash_attn: sdpa | flash_attention_2 | eager")
    ap.add_argument("--jpeg-quality", type=int, default=20)
    ap.add_argument("--int8-skip", default="", help="variants int8_*: comma list of linear-name substrings kept fp16")
    ap.add_argument("--int8-calib", type=int, default=0,
                    help="variants int8_*: SmoothQuant, calibrated on this many frames run just before quantising")
    ap.add_argument("--int8-alpha", type=float, default=0.5)
    ap.add_argument("--int8-mode", default="w8a8", help="w8a8 | w8 | a8 (the last two: parity-only fp16 GEMMs)")
    ap.add_argument("--int8-calib-out", default="", help="save the calibration stats (torch.save) for the node's int8_calib_path")
    ap.add_argument("--ref-out", default="", help="write this run's baseline outputs (npz) for a later run to compare against")
    ap.add_argument("--ref-in", default="", help="compare every variant against these outputs instead of this run's baseline")
    ap.add_argument("--json-out", default="", help="write the summary table as JSON")
    ap.add_argument("--profile", action="store_true", help="torch.profiler one baseline frame")
    ap.add_argument("--profile-vit", action="store_true",
                    help="after all variants: torch.profiler on one ViT call alone, kernels grouped GEMM/attention/glue")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]

    from minddrive_ros.agent_constants import (
        CAMERA_ORDER, AGENT_HZ, build_agent_results, batch_to_device, custom_wrap_fp16_model)
    from minddrive_ros.camera_input import decode_image
    from minddrive_ros import minddrive_speedups as sp
    from mmcv.parallel.collate import collate
    from mmcv.core.bbox import get_box_type
    from pyquaternion import Quaternion

    log(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}  precision {args.precision}")
    log("loading model ...")
    model, pipeline = build_model(args.config, args.checkpoint, args.precision)
    torch.cuda.synchronize()
    log(f"model loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
        f"LLM dtype {next(sp._expert_models(model)[0].parameters()).dtype}, "
        f"ViT dtype {next(model.img_backbone.parameters()).dtype}, "
        f"attn {sp._expert_models(model)[0].config._attn_implementation}")

    imgs_src = [load_test_image(p) for p in args.images.split(",") if p]
    frames: List[List[np.ndarray]] = []
    for f in range(args.frames + 1):
        base = imgs_src[f % len(imgs_src)]
        recs = [(0.0, base.shape[0], base.shape[1], "bgr8",
                 np.ascontiguousarray(np.roll(base, 7 * i + 3 * f, axis=1)).tobytes()) for i in range(6)]
        frames.append([decode_image(r, args.jpeg_quality) for r in recs])
    par = sp.ParallelDecoder(decode_image, workers=6)

    flags: Dict[str, bool] = {}

    def build_batch(frame_idx: int, imgs: List[np.ndarray]):
        can_bus = np.zeros(18)
        can_bus[0] = 0.5 * frame_idx * 2.5 / AGENT_HZ * 20
        can_bus[3:7] = list(Quaternion(axis=[0, 0, 1], radians=0.0))
        can_bus[7] = 2.5
        results = build_agent_results(dict(zip(CAMERA_ORDER, imgs)), can_bus, 0.0, 4, "bench",
                                      frame_idx, frame_idx / AGENT_HZ, get_box_type)
        t0 = time.perf_counter()
        results = pipeline(results)
        t1 = time.perf_counter()
        batch = collate([results], samples_per_gpu=1)
        t2 = time.perf_counter()
        batch_to_device(batch, torch.device("cuda"))
        if flags.get("overlap"):
            sp.reorder_rounds_for_overlap(batch, model)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return batch, {"pipeline": (t1 - t0) * 1e3, "collate": (t2 - t1) * 1e3, "to_device": (t3 - t2) * 1e3}

    log("prep timing ...")
    recs6 = [(0.0, 900, 1600, "bgr8", np.ascontiguousarray(np.roll(imgs_src[0], 7 * i, axis=1)).tobytes()) for i in range(6)]
    for name, fn in [("decode x6 sequential", lambda: [decode_image(r, args.jpeg_quality) for r in recs6]),
                     ("decode x6 parallel", lambda: par(recs6, args.jpeg_quality))]:
        fn()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        log(f"  {name:24s} {np.mean(ts):6.1f} ms")
    for _ in range(2):
        batch, pt = build_batch(0, frames[0])
    log("  pipeline sequential: " + "  ".join(f"{k}={v:.1f}ms" for k, v in pt.items()))
    sp.parallelize_pipeline(pipeline, 6, log)
    for _ in range(2):
        batch, pt = build_batch(0, frames[0])
    log("  pipeline parallel:   " + "  ".join(f"{k}={v:.1f}ms" for k, v in pt.items()))
    ids = batch["input_ids"][0][0]
    log(f"  prompt token ids per round: {[int(t.numel()) for t in ids]}  img {tuple(batch['img'][0].shape)} {batch['img'][0].dtype}")

    timer = sp.StageTimer()
    timer.wrap_minddrive(model)

    def run_frames(tag: str, n: int) -> Dict:
        """Reset temporal memory, run n+1 frames (first = warm-up, untimed)."""
        model.test_flag = False
        sv = getattr(model.img_backbone, "_md_stagger", None)
        if sv is not None:
            sv.reset()
        outs, totals, stages_acc = [], [], {}
        for f in range(n + 1):
            batch, _ = build_batch(f, frames[f])
            torch.manual_seed(1234 + f)
            timer.enabled = f > 0
            timer._pending.clear()
            torch.cuda.synchronize()
            t = time.perf_counter()
            sp.mark_step()
            with torch.no_grad():
                custom_wrap_fp16_model(model)
                out = model(batch, return_loss=False)
            torch.cuda.synchronize()
            total = (time.perf_counter() - t) * 1e3
            pb = out[0]["pts_bbox"]
            rec = {"speed": pb["ego_fut_preds"].float().cpu().numpy(),
                   "path": pb["pw_ego_fut_pred"].float().cpu().numpy(),
                   "speed_value": int(pb.get("speed_value", -1)), "path_value": int(pb.get("path_value", -1))}
            if f == 0:
                log(f"  [{tag}] warm-up forward {total:.0f} ms")
                continue
            st = timer.report()
            for k, v in st.items():
                stages_acc[k] = stages_acc.get(k, 0.0) + v / n
            totals.append(total)
            outs.append(rec)
            log(f"  [{tag}] frame {f}: {total:.0f} ms  {sp.StageTimer.format(st, total)}  "
                f"decision={rec['speed_value']}/{rec['path_value']}")
        return {"totals": totals, "stages": stages_acc, "outs": outs}

    ref = None
    if args.ref_in and os.path.exists(args.ref_in):
        z = np.load(args.ref_in, allow_pickle=True)
        ref = [{"speed": z["speed"][i], "path": z["path"][i],
                "speed_value": int(z["speed_value"][i]), "path_value": int(z["path_value"][i])}
               for i in range(len(z["speed"]))]
        log(f"reference outputs loaded from {args.ref_in} ({len(ref)} frames, {z['label']})")

    results: Dict[str, Dict] = {}
    for v in variants:
        log(f"=== variant: {v} ===")
        t_apply = time.perf_counter()
        try:
            if v == "merge_lora":
                sp.merge_lora_experts(model, log)
                timer.wrap_llm(model)
            elif v == "flash_attn":
                sp.set_llm_attention(model, args.llm_attn, log)
                timer.wrap_llm(model)
            elif v == "vit_glue":
                sp.patch_vit_blocks(model, log)
            elif v == "map_slice":
                sp.slice_map_head_one2one(model, log)
            elif v == "down_t":
                sp.transpose_llm_down_proj(model, log)
            elif v == "vit_t":
                sp.transpose_vit_linears(model, log=log)
            elif v == "vit_win":
                sp.patch_vit_window_blocks(model, log)
            elif v.startswith("vision"):
                size = int(v[6:])
                sp.set_pipeline_input_size(pipeline, size)
                sp.set_vit_input_size(model, size, log)
                log(f"pipeline resize -> {size}x{size}")
            elif v.startswith("stagger"):
                sv = sp.install_staggered_views(model, int(v[7:]), log)
                sv.reset()
            elif v in ("int8_llm", "int8_vit"):
                calib = None
                if args.int8_calib > 0:
                    calib = sp.collect_int8_calibration(model, [v[5:]], lambda: run_frames("calib", args.int8_calib),
                                                        args.int8_skip.split(","), log)
                    if args.int8_calib_out:
                        prev = torch.load(args.int8_calib_out) if os.path.exists(args.int8_calib_out) else {}
                        prev.update(calib)
                        torch.save(prev, args.int8_calib_out)
                        log(f"calibration stats written to {args.int8_calib_out} ({len(prev)} linears)")
                sp.quantize_linears_int8(model, [v[5:]], args.int8_skip.split(","), log, calib, args.int8_alpha, args.int8_mode)
                timer.wrap_llm(model)
            elif v == "compile_heads":
                sp.compile_submodules(model, ["heads"], args.compile_mode, log)
            elif v == "compile_llm":
                sp.compile_submodules(model, ["llm"], args.compile_mode, log)
            elif v == "compile_vit":
                sp.compile_submodules(model, ["vit"], args.compile_mode, log)
            elif v == "graph_vit":
                sp.graph_vit(model, log)
            elif v == "graph_heads":
                sp.graph_heads(model, log)
            elif v == "logits_slice":
                sp.slice_decision_logits(model, log)
                timer.wrap_llm(model)
            elif v == "overlap_experts":
                sp.overlap_experts(model, log)
                flags["overlap"] = True
                timer.wrap_llm(model)
            elif v != "baseline":
                log(f"unknown variant {v}, skipping"); continue
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"apply failed: {type(e).__name__}: {e}"); continue
        try:
            r = run_frames(v, args.frames)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"variant {v} FAILED: {type(e).__name__}: {e}")
            if v.startswith("compile") or v.startswith("graph"):
                log(f"  restored {sp.uncompile_submodules(model)} eager forwards; ungraph {sp.ungraph_vit(model)}")
            continue
        r["apply_s"] = time.perf_counter() - t_apply - sum(r["totals"]) / 1e3
        if ref is None:
            ref = r["outs"]
            if args.ref_out:
                np.savez(args.ref_out, speed=np.stack([o["speed"] for o in ref]), path=np.stack([o["path"] for o in ref]),
                         speed_value=np.array([o["speed_value"] for o in ref]), path_value=np.array([o["path_value"] for o in ref]),
                         label=f"{args.precision} {v}")
                log(f"reference outputs written to {args.ref_out}")
        r["d_speed"] = max(float(np.abs(a["speed"] - b["speed"]).max()) for a, b in zip(r["outs"], ref))
        r["d_path"] = max(float(np.abs(a["path"] - b["path"]).max()) for a, b in zip(r["outs"], ref))
        r["decision_agree"] = sum(int(a["speed_value"] == b["speed_value"] and a["path_value"] == b["path_value"])
                                  for a, b in zip(r["outs"], ref))
        results[v] = r
        log(f"  [{v}] mean forward {np.mean(r['totals']):.0f} ms  max|d speed traj| {r['d_speed']:.4f} m  "
            f"max|d path| {r['d_path']:.4f} m  decisions agree {r['decision_agree']}/{len(ref)}")
        if args.profile and v == "baseline":
            log("=== torch.profiler (one frame) ===")
            from torch.profiler import ProfilerActivity, profile
            batch, _ = build_batch(99, frames[1])
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with torch.no_grad():
                    model(batch, return_loss=False)
                torch.cuda.synchronize()
            try:
                ka = prof.key_averages()
                attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
                cuda_total = sum(getattr(e, attr) for e in ka) / 1e3
                n_kernels = sum(e.count for e in ka if getattr(e, attr) > 0)
                log(f"sum of self GPU time {cuda_total:.0f} ms over ~{n_kernels} kernel launches")
                print(ka.table(sort_by=attr, row_limit=25), flush=True)
            except Exception as e:
                log(f"profiler table failed: {type(e).__name__}: {e}")

    if args.profile_vit:
        from torch.profiler import ProfilerActivity, profile
        vit = model.img_backbone
        x = batch["img"][0]
        x = x.reshape(-1, *x.shape[-3:]).to(next(vit.parameters()).dtype).cuda().contiguous()
        with torch.no_grad():
            for _ in range(2):
                vit(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                vit(x)
                torch.cuda.synchronize()
        ka = prof.key_averages()
        attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
        groups: Dict[str, List[float]] = {}
        for e in ka:
            t = getattr(e, attr) / 1e3
            if t <= 0:
                continue
            n = e.key.lower()
            if any(k in n for k in ("gemm", "cublas", "cutlass", "xmma", "sgemm", "hgemm", "nvjet")):
                g = "GEMM"
            elif any(k in n for k in ("flash", "fmha", "attention", "sdp")):
                g = "attention"
            elif n.startswith("triton_") or "triton" in n:
                g = "inductor fused (glue)"
            elif any(k in n for k in ("memcpy", "memset", "copy")):
                g = "copies"
            else:
                g = "other"
            groups.setdefault(g, [0.0, 0])
            groups[g][0] += t; groups[g][1] += e.count
        total = sum(v[0] for v in groups.values())
        log(f"=== ViT alone, one call, GPU kernel time {total:.0f} ms ===")
        for g, (t, c) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
            log(f"  {g:24s} {t:7.1f} ms  {c:5d} launches")
        print(ka.table(sort_by=attr, row_limit=20), flush=True)

    print(f"\n==== SUMMARY ({args.precision}; mean per frame, ms) ====")
    keys = sorted({k for r in results.values() for k in r["stages"]})
    print(f"{'variant':14s} {'total':>7s} " + " ".join(f"{k:>12s}" for k in keys)
          + f" {'d_speed m':>10s} {'d_path m':>9s} {'agree':>6s} {'apply s':>8s}")
    for v, r in results.items():
        print(f"{v:14s} {np.mean(r['totals']):7.0f} " + " ".join(f"{r['stages'].get(k, 0.0):12.0f}" for k in keys)
              + f" {r['d_speed']:10.4f} {r['d_path']:9.4f} {r['decision_agree']:>3d}/{len(ref):<2d} {r['apply_s']:8.0f}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({v: {"total": float(np.mean(r["totals"])), "stages": r["stages"], "d_speed": r["d_speed"],
                           "d_path": r["d_path"], "agree": r["decision_agree"], "n": len(ref), "apply_s": r["apply_s"]}
                       for v, r in results.items()}, fh, indent=1)
    par.shutdown()


if __name__ == "__main__":
    main()
