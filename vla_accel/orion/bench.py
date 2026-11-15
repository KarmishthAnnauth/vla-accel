"""Cumulative ORION speedup bench on a Bench2Drive clip.

Loads the model once, applies the variants in order, and reports per-stage CUDA-event
time and max trajectory deviation from the untouched model. `--variants baseline --profile`
gives the baseline stage profile.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch

from vla_accel.common import b2d, jetson
from vla_accel.orion import inputs, model as om, speedups as sp

ALL_VARIANTS = ["baseline", "merge_lora", "flash_attn", "compile_heads", "compile_llm", "vit_glue",
                "compile_vit", "map_slice", "overlap_heads", "compile_posembed", "int8_llm", "down_t",
                "vision_lite", "trt_vit"]
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped log line."""
    print(f"[bench +{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    """Run the variant chain and print the summary table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=om.CONFIG)
    ap.add_argument("--checkpoint", default=om.CHECKPOINT)
    ap.add_argument("--precision", default="fp16")
    inputs.add_b2d_args(ap, frames=4, split="held")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--compile-mode", default="reduce-overhead")
    ap.add_argument("--profile", action="store_true", help="torch.profiler one baseline frame")
    ap.add_argument("--jpeg-quality", type=int, default=inputs.JPEG_QUALITY)
    ap.add_argument("--engine", default=os.path.join(om.ENGINES_DIR, "orion_vit_fp16.plan"),
                    help="TensorRT plan for the ViT (variant trt_vit)")
    ap.add_argument("--int8-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--int8-smooth-alpha", type=float, default=0.0, help="SmoothQuant alpha (0 = off)")
    ap.add_argument("--int8-calib-frames", type=int, default=3, help="frames from the first calibration clip")
    ap.add_argument("--int8-skip-layers", default="", help="decoder layers kept fp16, e.g. 0,1,30,31")
    ap.add_argument("--vit-input-size", type=int, default=640, help="variant vision_lite: ViT input (multiple of 256)")
    ap.add_argument("--rear-refresh-every", type=int, default=2, help="variant vision_lite: rear views refreshed every N frames")
    args = ap.parse_args()
    args.b2d_dir = os.path.abspath(args.b2d_dir)
    variants = [v for v in args.variants.split(",") if v]
    jetson.warn_if_unpinned()

    clip = inputs.select_clips(args.b2d_dir, args.clips, args.split)[0]
    idxs = b2d.frame_indices(clip, args.frames + 1, args.stride, args.start)
    frames = [(i, *inputs.load_frame(clip, i, args.jpeg_quality)) for i in idxs]
    log(f"clip {os.path.basename(clip)}: frames {idxs} (first is warm-up)")

    log(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}  GPU {jetson.gpu_mhz()} MHz")
    log("loading model ...")
    model, cfg = om.build_model(args.config, args.checkpoint, args.precision)
    pipe_holder = {"pipe": om.build_pipeline(cfg)}
    torch.cuda.synchronize()
    log(f"model loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def batch_of(i: int):
        idx, imgs, anno = frames[i]
        return inputs.build_batch(pipe_holder["pipe"], imgs, anno, idx)

    log("prep timing (3 reps each) ...")
    records = [(clip, c, idxs[0]) for c in inputs.CAMERA_ORDER]
    par = sp.ParallelDecoder(inputs.load_camera, workers=6)
    for name, fn in [("load x6 sequential", lambda: [inputs.load_camera(r, args.jpeg_quality) for r in records]),
                     ("load x6 parallel", lambda: par(records, args.jpeg_quality))]:
        fn()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        log(f"  {name:24s} {np.mean(ts):6.1f} ms")
    for _ in range(2):
        batch, pt = batch_of(0)
    log("  " + "  ".join(f"{k}={v:.1f}ms" for k, v in pt.items()))
    n_tok = sum(int(t.numel()) for t in batch["input_ids"][0][0]) if torch.is_tensor(batch["input_ids"][0][0][0]) else -1
    log(f"  prompt token ids in batch: {n_tok}  img tensor {tuple(batch['img'][0].shape)} {batch['img'][0].dtype}")

    timer = sp.StageTimer()
    timer.wrap_orion(model)
    om.custom_wrap_fp16_model(model)

    def run_frames(tag: str) -> Dict:
        """Reset temporal memory and run the clip frames; the first is an untimed warm-up."""
        model.test_flag = False
        if getattr(model.img_backbone, "_orion_stagger", None) is not None:
            model.img_backbone._orion_stagger.reset()
        trajs, totals, stages_acc = [], [], {}
        n = len(frames) - 1
        for f in range(len(frames)):
            batch, _ = batch_of(f)
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

    def run_calibration():
        """Push the calibration frames through the model (SmoothQuant stats)."""
        cal_clip = inputs.select_clips(args.b2d_dir, split="calib")[0]
        model.test_flag = False
        for idx in b2d.frame_indices(cal_clip, args.int8_calib_frames, args.stride, args.start):
            imgs, anno = inputs.load_frame(cal_clip, idx, args.jpeg_quality)
            cb, _ = inputs.build_batch(pipe_holder["pipe"], imgs, anno, idx)
            with torch.inference_mode():
                model(cb, return_loss=False)

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
                skip = [int(i) for i in args.int8_skip_layers.split(",") if i]
                if args.int8_smooth_alpha > 0:
                    was_compiled = sp.uncompile_submodules(model) > 0
                    stats = sp.collect_llm_act_stats(model, run_calibration, targets)
                    k0 = next(iter(stats))
                    log(f"calibration stats: {len(stats)} projections, e.g. {k0} "
                        f"max|x| {stats[k0].max():.1f} / median {stats[k0].median():.2f}")
                    sp.smooth_llm(model, stats, args.int8_smooth_alpha, log)
                    sp.quantize_llm_int8(model, targets, log, skip_layers=skip)
                    if was_compiled:
                        sp.compile_submodules(model, ["heads", "llm", "vit"], args.compile_mode, log)
                else:
                    sp.quantize_llm_int8(model, targets, log, skip_layers=skip)
            elif v == "vision_lite":
                if args.vit_input_size != 640:
                    pipe_holder["pipe"] = om.build_pipeline(cfg, args.vit_input_size, workers=6)
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
                if not os.path.exists(args.engine):
                    log(f"trt_vit: no engine at {args.engine}, skipping"); continue
                if not sp.install_trt_backbone(model, args.engine, log):
                    continue
            elif v != "baseline":
                log(f"unknown variant {v}, skipping"); continue
        except Exception as e:
            log(f"apply failed: {type(e).__name__}: {e}"); continue
        try:
            r = run_frames(v)
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
            batch, _ = batch_of(len(frames) - 1)
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
