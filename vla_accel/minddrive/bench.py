"""MindDrive cumulative speedup bench on Bench2Drive clips: per-stage timing and trajectory/decision parity.

Loads the model once and applies the variants in order. Precision is a load-time property, so fp32 and fp16
are separate runs; --ref-out / --ref-in carry the fp32 reference outputs into the fp16 run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from vla_accel.common import b2d, jetson

ALL_VARIANTS = ["baseline", "merge_lora", "flash_attn", "vit_glue", "map_slice", "down_t", "vit_t",
                "vit_win", "int8_llm", "int8_vit", "vision512", "compile_heads", "compile_llm", "compile_vit", "graph_vit",
                "graph_heads", "logits_slice", "overlap_experts", "stagger2"]
T0 = time.time()

Plan = List[Tuple[str, List[int]]]


def log(msg: str) -> None:
    """Timestamped log line."""
    print(f"[bench +{time.time() - T0:7.1f}s] {msg}", flush=True)


def build_model(cfg_path: str, ckpt_path: str, precision: str):
    """Build MindDrive as the agent does, at fp16 or fp32, and its inference pipeline."""
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


def load_camera(record: tuple, jpeg_quality: int) -> np.ndarray:
    """One B2D camera image (clip, cam, idx) as BGR 1600x900, JPEG round-tripped like the agent."""
    clip, cam, idx = record
    img = b2d.load_image(clip, cam, idx, jpeg_quality=jpeg_quality)
    if img.shape[:2] != (900, 1600):
        img = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_LINEAR)
    return img


def make_plan(clips: List[str], frames: int, stride: int, start: int) -> Plan:
    """(clip, frame indices) for every selected clip."""
    plan = [(c, b2d.frame_indices(c, frames, stride, start)) for c in clips]
    return [(c, idxs) for c, idxs in plan if idxs]


def select_clips(args) -> Tuple[List[str], List[str]]:
    """(bench clips, calibration clips) from --clips or the held-out / calibration split."""
    calib, held = b2d.split_clips(args.b2d_dir)
    if args.clips:
        names = [c for c in args.clips.split(",") if c]
        bench = [os.path.join(args.b2d_dir, n) for n in names]
    else:
        bench = (held or calib)[: args.num_clips]
    calib = [c for c in calib if c not in bench] or bench
    return bench, calib


def main() -> None:
    """Run the variant chain and print / save the summary table."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py",
                    help="relative to the MindDrive checkout (the working directory)")
    ap.add_argument("--checkpoint", default=os.environ.get("MINDDRIVE_CKPT", "/models/MindDrive/minddrive_3b_rltrain.pth"))
    ap.add_argument("--b2d-dir", default=b2d.B2D_DIR)
    ap.add_argument("--clips", default="", help="comma list of clip folder names (default: held-out split)")
    ap.add_argument("--num-clips", type=int, default=2, help="clips taken from the held-out split when --clips is unset")
    ap.add_argument("--frames", type=int, default=4, help="frames per clip")
    ap.add_argument("--stride", type=int, default=10, help="anno frames apart (B2D is 10 Hz)")
    ap.add_argument("--start", type=int, default=10, help="first anno frame of each clip")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--llm-attn", default="flash_attention_2", help="variant flash_attn: sdpa | flash_attention_2 | eager")
    ap.add_argument("--jpeg-quality", type=int, default=20)
    ap.add_argument("--int8-skip", default="", help="variants int8_*: comma list of linear-key substrings kept fp16")
    ap.add_argument("--int8-calib", type=int, default=0,
                    help="variants int8_*: SmoothQuant, calibrated on this many frames of the calibration clips")
    ap.add_argument("--int8-alpha", type=float, default=0.5)
    ap.add_argument("--int8-mode", default="w8a8", help="w8a8 | w8 | a8 (the last two: parity-only fp16 GEMMs)")
    ap.add_argument("--int8-calib-out", default="", help="save the calibration stats (torch.save)")
    ap.add_argument("--ref-out", default="", help="write this run's baseline outputs (npz)")
    ap.add_argument("--ref-in", default="", help="compare every variant against these outputs instead of this run's baseline")
    ap.add_argument("--json-out", default="", help="write the summary table as JSON")
    ap.add_argument("--profile", action="store_true", help="torch.profiler on one baseline frame")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]

    from vla_accel.minddrive.inputs import (
        CAMERA_ORDER, MEMORY_MAX_DT, batch_to_device, build_agent_results, can_bus_from_anno,
        custom_wrap_fp16_model)
    from vla_accel.minddrive import speedups as sp
    from mmcv.parallel.collate import collate
    from mmcv.core.bbox import get_box_type

    jetson.warn_if_unpinned()
    bench_clips, calib_clips = select_clips(args)
    plan = make_plan(bench_clips, args.frames, args.stride, args.start)
    n_frames = sum(len(i) for _, i in plan)
    if not n_frames:
        raise SystemExit("no frames selected (check --b2d-dir, --clips, --start)")
    log(f"{len(plan)} clips, {n_frames} frames, stride {args.stride}: {[os.path.basename(c) for c, _ in plan]}")
    if args.stride / b2d.B2D_HZ >= MEMORY_MAX_DT:
        log(f"stride {args.stride} is >= {MEMORY_MAX_DT} s: the heads drop temporal memory every frame")

    log(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}  GPU {jetson.gpu_mhz()} MHz  "
        f"precision {args.precision}")
    log("loading model ...")
    model, pipeline = build_model(args.config, args.checkpoint, args.precision)
    torch.cuda.synchronize()
    log(f"model loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
        f"LLM dtype {next(sp._expert_models(model)[0].parameters()).dtype}, "
        f"ViT dtype {next(model.img_backbone.parameters()).dtype}, "
        f"attn {sp._expert_models(model)[0].config._attn_implementation}")

    loader = sp.ParallelDecoder(load_camera, workers=6)
    flags: Dict[str, bool] = {}

    def load_frame(clip: str, idx: int) -> Tuple[List[np.ndarray], dict]:
        imgs = loader([(clip, cam, idx) for cam in CAMERA_ORDER], args.jpeg_quality)
        return imgs, b2d.load_anno(clip, idx)

    def build_batch(clip: str, idx: int, step: int, imgs: List[np.ndarray], anno: dict):
        can_bus, ego_theta, command = can_bus_from_anno(anno)
        results = build_agent_results(dict(zip(CAMERA_ORDER, imgs)), can_bus, ego_theta, command,
                                      os.path.basename(clip), step, idx / b2d.B2D_HZ, get_box_type)
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

    c0, i0 = plan[0]
    imgs0, anno0 = load_frame(c0, i0[0])

    log("prep timing ...")
    recs = [(c0, cam, i0[0]) for cam in CAMERA_ORDER]
    for name, fn in [("load x6 sequential", lambda: [load_camera(r, args.jpeg_quality) for r in recs]),
                     ("load x6 parallel", lambda: loader(recs, args.jpeg_quality))]:
        fn()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        log(f"  {name:24s} {np.mean(ts):6.1f} ms")
    for _ in range(2):
        batch, pt = build_batch(c0, i0[0], 0, imgs0, anno0)
    log("  pipeline sequential: " + "  ".join(f"{k}={v:.1f}ms" for k, v in pt.items()))
    sp.parallelize_pipeline(pipeline, 6, log)
    for _ in range(2):
        batch, pt = build_batch(c0, i0[0], 0, imgs0, anno0)
    log("  pipeline parallel:   " + "  ".join(f"{k}={v:.1f}ms" for k, v in pt.items()))
    ids = batch["input_ids"][0][0]
    log(f"  prompt token ids per round: {[int(t.numel()) for t in ids]}  img {tuple(batch['img'][0].shape)} {batch['img'][0].dtype}")

    timer = sp.StageTimer()
    timer.wrap_minddrive(model)

    def reset_memory() -> None:
        model.test_flag = False
        if getattr(model, "planning_memory", None) is not None:
            model.planning_memory = None
        sv = getattr(model.img_backbone, "_md_stagger", None)
        if sv is not None:
            sv.reset()

    def forward(batch, seed: int):
        torch.manual_seed(seed)
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
        return rec, total

    def run_frames(tag: str, run_plan: Plan) -> Dict:
        """One untimed warm-up forward, then every clip of run_plan as a sequence, memory reset at clip start."""
        reset_memory()
        timer.enabled = False
        clip0, idx0 = run_plan[0][0], run_plan[0][1][0]
        imgs, anno = load_frame(clip0, idx0)
        batch, _ = build_batch(clip0, idx0, 0, imgs, anno)
        _, ms = forward(batch, 1234)
        log(f"  [{tag}] warm-up forward {ms:.0f} ms")
        n = sum(len(i) for _, i in run_plan)
        outs, totals, stages_acc, keys = [], [], {}, []
        for clip, idxs in run_plan:
            reset_memory()
            name = os.path.basename(clip)
            for step, idx in enumerate(idxs):
                imgs, anno = load_frame(clip, idx)
                batch, _ = build_batch(clip, idx, step, imgs, anno)
                timer.enabled = True
                timer._pending.clear()
                rec, total = forward(batch, 1235 + len(keys))
                timer.enabled = False
                st = timer.report()
                for k, v in st.items():
                    stages_acc[k] = stages_acc.get(k, 0.0) + v / n
                totals.append(total)
                outs.append(rec)
                keys.append(f"{name}:{idx}")
                log(f"  [{tag}] {name}:{idx}: {total:.0f} ms  {sp.StageTimer.format(st, total)}  "
                    f"decision={rec['speed_value']}/{rec['path_value']}")
        return {"totals": totals, "stages": stages_acc, "outs": outs, "keys": keys}

    ref, ref_keys = None, None
    if args.ref_in and os.path.exists(args.ref_in):
        z = np.load(args.ref_in, allow_pickle=True)
        ref = [{"speed": z["speed"][i], "path": z["path"][i],
                "speed_value": int(z["speed_value"][i]), "path_value": int(z["path_value"][i])}
               for i in range(len(z["speed"]))]
        ref_keys = list(z["keys"]) if "keys" in z else None
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
                    per_clip = max(1, -(-args.int8_calib // len(calib_clips)))
                    cplan = make_plan(calib_clips, per_clip, args.stride, args.start)
                    budget, trimmed = args.int8_calib, []
                    for c, idxs in cplan:
                        if budget > 0:
                            trimmed.append((c, idxs[:budget]))
                            budget -= len(trimmed[-1][1])
                    calib = sp.collect_int8_calibration(model, [v[5:]], lambda: run_frames("calib", trimmed),
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
            r = run_frames(v, plan)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"variant {v} FAILED: {type(e).__name__}: {e}")
            if v.startswith("compile") or v.startswith("graph"):
                log(f"  restored {sp.uncompile_submodules(model)} eager forwards; ungraph {sp.ungraph_vit(model)}")
            continue
        r["apply_s"] = time.perf_counter() - t_apply - sum(r["totals"]) / 1e3
        if ref is None:
            ref, ref_keys = r["outs"], r["keys"]
            if args.ref_out:
                np.savez(args.ref_out, speed=np.stack([o["speed"] for o in ref]), path=np.stack([o["path"] for o in ref]),
                         speed_value=np.array([o["speed_value"] for o in ref]), path_value=np.array([o["path_value"] for o in ref]),
                         keys=np.array(ref_keys), label=f"{args.precision} {v}")
                log(f"reference outputs written to {args.ref_out}")
        if ref_keys is not None and list(ref_keys) != r["keys"]:
            raise SystemExit("reference frames differ from this run's frames (use the same clip/frame options)")
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
            imgs, anno = load_frame(c0, i0[-1])
            batch, _ = build_batch(c0, i0[-1], 0, imgs, anno)
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

    if not results:
        raise SystemExit("no variant completed")
    print(f"\n==== SUMMARY ({args.precision}; mean per frame, ms; {n_frames} B2D frames) ====")
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
    loader.shutdown()


if __name__ == "__main__":
    main()
