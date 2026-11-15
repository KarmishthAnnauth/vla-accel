"""Stage profile of the unmodified Alpamayo 1.5 rollout (1 sample, 5 Euler steps) on a Bench2Drive frame.

``other`` is the model time the timed stages do not account for.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from vla_accel.alpamayo import common


def run_once(model, processor, payload, args, timer):
    """One timed rollout; returns (stage row, reasoning text, trajectory)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs = common.make_inputs(processor, payload, nav_text=args.nav_text)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    fn = (
        model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav
        if args.nav_text
        else model.sample_trajectories_from_data_with_vlm_rollout
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = fn(
            data=inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=1,
            num_traj_sets=1,
            max_generation_length=args.max_gen,
            return_extra=True,
        )
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    rec = timer.rec
    llm = rec["llm"]
    n_vlm_calls = len(rec["lm_head"])
    row = {
        "preprocess": (t1 - t0) * 1e3,
        "vit": sum(rec["vit"]),
        "prefill": llm[0],
        "decode": sum(llm[1:n_vlm_calls]) if not args.nav_text else float("nan"),
        "decode_steps": n_vlm_calls - 1,
        "lm_head": sum(rec["lm_head"]),
        "generate": sum(rec["generate"]),
        "expert": sum(rec["expert"]),
        "expert_calls": len(rec["expert"]),
        "diffusion": sum(rec["diffusion"]),
        "to_traj": sum(rec["to_traj"]),
        "model": (t2 - t1) * 1e3,
        "total": (t2 - t0) * 1e3,
        "prompt_len": int(inputs["tokenized_data"]["input_ids"].shape[1]),
        "mhz": common.gpu_mhz(),
    }
    row["gen_overhead"] = row["generate"] - row["vit"] - sum(llm[:n_vlm_calls]) - row["lm_head"]
    row["other"] = row["model"] - row["generate"] - row["diffusion"] - row["to_traj"]
    timer.reset()
    return row, str(extra["cot"].reshape(-1)[0]), pred_xyz


def main() -> None:
    """Profile each camera count."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-gen", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--nav-text", default=None)
    common.add_b2d_args(ap, frames=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, processor = common.load_model()
    model.diffusion.num_inference_steps = args.steps
    timer = common.StageTimer().attach(model)
    _, held = common.split(args)
    clip = held[0]
    t = common.scene_frames(clip, args.frames, args.stride)[0]
    print(f"input: {clip} frame {t}", flush=True)
    results = {}
    for n_cams in args.cams:
        payload = common.b2d_payload(clip, t, n_cams)
        rows = []
        for i in range(args.warmup + args.runs):
            torch.manual_seed(i)
            torch.cuda.manual_seed_all(i)
            row, cot, _ = run_once(model, processor, payload, args, timer)
            tag = "warm" if i < args.warmup else "run "
            print(f"[{n_cams} cam] {tag} {i}: total={row['total']:.0f} ms "
                  f"tokens={row['decode_steps']} gpu={row['mhz']}MHz cot={cot[:90]!r}", flush=True)
            if i >= args.warmup:
                rows.append(row)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[n_cams] = {"mean": mean, "rows": rows}
        print(f"\n== {n_cams} camera(s) x {common.NUM_FRAMES} frames, prompt {mean['prompt_len']:.0f} "
              f"tokens, {mean['decode_steps']:.1f} generated, {args.steps} Euler steps ==")
        for k in ("preprocess", "vit", "prefill", "decode", "lm_head", "gen_overhead",
                  "diffusion", "to_traj", "other", "total"):
            extra = ""
            if k == "decode" and mean["decode_steps"]:
                extra = f"   ({mean['decode'] / mean['decode_steps']:.1f} ms/token)"
            if k == "diffusion":
                extra = (f"   (expert {mean['expert']:.1f} ms in {mean['expert_calls']:.0f} calls)")
            print(f"  {k:<13}{mean[k]:8.1f} ms  {100 * mean[k] / mean['total']:5.1f} %{extra}")
        print(flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
