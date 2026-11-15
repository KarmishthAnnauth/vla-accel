#!/usr/bin/env python3
"""FastReCogDrive vs the unmodified agent on Bench2Drive frames, same diffusion noise.

Reports hidden-state and trajectory differences, fallback, repeatability and speed; ``--noise-floor`` adds a per-stage deviation breakdown.
"""
import argparse
import os
import statistics
import time

import numpy as np
import torch
from PIL import Image

from vla_accel.common import jetson
from vla_accel.recogdrive import common as C
from vla_accel.recogdrive import inputs as I


def rel(a, b):
    """Relative L2, row-cosine and max-abs difference of two tensors, as a string."""
    a, b = a.float(), b.float()
    row = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (f"rel-L2 {((a - b).norm() / b.norm()).item():.4f}  row-cos min {row.min().item():.5f} "
            f"mean {row.mean().item():.6f}  max|d| {(a - b).abs().max().item():.3f} (|ref| max {b.abs().max().item():.0f})")


def noise_floor(agent, fast, cases):
    """Stage-by-stage deviation of the fast path against the reference and the reference's own bf16 noise floor."""
    lm = agent.backbone.model.language_model
    seen = {}
    orig_lm = lm.forward

    def spy_lm(*a, **k):
        seen["embeds"], seen["mask"] = k["inputs_embeds"].clone(), k["attention_mask"]
        return orig_lm(*a, **k)
    orig_ga = agent.action_head.get_action

    def spy_ga(vl, ai_, *a, **k):
        seen["vl"] = vl.clone()
        seen["ai"] = ai_
        return orig_ga(vl, ai_, *a, **k)
    lm.forward, agent.action_head.get_action = spy_lm, spy_ga
    try:
        for k, (name, ai, _) in enumerate(cases[:2]):
            torch.manual_seed(k)
            ref = np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64)
            vl_ref, embeds, mask = seen["vl"], seen["embeds"], seen["mask"]
            n_pad = int((mask == 0).sum())
            real = embeds[:, n_pad:]
            torch.manual_seed(k)
            out = np.asarray(fast.compute_trajectory(ai).poses, dtype=np.float64)
            vl_fast = fast.hidden_state().clone()
            print(f"\n=== {name}: {real.shape[1]} real tokens, {n_pad} pads")
            win = fast._win[0, :real.shape[1] - fast._n_prefix]
            print("LLM input embeddings (image+tail):", "bit-identical" if torch.equal(win, real[0, fast._n_prefix:])
                  else rel(win, real[0, fast._n_prefix:]))
            with torch.no_grad():
                alt = lm.model(inputs_embeds=real, output_hidden_states=True, return_dict=True).hidden_states[-1]
            print("reference padded vs stock unpadded (noise floor):", rel(alt[0], vl_ref[0, n_pad:]))
            print("fast vs reference padded                        :", rel(vl_fast[0, n_pad:], vl_ref[0, n_pad:]))
            print("fast vs stock unpadded                          :", rel(vl_fast[0, n_pad:], alt[0]))
            print("  prefix rows only, fast vs reference           :",
                  rel(vl_fast[0, n_pad:n_pad + fast._n_prefix], vl_ref[0, n_pad:n_pad + fast._n_prefix]))
            print("  prefix rows only, stock unpadded vs reference :",
                  rel(alt[0, :fast._n_prefix], vl_ref[0, n_pad:n_pad + fast._n_prefix]))
            if n_pad:
                print("  padding rows, fast vs reference               :", rel(vl_fast[0, :n_pad], vl_ref[0, :n_pad]),
                      "| reference pad rows all equal:", bool((vl_ref[0, :n_pad] == vl_ref[0, :1]).all()))
            fast._ctx[0, :vl_ref.shape[1]] = vl_ref[0]
            torch.manual_seed(k)
            for i in range(fast._steps + 1):
                fast._noise[i] = torch.randn((1, 8, 3), device="cuda")
            p = fast._planner().float().cpu().numpy()[0]
            print(f"planner alone (reference hidden states in): max|d| {np.abs(p - ref).max():.2e}")
            with torch.no_grad():
                torch.manual_seed(k)
                vl_alt = torch.cat([vl_ref[:, :n_pad], alt.float()], dim=1)
                t_alt = orig_ga(vl_alt, seen["ai"])["pred_traj"].float().cpu().numpy()[0]
            print(f"trajectory: fast vs reference {np.abs(out[:, :2] - ref[:, :2]).max():.4f} m | "
                  f"stock-unpadded LLM + reference planner vs reference {np.abs(t_alt[:, :2] - ref[:, :2]).max():.4f} m")
    finally:
        lm.forward, agent.action_head.get_action = orig_lm, orig_ga
        fast._ctx_layout = (-1, -1)  # the context was overwritten; force a full relayout


def main():
    p = C.add_b2d_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--eager", action="store_true", help="no CUDA graphs")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--no-ref-bench", action="store_true")
    p.add_argument("--vit-engine", default="", help="opt-in TensorRT ViT engine")
    p.add_argument("--noise-floor", action="store_true", help="also print the per-stage deviation breakdown")
    args = p.parse_args()
    jetson.warn_if_unpinned()

    cases = C.load_inputs(args)
    agent = C.build_agent()
    from vla_accel.recogdrive.speedups import FastReCogDrive
    from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

    t0 = time.time()
    fast = FastReCogDrive(agent, capture=not args.eager, log=print, profile=True, vit_engine=args.vit_engine)
    print(f"fast path built in {time.time() - t0:.1f} s; GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

    seen = {}
    orig_get_action = agent.action_head.get_action

    def spy(vl, *a, **k):
        seen["vl"] = vl.detach().clone()
        return orig_get_action(vl, *a, **k)
    agent.action_head.get_action = spy

    print("\ncase                        speed cmd       tokens  pads | hidden max|d|  mean|d|   cos    "
          "| traj max|d| xy [m]  heading [rad] | pixels")
    worst_xy = worst_h = 0.0
    for k, (name, ai, _) in enumerate(cases):
        torch.manual_seed(100 + k)
        ref = np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64)
        vl_ref = seen["vl"]
        torch.manual_seed(100 + k)
        out = np.asarray(fast.compute_trajectory(ai).poses, dtype=np.float64)
        vl = fast.hidden_state()
        assert fast.fallbacks == 0, "fast path fell back"
        assert vl.shape == vl_ref.shape, (vl.shape, vl_ref.shape)
        n_pad, n_real = fast._ctx_layout
        d = (vl.float() - vl_ref.float()).abs()
        cos = torch.nn.functional.cosine_similarity(vl.float().flatten(), vl_ref.float().flatten(), dim=0).item()
        px_ref = load_image(str(ai.cameras[-1].cam_f0.image)).cuda().to(torch.bfloat16)
        px_same = bool((px_ref == fast._pixels).all())
        dxy = np.abs(out[:, :2] - ref[:, :2]).max()
        dh = np.abs(out[:, 2] - ref[:, 2]).max()
        worst_xy, worst_h = max(worst_xy, dxy), max(worst_h, dh)
        status = ai.ego_statuses[-1]
        cmd = I.COMMAND_NAMES[int(np.argmax(status.driving_command))]
        print(f"{name:27s} {np.linalg.norm(status.ego_velocity):5.1f} {cmd:8s}  {n_real:5d}  {n_pad:3d} | "
              f"{d.max().item():9.4f}    {d.mean().item():.5f}  {cos:.6f} | {dxy:12.4f}        {dh:9.5f}    | "
              f"{'identical' if px_same else 'DIFFER'}")
        if n_pad:
            dp = (vl[0, :n_pad].float() - vl_ref[0, :n_pad].float()).abs().max().item()
            print(f"      padding rows: max|d| {dp:.5f}")
    print(f"worst trajectory deviation: {worst_xy:.4f} m, {worst_h:.5f} rad")

    name0, ai0, rgb0 = cases[0]
    trajs = []
    for s in range(6):
        torch.manual_seed(s)
        trajs.append(np.asarray(agent.compute_trajectory(ai0).poses, dtype=np.float64))
    spread = np.abs(np.stack(trajs)[:, :, :2] - np.mean(trajs, axis=0)[None, :, :2]).max()
    print(f"for scale: the reference's own seed-to-seed spread on one input is {spread:.3f} m (diffusion sampling noise)")

    odd = I.write_frame(np.asarray(Image.fromarray(rgb0).resize((1024, 768))), args.frame_dir, "odd_4x3")
    ai_odd = I.build_agent_input(
        [s.ego_pose for s in ai0.ego_statuses], [s.ego_velocity for s in ai0.ego_statuses],
        ai0.ego_statuses[-1].ego_acceleration, ai0.ego_statuses[-1].driving_command, odd)
    torch.manual_seed(3)
    r_ref = np.asarray(agent.compute_trajectory(ai_odd).poses)
    torch.manual_seed(3)
    r_fast = np.asarray(fast.compute_trajectory(ai_odd).poses)
    print(f"4:3 frame: fast.plan -> {fast.plan(ai_odd)}, compute_trajectory used the reference "
          f"({'same result' if np.array_equal(r_ref, r_fast) else 'DIFFERENT'}; fallbacks={fast.fallbacks})")
    os.remove(odd)
    fast.fallbacks = 0

    _, ai1, rgb1 = cases[1 % len(cases)]
    torch.manual_seed(5)
    p_file = np.asarray(fast.plan(ai1).poses)
    torch.manual_seed(5)
    p_arr = np.asarray(fast.plan(ai1, image=rgb1).poses)
    print("frame passed as array vs read from file:", "bit-identical" if np.array_equal(p_file, p_arr) else "DIFFERS")
    print("verify():", fast.verify(ai1))

    torch.manual_seed(7)
    r1 = np.asarray(fast.compute_trajectory(ai0).poses)
    torch.manual_seed(8)
    fast.compute_trajectory(ai1)
    torch.manual_seed(7)
    r2 = np.asarray(fast.compute_trajectory(ai0).poses)
    print("repeat after a different input:", "bit-identical" if np.array_equal(r1, r2) else f"DIFFERS {np.abs(r1 - r2).max()}")

    agent.action_head.get_action = orig_get_action
    if args.noise_floor:
        noise_floor(agent, fast, cases)

    def bench(fn, n):
        rows = []
        for i in range(n + 3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(ai0)
            torch.cuda.synchronize()
            rows.append((time.perf_counter() - t0) * 1e3)
        return rows[3:]
    stage = {}

    def fast_fn(x):
        fast.plan(x, image=rgb0)
        for k_, v_ in fast.timing.items():
            stage.setdefault(k_, []).append(v_)
    fr = bench(fast_fn, args.runs)
    print(f"\nfast : mean {statistics.mean(fr):7.1f} ms  sd {statistics.pstdev(fr):.1f}  min {min(fr):.1f}  "
          f"max {max(fr):.1f}   (gpu {jetson.gpu_mhz()} MHz)")
    print("       stages: " + "  ".join(f"{k_}={statistics.mean(v_[3:]):.1f}" for k_, v_ in stage.items()))
    if not args.no_ref_bench:
        rr = bench(agent.compute_trajectory, max(6, args.runs // 2))
        print(f"ref  : mean {statistics.mean(rr):7.1f} ms  sd {statistics.pstdev(rr):.1f}  min {min(rr):.1f}  max {max(rr):.1f}")
        print(f"speedup {statistics.mean(rr) / statistics.mean(fr):.2f}x")
    print("GPU peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")


if __name__ == "__main__":
    main()
