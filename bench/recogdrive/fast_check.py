#!/usr/bin/env python3
"""Correctness and speed of FastReCogDrive against the unmodified agent.

For each case both paths plan from the same input with the same torch seed, so
they draw the same diffusion noise; what differs is only the arithmetic path.
"""
import argparse
import glob
import statistics
import time

import numpy as np
import torch

import common as C

p = argparse.ArgumentParser()
p.add_argument("--eager", action="store_true", help="no CUDA graphs")
p.add_argument("--runs", type=int, default=20)
p.add_argument("--no-ref-bench", action="store_true")
args = p.parse_args()

agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

t0 = time.time()
fast = FastReCogDrive(agent, capture=not args.eager, log=print, profile=True)
print(f"fast path built in {time.time() - t0:.1f} s; GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

seen = {}
orig_get_action = agent.action_head.get_action
def spy(vl, *a, **k):
    seen["vl"] = vl.detach().clone()
    return orig_get_action(vl, *a, **k)
agent.action_head.get_action = spy

frames = sorted(glob.glob(C.os.path.join(C.os.path.dirname(C.os.path.abspath(__file__)), "frames/*.bmp")))
cases = []
for i, (speed, cmd, yaw) in enumerate([(0.0, "straight", 0.0), (5.0, "straight", 0.0), (8.0, "left", 0.25),
                                       (12.0, "right", -0.2), (20.0, "straight", 0.03), (33.0, "left", 0.4),
                                       (55.0, "right", -0.75), (80.0, "left", 2.5), (95.0, "right", -2.9)]):
    cases.append((frames[i % len(frames)], speed, cmd, yaw))

print("\ncase                               tokens  pads | hidden max|d|  mean|d|   cos    | traj max|d| xy [m]  heading [rad] | pixels")
worst_xy = worst_h = 0.0
for k, (frame, speed, cmd, yaw) in enumerate(cases):
    ai = C.make_input(frame, speed=speed, command=cmd, yaw_rate=yaw)
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
    px_ref = load_image(frame).cuda().to(torch.bfloat16)
    px_same = bool((px_ref == fast._pixels).all())
    dxy = np.abs(out[:, :2] - ref[:, :2]).max(); dh = np.abs(out[:, 2] - ref[:, 2]).max()
    worst_xy, worst_h = max(worst_xy, dxy), max(worst_h, dh)
    print(f"{C.os.path.basename(frame)} v={speed:4.1f} {cmd:8s} yaw={yaw:+.2f}  {n_real:5d}  {n_pad:3d} | "
          f"{d.max().item():9.4f}    {d.mean().item():.5f}  {cos:.6f} | {dxy:12.4f}        {dh:9.5f}    | "
          f"{'identical' if px_same else 'DIFFER'}")
    if n_pad:
        dp = (vl[0, :n_pad].float() - vl_ref[0, :n_pad].float()).abs().max().item()
        print(f"      padding rows: max|d| {dp:.5f}")
print(f"worst trajectory deviation: {worst_xy:.4f} m, {worst_h:.5f} rad")

ai = C.make_input(frames[0], speed=8.0, command="straight")
trajs = []
for s in range(6):
    torch.manual_seed(s)
    trajs.append(np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64))
spread = np.abs(np.stack(trajs)[:, :, :2] - np.mean(trajs, axis=0)[None, :, :2]).max()
print(f"for scale: the reference's own seed-to-seed spread on one input is {spread:.3f} m (diffusion sampling noise)")

from PIL import Image
odd = C.os.path.join(C.os.path.dirname(frames[0]), "odd_4x3.bmp")
Image.open(frames[0]).convert("RGB").resize((1024, 768)).save(odd)
ai_odd = C.make_input(odd, speed=5.0)
torch.manual_seed(3); r_ref = np.asarray(agent.compute_trajectory(ai_odd).poses)
torch.manual_seed(3); r_fast = np.asarray(fast.compute_trajectory(ai_odd).poses)
print(f"4:3 frame: fast.plan -> {fast.plan(ai_odd)}, compute_trajectory used the reference "
      f"({'same result' if np.array_equal(r_ref, r_fast) else 'DIFFERENT'}; fallbacks={fast.fallbacks})")
C.os.remove(odd); fast.fallbacks = 0

rgb = np.asarray(Image.open(frames[1]).convert("RGB"))
ai1 = C.make_input(frames[1], speed=5.0)
torch.manual_seed(5); p_file = np.asarray(fast.plan(ai1).poses)
torch.manual_seed(5); p_arr = np.asarray(fast.plan(ai1, image=rgb).poses)
print("frame passed as array vs read from file:", "bit-identical" if np.array_equal(p_file, p_arr) else "DIFFERS")
print("verify():", fast.verify(ai1))

a = C.make_input(frames[0], speed=5.0); b = C.make_input(frames[1], speed=33.0, command="left", yaw_rate=0.4)
torch.manual_seed(7); r1 = np.asarray(fast.compute_trajectory(a).poses)
torch.manual_seed(8); fast.compute_trajectory(b)
torch.manual_seed(7); r2 = np.asarray(fast.compute_trajectory(a).poses)
print("repeat after a different-length prompt:", "bit-identical" if np.array_equal(r1, r2) else f"DIFFERS {np.abs(r1 - r2).max()}")

agent.action_head.get_action = orig_get_action
ai = C.make_input(frames[0], speed=8.0, command="straight")
def bench(fn, n):
    rows = []
    for i in range(n + 3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(ai); torch.cuda.synchronize()
        rows.append((time.perf_counter() - t0) * 1e3)
    return rows[3:]
stage = {}
rgb8 = np.asarray(Image.open(frames[0]).convert("RGB"))
def fast_fn(x):
    fast.plan(x, image=rgb8)
    for k_, v_ in fast.timing.items():
        stage.setdefault(k_, []).append(v_)
fr = bench(fast_fn, args.runs)
print(f"\nfast : mean {statistics.mean(fr):7.1f} ms  sd {statistics.pstdev(fr):.1f}  min {min(fr):.1f}  max {max(fr):.1f}   (gpu {C.gpu_mhz()} MHz)")
print("       stages: " + "  ".join(f"{k_}={statistics.mean(v_[3:]):.1f}" for k_, v_ in stage.items()))
if not args.no_ref_bench:
    rr = bench(agent.compute_trajectory, max(6, args.runs // 2))
    print(f"ref  : mean {statistics.mean(rr):7.1f} ms  sd {statistics.pstdev(rr):.1f}  min {min(rr):.1f}  max {max(rr):.1f}")
    print(f"speedup {statistics.mean(rr) / statistics.mean(fr):.2f}x")
print("GPU peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
