#!/usr/bin/env python3
"""What a different vision encoder does to the *plan*: the fast path with its
ViT stage replaced, against the unmodified agent, same input, same noise.

  vit_e2e.py <engine> [<engine> ...]
"""
import glob
import sys

import numpy as np
import torch

import common as C
from trt_vit import Engine

agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
fast = FastReCogDrive(agent, log=lambda m: None)
frames = sorted(glob.glob(C.os.path.join(C.os.path.dirname(C.os.path.abspath(__file__)), "frames/*.bmp")))
cases = [(f, v, c, y) for f in frames for (v, c, y) in ((0.0, "straight", 0.0), (6.0, "left", 0.2), (12.0, "straight", 0.0), (9.0, "right", -0.25))]
inputs = [C.make_input(f, speed=v, command=c, yaw_rate=y) for f, v, c, y in cases]


def plans(seed0=100):
    out = []
    for k, ai in enumerate(inputs):
        torch.manual_seed(seed0 + k)
        out.append(np.asarray(fast.plan(ai).poses, dtype=np.float64))
    return np.stack(out)


ref = []
for k, ai in enumerate(inputs):
    torch.manual_seed(100 + k)
    ref.append(np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64))
ref = np.stack(ref)
other = []
for k, ai in enumerate(inputs):
    torch.manual_seed(500 + k)
    other.append(np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64))
other = np.stack(other)


def report(name, p):
    d = np.linalg.norm(p[:, :, :2] - ref[:, :, :2], axis=-1)
    print(f"{name:34s} mean {d.mean():.3f} m | final-pose mean {d[:, -1].mean():.3f} max {d[:, -1].max():.3f} m | "
          f"worst pose {d.max():.3f} m | heading max {np.abs(p[:, :, 2] - ref[:, :, 2]).max():.4f} rad")


print(f"{len(inputs)} cases (3 frames x 4 ego states); deviation of the planned xy from the reference plan, same noise")
report("reference, other noise (scale)", other)
report("fast path, PyTorch ViT (exact)", plans())
stock = fast._vit
for path in sys.argv[1:]:
    eng = Engine(path)
    fast._vit = lambda: eng(fast._pixels.float()).to(torch.bfloat16)
    report(C.os.path.basename(path), plans())
    fast._vit = stock
    del eng
