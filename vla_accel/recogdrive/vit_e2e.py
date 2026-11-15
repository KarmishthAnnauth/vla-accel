#!/usr/bin/env python3
"""Plan deviation caused by a TensorRT vision encoder: fast path with the engine vs the unmodified agent, same noise.

  python3 -m vla_accel.recogdrive.vit_e2e <engine> [<engine> ...]
"""
import argparse
import os

import numpy as np
import torch

from vla_accel.recogdrive import common as C
from vla_accel.recogdrive.trt_vit import Engine


def main():
    p = C.add_b2d_args(argparse.ArgumentParser(description=__doc__), frames=3, clips=4)
    p.add_argument("engines", nargs="*")
    args = p.parse_args()

    inputs = [ai for _, ai, _ in C.load_inputs(args)]
    agent = C.build_agent()
    from vla_accel.recogdrive.speedups import FastReCogDrive
    fast = FastReCogDrive(agent, log=lambda m: None)

    def plans(seed0=100):
        out = []
        for k, ai in enumerate(inputs):
            torch.manual_seed(seed0 + k)
            out.append(np.asarray(fast.plan(ai).poses, dtype=np.float64))
        return np.stack(out)

    def reference(seed0):
        out = []
        for k, ai in enumerate(inputs):
            torch.manual_seed(seed0 + k)
            out.append(np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64))
        return np.stack(out)

    ref, other = reference(100), reference(500)

    def report(name, p):
        d = np.linalg.norm(p[:, :, :2] - ref[:, :, :2], axis=-1)
        print(f"{name:34s} mean {d.mean():.3f} m | final-pose mean {d[:, -1].mean():.3f} max {d[:, -1].max():.3f} m | "
              f"worst pose {d.max():.3f} m | heading max {np.abs(p[:, :, 2] - ref[:, :, 2]).max():.4f} rad")

    print(f"{len(inputs)} Bench2Drive frames; deviation of the planned xy from the reference plan, same noise")
    report("reference, other noise (scale)", other)
    report("fast path, PyTorch ViT (exact)", plans())
    stock = fast._vit
    for path in args.engines:
        eng = Engine(path)
        fast._vit = lambda: eng(fast._pixels.float()).to(torch.bfloat16)
        report(os.path.basename(path), plans())
        fast._vit = stock
        del eng


if __name__ == "__main__":
    main()
