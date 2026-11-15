"""Compare a TensorRT ViT engine with the PyTorch backbone on export_vit_onnx.py's parity tensors, and time it."""
from __future__ import annotations

import argparse
import os
import time

import torch

from vla_accel.common import jetson
from vla_accel.orion import model as om


def main() -> None:
    """Load the engine, print feature parity and per-iteration latency."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=os.path.join(om.ENGINES_DIR, "orion_vit_fp16.plan"))
    ap.add_argument("--parity", default=os.path.join(om.ENGINES_DIR, "orion_vit_parity.pt"))
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    jetson.warn_if_unpinned()
    from vla_accel.orion.speedups import TrtBackbone

    d = torch.load(args.parity, map_location="cuda")
    x, y_ref, y_exp = d["x"], d["y_ref"], d["y_exp"]
    bb = TrtBackbone(args.engine, print)
    y = bb(x)[0].float()

    def rel(a, b):
        """Max abs and relative L2 difference."""
        return float((a - b).abs().max()), float((a - b).norm() / b.norm())
    print(f"engine vs ORION fp16/flash path: max|d|={rel(y, y_ref)[0]:.4f} rel={rel(y, y_ref)[1]:.2e}")
    print(f"engine vs fp32 export graph:     max|d|={rel(y, y_exp)[0]:.4f} rel={rel(y, y_exp)[1]:.2e}")
    print(f"(reference spread fp32 vs fp16:  rel={rel(y_exp, y_ref)[1]:.2e})")
    for _ in range(3):
        bb(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.iters):
        bb(x)
    torch.cuda.synchronize()
    print(f"engine forward: {(time.perf_counter() - t) * 1e3 / args.iters:.1f} ms/iter (incl. output clone)")


if __name__ == "__main__":
    main()
