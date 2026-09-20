#!/usr/bin/env python3
"""Compare the TensorRT ViT engine against the PyTorch backbone on the tensors
saved by export_vit_onnx.py (`*_parity.pt`), and time it.

  docker exec orion_ros bash -c 'cd /root/Orion && python3 \
      /benchmarking/alpamayo-autoware/src/orion_ros/tools/check_vit_engine.py \
      --engine /benchmarking/alpamayo-autoware/src/orion_ros/engines/orion_vit_fp16.plan'
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--parity", default="/benchmarking/alpamayo-autoware/src/orion_ros/engines/orion_vit_parity.pt")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    from orion_ros.orion_speedups import TrtBackbone

    d = torch.load(args.parity, map_location="cuda")
    x, y_ref, y_exp = d["x"], d["y_ref"], d["y_exp"]
    bb = TrtBackbone(args.engine, print)
    y = bb(x)[0].float()

    def rel(a, b):
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
