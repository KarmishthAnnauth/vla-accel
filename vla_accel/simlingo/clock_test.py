"""GPU clock (DVFS) effect on the untouched SimLingo model, one Bench2Drive frame.

A: cold clock back-to-back; B: after a 15 s matmul spin; C: with a 1.5 s idle gap between calls.
Run it once with clocks unpinned and once after `sudo jetson_clocks` to see the governor's share.
"""

import argparse
import time

import torch

from vla_accel.simlingo import inputs


def main():
    ap = inputs.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--spin-s", type=float, default=15.0)
    ap.add_argument("--gap-s", type=float, default=1.5)
    args = ap.parse_args()

    model, ctx = inputs.build_model(args)
    tag, example, _ = inputs.first_example(args, ctx)
    dev = ctx.device

    def infer():
        torch.cuda.synchronize()
        a = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(example)
        torch.cuda.synchronize()
        return (time.perf_counter() - a) * 1000

    print(f"\nframe {tag}")
    print(f"--- A. cold clock, back-to-back (no warm-up spin) ---   start gpu={inputs.gpu_mhz()}MHz")
    for i in range(8):
        t = infer()
        print(f"   {i}: {t:7.0f} ms   gpu={inputs.gpu_mhz():5d} MHz")

    print(f"\n--- B. force the clock up with a {args.spin_s:.0f}s matmul spin ---")
    spin = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < args.spin_s:
        for _ in range(20):
            torch.mm(spin, spin)
    torch.cuda.synchronize()
    print(f"   gpu after spin = {inputs.gpu_mhz()} MHz")
    for i in range(5):
        t = infer()
        print(f"   {i}: {t:7.0f} ms   gpu={inputs.gpu_mhz():5d} MHz")

    print(f"\n--- C. duty cycle: {args.gap_s}s idle gap between inferences ---")
    for i in range(6):
        time.sleep(args.gap_s)
        t = infer()
        print(f"   {i}: {t:7.0f} ms   gpu={inputs.gpu_mhz():5d} MHz")


if __name__ == "__main__":
    main()
