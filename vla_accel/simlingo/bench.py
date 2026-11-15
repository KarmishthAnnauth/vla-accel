"""Correctness and speed of ``fast_inference`` against the untouched SimLingo model on Bench2Drive frames.

    python3 -m vla_accel.simlingo.bench --b2d-dir /data/b2d --clip 0 --frames 5
"""

import argparse

import torch

from vla_accel.common import jetson
from vla_accel.simlingo import inputs
from vla_accel.simlingo.fast_inference import optimize_for_inference, verify


def _run(model, ex):
    """One forward under bf16 autocast."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(ex)


def main():
    ap = inputs.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--iters", type=int, default=6, help="timed calls per path")
    args = ap.parse_args()
    jetson.warn_if_unpinned()

    print("loading model...", flush=True)
    model, ctx = inputs.build_model(args)
    frames = list(inputs.examples(args, ctx))
    tag0, ex0, n_tok0 = frames[0]
    print(f"{len(frames)} frames, first {tag0}: {n_tok0} prompt tokens   gpu: {inputs.gpu_mhz()} MHz\n")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_ms, base_sd = inputs.timeit(lambda: model(ex0), n=args.iters)
    base = [_run(model, ex) for _, ex, _ in frames]
    print(f"BASELINE (untouched)   {base_ms:8.1f} ms (sd {base_sd:.1f})   language={base[0][2][0]!r}")

    report = optimize_for_inference(model)
    print(f"\noptimize_for_inference -> {report}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fast_ms, fast_sd = inputs.timeit(lambda: model(ex0), n=args.iters)
    fast = [_run(model, ex) for _, ex, _ in frames]
    print(f"OPTIMISED              {fast_ms:8.1f} ms (sd {fast_sd:.1f})   language={fast[0][2][0]!r}")
    print(f"  cuda graph active: {model._fast_inference._graph is not None}")

    print("\nper frame vs BASELINE (includes the bf16 LoRA-merge perturbation):")
    print(f"  {'frame':28s} {'tokens':>6s} {'route m':>8s} {'speed m':>8s} text")
    worst_r = worst_s = 0.0
    all_text = True
    for (tag, _, nt), (bs, br, bl), (fs, fr, fl) in zip(frames, base, fast):
        rd = float((fr.float() - br.float()).abs().max())
        sd = float((fs.float() - bs.float()).abs().max())
        same = fl[0].strip() == bl[0].strip()
        worst_r, worst_s, all_text = max(worst_r, rd), max(worst_s, sd), all_text and same
        print(f"  {tag:28s} {nt:6d} {rd:8.4f} {sd:8.4f} {same}")
    print(f"  max route |diff| {worst_r:.4f} m   max speed |diff| {worst_s:.4f} m   "
          f"text identical: {all_text}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = verify(model, ex0)
    print(f"\nfast vs merged original forward (cache + graph only): route {v['route_max_diff_m']:.4f} m, "
          f"speed {v['speed_max_diff_m']:.4f} m, text match {v['language_match']}, ok {v['ok']}")

    print(f"\n  SPEEDUP: {base_ms / fast_ms:.2f}x   ({base_ms:.0f} -> {fast_ms:.0f} ms)")

    print("\nrepeat-call stability (first frame, 5 calls):")
    firsts = []
    for i in range(5):
        _, r, l = _run(model, ex0)
        firsts.append(float(r[0, 0, 0]))
        print(f"  call {i}: route[0,0]={r[0, 0, 0].float():+.5f}  lang={l[0]!r}")
    print(f"  all identical: {len(set(f'{x:.6f}' for x in firsts)) == 1}")

    print("\nre-run after the other frames (stale-cache check across prompt lengths):")
    for (tag, ex, nt), (_, fr, _) in zip(frames, fast):
        _, r, _ = _run(model, ex)
        print(f"  {tag:28s} tokens={nt}  identical: {torch.equal(r, fr)}")


if __name__ == "__main__":
    main()
