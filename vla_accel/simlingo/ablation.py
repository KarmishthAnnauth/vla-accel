"""Cumulative ablation of the SimLingo fast path on one Bench2Drive frame.

Rows: baseline, + LoRA merge and SDPA, + KV cache (eager decode), + CUDA-graph decode; each vs the baseline.
"""

import argparse

import torch

from vla_accel.common import jetson
from vla_accel.simlingo import fast_inference as fi
from vla_accel.simlingo import inputs


def main():
    ap = inputs.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    jetson.warn_if_unpinned()

    model, ctx = inputs.build_model(args)
    tag, example, n_tok = inputs.first_example(args, ctx)
    print(f"frame {tag}: {n_tok} prompt tokens   gpu: {inputs.gpu_mhz()} MHz", flush=True)

    def run(fn):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ms = inputs.timeit(lambda: fn(example), n=args.iters, warmup=3)
            s, r, l = fn(example)
        return ms, s, r, l

    ms, b_s, b_r, b_l = run(model)
    rows = [("baseline", ms, 0.0, 0.0, True)]

    def diff(s, r, l):
        return (float((r.float() - b_r.float()).abs().max()),
                float((s.float() - b_s.float()).abs().max()),
                l[0].strip() == b_l[0].strip())

    assert fi._merge_lora(model)
    assert fi._force_sdpa(model.language_model.model)
    ms, s, r, l = run(model.forward)
    rows.append(("+ LoRA merge, SDPA", ms, *diff(s, r, l)))

    eager = fi._FastForward(model, max_cache_len=768, use_cuda_graph=False, max_new_tokens=100)
    ms, s, r, l = run(eager)
    rows.append(("+ KV cache (eager decode)", ms, *diff(s, r, l)))

    graphed = fi._FastForward(model, max_cache_len=768, use_cuda_graph=True, max_new_tokens=100)
    ms, s, r, l = run(graphed)
    assert graphed._graph is not None, "graph capture failed"
    rows.append(("+ CUDA-graph decode", ms, *diff(s, r, l)))

    print(f"gpu after: {inputs.gpu_mhz()} MHz   language: {b_l[0]!r}")
    print(f"{'step':28s} {'mean ms':>8s} {'sd':>6s} {'route m':>8s} {'speed m':>8s} text")
    for name, (m, sd), rd, sp, same in rows:
        print(f"{name:28s} {m:8.1f} {sd:6.1f} {rd:8.4f} {sp:8.4f} {same}")


if __name__ == "__main__":
    main()
