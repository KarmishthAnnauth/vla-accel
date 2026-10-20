"""Step-by-step ablation of the SimLingo fast path: LoRA merge + SDPA, KV cache
(eager decode), CUDA-graph decode.  Model forward only, same input as
simlingo_fast_check.py, each step compared with the untouched model.
Run inside the sim_ros container with clocks pinned.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import importlib.util
from simlingo_fast_check import build_example, build_model, gpu_mhz

_CONV = ("/benchmarking/simlingo/pretrained/InternVL2-1B/models--OpenGVLab--InternVL2-1B/"
         "snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187/conversation.py")
_spec = importlib.util.spec_from_file_location


def _patched_spec(name, location=None, *a, **k):
    if isinstance(location, str) and location.endswith("/pretrained/InternVL2-1B/conversation.py"):
        location = _CONV
    return _spec(name, location, *a, **k)


importlib.util.spec_from_file_location = _patched_spec


def bench(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    model, cfg, tok, cache_dir, variant = build_model()
    example, n_tok = build_example(cfg, tok, cache_dir, variant)
    print(f"prompt tokens: {n_tok}   gpu: {gpu_mhz()} MHz", flush=True)

    def run(fn):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ms = bench(lambda: fn(example))
            s, r, l = fn(example)
        return ms, s, r, l

    ms, b_s, b_r, b_l = run(model)
    rows = [("baseline", ms, 0.0, 0.0, True)]

    def diff(s, r, l):
        return (float((r.float() - b_r.float()).abs().max()),
                float((s.float() - b_s.float()).abs().max()),
                l[0].strip() == b_l[0].strip())

    from simlingo_training.models import fast_inference as fi
    assert fi._merge_lora(model)
    assert fi._force_sdpa(model.language_model.model)
    ms, s, r, l = run(model.forward)
    rows.append(("+ LoRA merge, SDPA", ms, *diff(s, r, l)))

    eager = fi._FastForward(model, max_cache_len=768, use_cuda_graph=False,
                            max_new_tokens=100)
    ms, s, r, l = run(eager)
    rows.append(("+ KV cache (eager decode)", ms, *diff(s, r, l)))

    graphed = fi._FastForward(model, max_cache_len=768, use_cuda_graph=True,
                              max_new_tokens=100)
    ms, s, r, l = run(graphed)
    assert graphed._graph is not None, "graph capture failed"
    rows.append(("+ CUDA-graph decode", ms, *diff(s, r, l)))

    print(f"gpu after: {gpu_mhz()} MHz   language: {b_l[0]!r}")
    print(f"{'step':28s} {'mean ms':>8s} {'sd':>6s} {'route m':>8s} {'speed m':>8s} text")
    for name, (m, sd), rd, sp, same in rows:
        print(f"{name:28s} {m:8.1f} {sd:6.1f} {rd:8.4f} {sp:8.4f} {same}")


if __name__ == "__main__":
    main()
