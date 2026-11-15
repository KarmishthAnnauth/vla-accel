#!/usr/bin/env python3
"""Stage profile of the unmodified ReCogDrive compute_trajectory on Bench2Drive frames.

Each stage is timed with CUDA sync on both sides; ``other`` is wall time no stage accounts for.
"""
import argparse
import json
import statistics
import time

import torch

from vla_accel.common import jetson
from vla_accel.recogdrive import common as C


class TokProxy:
    """Tokenizer wrapper that times each call as ``tokenize``."""

    def __init__(self, t, st):
        self.__dict__["_t"] = t
        self.__dict__["_st"] = st

    def __getattr__(self, n):
        return getattr(self._t, n)

    def __setattr__(self, n, v):
        setattr(self._t, n, v)

    def __call__(self, *a, **k):
        with self._st("tokenize"):
            return self._t(*a, **k)


def main():
    p = C.add_b2d_args(argparse.ArgumentParser(description=__doc__), frames=1, clips=1)
    p.add_argument("--runs", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--json", default="")
    args = p.parse_args()
    jetson.warn_if_unpinned()

    cases = C.load_inputs(args)
    agent = C.build_agent()
    st = C.Stages()

    import navsim.agents.recogdrive.recogdrive_agent as RA
    load_image = RA.load_image

    def timed_load_image(*a, **k):
        with st("load_image"):
            return load_image(*a, **k)
    RA.load_image = timed_load_image

    bb = agent.backbone
    vlm = bb.model
    bb.tokenizer = TokProxy(bb.tokenizer, st)
    st.wrap(vlm, "extract_feature", "vit+mlp1")
    st.wrap(vlm.language_model.model, "forward", "llm_body")
    st.wrap(vlm.language_model.lm_head, "forward", "lm_head")
    st.wrap(agent.action_head, "get_action", "planner")

    rows = []
    for i in range(args.runs):
        _, agent_input, _ = cases[i % len(cases)]
        st.t = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.compute_trajectory(agent_input)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
        row = dict(st.t)
        row["wall"] = wall
        row["other"] = wall - sum(st.t.values())
        row["gpu_mhz"] = jetson.gpu_mhz()
        rows.append(row)
        print(f"run {i}: " + " ".join(f"{k}={v:.0f}" for k, v in row.items()))

    keep = rows[args.warmup:]
    print(f"\nmean over {len(keep)} runs (first {args.warmup} discarded):")
    summary = {}
    wall_mean = statistics.mean([r["wall"] for r in keep])
    for k in keep[0]:
        vals = [r[k] for r in keep]
        summary[k] = statistics.mean(vals)
        if k != "gpu_mhz":
            print(f"  {k:12s} {summary[k]:8.1f} ms  {100 * summary[k] / wall_mean:5.1f} %  (sd {statistics.pstdev(vals):.1f})")
    print("gpu clock", summary["gpu_mhz"], "MHz; GPU peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "mean": summary}, f, indent=1)


if __name__ == "__main__":
    main()
