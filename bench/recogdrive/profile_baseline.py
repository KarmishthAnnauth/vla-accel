#!/usr/bin/env python3
"""Stage-level profile of the unmodified ReCogDrive compute_trajectory.

Every stage is timed with torch.cuda.synchronize() on both sides; `other` is
the wall time the stages do not account for.
"""
import argparse
import json
import statistics
import time

import torch

import common as C

p = argparse.ArgumentParser()
p.add_argument("--image", default=C.os.path.join(C.os.path.dirname(__file__), "frames/carla_0.bmp"))
p.add_argument("--runs", type=int, default=12)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--json", default="")
args = p.parse_args()

agent = C.build_agent()
st = C.Stages()

import navsim.agents.recogdrive.recogdrive_agent as RA
RA.load_image = (lambda f: (lambda *a, **k: st_call("load_image", f, *a, **k)))(RA.load_image)


def st_call(name, fn, *a, **k):
    with st(name):
        return fn(*a, **k)


bb = agent.backbone
vlm = bb.model
tok = bb.tokenizer
orig_tok_call = type(tok).__call__
st.wrap(bb, "tokenizer", "tokenize") if False else None


class TokProxy:
    def __init__(self, t): self.__dict__["_t"] = t
    def __getattr__(self, n): return getattr(self._t, n)
    def __setattr__(self, n, v): setattr(self._t, n, v)
    def __call__(self, *a, **k):
        with st("tokenize"):
            return self._t(*a, **k)


bb.tokenizer = TokProxy(tok)
st.wrap(vlm, "extract_feature", "vit+mlp1")
st.wrap(vlm.language_model.model, "forward", "llm_body")
st.wrap(vlm.language_model.lm_head, "forward", "lm_head")
st.wrap(agent.action_head, "get_action", "planner")
orig_fb = agent.get_feature_builders

agent_input = C.make_input(args.image)
rows = []
info = {}
for i in range(args.runs):
    st.t = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    traj = agent.compute_trajectory(agent_input)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    row = dict(st.t)
    row["wall"] = wall
    row["other"] = wall - sum(st.t.values())
    row["gpu_mhz"] = C.gpu_mhz()
    rows.append(row)
    print(f"run {i}: " + " ".join(f"{k}={v:.0f}" for k, v in row.items()))

keep = rows[args.warmup:]
print(f"\nmean over {len(keep)} runs (first {args.warmup} discarded):")
summary = {}
for k in keep[0]:
    vals = [r[k] for r in keep]
    summary[k] = statistics.mean(vals)
    if k != "gpu_mhz":
        print(f"  {k:12s} {summary[k]:8.1f} ms  {100 * summary[k] / statistics.mean([r['wall'] for r in keep]):5.1f} %  (sd {statistics.pstdev(vals):.1f})")
print("gpu clock", summary["gpu_mhz"], "MHz; GPU peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
if args.json:
    json.dump({"rows": rows, "mean": summary}, open(args.json, "w"), indent=1)
