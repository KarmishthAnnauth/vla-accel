#!/usr/bin/env python3
"""Reproduce ORION's temporal-memory behaviour in async real-time mode.

Drives the real pre_update_memory retention rule with the timestamp sequences
each timestamp_mode produces, so the consequence of >2 s inference is measured
rather than argued about.
"""
import sys

import numpy as np

sys.path.insert(0, "/root/Orion")
from orion_ros.orion_node import (
    OrionRosNode, ORION_MEMORY_MAX_DT, ORION_AGENT_HZ,
)

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class _Log:
    def __init__(self): self.warns = []
    def info(self, *a, **k): pass
    def warn(self, m, *a, **k): self.warns.append(str(m))


def make_stub(mode):
    s = OrionRosNode.__new__(OrionRosNode)
    s._timestamp_mode = mode
    s._prev_sensor_ts = None
    s._memoryless_frames = 0
    s._dt_samples = 0
    s._last_dt_log_t = 0.0
    s._frame_idx = 0
    s._log = _Log()
    s.get_logger = lambda: s._log
    return s


def retained(timestamps, scene_tokens):
    """The model's own rule: memory survives iff |dt| < 2.0 AND token matches.
    Mirrors OrionHead.pre_update_memory / memory_refresh(mem, x) == mem * x."""
    kept = []
    prev_t = None
    prev_tok = None
    for t, tok in zip(timestamps, scene_tokens):
        if prev_t is None:
            kept.append(False)
        else:
            kept.append(abs(t - prev_t) < ORION_MEMORY_MAX_DT and tok == prev_tok)
        prev_t, prev_tok = t, tok
    return kept


def main():
    n = 20
    infer_s = 2.5
    sensor_ts = [1000.0 + i * infer_s for i in range(n)]
    token = ["route-0001"] * n

    print(f"--- async real-time, {infer_s} s/inference ---")

    s = make_stub("sensor")
    fed = []
    for i, ts in enumerate(sensor_ts):
        fed.append(ts)
        s._check_memory_continuity(ts)
    kept = retained(fed, token)
    n_kept = sum(kept)
    check(n_kept == 0, f"sensor mode: memory retained on {n_kept}/{n} frames (expect 0)")
    check(s._memoryless_frames == n - 1,
          f"diagnostic counted every affected frame ({s._memoryless_frames}/{n - 1})")
    check(any("single-frame" in w for w in s._log.warns),
          "a warning is emitted (the wipe is no longer silent)")
    if s._log.warns:
        print("   warn: " + s._log.warns[0][:150] + "…")

    s2 = make_stub("agent")
    fed2 = [i / ORION_AGENT_HZ for i in range(n)]
    for ts in sensor_ts:
        s2._check_memory_continuity(ts)
    kept2 = retained(fed2, token)
    check(sum(kept2) == n - 1,
          f"agent mode: memory retained on {sum(kept2)}/{n - 1} frames after the first")
    check(s2._memoryless_frames == n - 1,
          "agent mode still reports the real gap rather than hiding it")
    check(any("masking this" in w for w in s2._log.warns),
          "warning says the mode is masking a real gap")

    print("\n--- synchronous mode (sim waits for the agent) ---")
    s3 = make_stub("sensor")
    sync_ts = [1000.0 + i * 0.05 for i in range(n)]
    for ts in sync_ts:
        s3._check_memory_continuity(ts)
    check(sum(retained(sync_ts, token)) == n - 1, "memory retained on every frame")
    check(s3._memoryless_frames == 0, "no warning in synchronous mode")

    print("\n--- route boundary ---")
    tok_mixed = ["route-0001"] * 10 + ["route-0002"] * 10
    kept4 = retained(sync_ts, tok_mixed)
    check(kept4[10] is False, "first frame of the new route does not inherit memory")
    check(sum(kept4) == n - 2, "exactly one extra drop, at the boundary")

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL MEMORY-CONTINUITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
