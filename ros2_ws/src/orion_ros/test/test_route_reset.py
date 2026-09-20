#!/usr/bin/env python3
"""Verify per-route context isolation in the ORION nodes.

Checks the boundary logic without loading the 38 GB checkpoint: a fake model
records whether forward_test would reset its heads (test_flag cleared), and the
route callback / reset method are driven directly.
"""
import sys
from collections import deque

import numpy as np
from geometry_msgs.msg import Pose
from carla_msgs.msg import CarlaRoute

sys.path.insert(0, "/root/Orion")
from team_code.pid_controller import PIDController
from orion_ros.orion_withpid_node import OrionWithPidRosNode
from orion_ros.orion_node import OrionRosNode

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass


class _FakeModel:
    """Stands in for the Orion detector. forward_test resets both heads exactly
    once per test_flag transition, as mmcv/models/detectors/orion.py does."""
    def __init__(self):
        self.test_flag = False
        self.memory = None
        self.resets = 0

    def forward_once(self, scene_token):
        if not self.test_flag:
            self.memory = None
            self.resets += 1
            self.test_flag = True
        if self.memory is not None and self.memory != scene_token:
            self.memory = None
        self.memory = scene_token


def make_stub(cls, with_pid):
    s = cls.__new__(cls)
    s._route = deque()
    s._frame_idx = 7
    s._last_ref_stamp = (1, 2)
    s._scene_token = "route-0000"
    s._route_serial = 0
    s._route_fingerprint = None
    s._pending_route_reset = False
    s._model = _FakeModel()
    s._driving_command = 4
    if with_pid:
        s._pid = PIDController()
    s.get_logger = lambda: _Log()
    return s


def route_msg(x0, x1, n=5):
    m = CarlaRoute()
    for i in range(n):
        p = Pose()
        p.position.x = float(x0 + i * (x1 - x0) / max(n - 1, 1))
        p.position.y = 0.0
        m.poses.append(p)
        m.road_options.append(4)
    return m


def run(cls, label, with_pid):
    print(f"\n--- {label} ---")
    s = make_stub(cls, with_pid)

    cls._route_callback(s, route_msg(0, 100))
    check(s._pending_route_reset, "new route arms a reset")
    tok1 = s._scene_token
    check(tok1 != "route-0000", f"scene token advanced ({tok1})")

    s._pending_route_reset = False
    cls._route_callback(s, route_msg(0, 100))
    check(not s._pending_route_reset, "identical replayed plan does NOT re-arm")
    check(s._scene_token == tok1, "scene token unchanged on replay")

    cls._route_callback(s, route_msg(500, 900))
    check(s._pending_route_reset, "different route re-arms a reset")
    tok2 = s._scene_token
    check(tok2 != tok1, f"scene token advanced again ({tok2})")

    s._model.forward_once(tok1)
    check(s._model.memory == tok1, "memory populated during route 1")
    if with_pid:
        pid_before = s._pid
        for _ in range(40):
            s._pid.turn_controller.step(0.9)
        check(abs(np.mean(s._pid.turn_controller._window)) > 0.5,
              "PID integral window is biased at end of route 1")
    s._frame_idx = 123

    cls._reset_for_new_route(s)
    check(not s._pending_route_reset, "reset clears the pending flag")
    check(s._frame_idx == 0, "frame counter restarts at 0")
    check(s._last_ref_stamp is None, "reference stamp cleared")
    check(s._model.test_flag is False, "test_flag cleared -> heads reset next forward")
    if with_pid:
        check(s._pid is not pid_before, "PID controller rebuilt (fresh object)")
        check(abs(np.mean(s._pid.turn_controller._window)) < 1e-9,
              "PID integral window is clean for route 2")

    resets_before = s._model.resets
    s._model.forward_once(tok2)
    check(s._model.resets == resets_before + 1, "forward_test performed the reset")
    check(s._model.memory == tok2, "memory now belongs to route 2 only")


def main():
    run(OrionWithPidRosNode, "orion_withpid_node (PID)", with_pid=True)
    run(OrionRosNode, "orion_node (Stanley)", with_pid=False)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL ROUTE-ISOLATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
