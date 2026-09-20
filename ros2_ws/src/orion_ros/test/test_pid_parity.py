#!/usr/bin/env python3
"""Verify orion_withpid_node._compute_control reproduces OrionAgent.run_step.

The reference below is transcribed from Orion/team_code/orion_b2d_agent.py's
run_step control block. Both sides drive the SAME PIDController class, each with
its own instance so the stateful 40-sample windows evolve independently but
identically.
"""
import sys
import numpy as np
from builtin_interfaces.msg import Time

sys.path.insert(0, "/root/Orion")
from team_code.pid_controller import PIDController
from orion_ros.orion_withpid_node import OrionWithPidRosNode, ORION_SPEED_CAP_MPS


class _Log:
    def warn(self, *a, **k): pass
    def info(self, *a, **k): pass


class _Stub:
    """The node, minus everything _compute_control does not touch."""
    def __init__(self):
        self._pid = PIDController()
    def get_logger(self):
        return _Log()


def agent_reference(pid, ego_fut_preds, speed, local_command_xy):
    """Verbatim from OrionAgent.run_step."""
    steer_traj, throttle_traj, brake_traj, _ = pid.control_pid(
        ego_fut_preds, np.float64(speed), local_command_xy)
    if brake_traj < 0.05:
        brake_traj = 0.0
    if throttle_traj > brake_traj:
        brake_traj = 0.0
    if speed > 5:
        throttle_traj = 0
    return (float(np.clip(float(steer_traj), -1, 1)),
            float(np.clip(float(throttle_traj), 0, 0.75)),
            float(np.clip(float(brake_traj), 0, 1)))


def main():
    rng = np.random.default_rng(0)
    stub = _Stub()
    ref_pid = PIDController()

    speeds = [0.0, 1.0, 3.0, 4.9, 5.0, 5.1, 8.0, 12.0]
    failures = 0
    checked = 0
    capped = 0

    for i in range(200):
        base = np.cumsum(rng.uniform(0.0, 1.2, size=(6, 2)) * np.array([0.3, 1.0]), axis=0)
        speed = speeds[i % len(speeds)]
        near = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20)])
        ego_xy = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5)])
        ego_theta = rng.uniform(-np.pi, np.pi)

        raw_theta = np.pi / 2.0 - ego_theta
        near_world = np.array([near[0] - ego_xy[0], near[1] - ego_xy[1]])
        rot = np.array([[np.cos(raw_theta), -np.sin(raw_theta)],
                        [np.sin(raw_theta),  np.cos(raw_theta)]])
        local_command_xy = rot @ near_world

        meta = {"near_node": near, "ego_xy": ego_xy, "ego_theta": ego_theta,
                "speed": speed, "stamp": Time()}

        got = OrionWithPidRosNode._compute_control(stub, base, meta)
        exp = agent_reference(ref_pid, base, speed, local_command_xy)

        checked += 1
        if speed > ORION_SPEED_CAP_MPS:
            capped += 1
        mine = (got.steer, got.throttle, got.brake)
        if not np.allclose(mine, exp, atol=1e-9):
            failures += 1
            if failures <= 3:
                print(f"  MISMATCH i={i} speed={speed}: node={mine} agent={exp}")

    print(f"checked {checked} frames ({capped} above the {ORION_SPEED_CAP_MPS} m/s cap)")
    if failures:
        print(f"FAIL: {failures} mismatches")
        return 1
    print("PASS: node control == agent control on every frame")

    over = [s for s in speeds if s > ORION_SPEED_CAP_MPS]
    assert over, "test did not exercise the cap"
    pid2 = PIDController()
    stub2 = _Stub()
    traj = np.cumsum(np.tile([0.0, 1.5], (6, 1)), axis=0)
    m = {"near_node": np.array([0.0, 30.0]), "ego_xy": np.zeros(2),
         "ego_theta": np.pi / 2, "speed": 9.0, "stamp": Time()}
    c = OrionWithPidRosNode._compute_control(stub2, traj, m)
    assert c.throttle == 0.0, f"speed cap not applied: throttle={c.throttle}"
    print("PASS: throttle forced to 0 above the speed cap")
    del pid2
    return 0


if __name__ == "__main__":
    sys.exit(main())
