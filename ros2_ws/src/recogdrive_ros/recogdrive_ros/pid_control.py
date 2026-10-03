"""Closed-loop control for recogdrive_node: the Bench2Drive PID on ReCogDrive's
plan, run the way orion_withpid_node runs it.

ReCogDrive's release has no controller (NAVSIM scores the plan in a
non-reactive simulation), so this is the Bench2Drive agents' control step:
``PIDController.control_pid`` (loaded from the user's checkout) every tick, then the brake rules and the 5 m/s
throttle cut of ``orion_b2d_agent.run_step`` / ``minddrive_b2d_agent.run_step``.

Frames.  ReCogDrive plans in NAVSIM's ego frame (x forward, y left, 8 poses
0.5 s apart).  ``control_pid`` reads waypoints as (x right, y forward) -- its
steering angle is ``pi/2 - atan2(y, x)``, positive to the right like CARLA's
steer -- hence ``to_pid_frame``.
"""
from __future__ import annotations

import importlib.util
import math
import os
from typing import Optional, Tuple

import numpy as np

from recogdrive_ros.agent_inputs import NAVSIM_INTERVAL_SEC, trajectory_to_world, world_to_frame

B2D_SPEED_CAP_MPS = 5.0
B2D_MAX_THROTTLE = 0.75


def load_pid_controller(team_code_dir: str):
    """The PIDController class from ``<team_code_dir>/pid_controller.py`` -- the
    Bench2Drive controller as shipped in the ORION / MindDrive checkouts
    (``Orion/team_code``).  Loaded from the user's checkout instead of being
    copied here: the file is Bench2Drive's (CC BY-NC-ND 4.0)."""
    path = os.path.join(team_code_dir, "pid_controller.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no pid_controller.py in {team_code_dir!r}: point pid_controller_dir at the "
            f"team_code directory of an ORION / MindDrive / Bench2Drive checkout")
    spec = importlib.util.spec_from_file_location("b2d_pid_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIDController


def rebase_plan(plan_xy: np.ndarray, age: float, plan_frame: np.ndarray,
                ego_now: np.ndarray, dt: float = NAVSIM_INTERVAL_SEC) -> np.ndarray:
    """The plan re-cut from where the ego is NOW, in its current frame
    (orion_withpid_node._rebase_plan).

    plan_xy     (n, 2) planned positions in the ego frame of the camera frame
    age         seconds since that camera frame
    plan_frame  (x, y, yaw) of that ego frame in the map
    ego_now     (x, y, yaw) of the ego in the map now

    Time: element j is where the model expected to be at ``age + dt * (j + 1)``
    -- the plan is already one inference old when it arrives, and
    ``control_pid`` reads its target speed from the distance to the first
    waypoint.  Past the plan's horizon the last segment's velocity is kept.
    Frame: the points are moved into the current ego frame with odometry, so
    the geometry stays relative to where the car actually is.
    """
    plan_xy = np.asarray(plan_xy, dtype=np.float64)
    knots = np.vstack((np.zeros((1, 2)), plan_xy))
    horizon = dt * (len(knots) - 1)
    v_end = (knots[-1] - knots[-2]) / dt

    def sample(t: float) -> np.ndarray:
        if t >= horizon:
            return knots[-1] + (t - horizon) * v_end
        u = max(t, 0.0) / dt
        i = min(int(math.floor(u)), len(knots) - 2)
        return knots[i] + (u - i) * (knots[i + 1] - knots[i])

    pts = np.array([sample(age + dt * (j + 1)) for j in range(len(plan_xy))])
    poses = np.column_stack((pts, np.zeros(len(pts))))
    return world_to_frame(trajectory_to_world(poses, plan_frame), ego_now)[:, :2]


def to_pid_frame(xy: np.ndarray) -> np.ndarray:
    """(x forward, y left) -> control_pid's (x right, y forward)."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack((-xy[..., 1], xy[..., 0]), axis=-1)


def compute_control(pid, waypoints_ego: np.ndarray, speed: float,
                    target_ego: Optional[np.ndarray] = None,
                    speed_cap_mps: float = B2D_SPEED_CAP_MPS) -> Tuple[float, float, float, dict]:
    """(steer, throttle, brake, pid metadata) for waypoints in the current ego
    frame (x forward, y left).  orion_b2d_agent.run_step after control_pid.

    target_ego: the next route point in the same frame.  control_pid only
    reports it (use_target_to_aim is False there); straight ahead if None.
    """
    waypoints = to_pid_frame(waypoints_ego)
    target = to_pid_frame(np.asarray(target_ego, dtype=np.float64)) if target_ego is not None \
        else np.array([0.0, 10.0])
    steer, throttle, brake, metadata = pid.control_pid(waypoints, np.float64(speed), target)
    steer = float(np.clip(steer, -1.0, 1.0))
    throttle = float(np.clip(throttle, 0.0, B2D_MAX_THROTTLE))
    brake = float(brake)
    if brake < 0.05:
        brake = 0.0
    if throttle > brake:
        brake = 0.0
    if speed_cap_mps > 0.0 and float(speed) > speed_cap_mps:
        throttle = 0.0
    return steer, throttle, brake, metadata
