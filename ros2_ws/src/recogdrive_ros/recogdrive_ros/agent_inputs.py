#!/usr/bin/env python3

"""Everything between the ROS topics and ``navsim.common.dataclasses.AgentInput``.

ReCogDrive is evaluated on NAVSIM: ``run_pdm_score_recogdrive.py`` loads an
``AgentInput`` from the logs with ``AgentInput.from_scene_dict_list`` and hands
it to ``ReCogDriveAgent.compute_trajectory``.  The ROS node calls that same
``compute_trajectory`` unmodified, so the only thing that has to be rebuilt here
is the ``AgentInput`` -- and it is rebuilt with NAVSIM's own conventions:

  * 4 ego statuses, 0.5 s apart (``num_history_frames: 4`` at 2 Hz), the last
    one being "now";
  * ``ego_pose`` of each status relative to the last one, through NAVSIM's own
    ``convert_absolute_to_relative_se2_array`` (x forward, y left, heading CCW),
    cast to float32 exactly like ``from_scene_dict_list``;
  * ``ego_velocity`` / ``ego_acceleration`` = (x, y) in the ego frame
    (``ego_dynamic_state[:2]`` / ``[2:]``);
  * ``driving_command`` = one-hot over [left, straight, right, unknown];
  * ``cameras[-1].cam_f0.image`` = the PATH of the front image
    (``load_image_path=True`` in the eval script): the agent opens it itself.

No ROS and no torch in this module, and the NAVSIM imports are deferred to
``build_agent_input``, so the sampling logic is unit-testable anywhere.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

NAVSIM_INTERVAL_SEC: float = 0.5
NUM_HISTORY_FRAMES: int = 4
TRAJECTORY_HORIZON_SEC: float = 4.0
NUM_TRAJECTORY_POSES: int = 8

CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT, CMD_UNKNOWN = 0, 1, 2, 3
COMMAND_NAMES = ["left", "straight", "right", "unknown"]

ROAD_OPTION_TO_COMMAND = {
    1: CMD_LEFT,
    2: CMD_RIGHT,
    3: CMD_STRAIGHT,
    4: CMD_STRAIGHT,
    5: CMD_LEFT,
    6: CMD_RIGHT,
}


def road_option_to_command(road_option: int) -> int:
    """CARLA RoadOption int -> NAVSIM command index (VOID / garbage -> straight)."""
    return ROAD_OPTION_TO_COMMAND.get(int(road_option), CMD_STRAIGHT)


def command_one_hot(command: int) -> np.ndarray:
    """The 4-element one-hot NAVSIM stores per frame as ``driving_command``."""
    if command not in (CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT):
        raise ValueError(f"command must be left/straight/right (0/1/2), got {command}")
    one_hot = np.zeros(4, dtype=np.int64)
    one_hot[command] = 1
    return one_hot


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseHistory:
    """Time-stamped ego samples, newest last, for cutting NAVSIM's 2 Hz history
    out of a 20 Hz odometry stream.

    Each sample is (t, x, y, yaw, vx, vy): the pose in the odometry (map) frame
    and the velocity in the vehicle frame.  Stamps are the odometry header
    stamps (simulation time), the same clock as the camera stamps the history
    is cut against.
    """

    def __init__(self, max_age_sec: float = 6.0) -> None:
        self._max_age = float(max_age_sec)
        self._samples: List[Tuple[float, float, float, float, float, float]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples = []

    @property
    def oldest_time(self) -> Optional[float]:
        return self._samples[0][0] if self._samples else None

    @property
    def newest_time(self) -> Optional[float]:
        return self._samples[-1][0] if self._samples else None

    def add(self, t: float, x: float, y: float, yaw: float, vx: float, vy: float) -> bool:
        """Append a sample.  Returns True when the buffer had to be restarted
        because time ran backwards (the simulator was reset under the node)."""
        restarted = False
        if self._samples:
            last_t = self._samples[-1][0]
            if t < last_t:
                self._samples = []
                restarted = True
            elif t == last_t:
                self._samples[-1] = (t, x, y, yaw, vx, vy)
                return False
        self._samples.append((float(t), float(x), float(y), float(yaw), float(vx), float(vy)))
        cutoff = t - self._max_age
        drop = 0
        while drop < len(self._samples) - 1 and self._samples[drop][0] < cutoff:
            drop += 1
        if drop:
            del self._samples[:drop]
        return restarted

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """(pose [x, y, yaw], velocity [vx, vy]) at time t: linear between the
        two neighbouring samples (yaw along the shorter arc), clamped to the
        buffer's ends."""
        if not self._samples:
            raise ValueError("pose history is empty")
        s = self._samples
        if t <= s[0][0]:
            a = s[0]
            return np.array(a[1:4], dtype=np.float64), np.array(a[4:6], dtype=np.float64)
        if t >= s[-1][0]:
            a = s[-1]
            return np.array(a[1:4], dtype=np.float64), np.array(a[4:6], dtype=np.float64)
        hi = len(s) - 1
        while s[hi - 1][0] > t:
            hi -= 1
        a, b = s[hi - 1], s[hi]
        u = (t - a[0]) / (b[0] - a[0])
        yaw = normalize_angle(a[3] + u * normalize_angle(b[3] - a[3]))
        pose = np.array([a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), yaw], dtype=np.float64)
        vel = np.array([a[4] + u * (b[4] - a[4]), a[5] + u * (b[5] - a[5])], dtype=np.float64)
        return pose, vel

    def window(self, t_ref: float, num_frames: int = NUM_HISTORY_FRAMES,
               interval: float = NAVSIM_INTERVAL_SEC) -> Tuple[np.ndarray, np.ndarray, int]:
        """NAVSIM's history at ``t_ref``: ``num_frames`` samples ``interval``
        apart, oldest first, the last one at ``t_ref``.

        Returns (poses (n, 3), velocities (n, 2), n_held) where ``n_held`` is the
        number of leading frames older than the buffer: those repeat the oldest
        pose there is, i.e. the ego is taken to have been standing where it
        was first seen.
        """
        if not self._samples:
            raise ValueError("pose history is empty")
        oldest = self._samples[0][0]
        poses = np.zeros((num_frames, 3), dtype=np.float64)
        vels = np.zeros((num_frames, 2), dtype=np.float64)
        held = 0
        for i in range(num_frames):
            t = t_ref - (num_frames - 1 - i) * interval
            if t < oldest - 1e-3:
                held += 1
            poses[i], vels[i] = self.sample(t)
        return poses, vels, held

    def finite_difference_velocity(self, t: float, span: float = 0.2) -> Optional[np.ndarray]:
        """Ego-frame velocity at ``t`` from the poses alone, over ``span``
        seconds.  A cross-check on the twist the odometry reports; None when
        the buffer does not cover the span."""
        if len(self._samples) < 2 or t - span < self._samples[0][0] - 1e-3:
            return None
        p0, _ = self.sample(t - span)
        p1, _ = self.sample(t)
        world = (p1[:2] - p0[:2]) / span
        c, s = math.cos(p1[2]), math.sin(p1[2])
        return np.array([c * world[0] + s * world[1], -s * world[0] + c * world[1]], dtype=np.float64)


def shift_along_heading(pose: np.ndarray, offset_x: float) -> np.ndarray:
    """The pose of a point ``offset_x`` metres ahead (+) of / behind (-) the
    given pose along its own heading.  NAVSIM's ego pose is the rear axle; a
    simulator's odometry origin usually is not."""
    if not offset_x:
        return np.array(pose, dtype=np.float64)
    return np.array([pose[0] + offset_x * math.cos(pose[2]),
                     pose[1] + offset_x * math.sin(pose[2]),
                     pose[2]], dtype=np.float64)


def build_agent_input(global_poses: np.ndarray, velocities: np.ndarray,
                      acceleration: np.ndarray, driving_command: np.ndarray,
                      image_path: str):
    """The ``AgentInput`` ``AgentInput.from_scene_dict_list`` would build for
    this instant: same pose conversion, same dtypes, the front image as a path.

    global_poses    (4, 3) float64, oldest first, last = now, (x, y, heading)
    velocities      (4, 2) ego-frame (vx, vy) per frame
    acceleration    (2,)   ego-frame (ax, ay) now.  The feature builder reads
                           the dynamics of the LAST status only; the earlier
                           statuses carry the same acceleration.
    driving_command (4,)   one-hot, see ``command_one_hot``
    """
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    global_poses = np.array(global_poses, dtype=np.float64)
    local_ego_poses = convert_absolute_to_relative_se2_array(
        StateSE2(*global_poses[-1]), global_poses)

    ego_statuses = [
        EgoStatus(
            ego_pose=np.array(local_ego_poses[i], dtype=np.float32),
            ego_velocity=np.array(velocities[i], dtype=np.float32),
            ego_acceleration=np.array(acceleration, dtype=np.float32),
            driving_command=np.array(driving_command),
        )
        for i in range(len(global_poses))
    ]
    empty = [Cameras(*[Camera() for _ in range(8)]) for _ in range(len(global_poses) - 1)]
    current = Cameras(Camera(image=image_path), *[Camera() for _ in range(7)])
    return AgentInput(ego_statuses=ego_statuses, cameras=empty + [current], lidars=[])


def trajectory_to_world(poses: np.ndarray, frame_pose: np.ndarray) -> np.ndarray:
    """Ego-frame (x, y, heading) rows -> the map frame, given the pose of that
    ego frame in the map."""
    poses = np.asarray(poses, dtype=np.float64)
    c, s = math.cos(frame_pose[2]), math.sin(frame_pose[2])
    out = np.empty_like(poses)
    out[:, 0] = frame_pose[0] + c * poses[:, 0] - s * poses[:, 1]
    out[:, 1] = frame_pose[1] + s * poses[:, 0] + c * poses[:, 1]
    out[:, 2] = poses[:, 2] + frame_pose[2]
    return out


def world_to_frame(world_poses: np.ndarray, frame_pose: np.ndarray) -> np.ndarray:
    """Inverse of ``trajectory_to_world``."""
    world_poses = np.asarray(world_poses, dtype=np.float64)
    c, s = math.cos(frame_pose[2]), math.sin(frame_pose[2])
    dx = world_poses[:, 0] - frame_pose[0]
    dy = world_poses[:, 1] - frame_pose[1]
    out = np.empty_like(world_poses)
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    out[:, 2] = np.arctan2(np.sin(world_poses[:, 2] - frame_pose[2]),
                           np.cos(world_poses[:, 2] - frame_pose[2]))
    return out


def segment_speeds(points_xy: np.ndarray, start_xy: np.ndarray,
                   interval: float = NAVSIM_INTERVAL_SEC) -> np.ndarray:
    """Speed implied by each waypoint: the distance from the previous one (the
    planning origin for the first) over the 0.5 s between them."""
    pts = np.vstack((np.asarray(start_xy, dtype=np.float64)[None, :2],
                     np.asarray(points_xy, dtype=np.float64)[:, :2]))
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / interval
