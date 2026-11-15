"""NAVSIM ``AgentInput`` for ReCogDrive, built from Bench2Drive clips.

Ego history, velocity, acceleration and command come from the 10 Hz annos; the front image is written to a file.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from vla_accel.common import b2d

NAVSIM_INTERVAL_SEC: float = 0.5
NUM_HISTORY_FRAMES: int = 4
TRAJECTORY_HORIZON_SEC: float = 4.0
NUM_TRAJECTORY_POSES: int = 8
NAVSIM_IMAGE_SIZE = (1920, 1080)
HISTORY_STRIDE = int(round(NAVSIM_INTERVAL_SEC * b2d.B2D_HZ))

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
    """CARLA RoadOption -> NAVSIM command index (VOID / unknown -> straight)."""
    return ROAD_OPTION_TO_COMMAND.get(int(road_option), CMD_STRAIGHT)


def command_one_hot(command: int) -> np.ndarray:
    """The 4-element one-hot NAVSIM stores as ``driving_command``."""
    if command not in (CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT):
        raise ValueError(f"command must be left/straight/right (0/1/2), got {command}")
    one_hot = np.zeros(4, dtype=np.int64)
    one_hot[command] = 1
    return one_hot


def build_agent_input(global_poses: np.ndarray, velocities: np.ndarray,
                      acceleration: np.ndarray, driving_command: np.ndarray,
                      image_path: str):
    """The AgentInput ``AgentInput.from_scene_dict_list`` builds: same pose conversion and dtypes, image as a path."""
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


def front_image(clip: str, idx: int, jpeg_quality: int = 0) -> np.ndarray:
    """B2D rgb_front as RGB uint8, centre-cropped to 16:9 and resized to NAVSIM's 1920x1080."""
    rgb = cv2.cvtColor(b2d.load_image(clip, "CAM_FRONT", idx, jpeg_quality), cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    w, h = im.size
    cw, ch = min(w, int(h * 16 / 9)), min(h, int(w * 9 / 16))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    if (cw, ch) != (w, h):
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
    if im.size != NAVSIM_IMAGE_SIZE:
        im = im.resize(NAVSIM_IMAGE_SIZE, Image.BICUBIC)
    return np.asarray(im)


def write_frame(image_rgb: np.ndarray, directory: str, name: str) -> str:
    """Write the frame as a lossless BMP the agent opens itself; returns an ASCII path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.bmp")
    Image.fromarray(image_rgb).save(path)
    return path


def ego_state(clip: str, idx: int) -> dict:
    """NAVSIM ego history at frame ``idx``: 4 world poses 0.5 s apart, ego-frame velocities and acceleration, command."""
    span = HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1) + 1
    poses = b2d.ego_history(clip, idx, span)[::HISTORY_STRIDE]
    frames = [max(k, 0) for k in range(idx - span + 1, idx + 1)][::HISTORY_STRIDE]
    annos = [b2d.load_anno(clip, k) for k in frames]
    velocities = np.array([[float(a.get("speed", 0.0)), 0.0] for a in annos], dtype=np.float64)
    acc = annos[-1].get("acceleration")
    # CARLA's ego frame is left-handed (y right); NAVSIM's y points left.
    acceleration = np.array([acc[0], -acc[1]], dtype=np.float64) if acc is not None else np.zeros(2)
    road_option = annos[-1].get("command_near", 4)
    return {"poses": poses, "velocities": velocities, "acceleration": acceleration,
            "road_option": int(road_option), "command": road_option_to_command(road_option),
            "held": sum(1 for k in range(idx - span + 1, idx + 1, HISTORY_STRIDE) if k < 0)}


def make_input(clip: str, idx: int, frame_dir: str, jpeg_quality: int = 0,
               image: Optional[np.ndarray] = None):
    """(AgentInput, front RGB array) for one B2D frame; the image is also written under ``frame_dir``."""
    if image is None:
        image = front_image(clip, idx, jpeg_quality)
    state = ego_state(clip, idx)
    path = write_frame(image, frame_dir, f"{os.path.basename(os.path.normpath(clip))}_{idx:05d}")
    agent_input = build_agent_input(state["poses"], state["velocities"], state["acceleration"],
                                    command_one_hot(state["command"]), path)
    return agent_input, image
