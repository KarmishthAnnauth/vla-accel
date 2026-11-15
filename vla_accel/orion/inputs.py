"""ORION input batches from Bench2Drive clips: six cameras, can_bus, command and calibration.

The results dict matches the reference agent (team_code/orion_b2d_agent.py::run_step).
"""
from __future__ import annotations

import os
import time
from typing import List, Sequence, Tuple

import numpy as np

from vla_accel.common import b2d

CAMERA_ORDER: List[str] = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
JPEG_QUALITY = 20  # the agent re-encodes every camera at q20

LIDAR2IMG = {
    "CAM_FRONT": np.array([[1.14251841e03, 8.00000000e02, 0.00000000e00, -9.52000000e02],
                           [0.00000000e00, 4.50000000e02, -1.14251841e03, -8.09704417e02],
                           [0.00000000e00, 1.00000000e00, 0.00000000e00, -1.19000000e00],
                           [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_FRONT_LEFT": np.array([[6.03961325e-14, 1.39475744e03, 0.00000000e00, -9.20539908e02],
                                [-3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_FRONT_RIGHT": np.array([[1.31064327e03, -4.77035138e02, 0.00000000e00, -4.06010608e02],
                                 [3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                                 [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_BACK": np.array([[-5.60166031e02, -8.00000000e02, 0.00000000e00, -1.28800000e03],
                          [5.51091060e-14, -4.50000000e02, -5.60166031e02, -8.58939847e02],
                          [1.22464680e-16, -1.00000000e00, 0.00000000e00, -1.61000000e00],
                          [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_BACK_LEFT": np.array([[-1.14251841e03, 8.00000000e02, 0.00000000e00, -6.84385123e02],
                               [-4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
                               [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
    "CAM_BACK_RIGHT": np.array([[3.60989788e02, -1.34723223e03, 0.00000000e00, -1.04238127e02],
                                [4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]),
}
LIDAR2CAM = {
    "CAM_FRONT": np.array([[1., 0., 0., 0.], [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19], [0., 0., 0., 1.]]),
    "CAM_FRONT_LEFT": np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),
    "CAM_FRONT_RIGHT": np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),
    "CAM_BACK": np.array([[-1., 0., 0., 0.], [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61], [0., 0., 0., 1.]]),
    "CAM_BACK_LEFT": np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),
    "CAM_BACK_RIGHT": np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]]),
}
LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])


def command2hot(command: int, max_dim: int = 6) -> np.ndarray:
    """One-hot of a CARLA RoadOption 1..6 (invalid -> LANEFOLLOW)."""
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot


def command2nohot(command: int, max_dim: int = 6) -> int:
    """Zero-based index of a CARLA RoadOption 1..6 (invalid -> LANEFOLLOW)."""
    if command < 0:
        command = 4
    command -= 1
    return command


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid ego pose."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def load_camera(record: Tuple[str, str, int], jpeg_quality: int = JPEG_QUALITY) -> np.ndarray:
    """One BGR camera image for a (clip, cam, idx) record."""
    clip, cam, idx = record
    return b2d.load_image(clip, cam, idx, jpeg_quality)


def load_frame(clip: str, idx: int, jpeg_quality: int = JPEG_QUALITY) -> Tuple[List[np.ndarray], dict]:
    """Six BGR images in ORION order and the annotation of one clip frame."""
    imgs = [b2d.load_image(clip, c, idx, jpeg_quality) for c in CAMERA_ORDER]
    return imgs, b2d.load_anno(clip, idx)


def can_bus_and_pose(anno: dict) -> Tuple[np.ndarray, np.ndarray]:
    """18-d can_bus and lidar-to-global pose from a B2D annotation, as the agent builds them."""
    from pyquaternion import Quaternion

    raw_theta = anno["theta"] if not np.isnan(anno["theta"]) else 0.0
    ego_theta = -raw_theta + np.pi / 2
    can_bus = np.zeros(18)
    can_bus[0], can_bus[1] = anno["x"], -anno["y"]
    can_bus[3:7] = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
    can_bus[7] = anno["speed"]
    can_bus[10:13] = anno["acceleration"]
    can_bus[11] *= -1
    can_bus[13:16] = -np.asarray(anno["angular_velocity"])
    can_bus[16], can_bus[17] = ego_theta, ego_theta / np.pi * 180
    ego2world = np.eye(4)
    ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
    ego2world[0:2, 3] = can_bus[0:2]
    return can_bus, ego2world @ LIDAR2EGO


def make_results(imgs: Sequence[np.ndarray], anno: dict, frame_idx: int) -> dict:
    """The agent's raw results dict, before the inference pipeline."""
    from mmcv.core.bbox import get_box_type

    can_bus, l2g = can_bus_and_pose(anno)
    cmd = int(anno.get("command_near", 4))
    res = {
        "lidar2img": np.stack([LIDAR2IMG[c] for c in CAMERA_ORDER]),
        "lidar2cam": np.stack([LIDAR2CAM[c] for c in CAMERA_ORDER]),
        "cam_intrinsic": [LIDAR2IMG[c] @ np.linalg.inv(LIDAR2CAM[c]) for c in CAMERA_ORDER],
        "img": list(imgs), "folder": " ", "scene_token": " ", "frame_idx": frame_idx,
        "timestamp": frame_idx / b2d.B2D_HZ, "box_type_3d": get_box_type("LiDAR")[0],
        "can_bus": can_bus, "command": command2nohot(cmd), "ego_fut_cmd": command2hot(cmd),
        "ego_pose": l2g, "ego_pose_inv": invert_matrix_egopose_numpy(l2g), "lidar2ego": LIDAR2EGO,
        "l2g_r_mat": l2g[0:3, 0:3], "l2g_t": l2g[0:3, 3],
    }
    res["img_shape"] = res["ori_shape"] = res["pad_shape"] = np.stack(imgs, axis=-1).shape
    return res


def build_batch(pipeline, imgs: Sequence[np.ndarray], anno: dict, frame_idx: int):
    """Pipeline, collate and move to GPU; returns (batch, {stage: ms})."""
    import torch
    from mmcv.parallel.collate import collate

    res = make_results(imgs, anno, frame_idx)
    t0 = time.perf_counter()
    res = pipeline(res)
    t1 = time.perf_counter()
    batch = collate([res], samples_per_gpu=1)
    t2 = time.perf_counter()
    for key, data in batch.items():
        if key != "img_metas" and torch.is_tensor(data[0]):
            data[0] = data[0].to("cuda")
        if key == "input_ids":
            for i in range(len(data[0])):
                for k in range(len(data[0][i])):
                    data[0][i][k] = data[0][i][k].to("cuda")
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    return batch, {"pipeline": (t1 - t0) * 1e3, "collate": (t2 - t1) * 1e3, "to_device": (t3 - t2) * 1e3}


def select_clips(b2d_dir: str, names: str = "", split: str = "all") -> List[str]:
    """Clips under b2d_dir: all, the calibration or held-out split, or a comma list of names/indices."""
    if names:
        clips = b2d.list_clips(b2d_dir)
        out = []
        for n in names.split(","):
            n = n.strip()
            out.append(clips[int(n)] if n.isdigit() else os.path.join(b2d_dir, n))
        return out
    if split in ("calib", "held"):
        calib, held = b2d.split_clips(b2d_dir)
        chosen = calib if split == "calib" else held
        if chosen:
            return chosen
    return b2d.list_clips(b2d_dir)  # too few clips for a split


def add_b2d_args(ap, frames: int, clips: str = "", split: str = "all") -> None:
    """Shared --b2d-dir / --clips / --split / --frames / --stride / --start options."""
    ap.add_argument("--b2d-dir", default=b2d.B2D_DIR, help="folder of Bench2Drive clips")
    ap.add_argument("--clips", default=clips, help="comma list of clip names or indices (overrides --split)")
    ap.add_argument("--split", choices=["all", "calib", "held"], default=split)
    ap.add_argument("--frames", type=int, default=frames, help="frames per clip")
    ap.add_argument("--stride", type=int, default=10, help="annotation frames apart (10 = 1 s)")
    ap.add_argument("--start", type=int, default=10, help="first annotation frame")


def frame_plan(args) -> List[Tuple[str, List[int]]]:
    """[(clip, frame indices)] from the shared B2D options."""
    clips = select_clips(os.path.abspath(args.b2d_dir), args.clips, args.split)
    return [(c, b2d.frame_indices(c, args.frames, args.stride, args.start)) for c in clips]

