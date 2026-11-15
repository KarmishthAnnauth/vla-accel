"""Bench2Drive (mini) clip reader shared by every model's bench.

A clip is a directory with ``camera/<cam>/%05d.jpg`` and ``anno/%05d.json.gz``
recorded at 10 Hz. Point ``B2D_DIR`` (or ``--b2d-dir``) at the folder holding
the extracted clips.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

B2D_DIR = os.environ.get("B2D_DIR", "/data/bench2drive-mini")
B2D_HZ = 10.0

# nuScenes-style camera name -> Bench2Drive camera folder.
CAMERAS = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def list_clips(root: str = B2D_DIR) -> List[str]:
    """All clip directories under ``root``, sorted."""
    clips = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "anno")))
    if not clips:
        raise FileNotFoundError(f"no Bench2Drive clips (*/anno) under {root}")
    return clips


def split_clips(root: str = B2D_DIR) -> Tuple[List[str], List[str]]:
    """(calibration clips, held-out clips): every sixth clip, from the third, is held out."""
    clips = list_clips(root)
    held = clips[2::6]
    return [c for c in clips if c not in held], held


def num_frames(clip: str) -> int:
    return len(glob.glob(os.path.join(clip, "anno", "*.json.gz")))


def frame_indices(clip: str, frames: int, stride: int = 10, start: int = 10) -> List[int]:
    """``frames`` indices ``stride`` apart from ``start`` (stride 10 = 1 s at 10 Hz)."""
    return list(range(start, num_frames(clip), stride))[:frames]


def load_anno(clip: str, idx: int) -> dict:
    with gzip.open(os.path.join(clip, "anno", f"{idx:05d}.json.gz")) as f:
        return json.load(f)


def load_image(clip: str, cam: str, idx: int, jpeg_quality: int = 0) -> np.ndarray:
    """One BGR camera image; ``jpeg_quality > 0`` re-encodes it as the CARLA agents do."""
    folder = CAMERAS.get(cam, cam)
    img = cv2.imread(os.path.join(clip, "camera", folder, f"{idx:05d}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{clip}/camera/{folder}/{idx:05d}.jpg")
    if jpeg_quality > 0:
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def load_cameras(clip: str, idx: int, cams: Sequence[str], jpeg_quality: int = 0) -> Dict[str, np.ndarray]:
    return {c: load_image(clip, c, idx, jpeg_quality) for c in cams}


def ego_box(anno: dict) -> dict:
    return next(b for b in anno["bounding_boxes"] if b.get("class") == "ego_vehicle")


def ego_pose(anno: dict) -> np.ndarray:
    """Ego (x, y, yaw) in a right-handed world frame (CARLA's y and yaw mirrored)."""
    ego = ego_box(anno)
    return np.array([ego["location"][0], -ego["location"][1], -np.radians(ego["rotation"][2])])


def ego_history(clip: str, idx: int, steps: int) -> np.ndarray:
    """Ego poses for frames ``idx-steps+1 .. idx`` as (steps, 3), world frame."""
    return np.stack([ego_pose(load_anno(clip, max(k, 0))) for k in range(idx - steps + 1, idx + 1)])


def to_local(poses: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Express world (x, y, yaw) poses in the frame of ``ref`` (x forward, y left)."""
    c, s = np.cos(ref[2]), np.sin(ref[2])
    d = poses[:, :2] - ref[:2]
    out = np.empty_like(poses)
    out[:, 0] = c * d[:, 0] + s * d[:, 1]
    out[:, 1] = -s * d[:, 0] + c * d[:, 1]
    out[:, 2] = (poses[:, 2] - ref[2] + np.pi) % (2 * np.pi) - np.pi
    return out


def target_point(clip: str, idx: int, ahead: int = 40) -> np.ndarray:
    """A route target ~``ahead`` frames later, from the ego's own future path, in the local frame."""
    n = num_frames(clip)
    ref = ego_pose(load_anno(clip, idx))
    fut = ego_pose(load_anno(clip, min(idx + ahead, n - 1)))
    return to_local(fut[None], ref)[0, :2]
