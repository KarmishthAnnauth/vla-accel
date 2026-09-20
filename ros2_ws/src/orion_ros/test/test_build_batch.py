#!/usr/bin/env python3
"""Push six distinct camera images through ORION's REAL inference pipeline.

Exercises orion_node._build_batch end to end (agent results dict -> ORION's own
inference_only_pipeline -> mm_collate) without loading the 38 GB checkpoint, then
asserts the six views are still distinct in the collated tensor. A regression to
replicate-the-front would make all six identical here.
"""
import sys
from collections import deque

import numpy as np
import torch
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

sys.path.insert(0, "/root/Orion")
from mmcv import Config
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type

from orion_ros.orion_node import OrionRosNode
from orion_ros.camera_input import ORION_CAMERA_ORDER

CFG = "/root/Orion/adzoo/orion/configs/orion_stage3_agent.py"
H, W = 900, 1600
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class _Log:
    def warn(self, *a, **k): pass
    def info(self, *a, **k): pass


class _Stub:
    """orion_node, minus __init__ and the model."""
    def __init__(self, pipeline):
        self._inference_only_pipeline = pipeline
        self._mm_collate = mm_collate_to_batch_form
        self._get_box_type = get_box_type
        self._device = torch.device("cpu")
        self._frame_idx = 0
        self._jpeg_quality = 20
        self._driving_command = 4
        self._route = deque()
        self._scene_token = "route-0001"
        self._timestamp_mode = "sensor"
        self._prev_sensor_ts = None
        self._memoryless_frames = 0
        self._dt_samples = 0
        self._last_dt_log_t = 0.0
    def get_logger(self):
        return _Log()


for _m in ("_odom_to_canbus_ego_pose", "_compute_driving_command", "_sanitize_cmd",
           "_check_memory_continuity"):
    if hasattr(OrionRosNode, _m):
        setattr(_Stub, _m, getattr(OrionRosNode, _m))


def cam_record(value: int) -> tuple:
    stamp = Time(sec=100, nanosec=0)
    data = np.full((H, W, 3), value, dtype=np.uint8).tobytes()
    return (stamp, H, W, "bgr8", data)


def main():
    cfg = Config.fromfile(CFG)
    pipeline = Compose([
        t for t in cfg.inference_only_pipeline
        if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)
    ])
    stub = _Stub(pipeline)

    odom = Odometry()
    odom.header.stamp = Time(sec=100, nanosec=0)
    odom.pose.pose.position.x, odom.pose.pose.position.y = 12.0, -3.0
    odom.pose.pose.orientation.w = 1.0
    imu = Imu()

    values = [10, 45, 80, 130, 175, 220]
    images = {cam: cam_record(v) for cam, v in zip(ORION_CAMERA_ORDER, values)}
    snapshot = {"images": images, "speed": 4.0, "odom": odom, "imu": imu}

    batch, meta = OrionRosNode._build_batch(stub, snapshot)
    check(True, "_build_batch completed through the real ORION pipeline")

    img = batch["img"][0]
    if isinstance(img, (list, tuple)):
        img = img[0]
    t = img.data if hasattr(img, "data") else img
    t = torch.as_tensor(t).float()
    while t.dim() > 4:
        t = t[0]
    check(t.shape[0] == 6, f"collated tensor carries 6 views (got shape {tuple(t.shape)})")

    means = [float(t[i].mean()) for i in range(t.shape[0])]
    print("   per-view means:", [f"{m:.1f}" for m in means])
    check(len(set(round(m, 2) for m in means)) == 6,
          "all six views are distinct after preprocessing")

    order_ok = all(means[i] < means[i + 1] for i in range(5))
    check(order_ok, "views stay in ORION_CAMERA_ORDER (monotonic by construction)")

    node_meta = batch["img_metas"]
    while not isinstance(node_meta, dict):
        node_meta = node_meta[0]
    l2i = node_meta["lidar2img"]
    l2i = l2i.data if hasattr(l2i, "data") else l2i
    l2i = [np.asarray(m) for m in l2i]
    check(len(l2i) == 6, f"6 lidar2img matrices present (got {len(l2i)})")
    check(len({m.tobytes() for m in l2i}) == 6,
          "the 6 calibration matrices differ (per-camera, not replicated)")

    check(node_meta.get("scene_token") == "route-0001",
          f"per-route scene_token reaches img_metas (got {node_meta.get('scene_token')!r})")

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("BUILD-BATCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
