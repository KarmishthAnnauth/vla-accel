#!/usr/bin/env python3

"""Shared 6-camera input handling for the ORION ROS nodes.

ORION is a surround-view model: ``OrionAgent.sensors()`` spawns six cameras and
``OrionAgent.tick()`` hands the pipeline six *different* images, one per mount.
The per-camera calibration (``lidar2img`` / ``lidar2cam``) that the node passes
alongside them encodes those six distinct poses, so feeding one view into every
slot tells the model it is looking backwards at a forward image. Detections and
map predictions behind and beside the ego are then invented from nothing.

This module owns the multi-camera plumbing so ``orion_node`` and
``orion_withpid_node`` share exactly one implementation:

  * per-camera subscriptions and latest-frame buffers,
  * a snapshot that is only handed out once every camera has produced a frame,
  * a skew check against the reference camera, because the agent's six images
    come from a single simulator tick whereas ROS delivers six independent
    streams,
  * the raw ``sensor_msgs/Image`` -> BGR decode, including the agent's lossy
    JPEG quality-20 re-encode.

Single-camera (replicate) mode is retained for bring-up and for benchmarking
against the old behaviour: pass one topic instead of six. It is NOT faithful to
the reference agent and the buffer says so, loudly, at startup.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from sensor_msgs.msg import Image as RosImage

ORION_CAMERA_ORDER: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

REFERENCE_CAMERA = "CAM_FRONT"

DEFAULT_CAMERA_TOPICS: List[str] = [
    f"/carla/hero/{cam}/image" for cam in ORION_CAMERA_ORDER
]


def decode_image(record: tuple, jpeg_quality: int = 20) -> np.ndarray:
    """Decode a raw sensor_msgs/Image record into BGR uint8.

    BGR (not RGB) is deliberate: the ORION agent feeds BGR and lets the pipeline's
    ``NormalizeMultiviewImage(to_rgb=True)`` do the conversion.

    record = (stamp, height, width, encoding, raw_bytes)
    """
    _, height, width, encoding, raw = record
    channels = 4 if encoding in ("rgba8", "bgra8") else 3
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
    if encoding == "bgra8":
        img = arr[:, :, :3]
    elif encoding == "bgr8":
        img = arr
    elif encoding == "rgba8":
        img = arr[:, :, 2::-1]
    elif encoding == "rgb8":
        img = arr[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    img = np.ascontiguousarray(img)

    if jpeg_quality > 0:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        _, enc = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def _stamp_tuple(stamp) -> tuple:
    return (stamp.sec, stamp.nanosec)


def _stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class MultiCameraBuffer:
    """Latest-frame buffers for the six ORION cameras.

    Holds only the raw record per camera (stamp/height/width/encoding/bytes) —
    decoding is deferred to the inference worker so the subscription callbacks
    stay cheap and never become the bottleneck on a 5.76 MB/frame stream.
    """

    def __init__(
        self,
        node,
        topics: Sequence[str],
        qos,
        sync_tolerance_sec: float = 0.1,
    ) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._sync_tolerance = float(sync_tolerance_sec)

        topics = [t for t in topics if t]
        if len(topics) == 1:
            self.replicate = True
            self._topics = {REFERENCE_CAMERA: topics[0]}
            self._logger.warn(
                "SINGLE-CAMERA MODE: '%s' will be replicated into all 6 ORION "
                "camera slots while the 6 distinct calibration matrices are still "
                "passed. The model will be shown a forward image labelled as the "
                "rear and side views. This is NOT faithful to OrionAgent and any "
                "Bench2Drive score from it is not comparable — pass 6 topics via "
                "the 'camera_topics' parameter for the real setup." % topics[0]
            )
        elif len(topics) == len(ORION_CAMERA_ORDER):
            self.replicate = False
            self._topics = dict(zip(ORION_CAMERA_ORDER, topics))
        else:
            raise ValueError(
                f"camera_topics must have exactly 1 (replicate) or "
                f"{len(ORION_CAMERA_ORDER)} entries in ORION order "
                f"{ORION_CAMERA_ORDER}; got {len(topics)}"
            )

        self._latest: Dict[str, tuple] = {}
        self._last_skew_log_t: float = 0.0
        self._skew_count: int = 0

        for cam, topic in self._topics.items():
            node.create_subscription(
                RosImage, topic, self._make_callback(cam), qos
            )
            self._logger.info(f"Subscribed to {cam}: {topic}")

    def _make_callback(self, cam: str):
        def _cb(msg: RosImage) -> None:
            self._latest[cam] = (
                msg.header.stamp, msg.height, msg.width, msg.encoding, bytes(msg.data)
            )
        return _cb

    @property
    def cameras(self) -> List[str]:
        """Cameras that are actually subscribed (1 in replicate mode, else 6)."""
        return list(self._topics.keys())

    def missing(self) -> List[str]:
        """Subscribed cameras that have not produced a frame yet."""
        return [c for c in self._topics if c not in self._latest]

    def reference_record(self) -> Optional[tuple]:
        return self._latest.get(REFERENCE_CAMERA)

    def reference_stamp(self) -> Optional[tuple]:
        rec = self.reference_record()
        return _stamp_tuple(rec[0]) if rec is not None else None

    def snapshot(self) -> Optional[Dict[str, tuple]]:
        """Latest record for every ORION camera, or None if any is still missing.

        In replicate mode the single front record is returned under all six keys,
        so callers index uniformly by camera name and never branch on the mode.
        """
        if self.missing():
            return None
        if self.replicate:
            rec = self._latest[REFERENCE_CAMERA]
            return {cam: rec for cam in ORION_CAMERA_ORDER}

        snap = dict(self._latest)
        self._check_skew(snap)
        return snap

    def _check_skew(self, snap: Dict[str, tuple]) -> None:
        """Warn when the six views did not come from roughly the same instant.

        The reference agent gets all six images from one simulator tick. Here they
        arrive as six independent DDS streams, so a stalled or lagging camera can
        pair a current front view with a stale rear one — the model is handed an
        inconsistent scene and there is nothing in its output that says so. This
        does not drop the frame (that would stall inference on one slow camera);
        it makes the condition visible.
        """
        ref = snap.get(REFERENCE_CAMERA)
        if ref is None:
            return
        ref_t = _stamp_sec(ref[0])
        worst_cam, worst_dt = None, 0.0
        for cam, rec in snap.items():
            dt = abs(_stamp_sec(rec[0]) - ref_t)
            if dt > worst_dt:
                worst_cam, worst_dt = cam, dt
        if worst_dt <= self._sync_tolerance:
            return

        self._skew_count += 1
        now = time.monotonic()
        if now - self._last_skew_log_t < 5.0:
            return
        self._last_skew_log_t = now
        self._logger.warn(
            f"camera skew: {worst_cam} is {worst_dt * 1e3:.0f} ms from "
            f"{REFERENCE_CAMERA} (tolerance {self._sync_tolerance * 1e3:.0f} ms, "
            f"{self._skew_count} frames affected). The six views are not from the "
            f"same instant; check for a lagging or dropped camera stream."
        )

    @property
    def skew_count(self) -> int:
        return self._skew_count
