#!/usr/bin/env python3

"""Front-camera input for the ReCogDrive ROS node.

ReCogDrive plans from ONE image: ``ReCogDriveFeatureBuilder`` reads
``cameras[-1].cam_f0`` (NAVSIM's front camera, 1920x1080) and nothing else, and
the eval script runs with ``agent.cam_type='single'``.  So unlike the ORION and
MindDrive nodes there is a single subscription here.

The agent does not take the image as an array.  With ``cache_hidden_state=False``
(the eval setting) the feature builder passes the image PATH and
``ReCogDriveAgent.forward`` opens it itself (``load_image`` ->
``Image.open(path).convert('RGB')`` -> ``dynamic_preprocess`` -> ``build_transform``).
To keep that code path byte-for-byte the node writes each frame to a file on
tmpfs and hands the agent its path; ``write_frame`` does that.  The default
format is BMP: lossless, so the pixels the agent reads are exactly the pixels
the topic carried.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np
from sensor_msgs.msg import Image as RosImage

FRAME_FORMATS = ("bmp", "png", "jpg")


_TO_BGR = {"bgra8": cv2.COLOR_BGRA2BGR, "bgr8": None, "rgba8": cv2.COLOR_RGBA2BGR, "rgb8": cv2.COLOR_RGB2BGR}
_TO_RGB = {"bgra8": cv2.COLOR_BGRA2RGB, "bgr8": cv2.COLOR_BGR2RGB, "rgba8": cv2.COLOR_RGBA2RGB, "rgb8": None}


def decode_image(record: tuple, rgb: bool = False) -> np.ndarray:
    """Decode a raw sensor_msgs/Image record into uint8 (H, W, 3).

    BGR by default (what cv2.imwrite wants; the agent's
    ``Image.open(...).convert('RGB')`` reads the file back as RGB); rgb=True
    gives those RGB pixels directly, for a consumer that takes the array.

    record = (stamp, height, width, encoding, raw_bytes)
    """
    _, height, width, encoding, raw = record
    table = _TO_RGB if rgb else _TO_BGR
    if encoding not in table:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    channels = 4 if encoding in ("rgba8", "bgra8") else 3
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
    code = table[encoding]
    return np.ascontiguousarray(arr) if code is None else cv2.cvtColor(arr, code)


def frame_path(directory: str, fmt: str = "bmp", name: str = "cam_f0") -> str:
    """Where write_frame puts the frame."""
    return os.path.join(directory, f"{name}.{fmt}")


def write_frame(image_bgr: np.ndarray, directory: str, fmt: str = "bmp",
                jpeg_quality: int = 95, name: str = "cam_f0") -> str:
    """Write the frame where the agent can open it and return the path.

    Written under a temporary name and renamed, so a reader never sees a
    half-written file.  The path is ASCII: the feature builder ships it to the
    agent as a tensor of character codes.
    """
    if fmt not in FRAME_FORMATS:
        raise ValueError(f"frame format must be one of {FRAME_FORMATS}, got {fmt!r}")
    os.makedirs(directory, exist_ok=True)
    path = frame_path(directory, fmt, name)
    tmp = os.path.join(directory, f".{name}.tmp.{fmt}")
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if fmt == "jpg" else []
    if not cv2.imwrite(tmp, image_bgr, params):
        raise IOError(f"cv2.imwrite failed for {tmp}")
    os.replace(tmp, path)
    return path


def _stamp_tuple(stamp) -> tuple:
    return (stamp.sec, stamp.nanosec)


class FrontCameraBuffer:
    """Latest-frame buffer for the front camera.

    Holds only the raw record (stamp/height/width/encoding/bytes); decoding is
    deferred to the inference worker so the subscription callback stays cheap
    on a multi-megabyte stream.
    """

    def __init__(self, node, topic: str, qos) -> None:
        self._logger = node.get_logger()
        self.topic = topic
        self._latest: Optional[tuple] = None
        self.received_at: Optional[float] = None
        node.create_subscription(RosImage, topic, self._callback, qos)
        self._logger.info(f"Subscribed to front camera: {topic}")

    def _callback(self, msg: RosImage) -> None:
        self._latest = (msg.header.stamp, msg.height, msg.width, msg.encoding, bytes(msg.data))
        self.received_at = time.monotonic()

    def has_frame(self) -> bool:
        return self._latest is not None

    def record(self) -> Optional[tuple]:
        return self._latest

    def stamp(self) -> Optional[tuple]:
        rec = self._latest
        return _stamp_tuple(rec[0]) if rec is not None else None
