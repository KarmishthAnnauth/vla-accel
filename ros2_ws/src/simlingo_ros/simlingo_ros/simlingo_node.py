#!/usr/bin/env python3
"""
ROS 2 node that runs SimLingo VLA inference and publishes Autoware trajectories.

Subscriptions:
  /carla/hero/rgb_0/image         sensor_msgs/Image       Raw RGBA from carla_ros_bridge
  /carla/hero/speed               std_msgs/Float32        Forward speed in m/s
  /carla/hero/odometry            nav_msgs/Odometry       Vehicle pose (map frame)
  /carla/hero/global_plan         carla_msgs/CarlaRoute   Route waypoints (TRANSIENT_LOCAL)

Publications:
  /carla/hero/vehicle_control_cmd  carla_msgs/CarlaEgoVehicleControl   (control_mode=pid)
  /simlingo/predicted_trajectory   autoware_planning_msgs/Trajectory
  /simlingo/language_output        std_msgs/String

Control:
  control_mode=pid (default) runs SimLingo's own longitudinal + lateral PID
  (agent_simlingo.py::control_pid, see simlingo_pid.py) in this node and publishes
  steer/throttle/brake straight to CARLA, exactly as the reference leaderboard agent
  does. No Stanley controller and no carla_ackermann_control node are involved —
  make sure carla_ackermann_control is NOT running, or it will fight this node for
  /carla/hero/vehicle_control_cmd.

  control_mode=trajectory publishes only the Autoware trajectory, for the old
  stanley_controller_node path.

  The trajectory and RViz markers are published in both modes, so the prediction
  stays inspectable while the PID drives.

Model loading:
  Loads directly from local paths — no HuggingFace download at runtime.
  The hydra config is located automatically from the checkpoint path, matching
  the convention used in agent_simlingo.py:
    <checkpoint>.ckpt  →  ../../../.hydra/config.yaml

Required parameters:
  checkpoint_path   (str)   Absolute path to the .ckpt weights file
  simlingo_path     (str)   Absolute path to the simlingo repository root
                            (so that simlingo_training and team_code are importable)
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import torch
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaRoute
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import ColorRGBA, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


_WP_DT = 0.25

_CAM_FOV = 110
_CAM_W   = 1024
_CAM_H   = 512

_MIN_TARGET_DIST_M = 7.5
_MAX_TARGET_DIST_M = 50.0


class _Phases:
    """Per-inference phase timing.

    Deliberately dumb: a perf_counter and a dict.  `mark()` closes the phase
    that was open and opens the next, so the marks partition the wall clock
    with no gaps -- which is the point.  Anything the marks do not cover shows
    up in `other`, so an unaccounted-for cost cannot hide.

    GPU clock is sampled per inference because the Orin's devfreq governor
    idles the GPU at 306 MHz and takes ~12 s of sustained load to reach 1300
    MHz; a run that never ramps looks exactly like a slow model.
    """

    GPU_FREQ = "/sys/class/devfreq/17000000.gpu/cur_freq"
    INNER = ("prompt", "assemble", "model", "control", "log", "publish")
    OUTER = ("pay_copy", "pay_tile", "pay_norm", "pay_route", "wait")
    ORDER = OUTER + INNER

    def __init__(self) -> None:
        self.d: dict = {}
        self._t0 = self._t = time.perf_counter()

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        self.d[name] = self.d.get(name, 0.0) + (now - self._t) * 1000.0
        self._t = now
        return self.d[name]

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def finish(self) -> dict:
        total = self.total_ms()
        inner = sum(self.d.get(k, 0.0) for k in self.INNER)
        self.d["other"] = total - inner
        self.d["total"] = total
        self.d["gpu_mhz"] = self.gpu_mhz()
        return self.d

    @classmethod
    def gpu_mhz(cls) -> int:
        try:
            with open(cls.GPU_FREQ) as fh:
                return int(fh.read().strip()) // 1_000_000
        except Exception:
            return -1

    @classmethod
    def line(cls, d: dict) -> str:
        outer = "  ".join(f"{k}={d[k]:.0f}" for k in cls.OUTER if k in d)
        inner = "  ".join(f"{k}={d[k]:.0f}" for k in cls.INNER if k in d)
        wall = sum(d.get(k, 0.0) for k in cls.OUTER) + d.get("total", 0.0)
        return (f"[PROF] wall={wall:7.0f}ms  total={d.get('total', 0.0):7.0f}ms  "
                f"[{outer}]  {inner}  other={d.get('other', 0.0):.0f}"
                f"  gpu={d.get('gpu_mhz', -1)}MHz")

    @classmethod
    def summary(cls, rows: list) -> str:
        if not rows:
            return "[PROF] no samples"
        keys = [k for k in (*cls.ORDER, "other", "total") if any(k in r for r in rows)]
        out = [f"[PROF] ---- {len(rows)} inferences: mean / min / max (ms) ----"]
        for k in keys:
            v = sorted(r.get(k, 0.0) for r in rows)
            out.append(f"[PROF]   {k:9s} {sum(v)/len(v):8.0f} {v[0]:8.0f} {v[-1]:8.0f}")
        g = [r.get("gpu_mhz", -1) for r in rows]
        out.append(f"[PROF]   gpu_mhz   {sum(g)/len(g):8.0f} {min(g):8d} {max(g):8d}")
        return "\n".join(out)


class SimLingoNode(Node):

    def __init__(self) -> None:
        super().__init__("simlingo_node")

        self.declare_parameter("simlingo_path",   "/workspace/simlingo")
        self.declare_parameter("checkpoint_path", "/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")
        self.declare_parameter("image_topic",      "/carla/hero/rgb_0/image")
        self.declare_parameter("image_compressed", False)
        self.declare_parameter("speed_topic",      "/carla/hero/speed")
        self.declare_parameter("odometry_topic",   "/carla/hero/odometry")
        self.declare_parameter("route_topic",      "/carla/hero/global_plan")
        self.declare_parameter("trajectory_topic", "/simlingo/predicted_trajectory")
        self.declare_parameter("language_topic",   "/simlingo/language_output")
        self.declare_parameter("control_topic",    "/carla/hero/vehicle_control_cmd")
        self.declare_parameter("inference_period_sec",      0.25)
        self.declare_parameter("min_trajectory_speed_mps",  0.5)
        self.declare_parameter("brake_speed_mps",           0.4)
        self.declare_parameter("control_mode",     "pid")
        self.declare_parameter("control_hz",            20.0)
        self.declare_parameter("fast_inference",        True)
        self.declare_parameter("profile",               True)
        self.declare_parameter("profile_every",         20)
        self.declare_parameter("control_timeout_sec",   0.5)
        self.declare_parameter("max_plan_age_sec",      5.0)
        self.declare_parameter("pid_window_rate_compensation", False)
        self.declare_parameter("initial_brake_sec",     0.0)

        simlingo_path = str(self.get_parameter("simlingo_path").value)
        if simlingo_path and simlingo_path not in sys.path:
            sys.path.insert(0, simlingo_path)

        self._min_traj_speed = float(self.get_parameter("min_trajectory_speed_mps").value)
        inference_period     = float(self.get_parameter("inference_period_sec").value)
        self._control_mode   = str(self.get_parameter("control_mode").value).lower()
        if self._control_mode not in ("pid", "trajectory"):
            raise ValueError(
                f"control_mode must be 'pid' or 'trajectory', got '{self._control_mode}'"
            )
        self._control_timeout = float(self.get_parameter("control_timeout_sec").value)
        self._initial_brake   = float(self.get_parameter("initial_brake_sec").value)
        self._control_hz      = float(self.get_parameter("control_hz").value)
        self._fast_inference  = bool(self.get_parameter("fast_inference").value)
        self._profile         = bool(self.get_parameter("profile").value)
        self._profile_every   = int(self.get_parameter("profile_every").value)
        self._phase_rows: List[dict] = []
        self._max_plan_age    = float(self.get_parameter("max_plan_age_sec").value)
        if self._control_hz <= 0.0:
            raise ValueError(f"control_hz must be > 0, got {self._control_hz}")

        self._latest_image: Optional[np.ndarray] = None
        self._img_first_stamp: Optional[float] = None
        self._img_first_wall:  Optional[float] = None
        self._img_last_stamp:  Optional[float] = None
        self._img_last_wall:   Optional[float] = None
        self._img_count: int = 0
        self._img_used:  int = 0
        self._latest_speed: float = 0.0
        self._latest_odom:  Optional[Odometry] = None
        self._route_wps:    List[np.ndarray] = []
        self._route_idx:    int = 0

        self._controller             = None
        self._last_control_sec: Optional[float] = None
        self._first_control_sec: Optional[float] = None
        self._braking_on_stall  = False

        self._plan: Optional[dict] = None
        self._plan_lock = threading.Lock()
        self._plan_seq  = 0
        self._plan_exhausted_seq = -1
        self._plan_progress     = 0
        self._plan_progress_seq = -1

        traj_topic = str(self.get_parameter("trajectory_topic").value)
        self._traj_pub = self.create_publisher(Trajectory, traj_topic, 10)
        self.get_logger().info(f"Publishing trajectories on {traj_topic}")

        marker_topic = traj_topic + "_markers"
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.get_logger().info(f"Publishing trajectory markers on {marker_topic}")

        lang_topic = str(self.get_parameter("language_topic").value)
        self._lang_pub = self.create_publisher(String, lang_topic, 10)
        self.get_logger().info(f"Publishing language output on {lang_topic}")

        self._control_pub = None
        if self._control_mode == "pid":
            control_topic = str(self.get_parameter("control_topic").value)
            self._control_pub = self.create_publisher(
                CarlaEgoVehicleControl, control_topic, 10
            )
            self.get_logger().info(
                f"control_mode=pid — publishing CarlaEgoVehicleControl on {control_topic}. "
                "carla_ackermann_control / stanley_controller_node must NOT be running."
            )
        else:
            self.get_logger().info(
                "control_mode=trajectory — trajectory only, an external controller must drive."
            )

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        image_topic = str(self.get_parameter("image_topic").value)
        image_compressed = (bool(self.get_parameter("image_compressed").value)
                            or image_topic.endswith("/compressed"))
        if image_compressed:
            self.create_subscription(RosCompressedImage, image_topic,
                                     self._image_cb_compressed, img_qos)
            self.get_logger().info(f"Subscribed to image (COMPRESSED): {image_topic}")
        else:
            self.create_subscription(RosImage, image_topic, self._image_cb, img_qos)
            self.get_logger().info(f"Subscribed to image (raw): {image_topic}")

        speed_topic = str(self.get_parameter("speed_topic").value)
        self.create_subscription(Float32, speed_topic, self._speed_cb, 10)
        self.get_logger().info(f"Subscribed to speed: {speed_topic}")

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        odom_topic = str(self.get_parameter("odometry_topic").value)
        self.create_subscription(Odometry, odom_topic, self._odometry_cb, odom_qos)
        self.get_logger().info(f"Subscribed to odometry: {odom_topic}")

        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        route_topic = str(self.get_parameter("route_topic").value)
        self.create_subscription(CarlaRoute, route_topic, self._route_cb, route_qos)
        self.get_logger().info(f"Subscribed to route: {route_topic}")

        self._model       = None
        self._tokenizer   = None
        self._cfg         = None
        self._transform   = None
        self._conv_module = None
        self._num_image_token: int = 256

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        self.get_logger().info("Loading SimLingo model (this may take a few minutes)...")
        self._executor.submit(self._setup_model).result()

        self._timer = self.create_timer(inference_period, self._timer_cb)

        if self._control_mode == "pid":
            self._ctrl_group = MutuallyExclusiveCallbackGroup()
            self._control_timer = self.create_timer(
                1.0 / self._control_hz, self._control_cb,
                callback_group=self._ctrl_group,
            )
            self.get_logger().info(
                f"Control loop at {self._control_hz:.1f} Hz, tracking the latest "
                f"map-anchored plan (max age {self._max_plan_age:.1f}s)."
            )
            if self._control_timeout > 0.0:
                self._watchdog = self.create_timer(
                    0.1, self._watchdog_cb, callback_group=self._ctrl_group)


    def _setup_model(self) -> None:
        """Load SimLingo from local paths. Runs in the worker thread."""
        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoConfig, AutoProcessor

        ckpt_path = str(self.get_parameter("checkpoint_path").value)
        if not ckpt_path:
            raise ValueError("Parameter 'checkpoint_path' is required.")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.get_logger().info(f"Loading checkpoint: {ckpt_path}")

        cfg_path = Path(ckpt_path).parent.parent.parent / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Expected .hydra/config.yaml at {cfg_path}. "
                "Ensure checkpoint_path points to the .ckpt file inside the training output directory."
            )
        cfg = OmegaConf.load(cfg_path)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self._cfg = cfg

        vlm_variant = cfg.model.vision_model.variant
        simlingo_path = str(self.get_parameter("simlingo_path").value)
        vlm_cache = str(Path(simlingo_path) / "pretrained" / vlm_variant.split("/")[1])

        processor = AutoProcessor.from_pretrained(vlm_variant, trust_remote_code=True, cache_dir=vlm_cache)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>",
                "<ORG_WAYPOINTS>", "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>",
                "<TARGET_POINT>",
            ]
        })
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer

        vlm_cfg          = AutoConfig.from_pretrained(vlm_variant, trust_remote_code=True, cache_dir=vlm_cache)
        image_size       = vlm_cfg.force_image_size or vlm_cfg.vision_config.image_size
        patch_size       = vlm_cfg.vision_config.patch_size
        downsample_ratio = vlm_cfg.downsample_ratio
        self._num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.get_logger().info(f"num_image_token per tile: {self._num_image_token}")

        conv_py = Path(vlm_cache) / "conversation.py"
        if not conv_py.exists():
            from huggingface_hub import hf_hub_download
            conv_py = Path(hf_hub_download(repo_id=vlm_variant, filename="conversation.py"))
        spec = importlib.util.spec_from_file_location("_simlingo_conv", str(conv_py))
        conv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conv_module)
        self._conv_module = conv_module

        from simlingo_training.utils.internvl2_utils import build_transform
        self._transform = build_transform(input_size=448)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self._model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=vlm_cache,
            _recursive_=False,
        ).to(self._device)
        torch.set_default_dtype(default_dtype)

        self._model.load_state_dict(torch.load(ckpt_path, map_location=self._device))
        self._model.eval()

        if self._fast_inference:
            try:
                from simlingo_training.models.fast_inference import optimize_for_inference
                report = optimize_for_inference(self._model)
                self.get_logger().info(f"fast_inference enabled: {report}")
            except Exception as exc:
                self.get_logger().error(
                    f"fast_inference failed ({type(exc).__name__}: {exc}); "
                    "falling back to the stock model")
        else:
            self.get_logger().info("fast_inference disabled by parameter")

        torch.cuda.empty_cache()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _d = torch.zeros(1, 1, device=self._device, dtype=torch.bfloat16)
            torch.mm(_d, _d)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if self._control_mode == "pid":
            self._controller = self._make_controller(announce=True)

        self.get_logger().info("SimLingo model ready.")

    def _make_controller(self, announce: bool = False):
        """A controller with no history in it.

        Imported here rather than at module scope: it prefers the upstream
        classes from team_code, which only resolve after simlingo_path was
        pushed onto sys.path in __init__.
        """
        from simlingo_ros.simlingo_pid import (
            UPSTREAM_CONTROLLERS,
            SimLingoPIDConfig,
            SimLingoPIDController,
        )

        pid_config = SimLingoPIDConfig(
            brake_speed=float(self.get_parameter("brake_speed_mps").value)
        )
        controller = SimLingoPIDController(
            config=pid_config,
            control_hz=self._control_hz,
            window_rate_compensation=bool(
                self.get_parameter("pid_window_rate_compensation").value
            ),
        )
        if announce:
            source = "team_code (upstream)" if UPSTREAM_CONTROLLERS else "bundled copy"
            self.get_logger().info(
                f"SimLingo PID ready (brake_speed={pid_config.brake_speed:.2f} m/s) — "
                f"controllers from {source}, "
                f"stepping at ~{self._control_hz:.1f} Hz, speed PID window "
                f"n={controller.speed_n}."
            )
        return controller

    @property
    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _note_image(self, msg) -> None:
        """Record when a frame was stamped and when it actually got here."""
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        wall  = time.time()
        if self._img_first_stamp is None:
            self._img_first_stamp, self._img_first_wall = stamp, wall
        self._img_last_stamp, self._img_last_wall = stamp, wall
        self._img_count += 1
        drift = ((wall - self._img_first_wall)
                 - (stamp - self._img_first_stamp))
        self.get_logger().info(
            f"[IMG] n={self._img_count} used={self._img_used} "
            f"drift={drift:+.2f}s "
            f"(stream {stamp - self._img_first_stamp:.1f}s vs wall "
            f"{wall - self._img_first_wall:.1f}s)",
            throttle_duration_sec=5.0,
        )

    def _image_cb(self, msg: RosImage) -> None:
        """Decode RGBA image from carla_ros_bridge and apply training-domain preprocessing."""
        channels = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            msg.height, msg.width, channels
        )
        img = img[:, :, :3]
        _, buf = cv2.imencode(".jpg", img)
        img    = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_h = int(img.shape[0] - (img.shape[0] * 4.8) // 16)
        self._latest_image = img[:crop_h, :, :]
        self._note_image(msg)

    def _image_cb_compressed(self, msg: RosCompressedImage) -> None:
        """Decode a JPEG frame from image_transport republish on the CARLA box.

        Same output as _image_cb: RGB, hood cropped.  The raw path re-encodes to
        JPEG itself to reproduce the compression artifacts the model was trained
        on -- here the frame arrived as JPEG, so that step is already done and
        repeating it would compress twice.
        """
        img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn("could not decode compressed frame; dropping")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_h = int(img.shape[0] - (img.shape[0] * 4.8) // 16)
        self._latest_image = img[:crop_h, :, :]
        self._note_image(msg)

    def _speed_cb(self, msg: Float32) -> None:
        self._latest_speed = float(msg.data)

    def _odometry_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _route_cb(self, msg: CarlaRoute) -> None:
        self._route_wps = [
            np.array([p.position.x, p.position.y], dtype=np.float64)
            for p in msg.poses
        ]
        self._route_idx = 0
        self._reset_for_new_route()
        self.get_logger().info(
            f"Route received: {len(self._route_wps)} waypoints."
        )
        _head = "  ".join(f"[{w[0]:+.2f},{w[1]:+.2f}]" for w in self._route_wps[:4])
        self.get_logger().info(
            f"[DIAG] route[0:4] (ROS map xy): {_head}"
            f"   route[-1]=[{self._route_wps[-1][0]:+.2f},{self._route_wps[-1][1]:+.2f}]"
        )

    def _reset_for_new_route(self) -> None:
        """Drop everything scoped to the route that just ended.

        Upstream gets this for free: leaderboard_evaluator.py builds a fresh
        ``agent_class_obj(...)`` for every route (:363) and destroy()s it
        afterwards (:177), so no state can cross a route boundary.  On this path
        only the thin gateway agent on the CARLA box is rebuilt -- the model
        node is one long-lived process started once by start_orin.sh for the
        whole run -- so the boundary has to be recreated by hand.

        Two pieces of carried-over state actually bite:

          * ``_plan`` holds the previous route's trajectory, anchored in the map
            frame at a spawn point the ego is no longer anywhere near.
            max_plan_age_sec brakes rather than follows it, but only once the
            plan is stale enough, so a quick route changeover could steer
            briefly toward the old path.
          * the controller's ``_stuck_time`` / ``_force_move_until``.  Routes
            tend to *end* with the ego blocked, which is precisely when the
            stuck timer is large, so the next route could open with the creep
            recovery already firing at throttle 0.4 before the model has said
            anything.

        Sensor state (image, odometry, speed) is deliberately left alone: it is
        overwritten continuously and is about the world, not the route.
        """
        with self._plan_lock:
            self._plan = None
            if self._control_mode == "pid" and self._controller is not None:
                self._controller = self._make_controller()
        self._plan_progress = 0
        self._plan_progress_seq = -1
        self._plan_exhausted_seq = -1
        self._first_control_sec = None
        self._last_control_sec  = None
        self._braking_on_stall  = False
        self._img_first_stamp = None
        self._img_first_wall  = None
        self._img_count = 0
        self._img_used  = 0
        self.get_logger().info(
            "New route — cleared the previous route's plan, PID history and "
            "stuck timer.")


    def _timer_cb(self) -> None:
        try:
            if self._active_future and not self._active_future.done():
                return
            payload = self._build_payload()
            if payload is None:
                return
            self.get_logger().info("Submitting inference job…")
            self._active_future = self._executor.submit(self._run_inference, payload)
            self._active_future.add_done_callback(self._on_done)
        except Exception as exc:
            self.get_logger().error(f"_timer_cb failed: {exc}", throttle_duration_sec=5.0)

    def _build_payload(self) -> Optional[dict]:
        """Gather the latest sensor data and preprocess images on the main thread."""
        if self._latest_image is None:
            self.get_logger().warn("Waiting for image…", throttle_duration_sec=5.0)
            return None
        if self._latest_odom is None:
            self.get_logger().warn("Waiting for odometry…", throttle_duration_sec=5.0)
            return None
        if not self._route_wps:
            self.get_logger().warn("Waiting for route…", throttle_duration_sec=5.0)
            return None

        _ph = _Phases()
        img   = self._latest_image.copy()
        _ph.mark("pay_copy")
        speed = self._latest_speed
        odom  = self._latest_odom
        stamp = odom.header.stamp

        self._img_used += 1
        if self._img_last_wall is not None:
            sat_for = time.time() - self._img_last_wall
            drift = ((self._img_last_wall - self._img_first_wall)
                     - (self._img_last_stamp - self._img_first_stamp))
            self.get_logger().info(
                f"[IMG] feeding model a frame that arrived {sat_for:.2f}s ago; "
                f"stream drift {drift:+.2f}s; received {self._img_count}, "
                f"used {self._img_used}")

        from simlingo_training.utils.internvl2_utils import dynamic_preprocess

        use_thumbnail = self._cfg.model.vision_model.use_global_img
        pil_img = PILImage.fromarray(img)
        _n = getattr(self, "_imgdump_n", 0)
        if _n < 3:
            self._imgdump_n = _n + 1
            try:
                import os as _os
                _os.makedirs("/benchmarking/imgdiag", exist_ok=True)
                pil_img.save(f"/benchmarking/imgdiag/frame_{_n}.png")
                self.get_logger().info(
                    f"[DIAG] wrote /benchmarking/imgdiag/frame_{_n}.png "
                    f"shape={img.shape} dtype={img.dtype} "
                    f"mean_rgb=({img[...,0].mean():.1f},{img[...,1].mean():.1f},{img[...,2].mean():.1f})"
                )
            except Exception as _e:
                self.get_logger().warn(f"[DIAG] image dump failed: {_e}")
        tiles   = dynamic_preprocess(
            pil_img, image_size=448, use_thumbnail=use_thumbnail, max_num=2
        )
        _ph.mark("pay_tile")
        pixel_values = torch.stack([self._transform(t) for t in tiles])
        _ph.mark("pay_norm")
        processed_image = pixel_values.unsqueeze(0).unsqueeze(0)
        num_patches = pixel_values.shape[0]

        ego_pos = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
        ])
        ego_yaw = _yaw_from_quaternion(odom.pose.pose.orientation)
        tp0, tp1 = self._get_target_points(ego_pos, ego_yaw)
        _q = odom.pose.pose.orientation
        _i = self._route_idx
        _sel = self._route_wps[min(_i + 1, len(self._route_wps) - 1)]
        self.get_logger().info(
            f"[DIAG] ego(ROS map)=[{ego_pos[0]:+.2f},{ego_pos[1]:+.2f}] "
            f"yaw={math.degrees(ego_yaw):+.1f}deg "
            f"quat=(x{_q.x:+.3f} y{_q.y:+.3f} z{_q.z:+.3f} w{_q.w:+.3f}) "
            f"frame={odom.header.frame_id}/{odom.child_frame_id} "
            f"idx={_i} sel_world=[{_sel[0]:+.2f},{_sel[1]:+.2f}] "
            f"-> tp0=[{tp0[0]:+.2f},{tp0[1]:+.2f}]",
            throttle_duration_sec=2.0,
        )
        target_points_np   = np.array([tp0, tp1], dtype=np.float32)
        target_point_torch = torch.from_numpy(tp0[np.newaxis]).float()
        _ph.mark("pay_route")

        return {
            "processed_image":    processed_image,
            "num_patches":        num_patches,
            "speed":              speed,
            "target_points_np":   target_points_np,
            "target_point_torch": target_point_torch,
            "_phases":            _ph.d,
            "_built_at":          time.perf_counter(),
            "stamp":              stamp,
            "ego_pos_3d":         np.array([
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
            ], dtype=np.float64),
            "ego_yaw":            ego_yaw,
        }

    def _get_target_points(
        self, ego_pos: np.ndarray, ego_yaw: float
    ):
        """Next two route waypoints in the ego frame, the way upstream picks them.

        A transcription of RoutePlanner.run_step (simlingo/team_code/nav_planner.py)
        together with the lines that consume it (agent_simlingo.py:444-452), with
        route progress tracked by an index instead of popleft()ing a deque.

        The point worth stating plainly, because the previous version of this
        function got it wrong: min_distance and max_distance decide *which
        waypoints are discarded*, never which one becomes the target.  The target
        is always the next route entry after the discarded ones -- route[1] -- at
        whatever distance that turns out to be.  Selecting candidates on "between
        7.5 m and 50 m away, and in front of the car" instead is what fed the
        model target points 90 degrees off its heading (and, when nothing passed
        that filter, the far end of the route), which made it plan a hard left on
        every frame and pinned the lateral PID at steer=-1.0.
        """
        def to_ego(wp: np.ndarray) -> np.ndarray:
            return self._map_to_model(wp, ego_pos, ego_yaw)[0]

        route = self._route_wps
        n = len(route)

        if n - self._route_idx > 2:
            to_pop = 0
            farthest_in_range = -np.inf
            cumulative_distance = 0.0
            for i in range(self._route_idx + 1, n):
                if cumulative_distance > _MAX_TARGET_DIST_M:
                    break
                cumulative_distance += float(np.linalg.norm(route[i] - route[i - 1]))
                d = float(np.linalg.norm(route[i] - ego_pos))
                if farthest_in_range < d <= _MIN_TARGET_DIST_M:
                    farthest_in_range = d
                    to_pop = i - self._route_idx
            self._route_idx = min(self._route_idx + to_pop, max(n - 2, 0))

        remaining = n - self._route_idx
        if remaining > 2:
            tp0, tp1 = route[self._route_idx + 1], route[self._route_idx + 2]
        elif remaining > 1:
            tp0 = tp1 = route[self._route_idx + 1]
        else:
            tp0 = tp1 = route[self._route_idx]

        return to_ego(tp0), to_ego(tp1)

    def _run_inference(self, payload: dict) -> dict:
        """Build DrivingInput, run the model, and publish results. Runs in the worker thread."""
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel

        start = time.time()
        ph = _Phases()
        ph.d["wait"] = max(0.0, (time.perf_counter() - payload["_built_at"]) * 1000.0)
        ph._t = time.perf_counter()

        processed_image   = payload["processed_image"]
        num_patches       = payload["num_patches"]
        speed             = payload["speed"]
        target_points_np  = payload["target_points_np"]
        target_point_torch = payload["target_point_torch"]

        speed_rounded = round(speed, 1)
        prompt = (
            f"Current speed: {speed_rounded} m/s. "
            "Target waypoint: <TARGET_POINT><TARGET_POINT>. "
            "Predict the waypoints."
        )

        tp_token_id = self._tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
        placeholder_batch = {tp_token_id: target_points_np}

        question = f"<image>\n{prompt}"
        template = self._conv_module.get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        system_header = (
            template.system_template.replace("{system_message}", template.system_message)
            + template.sep
        )
        query = query.replace(system_header, "")

        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self._num_image_token * num_patches
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)

        tok = self._tokenizer(
            [query],
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        ph.mark("prompt")

        device = self._device
        ll = LanguageLabel(
            phrase_ids=tok["input_ids"].to(device),
            phrase_valid=(tok["input_ids"] != self._tokenizer.pad_token_id).to(device),
            phrase_mask=(tok["input_ids"] != self._tokenizer.pad_token_id).to(device),
            placeholder_values=[placeholder_batch],
            language_string=[query],
            loss_masking=None,
        )

        _, _, n_tiles, C, H, W = processed_image.shape
        focal = W / (2.0 * math.tan(_CAM_FOV * math.pi / 360.0))
        K = torch.tensor(
            [[focal, 0.0, W / 2.0],
             [0.0, focal, H / 2.0],
             [0.0, 0.0,  1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        E = torch.eye(4, dtype=torch.float32)
        E[:3, 3] = torch.tensor([-1.5, 0.0, 2.0])
        E = E.unsqueeze(0).to(device)

        model_input = DrivingInput(
            camera_images=processed_image.to(device).bfloat16(),
            image_sizes=None,
            camera_intrinsics=K,
            camera_extrinsics=E,
            vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=device),
            target_point=target_point_torch.to(device),
            prompt=ll,
            prompt_inference=ll,
        )

        ph.mark("assemble")

        if self._profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_speed_wps, pred_route, language = self._model(model_input)
        if self._profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        ph.mark("model")

        pred_route     = pred_route.float().cpu()     if pred_route     is not None else None
        pred_speed_wps = pred_speed_wps.float().cpu() if pred_speed_wps is not None else None

        control_info = None
        if self._control_mode == "pid":
            control_info = self._store_plan(pred_route, pred_speed_wps, payload)
        ph.mark("control")

        if language:
            lang_str = language[0] if isinstance(language, (list, tuple)) else str(language)
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"[SIMLINGO] Prompt : {prompt}\n"
                f"[SIMLINGO] Output : {lang_str}\n"
                f"{'='*60}"
            )
        else:
            self.get_logger().warn("[SIMLINGO] Model returned no language output.")

        if pred_route is not None and pred_route.numel() > 0:
            route_np = pred_route[0].numpy()
            pts_str = "  ".join(
                f"[{pt[0]:+.2f}, {pt[1]:+.2f}]" for pt in route_np[:5]
            )
            self.get_logger().info(
                f"[SIMLINGO] Route wps (first 5, ego x/y m, y=right): {pts_str}"
                f"  … ({len(route_np)} total)"
            )
        else:
            self.get_logger().warn("[SIMLINGO] pred_route is None or empty.")

        if pred_speed_wps is not None and pred_speed_wps.shape[1] >= 3:
            spd = pred_speed_wps[0].numpy()
            desired_speed = float(np.linalg.norm(spd[0] - spd[2]) * 2.0)
            spd_str = "  ".join(
                f"[{pt[0]:+.2f}, {pt[1]:+.2f}]" for pt in spd[:5]
            )
            self.get_logger().info(
                f"[SIMLINGO] Speed wps (first 5): {spd_str}"
                f"  → desired_speed = {desired_speed:.2f} m/s"
            )
        else:
            self.get_logger().warn("[SIMLINGO] pred_speed_wps is None or too short.")

        tp = target_points_np[0]
        self.get_logger().info(
            f"[SIMLINGO] Target point (ego): [{tp[0]:+.2f}, {tp[1]:+.2f}] m"
            f"  |  ego speed = {speed:.2f} m/s"
        )

        ph.mark("log")

        traj_msg = self._to_autoware_trajectory(pred_route, pred_speed_wps, payload["stamp"])
        self._traj_pub.publish(traj_msg)

        if pred_route is not None and pred_route.numel() > 0:
            marker_array = self._trajectory_to_markers(
                pred_route[0].float().cpu().numpy(),
                ego_pos=payload["ego_pos_3d"],
                ego_yaw=payload["ego_yaw"],
                stamp=payload["stamp"],
            )
            self._marker_pub.publish(marker_array)

        if language:
            lang_str = language[0] if isinstance(language, (list, tuple)) else str(language)
            lang_msg = String()
            lang_msg.data = str(lang_str)
            self._lang_pub.publish(lang_msg)

        ph.mark("publish")
        phases = ph.finish()
        phases.update(payload.get("_phases", {}))

        stats = {"duration_sec": time.time() - start, "num_pts": len(traj_msg.points),
                 "_phases": phases}
        if control_info is not None:
            stats.update(control_info)
        return stats


    @staticmethod
    def _map_to_model(pts_map: np.ndarray, ego_pos: np.ndarray,
                      ego_yaw: float) -> np.ndarray:
        """ROS map frame -> model ego frame (x forward, y right). Returns [N, 2]."""
        pts = np.asarray(pts_map, dtype=np.float64).reshape(-1, 2)
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)
        R = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
        local = (pts - np.asarray(ego_pos, dtype=np.float64).reshape(1, 2)) @ R.T
        return np.stack([local[:, 0], -local[:, 1]], axis=1)

    @staticmethod
    def _model_to_map(pts_model: np.ndarray, ego_pos: np.ndarray,
                      ego_yaw: float) -> np.ndarray:
        """Model ego frame (x forward, y right) -> ROS map frame. Returns [N, 2]."""
        pts = np.asarray(pts_model, dtype=np.float64).reshape(-1, 2)
        local = np.stack([pts[:, 0], -pts[:, 1]], axis=1)
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        return local @ R.T + np.asarray(ego_pos, dtype=np.float64).reshape(1, 2)


    def _store_plan(
        self,
        pred_route:     Optional[torch.Tensor],
        pred_speed_wps: Optional[torch.Tensor],
        payload:        dict,
    ) -> Optional[dict]:
        """Freeze the model's prediction in the map frame for the control loop.

        The model predicts in the ego frame of the image it was handed.  By the
        time inference returns, that frame is 1.2-2.3 s old and -- at 5 m/s --
        some ten metres behind the car.  Anchoring the prediction to the pose it
        was actually computed from is what makes it trackable: the control loop
        can then keep following the same *physical* path from wherever the ego
        has got to, instead of re-reading a stale ego-frame path as though it
        started at the car's current position.

        Re-reading it that way is not a hypothetical error, it is the thing this
        replaced: the old code applied one control per prediction, so the ego
        was steered at ~0.5 Hz and each command -- including a saturated
        steering angle -- latched for ~2 s of simulated time.
        """
        if pred_route is None or pred_route.numel() == 0:
            self.get_logger().warn("No route prediction — plan not updated.",
                                   throttle_duration_sec=2.0)
            return None
        if pred_speed_wps is None or pred_speed_wps.shape[1] < 3:
            self.get_logger().warn("No speed waypoints — plan not updated.",
                                   throttle_duration_sec=2.0)
            return None

        route_model = pred_route[0].numpy().astype(np.float64)
        route_map   = self._model_to_map(
            route_model, payload["ego_pos_3d"][:2], payload["ego_yaw"])
        horizon_m = float(np.linalg.norm(np.diff(route_map, axis=0), axis=1).sum())

        with self._plan_lock:
            self._plan_seq += 1
            self._plan = {
                "route_map":   route_map,
                "speed_wps":   pred_speed_wps[0].numpy(),
                "created_sec": self.get_clock().now().nanoseconds * 1e-9,
                "seq":         self._plan_seq,
                "horizon_m":   horizon_m,
            }
        return {"horizon_m": horizon_m}

    def _control_cb(self) -> None:
        """Track the latest plan from the ego's current pose, at control_hz.

        This is the loop SimLingo expects to exist.  Upstream calls control_pid
        once per simulator frame -- agent_simlingo.py:672 infers every step and
        config_simlingo.py sets carla_fps = 20 -- so the policy is tuned to
        correct itself every 50 ms.

        Inference on the Orin takes 1.2-2.3 s, and under a real-time async
        simulator that is 1.2-2.3 s of *simulated* time as well.  Publishing one
        control per inference therefore steered the ego at a measured 0.50 Hz,
        40x coarser than the policy's own rate, with every command held in
        between.  Measured on 2026-09-11, route 1: `steer=-1.000` issued at rest
        and held 2.4 s took the ego from yaw +0.0 deg to +83.3 deg, the next
        command to -137.3 deg, and the route was over three seconds after it
        began.  Nothing downstream of that -- target points behind the car, tiny
        predicted speeds, saturated steering -- was a reading of the policy; it
        was a reading of a car that had already crashed.

        Splitting the two rates costs nothing in fidelity: the control law,
        its gains and its inputs are unchanged, and the model's latency shows up
        as what it honestly is -- the plan being stale -- rather than as a
        steering artefact.
        """
        if self._control_pub is None:
            return
        odom = self._latest_odom
        if odom is None:
            return
        with self._plan_lock:
            plan = self._plan
            controller = self._controller
        if plan is None or controller is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        stamp   = odom.header.stamp
        speed   = self._latest_speed

        age = now_sec - plan["created_sec"]
        if 0.0 < self._max_plan_age < age:
            self.get_logger().warn(
                f"Newest plan is {age:.1f}s old (> {self._max_plan_age:.1f}s) — braking.",
                throttle_duration_sec=2.0)
            self._publish_control(0.0, 0.0, 1.0, stamp)
            return

        ego_pos   = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
        ego_yaw   = _yaw_from_quaternion(odom.pose.pose.orientation)
        route_now = self._map_to_model(plan["route_map"], ego_pos, ego_yaw)

        if plan["seq"] != self._plan_progress_seq:
            self._plan_progress_seq = plan["seq"]
            self._plan_progress = 0
        nearest = int(np.argmin(np.linalg.norm(route_now, axis=1)))
        self._plan_progress = max(nearest, self._plan_progress)
        ahead = route_now[self._plan_progress + 1:]
        if ahead.shape[0] < 2:
            if plan["seq"] != self._plan_exhausted_seq:
                self._plan_exhausted_seq = plan["seq"]
                self.get_logger().warn(
                    f"OUTRUN: plan {plan['seq']} ({plan['horizon_m']:.1f} m) "
                    f"consumed in {age:.2f}s at {speed:.2f} m/s "
                    f"(reached waypoint {self._plan_progress}/"
                    f"{route_now.shape[0]}) — braking until the next "
                    f"prediction. The car is driving faster than the model "
                    f"can re-plan.")
            self._publish_control(0.0, 0.0, 1.0, stamp)
            return

        left_m = float(np.linalg.norm(np.diff(ahead, axis=0), axis=1).sum())

        dt_sec = (now_sec - self._last_control_sec
                  if self._last_control_sec is not None else 0.0)

        steer, throttle, brake, desired_speed = controller.step(
            route_waypoints=ahead,
            speed=speed,
            speed_waypoints=plan["speed_wps"],
        )

        throttle, brake = controller.apply_creep(
            throttle, brake, speed, now_sec, dt_sec
        )

        if self._first_control_sec is None:
            self._first_control_sec = now_sec
        if now_sec - self._first_control_sec < self._initial_brake:
            steer, throttle, brake = 0.0, 0.0, True

        self._publish_control(steer, throttle, 1.0 if brake else 0.0, stamp)

        self.get_logger().info(
            f"[SIMLINGO] PID: steer={steer:+.3f}  throttle={throttle:.2f}  "
            f"brake={int(brake)}  desired={desired_speed:.2f} m/s  "
            f"actual={speed:.2f} m/s  | plan {plan['seq']} age={age:.2f}s "
            f"{ahead.shape[0]}/{route_now.shape[0]} wps ahead "
            f"(at {self._plan_progress}) horizon={plan['horizon_m']:.1f}m "
            f"left={left_m:.1f}m",
            throttle_duration_sec=1.0,
        )

    def _publish_control(self, steer: float, throttle: float, brake: float, stamp) -> None:
        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = stamp
        cmd.steer        = float(steer)
        cmd.throttle     = float(throttle)
        cmd.brake        = float(brake)
        cmd.hand_brake   = False
        cmd.reverse      = False
        cmd.manual_gear_shift = False
        cmd.gear         = 1
        self._control_pub.publish(cmd)
        self._last_control_sec = self.get_clock().now().nanoseconds * 1e-9
        self._braking_on_stall = False

    def _watchdog_cb(self) -> None:
        """Brake if the control loop stopped publishing.

        CARLA keeps applying the last CarlaEgoVehicleControl it received, so a
        node that hangs would otherwise leave the car rolling on its last
        throttle value.

        This guards _control_cb, not inference: a slow model is normal and is
        handled by tracking the existing plan, while a model that has stopped
        predicting entirely is caught by max_plan_age_sec.  So control_timeout
        is a small multiple of the control period -- if it is ever raised above
        the inference time again, it is guarding the wrong thing.
        """
        if self._last_control_sec is None:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_control_sec <= self._control_timeout:
            return
        if not self._braking_on_stall:
            self.get_logger().warn(
                f"No control for {now_sec - self._last_control_sec:.1f}s — braking."
            )
        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.steer, cmd.throttle, cmd.brake = 0.0, 0.0, 1.0
        cmd.gear = 1
        self._control_pub.publish(cmd)
        self._braking_on_stall = True


    def _to_autoware_trajectory(
        self,
        pred_route:     Optional[torch.Tensor],
        pred_speed_wps: Optional[torch.Tensor],
        stamp,
    ) -> Trajectory:
        """Convert SimLingo's ego-relative 2-D waypoints to an Autoware Trajectory message.

        pred_route:     [B, 20, 2] — spatial route waypoints (ego frame, x=forward, y=left)
        pred_speed_wps: [B, 10, 2] — speed waypoints used to derive longitudinal velocity
        """
        traj = Trajectory()
        traj.header.stamp    = stamp
        traj.header.frame_id = "base_link"

        if pred_route is None or pred_route.numel() == 0:
            return traj

        route_np = pred_route[0].numpy()

        desired_speed = self._min_traj_speed
        if pred_speed_wps is not None and pred_speed_wps.shape[1] >= 3:
            spd = pred_speed_wps[0].numpy()
            desired_speed = max(
                float(np.linalg.norm(spd[0] - spd[2]) * 2.0),
                self._min_traj_speed,
            )

        for idx, pt in enumerate(route_np):
            tp = TrajectoryPoint()
            tp.pose.position.x = float(pt[0])
            tp.pose.position.y = float(pt[1])
            tp.pose.position.z = 0.0
            tp.pose.orientation.w = 1.0
            tp.longitudinal_velocity_mps = float(desired_speed)
            tp.lateral_velocity_mps      = 0.0
            tp.acceleration_mps2         = 0.0
            tp.heading_rate_rps          = 0.0
            sec_f = idx * _WP_DT
            tp.time_from_start = Duration(sec=int(sec_f), nanosec=int((sec_f % 1) * 1e9))
            traj.points.append(tp)

        return traj

    def _trajectory_to_markers(
        self,
        route_np: np.ndarray,
        ego_pos: np.ndarray,
        ego_yaw: float,
        stamp,
    ) -> MarkerArray:
        """Convert ego-frame 2-D waypoints to a map-frame MarkerArray for RViz."""
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)

        line_marker = Marker()
        line_marker.header.stamp    = stamp
        line_marker.header.frame_id = "map"
        line_marker.ns              = "simlingo_trajectory"
        line_marker.id              = 0
        line_marker.type            = Marker.LINE_STRIP
        line_marker.action          = Marker.ADD
        line_marker.scale.x         = 0.3
        line_marker.color           = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
        line_marker.pose.orientation.w = 1.0

        for pt in route_np:
            px, py = float(pt[0]), float(pt[1])
            p   = Point()
            p.x = cos_y * px - sin_y * py + ego_pos[0]
            p.y = sin_y * px + cos_y * py + ego_pos[1]
            p.z = float(ego_pos[2])
            line_marker.points.append(p)

        marker_array = MarkerArray()
        marker_array.markers.append(line_marker)
        return marker_array


    def _on_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
            return
        if metrics:
            self.get_logger().info(
                f"Inference done in {metrics['duration_sec']:.2f} s "
                f"({metrics['num_pts']} trajectory points)."
            )
            phases = metrics.get("_phases")
            if self._profile and phases:
                self.get_logger().info(_Phases.line(phases))
                self._phase_rows.append(phases)
                if (self._profile_every > 0
                        and len(self._phase_rows) % self._profile_every == 0):
                    self.get_logger().info(_Phases.summary(self._phase_rows))

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def _yaw_from_quaternion(q) -> float:
    """Extract yaw angle from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimLingoNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
