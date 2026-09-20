#!/usr/bin/env python3

"""ROS 2 node that streams CARLA sensor topics into the ORION VLA model and
publishes an Autoware trajectory (+ reasoning text) for inference benchmarking.

Design goal: the payload handed to the ORION model must be prepared in EXACTLY
the same way as the reference closed-loop agent
(``Orion/team_code/orion_b2d_agent.py::OrionAgent.run_step``). To guarantee that,
this node reuses ORION's own code rather than reimplementing the preprocessing:

  raw 6-cam images + odometry
      -> build the same ``results`` dict the agent builds
      -> ``Compose(cfg.inference_only_pipeline)``   (resize/crop, normalize, pad,
                                                      VQA tokenization)
      -> ``mmcv.parallel.collate``  (a.k.a. mm_collate_to_batch_form)
      -> ``model(batch, return_loss=False)``
      -> output[0]['pts_bbox']['ego_fut_preds']  -> (6, 2) ego trajectory

Only the *raw input sources* differ from the CARLA agent. They are the same
carla-simulator/ros-bridge topics consumed by the SimLingo reference node:
  * 6x ``sensor_msgs/Image`` (raw rgba8/bgra8), one per ORION camera in
    ORION_CAMERA_ORDER -> each decoded to BGR into its OWN input slot, matching
    OrionAgent.tick(). See orion_ros.camera_input.
  * ``std_msgs/Float32`` speed                          -> can_bus[7]
  * ``nav_msgs/Odometry``                               -> ego_pose / heading
  * ``sensor_msgs/Imu``                                 -> can_bus accel + angular
  * ``carla_msgs/CarlaRoute`` (latched)                 -> driving command

NOTE (deferred, to reconcile together later):
  * Topic names are all parameters with placeholder defaults.
  * The odometry->can_bus coordinate conventions replicate the agent's *formulas*
    (sign flips, ego_theta = -yaw + pi/2) under the assumption that the odometry
    frame matches CARLA's world frame and odom-yaw matches the CARLA compass.
    These sign/frame choices are isolated in ``_odom_to_canbus_ego_pose`` and
    marked with TODOs so we can adjust them once the real CARLA bridge topics and
    frames are known.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from carla_msgs.msg import CarlaRoute
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import ColorRGBA, Float32, String

from orion_ros.camera_input import (
    DEFAULT_CAMERA_TOPICS,
    ORION_CAMERA_ORDER,
    MultiCameraBuffer,
    decode_image,
)
from orion_ros import orion_speedups
from visualization_msgs.msg import Marker, MarkerArray

try:
    from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint

    _HAS_AUTOWARE_TRAJ = True
except ImportError:
    _HAS_AUTOWARE_TRAJ = False


ORION_TRAJ_DT = 0.5

ORION_MEMORY_MAX_DT = 2.0
ORION_AGENT_HZ = 20.0
ORION_FUT_TS = 6

ROUTE_MIN_DIST = 4.0
ROUTE_MAX_DIST = 50.0


def command2hot(command: int, max_dim: int = 6) -> np.ndarray:
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot


def command2nohot(command: int, max_dim: int = 6) -> int:
    if command < 0:
        command = 4
    command -= 1
    return command


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


_CUSTOM_FP16 = dict(map_head=False, pts_bbox_head=False)


def custom_wrap_fp16_model(model) -> None:
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True
    for module_name, v in _CUSTOM_FP16.items():
        if module_name in model._modules:
            model._modules[module_name].fp16_enabled = v


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


class OrionRosNode(Node):
    """ROS 2 node that runs ORION inference on live CARLA camera + odometry streams."""

    def __init__(self) -> None:
        super().__init__("orion_node")

        self.declare_parameter("orion_repo_path", "")
        self.declare_parameter(
            "orion_config_path",
            "adzoo/orion/configs/orion_stage3_agent.py",
        )
        self.declare_parameter("orion_checkpoint_path", "ckpts/Orion.pth")
        self.declare_parameter("precision", "fp16")
        self.declare_parameter("timestamp_mode", "sensor")

        self.declare_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        self.declare_parameter("front_camera_topic", "")
        self.declare_parameter("camera_sync_tolerance_sec", 0.1)
        self.declare_parameter("speed_topic", "/carla/hero/speed")
        self.declare_parameter("odometry_topic", "/carla/hero/odometry")
        self.declare_parameter("imu_topic", "/carla/hero/imu")
        self.declare_parameter("route_topic", "/carla/hero/global_plan")
        self.declare_parameter("trajectory_topic", "/orion/predicted_trajectory")
        self.declare_parameter("cot_topic", "/orion/cot")

        self.declare_parameter("inference_period_sec", 0.05)
        self.declare_parameter("require_new_frame", True)
        self.declare_parameter("publish_cot", True)
        self.declare_parameter("driving_command", 4)
        self.declare_parameter("replicate_jpeg_quality", 20)
        self.declare_parameter("merge_lora", True)
        self.declare_parameter("llm_flash_attn", True)
        self.declare_parameter("compile_targets", "heads,llm,vit")
        self.declare_parameter("compile_mode", "default")
        self.declare_parameter("vit_glue", True)
        self.declare_parameter("backbone_engine", "")
        self.declare_parameter("profile_stages", False)
        self.declare_parameter("decode_workers", 6)
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("min_trajectory_speed_mps", 0.0)

        self._device = torch.device("cuda")

        front_topic = str(self.get_parameter("front_camera_topic").value or "")
        if front_topic:
            self._camera_topics = [front_topic]
        else:
            self._camera_topics = [
                str(t) for t in (self.get_parameter("camera_topics").value or []) if str(t)
            ]
        if not self._camera_topics:
            raise ValueError(
                "camera_topics must list 6 raw sensor_msgs/Image topics in ORION order"
            )
        self._camera_sync_tol = float(
            self.get_parameter("camera_sync_tolerance_sec").value
        )
        self._timestamp_mode = str(self.get_parameter("timestamp_mode").value or "sensor")
        if self._timestamp_mode not in ("sensor", "agent"):
            raise ValueError("timestamp_mode must be 'sensor' or 'agent'")

        self._inference_period = float(self.get_parameter("inference_period_sec").value)
        self._require_new_frame = bool(self.get_parameter("require_new_frame").value)
        self._publish_cot = bool(self.get_parameter("publish_cot").value)
        self._driving_command = int(self.get_parameter("driving_command").value)
        self._jpeg_quality = int(self.get_parameter("replicate_jpeg_quality").value)
        self._merge_lora = bool(self.get_parameter("merge_lora").value)
        self._llm_flash_attn = bool(self.get_parameter("llm_flash_attn").value)
        self._compile_targets = [t.strip() for t in
                                 str(self.get_parameter("compile_targets").value or "").split(",")
                                 if t.strip()]
        self._compile_mode = str(self.get_parameter("compile_mode").value or "default")
        self._profile_stages = bool(self.get_parameter("profile_stages").value)
        self._vit_glue = bool(self.get_parameter("vit_glue").value)
        self._backbone_engine = str(self.get_parameter("backbone_engine").value or "")
        if self._backbone_engine:
            self._compile_targets = [t for t in self._compile_targets if t != "vit"]
        self._decoder = orion_speedups.ParallelDecoder(
            decode_image, workers=int(self.get_parameter("decode_workers").value))
        self._stage_timer: Optional[orion_speedups.StageTimer] = None
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._map_frame = str(self.get_parameter("map_frame_id").value)
        self._min_traj_speed = float(self.get_parameter("min_trajectory_speed_mps").value)

        qos = 10
        traj_topic = self.get_parameter("trajectory_topic").value
        if _HAS_AUTOWARE_TRAJ:
            self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, qos)
            self.get_logger().info(f"Publishing Autoware trajectories on {traj_topic}")
        else:
            self._trajectory_pub = None
            self.get_logger().warn(
                "autoware_planning_msgs not found; trajectory will only be published "
                "as markers. Install autoware_planning_msgs for full parity."
            )
        self._marker_pub = self.create_publisher(MarkerArray, traj_topic + "_markers", qos)

        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, qos)
        self.get_logger().info(f"CoT/reasoning topic: {cot_topic} (publish_cot={self._publish_cot})")

        self._cams: Optional[MultiCameraBuffer] = None
        self._latest_speed: float = 0.0
        self._latest_odom: Optional[Odometry] = None
        self._latest_imu: Optional[Imu] = None
        self._route: deque = deque()
        self._last_ref_stamp: Optional[tuple] = None
        self._frame_idx: int = 0
        self._scene_token: str = "route-0000"
        self._route_serial: int = 0
        self._route_fingerprint = None
        self._pending_route_reset: bool = False
        self._prev_sensor_ts = None
        self._memoryless_frames: int = 0
        self._dt_samples: int = 0
        self._last_dt_log_t: float = 0.0
        self._last_skip_log_t: float = 0.0

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None
        self._model = None
        self._inference_only_pipeline = None
        self._mm_collate = None
        self._get_box_type = None

        self.get_logger().info("Loading ORION model ...")
        self._executor.submit(self._setup_model).result()

        self._cams = MultiCameraBuffer(
            self, self._camera_topics, img_qos, self._camera_sync_tol
        )

        speed_topic = self.get_parameter("speed_topic").value
        self.create_subscription(Float32, speed_topic, self._speed_callback, 10)
        self.get_logger().info(f"Subscribed to speed topic: {speed_topic}")

        odom_topic = self.get_parameter("odometry_topic").value
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        imu_topic = self.get_parameter("imu_topic").value
        self.create_subscription(Imu, imu_topic, self._imu_callback, odom_qos)
        self.get_logger().info(f"Subscribed to IMU topic: {imu_topic}")

        route_topic = self.get_parameter("route_topic").value
        self.create_subscription(CarlaRoute, route_topic, self._route_callback, route_qos)
        self.get_logger().info(f"Subscribed to route topic: {route_topic}")

        self._timer = self.create_timer(self._inference_period, self._timer_callback)

    def _setup_model(self) -> None:
        """Build the ORION model + the exact inference pipeline used by the agent."""
        repo_path = str(self.get_parameter("orion_repo_path").value or "")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        if repo_path:
            os.chdir(repo_path)

        from mmcv import Config
        from mmcv.models import build_model
        from mmcv.utils import load_checkpoint
        from mmcv.datasets.pipelines import Compose
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form
        from mmcv.core.bbox import get_box_type

        self._mm_collate = mm_collate_to_batch_form
        self._get_box_type = get_box_type

        cfg_path = str(self.get_parameter("orion_config_path").value)
        ckpt_path = str(self.get_parameter("orion_checkpoint_path").value)
        cfg = Config.fromfile(cfg_path)

        precision = str(self.get_parameter("precision").value or "").lower()
        if precision == "fp16":
            cfg.model["fp16_infer"] = True
            cfg.model["fp16_eval"] = False
            cfg.model["fp32_infer"] = False
            self.get_logger().info("Inference precision: FP16 (fp16_infer)")
        elif precision == "fp32":
            cfg.model["fp16_infer"] = False
            cfg.model["fp16_eval"] = False
            cfg.model["fp32_infer"] = True
            self.get_logger().info("Inference precision: FP32 (fp32_infer)")
        else:
            self.get_logger().info("Inference precision: using config defaults")

        if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
            import importlib
            module_path = cfg.plugin_dir.rstrip("/").replace("/", ".")
            importlib.import_module(module_path)

        self._model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                                  test_cfg=cfg.get("test_cfg"))
        load_checkpoint(self._model, ckpt_path, map_location="cpu")
        self._model.cuda()
        self._model.eval()

        info = self.get_logger().info
        if self._merge_lora:
            orion_speedups.merge_lora(self._model, info)
        if self._llm_flash_attn:
            orion_speedups.patch_llm_flash_attention(self._model, info)
        if self._vit_glue and not self._backbone_engine:
            orion_speedups.patch_vit_blocks(self._model, info)
        if self._compile_targets:
            orion_speedups.compile_submodules(
                self._model, self._compile_targets, self._compile_mode, info)
        if self._backbone_engine:
            orion_speedups.install_trt_backbone(self._model, self._backbone_engine, info)
        custom_wrap_fp16_model(self._model)
        if self._profile_stages:
            self._stage_timer = orion_speedups.StageTimer()
            self._stage_timer.wrap_orion(self._model)

        pipeline_cfg = [
            t for t in cfg.inference_only_pipeline
            if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)
        ]
        self._inference_only_pipeline = Compose(pipeline_cfg)
        orion_speedups.parallelize_pipeline(
            self._inference_only_pipeline,
            workers=int(self.get_parameter("decode_workers").value),
            log=self.get_logger().info)

        torch.cuda.synchronize()
        self.get_logger().info("ORION model loaded and ready.")

    def _speed_callback(self, msg: Float32) -> None:
        self._latest_speed = float(msg.data)

    def _odometry_callback(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _imu_callback(self, msg: Imu) -> None:
        self._latest_imu = msg

    def _route_callback(self, msg: CarlaRoute) -> None:
        n = min(len(msg.poses), len(msg.road_options))
        route = deque()
        for i in range(n):
            p = msg.poses[i]
            route.append(
                (np.array([p.position.x, p.position.y], dtype=np.float64),
                 int(msg.road_options[i]))
            )
        self._route = route

        fp = (len(route),
              tuple(route[0][0]) if route else (),
              tuple(route[-1][0]) if route else ())
        if fp != self._route_fingerprint:
            self._route_fingerprint = fp
            self._route_serial += 1
            self._scene_token = f"route-{self._route_serial:04d}"
            self._pending_route_reset = True
            self.get_logger().info(
                f"Route received: {len(route)} waypoints — NEW route "
                f"({self._scene_token}); context will be reset before the next "
                f"inference."
            )
        else:
            self.get_logger().info(
                f"Route received: {len(route)} waypoints — same plan as "
                f"{self._scene_token}, keeping context."
            )

    def _reset_for_new_route(self) -> None:
        """Drop everything scoped to the route that just ended.

        Upstream gets this for free: leaderboard_evaluator.py builds a fresh
        agent object for every route and destroy()s it afterwards, so no state
        can cross a route boundary. Here only the thin gateway agent on the CARLA
        box is rebuilt -- this node is one long-lived process started once by
        start_orion.sh for the whole run -- so the boundary has to be recreated
        by hand. Same problem, and the same fix, as
        simlingo_node._reset_for_new_route.

        What actually carries over, worst first:

          * OrionHead / map-head temporal memory. `pre_update_memory` keeps the
            rolling memory whenever `scene_token` matches AND the stamp delta is
            under 2 s; with a constant token the first frame of a new route would
            be fused with the last frames of the previous one -- a different map,
            kilometres away. Clearing `test_flag` makes the model's own
            `forward_test` call `reset_memory()` on both heads on the next pass,
            which is the same path it uses for the very first frame.
          * The PID's 40-sample integral windows. OrionAgent.setup() builds a new
            PIDController per route. Routes tend to END with the ego blocked, so
            the windows are at their most biased exactly when the next route
            starts -- and at ORION's frame rate 40 samples is many seconds.
          * frame_idx, which the agent restarts from 0 each route.

        Sensor buffers (images, odometry, speed, IMU) are deliberately left
        alone: they are overwritten continuously and describe the world, not the
        route.
        """
        model = self._model
        if model is not None and hasattr(model, "test_flag"):
            model.test_flag = False
        self._frame_idx = 0
        self._last_ref_stamp = None
        self._prev_sensor_ts = None
        self._reset_controller_state()
        self._pending_route_reset = False
        self.get_logger().info(
            f"Context reset for {self._scene_token}: temporal memory, "
            f"frame counter and controller state cleared."
        )

    def _reset_controller_state(self) -> None:
        return

    def _check_memory_continuity(self, sensor_ts: float) -> None:
        """Report when ORION is running without its temporal memory.

        `pre_update_memory` zeroes the whole memory when consecutive timestamps
        are >= 2 s apart. In CARLA's synchronous mode that never happens (the sim
        waits for the agent, so frames are 0.05 s apart in sim time no matter how
        slow inference is). In async real-time it happens on EVERY frame as soon
        as inference exceeds ~2 s — the model silently becomes a single-frame
        detector, and no log line anywhere says so. Hence this one.
        """
        prev, self._prev_sensor_ts = self._prev_sensor_ts, sensor_ts
        if prev is None:
            return
        dt = abs(sensor_ts - prev)
        self._dt_samples += 1
        if dt < ORION_MEMORY_MAX_DT:
            return
        self._memoryless_frames += 1

        now = time.monotonic()
        if now - self._last_dt_log_t < 10.0:
            return
        self._last_dt_log_t = now
        pct = 100.0 * self._memoryless_frames / max(self._dt_samples, 1)
        extra = ""
        if self._timestamp_mode == "agent":
            extra = (" — timestamp_mode=agent is masking this: the model is told "
                     f"{1.0 / ORION_AGENT_HZ:.2f} s elapsed, so memory is kept but "
                     "its motion reasoning is given a false dt")
        self.get_logger().warn(
            f"temporal memory: {dt:.2f} s between frames exceeds ORION's "
            f"{ORION_MEMORY_MAX_DT:.1f} s retention window, so the rolling memory "
            f"is zeroed and ORION runs single-frame "
            f"({self._memoryless_frames}/{self._dt_samples} frames, {pct:.0f}%)"
            f"{extra}. Latency benchmarks are unaffected; driving quality is not "
            f"comparable to the paper's synchronous-mode results."
        )

    def _log_dispatch_state(self, reason: str, interval_sec: float = 2.0) -> None:
        """Throttled log explaining why the timer is NOT dispatching inference.
        The timer fires at ~20 Hz, so without throttling these would flood the log.
        Use it to tell apart the silent-skip branches: a stalled camera stream
        ("no new frame") vs. a hung forward pass ("busy") vs. missing inputs."""
        now = time.monotonic()
        if now - self._last_skip_log_t < interval_sec:
            return
        self._last_skip_log_t = now
        self.get_logger().info(f"[dispatch] {reason}")

    def _timer_callback(self) -> None:
        if self._active_future is not None and not self._active_future.done():
            self._log_dispatch_state("busy: inference still in flight")
            return

        missing = self._cams.missing() if self._cams is not None else ["(no subs)"]
        if missing:
            self._log_dispatch_state(
                f"waiting: no frame yet from {', '.join(missing)}"
            )
            return
        if self._latest_odom is None:
            self._log_dispatch_state("waiting: no odometry received yet")
            return

        ref_stamp = self._cams.reference_stamp()
        if self._require_new_frame and ref_stamp == self._last_ref_stamp:
            self._log_dispatch_state(
                f"skipped: no new front frame since last run "
                f"(stamp held at {ref_stamp[0]}.{ref_stamp[1]:09d}) — camera stream stalled?"
            )
            return
        self._last_ref_stamp = ref_stamp

        images = self._cams.snapshot()
        if images is None:
            self._log_dispatch_state("waiting: incomplete camera set")
            return
        snapshot = {
            "images": images,
            "speed": self._latest_speed,
            "odom": self._latest_odom,
            "imu": self._latest_imu,
        }
        self._active_future = self._executor.submit(self._run_inference, snapshot)
        self._active_future.add_done_callback(self._on_future_done)

    def _compute_driving_command(self, ego_xy: np.ndarray) -> int:
        """Time-varying navigation command fed to ORION (command2hot encoding):
          1=left, 2=right, 3=straight, 4=follow-lane, 5=lane-change-L, 6=lane-change-R.

        Faithful port of Bench2Drive RoutePlanner.run_step (team_code/planner.py):
        advance along the latched CarlaRoute using the ego position, popping
        waypoints already passed (within ROUTE_MIN_DIST, scanning up to
        ROUTE_MAX_DIST ahead), and return the road_option of the new front
        waypoint. Route poses and ego position are both in the ROS map frame, so
        the distance math matches the source (consistent frames). Falls back to
        the static `driving_command` parameter until the route arrives.
        """
        route = self._route
        if not route:
            return self._driving_command
        if len(route) == 1:
            return self._sanitize_cmd(route[0][1])

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(route)):
            if cumulative_distance > ROUTE_MAX_DIST:
                break
            cumulative_distance += np.linalg.norm(route[i][0] - route[i - 1][0])
            distance = np.linalg.norm(route[i][0] - ego_xy)
            if distance <= ROUTE_MIN_DIST and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i
        for _ in range(to_pop):
            if len(route) > 2:
                route.popleft()
        return self._sanitize_cmd(route[0][1])

    def _sanitize_cmd(self, cmd: int) -> int:
        """road_options is a uint8[] field, so CARLA's VOID (-1) arrives as 255.
        Anything outside the valid 1..6 RoadOption range -> static fallback."""
        cmd = int(cmd)
        return cmd if 1 <= cmd <= 6 else self._driving_command

    def _odom_to_canbus_ego_pose(self, odom: Odometry, imu: Optional[Imu], speed: float):
        """Build the 18-dim can_bus vector and the ego_pose/lidar2global matrices
        from a carla-simulator/ros-bridge ``nav_msgs/Odometry`` (+ ``sensor_msgs/Imu``).

        IMPORTANT — no manual sign flips here. The CARLA agent
        (orion_b2d_agent.py) reads RAW CARLA sensors (left-handed) and manually
        applies the CARLA->ROS conversion: ``can_bus[1] = -pos.y``,
        ``ego_theta = -compass + pi/2``, ``can_bus[11] *= -1``,
        ``can_bus[13:16] = -angular_velocity``. The ros-bridge has ALREADY done
        exactly that conversion (transforms.py negates y, negates yaw, negates
        angular y/z, and rotates linear velocity into the body frame). So we feed
        the bridge values directly; the resulting can_bus is numerically
        equivalent to what the agent produces for the same physical state. The
        ego_pose is the vehicle pose in the ROS "map" (ENU) frame == the agent's
        ego2world frame.
        """
        pose = odom.pose.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        ego_theta = float(Rotation.from_quat(quat).as_euler("zyx")[0])

        if speed is None or speed <= 0.0:
            lin = odom.twist.twist.linear
            speed = math.hypot(lin.x, lin.y)

        if imu is not None:
            acceleration = np.array(
                [imu.linear_acceleration.x, imu.linear_acceleration.y,
                 imu.linear_acceleration.z], dtype=np.float64
            )
            angular_velocity = np.array(
                [imu.angular_velocity.x, imu.angular_velocity.y,
                 imu.angular_velocity.z], dtype=np.float64
            )
        else:
            acceleration = np.zeros(3)
            ang = odom.twist.twist.angular
            angular_velocity = np.array([ang.x, ang.y, ang.z], dtype=np.float64)

        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        can_bus = np.zeros(18)
        can_bus[0] = pos[0]
        can_bus[1] = pos[1]
        can_bus[3:7] = rotation
        can_bus[7] = speed
        can_bus[10:13] = acceleration
        can_bus[13:16] = angular_velocity
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180

        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]

        lidar2global = ego2world @ LIDAR2EGO
        ego_pose = lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        return can_bus, ego_pose, ego_pose_inv, lidar2global

    def _build_batch(self, snapshot: dict):
        """Reproduce the agent's `results` dict, run the ORION inference pipeline,
        and collate into a model-ready batch. Returns (batch, meta)."""
        odom = snapshot["odom"]

        results: dict = {}
        results["lidar2img"] = []
        results["lidar2cam"] = []
        results["cam_intrinsic"] = []
        results["img"] = []
        results["folder"] = " "
        results["scene_token"] = self._scene_token
        frame_idx = self._frame_idx
        results["frame_idx"] = frame_idx
        self._frame_idx += 1

        sensor_ts = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        if self._timestamp_mode == "agent":
            results["timestamp"] = frame_idx / ORION_AGENT_HZ
        else:
            results["timestamp"] = sensor_ts
        self._check_memory_continuity(sensor_ts)
        results["box_type_3d"], _ = self._get_box_type("LiDAR")

        images = snapshot["images"]
        results["img"] = self._decoder(
            [images[cam] for cam in ORION_CAMERA_ORDER], self._jpeg_quality)
        for cam in ORION_CAMERA_ORDER:
            results["lidar2img"].append(LIDAR2IMG[cam])
            results["lidar2cam"].append(LIDAR2CAM[cam])
            results["cam_intrinsic"].append(
                np.matmul(LIDAR2IMG[cam], np.linalg.inv(LIDAR2CAM[cam]))
            )
        results["lidar2img"] = np.stack(results["lidar2img"], axis=0)
        results["lidar2cam"] = np.stack(results["lidar2cam"], axis=0)

        can_bus, ego_pose, ego_pose_inv, lidar2global = self._odom_to_canbus_ego_pose(
            odom, snapshot.get("imu"), float(snapshot.get("speed", 0.0))
        )
        results["can_bus"] = can_bus
        results["ego_pose"] = ego_pose
        results["ego_pose_inv"] = ego_pose_inv
        results["lidar2ego"] = LIDAR2EGO
        results["l2g_r_mat"] = lidar2global[0:3, 0:3]
        results["l2g_t"] = lidar2global[0:3, 3]

        ego_xy = np.array(
            [odom.pose.pose.position.x, odom.pose.pose.position.y], dtype=np.float64
        )
        command = self._compute_driving_command(ego_xy)
        results["command"] = command2nohot(command)
        results["ego_fut_cmd"] = command2hot(command)

        stacked_imgs = np.stack(results["img"], axis=-1)
        results["img_shape"] = stacked_imgs.shape
        results["ori_shape"] = stacked_imgs.shape
        results["pad_shape"] = stacked_imgs.shape

        results = self._inference_only_pipeline(results)
        batch = self._mm_collate([results], samples_per_gpu=1)

        for key, data in batch.items():
            if key != "img_metas":
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self._device)
            if key == "input_ids":
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        data[0][i][k] = data[0][i][k].to(self._device)

        meta = {
            "stamp": odom.header.stamp,
            "ego_pos": np.array(
                [odom.pose.pose.position.x, odom.pose.pose.position.y,
                 odom.pose.pose.position.z]
            ),
            "ego_rot": Rotation.from_quat(
                [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                 odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
            ).as_matrix(),
        }
        return batch, meta

    @torch.no_grad()
    def _run_inference(self, snapshot: dict) -> dict:
        t_start = time.time()
        if self._pending_route_reset:
            self._reset_for_new_route()
        batch, meta = self._build_batch(snapshot)
        t_prep = time.time()

        with torch.inference_mode():
            output = self._model(batch, return_loss=False)
        torch.cuda.synchronize()
        t_forward = time.time()
        stages = self._stage_timer.report() if self._stage_timer is not None else None

        out = output[0]
        ego_fut_preds = out["pts_bbox"]["ego_fut_preds"].cpu().numpy()

        ego_fut_preds = np.stack(
            [ego_fut_preds[:, 1], -ego_fut_preds[:, 0]], axis=1
        )

        cot = self._extract_reasoning(out.get("text_out")) if self._publish_cot else None

        if self._trajectory_pub is not None and _HAS_AUTOWARE_TRAJ:
            traj_msg = self._to_autoware_trajectory(ego_fut_preds, meta["stamp"])
            self._trajectory_pub.publish(traj_msg)
        self._marker_pub.publish(
            self._trajectory_to_markers(ego_fut_preds, meta["ego_pos"], meta["ego_rot"],
                                        meta["stamp"])
        )
        if cot:
            self._cot_pub.publish(String(data=cot))

        return {
            "prep_ms": (t_prep - t_start) * 1e3,
            "forward_ms": (t_forward - t_prep) * 1e3,
            "total_ms": (t_forward - t_start) * 1e3,
            "num_pts": len(ego_fut_preds),
            "stages": stages,
        }

    def _extract_reasoning(self, text_out) -> Optional[str]:
        """text_out is a list like [{'Q': str, 'A': [str]}, ...] from simple_test_pts."""
        if not text_out:
            return None
        parts = []
        for qa in text_out:
            try:
                q = qa.get("Q")
                a = qa.get("A")
                a = a[0] if isinstance(a, (list, tuple)) and a else a
                if a:
                    parts.append(f"Q: {q}\nA: {a}" if q else str(a))
            except AttributeError:
                continue
        return "\n\n".join(parts) if parts else None

    def _to_autoware_trajectory(self, traj_np: np.ndarray, stamp) -> "Trajectory":
        traj_msg = Trajectory()
        traj_msg.header.stamp = stamp
        traj_msg.header.frame_id = self._base_frame

        last_xy: Optional[tuple] = None
        for idx, point in enumerate(traj_np):
            tp = TrajectoryPoint()
            tp.pose.position.x = float(point[0])
            tp.pose.position.y = float(point[1])
            tp.pose.position.z = 0.0

            if idx + 1 < len(traj_np):
                dx = float(traj_np[idx + 1][0]) - float(point[0])
                dy = float(traj_np[idx + 1][1]) - float(point[1])
            elif last_xy is not None:
                dx = float(point[0]) - last_xy[0]
                dy = float(point[1]) - last_xy[1]
            else:
                dx, dy = 1.0, 0.0
            yaw = math.atan2(dy, dx)
            q = Rotation.from_euler("z", yaw).as_quat()
            tp.pose.orientation.x = float(q[0])
            tp.pose.orientation.y = float(q[1])
            tp.pose.orientation.z = float(q[2])
            tp.pose.orientation.w = float(q[3])

            if last_xy is None:
                speed = 0.0
            else:
                dist = math.hypot(float(point[0]) - last_xy[0], float(point[1]) - last_xy[1])
                speed = dist / ORION_TRAJ_DT
            tp.longitudinal_velocity_mps = float(max(speed, self._min_traj_speed))
            tp.lateral_velocity_mps = 0.0
            tp.acceleration_mps2 = 0.0
            tp.heading_rate_rps = 0.0

            t = idx * ORION_TRAJ_DT
            tp.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))

            traj_msg.points.append(tp)
            last_xy = (float(point[0]), float(point[1]))

        if len(traj_msg.points) >= 2:
            traj_msg.points[0].longitudinal_velocity_mps = (
                traj_msg.points[1].longitudinal_velocity_mps
            )
        return traj_msg

    def _trajectory_to_markers(self, traj_np: np.ndarray, ego_pos: np.ndarray,
                               ego_rot: np.ndarray, stamp) -> MarkerArray:
        marker_array = MarkerArray()
        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = self._map_frame
        line.ns = "orion_trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.3
        line.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        line.pose.orientation.w = 1.0
        for point in traj_np:
            local = np.array([float(point[0]), float(point[1]), 0.0])
            world = ego_rot @ local + ego_pos
            line.points.append(Point(x=float(world[0]), y=float(world[1]), z=float(world[2])))
        marker_array.markers.append(line)
        return marker_array

    def _on_future_done(self, future: Future) -> None:
        try:
            m = future.result()
        except Exception as exc:
            self.get_logger().error(f"ORION inference failed: {exc}")
            return
        if not m:
            return
        self.get_logger().info(
            f"ORION inference: total={m['total_ms']:.1f} ms "
            f"(prep={m['prep_ms']:.1f}, forward={m['forward_ms']:.1f}), "
            f"{m['num_pts']} waypoints, ~{1000.0 / max(m['total_ms'], 1e-3):.2f} FPS"
        )
        if m.get("stages"):
            self.get_logger().info(
                "ORION stages (ms): "
                + orion_speedups.StageTimer.format(m["stages"], m["forward_ms"]))

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._decoder.shutdown()
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = OrionRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
