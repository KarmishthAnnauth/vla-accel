#!/usr/bin/env python3

"""ROS 2 node that streams CARLA sensor topics into the ORION VLA model and
drives a self-contained closed loop: the model's predicted trajectory is fed to
a Bench2Drive PID controller in-node, which publishes carla_msgs/CarlaEgoVehicleControl
(steer/throttle/brake) directly on /carla/hero/vehicle_control_cmd — exactly like
the leaderboard agent's carla.VehicleControl, so no carla_ackermann_control node is
needed. This is the PID variant of orion_node.py (which instead publishes an
Autoware Trajectory for a downstream Stanley controller).

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
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaRoute
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
    REFERENCE_CAMERA,
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

ORION_SPEED_CAP_MPS = 5.0

AGENT_GNSS_MOUNT_X = -1.4

MAX_STEER_ANGLE = 0.7

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


class OrionWithPidRosNode(Node):
    """ORION inference on live CARLA streams, with an in-node PID controller that
    publishes CarlaEgoVehicleControl directly (self-contained closed loop)."""

    def __init__(self) -> None:
        super().__init__("orion_withpid_node")

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
        self.declare_parameter("pipeline_prep", True)
        self.declare_parameter("pipeline_margin_ms", 40.0)
        self.declare_parameter("map_head_slice", True)
        self.declare_parameter("vit_input_size", 640)
        self.declare_parameter("rear_view_refresh_every", 1)
        self.declare_parameter("llm_down_proj_transpose", True)
        self.declare_parameter("llm_int8", False)
        self.declare_parameter("llm_int8_stats", "")
        self.declare_parameter("llm_int8_alpha", 0.8)
        self.declare_parameter("llm_int8_skip_layers", [0, 1, 30, 31])
        self.declare_parameter("llm_int8_targets", "gate_proj,up_proj,down_proj")
        self.declare_parameter("overlap_heads", False)
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("min_trajectory_speed_mps", 0.0)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("plan_stall_timeout_sec", 6.0)
        self.declare_parameter("brake_when_stale", True)
        self.declare_parameter("gnss_mount_offset_x", AGENT_GNSS_MOUNT_X)
        self.declare_parameter("control_trace", False)
        self.declare_parameter("control_trace_path", "")

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
        self._pipeline_prep = bool(self.get_parameter("pipeline_prep").value)
        self._pipeline_margin_ms = float(self.get_parameter("pipeline_margin_ms").value)
        self._map_head_slice = bool(self.get_parameter("map_head_slice").value)
        self._vit_input_size = int(self.get_parameter("vit_input_size").value)
        self._rear_refresh = int(self.get_parameter("rear_view_refresh_every").value)
        self._stagger: Optional[orion_speedups.StaggeredViews] = None
        self._llm_down_t = bool(self.get_parameter("llm_down_proj_transpose").value)
        self._llm_int8 = bool(self.get_parameter("llm_int8").value)
        self._llm_int8_stats = str(self.get_parameter("llm_int8_stats").value or "")
        self._llm_int8_alpha = float(self.get_parameter("llm_int8_alpha").value)
        self._llm_int8_skip = [int(i) for i in (self.get_parameter("llm_int8_skip_layers").value or [])]
        self._llm_int8_targets = [t.strip() for t in
                                  str(self.get_parameter("llm_int8_targets").value or "").split(",") if t.strip()]
        self._overlap_heads = bool(self.get_parameter("overlap_heads").value)
        self._backbone_engine = str(self.get_parameter("backbone_engine").value or "")
        if self._backbone_engine:
            self._compile_targets = [t for t in self._compile_targets if t != "vit"]
        self._decoder = orion_speedups.ParallelDecoder(
            decode_image, workers=int(self.get_parameter("decode_workers").value))
        self._stage_timer: Optional[orion_speedups.StageTimer] = None
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._map_frame = str(self.get_parameter("map_frame_id").value)
        self._min_traj_speed = float(self.get_parameter("min_trajectory_speed_mps").value)
        self._control_hz = float(self.get_parameter("control_hz").value)
        self._plan_stall_timeout = float(
            self.get_parameter("plan_stall_timeout_sec").value)
        self._brake_when_stale = bool(self.get_parameter("brake_when_stale").value)
        self._gnss_offset_x = float(self.get_parameter("gnss_mount_offset_x").value)
        if self._gnss_offset_x:
            self.get_logger().info(
                f"ego localised {self._gnss_offset_x:+.2f} m (vehicle frame) from the "
                f"odometry origin, matching the reference agent's GNSS mount")
        self._control_trace = bool(self.get_parameter("control_trace").value)
        self._trace_fh = None
        self._trace_tick = 0
        self._trace_plan_seq = 0
        self._trace_last_arrived = None
        self._trace_ticks_on_plan = 0
        self._last_pid_meta = None
        if self._control_trace:
            import os as _os
            _p = str(self.get_parameter("control_trace_path").value or "")
            if not _p:
                _p = "/benchmarking/log/control_trace_%s.csv" % (
                    time.strftime("%Y%m%d_%H%M%S"))
            try:
                _os.makedirs(_os.path.dirname(_p), exist_ok=True)
                self._trace_fh = open(_p, "w", buffering=1)
                self._trace_fh.write(
                    "t,tick,plan_seq,swap,ticks_on_plan,plan_age,stalled,"
                    "speed,desired_speed,delta,angle,angle_final,aim_x,aim_y,"
                    "steer,throttle,brake\n")
                self.get_logger().info("control trace -> %s" % _p)
            except Exception as _exc:
                self.get_logger().warn(
                    "control_trace disabled: cannot write %s (%s)" % (_p, _exc))
                self._control_trace = False

        qos = 10
        self.declare_parameter("control_topic", "/carla/hero/vehicle_control_cmd")
        control_topic = self.get_parameter("control_topic").value
        self._control_pub = self.create_publisher(
            CarlaEgoVehicleControl, control_topic,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1))
        self.get_logger().info(f"Publishing CarlaEgoVehicleControl on {control_topic}")

        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, qos)
        self.get_logger().info(f"CoT/reasoning topic: {cot_topic} (publish_cot={self._publish_cot})")

        self._cams: Optional[MultiCameraBuffer] = None
        self._latest_speed: float = 0.0
        self._latest_odom: Optional[Odometry] = None
        self._traj_state: Optional[tuple] = None
        self._last_control: Optional[tuple] = None
        self._ctrl_count: int = 0
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
        self._prep_executor = ThreadPoolExecutor(max_workers=1)
        self._prep_future: Optional[Future] = None
        self._prep_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._fwd_expected_end: Optional[float] = None
        self._fwd_ms_avg: Optional[float] = None
        self._prep_ms_avg: float = 150.0
        self._last_fwd_start: Optional[float] = None
        self._last_period_mark: Optional[float] = None
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
        self._control_timer = self.create_timer(
            1.0 / self._control_hz, self._control_callback)
        self.get_logger().info(
            f"Control loop at {self._control_hz:g} Hz "
            f"(car stopped if no new plan arrives for "
            f"{self._plan_stall_timeout:g} s)")

    def _setup_model(self) -> None:
        """Build the ORION model + the exact inference pipeline used by the agent."""
        repo_path = str(self.get_parameter("orion_repo_path").value or "")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        if repo_path:
            os.chdir(repo_path)

        from team_code.pid_controller import PIDController
        self._pid = PIDController()

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
        if self._vit_input_size != 640:
            orion_speedups.set_vit_input_size(self._model, self._vit_input_size, info)
        if self._llm_int8:
            orion_speedups.apply_llm_int8(self._model, self._llm_int8_stats, self._llm_int8_alpha,
                                          self._llm_int8_skip, info, self._llm_int8_targets)
            if self._llm_down_t and "down_proj" not in self._llm_int8_targets:
                orion_speedups.transpose_llm_down_proj(self._model, info)
        elif self._llm_down_t:
            orion_speedups.transpose_llm_down_proj(self._model, info)
        if self._compile_targets:
            orion_speedups.compile_submodules(
                self._model, self._compile_targets, self._compile_mode, info)
        if self._backbone_engine:
            orion_speedups.install_trt_backbone(self._model, self._backbone_engine, info)
        if self._map_head_slice:
            orion_speedups.slice_map_head_one2one(self._model, info)
        if self._rear_refresh > 1:
            self._stagger = orion_speedups.install_staggered_views(self._model, self._rear_refresh, info)
        if self._overlap_heads:
            orion_speedups.overlap_heads(self._model, info)
        custom_wrap_fp16_model(self._model)
        if self._profile_stages:
            self._stage_timer = orion_speedups.StageTimer()
            self._stage_timer.wrap_orion(self._model)

        pipeline_cfg = [
            t for t in cfg.inference_only_pipeline
            if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)
        ]
        if self._vit_input_size != 640:
            orion_speedups.set_pipeline_input_size(pipeline_cfg, self._vit_input_size)
            self.get_logger().info(f"pipeline: ViT input {self._vit_input_size}x{self._vit_input_size}")
        self._inference_only_pipeline = Compose(pipeline_cfg)
        orion_speedups.parallelize_pipeline(
            self._inference_only_pipeline,
            workers=int(self.get_parameter("decode_workers").value),
            log=self.get_logger().info)

        self._warmup()
        self.get_logger().info("ORION model loaded and ready.")

    def _warmup(self) -> None:
        """Run ONE full forward on a black dummy frame so the first real
        inference isn't a cold-start outlier (CUDA kernel JIT, cuDNN autotune,
        allocator). Goes through the real pipeline so shapes are guaranteed
        correct. Runs in the model worker thread (same CUDA context as
        inference). Best-effort: logged, never fatal."""
        try:
            t0 = time.time()
            dummy = np.zeros((900, 1600, 3), dtype=np.uint8)
            lidar2global = np.eye(4) @ LIDAR2EGO
            results = {
                "lidar2img":     np.stack([LIDAR2IMG[c] for c in ORION_CAMERA_ORDER]),
                "lidar2cam":     np.stack([LIDAR2CAM[c] for c in ORION_CAMERA_ORDER]),
                "cam_intrinsic": [np.matmul(LIDAR2IMG[c], np.linalg.inv(LIDAR2CAM[c]))
                                  for c in ORION_CAMERA_ORDER],
                "img":         [dummy for _ in ORION_CAMERA_ORDER],
                "folder": " ", "scene_token": " ", "frame_idx": 0, "timestamp": 0.0,
                "box_type_3d": self._get_box_type("LiDAR")[0],
                "can_bus":     np.zeros(18),
                "command":     command2nohot(4),
                "ego_fut_cmd": command2hot(4),
                "ego_pose":    lidar2global,
                "ego_pose_inv": invert_matrix_egopose_numpy(lidar2global),
                "lidar2ego":   LIDAR2EGO,
                "l2g_r_mat":   lidar2global[0:3, 0:3],
                "l2g_t":       lidar2global[0:3, 3],
            }
            stacked = np.stack(results["img"], axis=-1)
            results["img_shape"] = results["ori_shape"] = results["pad_shape"] = stacked.shape
            results = self._inference_only_pipeline(results)
            batch = self._mm_collate([results], samples_per_gpu=1)
            for key, data in batch.items():
                if key != "img_metas" and torch.is_tensor(data[0]):
                    data[0] = data[0].to(self._device)
                if key == "input_ids":
                    for i in range(len(data[0])):
                        for k in range(len(data[0][i])):
                            data[0][i][k] = data[0][i][k].to(self._device)
            with torch.inference_mode():
                self._model(batch, return_loss=False)
                if self._stagger is not None:
                    self._model(batch, return_loss=False)
            torch.cuda.synchronize()
            if self._stagger is not None:
                self._stagger.reset()
            if hasattr(self._model, "test_flag"):
                self._model.test_flag = False
            self.get_logger().info(f"Model warmup OK in {(time.time() - t0) * 1e3:.0f} ms")
        except Exception as e:
            self.get_logger().warn(f"Model warmup skipped ({type(e).__name__}: {e})")

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
        if self._stagger is not None:
            self._stagger.reset()
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
        from team_code.pid_controller import PIDController
        self._pid = PIDController()
        self._traj_state = None
        self._last_control = None

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
            self._maybe_pipeline_prep()
            self._log_dispatch_state("busy: inference still in flight")
            return
        if self._prep_future is not None:
            self._try_start_from_prepared()
            return
        snapshot = self._take_snapshot()
        if snapshot is None:
            return
        self._start_forward(self._executor.submit(self._run_inference, snapshot),
                            include_prep=True)

    def _take_snapshot(self) -> Optional[dict]:
        """Gate on complete, fresh inputs and snapshot the LATEST of each."""
        missing = self._cams.missing() if self._cams is not None else ["(no subs)"]
        if missing:
            self._log_dispatch_state(
                f"waiting: no frame yet from {', '.join(missing)}"
            )
            return None
        if self._latest_odom is None:
            self._log_dispatch_state("waiting: no odometry received yet")
            return None

        ref_stamp = self._cams.reference_stamp()
        if self._require_new_frame and ref_stamp == self._last_ref_stamp:
            self._log_dispatch_state(
                f"skipped: no new front frame since last run "
                f"(stamp held at {ref_stamp[0]}.{ref_stamp[1]:09d}) — camera stream stalled?"
            )
            return None
        self._last_ref_stamp = ref_stamp

        images = self._cams.snapshot()
        if images is None:
            self._log_dispatch_state("waiting: incomplete camera set")
            return None
        return {
            "images": images,
            "speed": self._latest_speed,
            "odom": self._latest_odom,
            "imu": self._latest_imu,
            "t_dispatch": time.time(),
        }

    def _start_forward(self, future: Future, include_prep: bool) -> None:
        now = time.time()
        self._last_fwd_start = now
        expected = (self._fwd_ms_avg or 0.0) + (self._prep_ms_avg if include_prep else 0.0)
        self._fwd_expected_end = now + expected / 1e3 if self._fwd_ms_avg else None
        self._active_future = future
        future.add_done_callback(self._on_future_done)

    def _maybe_pipeline_prep(self) -> None:
        """While a forward is in flight: start preparing the next frame so it
        is ready right when the GPU frees up. Started late on purpose (see the
        pipeline_prep parameter) so the input is no older than it would be in
        the sequential path."""
        if (not self._pipeline_prep or self._prep_future is not None
                or self._fwd_expected_end is None or self._fwd_ms_avg is None
                or self._fwd_ms_avg < 2.0 * self._prep_ms_avg + self._pipeline_margin_ms):
            return
        if time.time() < self._fwd_expected_end - (self._prep_ms_avg + self._pipeline_margin_ms) / 1e3:
            return
        snapshot = self._take_snapshot()
        if snapshot is None:
            return
        self._prep_future = self._prep_executor.submit(self._prepare, snapshot)
        self._prep_future.add_done_callback(self._on_prep_done)

    def _on_prep_done(self, future: Future) -> None:
        self._try_start_from_prepared()

    def _try_start_from_prepared(self) -> None:
        """Start the forward on the prepared batch if the GPU worker is free.
        Called from the timer, the prep-done callback and the forward-done
        callback; the lock makes exactly one of them dispatch."""
        with self._dispatch_lock:
            fut = self._prep_future
            if fut is None or not fut.done():
                return
            if self._active_future is not None and not self._active_future.done():
                return
            self._prep_future = None
            try:
                prepared = fut.result()
            except Exception as exc:
                import traceback
                self.get_logger().error(
                    f"ORION prep failed: {exc}\n{traceback.format_exc()}")
                return
            self._start_forward(self._executor.submit(self._forward, prepared),
                                include_prep=False)

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

    def _agent_ego_xy(self, xy, ego_theta: float) -> np.ndarray:
        """Vehicle-origin ENU position -> the position the reference agent uses.

        The agent's GNSS sits at (gnss_mount_offset_x, 0) in the vehicle frame;
        rotate that into ENU by the vehicle->ENU yaw and add it. Yaw only, like
        the agent, which works in a flat 2-D world.
        """
        xy = np.asarray(xy, dtype=np.float64)[:2]
        d = self._gnss_offset_x
        if not d:
            return xy.copy()
        return xy + np.array([d * math.cos(ego_theta), d * math.sin(ego_theta)],
                             dtype=np.float64)

    def _odom_to_canbus_ego_pose(self, odom: Odometry, imu: Optional[Imu], speed: float):
        """Build the 18-dim can_bus vector and the ego_pose/lidar2global matrices
        from a carla-simulator/ros-bridge ``nav_msgs/Odometry`` (+ ``sensor_msgs/Imu``).

        IMPORTANT — no manual sign flips here. The CARLA agent
        (orion_b2d_agent.py) reads RAW CARLA sensors (left-handed) and manually
        applies the CARLA->ROS conversion: ``can_bus[1] = -pos.y``,
        ``ego_theta = -compass + pi/2``, ``can_bus[11] *= -1``,
        ``can_bus[13:16] = -angular_velocity``. The ros-bridge does MOST of that
        conversion already (transforms.py negates position y, negates yaw, and
        rotates linear velocity into the body frame), so those need no flip here.

        Angular velocity is the exception, and it was wrong until 2026-09-16.
        ``carla_ros_bridge/imu.py`` publishes ``(-gx, +gy, -gz)`` -- it negates
        gyro **x and z**, not y, because angular velocity is a pseudovector and
        flipping the y axis inverts the other two components. The agent instead
        negates all three (``can_bus[13:16] = -angular_velocity`` over the RAW
        CARLA gyro), giving ``(-gx, -gy, -gz)``. Feeding the bridge values
        straight through therefore put ``can_bus[14]`` at ``+gy`` where the
        reference has ``-gy``, so the y component is flipped below. The
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

        gnss_xy = self._agent_ego_xy(pos[:2], ego_theta)

        can_bus = np.zeros(18)
        can_bus[0] = gnss_xy[0]
        can_bus[1] = gnss_xy[1]
        can_bus[3:7] = rotation
        can_bus[7] = speed
        can_bus[10:13] = acceleration
        can_bus[13:16] = np.array(
            [angular_velocity[0], -angular_velocity[1], angular_velocity[2]],
            dtype=np.float64)
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

        ego_xy = can_bus[0:2].copy()
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

        near_node = (
            np.array(self._route[1][0], dtype=np.float64) if len(self._route) > 1
            else (np.array(self._route[0][0], dtype=np.float64) if self._route else None)
        )

        meta = {
            "stamp": snapshot["images"][REFERENCE_CAMERA][0],
            "ego_pos": np.array(
                [odom.pose.pose.position.x, odom.pose.pose.position.y,
                 odom.pose.pose.position.z]
            ),
            "ego_rot": Rotation.from_quat(
                [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                 odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
            ).as_matrix(),
            "ego_xy": ego_xy,
            "ego_theta": float(can_bus[16]),
            "near_node": near_node,
            "speed": float(snapshot.get("speed", 0.0)),
        }
        return batch, meta

    def _run_inference(self, snapshot: dict) -> dict:
        """Sequential path: prep + forward on the GPU worker."""
        return self._forward(self._prepare(snapshot))

    def _prepare(self, snapshot: dict) -> dict:
        """CPU half of a frame (decode, pipeline, collate, H2D). Runs on the GPU
        worker in the sequential path or on the prep worker when pipelined;
        the lock keeps _build_batch's bookkeeping (frame_idx, memory checks,
        route planner) strictly ordered."""
        with self._prep_lock:
            t0 = time.time()
            batch, meta = self._build_batch(snapshot)
            prep_ms = (time.time() - t0) * 1e3
        self._prep_ms_avg = 0.8 * self._prep_ms_avg + 0.2 * prep_ms
        return {"batch": batch, "meta": meta, "snapshot": snapshot, "prep_ms": prep_ms}

    @torch.no_grad()
    def _forward(self, prepared: dict) -> dict:
        t_start = time.time()
        if self._pending_route_reset:
            self._reset_for_new_route()
            prepared = self._prepare(prepared["snapshot"])
        batch, meta, snapshot = prepared["batch"], prepared["meta"], prepared["snapshot"]
        t_prep = time.time()

        with torch.inference_mode():
            output = self._model(batch, return_loss=False)
        torch.cuda.synchronize()
        t_forward = time.time()
        stages = self._stage_timer.report() if self._stage_timer is not None else None
        forward_ms = (t_forward - t_prep) * 1e3
        self._fwd_ms_avg = forward_ms if self._fwd_ms_avg is None else 0.8 * self._fwd_ms_avg + 0.2 * forward_ms
        period_ms = (t_start - self._last_period_mark) * 1e3 if self._last_period_mark else None
        self._last_period_mark = t_start

        out = output[0]
        ego_fut_preds = out["pts_bbox"]["ego_fut_preds"].cpu().numpy()

        self._traj_state = (
            ego_fut_preds.copy(),
            np.asarray(meta["ego_xy"], dtype=np.float64).copy(),
            float(meta["ego_theta"]),
            float(snapshot.get("t_dispatch", time.time())),
            time.time(),
        )

        cot = self._extract_reasoning(out.get("text_out")) if self._publish_cot else None
        if cot:
            self._cot_pub.publish(String(data=cot))

        return {
            "prep_ms": prepared["prep_ms"],
            "forward_ms": forward_ms,
            "total_ms": prepared["prep_ms"] + forward_ms,
            "period_ms": period_ms,
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

    @staticmethod
    def _rot(angle: float) -> np.ndarray:
        c, sn = math.cos(angle), math.sin(angle)
        return np.array([[c, -sn], [sn, c]])

    def _rebase_plan(self, traj: np.ndarray, age: float,
                     ego_xy0: np.ndarray, theta0: float,
                     ego_xy_now: np.ndarray, theta_now: float) -> np.ndarray:
        """The plan, re-cut from where the ego is NOW and in its current frame.

        Two corrections, both required, for different reasons:

        **Time.** ORION's waypoints are 0.5 s apart measured from the frame it
        looked at, and that frame is ~2 s old by the time the forward pass
        returns. Handing control_pid `waypoints[0]` unmodified aims the car at a
        point it drove past a second ago -- and worse, control_pid reads
        ``desired_speed`` from ``||waypoints[0]||``, so a point the car has
        nearly reached reads as "slow down". The plan is therefore resampled
        from `age` onwards: element j is where the model expected to be at
        ``age + 0.5*(j+1)``.

        **Frame.** The waypoints are in the ego frame of the capture instant and
        the ego has moved since, so they are rotated and translated into the
        current ego frame using odometry. This is also what makes the scheme
        self-correcting: if the car did not go where the model predicted, the
        geometry is still relative to where it actually is.

        Beyond the 3 s horizon the last segment's velocity is extended. That is
        not decoration -- a 2 s forward pass eats 2 s of a 3 s plan, so without
        it the controller runs dry for the second half of every inference cycle
        and the car would brake every other second.

        The plan is treated as rooted at the ego, which is how control_pid reads
        it (``||waypoints[0]||`` as distance travelled in 0.5 s). The LIDAR2EGO
        offset is not applied for that reason; it would also almost cancel,
        being constant in the vehicle frame.
        """
        knots = np.vstack((np.zeros((1, 2)), np.asarray(traj, dtype=np.float64)))
        horizon = ORION_TRAJ_DT * (len(knots) - 1)
        v_end = (knots[-1] - knots[-2]) / ORION_TRAJ_DT

        def sample(t: float) -> np.ndarray:
            if t >= horizon:
                return knots[-1] + (t - horizon) * v_end
            u = max(t, 0.0) / ORION_TRAJ_DT
            i = min(int(math.floor(u)), len(knots) - 2)
            return knots[i] + (u - i) * (knots[i + 1] - knots[i])

        pts = np.array([sample(age + ORION_TRAJ_DT * (j + 1))
                        for j in range(len(traj))])

        raw0 = np.pi / 2.0 - theta0
        raw_now = np.pi / 2.0 - theta_now
        o = self._rot(raw0) @ (np.asarray(ego_xy_now) - np.asarray(ego_xy0))
        return (self._rot(raw_now - raw0) @ (pts - o).T).T

    def _control_callback(self) -> None:
        """Track the latest plan. Runs at control_hz, including during a forward pass."""
        state = self._traj_state
        if state is None or self._latest_odom is None:
            return
        traj, ego_xy0, theta0, t0, t_arrived = state

        now = time.time()
        stalled = now - t_arrived
        if stalled > self._plan_stall_timeout:
            self._stop_the_car(
                f"no new plan for {stalled:.1f}s "
                f"(limit {self._plan_stall_timeout:g}s) -- the model has stopped "
                f"answering")
            return
        age = now - t0

        odom = self._latest_odom
        ego_xy_now = np.array([odom.pose.pose.position.x,
                               odom.pose.pose.position.y], dtype=np.float64)
        theta_now = float(Rotation.from_quat([
            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w,
        ]).as_euler("zyx")[0])
        ego_xy_now = self._agent_ego_xy(ego_xy_now, theta_now)

        plan = self._rebase_plan(traj, age, ego_xy0, theta0, ego_xy_now, theta_now)

        near_node = (
            np.array(self._route[1][0], dtype=np.float64) if len(self._route) > 1
            else (np.array(self._route[0][0], dtype=np.float64) if self._route else None)
        )
        meta = {
            "stamp": self.get_clock().now().to_msg(),
            "ego_xy": ego_xy_now,
            "ego_theta": theta_now,
            "near_node": near_node,
            "speed": float(self._latest_speed),
        }
        control = self._compute_control(plan, meta)
        if control is None:
            return
        self._control_pub.publish(control)
        self._last_control = (control.steer, control.throttle, control.brake)
        self._ctrl_count += 1
        if self._control_trace:
            self._write_control_trace(t_arrived, age, stalled, control)

    def _write_control_trace(self, t_arrived, age, stalled, control) -> None:
        """One CSV row per control tick. See the control_trace parameter."""
        if self._trace_fh is None:
            return
        m = self._last_pid_meta or {}
        swap = 0
        if t_arrived != self._trace_last_arrived:
            self._trace_last_arrived = t_arrived
            self._trace_plan_seq += 1
            self._trace_ticks_on_plan = 0
            swap = 1
        self._trace_ticks_on_plan += 1
        self._trace_tick += 1
        nan = float("nan")
        aim = m.get("aim") or (nan, nan)
        def g(k):
            v = m.get(k, nan)
            try:
                return float(v)
            except (TypeError, ValueError):
                return nan
        try:
            self._trace_fh.write(
                "%.6f,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.5f,%.5f,"
                "%.4f,%.4f,%.5f,%.4f,%.4f\n" % (
                    time.time(), self._trace_tick, self._trace_plan_seq, swap,
                    self._trace_ticks_on_plan, age, stalled,
                    g("speed"), g("desired_speed"), g("delta"),
                    g("angle"), g("angle_final"),
                    float(aim[0]), float(aim[1]),
                    control.steer, control.throttle, control.brake))
        except Exception:
            pass

    def _stop_the_car(self, why: str) -> None:
        if not self._brake_when_stale:
            return
        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.steer = 0.0
        cmd.throttle = 0.0
        cmd.brake = 1.0
        self._control_pub.publish(cmd)
        self._last_control = (0.0, 0.0, 1.0)
        self.get_logger().warn(f"braking: {why}", throttle_duration_sec=5.0)

    def _compute_control(self, ego_fut_preds: np.ndarray, meta: dict) -> Optional[CarlaEgoVehicleControl]:
        """Bench2Drive PID over the predicted trajectory -> CarlaEgoVehicleControl.
        Faithful port of orion_b2d_agent: build the near-waypoint target in the
        vehicle-local frame, run control_pid, and publish the resulting
        steer/throttle/brake DIRECTLY as carla.VehicleControl (no AckermannDrive
        round-trip, so no carla_ackermann_control node is needed).

        Frame note: this node works in the ROS map/ENU frame, where ego_theta is
        the vehicle->ENU yaw (== the agent's ego_theta). The agent's raw compass
        heading is therefore raw_theta = pi/2 - ego_theta. Route and ego position
        are already ENU (the ros-bridge applied the CARLA->ROS conversion), so the
        world->local rotation needs no Y-flip — unlike orion_lite which flips Y on
        the raw CARLA coordinates.
        """
        near_node = meta.get("near_node")
        if near_node is None:
            self.get_logger().warn("No global plan yet; not publishing control.",
                                   throttle_duration_sec=5.0)
            return None

        ego_xy = meta["ego_xy"]
        raw_theta = np.pi / 2.0 - meta["ego_theta"]
        near_xy_world = np.array([near_node[0] - ego_xy[0], near_node[1] - ego_xy[1]])
        rot = np.array([[math.cos(raw_theta), -math.sin(raw_theta)],
                        [math.sin(raw_theta),  math.cos(raw_theta)]])
        local_command_xy = rot @ near_xy_world

        steer, throttle, brake, metadata = self._pid.control_pid(
            ego_fut_preds, np.float64(meta["speed"]), local_command_xy)
        self._last_pid_meta = metadata

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 0.75))
        brake = float(brake)
        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0
        if float(meta["speed"]) > ORION_SPEED_CAP_MPS:
            throttle = 0.0

        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = meta["stamp"]
        cmd.steer = steer
        cmd.throttle = throttle
        cmd.brake = brake
        return cmd

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
        self._try_start_from_prepared()
        try:
            m = future.result()
        except Exception as exc:
            import traceback
            self.get_logger().error(
                f"ORION inference failed: {exc}\n{traceback.format_exc()}")
            return
        if not m:
            return
        ctrl = ""
        if self._last_control is not None:
            steer, throttle, brake = self._last_control
            ctrl = (f" -> latest control steer={steer:+.3f} "
                    f"throttle={throttle:.3f} brake={brake:.3f}")
        published, self._ctrl_count = self._ctrl_count, 0
        period = f", period={m['period_ms']:.0f} ms" if m.get("period_ms") else ""
        self.get_logger().info(
            f"ORION inference: total={m['total_ms']:.1f} ms "
            f"(prep={m['prep_ms']:.1f}, forward={m['forward_ms']:.1f}{period}), "
            f"{m['num_pts']} waypoints, ~{1000.0 / max(m['total_ms'], 1e-3):.2f} FPS, "
            f"{published} controls published since the last one"
            f"{ctrl}"
        )
        if m.get("stages"):
            self.get_logger().info(
                "ORION stages (ms): "
                + orion_speedups.StageTimer.format(m["stages"], m["forward_ms"]))

    def destroy_node(self) -> None:
        if self._trace_fh is not None:
            try:
                self._trace_fh.close()
            except Exception:
                pass
            self._trace_fh = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._prep_executor.shutdown(wait=False, cancel_futures=True)
        self._decoder.shutdown()
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    os.environ.setdefault(
        "RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}")
    rclpy.init(args=args)
    node = OrionWithPidRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
