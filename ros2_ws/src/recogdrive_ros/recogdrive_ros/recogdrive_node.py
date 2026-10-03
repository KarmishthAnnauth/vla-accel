#!/usr/bin/env python3

"""ROS 2 node that streams CARLA sensor topics into the ReCogDrive VLA model
(xiaomi-research/recogdrive, ICLR'26) and publishes the trajectory it plans.

It is the ReCogDrive counterpart of ``orion_ros/orion_node.py`` and
``minddrive_ros/minddrive_node.py`` and keeps their structure (latest-only
buffers, subscriptions created after the model load, one inference in flight
dispatched on a fresh front frame, the Bench2Drive route-planner port, the
per-route reset).

Design goal: inference must run EXACTLY as the source code runs it.  The
reference is the NAVSIM evaluation,
``scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh`` ->
``navsim/planning/script/run_pdm_score_recogdrive.py``:

    agent = instantiate(cfg.agent)          # ReCogDriveAgent(cache_hidden_state=False,
                                            #   vlm_type='internvl', dit_type='small',
                                            #   sampling_method='ddim', grpo=False, ...)
    agent.initialize()                      # loads the planner checkpoint
    agent_input = scene_loader.get_agent_input_from_token(token)
    trajectory = agent.compute_trajectory(agent_input)

This node makes those same three calls on the same class, unmodified.  Nothing
of the model is re-implemented here: the InternVL image tiling and
normalisation (``load_image``), the system message, the prompt with the
history and the navigation command, the 2800-token left padding, the bf16
flash-attention backbone, the last hidden state handed to the DiT planner, the
5-step DDIM sampling and the de-normalisation all run inside
``compute_trajectory``.

    front image + odometry + imu + route
        -> AgentInput (agent_inputs.build_agent_input: NAVSIM's own pose
           conversion and dtypes; the image as a file path, as the eval's
           ``load_image_path=True`` does)
        -> ReCogDriveAgent.compute_trajectory(agent_input)
        -> Trajectory.poses  (8, 3): x, y, heading in the ego frame,
                                     0.5 s apart, 4 s horizon

Only the raw input sources differ from NAVSIM.  They are the
carla-simulator/ros-bridge topics the other nodes already consume:
  * ``sensor_msgs/Image`` front camera  -> written to tmpfs, path -> cam_f0
  * ``nav_msgs/Odometry``               -> the 4 x 0.5 s ego history and the
                                           ego-frame velocity
  * ``sensor_msgs/Imu``                 -> the ego-frame acceleration
  * ``carla_msgs/CarlaRoute`` (latched) -> the navigation command

Output.  The ReCogDrive release contains no closed-loop controller (NAVSIM is
open loop; the Bench2Drive evaluation code is not published), so unlike the
MindDrive and ORION-with-PID nodes there is no reference PID to reproduce and
none is invented here.  The plan is published as
``autoware_planning_msgs/Trajectory`` in the ego frame, exactly the message
``orion_node`` / ``simlingo_node`` publish, for ``stanley_controller_node`` to
track, plus a ``nav_msgs/Path`` of the same plan in the map frame.

The diffusion planner samples: ``get_action`` starts from Gaussian noise and
the eval never seeds it, so two forwards on identical inputs differ slightly.
That is the reference behaviour and is left alone (``seed`` >= 0 seeds torch
once at start-up for a repeatable run).
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaRoute
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from recogdrive_ros.agent_inputs import (
    COMMAND_NAMES,
    NAVSIM_INTERVAL_SEC,
    NUM_HISTORY_FRAMES,
    NUM_TRAJECTORY_POSES,
    TRAJECTORY_HORIZON_SEC,
    PoseHistory,
    build_agent_input,
    command_one_hot,
    road_option_to_command,
    segment_speeds,
    shift_along_heading,
    trajectory_to_world,
    world_to_frame,
)
from recogdrive_ros.pid_control import (
    B2D_SPEED_CAP_MPS, compute_control, load_pid_controller, rebase_plan)
from recogdrive_ros.camera_input import (
    FRAME_FORMATS, FrontCameraBuffer, decode_image, frame_path, write_frame)

ROUTE_MIN_DIST = 4.0
ROUTE_MAX_DIST = 50.0


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class RecogDriveRosNode(Node):
    """ReCogDrive inference on live CARLA streams, publishing the planned
    trajectory."""

    def __init__(self) -> None:
        super().__init__("recogdrive_node")

        self.declare_parameter("recogdrive_repo_path", "/benchmarking/recogdrive")
        self.declare_parameter("vlm_path", "/models/ReCogDrive/ReCogDrive-VLM-2B")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("vlm_type", "internvl")
        self.declare_parameter("vlm_size", "small")
        self.declare_parameter("dit_type", "small")
        self.declare_parameter("sampling_method", "ddim")
        self.declare_parameter("cam_type", "single")
        self.declare_parameter("allow_partial_checkpoint", False)
        self.declare_parameter("seed", -1)
        self.declare_parameter("fast_inference", True)
        self.declare_parameter("profile", True)
        self.declare_parameter("vit_engine", "")

        self.declare_parameter("front_camera_topic", "/carla/hero/CAM_FRONT/image")
        self.declare_parameter("odometry_topic", "/carla/hero/odometry")
        self.declare_parameter("imu_topic", "/carla/hero/imu")
        self.declare_parameter("route_topic", "/carla/hero/global_plan")
        self.declare_parameter("trajectory_topic", "/recogdrive/predicted_trajectory")
        self.declare_parameter("path_topic", "/recogdrive/predicted_path")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("map_frame_id", "map")

        self.declare_parameter("inference_period_sec", 0.05)
        self.declare_parameter("require_new_frame", True)
        self.declare_parameter("driving_command", 4)
        self.declare_parameter("frame_dir", "/dev/shm/recogdrive_ros")
        self.declare_parameter("frame_format", "bmp")
        self.declare_parameter("frame_jpeg_quality", 95)
        self.declare_parameter("ego_reference_offset_x", 0.0)
        self.declare_parameter("require_full_history", False)

        self.declare_parameter("rebase_to_latest_pose", True)
        self.declare_parameter("min_trajectory_speed_mps", 0.0)

        self.declare_parameter("control_mode", "pid")
        self.declare_parameter("control_topic", "/carla/hero/vehicle_control_cmd")
        self.declare_parameter("pid_controller_dir", "/benchmarking/Orion/team_code")
        self.declare_parameter("speed_topic", "/carla/hero/speed")
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("plan_stall_timeout_sec", 6.0)
        self.declare_parameter("brake_when_stale", True)
        self.declare_parameter("speed_cap_mps", B2D_SPEED_CAP_MPS)

        self._inference_period = float(self.get_parameter("inference_period_sec").value)
        self._require_new_frame = bool(self.get_parameter("require_new_frame").value)
        self._driving_command = int(self.get_parameter("driving_command").value)
        self._frame_dir = str(self.get_parameter("frame_dir").value)
        self._frame_format = str(self.get_parameter("frame_format").value or "bmp").lower()
        if self._frame_format not in FRAME_FORMATS:
            raise ValueError(f"frame_format must be one of {FRAME_FORMATS}")
        self._frame_jpeg_quality = int(self.get_parameter("frame_jpeg_quality").value)
        self._ref_offset_x = float(self.get_parameter("ego_reference_offset_x").value)
        self._require_full_history = bool(self.get_parameter("require_full_history").value)
        self._rebase = bool(self.get_parameter("rebase_to_latest_pose").value)
        self._min_traj_speed = float(self.get_parameter("min_trajectory_speed_mps").value)
        self._control_mode = str(self.get_parameter("control_mode").value or "pid").lower()
        if self._control_mode not in ("pid", "stanley"):
            raise ValueError("control_mode must be 'pid' or 'stanley'")
        self._control_hz = float(self.get_parameter("control_hz").value)
        self._plan_stall_timeout = float(self.get_parameter("plan_stall_timeout_sec").value)
        self._brake_when_stale = bool(self.get_parameter("brake_when_stale").value)
        self._speed_cap = float(self.get_parameter("speed_cap_mps").value)
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._map_frame = str(self.get_parameter("map_frame_id").value)
        if self._ref_offset_x:
            self.get_logger().info(
                f"ego localised {self._ref_offset_x:+.2f} m (vehicle frame) from the "
                f"odometry origin (NAVSIM's ego pose is the rear axle)")

        traj_topic = str(self.get_parameter("trajectory_topic").value)
        self._trajectory_pub = None
        try:
            from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
            self._Trajectory, self._TrajectoryPoint = Trajectory, TrajectoryPoint
            self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, 10)
            self.get_logger().info(f"Publishing autoware_planning_msgs/Trajectory on {traj_topic}")
        except ImportError:
            self.get_logger().warn(
                "autoware_planning_msgs is not available: only the nav_msgs/Path is published")
        path_topic = str(self.get_parameter("path_topic").value)
        self._path_pub = self.create_publisher(Path, path_topic, 10)
        self.get_logger().info(f"Publishing nav_msgs/Path ({self._map_frame} frame) on {path_topic}")

        self._control_pub = None
        self._PIDController = None
        if self._control_mode == "pid":
            self._PIDController = load_pid_controller(
                str(self.get_parameter("pid_controller_dir").value))
            control_topic = str(self.get_parameter("control_topic").value)
            self._control_pub = self.create_publisher(CarlaEgoVehicleControl, control_topic, 10)
            self.get_logger().info(
                f"control_mode=pid: publishing CarlaEgoVehicleControl on {control_topic} at "
                f"{self._control_hz:g} Hz (Bench2Drive PID"
                f"{f', throttle cut above {self._speed_cap:g} m/s' if self._speed_cap > 0 else ''}). "
                f"carla_ackermann_control / stanley_controller_node must NOT be running.")
        else:
            self.get_logger().info(
                "control_mode=stanley: no control from this node; stanley_controller_node "
                "tracks the trajectory (needs carla_ackermann_control on the CARLA side)")
        self._pid = self._PIDController() if self._PIDController else None
        self._plan_state: Optional[tuple] = None
        self._latest_speed: Optional[float] = None
        self._odom_speed: float = 0.0
        self._last_control: Optional[tuple] = None
        self._warned_odom_speed = False

        self._cam: Optional[FrontCameraBuffer] = None
        self._history = PoseHistory(max_age_sec=6.0)
        self._latest_origin_pose: Optional[np.ndarray] = None
        self._latest_odom_mono: float = 0.0
        self._latest_accel: Optional[np.ndarray] = None
        self._route: deque = deque()
        self._route_serial: int = 0
        self._route_fingerprint = None
        self._route_change_mono: float = 0.0
        self._last_ref_stamp: Optional[tuple] = None
        self._last_skip_log_t: float = 0.0
        self._last_period_mark: Optional[float] = None
        self._last_clock_warn_t: float = 0.0
        self._last_twist_warn_t: float = 0.0
        self._warned_no_imu = False
        self._frame_count: int = 0

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None
        self._agent = None
        self._fast = None
        self._stages: Optional[dict] = None
        self._reference_frames: int = 0

        self.get_logger().info("Loading ReCogDrive model ...")
        self._executor.submit(self._setup_model).result()

        self._cam = FrontCameraBuffer(
            self, str(self.get_parameter("front_camera_topic").value), img_qos)
        odom_topic = self.get_parameter("odometry_topic").value
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        imu_topic = self.get_parameter("imu_topic").value
        self.create_subscription(Imu, imu_topic, self._imu_callback, odom_qos)
        route_topic = self.get_parameter("route_topic").value
        self.create_subscription(CarlaRoute, route_topic, self._route_callback, route_qos)
        self.get_logger().info(
            f"Subscribed: odometry {odom_topic}, imu {imu_topic}, route {route_topic}")

        if self._control_mode == "pid":
            speed_topic = str(self.get_parameter("speed_topic").value)
            self.create_subscription(Float32, speed_topic, self._speed_callback, 10)
            self.get_logger().info(f"Subscribed: speed {speed_topic}")
            self._control_timer = self.create_timer(1.0 / self._control_hz, self._control_callback)

        self._timer = self.create_timer(self._inference_period, self._timer_callback)

    def _setup_model(self) -> None:
        repo_path = str(self.get_parameter("recogdrive_repo_path").value or "")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        seed = int(self.get_parameter("seed").value)
        if seed >= 0:
            torch.manual_seed(seed)
            self.get_logger().info(f"torch seeded with {seed} (the reference eval does not seed)")

        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        import navsim
        from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
        self.get_logger().info(f"navsim: {os.path.dirname(os.path.abspath(navsim.__file__))}")

        vlm_path = str(self.get_parameter("vlm_path").value or "")
        ckpt_path = str(self.get_parameter("checkpoint_path").value or "")
        vlm_type = str(self.get_parameter("vlm_type").value)
        vlm_size = str(self.get_parameter("vlm_size").value)
        dit_type = str(self.get_parameter("dit_type").value)
        sampling_method = str(self.get_parameter("sampling_method").value)
        cam_type = str(self.get_parameter("cam_type").value)
        if not os.path.isfile(os.path.join(vlm_path, "config.json")):
            raise FileNotFoundError(f"vlm_path has no config.json: {vlm_path}")
        if not ckpt_path:
            raise ValueError(
                "checkpoint_path is empty: ReCogDriveAgent would run its diffusion "
                "planner from random weights. Pass the planner .ckpt.")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint_path does not exist: {ckpt_path}")
        self.get_logger().info(
            f"ReCogDriveAgent(vlm_type={vlm_type}, vlm_size={vlm_size}, dit_type={dit_type}, "
            f"sampling_method={sampling_method}, cam_type={cam_type}, cache_hidden_state=False, "
            f"grpo=False); vlm {vlm_path}; checkpoint {ckpt_path}")

        t0 = time.time()
        agent = ReCogDriveAgent(
            trajectory_sampling=TrajectorySampling(
                time_horizon=int(TRAJECTORY_HORIZON_SEC), interval_length=NAVSIM_INTERVAL_SEC),
            vlm_path=vlm_path,
            checkpoint_path=ckpt_path,
            cam_type=cam_type,
            vlm_type=vlm_type,
            dit_type=dit_type,
            sampling_method=sampling_method,
            cache_mode=False,
            cache_hidden_state=False,
            grpo=False,
            vlm_size=vlm_size,
        )
        self.get_logger().info(f"agent built in {time.time() - t0:.0f} s; loading checkpoint")

        t1 = time.time()
        seen = {}
        original_load = agent.load_state_dict

        def _observed_load(state_dict, *args, **kwargs):
            seen["loaded"] = set(state_dict.keys())
            return original_load(state_dict, *args, **kwargs)

        agent.load_state_dict = _observed_load
        try:
            agent.initialize()
        finally:
            del agent.load_state_dict
        self._report_checkpoint_coverage(agent, seen.get("loaded"))
        self.get_logger().info(f"checkpoint loaded in {time.time() - t1:.0f} s")

        self._agent = agent
        self.get_logger().info(
            f"model on GPU ({torch.cuda.memory_allocated() / 1e9:.1f} GB allocated) "
            f"after {time.time() - t0:.0f} s; backbone dtype "
            f"{next(agent.backbone.parameters()).dtype}, planner dtype "
            f"{next(agent.action_head.parameters()).dtype}")

        if bool(self.get_parameter("fast_inference").value):
            self._setup_fast_inference()
        else:
            self.get_logger().info("fast_inference:=false -- reference path (ReCogDriveAgent.compute_trajectory)")
        self._warmup()
        self.get_logger().info("ReCogDrive model loaded and ready.")

    def _setup_fast_inference(self) -> None:
        """Build the fast path around the agent; on any failure keep the
        reference path."""
        try:
            from recogdrive_ros.recogdrive_speedups import FastReCogDrive
            t0 = time.time()
            self._fast = FastReCogDrive(
                self._agent, log=self.get_logger().info,
                profile=bool(self.get_parameter("profile").value),
                vit_engine=str(self.get_parameter("vit_engine").value or ""))
            self.get_logger().info(
                f"fast inference path built in {time.time() - t0:.0f} s "
                f"({torch.cuda.memory_allocated() / 1e9:.1f} GB allocated)")
        except Exception as e:
            import traceback
            self._fast = None
            self.get_logger().error(
                f"fast inference path unavailable ({type(e).__name__}: {e}); using the "
                f"reference path\n{traceback.format_exc()}")

    def _report_checkpoint_coverage(self, agent, loaded: Optional[set]) -> None:
        """Say what initialize() put into the planner, and stop if learned
        planner weights were left at their random initialisation."""
        if loaded is None:
            raise RuntimeError("ReCogDriveAgent.initialize() loaded nothing (no checkpoint_path?)")
        head_params = {f"action_head.{n}" for n, _ in agent.action_head.named_parameters()}
        head_buffers = {f"action_head.{n}" for n, _ in agent.action_head.named_buffers()}
        missing_params = sorted(head_params - loaded)
        missing_buffers = sorted(head_buffers - loaded)
        backbone_loaded = sum(1 for k in loaded if k.startswith("backbone."))
        self.get_logger().info(
            f"checkpoint -> agent: {len(loaded)} tensors loaded "
            f"({len(head_params & loaded)}/{len(head_params)} planner parameters, "
            f"{backbone_loaded} backbone tensors"
            f"{' -- the backbone keeps the vlm_path weights' if not backbone_loaded else ''})")
        if missing_buffers:
            self.get_logger().info(
                f"{len(missing_buffers)} planner buffers not in the checkpoint (rebuilt by the "
                f"constructor): {missing_buffers[:4]}{' ...' if len(missing_buffers) > 4 else ''}")
        if missing_params:
            msg = (f"{len(missing_params)} of {len(head_params)} planner parameters were NOT "
                   f"loaded (name or shape mismatch; initialize() skips them silently), e.g. "
                   f"{missing_params[:4]}. Wrong checkpoint for vlm_size / dit_type / "
                   f"sampling_method?")
            if not bool(self.get_parameter("allow_partial_checkpoint").value):
                raise RuntimeError(msg + " Set allow_partial_checkpoint:=true to run anyway.")
            self.get_logger().error(msg + " Running anyway (allow_partial_checkpoint).")

    def _warmup(self) -> None:
        """One full compute_trajectory on a black 1920x1080 frame (NAVSIM's
        cam_f0 size) so the first real inference is not a cold-start outlier.
        Best-effort."""
        try:
            t0 = time.time()
            black = np.zeros((1080, 1920, 3), dtype=np.uint8)
            path = write_frame(black, self._frame_dir, self._frame_format,
                               self._frame_jpeg_quality, name="warmup")
            agent_input = build_agent_input(
                np.zeros((NUM_HISTORY_FRAMES, 3)), np.zeros((NUM_HISTORY_FRAMES, 2)),
                np.zeros(2), command_one_hot(road_option_to_command(4)), path)
            if self._fast is not None:
                self._verify_fast(agent_input)
            poses = self._plan(agent_input)
            os.remove(path)
            self.get_logger().info(
                f"Model warmup OK in {(time.time() - t0) * 1e3:.0f} ms: trajectory "
                f"{tuple(poses.shape)}, GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        except Exception as e:
            import traceback
            self.get_logger().warn(f"Model warmup skipped ({type(e).__name__}: {e})\n"
                                   f"{traceback.format_exc()}")

    def _verify_fast(self, agent_input) -> None:
        """Reference and fast path on one input with the same noise; the fast
        path is dropped if they disagree."""
        report = self._fast.verify(agent_input)
        if not report.get("covered"):
            self.get_logger().warn("fast path does not cover the warm-up frame; not verified")
            return
        hidden, traj = report["hidden_max_abs"], report["trajectory_max_abs"]
        self.get_logger().info(
            f"fast path vs reference, same noise: VLM hidden states max |diff| {hidden:g}"
            f"{' (bit-identical)' if hidden == 0.0 else ''}, trajectory max |diff| {traj:.2e} (m, rad)")
        if not traj < 0.5:
            self.get_logger().error(
                "fast path disagrees with the reference; using the reference path")
            self._fast = None
        elif hidden != 0.0 and self._fast.exact:
            self.get_logger().warn(
                "fast path is not bit-identical on this GPU / library build (a bf16 "
                "rounding difference, see recogdrive_speedups); the trajectory agrees")

    def _plan(self, agent_input, image_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """The (8, 3) poses as numpy: the fast path if it is on and covers this
        input, else ReCogDriveAgent.compute_trajectory.

        image_rgb: the frame agent_input names when it has NOT been written to
        that file (the fast path takes the array; the reference needs the
        file, so it is written if the reference ends up running)."""
        trajectory = None
        self._stages = None
        if self._fast is not None:
            trajectory = self._fast.plan(agent_input, image=image_rgb)
            if trajectory is not None:
                self._stages = dict(self._fast.timing)
            else:
                self._reference_frames += 1
                if self._reference_frames == 1:
                    self.get_logger().warn(
                        "frame outside the fast path (tile grid or prompt length): such "
                        "frames use the reference path")
        if trajectory is None:
            if image_rgb is not None:
                write_frame(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), self._frame_dir,
                            self._frame_format, self._frame_jpeg_quality)
            trajectory = self._agent.compute_trajectory(agent_input)
        torch.cuda.synchronize()
        poses = trajectory.poses
        if isinstance(poses, torch.Tensor):
            poses = poses.numpy()
        poses = np.asarray(poses, dtype=np.float64)
        if poses.shape != (NUM_TRAJECTORY_POSES, 3):
            raise RuntimeError(f"unexpected trajectory shape {poses.shape}")
        return poses

    def _odometry_callback(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        yaw = _yaw_from_quaternion(pose.orientation)
        origin = np.array([pose.position.x, pose.position.y, yaw], dtype=np.float64)
        lin = msg.twist.twist.linear
        vx, vy = float(lin.x), float(lin.y)
        ref = origin
        if self._ref_offset_x:
            ref = shift_along_heading(origin, self._ref_offset_x)
            vy += self._ref_offset_x * float(msg.twist.twist.angular.z)
        if self._history.add(_stamp_sec(msg.header.stamp), ref[0], ref[1], ref[2], vx, vy):
            self.get_logger().warn(
                "odometry time ran backwards (simulation restarted?): ego history cleared")
        self._odom_speed = math.hypot(vx, float(lin.y))
        self._latest_origin_pose = origin
        self._latest_odom_mono = time.monotonic()

    def _speed_callback(self, msg: Float32) -> None:
        self._latest_speed = float(msg.data)

    def _imu_callback(self, msg: Imu) -> None:
        self._latest_accel = np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y], dtype=np.float64)

    def _route_callback(self, msg: CarlaRoute) -> None:
        n = min(len(msg.poses), len(msg.road_options))
        route = deque()
        for i in range(n):
            p = msg.poses[i]
            route.append((np.array([p.position.x, p.position.y], dtype=np.float64),
                          int(msg.road_options[i])))
        self._route = route

        fp = (len(route),
              tuple(route[0][0]) if route else (),
              tuple(route[-1][0]) if route else ())
        if fp != self._route_fingerprint:
            self._route_fingerprint = fp
            self._route_serial += 1
            self._history.clear()
            self._last_ref_stamp = None
            self._route_change_mono = time.monotonic()
            self._plan_state = None
            if self._PIDController:
                self._pid = self._PIDController()
            self.get_logger().info(
                f"Route received: {len(route)} waypoints -- NEW route "
                f"(#{self._route_serial}); ego history cleared.")
        else:
            self.get_logger().info(
                f"Route received: {len(route)} waypoints -- same plan as route "
                f"#{self._route_serial}, keeping context.")

    def _log_dispatch_state(self, reason: str, interval_sec: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_skip_log_t < interval_sec:
            return
        self._last_skip_log_t = now
        self.get_logger().info(f"[dispatch] {reason}")

    def _timer_callback(self) -> None:
        if self._active_future is not None and not self._active_future.done():
            self._log_dispatch_state("busy: inference still in flight")
            return
        snapshot = self._take_snapshot()
        if snapshot is None:
            return
        self._active_future = self._executor.submit(self._run_inference, snapshot)
        self._active_future.add_done_callback(self._on_future_done)

    def _take_snapshot(self) -> Optional[dict]:
        if self._cam is None or not self._cam.has_frame():
            self._log_dispatch_state(
                f"waiting: no frame yet from {self._cam.topic if self._cam else '(no subs)'}")
            return None
        if self._latest_origin_pose is None or len(self._history) == 0:
            self._log_dispatch_state("waiting: no odometry received yet")
            return None
        if self._latest_odom_mono < self._route_change_mono:
            self._log_dispatch_state("waiting: no odometry since the new route arrived")
            return None
        if self._cam.received_at is None or self._cam.received_at < self._route_change_mono:
            self._log_dispatch_state("waiting: camera frame from before the route change")
            return None
        record = self._cam.record()
        ref_stamp = (record[0].sec, record[0].nanosec)
        if self._require_new_frame and ref_stamp == self._last_ref_stamp:
            waited = time.monotonic() - (self._cam.received_at or 0.0)
            if waited > 2.0:
                self._log_dispatch_state(
                    f"skipped: no new front frame for {waited:.1f} s "
                    f"(stamp held at {ref_stamp[0]}.{ref_stamp[1]:09d}) -- camera stream stalled?")
            return None

        t_img = _stamp_sec(record[0])
        newest = self._history.newest_time
        t_ref = min(t_img, newest)
        if abs(t_img - newest) > 1.0:
            t_ref = newest
            now = time.monotonic()
            if now - self._last_clock_warn_t > 10.0:
                self._last_clock_warn_t = now
                self.get_logger().warn(
                    f"front image stamp {t_img:.3f} and newest odometry stamp {newest:.3f} are "
                    f"{abs(t_img - newest):.2f} s apart: not one clock, or one stream lags. "
                    f"Using the newest odometry sample as the image's pose.")
        poses, velocities, held = self._history.window(t_ref)
        if held and self._require_full_history:
            self._log_dispatch_state(
                f"waiting: {NAVSIM_INTERVAL_SEC * (NUM_HISTORY_FRAMES - 1):.1f} s of ego "
                f"history needed, {t_ref - self._history.oldest_time:.2f} s buffered")
            return None
        self._last_ref_stamp = ref_stamp
        self._check_twist_frame(t_ref, velocities[-1])

        road_option = self._compute_driving_command(self._latest_origin_pose[:2])
        return {
            "record": record,
            "poses": poses,
            "velocities": velocities,
            "acceleration": self._acceleration(t_ref),
            "road_option": road_option,
            "command": road_option_to_command(road_option),
            "held": held,
            "t_dispatch": time.time(),
            "route_serial": self._route_serial,
        }

    def _acceleration(self, t_ref: float) -> np.ndarray:
        """Ego-frame (ax, ay): the IMU when there is one, else the change of
        the odometry velocity over the last 0.2 s."""
        if self._latest_accel is not None:
            return self._latest_accel.copy()
        if not self._warned_no_imu:
            self._warned_no_imu = True
            self.get_logger().warn(
                "no IMU message yet: acceleration is differentiated from the odometry twist")
        span = 0.2
        oldest = self._history.oldest_time
        if oldest is None or t_ref - span < oldest:
            return np.zeros(2, dtype=np.float64)
        (p0, v0), (p1, v1) = self._history.sample(t_ref - span), self._history.sample(t_ref)

        def to_world(pose, vel):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            return np.array([c * vel[0] - s * vel[1], s * vel[0] + c * vel[1]])

        a_world = (to_world(p1, v1) - to_world(p0, v0)) / span
        c, s = math.cos(p1[2]), math.sin(p1[2])
        return np.array([c * a_world[0] + s * a_world[1], -s * a_world[0] + c * a_world[1]])

    def _check_twist_frame(self, t_ref: float, velocity: np.ndarray) -> None:
        """The model is told (vx, vy) in the ego frame.  If the odometry twist
        were in the map frame instead the numbers would be wrong with nothing
        to show for it, so compare against the velocity the poses imply."""
        implied = self._history.finite_difference_velocity(t_ref)
        if implied is None or np.linalg.norm(implied - velocity) < 1.5:
            return
        now = time.monotonic()
        if now - self._last_twist_warn_t < 10.0:
            return
        self._last_twist_warn_t = now
        self.get_logger().warn(
            f"odometry twist ({velocity[0]:+.2f}, {velocity[1]:+.2f}) m/s disagrees with the "
            f"velocity its poses imply in the ego frame ({implied[0]:+.2f}, {implied[1]:+.2f}): "
            f"is the twist expressed in the vehicle frame?")

    def _compute_driving_command(self, ego_xy: np.ndarray) -> int:
        """Faithful port of Bench2Drive RoutePlanner.run_step (team_code/planner.py),
        as in the ORION / MindDrive nodes. Returns a CARLA RoadOption."""
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
        """CARLA's VOID (-1) and anything else out of range -> the fallback."""
        cmd = int(cmd)
        return cmd if 1 <= cmd <= 6 else self._driving_command

    def _run_inference(self, snapshot: dict):
        t_start = time.time()
        from_array = self._fast is not None and self._frame_format != "jpg"
        image = decode_image(snapshot["record"], rgb=from_array)
        if from_array:
            image_path = frame_path(self._frame_dir, self._frame_format)
        else:
            image_path = write_frame(image, self._frame_dir, self._frame_format,
                                     self._frame_jpeg_quality)
        agent_input = build_agent_input(
            snapshot["poses"], snapshot["velocities"], snapshot["acceleration"],
            command_one_hot(snapshot["command"]), image_path)
        t_prep = time.time()

        poses = self._plan(agent_input, image if from_array else None)
        t_forward = time.time()

        if snapshot["route_serial"] != self._route_serial:
            self.get_logger().info("plan discarded: route changed during the forward")
            return None

        self._plan_state = (poses[:, :2].copy(), np.array(snapshot["poses"][-1], dtype=np.float64),
                            float(snapshot["t_dispatch"]), time.time())
        self._publish(poses, snapshot)
        self._frame_count += 1
        period_ms = (t_start - self._last_period_mark) * 1e3 if self._last_period_mark else None
        self._last_period_mark = t_start
        return {
            "prep_ms": (t_prep - t_start) * 1e3,
            "forward_ms": (t_forward - t_prep) * 1e3,
            "total_ms": (t_forward - t_start) * 1e3,
            "period_ms": period_ms,
            "stages": self._stages,
            "path": "reference" if self._stages is None and self._fast is None else
                    ("reference (frame outside the fast path)" if self._stages is None else "fast"),
            "image": f"{image.shape[1]}x{image.shape[0]}",
            "command": COMMAND_NAMES[snapshot["command"]],
            "road_option": snapshot["road_option"],
            "held": snapshot["held"],
            "speed": float(np.linalg.norm(snapshot["velocities"][-1])),
            "control": self._last_control,
            "poses": poses,
        }

    def _control_callback(self) -> None:
        """Track the latest plan with the Bench2Drive PID.  Runs at control_hz,
        including while a forward pass is in flight."""
        state = self._plan_state
        origin = self._latest_origin_pose
        if state is None or origin is None:
            return
        plan_xy, plan_frame, t_seen, t_arrived = state
        now = time.time()
        stalled = now - t_arrived
        if stalled > self._plan_stall_timeout:
            self._stop_the_car(f"no new plan for {stalled:.1f} s (limit "
                               f"{self._plan_stall_timeout:g} s) -- the model has stopped answering")
            return
        ego_now = shift_along_heading(origin, self._ref_offset_x)
        waypoints = rebase_plan(plan_xy, now - t_seen, plan_frame, ego_now)

        target = None
        if self._route:
            node = self._route[1][0] if len(self._route) > 1 else self._route[0][0]
            target = world_to_frame(np.array([[node[0], node[1], 0.0]]), ego_now)[0, :2]
        speed = self._latest_speed
        if speed is None:
            speed = self._odom_speed
            if not self._warned_odom_speed:
                self._warned_odom_speed = True
                self.get_logger().warn(
                    "no speedometer message yet: using the odometry twist's magnitude as speed")
        steer, throttle, brake, _ = compute_control(
            self._pid, waypoints, speed, target, self._speed_cap)

        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.steer, cmd.throttle, cmd.brake = steer, throttle, brake
        self._control_pub.publish(cmd)
        self._last_control = (steer, throttle, brake)

    def _stop_the_car(self, why: str) -> None:
        if not self._brake_when_stale:
            return
        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.steer, cmd.throttle, cmd.brake = 0.0, 0.0, 1.0
        self._control_pub.publish(cmd)
        self._last_control = (0.0, 0.0, 1.0)
        self.get_logger().warn(f"braking: {why}", throttle_duration_sec=5.0)

    def _publish(self, poses: np.ndarray, snapshot: dict) -> None:
        stamp = snapshot["record"][0]
        plan_frame = snapshot["poses"][-1]
        world = trajectory_to_world(np.vstack((np.zeros((1, 3)), poses)), plan_frame)

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self._map_frame
        for x, y, yaw in world:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(ps)
        self._path_pub.publish(path)

        if self._trajectory_pub is None:
            return
        origin_then = shift_along_heading(plan_frame, -self._ref_offset_x)
        latest = self._latest_origin_pose
        frame = latest if (self._rebase and latest is not None) else origin_then
        local = world_to_frame(world, frame)
        speeds = segment_speeds(poses[:, :2], np.zeros(2))

        msg = self._Trajectory()
        msg.header.stamp = stamp
        msg.header.frame_id = self._base_frame
        for i, (x, y, yaw) in enumerate(local):
            tp = self._TrajectoryPoint()
            tp.pose.position.x, tp.pose.position.y = float(x), float(y)
            tp.pose.orientation.z = math.sin(yaw / 2.0)
            tp.pose.orientation.w = math.cos(yaw / 2.0)
            speed = speeds[max(i - 1, 0)]
            tp.longitudinal_velocity_mps = float(max(speed, self._min_traj_speed))
            t = i * NAVSIM_INTERVAL_SEC
            tp.time_from_start = Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
            msg.points.append(tp)
        self._trajectory_pub.publish(msg)

    def _on_future_done(self, future: Future) -> None:
        try:
            m = future.result()
        except Exception as exc:
            import traceback
            self.get_logger().error(
                f"ReCogDrive inference failed: {exc}\n{traceback.format_exc()}")
            return
        if not m:
            return
        poses = m["poses"]
        period = f", period={m['period_ms']:.0f} ms" if m.get("period_ms") else ""
        held = f", history: {m['held']} of {NUM_HISTORY_FRAMES} frames held" if m["held"] else ""
        c = m.get("control")
        ctrl = f"; control steer={c[0]:+.3f} throttle={c[1]:.2f} brake={c[2]:.2f}" if c else ""
        stages = m.get("stages")
        stages = (" [" + " ".join(f"{k}={v:.0f}" for k, v in stages.items()) + "]") if stages else ""
        self.get_logger().info(
            f"ReCogDrive inference: total={m['total_ms']:.1f} ms "
            f"(prep={m['prep_ms']:.1f}, forward={m['forward_ms']:.1f}{stages}{period}, {m['path']}), "
            f"image {m['image']}, cmd={m['command']} (road option {m['road_option']}), "
            f"v={m['speed']:.2f} m/s{held}; plan wp0=({poses[0, 0]:+.2f}, {poses[0, 1]:+.2f}) "
            f"end=({poses[-1, 0]:+.2f}, {poses[-1, 1]:+.2f}, {math.degrees(poses[-1, 2]):+.1f} deg){ctrl}")

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    os.environ.setdefault("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}")
    rclpy.init(args=args)
    node = RecogDriveRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
