#!/usr/bin/env python3

"""ROS 2 node that streams CARLA sensor topics into the MindDrive VLA model and
closes the loop in-node: the model's speed and path trajectories are fed to the
Bench2Drive decision-expert PID (``team_code/pid_controller_de.py``), which
publishes ``carla_msgs/CarlaEgoVehicleControl`` on
``/carla/hero/vehicle_control_cmd`` -- exactly what the leaderboard agent's
``carla.VehicleControl`` is on the CARLA side.

It is the MindDrive counterpart of ``orion_ros/orion_withpid_node.py`` and keeps
that node's structure (six-camera buffer, route-planner port, per-route context
reset, 20 Hz control loop over a re-based plan, prep pipelining).

Inference speedups (``minddrive_speedups.py``) are OFF unless asked for: with
every speedup parameter at its default the model object that runs is the one
``build_model`` + ``load_checkpoint`` produce, untouched, in the precision the
config asks for (fp32).  Each speedup was measured for time and for its effect
on the trajectories and the decision expert's choice by
``tools/bench_minddrive.py`` (see MINDDRIVE_ROS_NODE.md §6) before it was
exposed here; ``start_minddrive.sh --fast`` turns on the validated set.

Design goal: the payload handed to MindDrive must be prepared in EXACTLY the
same way as the reference closed-loop agent
(``MindDrive/team_code/minddrive_b2d_agent.py::MinddriveAgent.run_step``).  So
this node reuses MindDrive's own code and never reimplements preprocessing:

  raw 6-cam images + odometry
      -> the same ``results`` dict the agent builds   (agent_constants.build_agent_results)
      -> ``Compose(cfg.inference_only_pipeline)``     (resize/crop, normalize, pad,
                                                        VQA prompt + tokenization)
      -> ``mmcv.parallel.collate``                    (mm_collate_to_batch_form)
      -> the agent's H2D loop                         (agent_constants.batch_to_device)
      -> ``custom_wrap_fp16_model(model)``            (before every forward, as the agent)
      -> ``model(batch, return_loss=False)``
      -> out[0]['pts_bbox']['ego_fut_preds']      (6, 2)   speed trajectory, 0.5 s steps
         out[0]['pts_bbox']['pw_ego_fut_pred']    (20, 2)  path trajectory (spatial)
         out[0]['pts_bbox']['speed_value' / 'path_value']  the decision expert's
                                                   discrete meta-actions (logged)

Only the raw input sources differ from the CARLA agent.  They are the
carla-simulator/ros-bridge topics the ORION and SimLingo nodes already consume:
  * 6x ``sensor_msgs/Image`` (bgr8 from image_decompress_node, or raw
    rgba8/bgra8), one per camera in CAMERA_ORDER -> decoded to BGR into its own
    slot, matching MinddriveAgent.tick().  See minddrive_ros.camera_input.
  * ``std_msgs/Float32`` speed                          -> can_bus[7]
  * ``nav_msgs/Odometry``                               -> ego_pose / heading
  * ``sensor_msgs/Imu``                                 -> can_bus accel + angular
  * ``carla_msgs/CarlaRoute`` (latched)                 -> driving command + route reset

Precision.  The reference config sets ``fp32_infer=True`` and the agent passes it
through unmodified; ``precision:=config`` (default) does the same.
``precision:=fp16`` flips the config's ``fp16_infer`` flag instead: LLM weights
in fp16, ``img_backbone.half()``, the two perception heads fp32 -- exactly what
the flag does upstream, once the loader bug that made it unusable for Qwen2 is
patched (minddrive_env/0002-fp16-qwen-load.patch).

Per-route context.  The leaderboard builds a fresh agent per route; this node is
one long-lived process, so the boundary is recreated by hand (``_reset_for_new_route``):
a new CarlaRoute clears the model's ``test_flag`` (its own ``forward_test`` then
calls ``reset_memory()`` on both heads, the same path it takes for the very first
frame), restarts ``frame_idx`` at 0, rebuilds the PIDController and drops the
plan the control loop was tracking.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import rclpy
import torch
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaRoute
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String

from minddrive_ros.agent_constants import (
    AGENT_GNSS_MOUNT_X,
    AGENT_HZ,
    CAMERA_ORDER,
    FUT_PS,
    LIDAR2EGO,
    MEMORY_MAX_DT,
    ROUTE_MAX_DIST,
    ROUTE_MIN_DIST,
    SPEED_CAP_MPS,
    TRAJ_DT,
    batch_to_device,
    build_agent_results,
    custom_wrap_fp16_model,
)
from minddrive_ros.camera_input import (
    DEFAULT_CAMERA_TOPICS,
    REFERENCE_CAMERA,
    MultiCameraBuffer,
    decode_image,
)
from minddrive_ros import minddrive_speedups as sp

SPEED_ACTIONS = ["maintain moderate speed", "stop", "maintain slow speed", "speed up",
                 "slow down", "maintain fast speed", "slow down rapidly"]
PATH_ACTIONS = ["lanefollow", "straight", "turn left", "change lane left",
                "turn right", "change lane right"]


def _action_name(table: List[str], idx) -> str:
    try:
        return table[int(idx)]
    except (TypeError, ValueError, IndexError):
        return f"?{idx}"


class MinddriveRosNode(Node):
    """MindDrive inference on live CARLA streams, with the Bench2Drive
    decision-expert PID in-node publishing CarlaEgoVehicleControl directly."""

    def __init__(self) -> None:
        super().__init__("minddrive_node")

        self.declare_parameter("minddrive_repo_path", "/benchmarking/MindDrive")
        self.declare_parameter("minddrive_config_path",
                               "adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py")
        self.declare_parameter("minddrive_checkpoint_path",
                               "/models/MindDrive/minddrive_3b_rltrain.pth")
        self.declare_parameter("timestamp_mode", "sensor")
        self.declare_parameter("precision", "config")
        self.declare_parameter("merge_lora", False)
        self.declare_parameter("llm_attn", "keep")
        self.declare_parameter("vit_glue", False)
        self.declare_parameter("down_proj_t", False)
        self.declare_parameter("vit_weight_t", False)
        self.declare_parameter("vit_window_nopad", False)
        self.declare_parameter("int8_targets", "")
        self.declare_parameter("int8_skip", "")
        self.declare_parameter("int8_calib_path", "")
        self.declare_parameter("int8_alpha", 0.5)
        self.declare_parameter("vit_input_size", 640)
        self.declare_parameter("rear_view_refresh_every", 1)
        self.declare_parameter("map_head_slice", False)
        self.declare_parameter("compile_targets", "")
        self.declare_parameter("compile_mode", "default")
        self.declare_parameter("cuda_graph_vit", False)
        self.declare_parameter("cuda_graph_heads", False)
        self.declare_parameter("logits_slice", False)
        self.declare_parameter("overlap_experts", False)
        self.declare_parameter("pipeline_prep", False)
        self.declare_parameter("pipeline_margin_ms", 40.0)
        self.declare_parameter("profile_stages", False)

        self.declare_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        self.declare_parameter("camera_sync_tolerance_sec", 0.1)
        self.declare_parameter("speed_topic", "/carla/hero/speed")
        self.declare_parameter("odometry_topic", "/carla/hero/odometry")
        self.declare_parameter("imu_topic", "/carla/hero/imu")
        self.declare_parameter("route_topic", "/carla/hero/global_plan")
        self.declare_parameter("control_topic", "/carla/hero/vehicle_control_cmd")
        self.declare_parameter("meta_action_topic", "/minddrive/meta_action")

        self.declare_parameter("inference_period_sec", 0.05)
        self.declare_parameter("require_new_frame", True)
        self.declare_parameter("driving_command", 4)
        self.declare_parameter("replicate_jpeg_quality", 20)
        self.declare_parameter("decode_workers", 6)
        self.declare_parameter("gnss_mount_offset_x", AGENT_GNSS_MOUNT_X)

        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("plan_stall_timeout_sec", 6.0)
        self.declare_parameter("brake_when_stale", True)
        self.declare_parameter("control_trace", False)
        self.declare_parameter("control_trace_path", "")

        self._device = torch.device("cuda")

        self._camera_topics = [
            str(t) for t in (self.get_parameter("camera_topics").value or []) if str(t)
        ]
        if not self._camera_topics:
            raise ValueError("camera_topics must list 6 sensor_msgs/Image topics in CAMERA_ORDER")
        self._camera_sync_tol = float(self.get_parameter("camera_sync_tolerance_sec").value)
        self._timestamp_mode = str(self.get_parameter("timestamp_mode").value or "sensor")
        if self._timestamp_mode not in ("sensor", "agent"):
            raise ValueError("timestamp_mode must be 'sensor' or 'agent'")
        self._inference_period = float(self.get_parameter("inference_period_sec").value)
        self._require_new_frame = bool(self.get_parameter("require_new_frame").value)
        self._driving_command = int(self.get_parameter("driving_command").value)
        self._jpeg_quality = int(self.get_parameter("replicate_jpeg_quality").value)
        self._decode_workers = max(1, int(self.get_parameter("decode_workers").value))
        self._precision = str(self.get_parameter("precision").value or "config").lower()
        if self._precision not in ("config", "fp32", "fp16"):
            raise ValueError("precision must be 'config', 'fp32' or 'fp16'")
        self._speedup_opts = {
            "merge_lora": bool(self.get_parameter("merge_lora").value),
            "llm_attn": str(self.get_parameter("llm_attn").value or "keep"),
            "vit_glue": bool(self.get_parameter("vit_glue").value),
            "down_proj_t": bool(self.get_parameter("down_proj_t").value),
            "vit_weight_t": bool(self.get_parameter("vit_weight_t").value),
            "vit_window_nopad": bool(self.get_parameter("vit_window_nopad").value),
            "int8_targets": [t.strip() for t in
                             str(self.get_parameter("int8_targets").value or "").split(",") if t.strip()],
            "int8_skip": [t.strip() for t in
                          str(self.get_parameter("int8_skip").value or "").split(",") if t.strip()],
            "int8_calib_path": str(self.get_parameter("int8_calib_path").value or ""),
            "int8_alpha": float(self.get_parameter("int8_alpha").value),
            "vit_input_size": int(self.get_parameter("vit_input_size").value),
            "rear_view_refresh_every": int(self.get_parameter("rear_view_refresh_every").value),
            "map_head_slice": bool(self.get_parameter("map_head_slice").value),
            "compile_targets": [t.strip() for t in
                                str(self.get_parameter("compile_targets").value or "").split(",") if t.strip()],
            "compile_mode": str(self.get_parameter("compile_mode").value or "default"),
            "cuda_graph_vit": bool(self.get_parameter("cuda_graph_vit").value),
            "cuda_graph_heads": bool(self.get_parameter("cuda_graph_heads").value),
            "logits_slice": bool(self.get_parameter("logits_slice").value),
            "overlap_experts": bool(self.get_parameter("overlap_experts").value),
        }
        self._pipeline_prep = bool(self.get_parameter("pipeline_prep").value)
        self._pipeline_margin_ms = float(self.get_parameter("pipeline_margin_ms").value)
        self._profile_stages = bool(self.get_parameter("profile_stages").value)
        self._stage_timer: Optional[sp.StageTimer] = None
        self._decoder = sp.ParallelDecoder(decode_image, workers=self._decode_workers)
        self._gnss_offset_x = float(self.get_parameter("gnss_mount_offset_x").value)
        if self._gnss_offset_x:
            self.get_logger().info(
                f"ego localised {self._gnss_offset_x:+.2f} m (vehicle frame) from the "
                f"odometry origin, matching the reference agent's GNSS mount")
        self._control_hz = float(self.get_parameter("control_hz").value)
        self._plan_stall_timeout = float(self.get_parameter("plan_stall_timeout_sec").value)
        self._brake_when_stale = bool(self.get_parameter("brake_when_stale").value)
        self._control_trace = bool(self.get_parameter("control_trace").value)
        self._trace_fh = None
        self._trace_tick = 0
        self._trace_plan_seq = 0
        self._trace_last_arrived = None
        self._trace_ticks_on_plan = 0
        self._last_pid_meta = None
        if self._control_trace:
            _p = str(self.get_parameter("control_trace_path").value or "")
            if not _p:
                _p = "/benchmarking/minddrive_env/logs/control_trace_%s.csv" % (
                    time.strftime("%Y%m%d_%H%M%S"))
            try:
                os.makedirs(os.path.dirname(_p), exist_ok=True)
                self._trace_fh = open(_p, "w", buffering=1)
                self._trace_fh.write(
                    "t,tick,plan_seq,swap,ticks_on_plan,plan_age,stalled,"
                    "speed,desired_speed,delta,angle,angle_final,aim_x,aim_y,"
                    "steer,throttle,brake\n")
                self.get_logger().info("control trace -> %s" % _p)
            except Exception as _exc:
                self.get_logger().warn("control_trace disabled: cannot write %s (%s)" % (_p, _exc))
                self._control_trace = False

        control_topic = self.get_parameter("control_topic").value
        self._control_pub = self.create_publisher(
            CarlaEgoVehicleControl, control_topic,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1))
        self.get_logger().info(f"Publishing CarlaEgoVehicleControl on {control_topic}")
        self._meta_pub = self.create_publisher(
            String, self.get_parameter("meta_action_topic").value, 10)

        self._cams: Optional[MultiCameraBuffer] = None
        self._latest_speed: float = 0.0
        self._latest_odom: Optional[Odometry] = None
        self._latest_odom_mono: float = 0.0
        self._latest_imu: Optional[Imu] = None
        self._route_change_mono: float = 0.0
        self._traj_state: Optional[tuple] = None
        self._last_control: Optional[tuple] = None
        self._ctrl_count: int = 0
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
        self._last_period_mark: Optional[float] = None

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None
        self._prep_executor = ThreadPoolExecutor(max_workers=1)
        self._prep_future: Optional[Future] = None
        self._prep_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._fwd_expected_end: Optional[float] = None
        self._fwd_ms_avg: Optional[float] = None
        self._prep_ms_avg: float = 250.0
        self._model = None
        self._inference_only_pipeline = None
        self._mm_collate = None
        self._get_box_type = None
        self._pid = None

        self.get_logger().info("Loading MindDrive model ...")
        self._executor.submit(self._setup_model).result()

        self._cams = MultiCameraBuffer(self, self._camera_topics, img_qos, self._camera_sync_tol)
        speed_topic = self.get_parameter("speed_topic").value
        self.create_subscription(Float32, speed_topic, self._speed_callback, 10)
        odom_topic = self.get_parameter("odometry_topic").value
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        imu_topic = self.get_parameter("imu_topic").value
        self.create_subscription(Imu, imu_topic, self._imu_callback, odom_qos)
        route_topic = self.get_parameter("route_topic").value
        self.create_subscription(CarlaRoute, route_topic, self._route_callback, route_qos)
        self.get_logger().info(
            f"Subscribed: speed {speed_topic}, odometry {odom_topic}, imu {imu_topic}, "
            f"route {route_topic}")

        self._timer = self.create_timer(self._inference_period, self._timer_callback)
        self._control_timer = self.create_timer(1.0 / self._control_hz, self._control_callback)
        self.get_logger().info(
            f"Control loop at {self._control_hz:g} Hz "
            f"(car stopped if no new plan arrives for {self._plan_stall_timeout:g} s)")

    def _setup_model(self) -> None:
        repo_path = str(self.get_parameter("minddrive_repo_path").value or "")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        if repo_path:
            os.chdir(repo_path)

        from team_code.pid_controller_de import PIDController
        self._PIDController = PIDController
        self._pid = PIDController()

        from mmcv import Config
        from mmcv.models import build_model
        from mmcv.utils import load_checkpoint
        from mmcv.datasets.pipelines import Compose
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form
        from mmcv.core.bbox import get_box_type
        import mmcv as _mmcv
        self._mm_collate = mm_collate_to_batch_form
        self._get_box_type = get_box_type
        self.get_logger().info(f"mmcv fork: {os.path.dirname(os.path.abspath(_mmcv.__file__))}")

        cfg_path = str(self.get_parameter("minddrive_config_path").value)
        ckpt_path = str(self.get_parameter("minddrive_checkpoint_path").value)
        cfg = Config.fromfile(cfg_path)
        if self._precision == "fp16":
            cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = True, False, False
            self.get_logger().info("precision: fp16_infer (LLM fp16, ViT half, heads fp32)")
        elif self._precision == "fp32":
            cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = False, False, True
            self.get_logger().info("precision: fp32_infer")
        else:
            self.get_logger().info("precision: the config's own flags")
        self.get_logger().info(
            f"config {cfg_path}: lm_model_type={cfg.model.get('lm_model_type')} "
            f"fp32_infer={cfg.model.get('fp32_infer')} fp16_infer={cfg.model.get('fp16_infer')} "
            f"llm={cfg.model.get('lm_head')}; checkpoint {ckpt_path}")

        if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
            import importlib
            importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))

        t0 = time.time()
        self._model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                                  test_cfg=cfg.get("test_cfg"))
        self.get_logger().info(f"build_model done in {time.time() - t0:.0f} s; loading checkpoint")
        t1 = time.time()
        load_checkpoint(self._model, ckpt_path, map_location="cpu")
        self.get_logger().info(f"load_checkpoint done in {time.time() - t1:.0f} s")
        self._model.cuda()
        self._model.eval()

        applied = sp.apply_speedups(self._model, self._speedup_opts, self.get_logger().info)
        self.get_logger().info(f"speedups applied: {applied or 'none (reference model)'}")
        if self._profile_stages:
            self._stage_timer = sp.StageTimer()
            self._stage_timer.wrap_minddrive(self._model)

        pipe_cfg = [t for t in cfg.inference_only_pipeline
                    if t["type"] not in ("LoadMultiViewImageFromFilesInCeph",)]
        size = int(self._speedup_opts.get("vit_input_size") or 640)
        if size != 640 and sp.set_pipeline_input_size(pipe_cfg, size):
            self.get_logger().info(f"pipeline: ViT input {size}x{size}")
        self._inference_only_pipeline = Compose(pipe_cfg)
        sp.parallelize_pipeline(self._inference_only_pipeline, self._decode_workers,
                                self.get_logger().info)
        self.get_logger().info(
            f"model on GPU ({torch.cuda.memory_allocated() / 1e9:.1f} GB allocated) "
            f"after {time.time() - t0:.0f} s; pipeline "
            f"{[t['type'] for t in cfg.inference_only_pipeline]}")

        self._warmup()
        self.get_logger().info("MindDrive model loaded and ready.")

    def _warmup(self) -> None:
        """One full forward on a black frame so the first real inference is not
        a cold-start outlier. Goes through the real pipeline. Best-effort."""
        try:
            t0 = time.time()
            black = np.zeros((900, 1600, 3), dtype=np.uint8)
            can_bus = np.zeros(18)
            can_bus[3:7] = list(Quaternion(axis=[0, 0, 1], radians=0.0))
            results = build_agent_results({c: black for c in CAMERA_ORDER}, can_bus, 0.0, 4,
                                          "warmup", 0, 0.0, self._get_box_type)
            results = self._inference_only_pipeline(results)
            batch = self._mm_collate([results], samples_per_gpu=1)
            batch_to_device(batch, self._device)
            if self._speedup_opts["overlap_experts"]:
                sp.reorder_rounds_for_overlap(batch, self._model)
            n_warm = 2 if (self._speedup_opts["compile_targets"] or self._speedup_opts["cuda_graph_vit"]
                           or self._speedup_opts["cuda_graph_heads"]) else 1
            for _ in range(n_warm):
                sp.mark_step()
                with torch.no_grad():
                    custom_wrap_fp16_model(self._model)
                    out = self._model(batch, return_loss=False)
                torch.cuda.synchronize()
            pb = out[0]["pts_bbox"]
            self._clear_model_memory()
            self.get_logger().info(
                f"Model warmup OK in {(time.time() - t0) * 1e3:.0f} ms: "
                f"speed traj {tuple(pb['ego_fut_preds'].shape)}, "
                f"path traj {tuple(pb['pw_ego_fut_pred'].shape)}, "
                f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        except Exception as e:
            import traceback
            self.get_logger().warn(f"Model warmup skipped ({type(e).__name__}: {e})\n"
                                   f"{traceback.format_exc()}")

    def _clear_model_memory(self) -> None:
        model = self._model
        if model is None:
            return
        if hasattr(model, "test_flag"):
            model.test_flag = False
        if getattr(model, "planning_memory", None) is not None:
            model.planning_memory = None

    def _speed_callback(self, msg: Float32) -> None:
        self._latest_speed = float(msg.data)

    def _odometry_callback(self, msg: Odometry) -> None:
        self._latest_odom = msg
        self._latest_odom_mono = time.monotonic()

    def _imu_callback(self, msg: Imu) -> None:
        self._latest_imu = msg

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
            self._scene_token = f"route-{self._route_serial:04d}"
            self._pending_route_reset = True
            self._traj_state = None
            self._last_control = None
            self._route_change_mono = time.monotonic()
            self.get_logger().info(
                f"Route received: {len(route)} waypoints -- NEW route ({self._scene_token}); "
                f"context will be reset before the next inference.")
        else:
            self.get_logger().info(
                f"Route received: {len(route)} waypoints -- same plan as {self._scene_token}, "
                f"keeping context.")

    def _reset_for_new_route(self) -> None:
        """Drop everything scoped to the route that just ended: the heads'
        temporal memory (via test_flag, the model's own first-frame path), the
        frame counter, the PID's 40-sample windows (MinddriveAgent.setup builds
        a new PIDController per route) and the plan the control loop tracks."""
        self._clear_model_memory()
        self._frame_idx = 0
        self._last_ref_stamp = None
        self._prev_sensor_ts = None
        self._pid = self._PIDController()
        self._traj_state = None
        self._last_control = None
        self._last_pid_meta = None
        self._pending_route_reset = False
        self.get_logger().info(
            f"Context reset for {self._scene_token}: temporal memory, frame counter "
            f"and controller state cleared.")

    def _check_memory_continuity(self, sensor_ts: float) -> None:
        prev, self._prev_sensor_ts = self._prev_sensor_ts, sensor_ts
        if prev is None:
            return
        dt = abs(sensor_ts - prev)
        self._dt_samples += 1
        if dt < MEMORY_MAX_DT:
            return
        self._memoryless_frames += 1
        now = time.monotonic()
        if now - self._last_dt_log_t < 10.0:
            return
        self._last_dt_log_t = now
        pct = 100.0 * self._memoryless_frames / max(self._dt_samples, 1)
        extra = ""
        if self._timestamp_mode == "agent":
            extra = (" -- timestamp_mode=agent is masking this: the model is told "
                     f"{1.0 / AGENT_HZ:.2f} s elapsed, so memory is kept with a false dt")
        self.get_logger().warn(
            f"temporal memory: {dt:.2f} s between frames exceeds the {MEMORY_MAX_DT:.1f} s "
            f"retention window, so the rolling memory is zeroed and MindDrive runs "
            f"single-frame ({self._memoryless_frames}/{self._dt_samples} frames, {pct:.0f}%)"
            f"{extra}.")

    def _log_dispatch_state(self, reason: str, interval_sec: float = 2.0) -> None:
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
        self._start_forward(self._executor.submit(self._run_inference, snapshot), include_prep=True)

    def _start_forward(self, future: Future, include_prep: bool) -> None:
        now = time.time()
        expected = (self._fwd_ms_avg or 0.0) + (self._prep_ms_avg if include_prep else 0.0)
        self._fwd_expected_end = now + expected / 1e3 if self._fwd_ms_avg else None
        self._active_future = future
        future.add_done_callback(self._on_future_done)

    def _maybe_pipeline_prep(self) -> None:
        """While a forward is in flight: start preparing the next frame so it
        is ready right when the GPU frees up. Started late on purpose so the
        input is no older than it would be in the sequential path."""
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
                self.get_logger().error(f"MindDrive prep failed: {exc}\n{traceback.format_exc()}")
                return
            self._start_forward(self._executor.submit(self._forward, prepared), include_prep=False)

    def _take_snapshot(self) -> Optional[dict]:
        missing = self._cams.missing() if self._cams is not None else ["(no subs)"]
        if missing:
            self._log_dispatch_state(f"waiting: no frame yet from {', '.join(missing)}")
            return None
        if self._latest_odom is None:
            self._log_dispatch_state("waiting: no odometry received yet")
            return None
        if self._latest_odom_mono < self._route_change_mono:
            self._log_dispatch_state("waiting: no odometry since the new route arrived")
            return None
        oldest_cam = self._cams.oldest_receive_time()
        if oldest_cam is None or oldest_cam < self._route_change_mono:
            self._log_dispatch_state("waiting: camera frames from before the route change")
            return None
        ref_stamp = self._cams.reference_stamp()
        if self._require_new_frame and ref_stamp == self._last_ref_stamp:
            self._log_dispatch_state(
                f"skipped: no new front frame since last run "
                f"(stamp held at {ref_stamp[0]}.{ref_stamp[1]:09d}) -- camera stream stalled?")
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
            "route_serial": self._route_serial,
        }

    def _compute_driving_command(self, ego_xy: np.ndarray) -> int:
        """Faithful port of Bench2Drive RoutePlanner.run_step (team_code/planner.py)."""
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
        """road_options is uint8[]: CARLA's VOID (-1) arrives as 255."""
        cmd = int(cmd)
        return cmd if 1 <= cmd <= 6 else self._driving_command

    def _agent_ego_xy(self, xy, ego_theta: float) -> np.ndarray:
        """Vehicle-origin ENU position -> the GNSS-mount position the agent uses."""
        xy = np.asarray(xy, dtype=np.float64)[:2]
        d = self._gnss_offset_x
        if not d:
            return xy.copy()
        return xy + np.array([d * math.cos(ego_theta), d * math.sin(ego_theta)], dtype=np.float64)

    def _odom_to_canbus(self, odom: Odometry, imu: Optional[Imu], speed: float):
        """The agent's 18-dim can_bus from ros-bridge Odometry (+ Imu).

        No manual sign flips except one: the bridge has already applied the
        CARLA->ROS conversion the agent does by hand (position y negated, yaw
        negated, accelerometer y negated), so those values are used directly.
        Angular velocity is the exception: carla_ros_bridge/imu.py publishes
        (-gx, +gy, -gz) whereas the agent negates all three over the raw gyro,
        so the y component is flipped here (settled for ORION on 2026-09-16).
        """
        pose = odom.pose.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        ego_theta = float(Rotation.from_quat(quat).as_euler("zyx")[0])

        if speed is None or speed <= 0.0:
            lin = odom.twist.twist.linear
            speed = math.hypot(lin.x, lin.y)

        if imu is not None:
            acceleration = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y,
                                     imu.linear_acceleration.z], dtype=np.float64)
            angular_velocity = np.array([imu.angular_velocity.x, imu.angular_velocity.y,
                                         imu.angular_velocity.z], dtype=np.float64)
        else:
            acceleration = np.zeros(3)
            ang = odom.twist.twist.angular
            angular_velocity = np.array([ang.x, ang.y, ang.z], dtype=np.float64)

        gnss_xy = self._agent_ego_xy(pos[:2], ego_theta)
        can_bus = np.zeros(18)
        can_bus[0] = gnss_xy[0]
        can_bus[1] = gnss_xy[1]
        can_bus[3:7] = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus[7] = speed
        can_bus[10:13] = acceleration
        can_bus[13:16] = np.array([angular_velocity[0], -angular_velocity[1], angular_velocity[2]],
                                  dtype=np.float64)
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180
        return can_bus, ego_theta

    def _build_batch(self, snapshot: dict):
        odom = snapshot["odom"]
        frame_idx = self._frame_idx
        self._frame_idx += 1
        sensor_ts = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        timestamp = frame_idx / AGENT_HZ if self._timestamp_mode == "agent" else sensor_ts
        self._check_memory_continuity(sensor_ts)

        images = snapshot["images"]
        decoded = self._decoder([images[cam] for cam in CAMERA_ORDER], self._jpeg_quality)
        images_bgr = dict(zip(CAMERA_ORDER, decoded))

        can_bus, ego_theta = self._odom_to_canbus(odom, snapshot.get("imu"),
                                                  float(snapshot.get("speed", 0.0)))
        ego_xy = can_bus[0:2].copy()
        command = self._compute_driving_command(ego_xy)

        results = build_agent_results(images_bgr, can_bus, ego_theta, command,
                                      self._scene_token, frame_idx, timestamp,
                                      self._get_box_type)
        results = self._inference_only_pipeline(results)
        batch = self._mm_collate([results], samples_per_gpu=1)
        batch_to_device(batch, self._device)
        if self._speedup_opts["overlap_experts"]:
            sp.reorder_rounds_for_overlap(batch, self._model)

        near_node = (np.array(self._route[1][0], dtype=np.float64) if len(self._route) > 1
                     else (np.array(self._route[0][0], dtype=np.float64) if self._route else None))
        meta = {
            "stamp": snapshot["images"][REFERENCE_CAMERA][0],
            "ego_xy": ego_xy,
            "ego_theta": ego_theta,
            "near_node": near_node,
            "speed": float(snapshot.get("speed", 0.0)),
            "command": command,
        }
        return batch, meta

    def _run_inference(self, snapshot: dict):
        """Sequential path: prep + forward on the GPU worker."""
        return self._forward(self._prepare(snapshot))

    def _prepare(self, snapshot: dict) -> dict:
        """CPU half of a frame (decode, pipeline, collate, H2D). Runs on the GPU
        worker in the sequential path or on the prep worker when pipelined;
        the lock keeps _build_batch's bookkeeping strictly ordered."""
        with self._prep_lock:
            if self._pending_route_reset:
                self._reset_for_new_route()
            t0 = time.time()
            batch, meta = self._build_batch(snapshot)
            prep_ms = (time.time() - t0) * 1e3
        self._prep_ms_avg = 0.8 * self._prep_ms_avg + 0.2 * prep_ms
        return {"batch": batch, "meta": meta, "snapshot": snapshot, "prep_ms": prep_ms}

    @torch.no_grad()
    def _forward(self, prepared: dict):
        t_start = time.time()
        snapshot = prepared["snapshot"]
        if snapshot.get("route_serial") != self._route_serial or self._pending_route_reset:
            self.get_logger().info("prepared batch discarded: route changed before the forward")
            return None
        batch, meta = prepared["batch"], prepared["meta"]
        route_serial = self._route_serial

        if self._stage_timer is not None:
            self._stage_timer._pending.clear()
        sp.mark_step()
        custom_wrap_fp16_model(self._model)
        output = self._model(batch, return_loss=False)
        torch.cuda.synchronize()
        t_forward = time.time()
        stages = self._stage_timer.report() if self._stage_timer is not None else None
        forward_ms = (t_forward - t_start) * 1e3
        self._fwd_ms_avg = forward_ms if self._fwd_ms_avg is None else 0.8 * self._fwd_ms_avg + 0.2 * forward_ms
        period_ms = (t_start - self._last_period_mark) * 1e3 if self._last_period_mark else None
        self._last_period_mark = t_start

        pb = output[0]["pts_bbox"]
        speed_traj = pb["ego_fut_preds"].cpu().numpy()
        path_traj = pb["pw_ego_fut_pred"].cpu().numpy()
        speed_action = _action_name(SPEED_ACTIONS, pb.get("speed_value"))
        path_action = _action_name(PATH_ACTIONS, pb.get("path_value"))

        if route_serial != self._route_serial:
            self.get_logger().info(
                f"plan discarded: route changed during the forward "
                f"(now {self._scene_token}); context reset follows")
            return None

        self._traj_state = (
            speed_traj.copy(),
            path_traj.copy(),
            np.asarray(meta["ego_xy"], dtype=np.float64).copy(),
            float(meta["ego_theta"]),
            float(snapshot.get("t_dispatch", time.time())),
            time.time(),
        )
        self._meta_pub.publish(String(data=f"speed={speed_action} path={path_action}"))
        return {
            "prep_ms": prepared["prep_ms"],
            "forward_ms": forward_ms,
            "total_ms": prepared["prep_ms"] + forward_ms,
            "period_ms": period_ms,
            "stages": stages,
            "speed_action": speed_action,
            "path_action": path_action,
            "command": meta["command"],
            "path_len": float(np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), path_traj)), axis=0), axis=1).sum()),
            "speed_wp0": float(np.linalg.norm(speed_traj[0])),
        }

    @staticmethod
    def _rot(angle: float) -> np.ndarray:
        c, sn = math.cos(angle), math.sin(angle)
        return np.array([[c, -sn], [sn, c]])

    def _plan_frame_offset(self, ego_xy0, theta0, ego_xy_now, theta_now):
        """Current ego position expressed in the plan's own frame, and the
        rotation from the plan frame to the current ego frame.  The plan frame
        is rot(raw) applied to ENU deltas, raw = pi/2 - ego_theta (see
        _compute_control), so composing the two poses gives
        p' = R(raw_now - raw0) (p - o)."""
        raw0 = np.pi / 2.0 - theta0
        raw_now = np.pi / 2.0 - theta_now
        o = self._rot(raw0) @ (np.asarray(ego_xy_now) - np.asarray(ego_xy0))
        return o, self._rot(raw_now - raw0)

    def _rebase_speed_traj(self, traj: np.ndarray, age: float, o: np.ndarray, R: np.ndarray) -> np.ndarray:
        """The 6 x 0.5 s speed waypoints, re-cut from NOW.

        The waypoints are 0.5 s apart measured from the frame the model looked
        at, and that frame is one inference old by the time the plan arrives.
        control_pid reads desired_speed from ||wp[0]|| and ||wp[1]-wp[0]||, so
        a point the car has nearly reached reads as "slow down".  The plan is
        resampled from `age` onwards (element j = where the model expected to be
        at age + 0.5 (j+1)), the last segment's velocity extended past the 3 s
        horizon (a forward pass eats a large part of it), then moved into the
        current ego frame.  Identical to orion_withpid_node._rebase_plan.
        """
        knots = np.vstack((np.zeros((1, 2)), np.asarray(traj, dtype=np.float64)))
        horizon = TRAJ_DT * (len(knots) - 1)
        v_end = (knots[-1] - knots[-2]) / TRAJ_DT

        def sample(t: float) -> np.ndarray:
            if t >= horizon:
                return knots[-1] + (t - horizon) * v_end
            u = max(t, 0.0) / TRAJ_DT
            i = min(int(math.floor(u)), len(knots) - 2)
            return knots[i] + (u - i) * (knots[i + 1] - knots[i])

        pts = np.array([sample(age + TRAJ_DT * (j + 1)) for j in range(len(traj))])
        return (R @ (pts - o).T).T

    def _rebase_path(self, path: np.ndarray, o: np.ndarray, R: np.ndarray) -> np.ndarray:
        """The 20 path points, re-cut from where the ego is NOW.

        pw_ego_fut_pred has no time axis -- it is a polyline along the intended
        path -- so the spatial analogue of the time resample above is used:
        project the current ego position onto the polyline, and resample the
        same number of points at the polyline's own spacing from that arc
        length onwards, extending the last segment's direction past the end.
        control_pid's aim search (the point nearest aim_dist=3.5 m) then sees
        the same kind of list the agent hands it: rooted at the ego, all ahead.
        Then moved into the current ego frame.
        """
        n = len(path)
        knots = np.vstack((np.zeros((1, 2)), np.asarray(path, dtype=np.float64)))
        seg = np.diff(knots, axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seglen)))
        total = float(cum[-1])
        if total <= 1e-6:
            return (R @ (knots[1:] - o).T).T
        ds = total / n

        s_now, best = 0.0, np.inf
        for i in range(len(seg)):
            if seglen[i] <= 1e-9:
                d = np.linalg.norm(o - knots[i])
                t = 0.0
            else:
                t = float(np.clip(np.dot(o - knots[i], seg[i]) / (seglen[i] ** 2), 0.0, 1.0))
                d = np.linalg.norm(o - (knots[i] + t * seg[i]))
            if d < best:
                best, s_now = d, cum[i] + t * seglen[i]

        nz = np.nonzero(seglen > 1e-9)[0]
        u_end = seg[nz[-1]] / seglen[nz[-1]] if len(nz) else np.zeros(2)

        def sample(s: float) -> np.ndarray:
            if s >= total:
                return knots[-1] + (s - total) * u_end
            i = int(np.searchsorted(cum, s, side="right") - 1)
            i = min(max(i, 0), len(seg) - 1)
            t = (s - cum[i]) / seglen[i] if seglen[i] > 1e-9 else 0.0
            return knots[i] + t * seg[i]

        pts = np.array([sample(s_now + ds * (j + 1)) for j in range(n)])
        return (R @ (pts - o).T).T

    def _control_callback(self) -> None:
        state = self._traj_state
        if state is None or self._latest_odom is None:
            return
        speed_traj, path_traj, ego_xy0, theta0, t0, t_arrived = state

        now = time.time()
        stalled = now - t_arrived
        if stalled > self._plan_stall_timeout:
            self._stop_the_car(
                f"no new plan for {stalled:.1f}s (limit {self._plan_stall_timeout:g}s) "
                f"-- the model has stopped answering")
            return
        age = now - t0

        odom = self._latest_odom
        ego_xy_now = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y],
                              dtype=np.float64)
        theta_now = float(Rotation.from_quat([
            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w,
        ]).as_euler("zyx")[0])
        ego_xy_now = self._agent_ego_xy(ego_xy_now, theta_now)

        o, R = self._plan_frame_offset(ego_xy0, theta0, ego_xy_now, theta_now)
        speed_plan = self._rebase_speed_traj(speed_traj, age, o, R)
        path_plan = self._rebase_path(path_traj, o, R)

        near_node = (np.array(self._route[1][0], dtype=np.float64) if len(self._route) > 1
                     else (np.array(self._route[0][0], dtype=np.float64) if self._route else None))
        meta = {
            "stamp": self.get_clock().now().to_msg(),
            "ego_xy": ego_xy_now,
            "ego_theta": theta_now,
            "near_node": near_node,
            "speed": float(self._latest_speed),
        }
        control = self._compute_control(path_plan, speed_plan, meta)
        if control is None:
            return
        self._control_pub.publish(control)
        self._last_control = (control.steer, control.throttle, control.brake)
        self._ctrl_count += 1
        if self._control_trace:
            self._write_control_trace(t_arrived, age, stalled, control)

    def _write_control_trace(self, t_arrived, age, stalled, control) -> None:
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
            try:
                return float(m.get(k, nan))
            except (TypeError, ValueError):
                return nan
        try:
            self._trace_fh.write(
                "%.6f,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f,%.5f,%.4f,%.4f\n" % (
                    time.time(), self._trace_tick, self._trace_plan_seq, swap,
                    self._trace_ticks_on_plan, age, stalled,
                    g("speed"), g("desired_speed"), g("delta"), g("angle"), g("angle_final"),
                    float(aim[0]), float(aim[1]), control.steer, control.throttle, control.brake))
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

    def _compute_control(self, path_wp: np.ndarray, speed_wp: np.ndarray,
                         meta: dict) -> Optional[CarlaEgoVehicleControl]:
        """control_pid(out_truck_path, out_truck, speed, local_command_xy) and
        the clipping / brake / 5 m/s rules that follow it in run_step.

        Frame note: ego_theta here is the vehicle->ENU yaw (== the agent's
        ego_theta), so the agent's raw compass heading is pi/2 - ego_theta, and
        route/ego positions are already ENU (the bridge applied the CARLA->ROS
        conversion), so the world->local rotation needs no Y flip.
        """
        near_node = meta.get("near_node")
        if near_node is None:
            self.get_logger().warn("No global plan yet; not publishing control.",
                                   throttle_duration_sec=5.0)
            return None
        ego_xy = meta["ego_xy"]
        raw_theta = np.pi / 2.0 - meta["ego_theta"]
        command_near_xy = np.array([near_node[0] - ego_xy[0], near_node[1] - ego_xy[1]])
        rotation_matrix = np.array([[math.cos(raw_theta), -math.sin(raw_theta)],
                                    [math.sin(raw_theta), math.cos(raw_theta)]])
        local_command_xy = rotation_matrix @ command_near_xy

        steer_traj, throttle_traj, brake_traj, metadata_traj = self._pid.control_pid(
            path_wp, speed_wp, np.float64(meta["speed"]), local_command_xy)
        self._last_pid_meta = metadata_traj

        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0
        if float(meta["speed"]) > SPEED_CAP_MPS:
            throttle_traj = 0
        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = meta["stamp"]
        cmd.steer = float(np.clip(float(steer_traj), -1, 1))
        cmd.throttle = float(np.clip(float(throttle_traj), 0, 0.75))
        cmd.brake = float(np.clip(float(brake_traj), 0, 1))
        return cmd

    def _on_future_done(self, future: Future) -> None:
        self._try_start_from_prepared()
        try:
            m = future.result()
        except Exception as exc:
            import traceback
            self.get_logger().error(
                f"MindDrive inference failed: {exc}\n{traceback.format_exc()}")
            return
        if not m:
            return
        ctrl = ""
        if self._last_control is not None:
            steer, throttle, brake = self._last_control
            ctrl = f" -> latest control steer={steer:+.3f} throttle={throttle:.3f} brake={brake:.3f}"
        published, self._ctrl_count = self._ctrl_count, 0
        period = f", period={m['period_ms']:.0f} ms" if m.get("period_ms") else ""
        self.get_logger().info(
            f"MindDrive inference: total={m['total_ms']:.1f} ms "
            f"(prep={m['prep_ms']:.1f}, forward={m['forward_ms']:.1f}{period}), "
            f"cmd={m['command']} decision: speed=\"{m['speed_action']}\" path=\"{m['path_action']}\", "
            f"path {m['path_len']:.1f} m, |wp0| {m['speed_wp0']:.2f} m, "
            f"{published} controls published since the last one{ctrl}")
        if m.get("stages"):
            self.get_logger().info("MindDrive stages (ms): "
                                   + sp.StageTimer.format(m["stages"], m["forward_ms"]))

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
    os.environ.setdefault("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}")
    rclpy.init(args=args)
    node = MinddriveRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
