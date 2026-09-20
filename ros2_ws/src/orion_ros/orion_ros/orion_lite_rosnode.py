#!/usr/bin/env python3

import os
import sys
import math
import time
import numpy as np
import torch
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from collections import deque
from scipy.optimize import fsolve

from sensor_msgs.msg import Image, Imu, NavSatFix
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float32, ColorRGBA
from carla_msgs.msg import CarlaEgoVehicleControl
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from pyquaternion import Quaternion as PyQuaternion

try:
    from carla_msgs.msg import CarlaRoute
    HAS_CARLA_MSGS = True
except ImportError:
    HAS_CARLA_MSGS = False

Config = build_model = load_checkpoint = Compose = None
mm_collate_to_batch_form = get_box_type = None
PIDController = RoutePlanner = None


def _import_orion_lite_deps(repo_path):
    """Prepend the Orion-Lite repo to sys.path / chdir into it, then import mmcv
    and team_code FROM it. Importing mmcv.models registers OrionDistilledNew into
    the same mmcv registry that build_model (also from this mmcv) consults."""
    global Config, build_model, load_checkpoint, Compose
    global mm_collate_to_batch_form, get_box_type, PIDController, RoutePlanner

    if 'mmcv' in sys.modules:
        raise RuntimeError(
            f"mmcv already imported from {getattr(sys.modules['mmcv'], '__file__', '?')} "
            "before repo_path was applied; nothing must import mmcv at module top.")

    if repo_path:
        while repo_path in sys.path:
            sys.path.remove(repo_path)
        sys.path.insert(0, repo_path)
        os.chdir(repo_path)

    from mmcv import Config as _Config
    import mmcv as _mmcv
    if repo_path and not os.path.realpath(_mmcv.__file__).startswith(os.path.realpath(repo_path)):
        raise RuntimeError(
            f"Loaded mmcv from {_mmcv.__file__}, expected it under {repo_path}. "
            "Another mmcv is shadowing the Orion-Lite fork.")
    from mmcv.models import build_model as _build_model
    from mmcv.utils import load_checkpoint as _load_checkpoint
    from mmcv.datasets.pipelines import Compose as _Compose
    from mmcv.parallel.collate import collate as _collate
    from mmcv.core.bbox import get_box_type as _get_box_type
    Config, build_model, load_checkpoint, Compose = _Config, _build_model, _load_checkpoint, _Compose
    mm_collate_to_batch_form, get_box_type = _collate, _get_box_type

    try:
        from team_code.pid_controller import PIDController as _PID
        from team_code.planner import RoutePlanner as _RP
    except ImportError:
        from pid_controller import PIDController as _PID
        from planner import RoutePlanner as _RP
    PIDController, RoutePlanner = _PID, _RP


MAX_STEER_ANGLE = 0.7


def command2hot(command, max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    out = np.zeros(max_dim)
    out[command] = 1
    return out

def command2nohot(command, max_dim=6):
    if command < 0:
        command = 4
    return command - 1

def invert_matrix_egopose_numpy(egopose):
    inv = np.zeros((4, 4), dtype=np.float32)
    R = egopose[:3, :3]
    t = egopose[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3]  = -R.T @ t
    inv[3, 3]   = 1.0
    return inv

_CUSTOM_FP16 = dict(map_head=False, pts_bbox_head=False)

def _custom_wrap_fp16(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for name, val in _CUSTOM_FP16.items():
        model._modules[name].fp16_enabled = val


class _RoadOptionValue:
    """Thin wrapper so route entries have a .value attribute like CARLA's enum."""
    def __init__(self, v):
        self.value = v


class OrionLiteRosNode(Node):
    CAM_NAMES = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
    ]

    LIDAR2IMG = {
        'CAM_FRONT': np.array([
            [ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
            [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
            [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT': np.array([
            [ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
            [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
            [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT': np.array([
            [ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00, -4.06010608e+02],
            [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
            [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK': np.array([
            [-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
            [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
            [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT': np.array([
            [-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
            [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
            [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_RIGHT': np.array([
            [ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
            [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
            [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
    }

    LIDAR2CAM = {
        'CAM_FRONT': np.array([
            [ 1.,   0.,   0.,   0.  ],
            [ 0.,   0.,  -1.,  -0.24],
            [ 0.,   1.,   0.,  -1.19],
            [ 0.,   0.,   0.,   1.  ]]),
        'CAM_FRONT_LEFT': np.array([
            [ 0.57357644,  0.81915204,  0.,  -0.22517331],
            [ 0.,          0.,         -1.,  -0.24      ],
            [-0.81915204,  0.57357644,  0.,  -0.82909407],
            [ 0.,          0.,          0.,   1.        ]]),
        'CAM_FRONT_RIGHT': np.array([
            [ 0.57357644, -0.81915204,  0.,   0.22517331],
            [ 0.,          0.,         -1.,  -0.24      ],
            [ 0.81915204,  0.57357644,  0.,  -0.82909407],
            [ 0.,          0.,          0.,   1.        ]]),
        'CAM_BACK': np.array([
            [-1.,  0.,   0.,   0.  ],
            [ 0.,  0.,  -1.,  -0.24],
            [ 0., -1.,   0.,  -1.61],
            [ 0.,  0.,   0.,   1.  ]]),
        'CAM_BACK_LEFT': np.array([
            [-0.34202014,  0.93969262,  0.,  -0.25388956],
            [ 0.,          0.,         -1.,  -0.24      ],
            [-0.93969262, -0.34202014,  0.,  -0.49288953],
            [ 0.,          0.,          0.,   1.        ]]),
        'CAM_BACK_RIGHT': np.array([
            [-0.34202014, -0.93969262,  0.,   0.25388956],
            [ 0.,          0.,         -1.,  -0.24      ],
            [ 0.93969262, -0.34202014,  0.,  -0.49288953],
            [ 0.,          0.,          0.,   1.        ]]),
    }

    LIDAR2EGO = np.array([
        [ 0.,  1.,  0., -0.39],
        [-1.,  0.,  0.,  0.  ],
        [ 0.,  0.,  1.,  1.84],
        [ 0.,  0.,  0.,  1.  ],
    ])

    def __init__(self):
        super().__init__('orion_lite_inference_node')

        if not HAS_CARLA_MSGS:
            self.get_logger().warn(
                'carla_msgs not found; route topic will not be handled')

        self.declare_parameter(
            'config_path', '/benchmarking/Orion-Lite/configs/orion_lite_closedloop.py')
        self.declare_parameter(
            'ckpt_path', '/benchmarking/orion_lite_checkpoints/fused_ckpts/orion_lite.pth')
        self.declare_parameter('repo_path', '/benchmarking/Orion-Lite')
        self.declare_parameter('front_camera_topic', '/carla/hero/rgb_0/image')
        self.declare_parameter('imu_topic',          '/carla/hero/imu')
        self.declare_parameter('gnss_topic',         '/carla/hero/gnss')
        self.declare_parameter('odometry_topic',     '/carla/hero/odometry')
        self.declare_parameter('speed_topic',        '/carla/hero/speed')
        self.declare_parameter('route_topic',        '/carla/hero/global_plan')
        self.declare_parameter('control_topic',      '/carla/hero/vehicle_control_cmd')
        self.declare_parameter('marker_topic',       '/orion_lite/predicted_trajectory_markers')
        self.declare_parameter('map_frame_id',       'map')

        config_path = self.get_parameter('config_path').value
        ckpt_path   = self.get_parameter('ckpt_path').value
        repo_path   = self.get_parameter('repo_path').value

        self.get_logger().info('Loading Orion-Lite model ...')
        _import_orion_lite_deps(repo_path)
        self._load_model(config_path, ckpt_path)

        self.bridge        = CvBridge()
        self.pidcontroller = PIDController()
        self.step          = -1

        self.lat_ref: float = 42.0
        self.lon_ref: float = 2.0
        self._latlon_initialized = False

        self.latest_imu:   Imu        = None
        self.latest_gnss:  NavSatFix  = None
        self.latest_odom:  Odometry   = None
        self.latest_speed: float      = 0.0
        self.route_planner: RoutePlanner = None

        self._setup_subscribers()

        control_topic = self.get_parameter('control_topic').value
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl, control_topic,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1))

        self._map_frame = self.get_parameter('map_frame_id').value
        marker_topic    = self.get_parameter('marker_topic').value
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.get_logger().info(
            f'Publishing trajectory markers on {marker_topic} (frame {self._map_frame})')

        self.get_logger().info(
            'OrionLiteRosNode ready, waiting for sensors and global plan...')


    def _load_model(self, config_path, ckpt_path):
        cfg = Config.fromfile(config_path)
        if getattr(cfg, 'plugin', False) and hasattr(cfg, 'plugin_dir'):
            import importlib
            module_path = cfg.plugin_dir.rstrip('/').replace('/', '.')
            importlib.import_module(module_path)

        self.model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'),
        )
        load_checkpoint(self.model, ckpt_path, map_location='cpu')
        self.model.cuda().eval()

        pipeline_cfg = [p for p in cfg.inference_only_pipeline
                        if p['type'] not in ['LoadMultiViewImageFromFilesInCeph']]
        self.inference_pipeline = Compose(pipeline_cfg)

        self._warmup()

    def _warmup(self):
        """Run ONE full forward on a black dummy frame so the first real
        inference isn't a cold-start outlier (CUDA kernel JIT, cuDNN autotune,
        allocator). Goes through the real pipeline so shapes are guaranteed
        correct. Best-effort: logged, never fatal."""
        try:
            t0 = time.time()
            dummy = np.zeros((900, 1600, 3), dtype=np.uint8)
            ego2world = np.eye(4)
            lidar2global = ego2world @ self.LIDAR2EGO
            results = {
                'lidar2img':     np.stack([self.LIDAR2IMG[c] for c in self.CAM_NAMES]),
                'lidar2cam':     np.stack([self.LIDAR2CAM[c] for c in self.CAM_NAMES]),
                'cam_intrinsic': [np.matmul(self.LIDAR2IMG[c], np.linalg.inv(self.LIDAR2CAM[c]))
                                  for c in self.CAM_NAMES],
                'img':         [dummy for _ in self.CAM_NAMES],
                'folder': ' ', 'scene_token': ' ', 'frame_idx': 0, 'timestamp': 0.0,
                'box_type_3d': get_box_type('LiDAR')[0],
                'can_bus':     np.zeros(18),
                'command':     command2nohot(4),
                'ego_fut_cmd': command2hot(4),
                'ego_pose':    lidar2global,
                'ego_pose_inv': invert_matrix_egopose_numpy(lidar2global),
                'lidar2ego':   self.LIDAR2EGO,
                'l2g_r_mat':   lidar2global[0:3, 0:3],
                'l2g_t':       lidar2global[0:3, 3],
            }
            stacked = np.stack(results['img'], axis=-1)
            results['img_shape'] = results['ori_shape'] = results['pad_shape'] = stacked.shape
            results = self.inference_pipeline(results)
            batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
            for key, data in batch.items():
                if key != 'img_metas' and torch.is_tensor(data[0]):
                    data[0] = data[0].cuda()
            _custom_wrap_fp16(self.model)
            with torch.no_grad():
                self.model(batch, return_loss=False)
            torch.cuda.synchronize()
            self.get_logger().info(f'Model warmup OK in {(time.time() - t0) * 1e3:.0f} ms')
        except Exception as e:
            self.get_logger().warn(f'Model warmup skipped ({type(e).__name__}: {e})')


    def _setup_subscribers(self):
        state_grp = ReentrantCallbackGroup()
        infer_grp = MutuallyExclusiveCallbackGroup()

        state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=10)
        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        route_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=1,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)

        imu_topic   = self.get_parameter('imu_topic').value
        gnss_topic  = self.get_parameter('gnss_topic').value
        odom_topic  = self.get_parameter('odometry_topic').value
        speed_topic = self.get_parameter('speed_topic').value
        front_topic = self.get_parameter('front_camera_topic').value
        route_topic = self.get_parameter('route_topic').value

        self.create_subscription(Imu,       imu_topic,   self._imu_cb,   state_qos, callback_group=state_grp)
        self.create_subscription(NavSatFix, gnss_topic,  self._gnss_cb,  state_qos, callback_group=state_grp)
        self.create_subscription(Odometry,  odom_topic,  self._odom_cb,  state_qos, callback_group=state_grp)
        self.create_subscription(Float32,   speed_topic, self._speed_cb, state_qos, callback_group=state_grp)

        if HAS_CARLA_MSGS:
            self.create_subscription(CarlaRoute, route_topic, self._route_cb,
                                     route_qos, callback_group=state_grp)
        else:
            self.get_logger().warn(
                f'Skipping {route_topic} subscription (no carla_msgs)')

        self.create_subscription(Image, front_topic, self._camera_cb,
                                 img_qos, callback_group=infer_grp)


    def _imu_cb(self, msg: Imu):
        self.latest_imu = msg

    def _gnss_cb(self, msg: NavSatFix):
        self.latest_gnss = msg
        self._try_init_latlon_ref()

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg
        self._try_init_latlon_ref()

    def _speed_cb(self, msg: Float32):
        self.latest_speed = msg.data

    def _try_init_latlon_ref(self):
        """Compute lat_ref/lon_ref once from the first GNSS + odometry pair.
        Identical fsolve logic to _init() in orion_b2d_agent.py."""
        if self._latlon_initialized or self.latest_gnss is None or self.latest_odom is None:
            return
        lat  = self.latest_gnss.latitude
        lon  = self.latest_gnss.longitude
        locx = self.latest_odom.pose.pose.position.x
        locy = self.latest_odom.pose.pose.position.y
        EARTH_RADIUS_EQUA = 6378137.0
        try:
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            solution = fsolve(equations, [0, 0])
            self.lat_ref, self.lon_ref = float(solution[0]), float(solution[1])
        except Exception as e:
            self.get_logger().warn(f'lat_ref/lon_ref init failed ({e}), using defaults')
            self.lat_ref, self.lon_ref = 0, 0
        self._latlon_initialized = True
        self.get_logger().info(
            f'lat_ref={self.lat_ref:.6f} lon_ref={self.lon_ref:.6f} initialized')

    def _route_cb(self, msg: 'CarlaRoute'):
        """
        Convert CarlaRoute to the format RoutePlanner expects.
        road_options values follow CARLA's RoadOption enum:
          0=VOID 1=LEFT 2=RIGHT 3=STRAIGHT 4=LANEFOLLOW 5=CHANGELEFT 6=CHANGERIGHT
        """
        plan = deque()
        for pose, opt_val in zip(msg.poses, msg.road_options):
            xy  = np.array([pose.position.x, pose.position.y])
            cmd = _RoadOptionValue(int(opt_val))
            plan.append((xy, cmd))

        planner = RoutePlanner(min_distance=4.0, max_distance=50.0)
        planner.route = plan
        self.route_planner = planner
        self.get_logger().info(f'Global plan received: {len(plan)} waypoints')


    def _camera_cb(self, front_msg: Image):
        if self.latest_imu is None:
            self.get_logger().warn('Waiting for IMU...', throttle_duration_sec=5.0)
            return
        if self.latest_gnss is None:
            self.get_logger().warn('Waiting for GNSS...', throttle_duration_sec=5.0)
            return
        if not self._latlon_initialized:
            self.get_logger().warn('Waiting for lat_ref/lon_ref initialization...',
                                   throttle_duration_sec=5.0)
            return
        if self.route_planner is None:
            self.get_logger().warn('Waiting for global plan...', throttle_duration_sec=5.0)
            return

        self.step += 1
        t0 = time.time()
        try:
            control = self._run_inference(front_msg)
            self.control_pub.publish(control)
            dt_ms = (time.time() - t0) * 1e3
            self.get_logger().info(
                f'step {self.step}: inference {dt_ms:.0f} ms '
                f'(~{1000.0 / max(dt_ms, 1e-3):.2f} FPS) -> '
                f'steer={control.steer:+.3f} throttle={control.throttle:.3f} '
                f'brake={control.brake:.3f}')
        except Exception as exc:
            self.get_logger().error(f'Inference error at step {self.step}: {exc}')


    @torch.no_grad()
    def _run_inference(self, front_msg: Image) -> CarlaEgoVehicleControl:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        bgr = self.bridge.imgmsg_to_cv2(front_msg, desired_encoding='bgr8')
        _, buf = cv2.imencode('.jpg', bgr, encode_param)
        front_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        imgs = {name: front_img for name in self.CAM_NAMES}

        gps = np.array([self.latest_gnss.latitude, self.latest_gnss.longitude])
        pos = self.gps_to_location(gps)
        pos_x, pos_y = pos[0], pos[1]

        q = self.latest_imu.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        raw_theta = -math.atan2(siny_cosp, cosy_cosp)

        imu = self.latest_imu
        acceleration     = np.array([imu.linear_acceleration.x,
                                      imu.linear_acceleration.y,
                                      imu.linear_acceleration.z])
        angular_velocity = np.array([imu.angular_velocity.x,
                                      imu.angular_velocity.y,
                                      imu.angular_velocity.z])
        speed = np.float64(self.latest_speed)

        (_, curr_cmd), (near_node, _) = self.route_planner.run_step(pos)

        ego_theta = -raw_theta + np.pi / 2
        rotation  = list(PyQuaternion(axis=[0, 0, 1], radians=ego_theta))

        can_bus = np.zeros(18)
        can_bus[0]     = pos_x
        can_bus[1]     = -pos_y
        can_bus[3:7]   = rotation
        can_bus[7]     = self.latest_speed
        can_bus[10:13] = acceleration
        can_bus[11]   *= -1
        can_bus[13:16] = -angular_velocity
        can_bus[16]    = ego_theta
        can_bus[17]    = ego_theta / np.pi * 180

        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = PyQuaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3]   = can_bus[0:2]
        lidar2global = ego2world @ self.LIDAR2EGO
        ego_pose_inv = invert_matrix_egopose_numpy(lidar2global)

        cmd_int      = curr_cmd.value
        command_1idx = cmd_int + 1 if cmd_int > 0 else -1

        results = {
            'lidar2img':     np.stack([self.LIDAR2IMG[c] for c in self.CAM_NAMES]),
            'lidar2cam':     np.stack([self.LIDAR2CAM[c] for c in self.CAM_NAMES]),
            'cam_intrinsic': [
                np.matmul(self.LIDAR2IMG[c], np.linalg.inv(self.LIDAR2CAM[c]))
                for c in self.CAM_NAMES
            ],
            'img':         [imgs[c] for c in self.CAM_NAMES],
            'folder':      ' ',
            'scene_token': ' ',
            'frame_idx':   self.step,
            'timestamp':   self.step / 20.0,
            'box_type_3d': get_box_type('LiDAR')[0],
            'can_bus':     can_bus,
            'command':     command2nohot(command_1idx),
            'ego_fut_cmd': command2hot(command_1idx),
            'ego_pose':    lidar2global,
            'ego_pose_inv': ego_pose_inv,
            'lidar2ego':   self.LIDAR2EGO,
            'l2g_r_mat':   lidar2global[0:3, 0:3],
            'l2g_t':       lidar2global[0:3, 3],
        }
        stacked = np.stack(results['img'], axis=-1)
        results['img_shape'] = results['ori_shape'] = results['pad_shape'] = stacked.shape

        results = self.inference_pipeline(results)

        batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in batch.items():
            if key != 'img_metas' and torch.is_tensor(data[0]):
                data[0] = data[0].cuda()

        _custom_wrap_fp16(self.model)
        out       = self.model(batch, return_loss=False)
        out_truck = out[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()

        if self.latest_odom is not None:
            self.marker_pub.publish(
                self._trajectory_to_markers(out_truck, self.latest_odom,
                                            front_msg.header.stamp))

        near_xy_world    = np.array([near_node[0] - pos_x, -(near_node[1] - pos_y)])
        rot              = np.array([[math.cos(raw_theta), -math.sin(raw_theta)],
                                     [math.sin(raw_theta),  math.cos(raw_theta)]])
        local_command_xy = rot @ near_xy_world

        steer, throttle, brake, metadata = self.pidcontroller.control_pid(
            out_truck, speed, local_command_xy)

        steer    = float(np.clip(steer,    -1.0, 1.0))
        throttle = float(np.clip(throttle,  0.0, 0.75))
        brake    = float(brake)

        if brake < 0.05:     brake = 0.0
        if throttle > brake: brake = 0.0

        cmd = CarlaEgoVehicleControl()
        cmd.header.stamp = front_msg.header.stamp
        cmd.steer    = float(steer)
        cmd.throttle = float(throttle)
        cmd.brake    = float(brake)
        return cmd


    def _trajectory_to_markers(self, ego_fut_preds: np.ndarray,
                               odom: Odometry, stamp) -> MarkerArray:
        """Convert ORION's ego-relative predicted trajectory to a map-frame
        MarkerArray (LINE_STRIP) for RViz.

        ORION's ego_fut_preds live in the LIDAR_TOP frame, where index 1 = forward
        and index 0 = right. ROS base_link is (+x = forward, +y = left), so we swap
        to (x_fwd = point[1], y_left = -point[0]) before rotating into the map frame
        with the odometry pose. Without this swap the trajectory is rotated 90°.
        (Mirrors orion_node.py's Autoware trajectory conversion.)
        """
        pos  = odom.pose.pose.position
        quat = odom.pose.pose.orientation
        ego_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        ego_rot = PyQuaternion(quat.w, quat.x, quat.y, quat.z).rotation_matrix

        marker = Marker()
        marker.header.stamp    = stamp
        marker.header.frame_id = self._map_frame
        marker.ns              = 'orion_lite_trajectory'
        marker.id              = 0
        marker.type            = Marker.LINE_STRIP
        marker.action          = Marker.ADD
        marker.scale.x         = 0.3
        marker.color           = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        marker.pose.orientation.w = 1.0

        for point in ego_fut_preds:
            local = np.array([float(point[1]), -float(point[0]), 0.0])
            world = ego_rot @ local + ego_pos
            marker.points.append(
                Point(x=float(world[0]), y=float(world[1]), z=float(world[2])))

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        return marker_array

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])


def main(args=None):
    rclpy.init(args=args)
    node = OrionLiteRosNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
