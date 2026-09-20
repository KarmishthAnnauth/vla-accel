"""
B 机 simlingo inference client (ROS bridge variant).

This is a ROS-side adaptation of simlingo/team_code/agent_simlingo.py:LingoAgent.
Instead of receiving sensor data via the leaderboard's `input_data` dict, we
subscribe to the topics that A's MyROS2Agent + carla_ros_bridge publish.

Pipeline (per simlingo):
    1) Subscribe sensor topics from A  over rosbridge (websocket).
    2) Build a `tick_data` dict that matches what simlingo's tick() expects.
    3) Run simlingo's tick (UKF + InternVL2 image preprocess + route_planner +
       command + target_point) to populate self.DrivingInput.
    4) Forward through the simlingo model -> (pred_speed_wps, pred_route, language).
    5) Run simlingo's control_pid -> (steer, throttle, brake).
    6) Publish CarlaEgoVehicleControl back to A with header.stamp = latest image stamp.

This file imports simlingo's own modules; PYTHONPATH on B  must contain
simlingo's repo root so that `team_code.*` and `simlingo_training.*` resolve.

deps (B ):
    pip install roslibpy numpy pillow opencv-python torch transformers \\
                hydra-core omegaconf filterpy scipy ujson
"""

from __future__ import annotations

import base64
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import roslibpy

import torch
import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from PIL import Image
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from transformers import AutoProcessor

import team_code.transfuser_utils as t_u
from team_code.config_simlingo import GlobalConfig
from team_code.nav_planner import RoutePlanner
from simlingo_training.utils.custom_types import DrivingInput
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

sys.path.insert(0, os.environ.get("BENCH2DRIVE_LEADERBOARD", ""))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

ROSBRIDGE_HOST = os.environ.get("ROSBRIDGE_HOST", "192.0.2.30")
ROSBRIDGE_PORT = int(os.environ.get("ROSBRIDGE_PORT", "9090"))

CKPT_PATH = os.environ.get("CKPT_PATH", "/model/simlingo/checkpoints/epoch=013.ckpt")


def bicycle_model_forward(x, dt, steer, throttle, brake):
    front_wb = -0.090769015
    rear_wb = 1.4178275
    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837
    locs_0, locs_1, yaw, speed = x[0], x[1], x[2], x[3]
    accel = brake_accel if brake else throt_accel * throttle
    wheel = steer_gain * steer
    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)
    return np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

def measurement_function_hx(vehicle_state):
    return vehicle_state

def state_mean(state, wm):
    x = np.zeros(4)
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(np.sum(np.dot(np.sin(state[:, 2]), wm)),
                      np.sum(np.dot(np.cos(state[:, 2]), wm)))
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x

def measurement_mean(state, wm):
    x = np.zeros(4)
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(np.sum(np.dot(np.sin(state[:, 2]), wm)),
                      np.sum(np.dot(np.cos(state[:, 2]), wm)))
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x

def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y

def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y

T_IMAGE   = "/carla/hero/rgb_0/image"
T_IMU     = "/carla/hero/imu"
T_GPS     = "/carla/hero/gps"
T_SPEED   = "/carla/hero/speed"
T_PLAN    = "/carla/hero/global_plan"
T_PLANGN  = "/carla/hero/global_plan_gnss"
T_STATUS  = "/carla/hero/status"
T_CTRL    = "/carla/hero/vehicle_control_cmd"


def decode_image_msg(msg) -> np.ndarray:
    """sensor_msgs/Image -> H,W,3 uint8 BGR (matches simlingo: it reads
    input_data['rgb_0'][1][:, :, :3] which is BGRA from CARLA -> takes BGR)."""
    raw = base64.b64decode(msg["data"])
    h, w = msg["height"], msg["width"]
    enc = msg["encoding"]
    if enc == "bgra8":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, :3].copy()
    if enc == "rgb8":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        return arr[:, :, ::-1].copy()
    if enc == "bgr8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
    raise ValueError(f"unsupported encoding {enc}")


def imu_to_compass(msg) -> float:
    """sensor_msgs/Imu's orientation is the ego compass yaw (rad).
    simlingo: t_u.preprocess_compass(input_data['imu'][1][-1])
    leaderboard's imu sensor returns 7-vector with index -1 being the compass.
    With ros bridge we have to recover compass from quaternion (z-axis yaw)."""
    q = msg["orientation"]
    siny_cosp = 2 * (q["w"] * q["z"] + q["x"] * q["y"])
    cosy_cosp = 1 - 2 * (q["y"] * q["y"] + q["z"] * q["z"])
    yaw = math.atan2(siny_cosp, cosy_cosp)
    yaw = -yaw
    return t_u.preprocess_compass(yaw)


class SimlingoROSClient:
    """Mirrors LingoAgent but driven by ROS subscriptions instead of leaderboard."""

    def __init__(self, ckpt_path: str):
        self.config = GlobalConfig()
        self.device = torch.device("cuda")
        self.step = -1
        self.initialized = False

        cfg_path = Path(ckpt_path).parent.parent.parent / ".hydra" / "config.yaml"
        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.model.vision_model.use_global_img = self.cfg.data_module.use_global_img

        processor = AutoProcessor.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
        self.tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        self.tokenizer.add_special_tokens({"additional_special_tokens": [
            "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
            "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>"
        ]})
        self.tokenizer.padding_side = "left"

        cache_dir = f"pretrained/{self.cfg.model.vision_model.variant.split('/')[1]}"
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            cfg_data_module=self.cfg.data_module,
            processor=processor,
            cache_dir=cache_dir,
            _recursive_=False,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)
        weights_path = Path(ckpt_path)
        if weights_path.is_dir():
            weights_path = weights_path / "pytorch_model.pt"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        if self.config.eval_route_as == -1:
            self.config.eval_route_as = self.model.route_as

        self.speed_controller = t_u.PIDController(
            k_p=self.config.speed_kp, k_i=self.config.speed_ki,
            k_d=self.config.speed_kd, n=self.config.speed_n,
        )
        from team_code.nav_planner import LateralPIDController
        self.turn_controller = LateralPIDController(inference_mode=False)

        self.points = MerweScaledSigmaPoints(n=4, alpha=1e-5, beta=2, kappa=0,
                                             subtract=residual_state_x)
        self.ukf = UKF(dim_x=4, dim_z=4, fx=bicycle_model_forward,
                       hx=measurement_function_hx,
                       dt=self.config.carla_frame_rate,
                       points=self.points,
                       x_mean_fn=state_mean, z_mean_fn=measurement_mean,
                       residual_x=residual_state_x,
                       residual_z=residual_measurement_h)
        self.ukf.P = np.diag([0.5, 0.5, 1e-6, 1e-6])
        self.ukf.R = np.diag([0.5, 0.5, 1e-15, 1e-15])
        self.ukf.Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
        self.filter_initialized = False

        self.commands = deque(maxlen=2); self.commands.append(4); self.commands.append(4)
        self.target_point_prev = np.array([1e5, 1e5, 1e5])
        self.image_buffer = deque(maxlen=1)
        self.last_command = -1
        self.last_command_tmp = -1
        self.control = type("VC", (), {"steer": 0.0, "throttle": 0.0, "brake": 0.0})()
        self.stuck_detector = 0
        self.force_move = 0
        self.T = 1
        self.DrivingInput: dict = {}

        self._lock = threading.Lock()
        self._latest = {
            "image": None, "stamp": None,
            "imu":   None,
            "gps":   None,
            "speed": None,
        }
        self._route_set = False
        self._route_planner: RoutePlanner | None = None
        self.lat_ref = 0.0
        self.lon_ref = 0.0

    def connect(self):
        self.client = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
        self.client.run()
        print(f"[B] connected ws://{ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}")

        roslibpy.Topic(self.client, T_IMAGE, "sensor_msgs/Image",
                       queue_length=1).subscribe(self._on_image)
        roslibpy.Topic(self.client, T_IMU,   "sensor_msgs/Imu",
                       queue_length=1).subscribe(self._on_imu)
        roslibpy.Topic(self.client, T_GPS,   "sensor_msgs/NavSatFix",
                       queue_length=1).subscribe(self._on_gps)
        roslibpy.Topic(self.client, T_SPEED, "std_msgs/Float32",
                       queue_length=1).subscribe(self._on_speed)
        roslibpy.Topic(self.client, T_PLANGN, "carla_msgs/CarlaGnssRoute",
                       queue_length=1).subscribe(self._on_route_gnss)

        self.ctrl_pub = roslibpy.Topic(self.client, T_CTRL,
                                       "carla_msgs/CarlaEgoVehicleControl",
                                       queue_length=1)
        self.ctrl_pub.advertise()

    def _on_image(self, msg):
        try:
            img = decode_image_msg(msg)
        except Exception as e:
            print("[B] image decode failed:", e); return
        with self._lock:
            self._latest["image"] = img
            self._latest["stamp"] = msg["header"]["stamp"]

    def _on_imu(self, msg):
        with self._lock:
            self._latest["imu"] = msg

    def _on_gps(self, msg):
        with self._lock:
            self._latest["gps"] = np.array([msg["latitude"], msg["longitude"], msg["altitude"]])

    def _on_speed(self, msg):
        with self._lock:
            self._latest["speed"] = float(msg["data"])

    def _on_route_gnss(self, msg):
        """Build simlingo-compatible global_plan_gps from carla_msgs/CarlaGnssRoute and
        seed self._route_planner.  We only do this once."""
        if self._route_set:
            return
        coords  = msg.get("coordinates", [])
        options = msg.get("road_options", [])
        if not coords or not options:
            return
        global_plan_gps = []
        for c, o in zip(coords, options):
            global_plan_gps.append(({"lat": c["latitude"],
                                     "lon": c["longitude"],
                                     "z":   c["altitude"]}, o))

        self.lat_ref = coords[0]["latitude"]
        self.lon_ref = coords[0]["longitude"]
        self._route_planner = RoutePlanner(
            min_distance=self.config.route_planner_min_distance,
            max_distance=self.config.route_planner_max_distance,
            lat_ref=self.lat_ref, lon_ref=self.lon_ref,
        )
        self._route_planner.set_route(global_plan_gps, gps=True)
        self._route_set = True
        print(f"[B] route loaded ({len(coords)} pts), lat_ref={self.lat_ref:.6f} lon_ref={self.lon_ref:.6f}")

    @torch.inference_mode()
    def tick(self, image_bgr: np.ndarray, imu_msg, gps: np.ndarray, speed: float):
        _, buf = cv2.imencode(".jpg", image_bgr)
        camera = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        rgb_pos = rgb_pos[: int(rgb_pos.shape[0] - (rgb_pos.shape[0] * 4.8) // 16), :, :]
        rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
        rgb = np.array([rgb_pos])
        self.image_buffer.append(rgb)

        T_, C, H, W = rgb.shape
        transform = build_transform(input_size=448)
        image_pil = Image.fromarray(rgb.squeeze(0).transpose(1, 2, 0))
        images = dynamic_preprocess(image_pil, image_size=448,
                                    use_thumbnail=self.cfg.model.vision_model.use_global_img,
                                    max_num=2)
        pixel_values = torch.stack([transform(im) for im in images])
        processed = pixel_values.unsqueeze(0)
        num_patches, h, w = processed.shape[1], processed.shape[3], processed.shape[4]
        processed = processed.view(1, self.T, num_patches, C, h, w)

        gps_pos = self._route_planner.convert_gps_to_carla(gps)
        compass = imu_to_compass(imu_msg)

        if not self.filter_initialized:
            self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
            self.filter_initialized = True
        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
        self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
        gps_filtered = self.ukf.x[0:2]

        wp_route = self._route_planner.run_step(np.append(gps_filtered, gps_pos[2]))
        if   len(wp_route) > 2:
            target_point, far_command           = wp_route[1]
            next_target_point, next_far_command = wp_route[2]
        elif len(wp_route) > 1:
            target_point, far_command           = wp_route[1]
            next_target_point, next_far_command = wp_route[1]
        else:
            target_point, far_command           = wp_route[0]
            next_target_point, next_far_command = wp_route[0]

        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp
        self.last_command_tmp = far_command
        if (np.asarray(target_point) != self.target_point_prev).all():
            self.target_point_prev = np.asarray(target_point)
            self.commands.append(getattr(far_command, "value", far_command))

        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        command_t = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target_point      = t_u.inverse_conversion_2d(target_point[:2],      gps_filtered, compass)
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], gps_filtered, compass)
        target_point_t = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

        target_points_np = np.array([ego_target_point, ego_next_target_point])
        route_t = torch.from_numpy(target_points_np).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.DrivingInput = {
            "camera_images":  processed.to(self.device, dtype=torch.bfloat16),
            "target_point":   target_point_t,
            "route":          route_t,
            "command":        command_t,
        }
        return speed

    def control_pid(self, pred_route, velocity, pred_speed_wps):
        """Direct copy of agent_simlingo.control_pid."""
        from scipy.interpolate import PchipInterpolator
        route_waypoints = pred_route[0].data.cpu().numpy()
        speed = float(velocity)
        speed_waypoints = pred_speed_wps[0].data.cpu().numpy()

        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0

        brake = (desired_speed < self.config.brake_speed) or ((speed / max(desired_speed, 1e-3)) > self.config.brake_ratio)
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = float(np.clip(throttle, 0.0, self.config.clip_throttle))
        if brake:
            throttle = 0.0

        wp = route_waypoints.copy()
        wp = np.concatenate((np.zeros_like(wp[:1]), wp))
        shift = np.roll(wp, 1, axis=0); shift[0] = shift[1]
        d = np.linalg.norm(wp - shift, axis=1); d = np.cumsum(d)
        d += np.arange(0, len(d)) * 1e-4
        interp = PchipInterpolator(d, wp, axis=0)
        x = np.arange(0.1, d[-1], 0.1)
        route_interp = interp(x) if len(x) else wp[None, -1]

        steer = self.turn_controller.step(route_interp, speed)
        steer = float(np.clip(steer, -1.0, 1.0))
        return round(steer, 3), throttle, bool(brake)

    def run_once(self):
        with self._lock:
            img    = self._latest["image"]
            stamp  = self._latest["stamp"]
            imu    = self._latest["imu"]
            gps    = self._latest["gps"]
            speed  = self._latest["speed"]
            self._latest["stamp"] = None
        if img is None or stamp is None or imu is None or gps is None or speed is None:
            return False
        if not self._route_set:
            return False

        self.step += 1

        if self.step < self.config.inital_frames_delay:
            self._publish_control(stamp, 0.0, 0.0, 1.0)
            return True

        self.tick(img, imu, gps, speed)

        model_input = DrivingInput(**self.DrivingInput)
        pred_speed_wps, pred_route, language = self.model(model_input)

        steer, throttle, brake = self.control_pid(
            pred_route.float() if pred_route is not None else None,
            speed,
            pred_speed_wps.float() if pred_speed_wps is not None else None,
        )

        if speed < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration
        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        self.control.steer    = float(steer)
        self.control.throttle = float(throttle)
        self.control.brake    = float(brake)
        self._publish_control(stamp, steer, throttle, brake)
        return True

    def _publish_control(self, stamp, steer, throttle, brake):
        self.ctrl_pub.publish(roslibpy.Message({
            "header": {"stamp": stamp, "frame_id": "ego_vehicle"},
            "throttle":          float(throttle),
            "steer":             float(steer),
            "brake":             float(brake),
            "hand_brake":        False,
            "reverse":           False,
            "gear":              1,
            "manual_gear_shift": False,
        }))


def main():
    cli = SimlingoROSClient(CKPT_PATH)
    cli.connect()
    print("[B] waiting for first sensor frame + global plan ...")
    try:
        while cli.client.is_connected:
            if not cli.run_once():
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        cli.ctrl_pub.unadvertise()
        cli.client.terminate()


if __name__ == "__main__":
    main()
