#!/usr/bin/env python3
"""Lateral Stanley controller: subscribes to odometry + SimLingo trajectory.

When a trajectory arrives (base_link frame), it is immediately transformed to
the map frame using the current odometry pose.  The control loop then runs at a
fixed rate (control_hz), re-evaluating the closest segment and recomputing
steer/speed from the *current* odometry pose every cycle.
"""

from __future__ import annotations

import math
from typing import List, Optional

import rclpy
from ackermann_msgs.msg import AckermannDrive
from autoware_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
from rclpy.node import Node


class StanleyControllerNode(Node):

    def __init__(self) -> None:
        super().__init__("stanley_controller")

        self.declare_parameter("trajectory_topic",       "/simlingo/predicted_trajectory")
        self.declare_parameter("odometry_topic",         "/carla/hero/odometry")
        self.declare_parameter("control_output_topic",   "/carla/hero/ackermann_cmd")
        self.declare_parameter("stanley_gain_k",         0.8)
        self.declare_parameter("stanley_softening_ks",   0.5)
        self.declare_parameter("wheelbase_m",            2.875)
        self.declare_parameter("max_steering_rad",       0.6)
        self.declare_parameter("speed_lookahead_steps",  5)
        self.declare_parameter("control_hz",             20.0)
        self.declare_parameter("max_acceleration_mps2",  3.0)
        self.declare_parameter("trajectory_timeout_sec", 8.0)

        self._k          = float(self.get_parameter("stanley_gain_k").value)
        self._ks         = float(self.get_parameter("stanley_softening_ks").value)
        self._wheelbase  = float(self.get_parameter("wheelbase_m").value)
        self._max_steer  = float(self.get_parameter("max_steering_rad").value)
        self._lookahead  = int(self.get_parameter("speed_lookahead_steps").value)
        self._max_accel  = float(self.get_parameter("max_acceleration_mps2").value)
        self._traj_timeout = float(self.get_parameter("trajectory_timeout_sec").value)
        control_hz       = float(self.get_parameter("control_hz").value)

        self._log_counter = 0
        self._log_every   = max(1, int(control_hz // 2))

        self._latest_odom: Optional[Odometry] = None
        self._traj_xs:     List[float] = []
        self._traj_ys:     List[float] = []
        self._traj_speeds: List[float] = []
        self._last_traj_sec: Optional[float] = None

        odom_topic = str(self.get_parameter("odometry_topic").value)
        self.create_subscription(Odometry, odom_topic, self._odometry_cb, 10)
        self.get_logger().info(f"Subscribed to odometry: {odom_topic}")

        traj_topic = str(self.get_parameter("trajectory_topic").value)
        self.create_subscription(Trajectory, traj_topic, self._trajectory_cb, 10)
        self.get_logger().info(f"Subscribed to trajectory: {traj_topic}")

        out_topic = str(self.get_parameter("control_output_topic").value)
        self._pub = self.create_publisher(AckermannDrive, out_topic, 10)
        self.get_logger().info(f"Publishing AckermannDrive on {out_topic}")

        self.create_timer(1.0 / control_hz, self._control_cb)
        self.get_logger().info(f"Stanley controller ready at {control_hz:.0f} Hz.")


    def _odometry_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _trajectory_cb(self, msg: Trajectory) -> None:
        """Convert trajectory from base_link to map frame and cache it."""
        if self._latest_odom is None:
            self.get_logger().warn("No odometry yet — dropping trajectory.")
            return
        if len(msg.points) < 2:
            return

        pose    = self._latest_odom.pose.pose
        tx, ty  = pose.position.x, pose.position.y
        yaw     = _yaw_from_quaternion(pose.orientation)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        xs, ys, speeds = [], [], []
        for p in msg.points:
            x_bl, y_bl = p.pose.position.x, p.pose.position.y
            xs.append(tx + x_bl * cos_y - y_bl * sin_y)
            ys.append(ty + x_bl * sin_y + y_bl * cos_y)
            speeds.append(float(p.longitudinal_velocity_mps))

        self._traj_xs     = xs
        self._traj_ys     = ys
        self._traj_speeds = speeds
        self._last_traj_sec = self.get_clock().now().nanoseconds * 1e-9


    def _control_cb(self) -> None:
        now_sec   = self.get_clock().now().nanoseconds * 1e-9
        traj_age  = (
            now_sec - self._last_traj_sec
            if self._last_traj_sec is not None
            else float("inf")
        )

        if traj_age > self._traj_timeout:
            stop = AckermannDrive()
            stop.speed        = 0.0
            stop.acceleration = -self._max_accel
            self._pub.publish(stop)
            return

        if self._latest_odom is None or len(self._traj_xs) < 2:
            return

        pose    = self._latest_odom.pose.pose
        tx, ty  = pose.position.x, pose.position.y
        yaw     = _yaw_from_quaternion(pose.orientation)

        front_x = tx + self._wheelbase * math.cos(yaw)
        front_y = ty + self._wheelbase * math.sin(yaw)

        xs, ys, speeds = self._traj_xs, self._traj_ys, self._traj_speeds

        min_dist    = float("inf")
        closest_seg = 0
        for i in range(len(xs) - 1):
            d = _seg_dist(front_x, front_y, xs[i], ys[i], xs[i + 1], ys[i + 1])
            if d < min_dist:
                min_dist    = d
                closest_seg = i

        i0, i1   = closest_seg, closest_seg + 1
        path_dx   = xs[i1] - xs[i0]
        path_dy   = ys[i1] - ys[i0]
        path_hdg  = math.atan2(path_dy, path_dx)

        heading_err = math.atan2(
            math.sin(path_hdg - yaw), math.cos(path_hdg - yaw)
        )

        seg_len = math.hypot(path_dx, path_dy)
        if seg_len > 1e-6:
            cte = (
                math.sin(path_hdg) * (front_x - xs[i0])
                - math.cos(path_hdg) * (front_y - ys[i0])
            )
        else:
            cte = 0.0

        speed_idx = min(i0 + self._lookahead, len(speeds) - 1)
        speed     = max(speeds[speed_idx], 0.0)

        steer = heading_err + math.atan2(self._k * cte, speed + self._ks)
        steer = max(-self._max_steer, min(self._max_steer, steer))

        self._log_counter += 1
        if self._log_counter % self._log_every == 0:
            self.get_logger().info(
                f"seg={closest_seg}/{len(xs) - 1}  "
                f"heading_err={math.degrees(heading_err):.1f}°  "
                f"cte={cte:.3f} m  "
                f"steer={math.degrees(steer):.1f}°  "
                f"speed={speed:.2f} m/s"
            )

        cmd = AckermannDrive()
        cmd.steering_angle = float(steer)
        cmd.speed          = float(speed)
        cmd.acceleration   = self._max_accel
        self._pub.publish(cmd)


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _seg_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy   = bx - ax, by - ay
    seg_sq   = dx * dx + dy * dy
    if seg_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StanleyControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
