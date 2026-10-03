#!/usr/bin/env python3
"""Publish synthetic CARLA topics (front camera, odometry, speed, IMU and a
latched CarlaRoute) so recogdrive_node runs real inferences and its controller
publishes commands without the simulator.  Smoke test only.

  # in the node's environment (start_recogdrive.sh --shell), same ROS_DOMAIN_ID:
  python3 tools/fake_carla_topics.py 60 [cam_hz] [route_id]
  python3 tools/control_listener.py 20        # what the PID is commanding

Publishes the camera straight to the *_decompressed topic the node reads
(bypassing the JPEG hop) at `cam_hz`, and odometry + IMU at 20 Hz like
carla_ros_bridge.  The ego drives at 2.5 m/s along +x; the route's road options
turn from lane-follow to LEFT 40 m in, which must show up in the node log as
cmd=left.  A second `route_id` publishes a different route (1000 m away), which
must show up as a NEW route and an ego-history reset.
"""
import sys
import time

import cv2
import numpy as np
import rclpy
from carla_msgs.msg import CarlaRoute
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu

CAMERA = "CAM_FRONT"
WIDTH, HEIGHT = 1920, 1080
SPEED = 2.5
ODOM_HZ = 20.0

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
cam_hz = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
route_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0

rclpy.init()
n = Node("fake_carla_pub")
src = "/benchmarking/imgdiag/frame_0.png"
img = cv2.imread(src)
if img is None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
img = cv2.resize(img, (WIDTH, HEIGHT))
bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])

cam_pub = n.create_publisher(Image, f"/carla/hero/{CAMERA}/image_decompressed", 5)
odom_pub = n.create_publisher(Odometry, "/carla/hero/odometry", 10)
imu_pub = n.create_publisher(Imu, "/carla/hero/imu", 10)
from std_msgs.msg import Float32
speed_pub = n.create_publisher(Float32, "/carla/hero/speed", 10)
route_pub = n.create_publisher(
    CarlaRoute, "/carla/hero/global_plan",
    QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
               depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

route = CarlaRoute()
x0 = 1000.0 * route_id
for i in range(60):
    p = Pose()
    p.position.x = x0 + 2.0 * i
    p.position.y = 0.0
    p.orientation.w = 1.0
    route.poses.append(p)
    route.road_options.append(4 if i < 20 else 1)
route_pub.publish(route)

cam_msg = Image()
cam_msg.header.frame_id = CAMERA
cam_msg.height, cam_msg.width, cam_msg.encoding, cam_msg.step = HEIGHT, WIDTH, "bgra8", WIDTH * 4
cam_msg.data = bgra.tobytes()

t0 = time.time()
k = 0
frames = 0
next_cam = 0.0
while time.time() - t0 < dur:
    elapsed = time.time() - t0
    stamp = n.get_clock().now().to_msg()
    o = Odometry()
    o.header.stamp = stamp
    o.header.frame_id = "map"
    o.child_frame_id = "hero"
    o.pose.pose.position.x = x0 + SPEED * elapsed
    o.pose.pose.orientation.w = 1.0
    o.twist.twist.linear.x = SPEED
    odom_pub.publish(o)
    im = Imu()
    im.header.stamp = stamp
    im.orientation.w = 1.0
    im.linear_acceleration.z = 9.81
    imu_pub.publish(im)
    speed_pub.publish(Float32(data=float(SPEED)))
    if elapsed >= next_cam:
        cam_msg.header.stamp = stamp
        cam_pub.publish(cam_msg)
        frames += 1
        next_cam += 1.0 / cam_hz
    k += 1
    rclpy.spin_once(n, timeout_sec=0.0)
    time.sleep(max(0.0, (t0 + (k / ODOM_HZ)) - time.time()))
n.destroy_node()
rclpy.shutdown()
print("published", k, "odometry ticks,", frames, "camera frames")
