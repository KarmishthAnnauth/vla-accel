#!/usr/bin/env python3
"""Publish synthetic CARLA topics (6 cameras, odometry, speed, IMU, and a
latched CarlaRoute) so minddrive_node runs real inferences AND publishes
controls without the simulator.  Smoke test only.

  docker exec minddrive_ros bash -c "source /opt/ros/humble/setup.bash;
      source /opt/minddrive_ws/install/setup.bash;
      python3 /benchmarking/alpamayo-autoware/src/minddrive_ros/tools/fake_carla_topics.py 60 [hz] [route_id]"

Publishes straight to the *_decompressed topics the node reads (bypassing the
JPEG hop).  A second `route_id` publishes a different route mid-run, which
must show up in the node log as a context reset.
"""
import os
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
from std_msgs.msg import Float32

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
hz = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
route_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0

rclpy.init()
n = Node("fake_carla_pub")
src = os.path.join(os.environ.get("VLA_FRAMES_DIR", "/benchmarking/imgdiag"),
                   "frame_0.png")
img = cv2.imread(src)
if img is None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8)
img = cv2.resize(img, (1600, 900))
bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
pubs = {c: n.create_publisher(Image, f"/carla/hero/{c}/image_decompressed", 5) for c in CAMS}
odom_pub = n.create_publisher(Odometry, "/carla/hero/odometry", 10)
spd_pub = n.create_publisher(Float32, "/carla/hero/speed", 10)
imu_pub = n.create_publisher(Imu, "/carla/hero/imu", 10)
route_pub = n.create_publisher(
    CarlaRoute, "/carla/hero/global_plan",
    QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
               depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

route = CarlaRoute()
x0 = 1000.0 * route_id
for i in range(40):
    p = Pose()
    p.position.x = x0 + 2.0 * i
    p.position.y = 0.0
    p.orientation.w = 1.0
    route.poses.append(p)
    route.road_options.append(4)
route_pub.publish(route)

msgs = {}
for i, c in enumerate(CAMS):
    m = Image()
    m.header.frame_id = c
    m.height, m.width, m.encoding, m.step = 900, 1600, "bgra8", 1600 * 4
    m.data = np.ascontiguousarray(np.roll(bgra, 5 * i, axis=1)).tobytes()
    msgs[c] = m

t0 = time.time()
k = 0
while time.time() - t0 < dur:
    stamp = n.get_clock().now().to_msg()
    o = Odometry()
    o.header.stamp = stamp
    o.header.frame_id = "map"
    o.pose.pose.position.x = x0 + 2.5 * (time.time() - t0)
    o.pose.pose.orientation.w = 1.0
    o.twist.twist.linear.x = 2.5
    odom_pub.publish(o)
    spd_pub.publish(Float32(data=2.5))
    im = Imu()
    im.header.stamp = stamp
    im.orientation.w = 1.0
    imu_pub.publish(im)
    for c in CAMS:
        msgs[c].header.stamp = stamp
        pubs[c].publish(msgs[c])
    k += 1
    rclpy.spin_once(n, timeout_sec=0.0)
    time.sleep(max(0.0, (t0 + (k / hz)) - time.time()))
n.destroy_node()
rclpy.shutdown()
print("published", k, "ticks")
