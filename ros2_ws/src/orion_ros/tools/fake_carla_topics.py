#!/usr/bin/env python3
"""Publish synthetic CARLA topics (6 cameras, odometry, speed, IMU) so the ORION
node runs real inferences without the simulator. Smoke test only: no route, so
the PID publishes no controls.

  docker exec orion_ros bash -c "source /opt/ros/humble/setup.bash;
      source /opt/orion_ws/install/setup.bash;
      python3 /benchmarking/alpamayo-autoware/src/orion_ros/tools/fake_carla_topics.py 45"
"""
import os, sys, time, numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
rclpy.init(); n = Node("fake_carla_pub")
_FRAMES = os.environ.get("VLA_FRAMES_DIR", "/benchmarking/imgdiag")
img = cv2.resize(cv2.imread(os.path.join(_FRAMES, "frame_0.png")), (1600, 900))
bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
pubs = {c: n.create_publisher(Image, f"/carla/hero/{c}/image_decompressed", 5) for c in CAMS}
odom_pub = n.create_publisher(Odometry, "/carla/hero/odometry", 10)
spd_pub = n.create_publisher(Float32, "/carla/hero/speed", 10)
imu_pub = n.create_publisher(Imu, "/carla/hero/imu", 10)
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
hz = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
msgs = {}
for i, c in enumerate(CAMS):
    m = Image(); m.header.frame_id = c
    m.height, m.width, m.encoding, m.step = 900, 1600, "bgra8", 1600 * 4
    m.data = np.ascontiguousarray(np.roll(bgra, 5 * i, axis=1)).tobytes()
    msgs[c] = m
t0 = time.time(); k = 0
while time.time() - t0 < dur:
    stamp = n.get_clock().now().to_msg()
    for c in CAMS:
        msgs[c].header.stamp = stamp
        pubs[c].publish(msgs[c])
    o = Odometry(); o.header.stamp = stamp; o.header.frame_id = "map"
    o.pose.pose.position.x = 0.5 * k; o.pose.pose.orientation.w = 1.0; o.twist.twist.linear.x = 2.5
    odom_pub.publish(o); spd_pub.publish(Float32(data=2.5))
    im = Imu(); im.header.stamp = stamp; im.orientation.w = 1.0; imu_pub.publish(im)
    k += 1; rclpy.spin_once(n, timeout_sec=0.0); time.sleep(max(0.0, (t0 + (k / hz)) - time.time()))
n.destroy_node(); rclpy.shutdown(); print("published", k, "ticks")
