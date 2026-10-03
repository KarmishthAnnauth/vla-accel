#!/usr/bin/env python3
"""Listen to /carla/hero/vehicle_control_cmd for N seconds and summarise what
recogdrive_node (control_mode=pid) is commanding.  Smoke test only.

  python3 control_listener.py [seconds]
"""
import sys
import time

import numpy as np
import rclpy
from carla_msgs.msg import CarlaEgoVehicleControl
from rclpy.node import Node

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
rclpy.init()
n = Node("control_listener")
rows = []
n.create_subscription(CarlaEgoVehicleControl, "/carla/hero/vehicle_control_cmd",
                      lambda m: rows.append((time.time(), m.steer, m.throttle, m.brake)), 50)
t_end = time.time() + dur
while time.time() < t_end:
    rclpy.spin_once(n, timeout_sec=0.1)
if not rows:
    print("no CarlaEgoVehicleControl received")
else:
    a = np.array(rows)
    span = a[-1, 0] - a[0, 0]
    gaps = np.diff(a[:, 0])
    print(f"{len(a)} commands in {span:.1f} s = {(len(a) - 1) / max(span, 1e-9):.1f} Hz (max gap {gaps.max() * 1e3:.0f} ms)")
    for name, col in (("steer", 1), ("throttle", 2), ("brake", 3)):
        print(f"  {name:8s} min {a[:, col].min():+.3f}  mean {a[:, col].mean():+.3f}  max {a[:, col].max():+.3f}")
    print("  last 5:", " | ".join(f"s={r[1]:+.3f} t={r[2]:.2f} b={r[3]:.2f}" for r in a[-5:]))
n.destroy_node()
rclpy.shutdown()
