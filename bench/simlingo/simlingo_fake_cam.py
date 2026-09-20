"""Synthetic CARLA image publisher: compressed RGB at 10 Hz, matching the real stream."""
import sys, time, numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
rclpy.init()
n = Node("fake_carla_cam")
qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
pub = n.create_publisher(CompressedImage, "/carla/hero/rgb_0/compressed", qos)
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (600, 1024, 3), dtype=np.uint8)
ok, buf = cv2.imencode(".jpg", frame)
payload = buf.tobytes()
print(f"publishing {len(payload)/1024:.0f} KB jpeg at {HZ} Hz", flush=True)
def tick():
    m = CompressedImage()
    t = n.get_clock().now().to_msg()
    m.header.stamp = t; m.header.frame_id = "hero"; m.format = "jpeg"; m.data = payload
    pub.publish(m)
n.create_timer(1.0/HZ, tick)
try: rclpy.spin(n)
except KeyboardInterrupt: pass
