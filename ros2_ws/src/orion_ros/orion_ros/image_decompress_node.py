#!/usr/bin/env python3

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageDecompressNode(Node):
    def __init__(self):
        super().__init__('image_decompress_node')
        self.declare_parameter('in_topic',  '/carla/hero/rgb_0/image/compressed')
        self.declare_parameter('out_topic', '/carla/hero/rgb_0/image_decompressed')
        in_topic  = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value

        self.bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self._pub = self.create_publisher(Image, out_topic, qos)
        self.create_subscription(CompressedImage, in_topic, self._cb, qos)
        self.get_logger().info(f'Decompressing {in_topic} -> {out_topic} (bgr8)')

    def _cb(self, msg: CompressedImage):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn('JPEG decode failed', throttle_duration_sec=5.0)
            return
        out = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        out.header = msg.header
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImageDecompressNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
