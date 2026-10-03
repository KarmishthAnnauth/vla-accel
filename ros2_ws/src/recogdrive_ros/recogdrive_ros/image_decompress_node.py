#!/usr/bin/env python3

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image


class ImageDecompressNode(Node):
    def __init__(self):
        super().__init__('image_decompress_node')
        self.declare_parameter('in_topic',  '/carla/hero/CAM_FRONT/image/compressed')
        self.declare_parameter('out_topic', '/carla/hero/CAM_FRONT/image_decompressed')
        in_topic  = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value

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
        out = Image()
        out.header = msg.header
        out.height, out.width = bgr.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = bgr.tobytes()
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
