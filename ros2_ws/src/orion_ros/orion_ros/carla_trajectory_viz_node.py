#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from visualization_msgs.msg import Marker, MarkerArray

try:
    import carla
    HAS_CARLA_PYTHON_API = True
except ImportError:
    carla = None
    HAS_CARLA_PYTHON_API = False


class CarlaTrajectoryVizNode(Node):
    def __init__(self):
        super().__init__('carla_trajectory_viz_node')

        self.declare_parameter('marker_topic', '/orion_lite/predicted_trajectory_markers')
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout_sec', 10.0)
        self.declare_parameter('draw_life_time', 1.0)
        self.declare_parameter('point_size', 0.1)
        self.declare_parameter('line_thickness', 0.1)
        self.declare_parameter('z_offset', 0.3)
        self.declare_parameter('color_r', 255)
        self.declare_parameter('color_g', 0)
        self.declare_parameter('color_b', 0)
        self.declare_parameter('draw_points', False)

        self._host         = self.get_parameter('carla_host').value
        self._port         = int(self.get_parameter('carla_port').value)
        self._timeout      = float(self.get_parameter('carla_timeout_sec').value)
        self._life_time    = float(self.get_parameter('draw_life_time').value)
        self._point_size   = float(self.get_parameter('point_size').value)
        self._line_thick   = float(self.get_parameter('line_thickness').value)
        self._z_offset     = float(self.get_parameter('z_offset').value)
        self._draw_points  = bool(self.get_parameter('draw_points').value)
        cr = int(self.get_parameter('color_r').value)
        cg = int(self.get_parameter('color_g').value)
        cb = int(self.get_parameter('color_b').value)
        self._override_color = None
        if min(cr, cg, cb) >= 0:
            clamp = lambda v: max(0, min(255, v))
            self._override_color = carla.Color(clamp(cr), clamp(cg), clamp(cb)) \
                if HAS_CARLA_PYTHON_API else True

        self._world = None

        if not HAS_CARLA_PYTHON_API:
            self.get_logger().error(
                'The `carla` Python package is not importable. This node must run '
                'in an environment with the CARLA Python API installed. Markers '
                'will be received but NOT drawn.')
        else:
            self._connect_to_carla()
            if self._world is None:
                self._retry_timer = self.create_timer(2.0, self._retry_connect)

        marker_topic = self.get_parameter('marker_topic').value
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(MarkerArray, marker_topic, self._marker_cb, qos)
        self.get_logger().info(f'Subscribed to markers: {marker_topic}')


    def _connect_to_carla(self):
        try:
            client = carla.Client(self._host, self._port)
            client.set_timeout(self._timeout)
            self._world = client.get_world()
            self.get_logger().info(
                f'Connected to CARLA at {self._host}:{self._port}.')
        except Exception as e:
            self._world = None
            self.get_logger().warn(
                f'Could not connect to CARLA at {self._host}:{self._port} ({e}); '
                'will retry.')

    def _retry_connect(self):
        if self._world is not None:
            self._retry_timer.cancel()
            return
        self._connect_to_carla()
        if self._world is not None:
            self._retry_timer.cancel()


    def _marker_cb(self, msg: MarkerArray):
        if self._world is None:
            self.get_logger().warn('No CARLA connection yet; dropping markers.',
                                   throttle_duration_sec=5.0)
            return

        debug = self._world.debug
        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue
            if marker.type not in (Marker.LINE_STRIP, Marker.LINE_LIST, Marker.POINTS):
                continue

            color = self._override_color if self._override_color is not None \
                else self._to_carla_color(marker.color)
            locs = [self._to_carla_location(p) for p in marker.points]
            if not locs:
                continue

            if self._draw_points:
                for loc in locs:
                    debug.draw_point(loc, size=self._point_size, color=color,
                                     life_time=self._life_time)

            if marker.type == Marker.LINE_STRIP:
                for a, b in zip(locs[:-1], locs[1:]):
                    debug.draw_line(a, b, thickness=self._line_thick, color=color,
                                    life_time=self._life_time)
            elif marker.type == Marker.LINE_LIST:
                for i in range(0, len(locs) - 1, 2):
                    debug.draw_line(locs[i], locs[i + 1], thickness=self._line_thick,
                                    color=color, life_time=self._life_time)

    def _to_carla_location(self, p) -> 'carla.Location':
        return carla.Location(x=float(p.x), y=-float(p.y), z=float(p.z) + self._z_offset)

    @staticmethod
    def _to_carla_color(c) -> 'carla.Color':
        scale = 255.0 if max(c.r, c.g, c.b) <= 1.0 else 1.0
        clamp = lambda v: max(0, min(255, int(round(v * scale))))
        return carla.Color(clamp(c.r), clamp(c.g), clamp(c.b))


def main(args=None):
    rclpy.init(args=args)
    node = CarlaTrajectoryVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
