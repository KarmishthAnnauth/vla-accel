from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch the Orion-Lite ROS 2 node on live CARLA ros-bridge topics.

    The node subscribes ONLY to the front camera and replicates that single
    view into all 6 ORION input streams. It runs the Orion-Lite model and its
    own PID + RoutePlanner closed-loop, publishing carla_msgs/CarlaEgoVehicleControl
    (steer/throttle/brake) directly on /carla/hero/vehicle_control_cmd — exactly
    like the leaderboard agent's carla.VehicleControl (no controller node needed).

    All model paths point at the Orion-Lite repo inside the container. Override
    any of them on the command line, e.g.:
      ros2 launch orion_ros orion_lite.launch.py \\
          ckpt_path:=/some/other/orion_lite.pth
    """
    repo_path_arg = DeclareLaunchArgument(
        "repo_path", default_value="/benchmarking/Orion-Lite")
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value="/benchmarking/Orion-Lite/configs/orion_lite_closedloop.py")
    ckpt_path_arg = DeclareLaunchArgument(
        "ckpt_path",
        default_value="/benchmarking/orion_lite_checkpoints/fused_ckpts/orion_lite.pth")

    compressed_topic_arg = DeclareLaunchArgument(
        "compressed_topic", default_value="/carla/hero/rgb_0/image/compressed")

    front_camera_topic_arg = DeclareLaunchArgument(
        "front_camera_topic", default_value="/carla/hero/rgb_0/image_decompressed")
    imu_topic_arg = DeclareLaunchArgument(
        "imu_topic", default_value="/carla/hero/imu")
    gnss_topic_arg = DeclareLaunchArgument(
        "gnss_topic", default_value="/carla/hero/gps")
    odometry_topic_arg = DeclareLaunchArgument(
        "odometry_topic", default_value="/carla/hero/odometry")
    speed_topic_arg = DeclareLaunchArgument(
        "speed_topic", default_value="/carla/hero/speed")
    route_topic_arg = DeclareLaunchArgument(
        "route_topic", default_value="/carla/hero/global_plan")
    control_topic_arg = DeclareLaunchArgument(
        "control_topic", default_value="/carla/hero/vehicle_control_cmd")

    image_decompress_node = Node(
        package="orion_ros",
        executable="image_decompress_node",
        name="image_decompress_node",
        output="screen",
        parameters=[
            {
                "in_topic":  LaunchConfiguration("compressed_topic"),
                "out_topic": LaunchConfiguration("front_camera_topic"),
            }
        ],
    )

    orion_lite_node = Node(
        package="orion_ros",
        executable="orion_lite_node",
        name="orion_lite_inference_node",
        output="screen",
        parameters=[
            {
                "repo_path":          LaunchConfiguration("repo_path"),
                "config_path":        LaunchConfiguration("config_path"),
                "ckpt_path":          LaunchConfiguration("ckpt_path"),
                "front_camera_topic": LaunchConfiguration("front_camera_topic"),
                "imu_topic":          LaunchConfiguration("imu_topic"),
                "gnss_topic":         LaunchConfiguration("gnss_topic"),
                "odometry_topic":     LaunchConfiguration("odometry_topic"),
                "speed_topic":        LaunchConfiguration("speed_topic"),
                "route_topic":        LaunchConfiguration("route_topic"),
                "control_topic":      LaunchConfiguration("control_topic"),
            }
        ],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}"),
            SetEnvironmentVariable("ORION_QFORMER_PATH", "/models/Orion/pretrain_qformer"),
            repo_path_arg,
            config_path_arg,
            ckpt_path_arg,
            compressed_topic_arg,
            front_camera_topic_arg,
            imu_topic_arg,
            gnss_topic_arg,
            odometry_topic_arg,
            speed_topic_arg,
            route_topic_arg,
            control_topic_arg,
            image_decompress_node,
            orion_lite_node,
        ]
    )
