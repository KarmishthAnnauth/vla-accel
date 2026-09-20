from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Launch the ORION node on live CARLA camera + odometry topics.

    NOTE: topic names are placeholders — set the real CARLA ROS bridge topic
    names together later. The node subscribes to all six ORION cameras and feeds
    each view into its own input slot, as OrionAgent.tick() does.
    """
    camera_topics = [
        "/carla/hero/CAM_FRONT/image",
        "/carla/hero/CAM_FRONT_LEFT/image",
        "/carla/hero/CAM_FRONT_RIGHT/image",
        "/carla/hero/CAM_BACK/image",
        "/carla/hero/CAM_BACK_LEFT/image",
        "/carla/hero/CAM_BACK_RIGHT/image",
    ]

    require_new_frame_arg = DeclareLaunchArgument(
        "require_new_frame",
        default_value="true",
        description="Gate inference on a fresh front frame (true) or run "
        "continuously on the latest data (false).",
    )

    link_ckpts = ExecuteProcess(
        cmd=["ln", "-sfn", "/models/Orion", "/root/Orion/ckpts"],
        output="screen",
    )
    orion_node = Node(
        package="orion_ros",
        executable="orion_node",
        name="orion_node",
        output="screen",
        parameters=[
            {
                        "orion_repo_path": "/root/Orion",
                        "orion_config_path":
                            "/root/Orion/adzoo/orion/configs/orion_stage3_agent.py",
                        "orion_checkpoint_path": "/models/Orion/Orion.pth",
                        "precision": "fp16",
                        "camera_topics": camera_topics,
                        "speed_topic": "/carla/hero/speed",
                        "odometry_topic": "/carla/hero/odometry",
                        "imu_topic": "/carla/hero/imu",
                        "route_topic": "/carla/hero/global_plan",
                        "trajectory_topic": "/orion/predicted_trajectory",
                        "cot_topic": "/orion/cot",
                        "inference_period_sec": 0.05,
                        "require_new_frame": ParameterValue(
                            LaunchConfiguration("require_new_frame"), value_type=bool
                        ),
                        "publish_cot": True,
                        "driving_command": 4,
                        "replicate_jpeg_quality": 20,
                    }
                ],
            )

    stanley_node = Node(
        package="orion_ros",
        executable="stanley_controller_node",
        name="stanley_controller",
        output="screen",
        parameters=[
            {
                "trajectory_topic":       "/orion/predicted_trajectory",
                "odometry_topic":         "/carla/hero/odometry",
                "control_output_topic":   "/carla/hero/ackermann_cmd",
                "stanley_gain_k":         1.5,
                "stanley_softening_ks":   0.5,
                "wheelbase_m":            2.875,
                "max_steering_rad":       0.6,
                "speed_lookahead_steps":  5,
                "control_hz":             20.0,
                "max_acceleration_mps2":  3.0,
                "trajectory_timeout_sec": 5.0,
            }
        ],
    )

    return LaunchDescription(
        [
            require_new_frame_arg,
            link_ckpts,
            RegisterEventHandler(
                OnProcessExit(target_action=link_ckpts, on_exit=[orion_node])
            ),
            TimerAction(period=90.0, actions=[stanley_node]),
        ]
    )
