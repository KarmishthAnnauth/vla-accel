"""
Launch SimLingo VLA inference with SimLingo's own PID controller.

SimLingo's route waypoints were trained and tuned against the PID controller in
agent_simlingo.py::control_pid, so that is what drives by default. simlingo_node
runs it in-node and publishes steer/throttle/brake straight to CARLA — no Stanley
controller and no carla_ackermann_control node.

  IMPORTANT: carla_ackermann_control must NOT be running in PID mode. It also
  publishes /carla/hero/vehicle_control_cmd and the two would fight. The CARLA-side
  leaderboard agent must be in direct-control mode (CONTROL_MODE), not ackermann.

Topic flow (no carla_bridge_node needed — carla_ros_bridge on the CARLA machine
publishes directly to /carla/hero/* topics over the shared ROS 2 DDS domain):

  CARLA machine (my_ros2_agent_simlingo.py + carla_ros_bridge)
      /carla/hero/rgb_0/image          → simlingo_node
      /carla/hero/speed                → simlingo_node
      /carla/hero/odometry             → simlingo_node
      /carla/hero/global_plan          → simlingo_node

  simlingo_node
      /carla/hero/vehicle_control_cmd  → leaderboard agent
      /simlingo/predicted_trajectory   (debug / RViz, published in both modes)
      /simlingo/language_output        (debug)

Set control_mode:=trajectory to go back to the old path, where simlingo_node only
publishes the Autoware trajectory and stanley_controller_node converts it to
/carla/hero/ackermann_cmd for carla_ackermann_control.

Launch arguments:
  checkpoint_path   Absolute path to the SimLingo .ckpt weights file
  simlingo_path     Absolute path to the simlingo repository root on this machine
  control_mode      "pid" (default) or "trajectory"
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:

    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
        description="Absolute path to the SimLingo .ckpt weights file.",
    )
    simlingo_path_arg = DeclareLaunchArgument(
        "simlingo_path",
        default_value="/workspace/simlingo",
        description=(
            "Absolute path to the simlingo repository root so that "
            "simlingo_training and team_code packages are importable."
        ),
    )
    control_mode_arg = DeclareLaunchArgument(
        "control_mode",
        default_value="pid",
        description=(
            "'pid': simlingo_node runs SimLingo's own control_pid and publishes "
            "CarlaEgoVehicleControl directly. 'trajectory': publish the Autoware "
            "trajectory only and let stanley_controller_node drive."
        ),
    )

    control_hz_arg = DeclareLaunchArgument(
        "control_hz",
        default_value="20.0",
        description=(
            "Rate of the control loop in Hz. SimLingo is a 20 Hz policy "
            "(agent_simlingo.py runs control_pid once per simulator frame at "
            "carla_fps=20), and the loop tracks the latest map-anchored plan "
            "at this rate regardless of how slowly inference produces plans."
        ),
    )
    brake_speed_arg = DeclareLaunchArgument(
        "brake_speed_mps",
        default_value="0.4",
        description=(
            "desired_speed below which the controller brakes instead of "
            "throttling. 0.4 is upstream (config_simlingo.py). Lower it only "
            "on evidence from a run whose control rate was correct."
        ),
    )
    control_timeout_arg = DeclareLaunchArgument(
        "control_timeout_sec",
        default_value="0.5",
        description=(
            "Full-brake if the control loop stops publishing for this long. "
            "Guards the control loop, not inference -- a slow model is normal "
            "and is handled by tracking the existing plan."
        ),
    )
    max_plan_age_arg = DeclareLaunchArgument(
        "max_plan_age_sec",
        default_value="5.0",
        description=(
            "Full-brake if the newest prediction is older than this, i.e. the "
            "model has stopped producing plans altogether."
        ),
    )

    def _f(name):
        """A launch argument as a float parameter.

        Launch substitutions resolve to strings; the node declares these as
        doubles, and handing a string to a double parameter is a hard error at
        startup rather than a coercion.
        """
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    use_stanley = IfCondition(
        PythonExpression(["'", LaunchConfiguration("control_mode"), "' == 'trajectory'"])
    )

    cuda_home = "/usr/local/cuda-12.6"
    ld_library_path = ":".join([
        f"{cuda_home}/targets/aarch64-linux/lib",
        f"{cuda_home}/lib64",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu/tegra",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])

    simlingo_node = Node(
        package="simlingo_ros",
        executable="simlingo_node",
        name="simlingo_node",
        output="screen",
        parameters=[
            {
                "checkpoint_path":          LaunchConfiguration("checkpoint_path"),
                "simlingo_path":            LaunchConfiguration("simlingo_path"),
                "image_topic":              "/carla/hero/rgb_0/compressed",
                "speed_topic":              "/carla/hero/speed",
                "odometry_topic":           "/carla/hero/odometry",
                "route_topic":              "/carla/hero/global_plan",
                "trajectory_topic":         "/simlingo/predicted_trajectory",
                "language_topic":           "/simlingo/language_output",
                "control_topic":            "/carla/hero/vehicle_control_cmd",
                "control_mode":             LaunchConfiguration("control_mode"),
                "inference_period_sec":     0.25,
                "min_trajectory_speed_mps": 0.5,
                "brake_speed_mps":          _f("brake_speed_mps"),
                "control_hz":               _f("control_hz"),
                "control_timeout_sec":      _f("control_timeout_sec"),
                "max_plan_age_sec":         _f("max_plan_age_sec"),
                "pid_window_rate_compensation": False,
            }
        ],
        additional_env={
            "CUDA_HOME":       cuda_home,
            "LD_LIBRARY_PATH": ld_library_path,
        },
    )

    stanley_node = Node(
        package="simlingo_ros",
        executable="stanley_controller_node",
        name="stanley_controller",
        output="screen",
        condition=use_stanley,
        parameters=[
            {
                "trajectory_topic":       "/simlingo/predicted_trajectory",
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

    return LaunchDescription([
        checkpoint_path_arg,
        simlingo_path_arg,
        control_mode_arg,
        control_hz_arg,
        brake_speed_arg,
        control_timeout_arg,
        max_plan_age_arg,
        SetEnvironmentVariable("CUDA_HOME", cuda_home),
        SetEnvironmentVariable("LD_LIBRARY_PATH", ld_library_path),
        simlingo_node,
        TimerAction(period=90.0, actions=[stanley_node]),
    ])
