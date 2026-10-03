from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Launch ReCogDrive on live CARLA topics: one image_decompress node (JPEG
    from carla-host -> raw Image locally), the recogdrive_node, which runs the
    model and publishes its trajectory, and the Stanley controller that tracks
    it (AckermannDrive on /carla/hero/ackermann_cmd, converted to
    /carla/hero/vehicle_control_cmd by carla_ackermann_control on the CARLA
    side, the same chain as orion.launch.py and simlingo's trajectory mode).

    ReCogDrive plans from the front camera alone, so there is one camera here,
    not six.  carla_ros_bridge publishes /carla/<role_name>/<sensor id>/image;
    `camera_id` is that sensor id on the CARLA-side agent.  NAVSIM's cam_f0,
    which the released weights were trained on, is 1920x1080.
    """
    camera_id = LaunchConfiguration("camera_id")
    compressed_topic = ["/carla/hero/", camera_id, "/image/compressed"]
    decompressed_topic = ["/carla/hero/", camera_id, "/image_decompressed"]
    raw_topic = ["/carla/hero/", camera_id, "/image"]
    front_camera_topic = PythonExpression([
        "'/carla/hero/", camera_id, "/image_decompressed' if '",
        LaunchConfiguration("compressed"), "'.lower() in ('true', '1') else '/carla/hero/",
        camera_id, "/image'"])

    args = [
        DeclareLaunchArgument(
            "vlm_path", default_value="/models/ReCogDrive/ReCogDrive-VLM-2B",
            description="The stage-1 VLM: an InternVL checkpoint directory (config.json, "
                        "weights, tokenizer, the model's own python files)."),
        DeclareLaunchArgument(
            "checkpoint_path", default_value="",
            description="The diffusion-planner checkpoint (.ckpt with a state_dict). Required."),
        DeclareLaunchArgument(
            "vlm_size", default_value="small",
            description="'small' = the 2B VLM (hidden 1536), 'large' = the 8B VLM (hidden 3584). "
                        "Must match vlm_path and the checkpoint."),
        DeclareLaunchArgument(
            "dit_type", default_value="small",
            description="DiT preset of the planner; 'small' for every released checkpoint."),
        DeclareLaunchArgument(
            "sampling_method", default_value="ddim",
            description="Planner sampler the checkpoint was trained with: ddim | ddpm | flow."),
        DeclareLaunchArgument(
            "recogdrive_repo_path", default_value="/benchmarking/recogdrive",
            description="The ReCogDrive checkout (its navsim package)."),
        DeclareLaunchArgument(
            "camera_id", default_value="CAM_FRONT",
            description="CARLA sensor id of the front camera (the topic name segment)."),
        DeclareLaunchArgument(
            "compressed", default_value="true",
            description="true: the camera arrives as JPEG on <camera>/image/compressed and "
                        "is decompressed locally. false: subscribe to the raw <camera>/image."),
        DeclareLaunchArgument(
            "control_mode", default_value="pid",
            description="pid: Bench2Drive PID in recogdrive_node, CarlaEgoVehicleControl on "
                        "/carla/hero/vehicle_control_cmd (carla_ackermann_control must NOT run). "
                        "stanley: Stanley node -> AckermannDrive (needs carla_ackermann_control)."),
        DeclareLaunchArgument(
            "pid_controller_dir", default_value="/benchmarking/Orion/team_code",
            description="pid: directory holding Bench2Drive's pid_controller.py (the team_code "
                        "of an ORION / MindDrive checkout)."),
        DeclareLaunchArgument(
            "speed_cap_mps", default_value="5.0",
            description="pid: throttle is cut above this speed, as the Bench2Drive reference "
                        "agents do (0 = no cap)."),
        DeclareLaunchArgument(
            "require_new_frame", default_value="true",
            description="Gate inference on a fresh front frame (true) or run "
                        "continuously on the latest data (false)."),
        DeclareLaunchArgument(
            "require_full_history", default_value="false",
            description="true: wait for 1.5 s of odometry before the first inference. "
                        "false: hold the oldest known pose for the frames before it."),
        DeclareLaunchArgument(
            "ego_reference_offset_x", default_value="0.0",
            description="Localise the ego this far (m, vehicle frame) from the odometry "
                        "origin. NAVSIM's ego pose is the rear axle."),
        DeclareLaunchArgument(
            "rebase_to_latest_pose", default_value="true",
            description="Publish the plan in the ego frame of the latest odometry (true) "
                        "or of the camera frame it was planned from (false)."),
        DeclareLaunchArgument(
            "min_trajectory_speed_mps", default_value="0.0",
            description="Lower bound on the speeds written to the trajectory (0 = the model's)."),
        DeclareLaunchArgument(
            "frame_format", default_value="bmp",
            description="File format of the frame handed to the agent: bmp | png (lossless), jpg."),
        DeclareLaunchArgument(
            "seed", default_value="-1",
            description=">= 0 seeds torch once at start-up; the reference eval does not seed."),
        DeclareLaunchArgument(
            "fast_inference", default_value="true",
            description="true: the fast path (same computation, VLM half bit-identical to "
                        "the reference). false: ReCogDriveAgent.compute_trajectory itself."),
        DeclareLaunchArgument(
            "vit_engine", default_value="",
            description="Opt-in TensorRT engine for the vision encoder (not bit-identical; "
                        "see recogdrive_speedups._TrtVit). Empty = exact PyTorch encoder."),
        DeclareLaunchArgument(
            "profile", default_value="true",
            description="Per-stage times in the inference log line (fast path)."),
        DeclareLaunchArgument(
            "allow_partial_checkpoint", default_value="false",
            description="Run even if planner weights are missing from the checkpoint."),
    ]

    image_decompress_node = Node(
        package="recogdrive_ros",
        executable="image_decompress_node",
        name="image_decompress_front",
        output="screen",
        condition=IfCondition(LaunchConfiguration("compressed")),
        parameters=[{"in_topic": compressed_topic, "out_topic": decompressed_topic}],
    )

    recogdrive_node = Node(
        package="recogdrive_ros",
        executable="recogdrive_node",
        name="recogdrive_node",
        output="screen",
        parameters=[
            {
                "recogdrive_repo_path": LaunchConfiguration("recogdrive_repo_path"),
                "vlm_path": LaunchConfiguration("vlm_path"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "vlm_type": "internvl",
                "vlm_size": LaunchConfiguration("vlm_size"),
                "dit_type": LaunchConfiguration("dit_type"),
                "sampling_method": LaunchConfiguration("sampling_method"),
                "cam_type": "single",
                "allow_partial_checkpoint": ParameterValue(
                    LaunchConfiguration("allow_partial_checkpoint"), value_type=bool),
                "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                "fast_inference": ParameterValue(
                    LaunchConfiguration("fast_inference"), value_type=bool),
                "profile": ParameterValue(LaunchConfiguration("profile"), value_type=bool),
                "vit_engine": LaunchConfiguration("vit_engine"),
                "front_camera_topic": ParameterValue(front_camera_topic, value_type=str),
                "odometry_topic": "/carla/hero/odometry",
                "imu_topic": "/carla/hero/imu",
                "route_topic": "/carla/hero/global_plan",
                "trajectory_topic": "/recogdrive/predicted_trajectory",
                "path_topic": "/recogdrive/predicted_path",
                "inference_period_sec": 0.05,
                "require_new_frame": ParameterValue(
                    LaunchConfiguration("require_new_frame"), value_type=bool),
                "require_full_history": ParameterValue(
                    LaunchConfiguration("require_full_history"), value_type=bool),
                "ego_reference_offset_x": ParameterValue(
                    LaunchConfiguration("ego_reference_offset_x"), value_type=float),
                "rebase_to_latest_pose": ParameterValue(
                    LaunchConfiguration("rebase_to_latest_pose"), value_type=bool),
                "min_trajectory_speed_mps": ParameterValue(
                    LaunchConfiguration("min_trajectory_speed_mps"), value_type=float),
                "frame_format": LaunchConfiguration("frame_format"),
                "control_mode": LaunchConfiguration("control_mode"),
                "pid_controller_dir": LaunchConfiguration("pid_controller_dir"),
                "speed_cap_mps": ParameterValue(LaunchConfiguration("speed_cap_mps"), value_type=float),
                "driving_command": 4,
            }
        ],
    )

    stanley_node = Node(
        package="recogdrive_ros",
        executable="stanley_controller_node",
        name="stanley_controller",
        output="screen",
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("control_mode"), "' == 'stanley'"])),
        parameters=[
            {
                "trajectory_topic":       "/recogdrive/predicted_trajectory",
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
            SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}"),
            *args,
            image_decompress_node,
            recogdrive_node,
            stanley_node,
        ]
    )
