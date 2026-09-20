from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import PythonExpression, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Launch the ORION node in PID / self-contained mode.

    Same model setup as orion.launch.py (full ORION at /root/Orion, weights at
    /models/Orion), but instead of publishing an Autoware Trajectory for a
    downstream Stanley controller, this node runs a Bench2Drive PID in-node and
    publishes carla_msgs/CarlaEgoVehicleControl (steer/throttle/brake) directly on
    /carla/hero/vehicle_control_cmd. There is therefore NO stanley_controller_node.
    """
    CAMERA_IDS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    compressed_camera_topics = [f"/carla/hero/{c}/image/compressed" for c in CAMERA_IDS]
    camera_topics = [f"/carla/hero/{c}/image_decompressed" for c in CAMERA_IDS]

    require_new_frame_arg = DeclareLaunchArgument(
        "require_new_frame",
        default_value="true",
        description="Gate inference on a fresh front frame (true) or run "
        "continuously on the latest data (false).",
    )
    profile_stages_arg = DeclareLaunchArgument(
        "profile_stages", default_value="false",
        description="Append per-stage CUDA-event timing to the inference log.")
    compile_targets_arg = DeclareLaunchArgument(
        "compile_targets", default_value="heads,llm,vit",
        description="Comma-separated torch.compile targets: vit, llm, heads ('' = eager).")
    backbone_engine_arg = DeclareLaunchArgument(
        "backbone_engine", default_value="",
        description="TensorRT plan for the ViT backbone ('' = PyTorch).")
    vit_glue_arg = DeclareLaunchArgument(
        "vit_glue", default_value="true",
        description="Fused-GEMM / SDPA rewrite of the EVA-ViT blocks (same weights).")
    pipeline_prep_arg = DeclareLaunchArgument(
        "pipeline_prep", default_value="true",
        description="Prepare frame N+1 on a second worker while frame N's forward runs.")
    map_head_slice_arg = DeclareLaunchArgument(
        "map_head_slice", default_value="true",
        description="Run the map head with its 300 one-to-one lane queries only (exact).")
    overlap_heads_arg = DeclareLaunchArgument(
        "overlap_heads", default_value="false",
        description="Run the map head on a side CUDA stream concurrently with the det head.")
    llm_int8_arg = DeclareLaunchArgument(
        "llm_int8", default_value="false",
        description="W8A8 dynamic INT8 for the LLM decoder projections (changes numerics).")
    llm_int8_stats_arg = DeclareLaunchArgument(
        "llm_int8_stats", default_value="/benchmarking/alpamayo-autoware/src/orion_ros/engines/llm_act_stats.pt",
        description="SmoothQuant calibration stats for llm_int8 (tools/int8_sweep.py --save-stats).")
    vision_lite_arg = DeclareLaunchArgument(
        "vision_lite", default_value="false",
        description="ViT at 512x512 + rear cameras refreshed every other frame "
                    "(changes the model's inputs; route-score it). false = base model.")
    precision_arg = DeclareLaunchArgument(
        "precision", default_value="fp16",
        description="How the model is BUILT. 'fp16' halves img_backbone and loads the "
                    "LLM in fp16; 'fp32' leaves both fp32, which is what the reference "
                    "agent does (its config sets fp32_infer=True and is passed through "
                    "unmodified). fp32 costs far more memory -- ~50 GB RSS was measured "
                    "against 61 GB of unified memory -- so it may not fit. Both paths "
                    "still get custom_wrap_fp16_model afterwards.")
    timestamp_mode_arg = DeclareLaunchArgument(
        "timestamp_mode", default_value="sensor",
        description="'sensor' = the real frame stamp (honest dt, ~0.7 s at this rate). "
                    "'agent' = frame_idx/20, reproducing orion_b2d_agent's formula -- "
                    "true under synchronous CARLA, but a 16x-wrong dt here. A/B only.")
    gnss_mount_offset_x_arg = DeclareLaunchArgument(
        "gnss_mount_offset_x", default_value="-1.4",
        description="Localise the ego this far (m, vehicle frame) from the odometry "
                    "origin, matching orion_b2d_agent's GNSS mount at x=-1.4. "
                    "0.0 = vehicle origin (pre-fix behaviour).")
    replicate_jpeg_quality_arg = DeclareLaunchArgument(
        "replicate_jpeg_quality", default_value="20",
        description="Re-encode each camera at this JPEG quality before the model, "
                    "reproducing OrionAgent.tick's q20 pass. Set 0 when the images "
                    "already crossed the wire at q20 -- otherwise the model sees TWO "
                    "lossy passes where the reference agent applies one.")
    control_trace_arg = DeclareLaunchArgument(
        "control_trace", default_value="false",
        description="Write one CSV row per control tick (PID tuning); see control_trace_path.")
    control_trace_path_arg = DeclareLaunchArgument(
        "control_trace_path", default_value="",
        description="Trace file; empty = /benchmarking/log/control_trace_<timestamp>.csv.")
    merge_lora_arg = DeclareLaunchArgument(
        "merge_lora", default_value="true",
        description="Fold the LoRA adapters into the LLM weights at load.")
    llm_flash_attn_arg = DeclareLaunchArgument(
        "llm_flash_attn", default_value="true",
        description="LLaMA prefill attention through flash_attn (fp16 only).")

    link_ckpts = ExecuteProcess(
        cmd=["ln", "-sfn", "/models/Orion", "/root/Orion/ckpts"],
        output="screen",
    )

    image_decompress_nodes = [
        Node(
            package="orion_ros",
            executable="image_decompress_node",
            name=f"image_decompress_{cam.lower()}",
            output="screen",
            parameters=[
                {
                    "in_topic":  compressed_camera_topics[i],
                    "out_topic": camera_topics[i],
                }
            ],
        )
        for i, cam in enumerate(CAMERA_IDS)
    ]

    orion_withpid_node = Node(
        package="orion_ros",
        executable="orion_withpid_node",
        name="orion_withpid_node",
        output="screen",
        parameters=[
            {
                "orion_repo_path": "/root/Orion",
                "orion_config_path":
                    "/root/Orion/adzoo/orion/configs/orion_stage3_agent.py",
                "orion_checkpoint_path": "/models/Orion/Orion.pth",
                "precision": ParameterValue(
                    LaunchConfiguration("precision"), value_type=str),
                "camera_topics": camera_topics,
                "speed_topic": "/carla/hero/speed",
                "odometry_topic": "/carla/hero/odometry",
                "imu_topic": "/carla/hero/imu",
                "route_topic": "/carla/hero/global_plan",
                "cot_topic": "/orion/cot",
                "control_topic": "/carla/hero/vehicle_control_cmd",
                "inference_period_sec": 0.05,
                "require_new_frame": ParameterValue(
                    LaunchConfiguration("require_new_frame"), value_type=bool
                ),
                "profile_stages": ParameterValue(
                    LaunchConfiguration("profile_stages"), value_type=bool),
                "compile_targets": ParameterValue(
                    LaunchConfiguration("compile_targets"), value_type=str),
                "backbone_engine": ParameterValue(
                    LaunchConfiguration("backbone_engine"), value_type=str),
                "vit_glue": ParameterValue(
                    LaunchConfiguration("vit_glue"), value_type=bool),
                "pipeline_prep": ParameterValue(
                    LaunchConfiguration("pipeline_prep"), value_type=bool),
                "map_head_slice": ParameterValue(
                    LaunchConfiguration("map_head_slice"), value_type=bool),
                "overlap_heads": ParameterValue(
                    LaunchConfiguration("overlap_heads"), value_type=bool),
                "llm_int8": ParameterValue(
                    LaunchConfiguration("llm_int8"), value_type=bool),
                "llm_int8_stats": ParameterValue(
                    LaunchConfiguration("llm_int8_stats"), value_type=str),
                "vit_input_size": ParameterValue(PythonExpression(
                    ["512 if '", LaunchConfiguration("vision_lite"), "' == 'true' else 640"]), value_type=int),
                "rear_view_refresh_every": ParameterValue(PythonExpression(
                    ["2 if '", LaunchConfiguration("vision_lite"), "' == 'true' else 1"]), value_type=int),
                "control_trace": ParameterValue(
                    LaunchConfiguration("control_trace"), value_type=bool),
                "control_trace_path": ParameterValue(
                    LaunchConfiguration("control_trace_path"), value_type=str),
                "merge_lora": ParameterValue(
                    LaunchConfiguration("merge_lora"), value_type=bool),
                "llm_flash_attn": ParameterValue(
                    LaunchConfiguration("llm_flash_attn"), value_type=bool),
                "publish_cot": True,
                "driving_command": 4,
                "timestamp_mode": ParameterValue(
                    LaunchConfiguration("timestamp_mode"), value_type=str),
                "gnss_mount_offset_x": ParameterValue(
                    LaunchConfiguration("gnss_mount_offset_x"), value_type=float),
                "replicate_jpeg_quality": ParameterValue(
                    LaunchConfiguration("replicate_jpeg_quality"), value_type=int),
            }
        ],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}"),
            require_new_frame_arg,
            profile_stages_arg,
            compile_targets_arg,
            backbone_engine_arg,
            vit_glue_arg,
            pipeline_prep_arg,
            map_head_slice_arg,
            overlap_heads_arg,
            llm_int8_arg,
            llm_int8_stats_arg,
            vision_lite_arg,
            precision_arg,
            timestamp_mode_arg,
            gnss_mount_offset_x_arg,
            replicate_jpeg_quality_arg,
            control_trace_arg,
            control_trace_path_arg,
            merge_lora_arg,
            llm_flash_attn_arg,
            *image_decompress_nodes,
            link_ckpts,
            RegisterEventHandler(
                OnProcessExit(target_action=link_ckpts, on_exit=[orion_withpid_node])
            ),
        ]
    )
