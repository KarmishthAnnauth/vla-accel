from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


CAMERA_IDS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

VARIANTS = {
    "3b": ("adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py",
           "/models/MindDrive/minddrive_3b_rltrain.pth"),
    "05b": ("adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py",
            "/models/MindDrive/minddrive_rltrain.pth"),
}


def generate_launch_description() -> LaunchDescription:
    """Launch MindDrive in its self-contained closed-loop form: six
    image_decompress nodes (JPEG from carla-host -> raw Image locally) and the
    minddrive_node, which runs the model and the Bench2Drive decision-expert
    PID and publishes carla_msgs/CarlaEgoVehicleControl directly."""
    compressed_camera_topics = [f"/carla/hero/{c}/image/compressed" for c in CAMERA_IDS]
    camera_topics = [f"/carla/hero/{c}/image_decompressed" for c in CAMERA_IDS]

    variant_arg = DeclareLaunchArgument(
        "variant", default_value="3b",
        description="Which MindDrive: '3b' (minddrive_3b_rltrain.pth, Qwen2.5-3B) or "
                    "'05b' (minddrive_rltrain.pth, Qwen2-0.5B).")
    require_new_frame_arg = DeclareLaunchArgument(
        "require_new_frame", default_value="true",
        description="Gate inference on a fresh front frame (true) or run "
                    "continuously on the latest data (false).")
    timestamp_mode_arg = DeclareLaunchArgument(
        "timestamp_mode", default_value="sensor",
        description="'sensor' = the real frame stamp (honest dt; memory zeroed when "
                    "consumed frames are >2 s apart). 'agent' = frame_idx/20, the "
                    "agent's formula under synchronous CARLA. A/B only.")
    gnss_mount_offset_x_arg = DeclareLaunchArgument(
        "gnss_mount_offset_x", default_value="-1.4",
        description="Localise the ego this far (m, vehicle frame) from the odometry "
                    "origin, matching the agent's GNSS mount at x=-1.4.")
    replicate_jpeg_quality_arg = DeclareLaunchArgument(
        "replicate_jpeg_quality", default_value="20",
        description="Re-encode each camera at this JPEG quality before the model, "
                    "reproducing MinddriveAgent.tick's q20 pass. 0 when the images "
                    "already crossed the wire at q20 (run_minddrive_ros.sh sets that).")
    control_trace_arg = DeclareLaunchArgument(
        "control_trace", default_value="false",
        description="Write one CSV row per control tick (PID tuning).")
    control_trace_path_arg = DeclareLaunchArgument(
        "control_trace_path", default_value="",
        description="Trace file; empty = /benchmarking/minddrive_env/logs/control_trace_<ts>.csv.")
    decode_workers_arg = DeclareLaunchArgument(
        "decode_workers", default_value="6",
        description="Threads for the six per-camera JPEG decodes (bit-exact).")
    precision_arg = DeclareLaunchArgument(
        "precision", default_value="config",
        description="'config' = the reference config's fp32_infer; 'fp16' = fp16_infer "
                    "(LLM fp16, ViT half, heads fp32).")
    merge_lora_arg = DeclareLaunchArgument(
        "merge_lora", default_value="false",
        description="Fold both LoRA adapter sets into two merged LLM copies (exact).")
    llm_attn_arg = DeclareLaunchArgument(
        "llm_attn", default_value="keep",
        description="Qwen2 attention: keep | sdpa | flash_attention_2 (fp16 only) | eager.")
    vit_glue_arg = DeclareLaunchArgument(
        "vit_glue", default_value="false",
        description="Fused qkv / w12 GEMMs + SDPA in the EVA-ViT blocks (same weights).")
    down_proj_t_arg = DeclareLaunchArgument(
        "down_proj_t", default_value="false",
        description="down_proj weight stored [K,N]-contiguous (same numbers).")
    vit_weight_t_arg = DeclareLaunchArgument(
        "vit_weight_t", default_value="false",
        description="EVA-ViT qkv/proj/w3 weights stored [K,N]-contiguous (same numbers; needs vit_glue).")
    vit_window_nopad_arg = DeclareLaunchArgument(
        "vit_window_nopad", default_value="false",
        description="Window-attention blocks run qkv/proj on the unpadded token grid (same math; needs vit_glue).")
    int8_targets_arg = DeclareLaunchArgument(
        "int8_targets", default_value="",
        description="Comma-separated W8A8 int8 targets: llm, vit ('' = none). A precision change.")
    int8_skip_arg = DeclareLaunchArgument(
        "int8_skip", default_value="",
        description="Comma-separated linear-name substrings kept fp16 under int8_targets.")
    int8_calib_path_arg = DeclareLaunchArgument(
        "int8_calib_path", default_value="",
        description="SmoothQuant stats from tools/bench_minddrive.py --int8-calib-out ('' = plain per-token int8).")
    int8_alpha_arg = DeclareLaunchArgument(
        "int8_alpha", default_value="0.5",
        description="SmoothQuant migration strength.")
    vit_input_size_arg = DeclareLaunchArgument(
        "vit_input_size", default_value="640",
        description="EVA-ViT input size (640 = trained; 512 = ORION's lite vision, a numerics change).")
    rear_view_refresh_every_arg = DeclareLaunchArgument(
        "rear_view_refresh_every", default_value="1",
        description="Rear cameras through the ViT every N frames (1 = every frame).")
    map_head_slice_arg = DeclareLaunchArgument(
        "map_head_slice", default_value="false",
        description="Map head with its 300 one-to-one lane queries only (exact).")
    compile_targets_arg = DeclareLaunchArgument(
        "compile_targets", default_value="",
        description="Comma-separated torch.compile targets: vit, llm, heads ('' = eager).")
    compile_mode_arg = DeclareLaunchArgument(
        "compile_mode", default_value="default",
        description="torch.compile mode for compile_targets.")
    cuda_graph_vit_arg = DeclareLaunchArgument(
        "cuda_graph_vit", default_value="false",
        description="Manual CUDA-graph replay of the ViT forward.")
    cuda_graph_heads_arg = DeclareLaunchArgument(
        "cuda_graph_heads", default_value="false",
        description="CUDA-graph trees (reduce-overhead) on the head transformer stacks.")
    logits_slice_arg = DeclareLaunchArgument(
        "logits_slice", default_value="false",
        description="Decision expert: vocab projection on the one position read (exact).")
    overlap_experts_arg = DeclareLaunchArgument(
        "overlap_experts", default_value="false",
        description="Run the two expert prefills concurrently on two streams (exact).")
    pipeline_prep_arg = DeclareLaunchArgument(
        "pipeline_prep", default_value="false",
        description="Prepare frame N+1 on a second worker while frame N's forward runs.")
    profile_stages_arg = DeclareLaunchArgument(
        "profile_stages", default_value="false",
        description="Append per-stage CUDA-event timing to the inference log.")

    image_decompress_nodes = [
        Node(
            package="minddrive_ros",
            executable="image_decompress_node",
            name=f"image_decompress_{cam.lower()}",
            output="screen",
            parameters=[{"in_topic": compressed_camera_topics[i],
                         "out_topic": camera_topics[i]}],
        )
        for i, cam in enumerate(CAMERA_IDS)
    ]

    variant = LaunchConfiguration("variant")
    minddrive_node = Node(
        package="minddrive_ros",
        executable="minddrive_node",
        name="minddrive_node",
        output="screen",
        parameters=[
            {
                "minddrive_repo_path": "/benchmarking/MindDrive",
                "minddrive_config_path": ParameterValue(PythonExpression([
                    "'", VARIANTS["05b"][0], "' if '", variant, "' == '05b' else '",
                    VARIANTS["3b"][0], "'"]), value_type=str),
                "minddrive_checkpoint_path": ParameterValue(PythonExpression([
                    "'", VARIANTS["05b"][1], "' if '", variant, "' == '05b' else '",
                    VARIANTS["3b"][1], "'"]), value_type=str),
                "camera_topics": camera_topics,
                "speed_topic": "/carla/hero/speed",
                "odometry_topic": "/carla/hero/odometry",
                "imu_topic": "/carla/hero/imu",
                "route_topic": "/carla/hero/global_plan",
                "control_topic": "/carla/hero/vehicle_control_cmd",
                "meta_action_topic": "/minddrive/meta_action",
                "inference_period_sec": 0.05,
                "require_new_frame": ParameterValue(
                    LaunchConfiguration("require_new_frame"), value_type=bool),
                "timestamp_mode": ParameterValue(
                    LaunchConfiguration("timestamp_mode"), value_type=str),
                "gnss_mount_offset_x": ParameterValue(
                    LaunchConfiguration("gnss_mount_offset_x"), value_type=float),
                "replicate_jpeg_quality": ParameterValue(
                    LaunchConfiguration("replicate_jpeg_quality"), value_type=int),
                "decode_workers": ParameterValue(
                    LaunchConfiguration("decode_workers"), value_type=int),
                "control_trace": ParameterValue(
                    LaunchConfiguration("control_trace"), value_type=bool),
                "control_trace_path": ParameterValue(
                    LaunchConfiguration("control_trace_path"), value_type=str),
                "driving_command": 4,
                "precision": ParameterValue(LaunchConfiguration("precision"), value_type=str),
                "merge_lora": ParameterValue(LaunchConfiguration("merge_lora"), value_type=bool),
                "llm_attn": ParameterValue(LaunchConfiguration("llm_attn"), value_type=str),
                "vit_glue": ParameterValue(LaunchConfiguration("vit_glue"), value_type=bool),
                "down_proj_t": ParameterValue(LaunchConfiguration("down_proj_t"), value_type=bool),
                "vit_weight_t": ParameterValue(LaunchConfiguration("vit_weight_t"), value_type=bool),
                "vit_window_nopad": ParameterValue(LaunchConfiguration("vit_window_nopad"), value_type=bool),
                "int8_targets": ParameterValue(LaunchConfiguration("int8_targets"), value_type=str),
                "int8_skip": ParameterValue(LaunchConfiguration("int8_skip"), value_type=str),
                "int8_calib_path": ParameterValue(LaunchConfiguration("int8_calib_path"), value_type=str),
                "int8_alpha": ParameterValue(LaunchConfiguration("int8_alpha"), value_type=float),
                "vit_input_size": ParameterValue(LaunchConfiguration("vit_input_size"), value_type=int),
                "rear_view_refresh_every": ParameterValue(LaunchConfiguration("rear_view_refresh_every"), value_type=int),
                "map_head_slice": ParameterValue(LaunchConfiguration("map_head_slice"), value_type=bool),
                "compile_targets": ParameterValue(LaunchConfiguration("compile_targets"), value_type=str),
                "compile_mode": ParameterValue(LaunchConfiguration("compile_mode"), value_type=str),
                "cuda_graph_vit": ParameterValue(LaunchConfiguration("cuda_graph_vit"), value_type=bool),
                "cuda_graph_heads": ParameterValue(LaunchConfiguration("cuda_graph_heads"), value_type=bool),
                "logits_slice": ParameterValue(LaunchConfiguration("logits_slice"), value_type=bool),
                "overlap_experts": ParameterValue(LaunchConfiguration("overlap_experts"), value_type=bool),
                "pipeline_prep": ParameterValue(LaunchConfiguration("pipeline_prep"), value_type=bool),
                "profile_stages": ParameterValue(LaunchConfiguration("profile_stages"), value_type=bool),
            }
        ],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}"),
            variant_arg,
            require_new_frame_arg,
            timestamp_mode_arg,
            gnss_mount_offset_x_arg,
            replicate_jpeg_quality_arg,
            control_trace_arg,
            control_trace_path_arg,
            decode_workers_arg,
            precision_arg,
            merge_lora_arg,
            llm_attn_arg,
            vit_glue_arg,
            down_proj_t_arg,
            vit_weight_t_arg,
            vit_window_nopad_arg,
            int8_targets_arg,
            int8_skip_arg,
            int8_calib_path_arg,
            int8_alpha_arg,
            vit_input_size_arg,
            rear_view_refresh_every_arg,
            map_head_slice_arg,
            compile_targets_arg,
            compile_mode_arg,
            cuda_graph_vit_arg,
            cuda_graph_heads_arg,
            logits_slice_arg,
            overlap_experts_arg,
            pipeline_prep_arg,
            profile_stages_arg,
            *image_decompress_nodes,
            minddrive_node,
        ]
    )
