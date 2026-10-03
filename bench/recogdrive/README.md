# ReCogDrive benches

ReCogDrive's optimisations live in
`ros2_ws/src/recogdrive_ros/recogdrive_ros/recogdrive_speedups.py`
(`FastReCogDrive`, a wrapper around the built agent). These scripts are how it
was arrived at and how it is checked. None of them need CARLA or ROS.

Run them through `./run.sh <script> [args]`, which activates the venv
(`env/recogdrive/setup_venv.sh`) and puts the job under the memory guard.
`common.py` reads three environment variables:

| Variable | Meaning |
|---|---|
| `VLA_BENCH_DIR` | Work directory holding the ReCogDrive checkout as `recogdrive/` |
| `RECOGDRIVE_WEIGHTS` | Folder with the 2B VLM files and `ReCogDrive_Diffusion_Planner_2B_RL.ckpt` |
| `RECOGDRIVE_ROS_SRC` | The `recogdrive_ros` package source (default: this repo's) |

`python3 make_frames.py` first: it turns `assets/frames/` into the three
1920×1080 test frames under `frames/`.

| Script | Purpose |
|---|---|
| `fast_check.py` | **The one to run first.** Fast path vs the untouched agent on the same inputs with the same diffusion noise: hidden-state difference, trajectory difference, pixel equality, fallback, repeatability, speed. |
| `profile_baseline.py` | Stage profile of the unmodified `compute_trajectory`. |
| `microbench.py` | Matmul ceiling per dtype, token accounting, cost vs sequence length / tiles / planner context. Decides which stage is compute-bound and which is dispatch-bound. |
| `opbench.py` | Per-op cost inside one LLM layer and one ViT layer. |
| `rowdep.py` | Does a bf16 row's result depend on the batch's row count? (It does.) |
| `castprobe.py` | Where eager bf16 rounds, and what `torch.compile` does to those roundings. |
| `noise_floor.py` | Stage-by-stage deviation of the fast path against the reference. |
| `fusebench.py`, `fusebench2.py` | Fused kernels and alternative matmul formulations: speed and bit-equality. |
| `planner_probe.py`, `planner_probe2.py` | Planner and image-stage breakdown. |
| `longprompt_check.py` | The no-padding branch (prompts over 2800 tokens). |
| `tok_probe.py` | Prompt token-length range, split tokenisation, PIL resize costs. |
| `init_time.py`, `engine_timing.py` | Start-up phases; frame time with a TensorRT encoder. |
| `trt_vit.py` | ViT → ONNX, TensorRT engines (fp16 / calibrator INT8 / Q/DQ), latency and accuracy. |
| `trt_qdq_probe.py` | Does TensorRT run a Q/DQ matmul in INT8 on this device, and how fast? |
| `vit_int8.py` | INT8 calibration, SmoothQuant-style rebalancing, accuracy simulation, Q/DQ ONNX. |
| `vit_e2e.py` | What a different vision encoder does to the planned trajectory. |

The write-up built from these is
`ros2_ws/src/recogdrive_ros/RECOGDRIVE_inference_optimisation.md`.
