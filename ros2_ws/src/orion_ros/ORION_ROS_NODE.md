# ORION VLA — Inference Requirements & ROS 2 Node Setup

This document describes (A) what the ORION model needs in order to run inference,
and (B) how `orion_ros/orion_node.py` provides exactly that from live CARLA
ROS 2 topics, for inference benchmarking.

ORION repo: https://github.com/xiaomi-mlab/Orion (ICCV'25).
Reference nodes studied: `alpamayo_ros/alpamayo_node.py`, `simlingo_ros/simlingo_node.py`.

---

## 1. What ORION requires for inference

ORION (`mmcv/models/detectors/orion.py::Orion`, an `MVXTwoStageDetector`) fuses:
- an EVA-ViT image backbone + a 3D detection head (`OrionHead`) + a map head
  (`OrionHeadM`),
- a LLaVA-LLaMA LLM (`LlavaLlamaForCausalLM`),
- a generative trajectory planner (VAE / diffusion / MLP head).

### 1.1 Model loading
- Built from an mmcv config via `build_model(cfg.model, ...)`, then weights loaded
  with `load_checkpoint(model, Orion.pth)`.
- The LLM is loaded by `mmcv/utils/misc.py::load_model` from a **local** path
  (`llm_path = 'ckpts/pretrain_qformer/'`) — fp32 by default, fp16 if `fp16_infer`.
  LoRA is applied (`use_lora=True`).
- A special `<waypoint_ego>` token is added; the LLM hidden state at that token
  position conditions the planner (the reasoning→action bridge).
- Precision flags (mutually exclusive) live on `cfg.model`:
  - `fp32_infer=True`  → fp32, metric branch skipped (closed-loop agent default).
  - `fp16_infer=True`  → fp16 LLM + `img_backbone.half()` (heads stay fp32),
    "faster close-loop infer"; metric branch skipped **iff** `fp16_eval=False`.
  - `fp16_eval=True`   → re-enables the metric branch (needs ground truth) — must
    be OFF for ROS inference.

### 1.2 Inputs (single timestep)
With this config `use_lidar=False` / `use_radar=False`, so ORION needs only:
- **6 camera images** at **1600×900**, fed as **BGR** (the pipeline's
  `NormalizeMultiviewImage` has `to_rgb=True`), in this fixed order:
  `CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT`.
- **Ego state** assembled into the agent's `results` dict (see §3.2): an 18-dim
  `can_bus` vector, `ego_pose`/`ego_pose_inv` (lidar→global 4×4), camera calibration
  (`lidar2img`, `lidar2cam`, `cam_intrinsic`, `lidar2ego`), `timestamp`, `frame_idx`.
- A **driving command** (`ego_fut_cmd`, one-hot of a CARLA RoadOption 1..6) —
  *not* a sensor; comes from the route planner. It selects the active trajectory
  mode, so it materially affects the output.
- A **text prompt** + tokenizer ids, produced inside the pipeline by
  `LoadAnnoatationCriticalVQATest`. With the agent config (`desc_qa=False`) this is
  planning-only (`<waypoint_ego>`), so **no chain-of-thought is generated** (fast).

These raw inputs are run through `cfg.inference_only_pipeline`
(`Compose`) → `mmcv.parallel.collate` (a.k.a. `mm_collate_to_batch_form`) →
`model(batch, return_loss=False)`. The image preprocessing (resize/crop to
640×640, normalize, pad), tokenization, and collation are all done **inside that
pipeline** — they must not be reimplemented.

### 1.3 Output
`simple_test` returns `bbox_list[0]` with:
- `pts_bbox['ego_fut_preds']` — the **(6, 2)** ego-frame future trajectory
  (6 waypoints, 0.5 s spacing, 3 s horizon; cumulative for the VAE head, masked by
  the active command),
- `pts_bbox` detection/map results,
- `text_out` — VQA/CoT text (empty on the planning-only agent config).

### 1.4 Environment prerequisites
- The ORION repo's **custom `mmcv` package** importable, plus `torch`,
  `transformers`, `diffusers`, and (per the config) `flash_attn`.
- Local weights: `ckpts/pretrain_qformer/` (base LLM + vision) and `Orion.pth`.
- The configs use **relative paths** (`ckpts/...`), resolved against the process
  CWD → must run with CWD = ORION repo root.
- GPU: ~32 GB fp32 / ~17 GB fp16 (per the README).

---

## 2. ROS 2 interface (CARLA ros-bridge, ego role_name = `hero`)

Message types confirmed against the live setup (`simlingo_node.py` and the
Bench2Drive `leaderboard/.../ros2_agent.py` publisher).

### Subscriptions
| Purpose | Topic (default) | Type | QoS |
|---|---|---|---|
| 6 cameras | `camera_topics` (list, ORION order) | `sensor_msgs/Image` (raw rgba8/bgra8) | RELIABLE, depth 5 |
| Speed | `/carla/hero/speed` | `std_msgs/Float32` | default (10) |
| Odometry | `/carla/hero/odometry` | `nav_msgs/Odometry` | BEST_EFFORT, depth 10 |
| IMU | `/carla/hero/imu` | `sensor_msgs/Imu` | BEST_EFFORT, depth 10 |
| Route | `/carla/hero/global_plan` | `carla_msgs/CarlaRoute` | RELIABLE, **TRANSIENT_LOCAL** (latched), depth 1 |

### Publications
| Purpose | Topic (default) | Type |
|---|---|---|
| Trajectory | `/orion/predicted_trajectory` | `autoware_planning_msgs/Trajectory` |
| Trajectory markers | `/orion/predicted_trajectory_markers` | `visualization_msgs/MarkerArray` (map frame) |
| CoT / reasoning | `/orion/cot` | `std_msgs/String` (silent on planning-only config) |

---

## 3. How the node reproduces the ORION payload

**Core principle: reuse ORION's own code, swap only the raw input sources.** The
node imports ORION's `Config`, `build_model`, `load_checkpoint`, `Compose`,
`mmcv.parallel.collate`, and `get_box_type`, then reproduces the closed-loop agent
(`team_code/orion_b2d_agent.py::OrionAgent.run_step`) one-to-one — only the raw
images, speed, ego pose and route come from ROS instead of CARLA sensors.

### 3.1 Model setup (`_setup_model`, worker thread)
- Inserts `orion_repo_path` on `sys.path` and `os.chdir(repo_path)` so the config's
  relative `ckpts/...` paths resolve (model weights **and** pipeline tokenizer).
- Applies the `precision` parameter by flipping `cfg.model` flags (fp16 default).
- `build_model` → `load_checkpoint` → `.cuda()` → `.eval()`.
- Builds `Compose(cfg.inference_only_pipeline)` **minus**
  `LoadMultiViewImageFromFilesInCeph` (images are in memory) — exactly as the agent.
- Runs in a single-worker `ThreadPoolExecutor` so all CUDA handles live in one
  thread (same rationale as the reference nodes).

### 3.2 Payload build (`_build_batch`, worker thread)
Reproduces the agent's `results` dict field-for-field:
- `img`: 6 decoded BGR frames; calibration `lidar2img`/`lidar2cam`/`cam_intrinsic`
  copied verbatim from the agent (`LIDAR2IMG`, `LIDAR2CAM`, `LIDAR2EGO` constants).
- `can_bus`, `ego_pose`, `ego_pose_inv`, `lidar2ego`, `l2g_r_mat`, `l2g_t` (§3.4).
- `command` / `ego_fut_cmd` from the route planner (§3.5).
- meta: `folder`, `scene_token`, `frame_idx` (monotonic counter), `timestamp`
  (odom stamp), `box_type_3d`, `img_shape`/`ori_shape`/`pad_shape`.
- Then `inference_only_pipeline(results)` → `collate([results])` → move tensors to
  GPU with the agent's exact per-key loop (incl. the nested `input_ids` handling).
- `custom_wrap_fp16_model(model)` is called each inference (mirrors the agent):
  enable `fp16_enabled` everywhere except `map_head` / `pts_bbox_head`.

### 3.3 Image decode (`_decode_image`)
Incoming `sensor_msgs/Image` is decoded to **BGR uint8** (the channel order ORION
feeds the pipeline):
- `bgra8` → first 3 channels (already BGR; the common CARLA case),
- `rgba8`/`rgb8` → reversed to BGR, `bgr8` → as-is.
Then the agent's lossy **JPEG quality-20 re-encode** is replicated
(`replicate_jpeg_quality`, default 20). Raw bytes are stashed in the callback and
decoded in the worker thread to keep the ROS executor light.

### 3.4 Odometry/IMU → `can_bus` & `ego_pose` (`_odom_to_canbus_ego_pose`)
**No manual sign flips.** The CARLA agent reads raw left-handed CARLA sensors and
manually converts to ROS (negate y, negate yaw, negate angular y/z, rotate velocity
to body frame). The `carla-simulator/ros-bridge` has **already applied that exact
conversion** (verified in `carla_common/transforms.py`). So the bridge values are
used directly, yielding a `can_bus`/`ego_pose` numerically equivalent to the agent's:
- `can_bus[0:2]` = odom position x, y (already ENU; **not** re-negated).
- `ego_theta` = yaw from the odom quaternion (== agent's ego2world heading).
- `can_bus[7]` = speed from `/carla/hero/speed` (== CARLA speedometer; falls back to
  odom twist magnitude).
- `can_bus[10:13]` = IMU linear acceleration; `can_bus[13:16]` = IMU angular velocity
  (bridge already in ROS body-frame signs; falls back to zeros / odom twist).
- `ego_pose` = `ego2world @ LIDAR2EGO`; `ego_pose_inv` its inverse.

### 3.5 Driving command (`_compute_driving_command`)
Faithful port of Bench2Drive `RoutePlanner.run_step` (`team_code/planner.py`):
- The latched `CarlaRoute` provides `poses[]` and a parallel `road_options[]`
  (the per-waypoint RoadOption ints) — confirmed in the leaderboard
  `ros2_agent.py::set_global_plan`.
- `_route_callback` stores a `deque` of `(xy, road_option)` in the ROS map frame.
- Each inference, using the ego xy from odometry, the node pops waypoints already
  passed (within `ROUTE_MIN_DIST = 4.0 m`, scanning up to `ROUTE_MAX_DIST = 50.0 m`)
  and returns the front waypoint's RoadOption → time-varying command.
- `road_options` is `uint8[]`, so CARLA's `VOID = -1` arrives as 255; values outside
  1..6 fall back to the static `driving_command` parameter (also the pre-route value).

### 3.6 Output conversion (`_to_autoware_trajectory`, `_trajectory_to_markers`)
- `ego_fut_preds` (6×2, ego frame) → `autoware_planning_msgs/Trajectory` in
  `base_link`: position x/y, synthesized yaw from successive points, per-point
  `longitudinal_velocity_mps` = spacing / 0.5 s, `time_from_start` at 0.5 s steps.
- Markers transform the waypoints to the `map` frame via the odometry pose.
- CoT text (if any) published on `/orion/cot`.

### 3.7 Always-latest scheduling
- Latest-only buffers (one slot per camera, latest odom/imu/speed); old frames are
  overwritten, never queued.
- One inference in flight (timer returns early if busy → no backlog).
- Snapshot the latest of every input at dispatch.
- `require_new_frame=True` gates on a fresh `CAM_FRONT` stamp so the same data is
  never reprocessed (and the temporal memory never sees a duplicated ego pose).

### 3.8 Benchmark logging
Each run logs `prep`, `forward`, and `total` latency (ms) and approximate FPS.
For clean numbers: restart the process between fp16/fp32 runs and discard the first
1–2 inferences (CUDA warm-up / kernel autotuning).

---

## 4. Parameters

| Parameter | Default | Notes |
|---|---|---|
| `orion_repo_path` | `""` | Path to cloned Orion repo (added to sys.path; CWD). |
| `orion_config_path` | `adzoo/orion/configs/orion_stage3_agent.py` | Must define `inference_only_pipeline`. |
| `orion_checkpoint_path` | `ckpts/Orion.pth` | Trained ORION weights. |
| `precision` | `fp16` | `fp16` / `fp32` / `""` (use config defaults). |
| `camera_topics` | `[""]` | 6 raw `sensor_msgs/Image` topics, ORION order. **Required.** |
| `speed_topic` | `/carla/hero/speed` | `std_msgs/Float32`. |
| `odometry_topic` | `/carla/hero/odometry` | `nav_msgs/Odometry`. |
| `imu_topic` | `/carla/hero/imu` | `sensor_msgs/Imu`. |
| `route_topic` | `/carla/hero/global_plan` | `carla_msgs/CarlaRoute` (latched). |
| `trajectory_topic` | `/orion/predicted_trajectory` | Autoware `Trajectory`. |
| `cot_topic` | `/orion/cot` | `std_msgs/String`. |
| `inference_period_sec` | `0.05` | Dispatch-check interval (runs as fast as it can). |
| `require_new_frame` | `True` | Only run on a fresh CAM_FRONT frame. |
| `publish_cot` | `True` | Silent on the planning-only config. |
| `driving_command` | `4` | Fallback RoadOption (4 = follow-lane) before route arrives. |
| `replicate_jpeg_quality` | `20` | Replicate the agent's lossy JPEG-q20 re-encode (<=0 disables). |
| `base_frame_id` | `base_link` | Trajectory frame. |
| `map_frame_id` | `map` | Marker frame. |
| `min_trajectory_speed_mps` | `0.0` | Velocity floor in the trajectory message. |

---

## 5. Build & run

```bash
# In the ROS 2 workspace
colcon build --packages-select orion_ros
source install/setup.bash

# Edit launch/orion.launch.py paths + camera topic names, then:
ros2 launch orion_ros orion.launch.py
# or
ros2 run orion_ros orion_node --ros-args \
  -p orion_repo_path:=/path/to/Orion \
  -p orion_config_path:=/path/to/Orion/adzoo/orion/configs/orion_stage3_agent.py \
  -p orion_checkpoint_path:=/path/to/Orion/ckpts/Orion.pth \
  -p precision:=fp16 \
  -p "camera_topics:=[/carla/hero/rgb_0/image, ... 6 in ORION order ...]"
```

Watch the `_setup_model` logs — env / path / flash-attn / weight issues surface
there first.

---

## 6. Open items / to verify (not yet locked down)

1. **Camera topic names + ORION-order mapping** — the 6 `rgb_*` ids must map to
   CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT.
2. **Camera rig must match the agent** — resolution **1600×900**, and FOV/position/
   yaw matching `OrionAgent.sensors()` (front/sides FOV 70, back FOV 110, the
   specific mounts). The hardcoded `lidar2img` calibration + `ida_aug_conf`
   (H=900, W=1600) assume that exact rig; a mismatch silently degrades perception.
3. **Output frame/axis** — assumed `base_link`, x-forward / y-left. Sanity-check by
   driving straight (trajectory should extend along +x). Confirm whether the
   controller wants `base_link` or `map`, and whether it interpolates the sparse
   6-point / 3-s trajectory.
4. **flash-attn** availability on the target GPU/arch (config sets `flash_attn=True`).

---

## 6. Inference speedups (2026-09-14)

Measured on this Orin (AGX 64 GB, clocks pinned, fp16) with
`tools/bench_orion.py`, which loads the model once and applies the speedups in
`orion_ros/orion_speedups.py` cumulatively, checking the (6, 2) trajectory
against the untouched model each time. Nothing edits ORION sources; every
speedup is a post-build transform on the model object, so the container's
pre-built `/root/Orion` stays the code that runs.

Baseline profile (torch.profiler, one frame): GPU kernel time 1713 ms of a
1739 ms forward, i.e. **GPU-bound, not launch-bound**. Elementwise kernels
(rope glue, LayerNorms, window partition, casts) were ~600 ms, GEMMs ~620 ms.

| variant (cumulative)         | forward ms | ViT | LLM | map head | max Δtraj |
|------------------------------|-----------:|----:|----:|---------:|----------:|
| baseline                     | 1727 | 877 | 630 | 101 | 0 |
| + `merge_lora`               | 1639 | 879 | 545 | 101 | 1.6 mm |
| + `llm_flash_attn`           | 1549 | 878 | 456 | 101 | 5.7 mm |
| + `compile_targets=heads`    | 1520 | 877 | 456 |  73 | 4.7 mm |
| + `compile_targets=llm`      | 1449 | 878 | 386 |  73 | 7.7 mm |
| + `compile_targets=vit`      | **1007** | 434 | 387 | 74 | 6.1 mm |

Prep (CPU, per frame): six JPEG-q20 re-encodes 81 → 15 ms on a thread pool;
pipeline (`ResizeCropFlipRotImage` PIL resize + `NormalizeMultiviewImage`)
175 → 86 ms with the same per-image code mapped over threads, bit-exact.
End to end in the node (synthetic topics via `tools/fake_carla_topics.py`):
~2030 ms → ~1150 ms per frame (prep 120, forward 1032); with the map slice
and prep pipelining below, ~1000 ms between frames. Startup grows by the
compile: "ORION model loaded and ready" at ~385 s instead of ~200 s.

Node parameters (all default on; forwarded as `key:=value` by
`start_orion.sh`):
- `merge_lora` — fold the peft LoRA adapters (q/k/v/o, r=16) into the weights.
- `llm_flash_attn` — LLaMA prefill through `flash_attn_func(causal=True)`
  instead of transformers-4.31 eager attention. Batch 1, no padding, so the
  combined mask is pure causal; anything else falls back to the original.
- `compile_targets` (`heads,llm,vit`) / `compile_mode` (`default`) —
  `torch.compile` on the EVA-ViT backbone, the LlamaModel forward and the
  PETR transformer stacks. Compiles during the warm-up forward: ~3.5 min the
  first time; inductor caches to `/benchmarking/.torchinductor_cache`.
  `reduce-overhead` (CUDA graphs) was tried: no steady-state gain and it
  re-records on the first real frame (+6 min), and the head stacks are
  called several times per forward so cudagraph outputs get overwritten.
- `profile_stages` — per-stage CUDA-event timing on the inference log line.
- `decode_workers` — threads for decode and pipeline (1 = sequential).

`vit_glue` (default on) additionally rewrites the EVA-ViT blocks with the same
weights (one fused QKV GEMM, native SDPA instead of the flash_attn wrapper,
one fused W1/W2 GEMM): compiled ViT 434 -> 420 ms and half the compile time.

### TensorRT for the ViT: tried, not adopted (2026-09-14)

`tools/export_vit_onnx.py` exports the backbone (flash off, position embedding
baked; fp32 graph, parity vs the fp16 flash path 0.75 % relative) and
`tools/check_vit_engine.py` checks an engine against it. Results for the
6x3x640x640 fp16 backbone (compiled PyTorch: **420 ms**):

| build | GPU time |
|---|---:|
| TRT 8.6.2 (container), raw ONNX, level 3 | 892 ms |
| TRT 8.6.2 (container), onnxsim graph, level 5 (32 min build) | 538 ms |
| TRT 10.3 (host JetPack), onnxsim graph, level 3 | 449 ms |

TRT 8.6 on Orin has no fused attention for the eight 1600-token global blocks
(81 ms each, unfused softmax). TRT 10.3 does fuse them (`_gemm_mha_v2`), and
its profile is 197 ms GEMM + ~70 ms attention + ~180 ms LayerNorm/pointwise,
i.e. the same memory-bandwidth-bound shape as the inductor-compiled ViT. So
fp16 TensorRT buys nothing here, and INT8 could only shrink the 197 ms GEMM
share (~60-90 ms at best) at PTQ-accuracy risk, plus the engine would have to
run outside the container (TRT 8.6 inside vs 10.3 on the host; CUDA IPC is
unsupported on Tegra, so a host-side ViT service would copy ~20 MB of
features per frame through /dev/shm). Not worth it; the node keeps the
compiled PyTorch ViT. `backbone_engine` stays as a hook for a future engine.

### Exact wins after the compile work (2026-09-14, bench 7)

| change | effect |
|---|---|
| `map_head_slice` (default on): map head runs its 300 one-to-one lane queries only; the 1500 one-to-many queries are an H-DETR training trick that the head's own attention mask isolates from everything used at inference | map head 73 -> 41 ms, forward 1007 -> 980 ms, trajectory within fp16 noise |
| `pipeline_prep` (default on): frame N+1 is decoded/preprocessed on a second worker, started `pipeline_margin_ms` + avg-prep before frame N's forward is expected to end, and the finished forward starts the next one straight from the prepared batch | measured in the node with synthetic 5 Hz topics: period between forwards 1000-1016 ms for a 995-1009 ms forward, i.e. the ~140 ms prep is hidden; input age unchanged; log line gains `period=` |
| `overlap_heads` (default off): map head on a side CUDA stream during the det head | no gain (982 vs 980 ms): the heads are bound by Python launch time on one thread, not by the GPU |
| compiling `position_embeding` | worse (23 vs 13 ms, plus a recompile); not offered |
| `llm_down_proj_transpose` (default on): the K=11008 down projection weight stored [K,N] contiguous, so cuBLAS picks a 30 % faster fp16 kernel; identical numbers | LLM 405 -> 357 ms compiled, forward 980 -> 928 ms |

Remaining budget per forward (928 ms): ViT 420 (GEMM ~150 + bandwidth-bound
glue), LLM prefill ~357 (of which ~300 GEMM), heads ~110, planner ~25. Below
this it is INT8 GEMMs (next section) or a smaller model (Orion-Lite) / fewer
image tokens, not a different runtime.

### INT8 GEMMs for the LLM: available, off by default (2026-09-14)

`Int8Linear` (orion_speedups.py) does W8A8 dynamic quantisation through
`torch._int_mm` (per-token activation scale, per-channel weight scale; weight
passed as `[N,K].t()` -- with a row-major `[K,N]` weight cuBLASLt is 2.6x
SLOWER than fp16, with the transposed layout it is 1.6-2.9x faster at the
prefill shapes). Compiled, the full-decoder version takes the LLM from 357 to
240 ms (forward 810 ms); the MLP-only recipe below takes it to 276 ms
(forward 917 -> 849 ms). But the numbers move:

| recipe (tools/int8_sweep.py, 3 CARLA frames) | waypoint-feature rel. error | max Δtraj |
|---|---:|---:|
| all 224 projections, no smoothing | -- | 0.40 m |
| all, SmoothQuant α=0.5 | 18 % | 0.15 m |
| all, α=0.8, layers 0,1,30,31 fp16 | 7 % | 0.05 m |
| attention only, α=0.8, same 4 layers fp16 | 5 % | 0.02 m |
| **MLP only, α=0.8, same 4 layers fp16** (the node's recipe) | 4.4 % | 0.03 m (0.046 m on the bench frames) |

Yardsticks: every fp16 speedup above stays < 0.01 m, and the fp16 model's own
VAE sampling moves the trajectory 0.006 m between seeds. Layers 0-1 carry
100x/50x activation outliers (the rest ~10x) but skipping them alone does not
help: the error is spread over all layers. So INT8 is a real change to the
planner's input, not noise, and it needs a closed-loop route score before
it is trusted. It is wired for that experiment:

    ./start_orion.sh llm_int8:=true      # uses engines/llm_act_stats.pt (SmoothQuant
                                          # calibration from tools/int8_sweep.py --phase 0 --save-stats)

Node parameters: `llm_int8`, `llm_int8_stats`, `llm_int8_alpha` (0.8),
`llm_int8_skip_layers` ([0,1,30,31]), `llm_int8_targets` (gate/up/down).

### Lite vision: 512x512 input + staggered rear views, opt-in (2026-09-14)

`./start_orion.sh vision_lite:=true` sets `vit_input_size=512` and
`rear_view_refresh_every=2` (base model: 640 / 1). Mechanics
(orion_speedups.py): the pipeline's `ResizeMultiview3D` is pointed at 512x512
(it rescales `cam_intrinsic`/`lidar2img` itself, so the PETR 3D position
embedding follows), the EVA-ViT's global-attention rotary table is rebuilt for
the 32x32 token grid (the absolute position embedding is interpolated by the
backbone anyway), and a `StaggeredViews` wrapper sends the three rear cameras
through the ViT only every other frame, reusing their previous features in
between so the heads always get six views. Bench 14 vs the 640 base (924 ms):

| frame | forward | ViT |
|---|---:|---:|
| full (512, six views) | ~780 ms | 265 ms |
| staggered (front three views) | 646 ms | 136 ms |
| average | ~710 ms | |

In the node with synthetic 5 Hz topics: forward 790-805 ms (full) / 685-712 ms
(staggered), period between forwards 680-810 ms, i.e. ~0.75 s per frame vs
~1.0 s for the base model. Warm-up 185 s (both ViT shapes compiled).

Trajectory shift vs the base on the bench frames: **0.47 m** (INT8: 0.03-0.05 m,
fp16 changes: < 0.01 m). This is perception at a resolution the model never
trained at plus rear views that are one inference period (~0.7 s) old, so it
is a different model in effect; only a closed-loop route score can accept
it, and a 512-px fine-tune would be the way to make it safe. Startup adds one
compile per ViT shape (static per shape: `automatic_dynamic_shapes` is off,
the dynamic recompile took ~4 min per shape). The stagger cache is reset on
warm-up and on every route reset.

