# MindDrive — ROS 2 node on the Orin

`minddrive_ros/minddrive_node.py` runs MindDrive (xiaomi-mlab/MindDrive, ECCV'26;
ORION's successor) on live CARLA ros-bridge topics and closes the loop in-node
with the Bench2Drive decision-expert PID, publishing
`carla_msgs/CarlaEgoVehicleControl` on `/carla/hero/vehicle_control_cmd`.
It is the MindDrive counterpart of `orion_ros/orion_withpid_node.py` and keeps
that node's structure; `orion_ros/ORION_ROS_NODE.md` §2–3 describe the topics,
QoS and the odometry → `can_bus` conventions, all of which are unchanged.

Launched by `benchmarking/start_minddrive.sh` (`--3b` default, `--05b`), which
preflights everything and runs `ros2 launch minddrive_ros minddrive.launch.py`
inside the `minddrive_ros` container. Driven from carla-host by
`vla-accel/baselines/run_minddrive_ros.sh`.

## 1. Faithfulness to the reference agent

`MindDrive/team_code/minddrive_b2d_agent.py::MinddriveAgent` is reproduced by
reusing its code, never re-implementing it:

| agent | node |
|---|---|
| `setup()`: `Config.fromfile`, `build_model`, `load_checkpoint(map_location='cpu')`, `.cuda().eval()`, `Compose(inference_only_pipeline minus LoadMultiViewImageFromFilesInCeph)` | `_setup_model`, identical calls, CWD = the checkout so `./Bench2DriveZoo/ckpts/...` resolves |
| `setup()`: `lidar2img`, `lidar2cam`, `lidar2ego`, `PIDController()` | `agent_constants.py` (verbatim copies), `team_code.pid_controller_de.PIDController` |
| `tick()`: six BGR views, JPEG q20 round trip, in `CAM_FRONT … CAM_BACK_RIGHT` order | `camera_input.decode_image` (q20 done on carla-host's wire pass, `replicate_jpeg_quality:=0`) |
| `run_step()`: the `results` dict, `can_bus`, `ego_pose`, `command2hot`, pipeline, `collate`, the per-key H2D loop, `custom_wrap_fp16_model` before every forward, `model(batch, return_loss=False)` | `agent_constants.build_agent_results` / `batch_to_device` / `custom_wrap_fp16_model`, `_run_inference` |
| `run_step()`: `control_pid(pw_ego_fut_pred, ego_fut_preds, speed, local_command_xy)`, brake < 0.05 → 0, throttle > brake → brake 0, speed > 5 → throttle 0, clips | `_compute_control`, verbatim |
| `_init()`: `RoutePlanner(4.0, 50.0)` | `_compute_driving_command`, a port of `planner.py::run_step` over the latched `CarlaRoute` |

Precision defaults to the config's own (`fp32_infer=True`). `precision:=fp16`
flips the config's `fp16_infer` flag instead (LLM fp16, `img_backbone.half()`,
heads fp32): upstream's `load_model` loaded the LLaMA class under that flag
whatever `lm_model_type` said, which cannot load a Qwen2 checkpoint, so patch
0002 makes it pick the same class the fp32 branch picks.

**Speedups are opt-in** (§6). With every speedup parameter at its default
nothing is applied: the model object that runs is what `build_model` +
`load_checkpoint` return, in fp32.

## 2. What the node adds (the async closed loop)

The reference agent runs in CARLA's synchronous mode: one forward per 20 Hz
tick, `control_pid` on a plan that is 50 ms old. Here inference takes ~2.4 s and
the sim does not wait, so (as in `orion_withpid_node`):

* **Always-latest inference.** One forward in flight, dispatched on a fresh
  `CAM_FRONT` frame with the latest of every other input.
* **20 Hz control on a re-based plan.** `control_pid` runs from a timer on the
  last plan, re-cut for where the ego is now: the 6 × 0.5 s speed waypoints are
  resampled in time from the plan's age and moved into the current ego frame
  (`_rebase_speed_traj`, identical to ORION's `_rebase_plan`); the 20 path
  points, which have no time axis, are resampled by arc length from the
  projection of the current ego onto the polyline (`_rebase_path`). A plan
  older than `plan_stall_timeout_sec` (6 s, from its *arrival*) brakes the car.
* **Per-route context.** A new `CarlaRoute` (fingerprinted, so the latched
  redelivery of the same plan is ignored) arms a reset that the inference
  thread applies before its next forward: `model.test_flag = False` (the
  model's own `forward_test` then calls `reset_memory()` on both heads, exactly
  its first-frame path), `frame_idx = 0`, a fresh `PIDController`, plan
  dropped. The plan tracked at the boundary is dropped immediately in the
  callback; a forward in flight across the boundary is discarded on return;
  and no new plan is built until every camera and the odometry have *arrived*
  after the change (the buffers otherwise still hold the previous route's last
  samples — a fresh frame paired with a stale pose built the new route's first
  plan 906 m away in the synthetic test).
* **Temporal memory.** `MinddriveHead.pre_update_memory` keeps its memory only
  while consecutive `timestamp`s are < 2 s apart and `scene_token` matches. At
  ~2.6 s between consumed frames (`timestamp_mode:=sensor`, the default and
  the ORION eval's setting) the memory is zeroed every frame and the node says
  so; `timestamp_mode:=agent` feeds `frame_idx/20` instead and keeps it alive
  with a false dt. Same trade-off as ORION, deliberately left the same.

## 3. Measured (2026-09-18, 3B, fp32, clocks pinned, cold cache)

```
  +99 s  build_model         (Qwen2.5-3B shards 6.4 GB + EVA-ViT)
 +188 s  load_checkpoint     (28 GB minddrive_3b_rltrain.pth off the exfat-FUSE SSD;
                              its value_net.* keys are reported as unexpected — expected,
                              rl_training=False builds no value net)
 +196 s  on the GPU: 14.6 GB allocated, 15.7 GB peak; host RSS peaks at 44 GB
 +270 s  "MindDrive model loaded and ready."   (cold container: + colcon build)
  per frame: prep ~270 ms (six decodes + pipeline + collate), forward ~2.35–2.4 s (3.3 s cold),
             period ~2.65 s, ~54 controls published per plan at 20 Hz
```

Synthetic end-to-end run (`tools/fake_carla_topics.py`, two routes): 22
forwards, decisions logged (`speed="maintain moderate speed" path="lanefollow"`
on the static test scene), route change → plan dropped, in-flight forward
discarded, context reset, first plan of the new route within 4 m of the ego.

## 4. Parameters (launch args forwarded as `key:=value` by start_minddrive.sh)

| arg | default | |
|---|---|---|
| `variant` | `3b` | `3b` / `05b` |
| `require_new_frame` | `true` | dispatch only on a fresh front frame |
| `timestamp_mode` | `sensor` | `sensor` / `agent` (see §2) |
| `gnss_mount_offset_x` | `-1.4` | the agent localises from its GNSS mount |
| `replicate_jpeg_quality` | `20` | run_minddrive_ros.sh passes 0: q20 already on the wire |
| `control_trace` / `control_trace_path` | `false` / auto | per-tick CSV under `minddrive_env/logs/` |
| `decode_workers` | `6` | threads for the six decodes (bit-exact) |

Node-only parameters (edit the launch file): `control_hz` 20, `plan_stall_timeout_sec` 6,
`brake_when_stale` true, `driving_command` 4 (fallback before a route arrives),
`camera_sync_tolerance_sec` 0.1, `meta_action_topic` `/minddrive/meta_action`.

## 5. Environment

`karmishthannauth/minddrive_env_ros:v01` = the ORION image + transformers
4.45.2 + the modules MindDrive imports at module level (`--no-deps`; see
`minddrive_env/Dockerfile`). The checkout is bind-mounted at
`/benchmarking/MindDrive` with the ORION build's three compiled `mmcv` `.so`
files (identical `csrc`) and two patches (`minddrive_env/0001-lazy-rl-runner-imports.patch`,
`0002-fp16-qwen-load.patch`).
`minddrive_env/smoke_infer.py` loads the model exactly as the agent does and
runs a few forwards on a black frame: the first thing to run when something
about the environment is in doubt.

## 6. Inference speedups (2026-09-18)

`minddrive_ros/minddrive_speedups.py`, measured by `tools/bench_minddrive.py`
(loads once, applies cumulatively, three real CARLA frames, every step checked
against the fp32 reference outputs saved by the fp32 run):

```
docker exec minddrive_ros bash -c "$(cat /benchmarking/minddrive_env/logs/bench_cmd.sh) \
    --precision fp32 --frames 3 --variants baseline --ref-out .../ref_fp32.npz"
docker exec minddrive_ros bash -c "$(cat /benchmarking/minddrive_env/logs/bench_cmd.sh) \
    --precision fp16 --frames 3 --ref-in .../ref_fp32.npz --variants baseline,merge_lora,..."
```

| step (cumulative) | forward ms | ViT | LLM decision + action | max Δ speed traj | max Δ path | decisions |
|---|---:|---:|---:|---:|---:|---|
| fp32 reference (`fp32_infer`, the config) | **2318** | 957 | 567 + 534 | 0 | 0 | 3/3 |
| `precision:=fp16` (`fp16_infer`, needs patch 0002) | 1645 | 877 | 250 + 258 | 0.021 m | 0.021 m | 3/3 |
| `merge_lora:=true` (two merged expert copies, +6 GB GPU) | 1571 | 879 | 245 + 226 | 0.023 m | 0.017 m | 3/3 |
| `vit_glue:=true` (fused qkv / w12, SDPA) | 1513 | 831 | 238 + 224 | 0.054 m | 0.045 m | 3/3 |
| `map_head_slice:=true` (300 one-to-one queries; exact) | 1451 | 831 | | 0.063 m | 0.040 m | 3/3 |
| `down_proj_t:=true` (exact) | 1402 | | 212 + 197 | 0.063 m | 0.040 m | 3/3 |
| `compile_targets:=llm` | 1353 | | 187 + 174 | 0.061 m | 0.039 m | 3/3 |
| `compile_targets:=llm,vit` | 939 | 418 | | 0.040 m | 0.030 m | 3/3 |
| `cuda_graph_vit:=true` | 937 | 416 | | 0.040 m | 0.030 m | 3/3 |
| `logits_slice:=true` (one-row vocab projection; exact) | **924** | 416 | 174 + 174 | 0.040 m | 0.030 m | 3/3 |
| `vit_weight_t:=true` (EVA-ViT qkv/proj/w3 stored [K,N]; exact; 2026-09-18 pm, 4 frames) | 910 | 405 | 172 + 170 | 0.068 m | 0.092 m | 4/4 |
| `vit_window_nopad:=true` (window blocks: qkv/proj on the unpadded 40x40 grid, 1600 not 2304 tokens/view; exact) | **874** | 372 | 172 + 170 | 0.049 m | 0.085 m | 4/4 |

Second round (2026-09-18 pm, same bench, 4 frames vs a regenerated fp32
reference): the fp16 GEMMs of the ViT and the LLM run at 20-30 TFLOPS, and
cuBLASLt's int8 tensor-core path (`torch._int_mm`) is 1.4-2x faster at the
same shapes, so W8A8 int8 linears were added (`int8_targets:=llm`, per-channel
weights, per-token dynamic activations, freed fp16 weights: -3 GB per expert).

| step (cumulative on the row above) | forward ms | ViT | LLM decision + action | max Δ speed traj | max Δ path | decisions |
|---|---:|---:|---:|---:|---:|---|
| `int8_targets:=llm`, no calibration | 829 | 404 | 131 + 132 | 1.19 m | 0.68 m | 4/4 |
| + `int8_targets:=llm,vit` (qkv + proj) | 847 | 423 | | 1.09 m | 1.11 m | 4/4 |

The ViT int8 is slower than the best fp16 layout (the w12/w3 shapes gain
nothing in int8 and the quant passes cost more than the qkv/proj GEMMs save):
dropped.  The LLM int8 is 80 ms faster but plain per-token quantisation moves
the trajectory by a metre, an order of magnitude above the fp16 noise
(0.07 m), although the decision expert's choice never changed.  Eager subset
runs (`--int8-skip`) put the error in both halves of the decoder, MLP
dominant: attention only 0.33 m, MLP only 1.13 m, all but down_proj 1.33 m --
the activation-outlier signature Qwen2.5 is known for, hence SmoothQuant
(`--int8-calib N --int8-calib-out <pt>` in the bench, `int8_calib_path:=` in
the node): per-input-channel |x| maxima from calibration forwards, part of
the range migrated into the weight columns, the activation rescaled at run
time (fused by inductor).

| LLM int8 variant (eager unless noted; calibrated on 2 bench frames, i.e. in-sample) | forward ms | max Δ speed traj | max Δ path |
|---|---:|---:|---:|
| plain per-token (no calibration), compiled chain | 829 | 1.19 m | 0.68 m |
| SmoothQuant α=0.5, compiled chain (`/benchmarking/minddrive_int8_calib.pt`) | **824** | 0.41 m | 0.47 m |
| SmoothQuant α=0.7 | | 0.49 m | 0.83 m |
| SmoothQuant α=0.85 | | 1.07 m | 0.85 m |
| SmoothQuant α=0.3 | | 1.29 m | 0.44 m |
| α=0.5, weights int8 only (`--int8-mode w8`, fp16 GEMM, parity probe) | | 0.10 m | 0.40 m |
| α=0.5, activations int8 only (`--int8-mode a8`, parity probe) | | 1.01 m | 0.44 m |
| ORION's recipe: α=0.8, MLP only, layers 0,1,34,35 fp16 (`--int8-skip self_attn,:0:,:1:,:34:,:35:`), compiled | 832 | 1.45 m | 1.75 m |
| same skips, α=0.5 | | 0.34 m | 0.64 m |
| layers 0,1,34,35 fp16 only, all projections, α=0.5 | | 0.58 m | 0.43 m |

ORION's LLaMA-7B took that recipe at 0.03-0.05 m; MindDrive's Qwen2.5-3B does
not (its int8 error is not concentrated in the outer layers or in attention),
so int8 stays a numerics decision here.  `--int8-skip` matches anywhere in the
linear key `llm:<expert>:<layer>:<name>`.

### 512x512 vision (ORION's "lite vision"), opt-in, 2026-09-18

`vit_input_size:=512` (+ `rear_view_refresh_every:=2`): same mechanics as
ORION's (pipeline `ResizeMultiview3D` -> 512, which rescales the intrinsics so
the PETR position embedding follows; the EVA-ViT global rope rebuilt for the
32x32 grid; the window blocks need no padding at 32x32; `StaggeredViews`
sends the rear cameras through the ViT every other frame; `CudaGraphed` keeps
one graph per batch shape).  Bench, fp16 chain + `vit_weight_t`, 4 frames vs
the fp32 640 reference:

| variant | forward ms | ViT | max Δ speed traj | max Δ path | decisions |
|---|---:|---:|---:|---:|---|
| 640 (adopted chain) | 910 | 405 | 0.07 m | 0.09 m | 4/4 |
| 512, six views, compiled + graphed | **705** | 211 | 1.81 m | 2.61 m | 4/4 |
| 512 + rear views every 2nd frame (steady state: full 705 / staggered 609, avg ~657) | ~657 | 211 / 116 | 1.94 m | 2.67 m | 4/4 |

ORION took the same change at 0.47 m; MindDrive's planner moves its
trajectory by 2-3 m (the same amplification seen under int8: its output is
far more sensitive to upstream perturbation than ORION's).  Off by default;
a closed-loop route score is the only way to accept it, and a 512-px
fine-tune the only way to make it safe.  Note for the bench: the first
3-view batch of `stagger2` triggers a recompile inside the timed frames
(`automatic_dynamic_shapes` is off), so use the per-frame lines, not the mean.

So the activation side sets the speed-trajectory error and both sides carry
the path error; SmoothQuant halves the damage but does not reach the fp16
floor.  **`int8_targets:=llm` is therefore available but NOT in `--fast`**:
adopting it is a numerics decision (824 vs 910 ms for ~0.4 m of waypoint
movement on these frames, decisions unchanged), to be made against a
closed-loop score, not the bench.  The next lever if it is wanted: LLM.int8()
style outlier decomposition (the few calibrated outlier channels in a small
fp16 GEMM, the rest int8) for the activation side, and a per-linear error
ranking to keep the most sensitive projections fp16.

Measured and not adopted (no gain): `llm_attn:=flash_attention_2` (the sdpa
path transformers picks already dispatches to a flash kernel; the swap changed
nothing measurable), `compile_targets` including `heads` and
`cuda_graph_heads` (the PETR stacks are Python-launch-bound: map head 37 →
44 ms, det head 59 → 62 ms), `overlap_experts:=true` (the decision-expert
prefill on a side stream, rounds reordered: both prefills stretch to ~350 ms,
total unchanged -- one 3B prefill already saturates the GPU).

What is left in the 874 ms: ViT 372 (memory-bound elementwise glue plus GEMMs,
the same backbone ORION's TensorRT experiment found no faster route for),
two LLM prefills of ~570 tokens at ~171 each, heads ~110, planner + copies ~60.
INT8 GEMMs (second round below) buy 86 ms on the LLM at a numerics cost and
nothing on the ViT; inductor refuses GEMM autotuning on the Orin's 16 SMs
(`max-autotune` is a no-op) and prompt prefix caching is worthless because
513 of the ~570 LLM tokens are the per-frame detection and map queries.
Below this it is a smaller model, not a different runtime.

The trajectory deltas are fp16 rounding amplified through the VAE planner
(they go up and down as later steps change kernels), never a systematic shift;
the decision expert's discrete choice never changed on the bench frames. Still,
`--fast` is not the reference numerics: results from it are labelled.

In the node (`start_minddrive.sh --fast`, prep pipelined behind the forward):
forward ~950 ms, **period ~960 ms** between forwards (was 2650 ms in fp32),
GPU peak 18.5 GB, ready after 360 s with a warm inductor cache. Because
consumed frames are now <2 s apart, the heads' temporal memory is *retained*
frame to frame under `--fast`; the default fp32 run zeroes it every frame
(`temporal memory` warning in the log).
