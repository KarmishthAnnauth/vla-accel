# Orin reference results

Raw output from the benches on the Jetson AGX Orin 64 GB, clocks pinned.
Diff Thor's runs against these.

## MindDrive (`tools/bench_minddrive.py`, 2026-09-18)

| File | What it is |
|---|---|
| `minddrive/orin_fp32.json` | The fp32 reference run (`fp32_infer`, the config's own setting): 2318 ms forward. Everything else is compared against this. |
| `minddrive/orin_fp16_chain.json` / `.log` | The full cumulative fp16 chain down to 874 ms. The `.log` has the per-step commentary. |
| `minddrive/orin_vit_window.json` | `vit_window_nopad` — window-attention blocks on the unpadded 40×40 grid (1600 not 2304 tokens/view), exact. |
| `minddrive/orin_vision512.json` | 512×512 ViT input + staggered rear views. Opt-in: a numerics change, ~0.47 m. |
| `minddrive/orin_chain_a.json` | Plain per-token INT8, no calibration: fast, 1.19 m off. |
| `minddrive/orin_smoothquant.json` | SmoothQuant α=0.5: 824 ms, 0.41 m off. Still not adopted. |
| `minddrive/orin_recipe_r1.json` | ORION's INT8 recipe (α=0.8, MLP only, outer layers fp16) applied to MindDrive — it does *worse* (1.45 m), which is the point. |

## ORION opt-ins on Bench2Drive (`orion_ros/tools/bench_orion_b2d.py`, 2026-10-06)

Four real six-camera clips (54 frames), fed as sequences with temporal memory,
with the planner noise seeded per frame identically in every setup.
`orion/orin_b2d_<setup>.npz` holds the trajectories and timings, `.log` the
run, and `orion/orin_b2d_compare.txt` the `--compare` table:

| Setup | Forward ms | Mean / p95 / max deviation vs default |
|---|---:|---|
| default | 929 | — |
| noise (another seed; the noise floor) | 949 | 0.003 / 0.008 / 0.026 m |
| int8 (LLM) | 857 | 0.058 / 0.155 / 0.179 m |
| lite (512 px ViT + staggered rear) | 687 | 1.03 / 3.04 / 6.56 m |
| both | 622 | 1.02 / 2.99 / 6.60 m |

On real clips INT8 stays at about 6 cm. Lite vision moves plans by a metre, so
it stays off.

## ORION and SimLingo

No JSON: their numbers were recorded in the write-ups as they were measured, and
those tables are the reference.

- ORION — `ros2_ws/src/orion_ros/ORION_ROS_NODE.md` §6 (the cumulative variant
  table, per-stage) and `docs/ORION_inference_optimisation_methodology.md`.
- SimLingo — `ros2_ws/src/simlingo_ros/SIMLINGO_inference_optimisation.md` §6
  (Tables 1–3: microbenchmarks, end-to-end latency, achieved policy rate).

Re-running either bench on Thor will produce JSON in the same shape as
MindDrive's, so consider saving it here alongside.

## ReCogDrive (`bench/recogdrive/`, 2026-10-03)

| File | What it is |
|---|---|
| `recogdrive/orin_profile_baseline.json` / `.log` | Stage profile of the unmodified agent: 1260 ms, per stage and per run. |
| `recogdrive/orin_fast_check.log` | Fast path vs reference on 9 cases with the same noise (hidden states and trajectories identical), fallback, repeatability, and speed: 614 ms vs 1249 ms. |
| `recogdrive/orin_vit_engines.log` | TensorRT vision encoders: latency, feature error vs the bf16 reference, layer precisions. |
| `recogdrive/orin_vit_plan_deviation.log` | What each of those encoders does to the planned trajectory, 12 cases, same noise. |
| `recogdrive/orin_frame_timing.log` | Whole-frame time with the exact encoder and with the partial-INT8 engine. |
| `recogdrive/orin_int8_calib_alpha0.5.log`, `..._alpha0.8.log` | INT8 calibration with rebalancing: simulated feature error per group of quantised layers, per-block sensitivity. |

## Alpamayo 1.5 (`bench/alpamayo/`, 2026-10-03 – 10-12)

| File | What it is |
|---|---|
| `alpamayo/step_ablation/row_*.json`, `run_*.log`, `summary_int8.log` | The cumulative step ablation, 12 held-out Bench2Drive scenes: 4142 ms reference → 836 ms TensorRT fp16 (steady scene). `summary_int8.log` is the `--summarize` table. Row 0's `reference.pt` is not kept (it is rewritten by `--row 0`). |
| `alpamayo/orin_profile_baseline.json` / `.log` | Stage profile of the unmodified rollout, 1 and 4 cameras. |
| `alpamayo/orin_fast_check*.json` / `.log` | `fast_check.py`: on the three CARLA frames (PyTorch, `_trt`, `_trt_int8`) and on the held-out Bench2Drive clips (`_b2d_pytorch`, `_b2d_fp16`, `_b2d_int8`). |
| `alpamayo/orin_trt_probe.log` | TensorRT fp16 vs the fast path per component (ViT, one decoder layer, one expert layer). |
| `alpamayo/orin_trt_build.log` | Building and checking the fp16 ViT + per-layer prefill engines. |
| `alpamayo/orin_trt_int8_*.log` | INT8: one-layer speed, the PyTorch simulation sweep, the engine builds (kept recipe, all linears, ViT). |
| `alpamayo/orin_microbench.log`, `orin_opbench.log` | Op-level timings behind the fusion / layout choices. |
