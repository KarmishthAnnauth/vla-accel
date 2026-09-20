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

## ORION and SimLingo

No JSON: their numbers were recorded in the write-ups as they were measured, and
those tables are the reference.

- ORION — `ros2_ws/src/orion_ros/ORION_ROS_NODE.md` §6 (the cumulative variant
  table, per-stage) and `docs/ORION_inference_optimisation_methodology.md`.
- SimLingo — `ros2_ws/src/simlingo_ros/SIMLINGO_inference_optimisation.md` §6
  (Tables 1–3: microbenchmarks, end-to-end latency, achieved policy rate).

Re-running either bench on Thor will produce JSON in the same shape as
MindDrive's, so consider saving it here alongside.
