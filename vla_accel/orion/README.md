# ORION

EVA-ViT-L vision backbone (six 1600×900 cameras, resized to 640×640) + 3D detection and map heads +
LLaVA-LLaMA-7B with LoRA adapters + VAE planner (6 waypoints, 3 s). The model is GPU-bound, not
launch-bound. All transforms are in `speedups.py` and act on the built model object, so the upstream
sources are unchanged.

## Results (Jetson AGX Orin)

Deviation is the mean waypoint deviation from the baseline over Bench2Drive-mini frames. Optional
configurations are each relative to the optimized model.

| Step | Time (s) | Δ Time (s) | Mean deviation (m) |
|---|---:|---:|---:|
| Baseline (fp16) | 2.03 | – | – |
| **Common steps** | | | |
| + LoRA merge | 1.94 | -0.09 | 0.003 |
| + Unused computation: map-head pruning | 1.87 | -0.07 | 0.003 |
| + Kernel fusion (`torch.compile`) | 1.23 | -0.64 | 0.004 |
| + CUDA graph capture | 1.25 | +0.02 | 0.004 |
| + Parallel and pipelined preprocessing | **0.96** | -0.29 | 0.005 |
| **Optional (relative to optimized)** | | | |
| INT8 LM | 0.87 | -0.09 | 0.131 |
| Lite vision | 0.73 | -0.23 | 0.563 |
| INT8 LM and lite vision | 0.67 | -0.29 | 0.663 |

- **LoRA merge:** adapters are folded into the LM weights, which halves the matrix products in the attention projections.
- **Map-head pruning:** drops 1500 of the 1800 lane queries. They are a training aid that is still computed and then masked at inference.
- **Kernel fusion:** `torch.compile` fuses the memory-bound element-wise chains in the LM, the vision backbone and the map head. This is the largest gain.
- **CUDA graphs:** no gain, as expected for a GPU-bound model.
- **Preprocessing:** the six images are prepared in parallel (0.29 → 0.11 s), and this overlaps with the previous forward pass.
- **INT8 LM (optional):** SmoothQuant W8A8 on the LM's MLP layers. The first two and last two layers stay fp16.
- **Lite vision (optional):** 512×512 input with the padding removed (36 % fewer tokens), and the rear cameras run only every second frame.

## Environment

Runs in a container on the Orin (JetPack R36.4.7). The image ships a pre-built ORION checkout at
`/root/Orion` with the compiled mmcv ops, and nothing is installed on top of it.

```bash
# 1. Container (torch 2.4 / CUDA 12, transformers 4.31, flash-attn 2.7, TensorRT 8.6.2)
docker pull karmishthannauth/orion_env_ros:v01

# 2. Weights: Orion.pth and pretrain_qformer/ (base LLM + tokenizer) from the upstream release
#    (https://github.com/xiaomi-mlab/Orion), in one folder
export WEIGHTS_DIR=/path/to/models/Orion B2D_DIR=/path/to/bench2drive-mini
```

`run.sh` mounts the weights, Bench2Drive mini and an Inductor cache (`CACHE_DIR`) into the container. To use your
own ORION build, set `ORION_REPO_HOST` (it needs the compiled `mmcv._ext`). Loading takes about 200 s, and the
first compile adds 2–4 min.

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
R=vla_accel/orion/run.sh; OUT=vla_accel/orion/results

$R bench --variants baseline --profile      # baseline stage profile
$R bench --compile-mode default \
  --variants baseline,merge_lora,flash_attn,compile_heads,compile_llm,compile_vit,vit_glue,map_slice,down_t
$R int8_sweep --phase 0 --save-stats vla_accel/orion/engines/llm_act_stats.pt   # INT8 calibration
for s in default int8 lite both; do $R bench_b2d --setup $s --out $OUT/$s.npz; done
$R bench_b2d --compare $OUT                 # optional configurations vs default
```

Raw Orin output is in `reference_results/`.
