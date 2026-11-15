# MindDrive

ORION's successor. It uses the same EVA-ViT backbone, six-camera input, heads and VAE planner, with the LM
replaced by Qwen2.5-3B running as two LoRA experts. The decision expert predicts a meta-action and the action
expert predicts the trajectory, so the LM runs twice per frame. The model is GPU-bound. All transforms are in
`speedups.py` and act on the built model object. The only upstream changes are the two load patches in `docker/`.

## Results (Jetson AGX Orin)

Deviation is the mean waypoint deviation from the fp32 baseline over Bench2Drive-mini frames. Optional
configurations are each relative to the optimized model.

| Step | Time (s) | Δ Time (s) | Mean deviation (m) |
|---|---:|---:|---:|
| Baseline (fp32) | 2.69 | – | – |
| + fp16 | 2.00 | -0.69 | 0.003 |
| **Common steps** | | | |
| + LoRA merge | 1.91 | -0.09 | 0.003 |
| + Unused computation: map-head pruning | 1.84 | -0.07 | 0.003 |
| + Unused computation: one-row LM head | 1.83 | -0.01 | 0.004 |
| + Kernel fusion (`torch.compile`) | 1.23 | -0.60 | 0.003 |
| + CUDA graph capture | 1.22 | -0.01 | 0.003 |
| + Parallel and pipelined preprocessing | **0.88** | -0.34 | 0.003 |
| **Optional (relative to optimized)** | | | |
| INT8 LM | 0.81 | -0.07 | 0.171 |
| Lite vision | 0.72 | -0.16 | 0.483 |
| INT8 LM and lite vision | 0.65 | -0.23 | 0.583 |

The predicted meta-action is unchanged in all frames for every non-optional step.

- **fp16:** the reference config runs in fp32. Switching to fp16 alone saves 0.69 s.
- **LoRA merge:** the adapters of both experts are folded into the weights.
- **Map-head pruning:** drops the 1500 lane queries that are only used in training, as in ORION.
- **One-row LM head:** the decision expert needs logits for one position only, not all prompt positions over the 151 680-token vocabulary.
- **Kernel fusion:** `torch.compile` over both experts, the backbone and the head stacks. This is the largest gain.
- **CUDA graphs:** a small gain, as in ORION.
- **Preprocessing:** the six images are prepared on parallel threads, which halves the prep time, and this overlaps with the forward pass.
- **INT8 LM (optional):** SmoothQuant W8A8 (α = 0.5) on the whole LM. ORION's MLP-only recipe drifts 1.45 m on Qwen2.5.
- **Lite vision (optional):** 512 px input and the rear cameras every second frame. This also changes the meta-actions.

## Environment

Runs in a container on the Orin, built locally on top of the ORION image.

```bash
# 1. Upstream checkout
git clone https://github.com/xiaomi-mlab/MindDrive ~/MindDrive

# 2. Weights under ~/models/MindDrive/: minddrive_3b_rltrain.pth, llava-qwen2.5-3b/,
#    llava-qwen2.5-3b-eva02_petr_proj.pth
export MINDDRIVE_DIR=~/MindDrive MODELS_DIR=~/models B2D_DIR=/data/bench2drive-mini

# 3. Build the image (orion_env_ros:v01 + transformers 4.45.2 + MindDrive's import deps)
docker pull karmishthannauth/orion_env_ros:v01
vla_accel/minddrive/docker/build_image.sh            # -> karmishthannauth/minddrive_env:v01

# 4. Prepare the checkout: symlinks, compiled mmcv ops, the two patches in docker/, import check
vla_accel/minddrive/setup_checkout.sh
```

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
R=vla_accel/minddrive/run.sh

# fp32 reference, then the fp16 optimized chain against it
$R bench --precision fp32 --variants baseline --profile --ref-out /out/ref_fp32.npz
$R bench --precision fp16 --ref-in /out/ref_fp32.npz \
  --variants baseline,merge_lora,vit_glue,map_slice,down_t,vit_t,vit_win,compile_llm,compile_vit,graph_vit,logits_slice
```

Add `int8_llm` (with `--int8-calib 8`) or `vision512,stagger2` to the variants for the optional configurations.
Outputs in `/out` land in `$OUT_DIR` (default `vla_accel/minddrive/out`), and the raw Orin output is in `reference_results/`.
