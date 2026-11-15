# ReCogDrive

InternVL-2B VLM (front camera, ego history, one-hot navigation command) + a transformer diffusion planner
conditioned on the VLM's last hidden state (8 waypoints over 4 s, 5 denoising steps). The fast path is
`FastReCogDrive` in `speedups.py`, a wrapper around the built agent that does not modify upstream code.
Any input it does not cover falls back to the unmodified agent.

## Results (Jetson AGX Orin)

Deviation is the mean waypoint deviation from the baseline over Bench2Drive-mini frames. Optional
configurations are each relative to the optimized model.

| Step | Time (s) | Δ Time (s) | Mean deviation (m) |
|---|---:|---:|---:|
| Baseline | 1.27 | – | – |
| **Common steps** | | | |
| + LoRA merge | – | – | – |
| + Unused computation: vocabulary projection | 1.20 | -0.07 | 0.000 |
| + Kernel fusion (`torch.compile`) | 0.80 | -0.40 | 0.000 |
| + CUDA graph capture | 0.76 | -0.04 | 0.000 |
| + Parallel and pipelined preprocessing | 0.73 | -0.03 | 0.000 |
| **Model-specific steps** | | | |
| + Constant-prompt tokenization | 0.70 | -0.03 | 0.000 |
| + KV cache | **0.66** | -0.04 | 0.000 |
| **Optional (relative to optimized)** | | | |
| TensorRT fp16 vision encoder | 0.64 | -0.02 | 0.048 |
| TensorRT INT8 vision encoder, selected layers | 0.60 | -0.06 | 0.195 |
| TensorRT INT8 vision encoder, all layers | 0.58 | -0.08 | 0.267 |

- **LoRA merge:** not applicable. The model is fully fine-tuned.
- **Vocabulary projection:** no text is generated, so the LM head over 2800 positions × 152k tokens is skipped.
- **Kernel fusion:** RMSNorm, RoPE, SiLU-gate and scaling chains are fused, and the planner's denoising step is compiled. This is the largest gain.
- **CUDA graphs:** the vision encoder, LM and planner are each replayed as one graph.
- **Preprocessing:** the two front-frame resizes run on parallel threads. Conversion and normalization of the 9 tiles is one GPU op, with pixels identical to the reference.
- **Constant-prompt tokenization:** the fixed 291-token prefix is tokenized once, and only the text after the image is tokenized per frame.
- **KV cache:** the prefix K/V are computed once. The planner's cross-attention K/V are computed once per frame instead of once per denoising step.
- **TensorRT (optional):** the vision encoder is compute-bound, so engines help, but fp16 instead of bf16 shifts the trajectory. INT8 uses SmoothQuant Q/DQ.

## Environment

Python 3.10 venv on the Orin (JetPack R36.4.7, CUDA 12.6), no container. Needs [`uv`](https://github.com/astral-sh/uv).

```bash
# 1. Upstream checkout (no patches needed)
git clone https://github.com/xiaomi-research/recogdrive /opt/recogdrive

# 2. Weights: ReCogDrive-VLM-2B and ReCogDrive_Diffusion_Planner_2B_RL.ckpt in one folder
export RECOGDRIVE_VENV=~/venvs/recogdrive RECOGDRIVE_REPO=/opt/recogdrive \
       RECOGDRIVE_WEIGHTS=/data/weights/ReCogDrive B2D_DIR=/data/bench2drive-mini

# 3. Build the venv: Jetson torch/triton/flash-attn wheels, the pins, the nuplan chain (--no-deps), import check
cd vla_accel/recogdrive && ./env/run_capped.sh ./env/setup_venv.sh
```

The venv uses system site-packages for JetPack's TensorRT. `env/stubs/decord` replaces `decord`, which has no aarch64 wheel.

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
V=vla_accel/recogdrive

$V/run.sh profile_baseline                     # stage profile of the unmodified agent
$V/run.sh bench --clips 3 --frames 3           # fast path vs reference: exactness and speed

# optional TensorRT vision encoder
$V/run.sh trt_vit export && $V/run.sh trt_vit build fp16
$V/run.sh bench --vit-engine $RECOGDRIVE_WEIGHTS/trt/vit_fp16.engine --no-ref-bench
```

The INT8 engines are built with `vit_int8.py` (calibration, SmoothQuant, Q/DQ ONNX) and then `trt_vit build qdq`.
Raw Orin output is in `reference_results/`.
