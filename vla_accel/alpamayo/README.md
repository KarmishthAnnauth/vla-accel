# Alpamayo 1.5

Cosmos-Reason2-8B VLM (reasoning text) + 2.3B flow-matching action decoder (64 waypoints, 6.4 s).
Run with one front camera × 4 frames (828 prompt tokens). The fast path is `FastAlpamayo` in
`speedups.py`. It wraps the loaded model and does not modify `alpamayo1_5` or `transformers`.

## Results (Jetson AGX Orin)

Measured in the order listed, on consecutive frames. Deviation is the mean waypoint deviation from
the baseline over the 12 Bench2Drive-mini scenes. The last column counts scenes with unchanged reasoning text.

| Step | Time (s) | Δ Time (s) | Mean deviation (m) | Same reasoning |
|---|---:|---:|---:|---:|
| Baseline | 4.14 | – | – | 12/12 |
| **Common steps** | | | | |
| + LoRA merge | – | – | – | – |
| + Unused computation (inference loop) | 4.07 | -0.07 | 0.030 | 12/12 |
| + CUDA graph capture | 3.63 | -0.44 | 0.030 | 12/12 |
| + Kernel fusion (`torch.compile`) | 3.29 | -0.34 | 0.070 | 12/12 |
| + Parallel and pipelined preprocessing | 3.29 | 0.00 | 0.070 | 12/12 |
| **Model-specific steps** | | | | |
| + Draft decoding | 0.92 | -2.37 | 0.070 | 12/12 |
| + TensorRT vision encoder and prefill | **0.84** | -0.08 | 0.140 | 12/12 |
| **Optional (relative to optimized)** | | | | |
| TensorRT action head | 0.84 | 0.00 | 0.140 | 12/12 |
| INT8 LM | 0.71 | -0.13 | 2.310 | 10/12 |
| INT8 vision encoder | 0.76 | -0.08 | 4.430 | 8/12 |

- **LoRA merge:** not applicable. The model is fully fine-tuned, so there are no adapters.
- **Unused computation:** own inference loop. ViT position tables are built once and the 4 images go through one attention batch.
- **CUDA graphs:** the vision encoder, prefill and action-head decode are each replayed as one graph.
- **Kernel fusion:** element-wise chains (RoPE, SiLU-gate, RMSNorm) are fused into single kernels that are bitwise-equal to the original ops.
- **Preprocessing:** not optimized. Its share of the latency is negligible.
- **Draft decoding:** the previous frame's reasoning is checked as a draft inside the prefill. Accepted tokens skip decoding.
- **TensorRT:** fp16 engines for the vision encoder and the per-layer prefill.
- **Optional:** the TensorRT action head gives no gain (memory-bound). INT8 (SmoothQuant W8A8) is faster but moves the trajectory too far.

## Environment

Python 3.10 venv on the Orin (JetPack R36.4.7), no container.

```bash
# 1. venv with system site-packages (TensorRT 10.3.0 bindings come from JetPack)
python3.10 -m venv --system-site-packages /opt/a1_5_venv
source /opt/a1_5_venv/bin/activate

# 2. Jetson aarch64 CUDA 12.6 wheels for torch / torchvision / triton / flash-attn, then the pins
pip install -r vla_accel/alpamayo/requirements.txt

# 3. Upstream Alpamayo 1.5 (NVIDIA release), no patches needed
pip install -e "$ALPAMAYO_SRC"
```

CUDA 12.6 must be installed, plus `libcusolver` from CUDA 12.8 (`run.sh` preloads it via
`ALPAMAYO_CUSOLVER`). Point `ALPAMAYO_MODEL` at the checkpoint and `B2D_DIR` at Bench2Drive mini.

## Run

```bash
export ALPAMAYO_VENV=/opt/a1_5_venv ALPAMAYO_MODEL=... B2D_DIR=... ALPAMAYO_TRT_DIR=...
R=vla_accel/alpamayo/run.sh

$R bench --frames 3                                  # PyTorch fast path: verify + timing
$R trt_build                                         # fp16 ViT + prefill engines (~15 GB)
$R bench --frames 3 --trt-dir "$ALPAMAYO_TRT_DIR"    # optimized model
$R ablation --row 0; for r in 1 2 3 4 5 6 7 8; do $R ablation --row $r; done; $R ablation --summarize
```

Raw Orin output for every script is in `reference_results/`.
