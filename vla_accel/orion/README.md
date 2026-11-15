# ORION on Jetson AGX Orin

ORION (Fu et al., ICCV 2025; [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion)) is a Bench2Drive
driving agent. Each frame it takes six 1600×900 cameras, an 18-d CAN-bus vector and a navigation command,
and outputs a 6-waypoint, 3 s ego trajectory. The inference path is an EVA-ViT-L backbone (640×640 input,
24 blocks), a PETR-style 3D detection head and a map head (1800 lane queries), a LLaVA-LLaMA-7B with rank-16
LoRA on attention that does one planning prefill of about 600 tokens, and a VAE trajectory decoder fed by the
LLM state at a waypoint token. The backbone and LLM run in fp16 and the heads in fp32, as in the reference
agent. On the Orin the forward pass is GPU-bound: 1713 ms of a 1739 ms forward is kernel time, about half
of it memory-bound element-wise work. Every speedup in this folder is applied to the built model object, so
the upstream sources are not changed.

## Acceleration steps

The first nine steps are on by default. They change only the fp16 rounding: the trajectory moves less than
1 cm, while two sampler seeds of the unmodified model differ by 0.6 cm.

| Step | What it does | Orin effect |
|---|---|---|
| `merge_lora` | Merges the peft LoRA adapters into the LLaMA weights | LLM 636 → 555 ms |
| `flash_attn` | Runs the batch-1 causal prefill through `flash_attn_func` instead of the eager fp32 attention in transformers 4.31 | LLM 555 → 467 ms |
| `compile_heads` | `torch.compile` on the PETR transformer stacks of both heads, without CUDA graphs | map head 101 → 73 ms |
| `compile_llm` | `torch.compile` on the LlamaModel forward | LLM → 386 ms |
| `compile_vit` | `torch.compile` on the EVA-ViT backbone | ViT 883 → 434 ms |
| `vit_glue` | Fused QKV GEMM, SDPA and fused SwiGLU W1/W2 in the ViT, with the same weights | ViT 434 → 420 ms, half the compile time |
| `map_slice` | Keeps only the 300 one-to-one lane queries; the attention mask already isolates the 1500 one-to-many queries, so the result is exact | map head 73 → 42 ms |
| `down_t` | Stores the K=11008 `down_proj` weight as [K,N] contiguous so cuBLAS picks a faster fp16 kernel | LLM 405 → 357 ms |
| parallel prep | Runs the six JPEG re-encodes and the per-image resize and normalise loops on a thread pool, with bit-identical output | decode 81 → 15 ms, pipeline 175 → 86 ms |
| `int8_llm` (opt-in) | W8A8 `torch._int_mm` on the MLP projections, SmoothQuant α=0.8, layers 0, 1, 30 and 31 kept in fp16 | forward 929 → 857 ms, trajectory moves 6 cm (mean) |
| `vision_lite` (opt-in) | 512×512 ViT input, and the rear cameras go through the ViT only every second frame | forward 929 → 687 ms, trajectory moves 1.0 m (mean): kept off |
| `overlap_heads` (tried) | Runs the map head on a side CUDA stream during the det head | no gain (982 vs 980 ms); the heads are bound by Python launch time |
| `compile_posembed` (tried) | `torch.compile` on the 3D position embedding | slower (23 vs 13 ms) |
| `trt_vit` (tried) | TensorRT engine for the ViT (`export_vit_onnx.py`, `check_vit_engine.py`) | TRT 8.6.2: 892 ms, or 538 ms with onnxsim and level 5. TRT 10.3: 449 ms. Compiled PyTorch: 420 ms. Not adopted |

TensorRT gains nothing because the ViT is memory-bound. The TRT 10.3 profile (197 ms GEMM, about 70 ms
attention, about 180 ms norm and point-wise) has the same shape as the profile of the Inductor-compiled ViT.
For INT8, quantising all 224 projections without smoothing moves the trajectory 0.40 m. SmoothQuant with
α=0.5 brings this to 0.15 m, and the MLP-only recipe with four fp16 layers to 0.03 m.

## Results (Jetson AGX Orin)

The Orin was a 64 GB AGX on JetPack R36.4.7, with the GPU pinned at 1.3 GHz and fp16 inference.
Cumulative forward time per frame in ms, from `bench.py`:

| Step | Forward | ViT | LLM | Map head | Max Δ trajectory |
|---|---:|---:|---:|---:|---:|
| Baseline | 1736 | 883 | 636 | 101 | – |
| + merge_lora | 1651 | 882 | 555 | 101 | 1.6 mm |
| + flash_attn | 1567 | 885 | 467 | 102 | 5.7 mm |
| + compile_heads | 1529 | 884 | 462 | 73 | 4.7 mm |
| + compile_llm | 1449 | 878 | 386 | 73 | 7.7 mm |
| + compile_vit | 1007 | 434 | 387 | 74 | 6.1 mm |
| + vit_glue | 1004 | 420 | 399 | 73 | 7.7 mm |
| + map_slice | 980 | 422 | 406 | 42 | 4.7 mm |
| + down_t | **928** | 419 | 357 | 42 | 4.7 mm |

With preparation included, a frame takes 2030 ms at baseline and about 1000 ms optimised: prep drops from
275 to 120 ms, and pipelining it behind the previous forward hides it. These cumulative numbers were
measured on CARLA camera frames, before `bench.py` switched to Bench2Drive input.

Default and opt-in setups on four Bench2Drive clips (54 frames). Each clip runs as a sequence with temporal
memory, and every setup uses the same per-frame seed. The data is in `reference_results/`, from
`bench_b2d.py --compare`:

| Setup | Forward ms | Prep ms | Deviation from default, mean / p95 / max |
|---|---:|---:|---|
| default | 929 | 207 | – |
| noise (another seed: the noise floor) | 949 | 207 | 0.003 / 0.008 / 0.026 m |
| int8 | 857 | 207 | 0.058 / 0.155 / 0.179 m |
| lite | 687 | 196 | 1.03 / 3.04 / 6.56 m |
| both | 622 | 196 | 1.02 / 2.99 / 6.60 m |

## Setup

- **Upstream checkout**: the image `karmishthannauth/orion_env_ros:v01` ships it pre-built at `/root/Orion`,
  with compiled mmcv ops. To use your own build instead, mount it with `ORION_REPO_HOST`; it needs the
  compiled `mmcv._ext`.
- **Weights**: `Orion.pth` (38 GB) and `pretrain_qformer/`, which holds the base LLM and tokenizer, both in
  one folder `$WEIGHTS_DIR`. It is mounted at `/models/Orion`, and `run.sh` links
  `/root/Orion/ckpts/pretrain_qformer` to it when that link is missing. Loading takes about 200 s, and the
  first compile adds 2–4 min. The Inductor cache is kept in `$CACHE_DIR`.
- **Patches**: none. All changes are runtime transforms in `speedups.py`.
- **Data**: the Bench2Drive mini clips, extracted to `$B2D_DIR` (`<clip>/camera/rgb_*/%05d.jpg`,
  `<clip>/anno/%05d.json.gz`).
- **Environment**: Docker with the NVIDIA runtime on the Orin. See `requirements.txt`; nothing is installed on
  top of the image.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
export WEIGHTS_DIR=/path/to/models/Orion B2D_DIR=/path/to/bench2drive-mini
```

## Run

From the repo root:

```bash
R=vla_accel/orion/run.sh; OUT=vla_accel/orion/results

# Baseline stage profile (torch.profiler, one frame)
$R bench --variants baseline --profile

# Cumulative ablation with per-stage time and max Δ trajectory vs the untouched model
$R bench --compile-mode default \
  --variants baseline,merge_lora,flash_attn,compile_heads,compile_llm,compile_vit,vit_glue,map_slice,down_t

# INT8 SmoothQuant calibration stats (engines/llm_act_stats.pt) and the recipe sweep
$R int8_sweep --phase 0 --save-stats vla_accel/orion/engines/llm_act_stats.pt
$R int8_sweep --phase 1

# Default and opt-in setups on Bench2Drive sequences, then the comparison
for s in default int8 lite both; do $R bench_b2d --setup $s --out $OUT/$s.npz; done
$R bench_b2d --setup default --seed-offset 1000 --out $OUT/noise.npz
$R bench_b2d --compare $OUT

# TensorRT ViT (negative result)
$R export_vit_onnx --out vla_accel/orion/engines/orion_vit.onnx
$R trtexec --onnx=vla_accel/orion/engines/orion_vit.onnx --saveEngine=vla_accel/orion/engines/orion_vit_fp16.plan \
  --fp16 --builderOptimizationLevel=5 --memPoolSize=workspace:8192M
$R check_vit_engine
$R bench --compile-mode default --variants baseline,trt_vit
```

Every bench takes the shared options `--b2d-dir`, `--clips` (names or indices), `--split all|calib|held`,
`--frames`, `--stride` and `--start`. `bench.py` uses the first held-out clip by default, `int8_sweep.py` the
first calibration clip, and `bench_b2d.py` all clips.

## Files

- `speedups.py`: all acceleration transforms (LoRA merge, flash prefill, compile, ViT glue, map slice,
  down_proj layout, INT8 and SmoothQuant, lite vision, TensorRT backbone, stage timer, parallel prep).
- `model.py`: builds the model, config and pipeline from the upstream checkout; paths come from env vars.
- `inputs.py`: camera calibration and command helpers, and the Bench2Drive frame → ORION batch path.
- `bench.py`: cumulative variant bench with the baseline profile and the trajectory check.
- `bench_b2d.py`: default and opt-in setups on Bench2Drive sequences, plus `--compare`.
- `int8_sweep.py`: INT8 recipe sweep; writes `engines/llm_act_stats.pt`.
- `export_vit_onnx.py`, `check_vit_engine.py`: ONNX export and TensorRT engine check for the ViT.
- `run.sh`: runs a module in the container. `requirements.txt` describes the environment.
- `reference_results/`: Orin logs, `.npz` trajectories and the comparison table from `bench_b2d.py`.
