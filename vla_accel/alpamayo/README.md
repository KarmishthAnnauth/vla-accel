# Alpamayo 1.5

Alpamayo 1.5 (NVIDIA) is a driving VLA. A Qwen3-VL 8B vision-language model
writes a short reasoning text about the scene. A 2.3B diffusion action expert
then attends into the VLM's KV cache and turns noise into a trajectory in 5
Euler steps. The fast path is a single module, `speedups.py` (`FastAlpamayo`).
It works on the loaded `Alpamayo1_5` model and edits neither `alpamayo1_5` nor
`transformers`. Its `verify()` checks it against the model's own
`sample_trajectories_from_data_with_vlm_rollout`. The unmodified model handles
any input the fast path does not cover: batches, several samples, or
navigation text with classifier-free guidance. All numbers below use one front
camera × 4 frames (828 prompt tokens) at 560×1008.

## Acceleration steps

The unmodified rollout is decode-bound. Each reasoning token costs ~100 ms,
because every decode step reads all 12.9 GiB of weights.

| Step | What it does | Orin effect (steady scene) |
|---|---|---|
| Own inference loop | Replaces `generate()`. Uses a pre-allocated KV buffer shared by the VLM and the expert, computes `lm_head` only on rows whose logits are read, and builds the ViT position tables once with the 4 images attended as one batch | 4142 → 4269 ms (slower until graphed) |
| CUDA graphs | Captures the ViT, the fixed-size prefill window, the decode step and the whole 5-step Euler loop of the expert | → 3632 ms |
| Fused kernels | Uses `torch.compile` chains for RoPE, SiLU-gate and RMSNorm, with bf16 rounding points replayed so each kernel reproduces the stock op. Also fuses the expert layer and uses a pre-transposed `down_proj` weight. Each fusion is kept only if it matches the stock op | → 3289 ms |
| Draft decoding | Appends the previous frame's reasoning to the prompt as a draft (≤ 32 tokens) and checks it inside the prefill pass. Accepted tokens already have their K/V. Greedy output is the same token sequence that sequential decoding gives, and under sampling the distribution does not change | → 915 ms (1492 ms changing scene) |
| TensorRT fp16 (default) | Runs the ViT engine plus one engine per decoder layer for the prefill. The RMSNorms are pinned to fp32 because x² overflows fp16 | → 836 ms (1396 ms changing) |
| TensorRT INT8 prefill (opt-in) | W8A8 static Q/DQ with SmoothQuant α = 0.7. `down_proj` stays fp16. Calibrated on 60 Bench2Drive prompts | → 706 ms, 0.53 m median shift |

Two variants were measured and not adopted. INT8 including `down_proj` gave a
3.4 m median shift, and an INT8 ViT gave 1.9 m. In both cases the errors also
made the reasoning run to the 64-token cap on some scenes. For the action
expert, TensorRT is no faster than the PyTorch path (0.97 vs 1.04 ms per
layer), so the expert stays in PyTorch (`reference_results/orin_trt_probe.log`).

## Results (Jetson AGX Orin)

Jetson AGX Orin 64 GB, JetPack R36.4.7, GPU pinned at 1300 MHz. These are the
`ablation.py` results on 12 held-out Bench2Drive scenes (3 frames from each of
4 clips), with greedy decoding and the same diffusion noise in every row.
"Steady" means the scene repeats, so the draft is accepted. "Changing" means
consecutive frames come from different scenes. Deviation is, per scene, the
largest |difference| of any waypoint coordinate against row 0.

| Row | Step | Steady ms | Changing ms | Median / worst dev. | Same text |
|---|---|---:|---:|---|---:|
| 0 | reference (unmodified, greedy) | 4142 | 4142 | — | 12/12 |
| 1 | + own inference loop | 4269 | 4269 | 0.03 / 0.28 m | 11/12 |
| 2 | + CUDA graphs | 3632 | 3632 | 0.03 / 0.28 m | 11/12 |
| 3 | + fused kernels | 3289 | 3289 | 0.07 / 0.39 m | 11/12 |
| 4 | + draft decoding | 915 | 1492 | 0.07 / 0.27 m | 11/12 |
| 5 | **+ TensorRT fp16 (default)** | **836** | **1396** | 0.14 / 1.01 m | 10/12 |
| 6 | opt-in: TensorRT INT8 prefill | 706 | 1268 | 0.53 / 2.31 m | 10/12 |
| 7 | rejected: INT8 incl. `down_proj` | 1862 | 1598 | 3.41 / 8.24 m | 7/12 |
| 8 | rejected: INT8 vision encoder | 1865 | 1718 | 1.89 / 4.43 m | 8/12 |

For a steady scene the time drops from 4.14 s to 0.84 s per frame (5.0×). For
a changing scene it is 1.40 s. The sampled reference takes 4634 ms (~25
tokens). Scene 7 is a near-tie that the reference never finishes reasoning on,
and it moves ~10 m in every row.

## Setup

- **Upstream:** NVIDIA's Alpamayo 1.5 release, installed editable into the
  venv (`pip install -e "$ALPAMAYO_SRC"`). No patches are needed.
- **Weights:** the Alpamayo 1.5 checkpoint directory, set as `ALPAMAYO_MODEL`.
  The released checkpoint has no LoRA to merge.
- **Data:** Bench2Drive clips in `B2D_DIR`. Held-out clips are every sixth clip
  of the sorted directory listing, starting from the third (`[2::6]`). The
  rest are the INT8 calibration clips. The Orin runs used 24 clips: 4 held out
  and 20 for calibration. A different clip set gives a different split, and
  `--clips` overrides the held-out set.
- **Environment:** a venv (`ALPAMAYO_VENV`, originally `a1_5_venv`) on the
  Orin, with no container. It has CUDA 12.6 plus a CUDA 12.8 `libcusolver`
  preloaded, the system TensorRT 10.3.0, and the pins in `requirements.txt`.
  `run.sh` sets the library paths and the preload, puts the repo root on
  `PYTHONPATH`, keeps the inductor cache in `TORCHINDUCTOR_CACHE_DIR`, and warns
  if the GPU clock is not pinned.
- **Engines:** they are specific to the layout (1 camera, 828-token prompt)
  and take about 15 GB. TensorRT-Edge-LLM's own Alpamayo action engine does
  not build on TensorRT 10.3, so only the VLM side uses TensorRT.

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
export ALPAMAYO_VENV=... ALPAMAYO_MODEL=... B2D_DIR=... ALPAMAYO_TRT_DIR=...
R=vla_accel/alpamayo/run.sh

$R bench --frames 3 --out bench.json                          # verify + timing, PyTorch fast path
$R profile_baseline --cams 1 4 --runs 5                        # stage profile of the unmodified rollout

$R trt_build                                                   # fp16 ViT + 36 layer engines -> $ALPAMAYO_TRT_DIR
$R bench --frames 3 --trt-dir "$ALPAMAYO_TRT_DIR"              # default (row 5)

$R trt_int8 llm-sim                                            # INT8 simulation sweep (alpha, linear subsets)
$R trt_int8 llm-build --alpha 0.7 --skip down_proj --out "$ALPAMAYO_TRT_DIR-int8"       # row 6
$R trt_int8 llm-build --alpha 0.7 --down-err 1 --out "$ALPAMAYO_TRT_DIR-int8-all"       # row 7
$R trt_int8 vit-int8 --out "$ALPAMAYO_TRT_DIR-vit-int8"        # row 8 (vit_int8.engine)

$R ablation --row 0                                            # reference first
for r in 1 2 3 4 5 6 7 8; do $R ablation --row $r; done
$R ablation --summarize
```

`ablation.py` reads the engine directories for rows 5–8 from `ALPAMAYO_TRT_DIR`,
`ALPAMAYO_TRT_INT8_DIR`, `ALPAMAYO_TRT_INT8_ALL_DIR` and
`ALPAMAYO_TRT_VIT_INT8_DIR`. Each defaults to the suffixed names above. The
row-8 directory must hold `vit_int8.engine` (and no other `vit_*.engine`) next
to links to the fp16 `llm/` and `manifest.json`. All scripts take
`--b2d-dir`, `--clips`, `--frames` (per clip) and `--stride` (0 spreads frames
evenly over the clip).

## Files

| File | Purpose |
|---|---|
| `speedups.py` | `FastAlpamayo`: the fast path and `verify()` |
| `common.py` | Model load, Bench2Drive payloads (4 front frames, 16-step ego history), tokenisation, stage timer |
| `bench.py` | `verify()` against the unmodified model, then per-stage timing (steady / changing / no draft) and the reference |
| `ablation.py` | The cumulative step table above, one row per process |
| `profile_baseline.py` | Stage profile of the unmodified rollout (1 or 4 cameras) |
| `trt_common.py` | TensorRT build/run helpers and the ONNX export modules |
| `trt_build.py` | fp16 ViT and per-layer prefill engines |
| `trt_int8.py` | INT8 simulation, INT8 prefill engines, INT8 ViT engine |
| `run.sh` | Runs a module in the venv with the right library paths |
| `requirements.txt` | Venv pins |
| `reference_results/` | Raw Orin output of every script, incl. `step_ablation/` |
