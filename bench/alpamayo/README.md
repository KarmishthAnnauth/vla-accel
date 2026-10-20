# Alpamayo 1.5 benches

Alpamayo 1.5 (NVIDIA, Qwen3-VL 8B VLM + a 2.3B diffusion action expert that
attends into the VLM's KV cache). Its fast path is one module,
`alpamayo_speedups.py` (`FastAlpamayo`). It works on the loaded
`Alpamayo1_5` model and edits neither `alpamayo1_5` nor `transformers`. Its
`verify()` checks it against the model's own
`sample_trajectories_from_data_with_vlm_rollout` before use. The unmodified
model is the fallback for any input the fast path does not cover: batches,
several samples, or navigation text with classifier-free guidance.

Measured on the Jetson AGX Orin 64 GB, GPU pinned at 1300 MHz, one front
camera × 4 frames (828 prompt tokens), 5 Euler steps.

## Result

`step_ablation.py` measures cumulative steps, one row per process, on 12
held-out Bench2Drive scenes, with greedy decoding and the same diffusion noise
in every row. "Steady" means the scene, and so the reasoning, repeats from the
previous frame. "Changing" means consecutive frames come from different scenes.
Deviation is, per scene, the largest |difference| of any waypoint coordinate
against row 0.

| Row | Step | Steady ms | Changing ms | Median / worst dev. | Same text |
|---|---|---:|---:|---|---:|
| 0 | reference (unmodified model, greedy) | 4142 | 4142 | — | 12/12 |
| 1 | + own inference loop | 4269 | 4269 | 0.03 / 0.28 m | 11/12 |
| 2 | + CUDA graphs | 3632 | 3632 | 0.03 / 0.28 m | 11/12 |
| 3 | + fused kernels (`torch.compile`, expert layer, `down_proj` Wᵀ) | 3289 | 3289 | 0.07 / 0.39 m | 11/12 |
| 4 | + draft decoding | 915 | 1492 | 0.07 / 0.27 m | 11/12 |
| 5 | **+ TensorRT fp16 (ViT + prefill), the default** | **836** | **1396** | 0.14 / 1.01 m | 10/12 |
| 6 | opt-in: TensorRT INT8 prefill | 706 | 1268 | 0.53 / 2.31 m | 10/12 |
| 7 | rejected: INT8 prefill incl. `down_proj` | 1862 | 1598 | 3.41 / 8.24 m | 7/12 |
| 8 | rejected: INT8 vision encoder | 1865 | 1718 | 1.89 / 4.43 m | 8/12 |

The headline is **4.14 s → 0.84 s per frame (5.0×)** for a steady scene. For a
changing scene it is 1.40 s. The sampled reference takes 4634 ms (~25 tokens).
Rows 7 and 8 are slower *on average* than row 6 because their errors make the
reasoning on two scenes run on to the 64-token cap. Scene 7 is a near-tie that
the reference never finishes reasoning on, and it moves ~10 m in every row.

## Where the time went, and what was done

From `profile_baseline.py` on the reference path (1 camera, 3380 ms on CARLA
frames with ~16 tokens):

| Stage | ms | Why | Fix |
|---|---:|---|---|
| vision encoder, 4 images | 236 | 2.4× its GEMM time | position tables computed once, the four images attend as one batch, CUDA graph; TensorRT fp16 in row 5 |
| LLM prefill, 828 tokens | 739 | compute-bound | fixed-size window under a CUDA graph; per-layer TensorRT fp16 engines in row 5 |
| LLM decode, ~16 steps | 1617 | ~100 ms/token: each step reads 12.9 GiB of weights | **draft decoding**, below |
| `lm_head` | 132 | logits for all 828 prompt rows | only the rows whose logits are read |
| `generate()` bookkeeping | 70 | | own loop |
| action expert, 5 steps | 569 | 110 ms/step for a 30 ms weight pass: launch overhead | the whole Euler loop is one CUDA graph attending into the same pre-allocated KV buffer (no copies, no crop) |

**Draft decoding** is the step that matters. A one-token decode step costs the
same as a 64-token one, because it is bound by reading the weights. So the
previous frame's reasoning text is appended to the prompt as a draft and
checked inside the prefill pass. Every draft token the model would have
produced at that position is accepted, and its K/V are already in the cache.
Sequential decoding resumes at the first disagreement. Under greedy decoding
the output is the same token sequence that sequential decoding gives. Under
sampling, draft tokens are accepted with their model probability and rejected
ones are resampled from the remainder, so the sampling distribution does not
change. A scene whose reasoning did not change costs no decode pass at all.
The draft is capped at `DRAFT_CAP = 32` tokens.

The released checkpoint has no LoRA to merge.

## TensorRT and INT8

- `trt_probe.py` asked whether TensorRT fp16 beats the PyTorch fast path at
  all, per component. It does for the prefill layers (12.3 vs 14.7 ms per
  layer) and the ViT; together that is row 5's −79 ms. For an expert layer the
  gain is marginal (0.97 vs 1.04 ms), so the expert stays in PyTorch.
- `trt_build.py` builds `vit_fp16.engine` plus one engine per decoder layer
  over the prefill window. A single 14 GiB engine would not build in memory.
  The layer-by-layer build reuses the timing cache and checks each layer on
  real hidden states. The RMSNorms are pinned to fp32 because their x² overflows
  fp16 on real activations. The engines are specific to the layout: 1 camera,
  828-token prompt, ~15 GB.
- `trt_int8.py` adds W8A8 static INT8 with explicit Q/DQ: per-channel weights,
  per-linear activation clip by output error, and SmoothQuant. The kept recipe
  is α = 0.7 with `down_proj` left in fp16 (`llm-build --alpha 0.7 --skip
  down_proj`), calibrated on 60 Bench2Drive prompts. It is opt-in only: the
  0.53 m median shift is too large for a default.
- TensorRT-Edge-LLM's own Alpamayo action engine does not build on TensorRT
  10.3 (JetPack 6). That is why only the VLM side goes through TensorRT here.

## Scripts

All of them run through `./run.sh <script> [args]`. It activates the venv,
sets the CUDA 12.6 library path and the CUDA 12.8 cuSOLVER preload, puts this
directory on `PYTHONPATH`, keeps the inductor cache out of `/tmp`, and warns if
the GPU clock is not pinned.

| Script | Purpose |
|---|---|
| `fast_check.py` | **Run first.** `verify()` against the unmodified model, then per-stage timing in three regimes (steady / changing / no draft) and the reference. `--b2d 3` uses the held-out Bench2Drive clips; the CARLA frames are too easy to judge accuracy on. `--out results/x.json` |
| `step_ablation.py` | The table above: `--row 0` first (writes the reference), then `--row 1` … `--row 8`, then `--summarize`. |
| `profile_baseline.py` | Stage profile of the unmodified rollout (`--cams 1 4 --runs 5`). |
| `microbench.py`, `opbench.py` | Single-op timings behind the fusion and layout choices. |
| `trt_probe.py` | `vit layer expert`: one engine per component vs the fast path. |
| `trt_build.py` | The fp16 engines the fast path loads from `trt_dir`. |
| `trt_int8.py` | `layer-speed`, `llm-sim` (PyTorch simulation of the quantised prefill), `llm-build`, `vit-int8`. |
| `common.py` | Model load, inputs in the node's shape, Bench2Drive payload loader, stage timing. |

## Environment

On the Orin this was a venv (`a1_5_venv`), with no container: torch 2.11
(CUDA 12.6), torchvision 0.26, transformers 4.57.1, triton 3.6, flash-attn
2.8.3, onnx 1.23.1, einops 0.8.2, and the system TensorRT 10.3.0. The
`alpamayo1_5` package from NVIDIA's Alpamayo 1.5 release is installed
editable into it.

The paths to set are `ALPAMAYO_MODEL` (weights), `VLA_FRAMES_DIR` (the frames
in `assets/frames/`), `B2D_DIR` (Bench2Drive clips for `--b2d` and
calibration), `ALPAMAYO_TRT_DIR` (engine output), and `BENCH_DIR` in `run.sh`.

Orin output for every script is in `reference_results/alpamayo/`.
