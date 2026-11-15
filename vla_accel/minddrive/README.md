# MindDrive

MindDrive (xiaomi-mlab/MindDrive) is ORION's successor: the same EVA-ViT image backbone, PETR detection and map
heads and VAE planner, with the LLM replaced by LLaVA-Qwen2.5-3B carrying two LoRA experts. Each frame the
*decision expert* scores the meta-actions (speed / path decision) and the *action expert* produces the waypoint
features, so one forward runs the ViT (six 640x640 views), two ~570-token LLM prefills, the heads and the planner.
This folder benchmarks and accelerates that forward offline on Bench2Drive clips on a Jetson AGX Orin 64 GB.
All speedups are post-build transforms on the model object in `speedups.py`; no MindDrive source is edited
beyond the two load patches in `docker/`.

## Acceleration steps

Cumulative, in bench order (`bench.py --variants`). "Exact" means same math; the remaining drift is fp16 rounding
amplified through the planner.

| Step (variant) | What it does | Orin effect (forward ms) |
|---|---|---|
| fp32 reference | The config's own `fp32_infer` | 2318 |
| fp16 (`--precision fp16`) | Config `fp16_infer`: LLM and ViT in fp16, heads fp32 (needs patch 0002) | 1645 |
| `merge_lora` | Both LoRA adapters folded into two merged LLM copies (+6 GB GPU) | 1571 |
| `vit_glue` | EVA-ViT: fused qkv and SwiGLU w12 GEMMs, SDPA attention | 1513 |
| `map_slice` | Drop the 1500 masked one-to-many lane queries of the map head (exact) | 1451 |
| `down_t` | LLM down_proj weight stored [K,N] contiguous (exact) | 1402 |
| `vit_t` | Same layout for the ViT qkv / proj / w3 (exact) | 1381 |
| `vit_win` | Window blocks run qkv/proj on the unpadded 40x40 grid, 1600 not 2304 tokens/view (exact) | 1401, pays off once compiled |
| `compile_llm` | `torch.compile` of both expert Qwen2Models | 1345 |
| `compile_vit` | `torch.compile` of the ViT | 887 |
| `graph_vit` | Manual CUDA-graph capture of the ViT | 886 |
| `logits_slice` | Vocab projection of the one used prefill position only (exact) | **874** |
| `int8_llm` (opt-in) | W8A8 int8 LLM linears on `torch._int_mm`, SmoothQuant alpha 0.5 | 824, but 0.41 m drift |
| `vision512` + `stagger2` (opt-in) | 512x512 ViT input; rear views refreshed every 2nd frame | 705 / ~657, but 1.8-2.6 m drift |

Measured without gain and left off by default: `flash_attn` (sdpa already uses a flash kernel), `compile_heads` /
`graph_heads` (the PETR stacks are launch-bound; slower), `overlap_experts` (one 3B prefill already fills the GPU),
`int8_vit` (slower than the fp16 layout).

## Results (Jetson AGX Orin)

MAXN, clocks pinned, 3B model, fp16 chain against the fp32 reference outputs. Deviation is the max over frames of
the waypoint difference (speed trajectory / path); the decision expert's choice never changed.

| Configuration | Forward ms | Speedup | Max deviation (speed / path) | File |
|---|---:|---:|---|---|
| fp32 reference | 2318 | 1.00x | 0 | `orin_fp32.json` |
| fp16 eager | 1645 | 1.41x | 0.02 / 0.02 m | `orin_fp16_chain.json` |
| Full exact chain (to `logits_slice`, with `vit_t`, `vit_win`) | **874** | **2.65x** | 0.05 / 0.09 m | `orin_vit_window.json` |
| + INT8 LLM, plain per-token | 829 | 2.80x | 1.19 / 0.68 m | `orin_chain_a.json` |
| + INT8 LLM, SmoothQuant alpha 0.5 | 824 | 2.81x | 0.41 / 0.47 m | `orin_smoothquant.json` |
| + INT8 LLM, ORION's recipe (alpha 0.8, MLP only, outer layers fp16) | 832 | 2.79x | 1.45 / 1.75 m | `orin_recipe_r1.json` |
| 512x512 vision instead of 640 | 705 | 3.29x | 1.81 / 2.61 m | `orin_vision512.json` |

INT8 and 512 px vision are numerics changes and stay opt-in: Qwen2.5's activation outliers spread the int8 error
over all layers (unlike ORION's LLaMA), and MindDrive's planner amplifies upstream perturbations far more than
ORION's. What remains in 874 ms: ViT ~370, two LLM prefills ~170 each, heads ~110, planner and copies ~60.

The reference results were recorded on 3-4 camera frames per run before the bench was moved to Bench2Drive input;
re-running `bench.py` produces JSON in the same format.

## Setup

1. **Upstream checkout**: clone xiaomi-mlab/MindDrive to `$MINDDRIVE_DIR`.
2. **Weights** under `$MODELS_DIR/MindDrive/`: `minddrive_3b_rltrain.pth` (28 GB), `llava-qwen2.5-3b/` (LLM and
   tokenizer) and `llava-qwen2.5-3b-eva02_petr_proj.pth`.
3. **Image**: `vla_accel/minddrive/docker/build_image.sh` builds `karmishthannauth/minddrive_env:v01` from
   `karmishthannauth/orion_env_ros:v01` plus transformers 4.45.2 and MindDrive's module-level imports
   (see `requirements.txt`). The base image's develop-mode mmcv is repointed to the checkout.
4. **Checkout preparation**: `vla_accel/minddrive/setup_checkout.sh` creates the `Bench2DriveZoo` and `ckpts/`
   symlinks, copies the three compiled mmcv ops from the ORION image (identical csrc), applies
   `docker/0001-lazy-rl-runner-imports.patch` (RL/CARLA imports made optional) and
   `docker/0002-fp16-qwen-load.patch` (`fp16_infer` loads the Qwen2 class), and checks the imports in the container.
5. **Data**: Bench2Drive mini clips under `$B2D_DIR`.

## Run

```bash
export MINDDRIVE_DIR=~/MindDrive MODELS_DIR=~/models B2D_DIR=/data/bench2drive-mini
sudo nvpmodel -m 0 && sudo jetson_clocks
vla_accel/minddrive/setup_checkout.sh

# fp32 reference (also the baseline stage profile), then the fp16 chain against it
vla_accel/minddrive/run.sh bench --precision fp32 --variants baseline --profile \
    --ref-out /out/ref_fp32.npz --json-out /out/orin_fp32.json
vla_accel/minddrive/run.sh bench --precision fp16 --ref-in /out/ref_fp32.npz \
    --variants baseline,merge_lora,vit_glue,map_slice,down_t,vit_t,vit_win,compile_llm,compile_vit,graph_vit,logits_slice \
    --json-out /out/orin_fp16_chain.json

# opt-in: SmoothQuant INT8 LLM calibrated on 8 frames of the calibration clips
vla_accel/minddrive/run.sh bench --precision fp16 --ref-in /out/ref_fp32.npz \
    --variants baseline,merge_lora,vit_glue,map_slice,down_t,vit_t,int8_llm,compile_llm,compile_vit,graph_vit,logits_slice \
    --int8-calib 8 --int8-calib-out /out/int8_calib.pt --json-out /out/orin_smoothquant.json

# opt-in: 512 px vision with staggered rear views
vla_accel/minddrive/run.sh bench --precision fp16 --ref-in /out/ref_fp32.npz \
    --variants baseline,merge_lora,vit_glue,map_slice,down_t,vit_t,vision512,compile_llm,compile_vit,graph_vit,logits_slice,stagger2
```

Input: `--b2d-dir`, `--clips` (folder names; default the first `--num-clips` held-out clips of
`b2d.split_clips`), `--frames` per clip, `--stride` (10 = 1 s) and `--start`. Each clip is a sequence: temporal
memory is reset at the clip start and kept while frames are < 2 s apart. The fp32 and fp16 runs must use the same
clip and frame options. `stagger2` recompiles on its first 3-view batch, so read its per-frame lines, not the mean.
Outputs written to `/out` land in `$OUT_DIR` (default `vla_accel/minddrive/out`).

## Files

- `speedups.py`: all speedups (LoRA merge, ViT rewrites, layouts, int8/SmoothQuant, compile, CUDA graphs, 512 px
  vision, staggered views), the stage timer and the CPU-prep thread pools.
- `bench.py`: cumulative variant bench on B2D clips: stage timing, trajectory and decision parity, fp32 reference
  in/out, INT8 calibration, optional `torch.profiler` of the baseline.
- `inputs.py`: the agent's fixed camera calibration, `results` dict, host-to-device copy, and the B2D anno to
  can_bus / yaw / command mapping.
- `run.sh`: runs a module of this folder in the container. `setup_checkout.sh`: prepares the checkout.
- `docker/`: `Dockerfile`, `build_image.sh`, and the two upstream patches.
- `requirements.txt`: what the image installs on top of the ORION image.
- `reference_results/`: the Orin runs above (`*.json` summaries, `orin_fp16_chain.log` full log).
