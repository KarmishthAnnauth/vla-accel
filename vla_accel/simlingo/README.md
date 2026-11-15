# SimLingo

InternVL2-1B: InternViT-300M vision encoder + Qwen2-0.5B language model with LoRA adapters.
It takes one front camera, the ego speed and two route target points, and predicts route and speed
waypoints (no-CoT mode). The fast path is `optimize_for_inference(model)` in `fast_inference.py`.
It only acts on the loaded model object, and `verify()` checks it against the unmodified model.

## Results (Jetson AGX Orin)

Deviation is the mean deviation of the predicted speed waypoints from the baseline over Bench2Drive-mini
frames. The predicted text is identical for all frames.

| Step | Time (s) | Δ Time (s) | Mean deviation (m) |
|---|---:|---:|---:|
| Baseline | 3.77 | – | – |
| + GPU clock pinning | 0.93 | -2.84 | 0.000 |
| **Common steps** | | | |
| + LoRA merge | 0.51 | -0.42 | 0.036 |
| + Unused computation: last-position logits | 0.36 | -0.15 | 0.033 |
| + Kernel fusion (`torch.compile`) | – | – | – |
| + CUDA graph capture | 0.36 | 0.00 | 0.033 |
| + Parallel and pipelined preprocessing | 0.34 | -0.02 | 0.033 |
| **Model-specific steps** | | | |
| + KV cache | **0.25** | -0.09 | 0.038 |

- **Clock pinning:** the model is launch-bound, so the default governor drops the GPU to 408–1122 MHz. `jetson_clocks` holds it at 1300 MHz.
- **LoRA merge:** adapters are merged into the base weights (`merge_and_unload()`).
- **Last-position logits:** the LM head is computed only on the position that is read.
- **Kernel fusion:** not possible, because the container has no Triton.
- **CUDA graphs:** no gain on the vision encoder, whose kernels are large. FlashAttention-2 is replaced by SDPA so the graph can be captured.
- **Preprocessing:** the JPEG round trip, hood crop and 448×448 tiling are parallelized and pipelined.
- **KV cache:** the original `greedy_sample` re-runs the full prompt for every token. A static cache prefills once and decodes one token per step.

## Environment

Runs in a container on the Orin (JetPack R36.4.7). Nothing is installed on top of it.

```bash
# 1. Container (torch 2.2 / CUDA 12.2 Jetson build, transformers 4.46.3, no Triton)
docker pull karmishthannauth/simlingo:humble-cyclonedd

# 2. Upstream checkout, with its pretrained/InternVL2-1B cache populated (no patches needed)
git clone https://github.com/RenzKa/simlingo ~/simlingo

# 3. Weights: the simlingo_hf release, checkpoint simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
#    placed under ~/models/simlingo_hf
```

`run.sh` mounts the repo, the checkout (`SIMLINGO_DIR`), the weights (`WEIGHTS_DIR`) and Bench2Drive mini
(`B2D_DIR`) into the container.

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
export SIMLINGO_DIR=~/simlingo WEIGHTS_DIR=~/models/simlingo_hf B2D_DIR=/data/bench2drive-mini
R=vla_accel/simlingo/run.sh

$R bench --clip 0 --frames 5      # fast path vs unmodified model
$R ablation --clip 0              # cumulative step table
$R profile_baseline --clip 0      # stage profile of the unmodified model
$R clock_test --clip 0            # effect of clock pinning
```
