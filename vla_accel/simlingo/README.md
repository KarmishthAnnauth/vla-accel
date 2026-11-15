# SimLingo on Jetson AGX Orin

SimLingo (Renz et al., CVPR 2025) is a vision-only driving VLA built on InternVL2-1B:
an InternViT-300M encoder (308.5 M parameters) and a Qwen2-0.5B language model with
LoRA adapters (647.3 M), 957 M in total, run in bf16. From one front camera image, the
ego speed and two route target points it first generates a short text prefix
(`"Waypoints:"`, 4 tokens, for the prompt used here). A driving head then reads 30
query tokens and predicts 20 route waypoints and 10 speed waypoints. On the Orin, the
model was not compute-bound. Two causes multiplied: the GPU governor held the clock at
31–63 % of peak, and `LLM.greedy_sample` keeps no KV cache. Without a cache, every
generated token and the driving head re-run the full 543-token prompt, which costs
N + 2 full forwards per frame.

## Acceleration steps

All steps except clock pinning are in `fast_inference.py` (`optimize_for_inference(model)`).

| Step | What it does | Orin effect |
|---|---|---|
| Clock pinning | `nvpmodel -m 0 && jetson_clocks` fixes the GPU at 1300 MHz | 3538 → 938 ms/frame (**3.77×**), SD 574 → 18 ms |
| LoRA merge | `merge_and_unload()`: one matmul per linear instead of three | full LM forward 133.8 → 78.7 ms |
| FA2 → SDPA | rebinds each Qwen2 attention layer; FA2's `unpad_input` cannot be graph-captured | prefill 78.7 → 63.3 ms; enables the graph |
| KV cache | `StaticCache`: prompt prefilled once, then 1-token decode steps | decode step 47.3 ms (eager) |
| CUDA-graph decode | the decode step is captured once and replayed for every frame and prompt length | decode step 47.3 → **10.6 ms** |
| Driving head as continuation | the 30 driving queries are appended to the same cache | removes the third full forward |
| Combined inference path | all of the above | 938 → 321 ms/frame (**2.92×**) |

The decode is dispatch-bound. A single-token forward with no cache costs 52 ms for
about 1 GFLOP of work. The vision encoder is compute-bound: graph capture only takes it
from 83.3 to 81.3 ms, so that step is not used. TensorRT or ONNX export would therefore
have reached under 10 % of the original latency. `torch.compile` cannot be used because
the Orin image has no Triton. A ROS-middleware explanation for the slow baseline was
also tested; it accounted for only 274 ms and was rejected.

## Results (Jetson AGX Orin)

These are per-frame latencies, including image preprocessing and prompt building. They
were measured on CARLA frames (1024×359 → 2 tiles → 543-token prompt) in the original
closed-loop deployment, with the first 5 frames dropped. JetPack R36.4.7 in MAXN mode.

| Configuration | n | Mean ms | SD | Median | Model forward ms |
|---|---:|---:|---:|---:|---:|
| A Baseline (governor) | 56 | 3538 | 574 | 3775 | 3450 |
| B A + clock pinning | 56 | 938 | 18 | 930 | 909 |
| C B + `fast_inference` | 364 | **321** | 11 | 323 | 294 |

**11.0× overall** (3.77× from clocks, 2.92× from the inference path). The one-off
CUDA-graph capture on the first call takes about 1 s.

How C compares with the untouched model on the same input:

| Property | Result |
|---|---|
| Route waypoints, max abs. deviation | 0.028 m |
| Speed waypoints, max abs. deviation | 0.054 m |
| Generated text | identical |
| Five repeat calls / interleaved prompt lengths | bit-identical |

The deviation comes from the bf16 LoRA merge. The cache and the graph give the same
result as the original forward. `verify()` checks this on any input.

## Setup

1. **Upstream checkout.** Clone SimLingo (github.com/RenzKa/simlingo). Its
   `pretrained/InternVL2-1B` cache must be populated (the `conversation.py` from an HF
   snapshot layout is found automatically). You do not need to patch or copy anything into
   the checkout: `fast_inference.py` only acts on the loaded model object and is imported
   from this package.
2. **Weights.** Use the `simlingo_hf` release, checkpoint
   `simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`, with its
   `simlingo/.hydra/config.yaml`. The config is found three levels above the checkpoint, or
   you can pass it with `--hydra-cfg`.
3. **Bench2Drive mini.** Extract the clips into one directory (`B2D_DIR`). The input is
   built like this:
   - The `rgb_front` image is scaled to 1024 px wide, centre-cropped to 1024×512 (the CARLA
     agent's frame), converted to RGB, and the bottom 4.8/16 is cropped. This gives
     1024×359, then 2 tiles and the same 543-token prompt as the measurements.
   - The prompt uses `speed`.
   - The target points are `x/y_command_near` and `x/y_command_far`, expressed in the ego
     frame with x forward and y right. If those keys are missing, the ego's own position
     4 s and 8 s later is used.
   - Note that B2D's front camera has a 70° field of view, while SimLingo was trained on a
     110° camera. Latency does not depend on this, but the predictions are off-distribution.
4. **Environment.** Use the container `karmishthannauth/simlingo:humble-cyclonedd`, which
   is the image used for all measurements. It is built on
   `karmishthannauth/simlingo_env:v01`, which provides the Python stack. Nothing is
   installed on top; see `requirements.txt` for the pinned versions.

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks          # skip only for clock_test's unpinned run
export SIMLINGO_DIR=~/simlingo WEIGHTS_DIR=~/models/simlingo_hf B2D_DIR=/data/bench2drive-mini

./vla_accel/simlingo/run.sh bench --clip 0 --frames 5        # fast path vs untouched model
./vla_accel/simlingo/run.sh profile_baseline --clip 0        # vision / LLM / other per forward
./vla_accel/simlingo/run.sh ablation --clip 0                # cumulative step table
./vla_accel/simlingo/run.sh clock_test --clip 0              # cold vs warm vs duty-cycled clock
```

Common options are `--clip <index|name>`, `--frames`, `--stride` (default 10, which is
1 s), `--start`, `--jpeg-quality` and `--ckpt`. Inside the container these are run as
`python3 -m vla_accel.simlingo.<module>`, with `PYTHONPATH` set to the repo root.

## Files

| File | Purpose |
|---|---|
| `fast_inference.py` | `optimize_for_inference(model)` and `verify(model, example)` |
| `inputs.py` | model loading, B2D front-camera and target-point input, `DrivingInput` building, timing helper |
| `bench.py` | speed and per-frame correctness of the fast path against the untouched model |
| `profile_baseline.py` | stage profile of the untouched model, with and without language |
| `ablation.py` | cumulative ablation: LoRA+SDPA, KV cache, CUDA graph |
| `clock_test.py` | DVFS effect: cold, spun-up and duty-cycled clock |
| `run.sh` | runs a module in the container with the repo, checkout, weights and B2D mounted |
| `requirements.txt` | environment reference (container image and its pinned versions) |
