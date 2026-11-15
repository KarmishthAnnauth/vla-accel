# ReCogDrive

ReCogDrive (xiaomi-research/recogdrive) is a NAVSIM planning agent. It plans from one front camera image
at 1920×1080. The image is cut into 9 tiles of 448² and encoded by InternViT-300M into 2304 image tokens. A
prompt (a 291-token system prompt, the image tokens, and the ego history and navigation command) is
left-padded to 2800 tokens and run through Qwen2.5-1.5B (InternVL3-2B, bf16) in **one forward pass**: no
tokens are generated. A 16-block fp32 diffusion transformer then denoises an 8-pose trajectory over 4 s in
5 DDIM steps, cross-attending to those 2800 hidden states. The checkpoint has no LoRA. The acceleration is
`FastReCogDrive` (`speedups.py`), a wrapper around the built `ReCogDriveAgent`. It edits no upstream
source, and its VLM output is **bit-identical** to the reference. Any input it does not cover (a frame that
is not 16:9, or a prompt longer than the window) falls back to the unmodified agent.

## Acceleration steps

| Step | What it does | Orin effect |
|---|---|---|
| Skip the LM head | Calls the decoder stack directly: only `hidden_states[-1]` is used, so the 2800 × 151k logits and their fp32 cast are never computed | LM head 58 ms and most of "other" (38 ms) removed |
| Prefix KV cache | The 291 system-prompt tokens are encoded once, from a full 2800-token stock pass so the bf16 rows match. The padding row is stored too | Part of LLM 544 → 392 ms |
| Fixed window + CUDA graphs | The LLM runs on 2304 image slots + 232 tail slots (static shapes). ViT, LLM and planner are each one CUDA graph | Step 1 total: 1255 → 769 ms |
| Planner context K/V once | Cross-attention K/V of the VLM tokens are computed once per frame instead of once per DDIM step | Planner 283 → 45 ms |
| Image and prompt | Threaded strip-wise PIL resizes (same pixels), ToTensor/Normalize on the GPU, only the text after `</img>` tokenised | Image + tokenise 99 → 34 ms |
| Pre-transposed weights | `torch.mm(x, Wᵀ)` instead of `F.linear` for the large LLM/ViT linears (bit-identical, ~2× faster for 8960→1536) | LLM 392 → 289 ms (together with the next step) |
| Exact fused kernels | `torch.compile` for RMSNorm, RoPE, SiLU-gate and the ViT layer-scale, with bf16 rounding restored (`_rne`). Each fused op is checked bit-for-bit at start-up | ViT 297 → 288 ms. Total 658 ms |
| Compiled planner step | The DDIM step is `torch.compile`d (fp32, ≤ 1e-6 from eager). The planner attention is written as matmul–softmax–matmul | Planner 45 → 28 ms. Total 636 ms |
| Frame as array | The front frame is handed over as an RGB array instead of being read back from a file | Image 31 → 22 ms. Total **615 ms** |
| *Opt-in:* TensorRT ViT | fp16 engine, or explicit-Q/DQ INT8 on 63 of 98 linears with SmoothQuant rebalancing (α = 0.8). fc2, blocks 0/1/12 and mlp1 stay fp16 | ViT 280 → 253 (fp16) / 230 ms (INT8). Frame ≈ 572–577 ms. **Not exact** |

## Results (Jetson AGX Orin)

Jetson AGX Orin 64 GB, JetPack R36.4.7, clocks pinned (GPU 1300 MHz), torch 2.10, flash-attn 2.8.3,
TensorRT 10.3. Batch 1, one front camera.

| Variant | Frame time | Speedup | VLM hidden states | Plan deviation vs reference, same noise (mean / final / worst) |
|---|---:|---:|---|---|
| Reference `compute_trajectory` | 1249–1260 ms | 1.00× | — | — |
| Fast path (default) | **614 ms** | **2.03×** | bit-identical | ~1e-5 m (max 3.8e-6) |
| + TensorRT fp16 ViT | ~590 ms | 2.1× | 4.0 % rel-L2 | 0.08 / 0.17 / 0.70 m |
| + INT8 Q/DQ ViT (63/98 linears) | 572–577 ms | 2.2× | 5.2 % rel-L2 | 0.13 / 0.24 / 0.80 m |
| *Reference vs itself, other noise* | | | | *0.20 / 0.39 / 0.80 m* |

Stage budget (ms), reference → fast: image + prompt 99 → 22, ViT 298 → 280, LLM 544 → 283, planner 283 → 28,
other 38 → ~2. The fast path costs +3 GB of GPU memory (7.9 GB allocated, peak 11.8 GB) and ~15–60 s of extra
start-up. The planner amplifies small feature changes, which is why the exact path is the default and
the TensorRT encoders are opt-in. The figures in `reference_results/` were measured on three CARLA frames
resized to 1920×1080, before the benches moved to Bench2Drive input.

## Setup

1. **Upstream checkout**: clone xiaomi-research/recogdrive and set `RECOGDRIVE_REPO` to it (default
   `/opt/recogdrive`). Nothing in it is patched.
2. **Weights**: put the ReCogDrive-VLM-2B files and `ReCogDrive_Diffusion_Planner_2B_RL.ckpt` in one folder
   and set `RECOGDRIVE_WEIGHTS` to it. `RECOGDRIVE_VLM` and `RECOGDRIVE_CKPT` override the two paths
   separately. TensorRT artefacts go to `RECOGDRIVE_TRT_DIR` (default `$RECOGDRIVE_WEIGHTS/trt`).
3. **Data**: extract Bench2Drive mini and set `B2D_DIR`. The benches use the held-out clips (every sixth
   clip, starting with the third). INT8 calibration uses the other clips. `rgb_front` (1600×900) is resized
   to 1920×1080. The ego history (4 poses 0.5 s apart), speed, acceleration and `command_near` are read from
   the annos.
4. **Environment**: a Python 3.10 venv on the host, with no container:
   `RECOGDRIVE_VENV=... RECOGDRIVE_REPO=... ./env/run_capped.sh ./env/setup_venv.sh`. It installs only wheels.
   The Jetson torch, torchvision, triton and flash-attn are pinned by URL and hash in
   `requirements-jetson.txt`. The nuplan-devkit chain is installed with `--no-deps`. The script copies
   `cuda.h` and `ptxas` into triton, and the venv sees JetPack's TensorRT through system site-packages.
   `env/stubs/decord` stands in for `decord`, which has no aarch64 wheel.

## Run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
export RECOGDRIVE_VENV=~/venvs/recogdrive RECOGDRIVE_REPO=/opt/recogdrive \
       RECOGDRIVE_WEIGHTS=/data/weights/ReCogDrive B2D_DIR=/data/bench2drive-mini
V=vla_accel/recogdrive

$V/run.sh profile_baseline                     # stage profile of the unmodified agent
$V/run.sh bench --clips 3 --frames 3           # fast path vs reference: exactness, fallback, speed
$V/run.sh bench --noise-floor                  # plus the per-stage deviation breakdown

# opt-in TensorRT / INT8 vision encoder
$V/run.sh trt_vit export
$V/run.sh trt_vit build fp16
$V/run.sh vit_int8 calib --alpha 0.8 --scales $RECOGDRIVE_WEIGHTS/trt/vit_smooth_a0.8.pt --export
$V/run.sh vit_int8 onnx --onnx $RECOGDRIVE_WEIGHTS/trt/vit_smooth_a0.8.onnx \
          --scales $RECOGDRIVE_WEIGHTS/trt/vit_smooth_a0.8.pt --skip ".fc2,L0.,L1.,L12.,mlp1" --tag qdq_a08_sel
$V/run.sh trt_vit build qdq --onnx $RECOGDRIVE_WEIGHTS/trt/vit_9x448_qdq_a08_sel.onnx --tag qdq_a08_sel
$V/run.sh trt_vit bench $RECOGDRIVE_WEIGHTS/trt/vit_fp16.engine $RECOGDRIVE_WEIGHTS/trt/vit_qdq_a08_sel.engine
$V/run.sh vit_e2e $RECOGDRIVE_WEIGHTS/trt/vit_fp16.engine $RECOGDRIVE_WEIGHTS/trt/vit_qdq_a08_sel.engine
$V/run.sh bench --vit-engine $RECOGDRIVE_WEIGHTS/trt/vit_qdq_a08_sel.engine --no-ref-bench
```

Every bench takes `--b2d-dir`, `--clip NAME` (repeatable), `--clips`, `--split held|calib|all`,
`--frames`, `--stride`, `--start` and `--jpeg-quality`. Without `run.sh`, activate the venv, set
`PYTHONPATH=<repo root>:$RECOGDRIVE_REPO:vla_accel/recogdrive/env/stubs` and
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, then run `python3 -m vla_accel.recogdrive.<module>`.

## Files

- `speedups.py`: `FastReCogDrive`, the fast path, plus `verify()` against the unmodified agent.
- `inputs.py`: builds the NAVSIM `AgentInput` from a Bench2Drive frame (front image, ego history, command).
- `common.py`: agent loading, Bench2Drive frame selection and stage timers.
- `bench.py`: fast path vs reference with the same diffusion noise, then speed. `--noise-floor` adds the per-stage deviation breakdown.
- `profile_baseline.py`: stage profile of the unmodified `compute_trajectory`.
- `trt_vit.py`: exports the ViT to ONNX, builds the TensorRT engines (fp16 / int8 / qdq) and checks their latency and accuracy.
- `vit_int8.py`: INT8 calibration, SmoothQuant rebalancing, simulated accuracy and the Q/DQ ONNX.
- `vit_e2e.py`: how much a TensorRT ViT engine moves the planned trajectory.
- `run.sh`: runs a module in the venv under the memory guard and warns if the clocks are not pinned.
- `requirements.txt`, `requirements-jetson.txt`, `requirements-nodeps.txt`: the venv's packages.
- `env/setup_venv.sh`, `env/check_imports.py`, `env/run_capped.sh`, `env/stubs/decord/`: build, check and guard the venv.
- `reference_results/`: Orin logs of the profile, bench, TensorRT engines, plan deviation and INT8 calibration.
