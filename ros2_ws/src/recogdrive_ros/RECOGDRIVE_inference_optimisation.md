# ReCogDrive inference optimisation on the Jetson AGX Orin

*ReCogDrive 2B (InternVL3-2B backbone + diffusion planner), October 2–3 2026.
Orin 64 GB, JetPack R36.4.7, clocks pinned, PyTorch 2.10 / flash-attn 2.8.3 /
triton 3.6 from the jetson-ai-lab `jp6/cu126` wheels, TensorRT 10.3.*

## Summary

| | Reference | Fast path | Factor |
|---|---:|---:|---:|
| `compute_trajectory`, standalone | 1255 ms | **615 ms** | 2.04× |
| Per frame in the ROS node | 1329 ms | **635 ms** | 2.09× |
| Plan-to-plan period in the node | 1350 ms | 650 ms | |
| VLM hidden states vs reference | | **bit-identical** | |
| Trajectory vs reference, same noise | | ~1e-5 m | |

The fast path is `recogdrive_ros/recogdrive_speedups.py` (`FastReCogDrive`): a
wrapper around the built `ReCogDriveAgent`. No ReCogDrive source is edited, and
the agent stays in place as the fallback for inputs the fast path does not
cover.

Unlike the three earlier models, the result here is not "close to" the
reference. The vision-language half produces the same bits; the fp32 diffusion
planner agrees to rounding. That turned out to matter: this planner amplifies
small feature changes (§6), so an approximate fast path would have needed a
closed-loop re-validation that an exact one does not.

An opt-in TensorRT INT8 vision encoder takes the frame to 577 ms at the cost of
a 0.13 m average plan shift (§7).

---

## 1. What the model does per frame

`ReCogDriveAgent.compute_trajectory` on one 1920×1080 front frame:

1. `load_image`: the frame is resized to 1792×896 and cut into 8 tiles of 448²,
   plus a 448² thumbnail: 9 tiles.
2. InternViT-300M encodes the 9 tiles into 9 × 256 = 2304 image tokens.
3. A prompt is built: a 291-token system prompt, the image tokens, and ~190
   tokens of ego history and navigation command. It is tokenised and
   left-padded to 2800 tokens.
4. Qwen2.5-1.5B (28 layers, bf16) runs **one forward pass**. Nothing is
   generated: the final hidden states of all 2800 positions are the output.
5. A 16-block diffusion transformer (fp32) denoises an 8-pose trajectory in 5
   DDIM steps, cross-attending to those 2800 hidden states.

There is no autoregressive decode and no LoRA (`use_llm_lora = 0` in the
released checkpoint), so two of the usual levers (KV-cached decoding, LoRA
merge) do not exist here in their usual form.

## 2. Where the time went

Stage profile of the unmodified agent, `torch.cuda.synchronize()` on both sides
of every stage, mean of 9 warm runs (`bench/recogdrive/profile_baseline.py`):

| Stage | ms | Share | Diagnosis |
|---|---:|---:|---|
| Image load + tiling | 72 | 5.7 % | CPU: two single-threaded PIL resizes, 9× ToTensor/Normalize |
| Tokenisation | 27 | 2.1 % | Slow tokenizer over a 2800-token string, 93 % of it constant |
| Vision encoder | 298 | 23.6 % | Compute-bound |
| LLM body | 486 | 38.5 % | Compute-bound, running at half the GPU's matmul rate |
| LM head | 58 | 4.6 % | **Waste**: logits over 2800 × 151 682 that nothing reads |
| Diffusion planner | 283 | 22.4 % | **Dispatch-bound**: 267 ms even with a 64-token context |
| Other | 38 | 3.0 % | Mostly `logits.float()`, a 1.7 GB conversion |
| **Total** | **1262** | | |

Microbenchmarks that decided what to do about each (`microbench.py`,
`opbench.py`):

- **Matmul ceiling.** A 2800×1536×8960 bf16 GEMM runs at 35.0 TFLOP/s (fp16
  28.8, fp32 3.3). The LLM as a whole delivered ~16, the ViT ~18.
- **LLM scaling.** 69 ms at 64 tokens, 445 ms at 2800: ~60 ms of dispatch
  overhead, the rest linear in tokens.
- **Planner scaling.** 267 ms with a 64-token context, 274 ms with 2800. A
  384-wide model cannot be compute-bound at that cost; it is ~4000 tiny kernel
  launches per frame.
- **Inside an LLM layer (14.0 ms).** Matmuls 9.3 ms, of which `down_proj`
  (8960→1536) takes 3.9 ms at 18 TFLOP/s, half the rate of the others.
  RMSNorm 1.9 ms, SiLU×gate 1.7 ms, RoPE 0.9 ms, attention 1.0 ms.
- **Inside a ViT layer (12.3 ms).** Matmuls 7.8 ms at ~30 TFLOP/s, attention
  2.1 ms, GELU 1.2 ms, layer-scale + residual 1.4 ms, LayerNorm 0.6 ms.

## 3. What was done

In the order it was built; totals are the standalone frame time after each step.

| Step | Total | Changed stage |
|---|---:|---|
| Reference | 1255 | |
| 1. Skip the LM head; KV-cache the prompt prefix; CUDA graphs; planner context K/V once; threaded resizes; tail-only tokenisation | 769 | LLM 486+58 → 392, planner 283 → 45, image+tokenise 99 → 34 |
| 2. Pre-transposed weights; exact fused kernels; own ViT loop | 658 | LLM 392 → 289, ViT 297 → 288 |
| 3. Compiled planner step; thumbnail resize in strips | 643 | planner 45 → 39 |
| 4. Planner attention written out | 636 | planner 39 → 28 |
| 5. Frame handed over as an array | 615 | image 31 → 22 |

### 3.1 LM head skipped

`Qwen2ForCausalLM.forward` projects all 2800 positions onto the vocabulary and
converts the result to fp32. The agent only reads `hidden_states[-1]`. Calling
the decoder stack directly removes 58 ms of matmul and most of the 38 ms of
"other". Exact.

### 3.2 KV cache for the prompt prefix

The 291 tokens before the image (system prompt, user header, `<img>`) are the
same every frame, and with causal attention their keys, values and final hidden
states cannot depend on what follows. They are computed once at start-up; each
frame runs only the image tokens and the tail, attending to the cached keys and
values (`flash_attn_func` aligns its causal mask bottom-right when there are
more keys than queries, which is exactly this case).

Padding rows are handled the same way. The reference's flash-attention path
gives a padded row zero attention output, so its final hidden state is one
constant vector; it is stored and written into the planner's context rather
than recomputed.

### 3.3 Fixed-size window, CUDA graphs

The tail's token count changes with the numbers in the ego history (measured
187–211 tokens). The LLM runs on a fixed window of 2304 + 232 slots; unused
slots sit at the end, where causal attention keeps them from influencing the
rows that are read. Static shapes make the whole LLM one CUDA graph. The vision
encoder and the planner are a graph each.

The planner's context is laid out exactly as the reference's left-padded
sequence, in a buffer with a capacity mask, so one planner graph serves every
prompt length.

### 3.4 Planner: context keys/values once, attention written out

The 8 cross-attention blocks project the 2800 VLM tokens to keys and values in
every one of the 5 denoising steps; the projections do not depend on the step.
They are computed once per frame.

`F.scaled_dot_product_attention` costs ~0.4 ms per call here whatever the size:
it is a kernel built for long sequences, called 40 times per frame with 8
queries. Written as matmul–softmax–matmul the planner drops from 39 to 28 ms.
The denoising step is `torch.compile`d (fp32, agrees with eager to 1e-6).

The diffusion noise is drawn outside the graph in the reference's order, so a
given torch seed produces the same noise on both paths. That is what makes a
same-seed comparison meaningful.

### 3.5 Pre-transposed weights

`F.linear` on bf16 runs the 8960→1536 projection at 18 TFLOP/s; the same
product as `torch.mm(x, W.t().contiguous())` runs at 33 and is bit-identical.
Applied to the large linears of both transformers (`gate_proj`, `up_proj`,
`down_proj`; ViT `qkv`, `fc1`, `fc2`): total matmul time 444 → 372 ms. Costs
2.7 GB for the transposed copies.

### 3.6 Exact fused kernels

RMSNorm, RoPE, the SiLU gate and the ViT's layer-scale are chains of small
elementwise ops, bandwidth-bound and several kernels each. `torch.compile`
fuses each chain into one kernel (RoPE 0.76 → 0.10 ms, RMSNorm 0.95 → ~0.5 ms),
but **not with the same result**: see §4.2.

### 3.7 Image and prompt

The two PIL resizes are the cost (30 ms and 16 ms). Each is done in horizontal
strips on a thread pool using `Image.resize(box=...)`; the pixels are identical
to the one-shot resize. ToTensor/Normalize run once on the GPU in fp32, which is
bit-identical to torchvision's CPU arithmetic. Only the text after `</img>` is
tokenised (the split is checked against a full tokenisation at start-up). In the
node, the frame is passed as an array instead of being written to a BMP and read
back.

## 4. Why bit-exactness took work

### 4.1 bf16 matmuls round by batch size

The first version computed the prefix cache in a 291-row pass. Its hidden states
differed from the reference by 4.4 % (relative L2), and same-seed trajectories
by 2–12 cm.

The stock model gives bit-identical rows whether the prompt is padded (2800
rows) or not (2787). But it does not for every row count (`rowdep.py`):

| Rows in the batch | MLP output for a given row |
|---|---|
| 291 … 2816 | one result |
| 1, 2, 16 | different |
| 2827 and above | different |

cuBLAS picks a kernel by problem size, and different kernels round the last
bf16 bit differently. bf16 has an 8-bit mantissa; through 28 layers one flipped
bit grows to ~5 % of the final hidden state (the stock model at 2827 rows vs
2800 rows differs by 5.4 %).

The fix: the cached prefix and padding rows are taken from one stock forward of
the reference's own length (2800 rows, left-padded), not computed alone. After
that change the fast path's hidden states are bit-identical.

### 4.2 Fused kernels drop intermediate rounding

An eager bf16 op is "compute in fp32, round to bf16". A fused kernel computes
the whole chain in fp32 and rounds once at the store; `torch.compile` drops
even an explicit `.to(bfloat16).to(float32)` round trip inside the chain
(`castprobe.py`). About a quarter of the elements come out one bit different.

Each fused function therefore puts the rounding back with integer arithmetic on
the fp32 bit pattern:

```python
def _rne(v):                      # fp32 -> nearest-even bf16 value, still fp32
    i = v.view(torch.int32)
    return ((i + 0x7FFF + ((i >> 16) & 1)) & -65536).view(torch.float32)
```

With `_rne` at every point where eager rounds, the compiled chain is
bit-identical. RMSNorm keeps its reduction as the eager kernel, because a fused
sum adds in a different order.

### 4.3 Self-checking at start-up

These properties depend on the GPU, cuBLAS and PyTorch builds, so nothing is
assumed. Every replaced op is compared bit-for-bit against the stock op at
start-up on a tensor of the real shape and dropped if a single bit differs. The
SiLU gate is checked on every finite bf16 value. The node then runs the
reference and the fast path on the warm-up frame with the same noise and logs
the difference:

```
fast path vs reference, same noise: VLM hidden states max |diff| 0 (bit-identical),
trajectory max |diff| 5.96e-06 (m, rad)
```

## 5. Final budget

| Stage | Reference | Fast path |
|---|---:|---:|
| Image + prompt | 99 | 22 |
| Vision encoder | 298 | 280 |
| LLM | 544 | 283 |
| Planner | 283 | 28 |
| Other | 38 | ~2 |
| **Total** | **1262** | **615** |

Both transformers now run their matmuls at 30–35 TFLOP/s, the measured ceiling
for bf16 on this GPU. What is left in them is matmul (~370 ms), attention
(~78 ms) and elementwise work that is already one kernel per chain.

Costs: +3 GB GPU memory (7.8 GB allocated in steady state; the node peaks at
10.6 GB), ~60 s extra start-up on the first launch and ~15 s once the compile
cache is warm. Frames that are not 16:9, or a prompt longer than the window,
fall back to the reference path.

## 6. The planner amplifies small feature changes

Measured on 12 cases (3 frames × 4 ego states), same diffusion noise on both
sides, deviation of the planned xy from the reference plan (`vit_e2e.py`):

| Vision encoder | Feature error vs reference | Mean | Final pose, mean | Worst pose |
|---|---:|---:|---:|---:|
| PyTorch, exact | 0 | 0 | 0 | 0 |
| TensorRT fp16 | 4.0 % | 0.08 m | 0.17 m | 0.70 m |
| *Reference vs itself, other noise* | | *0.20 m* | *0.39 m* | *0.80 m* |

The fp16 engine is mathematically closer to exact fp32 than the bf16 reference
is (the reference's own ViT is 3.8 % from fp32). It still moves the plan by up
to 0.70 m in the worst case. Any optimisation that changes rounding lands in
this regime, which is the argument for keeping the default path exact.

## 7. TensorRT and INT8 for the vision encoder

`bench/recogdrive/trt_vit.py`, `vit_int8.py`, `vit_e2e.py`.

| Variant | ViT | Frame | Feature error | Plan deviation: mean / final / worst |
|---|---:|---:|---:|---|
| PyTorch fast path | 280 ms | 617 ms | 0 | 0 |
| TensorRT fp16 | 253 ms | ~590 ms | 4.0 % | 0.08 / 0.17 / 0.70 m |
| **INT8, 63 of 98 linears, rebalanced** | 230 ms | **577 ms** | 5.2 % | 0.13 / 0.24 / 0.80 m |
| INT8, all 98 linears | 202 ms | ~540 ms | 18 % | 0.33 / 0.77 / 2.0 m |

What was found:

- **TensorRT's calibrator-based INT8 quantises nothing here.** Every
  transformer layer stays fp16 ("Missing scale and zero-point ... fall back to
  non-int8"); the engine is the fp16 engine.
- **Explicit Q/DQ nodes work.** With QuantizeLinear/DequantizeLinear on the
  input and weight of each linear, TensorRT runs the matmuls in INT8 at 2.2× the
  fp16 rate in isolation (`trt_qdq_probe.py`).
- **Most of that is lost to conversions.** Each INT8 matmul sits between fp16
  ops (LayerNorm, attention, GELU), and the quantise/dequantise hops are
  bandwidth-bound. All 98 linears in INT8 save 51 ms over the fp16 engine.
- **Naive W8A8 is too lossy**: 18 % feature error. `fc1` (13 % alone) and `fc2`
  (11 %) and blocks 0, 1, 12 carry it; `qkv` and `proj` are mild.
- **Rebalancing fixes the fixable part.** SmoothQuant-style scaling (α = 0.8)
  folded into the LayerNorm affine for `qkv` and `fc1`, and into the `v` rows of
  `qkv` for `proj`: an exact reparameterisation of the fp32 model. `fc2`'s input
  is a GELU output and cannot be rescaled this way, so `fc2` stays fp16, as do
  blocks 0, 1, 12 and the projector. Result: 5.2 % error against a reference
  whose own rounding is 3.8 %.

Wired as an opt-in (`vit_engine:=<engine>`, `start_recogdrive.sh --int8`). Not
the default, for two reasons: it was calibrated on three frames plus augmented
variants (no driving data was available on the device), and its effect on a
closed-loop score has not been measured.

## 8. Not done

- **LoRA merge**: no LoRA in the checkpoint.
- **TensorRT for the LLM**: its matmuls are already at the hardware ceiling in
  bf16, and a KV-cached prefix would have to be rebuilt inside the runtime.
- **INT8 for the LLM**: not tried. The experience from ORION and MindDrive
  (`MINDDRIVE_ROS_NODE.md`) and §6 here both argue for a closed-loop harness
  first.
- **Fewer image tiles**: changes what the model sees relative to training.

## 9. Environment

No container. A venv on the external SSD on top of the host's ROS 2 Humble
(`env/recogdrive/setup_venv.sh`). Things that had to be worked around:

- A bare `torch==2.10.0` from the jetson-ai-lab index resolves to PyPI's CPU
  wheel (the index mirrors PyPI); the CUDA wheels are pinned by URL and hash.
- The Jetson triton wheel ships without `cuda.h` and `ptxas`, so every
  `torch.compile` fails until they are copied from the system CUDA 12.6.
- exfat over FUSE has no symlinks: the venv is created with `--copies` elsewhere
  and moved.
- torch ≥ 2.6 refuses the planner checkpoint (`weights_only`);
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` restores the behaviour the source was
  written for.
- Heavy jobs run under `run_capped.sh`: a cgroup memory cap plus a watchdog,
  after a flash-attn source build exhausted memory and took the Orin down.

## 10. Control

ReCogDrive's release has no controller (NAVSIM scores the plan in a non-reactive
simulation). The node runs the Bench2Drive PID the way `orion_withpid_node`
does: at 20 Hz, on the latest plan re-cut for its age and the current pose,
with the reference agents' brake rules and 5 m/s throttle cap, publishing
`CarlaEgoVehicleControl` directly. `pid_controller.py` is loaded from the
user's ORION / MindDrive checkout (`pid_controller_dir`), not vendored.
`control_mode:=stanley` keeps the Trajectory → Stanley → AckermannDrive path.

## 11. Reproduction

```bash
cd bench/recogdrive
python3 make_frames.py                       # frames/carla_{0,1,2}.bmp from assets/frames
./run.sh profile_baseline.py                 # §2
./run.sh fast_check.py                       # correctness + speed, the one to run first
./run.sh microbench.py; ./run.sh opbench.py  # ceiling, scaling, per-op costs
./run.sh rowdep.py; ./run.sh castprobe.py    # §4.1, §4.2
./run.sh trt_vit.py export && ./run.sh trt_vit.py build fp16
./run.sh vit_int8.py calib --alpha 0.8 --scales <trt>/vit_smooth_a0.8.pt --export
./run.sh vit_int8.py onnx --onnx <trt>/vit_smooth_a0.8.onnx --scales <trt>/vit_smooth_a0.8.pt \
         --skip ".fc2,L0.,L1.,L12.,mlp1" --tag qdq_a08_sel
./run.sh trt_vit.py build qdq --onnx <trt>/vit_9x448_qdq_a08_sel.onnx --tag qdq_a08_sel
./run.sh vit_e2e.py <trt>/vit_fp16.engine <trt>/vit_qdq_a08_sel.engine   # §6, §7
```

Orin output of these is in `reference_results/recogdrive/`.

## 12. Limitations

- Measured on one Orin. The row-count boundaries in §4.1 belong to this cuBLAS
  build; the start-up checks exist so that a different device degrades to the
  stock ops instead of silently diverging.
- The node has run against synthetic topics only (`tools/fake_carla_topics.py`),
  not against CARLA. Latency and the control path are verified; driving quality
  is not.
- The released weights are NAVSIM-trained (real camera images). Their behaviour
  on CARLA renders is a separate question from anything measured here.
- Three test frames. Exactness holds by construction and is checked at every
  start-up; the INT8 deviation figures are from those three frames.
- Batch size one, one front camera, 16:9.
