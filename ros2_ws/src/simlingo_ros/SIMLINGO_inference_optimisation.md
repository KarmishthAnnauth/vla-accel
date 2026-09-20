# Reducing Closed-Loop Inference Latency of a Vision–Language–Action Model on an Embedded Accelerator

*A case study on SimLingo deployed to the NVIDIA Jetson AGX Orin*

---

## Abstract

The SimLingo vision–language–action (VLA) model, deployed as a ROS 2 node on an
NVIDIA Jetson AGX Orin for closed-loop evaluation in CARLA, exhibited a
per-frame inference latency of approximately 3.5 s, precluding operation at any
control rate useful for driving. This work reports a profiling-driven latency
reduction achieved without model compression, quantisation, or graph export to a
dedicated inference runtime. Phase-level instrumentation localised the cost
almost entirely to the model forward pass, and subsequent microbenchmarking
identified two independent and multiplicative causes: (i) dynamic
voltage–frequency scaling (DVFS) holding the GPU at 31–63 % of its peak clock,
and (ii) an absent key–value (KV) cache in the model's autoregressive decoding
routine, which caused the full prompt to be re-evaluated for every generated
token. Addressing both reduced mean end-to-end latency from 3538 ms to 321 ms, a
speedup of 11.0×, with a maximum deviation of 0.028 m in the predicted route
relative to the unoptimised model. The study additionally illustrates a
methodological point: the intuitive first-line remedies for slow neural network
inference — TensorRT or ONNX Runtime export — would have addressed less than
10 % of the observed latency, and were correctly deprioritised only because
measurement preceded optimisation.

---

## 1. Problem Statement

SimLingo [Renz et al., CVPR 2025] is a vision-only VLA model for autonomous
driving, built on the InternVL2-1B backbone. In the deployment under study, the
model runs on a Jetson AGX Orin and communicates over ROS 2 with a CARLA
simulator hosted on a separate workstation, receiving camera images and odometry
and returning a predicted trajectory.

The node's own timing instrumentation reported inference times of 2.93–3.89 s
per frame. At a vehicle speed of 5 m/s this corresponds to approximately 17 m of
travel between successive policy updates, which is not merely suboptimal but
qualitatively incompatible with closed-loop control: the planner acts on a state
estimate that is already several vehicle lengths stale. Reducing this latency was
therefore a precondition for meaningful closed-loop evaluation, independent of
any question about the policy's driving competence.

The initial hypothesis, and the one motivating this investigation, was that the
model was compute-bound and would require export to TensorRT or ONNX Runtime.
This hypothesis proved incorrect and is examined in Section 8.

---

## 2. Experimental Setup

### 2.1 Hardware

| Component | Specification |
|---|---|
| Platform | NVIDIA Jetson AGX Orin Developer Kit |
| CPU | 12-core ARM Cortex-A78AE |
| Memory | 61 GiB unified (LPDDR5) |
| GPU | Ampere, 1300.5 MHz peak, 306 MHz idle floor |
| Power mode | MAXN (`nvpmodel -m 0`) |
| L4T / JetPack | R36.4.7 (36.4.7-20250918154033) |

### 2.2 Software

All measurements were taken inside the deployment container, whose toolchain
differs from the host interpreter:

| Component | Version |
|---|---|
| PyTorch | 2.2.0a0+6a974be (CUDA 12.2) |
| Transformers | 4.46.3 |
| ROS 2 | Humble, `rmw_cyclonedds_cpp` |
| Triton | **not installed** (see §5.4) |

### 2.3 Model

SimLingo checkpoint `epoch=013`, configured per its accompanying Hydra config:

| Component | Parameters |
|---|---|
| Vision encoder (InternViT-300M) | 308.5 M |
| Language model (Qwen2-0.5B + LoRA) | 647.3 M |
| **Total** | **957.2 M** |

LoRA adapters (rank 32, α = 64, `target_modules="all-linear"`) are applied to
the language model and were *not* merged into the base weights at load time.
Inference runs in bfloat16 under `torch.autocast`.

The deployed prompt is the non-chain-of-thought variant
(`"… Predict the waypoints."`), for which the model emits the four-token prefix
`"Waypoints:"` before the driving head is evaluated. This was verified across
112 consecutive inferences, all of which produced identical text. This detail
matters: the cost of the decoding defect described in §4.2 scales with the number
of generated tokens, and the reported speedups are therefore *conservative*
relative to a chain-of-thought configuration.

---

## 3. Methodology

The investigation deliberately followed a measure-then-optimise discipline, in
three stages.

**Stage 1 — Establish the model's intrinsic cost.** The model was loaded and
exercised in isolation, outside the ROS process, on a synthetic input matching
the deployed tensor shapes (1024 × 359 frame → 2 tiles → 543-token prompt). This
established a reference latency independent of middleware.

**Stage 2 — Attribute the deployed latency.** Because the isolated measurement
(≈ 790 ms) disagreed with the deployed measurement (≈ 3500 ms) by a factor of
4.4, the discrepancy itself became the object of study. Phase-level
instrumentation was added to the ROS node, partitioning each inference into
mutually exclusive intervals whose sum is checked against the measured wall time,
with any unattributed remainder reported explicitly as `other`. A
`torch.cuda.synchronize()` barrier was placed on both sides of the model call so
that GPU time is attributed to the model rather than absorbed by the first
subsequent device-to-host transfer. Per-frame GPU clock frequency was sampled
from `sysfs` alongside each measurement.

**Stage 3 — Microbenchmark the dominant phase.** Once the model forward was
confirmed as the dominant term, its internals were benchmarked in isolation to
separate arithmetic cost from dispatch overhead.

An intermediate hypothesis attributing the discrepancy to ROS middleware
contention was formulated and *falsified* in Stage 2; this is reported in §4.3
because the falsification is itself informative.

---

## 4. Diagnosis

### 4.1 Phase attribution

Instrumented measurement over 56 inferences (first five discarded as warm-up)
localised the cost unambiguously:

| Phase | Mean (ms) | Share |
|---|---|---|
| **Model forward** | **3450.2** | **97.5 %** |
| Image tiling (`dynamic_preprocess`) | 26.9 | 0.8 % |
| Diagnostic logging | 15.1 | 0.4 % |
| Prompt construction + tokenisation | ~10 | 0.3 % |
| Trajectory construction and publication | ~6 | 0.2 % |
| Payload queueing delay | 1.4 | < 0.1 % |
| Unattributed (`other`) | 0.0 | 0 % |

The `other` term is zero on every sample, confirming the partition is complete
and that no cost is hidden outside the instrumented intervals.

### 4.2 Cause I — absent KV cache in autoregressive decoding

Inspection of `simlingo_training/models/language_model/llm.py:178`
(`LLM.greedy_sample`) revealed that the decoding loop maintains no KV cache. At
each step the newly sampled token embedding is concatenated to the running
sequence (`llm.py:234`) and the *entire* sequence is passed through all 24
transformer layers (`llm.py:218`). The driving head then performs a third full
forward pass over the same prefix (`driving.py:156`).

For a prompt of length *L* and *N* generated tokens, the cost is therefore
(*N* + 2) full forward passes rather than one prefill plus *N* incremental
decode steps. With *L* = 543 and *N* = 4 this is six full forwards per frame.

Microbenchmarking (Table 1) confirmed that a full forward pass is essentially
flat in sequence length over the relevant range, indicating that the cost is not
attention over the context but per-layer overhead.

### 4.3 Cause II — DVFS throttling, and the falsified middleware hypothesis

The instrumented GPU clock readings distinguished the deployed configuration
from the isolated one decisively:

| Configuration | GPU clock distribution |
|---|---|
| Isolated benchmark | 1300 MHz (100 %) |
| Deployed ROS node | 816 MHz (74 %), 408 MHz (16 %), 1020–1122 MHz (8 %), 612 MHz (2 %) |

The CPU governor (`schedutil`) concurrently held the cores at 1.4–1.5 GHz
against a 2.2 GHz maximum.

A prior hypothesis attributing the gap to ROS middleware contention — Global
Interpreter Lock (GIL) competition between the inference worker thread and the
`MultiThreadedExecutor` — was tested by reconstructing the node's concurrency
structure (10 Hz compressed-image subscription with JPEG decoding, 20 Hz control
timer, single-worker thread pool, two-thread executor) around the isolated model.
This reproduction yielded 1068 ms against 794 ms for the model alone: a 274 ms
penalty, insufficient by an order of magnitude to explain the observed gap. The
hypothesis was rejected, and the instrumented `wait = 1.4 ms` measurement in §4.1
subsequently confirmed that no queueing delay was present.

### 4.4 The interaction between the two causes

The two causes are not independent in mechanism, though they are separable in
remedy. Two measurements bound the decode's dispatch overhead. A single-token
forward pass with no cache whatsoever costs 52.1 ms against roughly 1 GFLOP of
arithmetic; and capturing the cached decode step into a CUDA graph removes 78 %
of its latency (47.3 → 10.6 ms) while performing identical arithmetic. The
decode is therefore overwhelmingly **dispatch-bound rather than compute-bound**.
A workload dominated by
launch overhead leaves the GPU idle between small kernels, and low measured
utilisation is precisely the input on which the `nvhost_podgov` governor bases a
decision to *reduce* clock frequency. Reduced clocks lengthen each kernel and the
dispatch path feeding it, further depressing measured utilisation.

This reciprocal relationship is offered as the most plausible explanation
consistent with the measurements, but it should be noted that it was not
isolated experimentally: the causal direction between low utilisation and low
clocks was inferred from the governor's documented behaviour rather than
demonstrated by controlled intervention. The *remedies* for the two causes were
validated independently and are reported separately in §6.

---

## 5. Interventions

### 5.1 Clock pinning

`jetson_clocks` was applied, which fixes CPU, GPU and memory controller
frequencies at their maxima by raising each `scaling_min_freq` to the
corresponding maximum. Note that this leaves the governor *name* unchanged
(`schedutil`), so verification must inspect frequency rather than governor
identity — an ambiguity that initially suggested the intervention had failed.

The setting does not survive a reboot and was therefore added as an idempotent
preflight check in the deployment script, which warns rather than aborting when
privilege escalation is unavailable.

### 5.2 LoRA merging

Adapter weights were folded into the base weights via PEFT's
`merge_and_unload()`, reducing each adapted linear layer from three matrix
multiplications to one. Since `target_modules="all-linear"` excludes the output
embedding by construction, the sampling logits — computed in `driving.py:149`
from `lm_head.weight` directly — are provably unaffected by the merge.

### 5.3 Attention backend substitution

The attention implementation was changed from FlashAttention-2 to SDPA. This was
not a performance measure but an enabling one: `_flash_attention_forward` calls
`unpad_input`, which performs a `torch.nonzero` on the attention mask, producing
a data-dependent output shape. CUDA graph capture of this operation fails with a
device-side assertion. Because Qwen2 binds its attention class at construction
time, setting `config._attn_implementation` alone is insufficient and the class
must be rebound on each layer.

### 5.4 KV caching and CUDA graph capture

`greedy_sample` was replaced by an implementation using a pre-allocated
`StaticCache`: the prompt is prefilled once, and each decode step processes a
single token. The decode step is captured into a `torch.cuda.CUDAGraph`, which
collapses several hundred Python-level kernel dispatches into a single replay.

All state the graph reads — token embedding, write position, attention mask —
resides in statically allocated tensors updated in place, so a single capture
serves every subsequent frame and accommodates varying prompt lengths without
recapture.

`torch.compile(mode="reduce-overhead")`, which would provide CUDA graphs
automatically, is unavailable: the container ships no Triton and the Inductor
backend fails outright. Raw graph capture requires neither.

### 5.5 Driving head as cache continuation

The driving head's 30 query tokens are appended to the existing cache rather than
triggering a third full forward pass. Care was required to reproduce the original
prefix exactly: `greedy_sample` appends the embedding of the end-of-sequence
token to the sequence *before* terminating, so the driving queries in the
original implementation attend to a prefix that includes it. The replacement
writes every sampled token — the terminator included — into the cache before
breaking.

---

## 6. Results

### 6.1 Microbenchmarks

**Table 1** — Language model, 543-token prompt, clocks pinned at 1300 MHz.

| Operation | Latency (ms) |
|---|---|
| Full forward, LoRA unmerged, FlashAttention-2 | 133.8 |
| Full forward, LoRA merged | 78.7 |
| Full forward, LoRA merged + SDPA (prefill) | 63.3 |
| Single-token forward, **no cache at all** (overhead floor) | 52.1 |
| Decode step, `DynamicCache` + LoRA merged, FlashAttention-2 | 56.2 |
| Decode step, `StaticCache` + SDPA, eager | 47.3 |
| **Decode step, CUDA graph replay** | **10.6** |

Full forward cost was measured at 139.3, 135.8, 135.1, 135.6 and 134.4 ms for
sequence lengths of 543, 560, 580, 600 and 650 tokens respectively (LoRA
unmerged, FlashAttention-2) — approximately invariant, and in fact weakly
*decreasing*, corroborating the overhead-bound diagnosis. Values in Table 1 come
from a separate run and differ by a few milliseconds from this series.

**Table 2** — Vision encoder (InternViT-300M, 2 tiles of 448 × 448).

| Operation | Latency (ms) |
|---|---|
| Forward, eager | 83.3 |
| Forward, CUDA graph replay | 81.3 |

The negligible improvement from graph capture indicates the vision encoder is
genuinely compute-bound, in contrast to the language model. It is consequently
the only component for which a dedicated inference runtime would be expected to
help (§8).

### 6.2 End-to-end latency

**Table 3** — Deployed ROS node, per-frame latency. First five inferences
discarded as warm-up; `wall` is the complete per-frame cost including payload
preparation.

| Configuration | *n* | Mean (ms) | SD | Min | Median | Max |
|---|---|---|---|---|---|---|
| **A** Baseline | 56 | 3538.0 | 574.5 | 1790 | 3774.5 | 3934 |
| **B** A + clock pinning | 56 | 937.5 | 17.9 | 918 | 930.0 | 993 |
| **C** B + cached/graphed inference | 364 | **320.9** | **11.0** | 296 | 323.0 | 360 |

Model forward component in isolation:

| Configuration | Mean (ms) | SD | GPU clock |
|---|---|---|---|
| A | 3450.2 | 569.0 | 408–1122 MHz (816 modal) |
| B | 909.4 | 17.7 | 1300 MHz (100 %) |
| C | 293.5 | 10.6 | 1300 MHz (100 %) |

**Cumulative speedup: 11.0×** (A → C), decomposing into 3.77× from clock pinning
and 2.92× from the inference-path changes.

Two secondary observations merit note. First, the standard deviation falls by a
factor of 52 between A and C; the baseline's high variance (SD = 574 ms, range
1790–3934 ms) is itself a signature of DVFS behaviour, since the governor
settles at different operating points across frames. Second, the maximum in
configuration C (360 ms) excludes a single 1048 ms first call corresponding to
CUDA graph capture, a one-off cost incurred at startup.

### 6.3 Achieved policy rate versus latency

Latency and policy rate are not equivalent here, because inference is triggered
by a periodic timer (`inference_period_sec` = 0.25 s) that skips a tick whenever
the previous inference is still running. The achieved period is therefore
quantised to a multiple of 250 ms:

| Configuration | Latency (ms) | Submit-to-submit interval (ms) | Rate (Hz) |
|---|---|---|---|
| A | 3538 | 3995 (median) | 0.27 |
| B | 938 | 1000 | 1.00 |
| C | 321 | 500 | 1.99 |

Each observed interval is the smallest multiple of 250 ms exceeding the
corresponding latency, confirming the quantisation. Consequently the **11.0×
latency reduction yields a 7.4× improvement in achieved policy rate**, and in
configuration C the node is idle for approximately 179 ms of every 500 ms cycle.
Reducing the timer period would recover the difference — an inference latency of
321 ms admits approximately 3.1 Hz — but this was not altered here, as the
control loop's tuning is coupled to the policy rate and revalidation would be
required.

### 6.4 Residual latency budget

Mean per-frame cost in configuration C:

| Phase | Mean (ms) |
|---|---|
| Model forward | 293.5 |
| Image tiling | 9.8 |
| Diagnostic logging | 3.8 |
| Prompt construction | 3.0 |
| Input assembly | 2.0 |
| Publication | 2.0 |
| Queueing delay | 1.0 |
| **Total** | **320.9** |

---

## 7. Validation

Because the interventions alter numerical results, equivalence was verified
rather than assumed. Against the unmodified model on identical input:

| Property | Result |
|---|---|
| Route waypoints, max abs. deviation | **0.028 m** |
| Speed waypoints, max abs. deviation | **0.054 m** |
| Generated text | identical |

The deviation originates from the bfloat16 LoRA merge — folding `BA` into the
base weights incurs rounding at bfloat16 precision — and not from the caching or
graph capture, which are arithmetically exact reorderings. The magnitude is well
below the spatial resolution at which the downstream PID controller tracks
waypoints.

Two failure modes specific to persistent state were tested explicitly, since the
`StaticCache` and captured graph are allocated once and reused across all frames:

- **Repeat invocation.** Five consecutive calls on identical input produced
  bit-identical outputs (`route[0,0] = +0.00592` in all cases), excluding
  cache contamination between frames.
- **Varying prompt length.** Prompts of 543 and 544 tokens (arising from
  differing decimal representations of vehicle speed) were interleaved; returning
  to a previously seen length reproduced the earlier output exactly
  (`+0.00368`), excluding stale-state corruption across length changes.

Across 364 logged inferences in configuration C, no capture failure or fallback
to the eager path occurred.

---

## 8. Discussion

### 8.1 On the deprioritisation of TensorRT and ONNX Runtime

The investigation was initiated by the question of whether TensorRT or ONNX
Runtime export would reduce inference latency. The measurements answer this
directly and negatively for the dominant cost.

Both runtimes address arithmetic efficiency: operator fusion, kernel selection,
precision reduction. Neither addresses DVFS behaviour, and neither would have
been necessary to remove the redundant forward passes, which are a property of
the model's Python decoding loop rather than of its computational graph. Of the
observed 3538 ms, approximately 2600 ms was attributable to clock throttling and
a further 600 ms to redundant prefills — together roughly 90 %, none of which is
reachable by graph export.

Furthermore, the autoregressive decode was measured to be dispatch-bound:
CUDA graph capture removed 78 % of the decode step's latency without altering
its arithmetic. Overhead of this kind is addressed by CUDA graphs, which both runtimes can also
provide, but which are obtainable directly from PyTorch at far lower integration
cost — particularly given that a KV-cached decoder would have had to be
reconstructed inside the target runtime regardless.

The vision encoder is the exception. At 83.3 ms with no measurable benefit from
graph capture, it is genuinely compute-bound, has static input shape, and
contains no data-dependent control flow. It is therefore a well-posed TensorRT
target — but it now represents roughly 26 % of a 321 ms budget, and an optimistic
1.5–2× improvement on it would yield perhaps 40 ms, or 12 % of total latency.

Had this been the first intervention attempted, it would have consumed
substantially more engineering effort than the two interventions that actually
mattered, while addressing a small fraction of the latency.

### 8.2 Generalisable observations

Three findings are likely to transfer to comparable deployments.

1. **Small language models on embedded accelerators are dispatch-bound, not
   compute-bound.** A 0.5 B-parameter decoder performing ~1 GFLOP per token
   required 52 ms per single-token forward pass, and 78 % of its cached decode
   latency was removed by CUDA graph capture alone — that is, by issuing the
   same kernels differently. Optimisation effort directed at arithmetic would
   have been misallocated.

2. **DVFS governors interact pathologically with launch-bound workloads.** A
   workload that keeps the accelerator idle between small kernels presents as
   low utilisation, which the governor answers by reducing clocks. On a
   development platform this is silent: no log records it, and the symptom is
   indistinguishable from a slow model.

3. **Instrumentation should partition time exhaustively.** The `other` residual
   was decisive: had the instrumentation reported only the phases anticipated,
   the middleware hypothesis of §4.3 could have survived much longer. A
   measured `other = 0` is what licensed the conclusion that the model forward
   was the whole story.

---

## 9. Limitations

- **Single-platform.** All results are specific to one Jetson AGX Orin under
  JetPack 36.4.7. DVFS behaviour is platform- and governor-specific.
- **The DVFS mechanism is inferred.** The reciprocal relationship proposed in
  §4.4 is consistent with all measurements but was not isolated by controlled
  intervention. The remedy's effect, by contrast, is directly measured.
- **Short generation only.** Speedups were measured with a four-token prefix.
  Microbenchmarks imply the advantage grows with generation length (projected
  ≈ 7× for 40 tokens), but this was not measured end-to-end in deployment.
- **Batch size one.** The fast path asserts a batch size of one, which is
  sufficient for single-vehicle closed-loop deployment but not for batched
  offline evaluation. The original implementation is retained and reachable.
- **Numerical deviation is not zero.** The 0.028 m route deviation is small
  relative to controller resolution, but its effect on closed-loop driving
  metrics over a full benchmark has not been quantified. The optimisation is
  therefore exposed as a runtime parameter that can be disabled.
- **Clock pinning has thermal consequences.** Sustained maximum clocks remove
  thermal headroom; long-duration behaviour under enclosure was not
  characterised.
- **Input staleness is unguarded.** The node consumes the most recent frame
  available when a payload is constructed (measured median age 70 ms against a
  10 Hz stream), but no check rejects a stale one. In three intervals of the
  logged run the image stream ceased and inference continued against a frozen
  frame whose age grew linearly at one inference period per cycle. This is a
  pre-existing property of the deployment, unaffected by the changes reported
  here, but it bounds the validity of any latency figure during stream
  interruptions.

---

## 10. Reproduction

| Artefact | Location |
|---|---|
| Fast inference path | `simlingo/simlingo_training/models/fast_inference.py` |
| Node instrumentation and integration | `alpamayo-autoware/src/simlingo_ros/simlingo_ros/simlingo_node.py` |
| Clock preflight check | `simlingo/start_orin.sh` |
| Correctness and speed harness | `benchmarking/simlingo_fast_check.py` |
| Isolated model profiler | `benchmarking/simlingo_profile_baseline.py` |
| ROS structure reproduction (§4.3) | `benchmarking/simlingo_ros_probe.py` |

Both optimisations are exposed as runtime parameters defaulting to enabled:

```bash
./start_orin.sh                          # both active
./start_orin.sh fast_inference:=false    # stock model, clocks still pinned
./start_orin.sh profile:=false           # suppress per-frame instrumentation
```

Per-frame instrumentation is emitted in the form:

```
[PROF] wall=324ms total=307ms [pay_copy=0 pay_tile=11 pay_norm=3 pay_route=1 wait=1]
       prompt=3 assemble=2 model=297 control=0 log=3 publish=2 other=0 gpu=1300MHz
```

---

## 11. Summary

| | Baseline | Final | Factor |
|---|---|---|---|
| Per-frame latency (mean) | 3538 ms | **321 ms** | **11.0×** |
| Model forward (mean) | 3450 ms | 294 ms | 11.7× |
| Standard deviation | 574 ms | 11 ms | 52× |
| Achieved policy rate (measured) | 0.27 Hz | 1.99 Hz | 7.4× |

The result was obtained without modifying the model architecture, reducing
numerical precision, compressing weights, or exporting to a dedicated inference
runtime. It required pinning two clock domains and correcting a decoding loop
that discarded its own intermediate state.
