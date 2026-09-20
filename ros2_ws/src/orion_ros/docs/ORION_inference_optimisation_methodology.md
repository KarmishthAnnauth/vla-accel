# Reducing the closed-loop inference latency of ORION on an NVIDIA Jetson AGX Orin

*Methodology draft. All figures were measured on 14 September 2026 on the target device; the tooling that produced them is in `alpamayo-autoware/src/orion_ros/tools/`.*

## 1. Experimental platform and baseline

The vision-language-action model under study is ORION (Fu et al., ICCV 2025), executed as a ROS 2 node inside a CARLA closed-loop evaluation. Per frame, the model consumes six 1600×900 camera images, an 18-dimensional CAN-bus vector and a navigation command, and emits a 6-waypoint trajectory over a 3 s horizon. Its inference path comprises (i) an EVA-ViT-L image backbone (24 blocks, embedding width 1024, 16 heads, 640×640 input, 16×16 patches; 16 window-attention blocks and 8 global-attention blocks), (ii) a PETR-style 3D detection head with 600 object queries and a 600-slot temporal memory, (iii) a map head with 1800 lane queries, (iv) a LLaVA-LLaMA-7B language model (32 layers, hidden size 4096) to which 513 learned vision tokens and a 71-token planning prompt are presented as a single prefill, and (v) a variational trajectory decoder conditioned on the LLM hidden state at a dedicated waypoint token. The LLM carries LoRA adapters (rank 16) on its attention projections.

The target platform is an NVIDIA Jetson AGX Orin (64 GB, JetPack R36.4.7, 61 GB unified memory, GPU clock pinned at 1.3 GHz). The model runs in a container providing PyTorch 2.4 (NVIDIA build 24.07), transformers 4.31, flash-attention 2.7 and TensorRT 8.6.2; the host provides TensorRT 10.3. All inference uses fp16 weights for the backbone and the LLM, with the perception heads in fp32, as in the reference agent.

The baseline node required 2030 ms per frame: 275 ms of CPU-side preparation (JPEG re-encoding to match the reference agent, resizing, normalisation, collation and host-to-device transfer) and 1750 ms for the forward pass.

## 2. Profiling and diagnosis

A standalone benchmark was written that loads the model once, applies candidate optimisations cumulatively, and reports per-stage GPU time via CUDA events wrapped around the backbone, the two heads, the LLM prefill and the planner. Each variant is additionally checked for output fidelity: with the random seed of the trajectory sampler fixed, the maximum absolute deviation of any waypoint coordinate from the unmodified fp16 model is recorded over three CARLA frames.

Kernel-level profiling with `torch.profiler` established that the forward pass was GPU-bound rather than launch-bound: 1713 ms of the 1739 ms wall time was occupied by kernels. Approximately 620 ms was spent in GEMMs and roughly 600 ms in element-wise kernels (rotary embeddings, layer normalisation, window partitioning and precision casts), the latter being memory-bandwidth-bound on this device. The stage decomposition of the baseline was: backbone 883 ms, LLM prefill 636 ms, map head 101 ms, detection head 66 ms, position embedding 13 ms, planner 25 ms.

## 3. Optimisations that preserve the computation

The following changes leave the mathematical function of the model unchanged up to fp16 rounding; the measured trajectory deviation remained below 1 cm throughout, against a seed-to-seed sampling spread of 0.6 cm in the unmodified model. All are implemented as post-construction transforms applied to the instantiated model object, so that the vendored ORION sources remain untouched.

**3.1 LoRA merge.** The reference loader wraps the LLM with PEFT and never merges the adapters, so every projection executes two additional low-rank matrix products per layer. Folding the adapters into the base weights reduced the LLM prefill from 636 to 555 ms.

**3.2 Fused attention for the prefill.** transformers 4.31 implements LLaMA attention by materialising the full score matrix in fp32. Because the planning prefill is a single unpadded sequence, the combined attention mask is purely causal and the attention was routed through `flash_attn_func(causal=True)`, with the original path retained as a fallback for padded or cached inputs. LLM prefill: 555 → 467 ms.

**3.3 Kernel fusion by compilation.** The backbone, the LLM decoder and the transformer stacks inside both heads were compiled with `torch.compile` (Inductor, default mode). The CUDA-graph mode (`reduce-overhead`) was evaluated and rejected: since the workload is not launch-bound it yielded no steady-state gain, it re-recorded its graphs on the first real frame at a cost of several minutes, and the head stacks are invoked several times per forward so that graph-owned outputs were overwritten. Compilation reduced the backbone from 883 to 434 ms, the LLM prefill to 386 ms and the map head from 101 to 73 ms, bringing the forward pass to 1007 ms. Compilation is performed during the node's warm-up forward and its artefacts are cached across container restarts.

**3.4 Backbone block reformulation.** The EVA-ViT attention and SwiGLU blocks were rewritten with identical weights: the three query/key/value projections were concatenated into a single GEMM, the flash-attention wrapper (which stacks and permutes key and value tensors) was replaced by `torch.nn.functional.scaled_dot_product_attention` on strided views, and the two SwiGLU input projections were fused into one GEMM. Backbone: 434 → 420 ms, with the compile time of the backbone halved.

**3.5 Parallel preprocessing.** The six per-camera JPEG re-encodes and the per-image loops inside the reference pipeline's resize and normalisation transforms were mapped over a thread pool; OpenCV and PIL release the interpreter lock, so the six views are processed concurrently. Decoding fell from 81 to 15 ms and the pipeline from 175 to 86 ms with bit-identical tensors. Preparation: 275 → 120 ms.

**3.6 Map-head query pruning.** The map head is configured with 1800 lane queries of which 300 are one-to-one queries and 1500 implement the hybrid-matching (one-to-many) training scheme. The head's self-attention mask forbids attention in both directions between the one-to-many block and the block containing the one-to-one and vision-language queries, cross-attention is per-query, and the temporal memory, decoded lanes and vision tokens are derived from the one-to-one slice only. Restricting inference to the 300 one-to-one queries therefore leaves every softmax over the same set of unmasked keys. Map head: 73 → 41 ms.

**3.7 Pipelining preparation with inference.** Preparation of frame *n*+1 was moved to a second worker thread, started at a time predicted from running averages so that it completes as the forward pass of frame *n* ends; the completion callback of the forward pass then dispatches the next one directly from the prepared batch. The input age is thereby unchanged relative to sequential execution while the preparation cost is hidden. Measured in the node with synthetic 5 Hz camera topics, the period between successive forward passes was 1000–1016 ms for forward passes of 995–1009 ms.

**3.8 GEMM operand layout.** For the LLM down-projection (K = 11 008), cuBLAS selects a 30 % faster fp16 kernel when the weight is stored transposed and contiguous; the change is numerically neutral. LLM prefill: 405 → 357 ms.

Two further candidates were evaluated and not adopted: executing the detection and map heads concurrently on two CUDA streams gave no gain, because the heads are bound by Python launch overhead on a single thread rather than by the GPU; and compiling the 3D position-embedding routine was slower than eager execution.

Table 1 summarises the cumulative effect. After these steps the node sustains one frame per 1.0 s with a forward pass of about 930 ms (backbone 420, LLM 350, heads 110, planner 25).

**Table 1.** Cumulative per-frame forward time (ms) and trajectory deviation from the unmodified model.

| Step | Forward | Backbone | LLM | Map head | Max. Δ trajectory |
|---|---:|---:|---:|---:|---:|
| Baseline | 1736 | 883 | 636 | 101 | – |
| + LoRA merge | 1651 | 882 | 555 | 101 | 1.6 mm |
| + fused prefill attention | 1567 | 885 | 467 | 102 | 5.7 mm |
| + compiled heads | 1529 | 884 | 462 | 73 | 4.7 mm |
| + compiled LLM | 1449 | 878 | 386 | 73 | 7.7 mm |
| + compiled backbone | 1007 | 434 | 387 | 74 | 6.1 mm |
| + backbone reformulation | 1004 | 420 | 399 | 73 | 7.7 mm |
| + map-head pruning | 980 | 422 | 406 | 42 | 4.7 mm |
| + down-projection layout | 928 | 419 | 357 | 42 | 4.7 mm |

## 4. TensorRT evaluation

To test whether a dedicated inference runtime could improve on the compiled PyTorch backbone, the EVA-ViT was exported to ONNX (opset 17, fp32 graph, standard attention in place of the flash-attention operator, absolute position embedding pre-resolved; relative deviation from the fp16 flash path 0.75 %) and built with TensorRT 8.6.2 in the container and with TensorRT 10.3 on the host. TensorRT 8.6 lacked a fused attention kernel for the 1600-token global blocks, which executed at 81 ms each; the resulting engine took 892 ms, or 538 ms after graph simplification and a level-5 tactic search. TensorRT 10.3 did fuse the attention but reached 449 ms against 420 ms for the compiled PyTorch backbone; its layer profile (197 ms GEMM, ~70 ms fused attention, ~180 ms normalisation and point-wise kernels) confirmed that the backbone is memory-bandwidth-bound and not amenable to a different runtime. TensorRT was therefore not adopted. An fp16 engine of the LLM was not attempted because its compiled prefill is already ~97 % GEMM time, which TensorRT executes through the same tensor-core kernels.

## 5. Post-training INT8 quantisation of the language model

**Method.** The LLM decoder projections were replaced by W8A8 linear layers: per-output-channel symmetric int8 weights and per-token dynamic symmetric int8 activations, accumulated in int32 by `torch._int_mm` and rescaled in fp32. A layout detail proved decisive: with the weight supplied as a row-major [K, N] matrix the int8 GEMM was 2.6× slower than fp16 on this platform, whereas the transposed [N, K] view used by cuBLASLt's int8 kernels was 1.6–2.9× faster than fp16 at the prefill shapes.

**Accuracy.** Naïve quantisation of all 224 projections displaced the trajectory by 0.40 m. SmoothQuant (Xiao et al., 2023) was implemented with calibration statistics gathered by forward hooks over three CARLA frames: per input channel, the activation range is migrated into the weights by a factor max|X|^α / max|W|^(1−α), folded exactly into the preceding RMSNorm (for the query/key/value and gate/up projections), into the value-projection rows (for the output projection, attention being linear in V per channel) and into the up-projection rows (for the down projection). A sweep over α, projection groups and fp16-retained layers was run in a single process by restoring fp16 weights between configurations. The first two decoder layers exhibit activation outliers of 100× and 50× the median channel magnitude (the remainder ≈10×), but retaining them in fp16 alone did not resolve the error, which is distributed across all layers.

**Table 2.** INT8 configurations, relative error of the LLM waypoint feature and maximum trajectory deviation (3 frames).

| Configuration | Feature error | Max. Δ trajectory |
|---|---:|---:|
| All projections, no smoothing | – | 0.40 m |
| All, SmoothQuant α = 0.5 | 18 % | 0.15 m |
| All, α = 0.8, layers 0, 1, 30, 31 in fp16 | 7 % | 0.05 m |
| Attention projections only, α = 0.8, same layers fp16 | 5 % | 0.02 m |
| MLP projections only, α = 0.8, same layers fp16 | 4.4 % | 0.03 m |

The MLP-only configuration was selected. Compiled, it reduces the LLM prefill from 346 to 276 ms and the forward pass from 917 to 849 ms (about 0.93 s per frame in the node). Since its deviation is five to ten times that of any fp16 change, it is exposed as an opt-in option whose acceptance is deferred to closed-loop route scoring.

## 6. Reduced-resolution and staggered vision input

As an opt-in option, the backbone input was reduced from 640×640 to 512×512 (1024 instead of 1600 tokens per view) and the three rear cameras were processed only on every other frame, their previous features being reused in between so that the heads always receive six views. The pipeline's resize transform rescales the camera intrinsics and lidar-to-image matrices, so the 3D position embedding follows; the backbone interpolates its absolute position embedding, and its global-attention rotary table was rebuilt for the 32×32 token grid. Static graphs were compiled for both batch shapes. In the node, full frames took ≈800 ms (backbone 240 ms) and staggered frames ≈700 ms (backbone 125 ms), for a period of 680–810 ms, i.e. ≈0.75 s per frame. The trajectory deviation from the base model on the bench frames was 0.47 m, an order of magnitude beyond the INT8 option: the model is operated at a resolution it was not trained at and with rear views one inference period old. Combined with the INT8 option, the per-frame time is estimated at ≈0.65 s (the savings are independent; the combination was not measured end to end).

## 7. Validation

Three instruments were used throughout. (i) The standalone benchmark (§2) for stage timing and trajectory fidelity with the temporal memory reset before each variant. (ii) A synthetic ROS publisher replaying CARLA frames at 5 Hz on all sensor topics, used to measure the deployed node's per-stage times and inter-frame period from its own log. (iii) Parity tools for the ONNX export (feature-level comparison with the fp16 flash path) and for the INT8 sweep (relative error of the waypoint feature and trajectory deviation). Closed-loop route scores on the CARLA Bench2Drive routes remain the acceptance criterion for the two options that alter the model's computation.

## 8. Summary

| Configuration | Per-frame time | Nature of change |
|---|---:|---|
| Baseline node | 2.03 s | – |
| Exact and fp16-neutral optimisations (§3) | 1.00 s | Δ trajectory < 1 cm |
| + INT8 LLM (§5) | ≈0.93 s | Δ trajectory 3–5 cm |
| + reduced/staggered vision (§6) | ≈0.65 s (estimated) | Δ trajectory 0.47 m |

Model loading takes ≈200 s; the compile-based warm-up adds 130–185 s at start-up.
