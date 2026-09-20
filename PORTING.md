# Porting to Jetson AGX Thor

Everything measured in this repo was measured on a Jetson AGX Orin 64 GB
(JetPack R36.4.7, CUDA 12, PyTorch 2.4/NVIDIA 24.07, transformers 4.31 for
ORION and 4.45.2 for MindDrive, flash-attention 2.7, TensorRT 8.6.2 in-container
/ 10.3 on the host). Thor is a different GPU generation on a different JetPack,
so **treat every number in the docs as the Orin baseline to beat, not as an
expected result.**

This file is what to rebuild, what to re-validate, and — more usefully — which
conclusions are likely to flip.

---

## 1. Nothing binary transfers

Rebuild, in this order. Each item blocks the ones after it.

1. **The base container images.** `env/Dockerfile.simlingo_ros` and
   `env/minddrive/Dockerfile` are built on JetPack 6 aarch64 bases
   (`karmishthannauth/simlingo_env:v01`, `orion_env_ros:v01`). Both need a
   JetPack 7 / CUDA 13 base and a matching PyTorch aarch64 wheel.
2. **The compiled mmcv ops.** ORION and MindDrive both need `mmcv._ext` built
   for the new compute capability. This is the single largest porting risk: the
   Orin setup works *only* because the ORION image ships a pre-built
   `/root/Orion`, and MindDrive reuses those same three `.so` files
   (`mmcv/ops/csrc` is identical across the two forks — see
   `scripts/start_minddrive.sh`). Build ORION's first, then copy.
3. **flash-attention.** ORION's LLM prefill path and MindDrive's
   `flash_attention_2` option both need a rebuilt wheel. If it will not build,
   both fall back cleanly — `patch_llm_flash_attention()` returns False and
   `set_llm_attention()` takes `sdpa` — so this is not a blocker, but the
   fallback costs ORION ~90 ms/frame on the Orin.
4. **The inductor cache.** `.torchinductor_cache` is architecture-specific.
   Delete it or every `torch.compile` variant will silently miscompare. Budget
   the first-run compile again: ~3.5 min for ORION.
5. **TensorRT plans.** Any `.plan` is locked to both the architecture and the
   TensorRT version. `orion_ros/engines/` is gitignored for this reason; rebuild
   with `tools/export_vit_onnx.py` then `tools/check_vit_engine.py`.

Dependency pins worth checking early: ORION needs **transformers 4.31**, which
is old enough that a CUDA 13 PyTorch may not tolerate it. MindDrive needs
**4.45.2** and cannot share ORION's image for exactly this reason. If 4.31 has
to move, `patch_llm_flash_attention()` is the code that assumes it — it exists
precisely because 4.31 has no SDPA path, so a newer transformers would make it
redundant rather than broken.

---

## 2. Which conclusions will probably flip

The optimisations are mostly safe. The *rankings and the negative results* are
what were specific to the Orin, and those are where re-measuring pays.

**SimLingo's clock pinning (3.77× of its 11×).** DVFS behaviour is a platform
property. Pin the clocks and re-run `bench/simlingo/_clock_test.py` before
anything else — if Thor's governor behaves differently under a launch-bound
duty cycle, SimLingo's headline number changes shape entirely.

**SimLingo's CUDA-graph capture (135 → 10.6 ms per decode step).** That win was
~93 % kernel-launch overhead being collapsed, which is a *CPU-side* cost. Thor's
CPU is a newer core at a higher clock, so the launch overhead — and therefore
the size of this win — should shrink. Re-run `_cpu_contention.py`. The graph
capture is still worth keeping; just do not expect 12.7×.

**ORION's `torch.compile` win (ViT 883 → 434 ms).** This was fusion of
memory-bandwidth-bound elementwise kernels (~600 ms of the 1713 ms). How much
survives depends on Thor's bandwidth-to-FLOP ratio. If Thor is
relatively more bandwidth-starved, the fusion win grows; if less, it shrinks and
the GEMMs dominate instead.

**"TensorRT for the ViT is a wash" (ORION).** Re-test. That verdict was against
TensorRT 8.6.2 on Orin. A newer TensorRT on a new architecture is exactly the
kind of thing that changes it.

**"INT8 costs MindDrive a metre of trajectory."** Re-test, and reconsider the
format. The int8 path uses `torch._int_mm` (cuBLASLt int8 tensor cores) and was
1.4–2× faster than fp16 at these shapes on Orin, but plain per-token
quantisation of Qwen2.5-3B moved the path 1.19 m, SmoothQuant α=0.5 only got it
to 0.41 m, and ORION's own α=0.8/MLP-only recipe made it worse (1.45 m). **If
Thor exposes FP8 tensor cores, that is a better answer than INT8 for this model
specifically** — the failure was activation outliers, which FP8's dynamic range
handles and INT8's does not. The whole quantisation section is worth reopening
rather than porting.

**"INT8 for the ViT is slower than the best fp16 layout."** Same caveat; this
was a kernel-selection result, not a property of the model.

**The GEMM layout tricks** (`down_proj_t`, `vit_weight_t` — storing weights
`[K,N]`-contiguous, worth ~50 ms and ~35 ms) are pure cuBLAS kernel-selection
artefacts. They are exact, so they cannot hurt, but they may become no-ops.

**What should transfer unchanged**, because it is arithmetic rather than
hardware: the LoRA merges, the map-head slice to 300 one-to-one queries, the
one-row logits slice, the fused-QKV/SwiGLU ViT rewrite, the unpadded
window-attention blocks, parallel image decode, and prep/inference pipelining.
All are exact or fp16-rounding-exact and the bench proves it per step.

---

## 3. Re-validation order

Each bench prints per-stage timing *and* max Δ trajectory against the
unoptimised model, so a variant that gets faster while drifting is visible
immediately. Work down the cumulative chain and stop where a step stops paying.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks        # first, always

# 1. Does the model still produce the reference output at all?
docker exec minddrive_ros ... minddrive_env/smoke_infer.py

# 2. Baseline, unoptimised, to establish Thor's starting point
... bench_orion.py --frames 4 --variants baseline
... bench_minddrive.py --precision fp32 --frames 3 --variants baseline \
      --ref-out /tmp/ref_fp32.npz

# 3. The full cumulative chain
... bench_orion.py --frames 4
... bench_minddrive.py --precision fp16 --frames 3 --ref-in /tmp/ref_fp32.npz

# 4. SimLingo: correctness first, then the platform questions
python3 bench/simlingo/simlingo_fast_check.py
python3 bench/simlingo/_clock_test.py
python3 bench/simlingo/_cpu_contention.py

# 5. ORION unit tests (no GPU needed for most)
pytest ros2_ws/src/orion_ros/test/

# 6. End-to-end in the node, no simulator
ros2 launch orion_ros orion_withpid.launch.py &
python3 ros2_ws/src/orion_ros/tools/fake_carla_topics.py 45
```

Diff step 3's output against `reference_results/`.

---

## 4. Things that will bite

- **`fp16_infer` is a load-time property**, not a flag you can toggle on a built
  model. fp32 and fp16 are separate bench invocations; that is what
  `--ref-out`/`--ref-in` exist for. Comparing an fp16 chain against fp16 eager
  instead of against the fp32 reference hides real drift.
- **MindDrive's fp16 path needs `env/minddrive/0002-fp16-qwen-load.patch`.**
  Upstream's `load_model` instantiates the LLaMA class under `fp16_infer`
  whatever `lm_model_type` says, so it cannot load either Qwen2 checkpoint —
  fp16 was a dead path for both MindDrive variants before this patch.
- **`env/minddrive/0001-lazy-rl-runner-imports.patch`** makes `import mmcv`
  survive without CARLA. The RL rollout runner imports the CARLA client at
  module level and there is no aarch64 CARLA wheel. Without it nothing in this
  repo runs. Same reason `simlingo_patch/team_code/nav_planner.py` guards its
  `GlobalRoutePlanner` import.
- **`merge_lora` costs memory before it saves time.** MindDrive keeps one merged
  copy per expert: +6 GB GPU. Fine in 64 GB unified memory; check the headroom.
- **Model load is minutes, not seconds** — ~200 s for ORION, ~385 s with the
  compile. The benches load once and apply variants cumulatively precisely
  because of this. Do not restructure them into one-variant-per-process.
- **`reduce-overhead` (CUDA graphs) was rejected for ORION** and the reason is
  structural, not performance: the head transformer stacks are called several
  times per forward, so graph-owned output tensors get overwritten. That will
  still be true on Thor. MindDrive's `graph_vit()` captures only the
  static-shape ViT, which is safe.
