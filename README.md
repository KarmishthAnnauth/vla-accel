# VLA inference acceleration — SimLingo, ORION, MindDrive

Everything needed to reproduce the inference-latency work on three
vision-language-action driving models, packaged so it can be re-run on a
**Jetson AGX Thor**. All figures in here were measured on a **Jetson AGX Orin
64 GB** (JetPack R36.4.7, clocks pinned) between September 12–18 2026; Thor is
the port target, not yet measured.

**Nothing in this repo requires CARLA.** Every model's speedups are measured by
a standalone bench that loads the checkpoint once and applies the optimisations
cumulatively against synthetic or recorded frames, and the ROS 2 nodes can be
driven end-to-end by the fake-topic publishers in `tools/`. The simulator is
only needed for closed-loop driving scores, which are out of scope here.

---

## What was achieved on the Orin

| Model | Baseline | Optimised | Speedup | Trajectory deviation |
|---|---:|---:|---:|---|
| **SimLingo** (InternVL2-1B + driving head) | 3538 ms/frame | **321 ms/frame** | **11.0×** | 0.028 m max route |
| **ORION** (EVA-ViT-L + LLaVA-LLaMA-7B) | 2030 ms/frame | **~1000 ms/frame** | **~2.0×** | < 1 cm |
| **MindDrive** (EVA-ViT + LLaVA-Qwen2.5-3B, 2 experts) | 2318 ms/frame (fp32) | **874 ms/frame** | **2.65×** | 0.05–0.09 m |

The three came apart in completely different places, which is the main reason
all three are worth re-running on Thor rather than assuming the same recipe:

- **SimLingo was not compute-bound at all.** Two multiplicative causes: the GPU
  governor sat at 31–63 % of peak clock, and the model's `greedy_sample` kept
  no KV cache, re-prefilling the whole 543-token prompt for every generated
  token. Clock pinning gave 3.77×, the inference-path changes 2.92×. TensorRT
  or ONNX export would have addressed under 10 % of the latency.
- **ORION was genuinely GPU-bound** — 1713 ms of kernel time in a 1739 ms
  forward, roughly half of it in memory-bandwidth-bound elementwise kernels.
  The win came from `torch.compile` fusion (ViT 883 → 434 ms), not from
  removing launch overhead. TensorRT for the ViT was tried and was a wash.
- **MindDrive** is ORION's ViT and heads with a Qwen2.5-3B and two LoRA experts
  alternating per frame, so ORION's helpers transferred verbatim while the LLM
  side had to be rewritten. INT8 is available but stays off: the Qwen2.5-3B's
  activation outliers move the trajectory by a metre even with SmoothQuant,
  where ORION's LLaMA-7B took the same recipe at 0.03–0.05 m.

Full methodology, per-stage tables and the negative results are in the docs
listed under each model below.

---

## Layout

```
ros2_ws/src/
  orion_ros/          ORION + Orion-Lite: node, speedups, bench, tests, docs
  minddrive_ros/      MindDrive: node, speedups, bench, docs
  simlingo_ros/       SimLingo: node, PID/Stanley controllers, docs
simlingo_patch/       files that must be dropped into a SimLingo checkout
bench/simlingo/       SimLingo's standalone profiling + diagnosis scripts
assets/frames/        three real camera frames used by every parity check
scripts/              container launchers (Orin paths; edit for Thor)
env/                  Dockerfiles, image build scripts, upstream patches
reference_results/    Orin bench output, to diff Thor against
```

### The acceleration frameworks

Each model's optimisations are a single module of **post-build transforms applied
to the instantiated model object**. No upstream model source is edited, so the
vendored checkouts (and, for ORION, the container's pre-built `/root/Orion` with
its compiled mmcv ops) stay exactly as they are.

| Model | Module | Lines |
|---|---|---:|
| ORION | `ros2_ws/src/orion_ros/orion_ros/orion_speedups.py` | 833 |
| MindDrive | `ros2_ws/src/minddrive_ros/minddrive_ros/minddrive_speedups.py` | 1260 |
| SimLingo | `simlingo_patch/simlingo_training/models/fast_inference.py` | 350 |

SimLingo is the exception to the "no source edits" rule: its fast path replaces
a method on `DrivingModel`, so it ships as a file that drops into the SimLingo
checkout at `simlingo_training/models/`. The ROS node imports it lazily and
falls back to the unoptimised path if it fails (`fast_inference` parameter).

### How each was evaluated, without CARLA

| Model | Bench | What it does |
|---|---|---|
| ORION | `orion_ros/tools/bench_orion.py` | Loads once (~200 s), applies 14 variants cumulatively, CUDA-event timing per stage (ViT / LLM / map head / det head / planner) + max Δ trajectory vs the untouched model |
| MindDrive | `minddrive_ros/tools/bench_minddrive.py` | Same, 19 variants; fp32 and fp16 are separate invocations with `--ref-out`/`--ref-in` carrying the fp32 reference across, so fp16 is compared to the true reference and not to fp16 eager |
| SimLingo | `bench/simlingo/simlingo_fast_check.py` | Correctness + speed of `fast_inference` against the untouched model on a node-shaped input |

Plus, per model, an end-to-end path with no simulator:

- `orion_ros/tools/fake_carla_topics.py` — six 1600×900 cameras, odometry,
  speed, IMU at a chosen rate. Smoke test: no route, so the PID publishes
  nothing, but the node runs real inferences.
- `minddrive_ros/tools/fake_carla_topics.py` — same plus a synthetic
  `CarlaRoute`, so the full decision→control loop runs.
- `bench/simlingo/simlingo_fake_cam.py` — compressed RGB at 10 Hz matching the
  real CARLA stream.

And the diagnosis scripts that found the SimLingo causes, kept because the same
questions will be worth asking on Thor:

| Script | Question it answers |
|---|---|
| `simlingo_profile_baseline.py` | Where does the per-frame time go — vision / language decode / driving head? |
| `_clock_test.py` | Does the devfreq governor explain the gap? Warm vs cold clocks, duty-cycled vs back-to-back. |
| `_cpu_contention.py` | How much does CPU contention inflate a launch-bound model? |
| `simlingo_ros_probe.py` | Which part of the ROS node's concurrency costs what (6 configurations, executor / timers / subscriptions / logging). |
| `simlingo_gil_test.py` | Is the ROS gap GIL contention from the 20 Hz control timer? |
| `simlingo_opt_prototype.py` | End-to-end optimised vs as-is, before the work was folded into `fast_inference.py`. |

ORION also has five unit tests (`orion_ros/test/`) covering batch construction,
camera handling, temporal-memory continuity, PID parity against the reference
agent, and route reset.

---

## Running it

### 1. Paths

`/benchmarking` is the **in-container** bind-mount point and appears throughout.
Almost every occurrence is an argparse default, a ROS parameter default or a
docstring, so it is overridable; keeping the same mount point is still by far
the least friction. The Orin layout was a single work directory bound at
`/benchmarking` containing this tooling next to the model checkouts:

```
<work>/                     ->  /benchmarking
  Orion/  MindDrive/  simlingo/  Orion-Lite/     model checkouts
  imgdiag/                                       the frames in assets/frames/
  alpamayo-autoware/src/{orion,minddrive,simlingo}_ros/   this repo's ros2_ws/src
```

Only the host-side paths in `scripts/start_*.sh` (`BENCH=`, `SSD_MOUNT=`,
`MODELS_HOST=`) must change for Thor. The two fake-topic publishers also honour
`VLA_FRAMES_DIR` if you keep the frames in this repo rather than at
`/benchmarking/imgdiag`.

### 2. Pin the clocks before measuring anything

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

This is not a detail. On the Orin it was 3.77× of SimLingo's 11×, and the
baseline's SD of 574 ms was itself the governor's signature. Any number taken
without it is noise.

### 3. Bench a model

```bash
# ORION
docker exec orion_ros bash -c 'source /opt/ros/humble/setup.bash;
    source /opt/orion_ws/install/setup.bash; cd /root/Orion;
    python3 /benchmarking/.../orion_ros/tools/bench_orion.py --frames 4'

# MindDrive: fp32 reference first, then the fp16 chain against it
docker exec minddrive_ros bash -c '... bench_minddrive.py --precision fp32 \
    --frames 3 --variants baseline --ref-out /tmp/ref_fp32.npz'
docker exec minddrive_ros bash -c '... bench_minddrive.py --precision fp16 \
    --frames 3 --ref-in /tmp/ref_fp32.npz'

# SimLingo
docker exec sim_ros python3 /benchmarking/bench/simlingo/simlingo_fast_check.py
```

Compare the result against `reference_results/` and the tables in the docs.

### 4. Run a node without the simulator

```bash
ros2 launch orion_ros orion_withpid.launch.py     # or minddrive / simlingo
python3 .../tools/fake_carla_topics.py 45         # duration_s [hz]
```

---

## Documentation

| Document | Contents |
|---|---|
| `ros2_ws/src/simlingo_ros/SIMLINGO_inference_optimisation.md` | Full write-up: diagnosis, the falsified middleware hypothesis, why TensorRT was deprioritised, limitations, reproduction (540 lines) |
| `ros2_ws/src/orion_ros/docs/ORION_inference_optimisation_methodology.md` | ORION methodology draft, publication-shaped |
| `ros2_ws/src/orion_ros/ORION_ROS_NODE.md` | ORION node contract, parameters, §6 speedup tables, TensorRT and INT8 negative results |
| `ros2_ws/src/minddrive_ros/MINDDRIVE_ROS_NODE.md` | MindDrive equivalent, incl. the full INT8 / SmoothQuant sweep |
| `ros2_ws/src/simlingo_ros/IMPLEMENTATION_NOTES.md` | SimLingo node implementation notes |
| `PORTING.md` | **Read this first when moving to Thor** — what to re-validate and what will not transfer |

## A note on the source

The code in this repository is published with inline comments stripped and
site-specific values replaced by placeholders. Module, class and function
docstrings are kept, so each file and each speedup still states what it does and
why it is safe. The longer reasoning — where the numerics drift, what was tried
and rejected, and the per-stage measurements — is in the Markdown documents
listed above, which remain the authoritative reference.

Placeholders you must set for your own site: `/home/USER/...` and
`/media/USER/EXTSSD` (paths), `carla-host` and the `192.0.2.x` addresses (the
simulator machine, if you use the closed-loop path), and `ztXXXXXXXXX` (network
interface). None of the benches in this repository need any of them — they
matter only for the closed-loop launchers in `scripts/`.

Container images are real and pullable: `karmishthannauth/orion_env_ros:v01`,
`karmishthannauth/minddrive_env_ros:v01` and
`karmishthannauth/simlingo:humble-cyclonedd`.

## Licence and attribution

Apache License 2.0 — see `LICENSE`. Attribution for the upstream projects is in
`NOTICE`.

This repository does not redistribute any model's source. The ORION, MindDrive,
SimLingo and Orion-Lite checkouts are supplied by you and imported at runtime;
every speedup is a post-build transform on the instantiated model object, and
the three small source changes that are needed ship as unified diffs under
`env/minddrive/` and `simlingo_patch/`. Model weights are not included.

Two upstream terms are worth knowing even though nothing here redistributes the
material they cover: **Bench2Drive** is CC BY-NC-ND 4.0, and the **SimLingo
dataset** is under a non-commercial licence from Wayve. They do not constrain
this repository, but they do constrain a full closed-loop evaluation.
**Orion-Lite publishes no licence at all**, so all rights are reserved by its
authors — ask them before using that path.

## Not included

Closed-loop CARLA evaluation: the Bench2Drive agents (`team_code/*_b2d_agent.py`
in each upstream repo), the leaderboard and scenario-runner harnesses, and the
CARLA-side bridge. The ROS nodes reproduce the agents' payload construction and
control law faithfully — that is what `test_pid_parity.py` checks — but driving
scores are not what this repo measures.

Model weights are not included and are not in git.
