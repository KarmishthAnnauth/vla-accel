# VLA ROS2 integration + optimization

ROS2 node implementations of VLAs and their inference optimization for near real-time operation on edge hardware. Currently only the Jetson AGX Orin has been tested. Inference is measured during evaluation of the VLAs with CARLA simulator, where the sim and model communicate via ROS2 over a common network. However, offline eval is also included for a rapid verification.

---

## What was achieved on the Orin

| Model | Baseline | Optimised | Speedup | Trajectory deviation |
|---|---:|---:|---:|---|
| **SimLingo** (InternVL2-1B + driving head) | 3538 ms/frame | **321 ms/frame** | **11.0×** | 0.028 m max route |
| **ORION** (EVA-ViT-L + LLaVA-LLaMA-7B) | 2030 ms/frame | **~1000 ms/frame** | **~2.0×** | < 1 cm |
| **MindDrive** (EVA-ViT + LLaVA-Qwen2.5-3B, 2 experts) | 2318 ms/frame (fp32) | **874 ms/frame** | **2.65×** | 0.05–0.09 m |

Note: Documentation on how the optimizations were achieved will be made shortly.

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
vendored checkouts remain unchanged.

| Model | Module | Lines |
|---|---|---:|
| ORION | `ros2_ws/src/orion_ros/orion_ros/orion_speedups.py` | 833 |
| MindDrive | `ros2_ws/src/minddrive_ros/minddrive_ros/minddrive_speedups.py` | 1260 |
| SimLingo | `simlingo_patch/simlingo_training/models/fast_inference.py` | 350 |



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

## ROS2 Documentation
MD files for notes on how the ROS2 nodes were made and the thought process behind them. Care was taken that inference payload is prepared exactly as the models are trained on for a fair evaluation.

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

Closed-loop CARLA evaluation which is part of my ongoing thesis.

Model weights are not included and are not in git. 
