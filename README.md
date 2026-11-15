# VLA inference acceleration on Jetson AGX Orin

Code for the thesis work on reducing the inference latency of five vision-language-action
driving models on an **NVIDIA Jetson AGX Orin 64 GB** (JetPack R36.4.7, clocks pinned).
Each model has a self-contained folder with its acceleration module, offline benchmarks on
the **Bench2Drive mini** dataset, environment and Orin reference results.

## Results (Jetson AGX Orin)

| Model | Baseline (s) | Optimized (s) | Speedup |
|---|---:|---:|---:|
| [SimLingo](vla_accel/simlingo) | 3.77 | **0.25** | **15.1×** |
| [ORION](vla_accel/orion) | 2.03 | **0.96** | **2.1×** |
| [MindDrive](vla_accel/minddrive) | 2.69 | **0.88** | **3.1×** |
| [ReCogDrive](vla_accel/recogdrive) | 1.27 | **0.66** | **1.9×** |
| [Alpamayo 1.5](vla_accel/alpamayo) | 4.14 | **0.84** | **4.9×** |


The README in each model folder gives the step-by-step table and the optional configurations
(TensorRT, INT8, lite vision).

## Layout

```
vla_accel/
  common/       b2d.py (Bench2Drive clip reader), jetson.py (GPU clock helpers)
  simlingo/     fast_inference.py + bench, profile, ablation, clock test
  orion/        speedups.py + cumulative bench, B2D opt-in bench, INT8 stats, TensorRT ViT
  minddrive/    speedups.py + cumulative bench, docker/ (image + upstream patches)
  recogdrive/   speedups.py + bench, profile, TensorRT/INT8 ViT, env/ (venv setup)
  alpamayo/     speedups.py + bench, ablation, profile, TensorRT fp16/INT8 builders
```

Each model folder contains `speedups.py` (or `fast_inference.py`), `bench.py`, `requirements.txt`,
`run.sh`, `README.md` and `reference_results/` with the raw Orin output.

## Environments (Jetson Orin)

| Model | Environment |
|---|---|
| SimLingo | `docker pull karmishthannauth/simlingo:humble-cyclonedd` (built on `karmishthannauth/simlingo_env:v01`) |
| ORION | `docker pull karmishthannauth/orion_env_ros:v01` (pre-built ORION checkout and compiled mmcv ops) |
| MindDrive | built locally on top of `karmishthannauth/orion_env_ros:v01`: `vla_accel/minddrive/docker/build_image.sh` |
| ReCogDrive | Python 3.10 venv: `vla_accel/recogdrive/env/setup_venv.sh` |
| Alpamayo 1.5 | Python 3.10 venv: `pip install -r vla_accel/alpamayo/requirements.txt` + `alpamayo1_5` |

The upstream model checkouts and weights are not included. Each model README lists where they
are mounted and which environment variables point to them.

## Running

1. **Pin the clocks.** Unpinned clocks change the result:

   ```bash
   sudo nvpmodel -m 0 && sudo jetson_clocks
   ```

2. **Get Bench2Drive mini.** Download it from
   [Thinklab-SJTU/Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive). Extract the clips (`<clip>/camera/rgb_*/%05d.jpg`,
   `<clip>/anno/%05d.json.gz`) into one folder and export `B2D_DIR=/path/to/bench2drive-mini`.

3. **Run a model's bench** through its `run.sh`, which starts the right container or venv and
   sets `PYTHONPATH`:

   ```bash
   vla_accel/simlingo/run.sh   bench
   vla_accel/orion/run.sh      bench --compile-mode default
   vla_accel/minddrive/run.sh  bench --precision fp32 --variants baseline --ref-out /out/ref_fp32.npz
   vla_accel/minddrive/run.sh  bench --precision fp16 --ref-in /out/ref_fp32.npz
   vla_accel/recogdrive/run.sh bench
   vla_accel/alpamayo/run.sh   bench --frames 3
   ```

   Every bench compares the accelerated model against the unmodified one on the same inputs,
   with the same sampler noise. It reports per-stage timing and trajectory deviation.


## Licence

Apache License 2.0; see `LICENSE`. Upstream attribution is in `NOTICE`. No model source or
weights are redistributed. Every speedup is a transform applied to the instantiated model object.
