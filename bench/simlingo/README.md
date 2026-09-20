# SimLingo benches

SimLingo's optimisations live in `simlingo_patch/simlingo_training/models/fast_inference.py`,
which drops into a SimLingo checkout at `simlingo_training/models/`. These
scripts are how it was arrived at and how it is checked.

All of them expect the SimLingo checkout at `/benchmarking/simlingo` and the
checkpoint at `/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`,
and run inside the `sim_ros` container. None of them need CARLA.

| Script | Purpose |
|---|---|
| `simlingo_fast_check.py` | **The one to run first.** Loads the real checkpoint, builds a node-shaped input (1024×359 → 2 tiles → 543-token prompt), compares the fast path against the untouched original for both speed and output. |
| `simlingo_profile_baseline.py` | Phase attribution: vision encode / autoregressive decode / driving head. Replicates `simlingo_node._load_model` + `_run_inference` on a synthetic image. |
| `simlingo_opt_prototype.py` | The five optimisations end-to-end against the as-is baseline, before they were folded into `fast_inference.py`. Useful as a readable statement of what each one does. |
| `_clock_test.py` | Does the devfreq governor explain the gap? Warm vs cold clocks, node duty cycle vs back-to-back. **Re-run this on Thor before anything else.** |
| `_cpu_contention.py` | How much a competing CPU load inflates a launch-bound model. |
| `simlingo_ros_probe.py` | Reproduces the node's concurrency structure in 6 configurations (`PROBE=bare\|full\|noctrl\|noimg\|nolog\|single`) to attribute cost to the executor, the control timer, the image subscription and logging. |
| `simlingo_gil_test.py` | Isolates GIL contention from the 20 Hz control timer specifically. |
| `simlingo_fake_cam.py` | Synthetic CARLA image publisher: compressed RGB at 10 Hz matching the real stream. `python3 simlingo_fake_cam.py [hz]` |

The write-up built from these is
`ros2_ws/src/simlingo_ros/SIMLINGO_inference_optimisation.md`.
