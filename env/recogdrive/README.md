# ReCogDrive environment

No container: a Python 3.10 venv on top of the host's ROS 2 Humble (JetPack 6).

```bash
./run_capped.sh ./setup_venv.sh        # idempotent; VENV=... and RECOGDRIVE_REPO=... to relocate
```

| File | What it is |
|---|---|
| `setup_venv.sh` | Builds the venv. Wheels only, nothing is compiled. |
| `requirements-jetson.txt` | torch 2.10 / torchvision / triton 3.6 / flash-attn 2.8.3 for JetPack 6 + CUDA 12.6, pinned by URL and hash. |
| `requirements.txt`, `requirements-nodeps.txt` | What ReCogDrive's inference imports; the nuplan-devkit chain without its dependencies. |
| `run_capped.sh` | Runs a job in a memory-capped cgroup with a watchdog. Use it for every heavy job. |
| `check_imports.py`, `smoke_infer.py` | Import check; one plan from one image without ROS. |
| `stubs/decord` | Placeholder for `decord` (no aarch64 wheel; imported but unused by inference). |

Run with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the planner checkpoint is a
Lightning pickle) and `PYTHONNOUSERSITE=1` (the venv's `activate` sets it).

For Thor all of `requirements-jetson.txt` changes; see `PORTING.md`.
