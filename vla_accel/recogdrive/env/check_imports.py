#!/usr/bin/env python3
"""Import everything ReCogDrive's inference needs and print the versions; exits non-zero on the first failure.

  python3 check_imports.py [/path/to/recogdrive checkout]
"""
import importlib
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RECOGDRIVE_REPO", "/opt/recogdrive")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MODULES = [
    "torch", "torchvision", "numpy", "transformers", "tokenizers", "accelerate",
    "peft", "timm", "einops", "flash_attn", "triton", "cv2",
    "diffusers.models.embeddings", "pytorch_lightning", "omegaconf", "pyquaternion",
    "shapely", "decord",
    "nuplan.planning.simulation.trajectory.trajectory_sampling",
    "navsim.common.dataclasses",
    "navsim.agents.recogdrive.recogdrive_agent",
]
for name in MODULES:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    top = sys.modules[name.split(".")[0]]
    print(f"ok   {name:58s} {str(getattr(top, '__version__', '')):28s} {getattr(top, '__file__', '')}")

import torch

print(f"cuda available: {torch.cuda.is_available()}")
if torch.version.cuda is None or not torch.cuda.is_available():
    print(f"FAIL torch {torch.__version__} has no usable CUDA (a PyPI CPU wheel?)")
    sys.exit(1)
