"""ORION model and pipeline construction from the upstream checkout (container /root/Orion)."""
from __future__ import annotations

import copy
import os
import sys

ORION_REPO = os.environ.get("ORION_REPO", "/root/Orion")
CONFIG = os.environ.get("ORION_CONFIG", os.path.join(ORION_REPO, "adzoo/orion/configs/orion_stage3_agent.py"))
CHECKPOINT = os.environ.get("ORION_CKPT", "/models/Orion/Orion.pth")
ENGINES_DIR = os.environ.get("ORION_ENGINES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines"))
INT8_STATS = os.path.join(ENGINES_DIR, "llm_act_stats.pt")

_CUSTOM_FP16 = dict(map_head=False, pts_bbox_head=False)


def use_repo(repo: str = ORION_REPO) -> None:
    """Make the upstream checkout importable and the CWD (its config uses relative ckpts/ paths)."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)


def custom_wrap_fp16_model(model) -> None:
    """Enable fp16 on every module except the det and map heads, as the reference agent does."""
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True
    for module_name, v in _CUSTOM_FP16.items():
        if module_name in model._modules:
            model._modules[module_name].fp16_enabled = v


def load_config(cfg_path: str = CONFIG, precision: str = "fp16"):
    """mmcv Config with the inference precision flags set."""
    from mmcv import Config

    cfg = Config.fromfile(cfg_path)
    if precision == "fp16":
        cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = True, False, False
    else:
        cfg.model["fp16_infer"], cfg.model["fp16_eval"], cfg.model["fp32_infer"] = False, False, True
    return cfg


def build_pipeline(cfg, vit_input_size: int = 640, workers: int = 0):
    """The agent's inference_only_pipeline, optionally resized and thread-parallel."""
    from mmcv.datasets.pipelines import Compose
    from vla_accel.orion import speedups as sp

    pipe_cfg = [copy.deepcopy(t) for t in cfg.inference_only_pipeline if t["type"] != "LoadMultiViewImageFromFilesInCeph"]
    if vit_input_size != 640:
        sp.set_pipeline_input_size(pipe_cfg, vit_input_size)
    pipeline = Compose(pipe_cfg)
    if workers > 1:
        sp.parallelize_pipeline(pipeline, workers)
    return pipeline


def build_model(cfg_path: str = CONFIG, ckpt_path: str = CHECKPOINT, precision: str = "fp16"):
    """Build ORION, load Orion.pth, move to GPU in eval mode; returns (model, cfg)."""
    use_repo()
    from mmcv.models import build_model as mm_build_model
    from mmcv.utils import load_checkpoint

    cfg = load_config(cfg_path, precision)
    model = mm_build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.cuda().eval()
    return model, cfg
