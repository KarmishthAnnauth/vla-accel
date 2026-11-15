"""Shared SimLingo model loading and Bench2Drive input building for every bench in this folder.

Front camera -> 1024x359 RGB (CARLA agent's frame and hood crop) -> 2 InternVL2 tiles -> 543-token prompt.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from vla_accel.common import b2d, jetson

SIMLINGO_DIR = os.environ.get("SIMLINGO_DIR", "/simlingo")
SIMLINGO_CKPT = os.environ.get(
    "SIMLINGO_CKPT", "/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")

SPECIAL_TOKENS = ["<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
                  "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>"]
CAM_W, CAM_H, CAM_FOV = 1024, 512, 110
TILE = 448


def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add checkout, checkpoint and Bench2Drive clip/frame options."""
    ap.add_argument("--simlingo-dir", default=SIMLINGO_DIR, help="SimLingo checkout")
    ap.add_argument("--ckpt", default=SIMLINGO_CKPT, help="pytorch_model.pt of the checkpoint")
    ap.add_argument("--hydra-cfg", default="",
                    help="training .hydra/config.yaml (default: three levels above --ckpt)")
    ap.add_argument("--b2d-dir", default=b2d.B2D_DIR)
    ap.add_argument("--clip", default="0", help="clip index into the sorted list, or a clip name")
    ap.add_argument("--frames", type=int, default=5, help="frames from the clip")
    ap.add_argument("--stride", type=int, default=10, help="anno frames apart (10 = 1 s)")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--jpeg-quality", type=int, default=0,
                    help="re-encode the B2D JPEG at this quality (0 = use as stored)")
    return ap


@dataclass
class Ctx:
    """Everything needed to turn a camera frame into a DrivingInput."""
    cfg: object
    tokenizer: object
    conv: object
    transform: object
    n_img_tok: int
    device: object


def _setup_path(simlingo_dir: str) -> None:
    """Make simlingo_training and team_code importable."""
    for p in (simlingo_dir + "/team_code", simlingo_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


def _conversation_module(cache_dir: str, variant: str):
    """Load InternVL2's conversation.py from the local cache (HF snapshot layout too)."""
    hits = [f"{cache_dir}/conversation.py"] + sorted(
        glob.glob(f"{cache_dir}/**/conversation.py", recursive=True))
    path = next((h for h in hits if os.path.isfile(h)), None)
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=variant, filename="conversation.py")
    spec = importlib.util.spec_from_file_location("_simlingo_conv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(args) -> Tuple[object, Ctx]:
    """Instantiate the Hydra DrivingModel, load the checkpoint, return (model, ctx)."""
    _setup_path(args.simlingo_dir)
    import hydra
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoProcessor
    from simlingo_training.utils.internvl2_utils import build_transform

    dev = torch.device("cuda")
    cfg_path = args.hydra_cfg or str(Path(args.ckpt).parent.parent.parent / ".hydra" / "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    variant = cfg.model.vision_model.variant
    cache = f"{args.simlingo_dir}/pretrained/{variant.split('/')[1]}"

    proc = AutoProcessor.from_pretrained(variant, trust_remote_code=True, cache_dir=cache)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tok.padding_side = "left"

    vcfg = AutoConfig.from_pretrained(variant, trust_remote_code=True, cache_dir=cache)
    n_img_tok = int(((vcfg.force_image_size or vcfg.vision_config.image_size)
                     // vcfg.vision_config.patch_size) ** 2 * (vcfg.downsample_ratio ** 2))

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(cfg.model, cfg_data_module=cfg.data_module, processor=proc,
                                    cache_dir=cache, _recursive_=False).to(dev)
    torch.set_default_dtype(default)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    ctx = Ctx(cfg=cfg, tokenizer=tok, conv=_conversation_module(cache, variant),
              transform=build_transform(input_size=TILE), n_img_tok=n_img_tok, device=dev)
    return model, ctx


def carla_frame(bgr: np.ndarray) -> np.ndarray:
    """B2D front image -> RGB at the agent's 1024x512 framing with the bottom 4.8/16 cropped."""
    h, w = bgr.shape[:2]
    new_h = round(h * CAM_W / w)
    img = cv2.resize(bgr, (CAM_W, new_h), interpolation=cv2.INTER_AREA)
    if new_h >= CAM_H:
        top = (new_h - CAM_H) // 2
        img = img[top:top + CAM_H]
    else:
        img = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_h = int(img.shape[0] - (img.shape[0] * 4.8) // 16)
    return img[:crop_h]


def _to_model_frame(world_xy: np.ndarray, ego: np.ndarray) -> np.ndarray:
    """CARLA world (x, y) points -> SimLingo ego frame (x forward, y right)."""
    pts = np.zeros((len(world_xy), 3))
    pts[:, 0], pts[:, 1] = world_xy[:, 0], -world_xy[:, 1]  # CARLA y -> right-handed
    loc = b2d.to_local(pts, ego)[:, :2]
    return np.stack([loc[:, 0], -loc[:, 1]], axis=1)


def target_points(clip: str, idx: int, anno: dict) -> np.ndarray:
    """Near and far route targets as (2, 2) in the model frame, from the anno or the ego's future path."""
    ego = b2d.ego_pose(anno)
    keys = ("x_command_near", "y_command_near", "x_command_far", "y_command_far")
    if all(anno.get(k) is not None for k in keys):
        world = np.array([[anno[keys[0]], anno[keys[1]]], [anno[keys[2]], anno[keys[3]]]], float)
        return _to_model_frame(world, ego).astype(np.float32)
    near, far = b2d.target_point(clip, idx, 40), b2d.target_point(clip, idx, 80)
    return np.array([[near[0], -near[1]], [far[0], -far[1]]], dtype=np.float32)  # y left -> right


def build_example(ctx: Ctx, rgb: np.ndarray, speed: float, tps: np.ndarray):
    """Build the DrivingInput the CARLA agent would; returns (example, prompt_tokens)."""
    import torch
    from PIL import Image as PILImage
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from simlingo_training.utils.internvl2_utils import dynamic_preprocess

    dev, tok = ctx.device, ctx.tokenizer
    tiles = dynamic_preprocess(PILImage.fromarray(rgb), image_size=TILE,
                               use_thumbnail=ctx.cfg.model.vision_model.use_global_img, max_num=2)
    pixel = torch.stack([ctx.transform(t) for t in tiles])
    processed = pixel.unsqueeze(0).unsqueeze(0)

    prompt = (f"Current speed: {round(speed, 1)} m/s. "
              "Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
    tpl = ctx.conv.get_conv_template("internlm2-chat")
    tpl.append_message(tpl.roles[0], f"<image>\n{prompt}")
    tpl.append_message(tpl.roles[1], None)
    query = tpl.get_prompt()
    query = query.replace(
        tpl.system_template.replace("{system_message}", tpl.system_message) + tpl.sep, "")
    query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * ctx.n_img_tok * pixel.shape[0]
                          + "</img>", 1)
    enc = tok([query], padding=True, return_tensors="pt",
              return_offsets_mapping=True, add_special_tokens=False)
    label = LanguageLabel(
        phrase_ids=enc["input_ids"].to(dev),
        phrase_valid=(enc["input_ids"] != tok.pad_token_id).to(dev),
        phrase_mask=(enc["input_ids"] != tok.pad_token_id).to(dev),
        placeholder_values=[{tok.convert_tokens_to_ids("<TARGET_POINT>"): tps}],
        language_string=[query], loss_masking=None)

    _, _, _, _, H, W = processed.shape
    focal = W / (2.0 * math.tan(CAM_FOV * math.pi / 360.0))
    K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]],
                     dtype=torch.float32).unsqueeze(0).to(dev)
    E = torch.eye(4, dtype=torch.float32)
    E[:3, 3] = torch.tensor([-1.5, 0.0, 2.0])
    E = E.unsqueeze(0).to(dev)

    example = DrivingInput(
        camera_images=processed.to(dev).bfloat16(), image_sizes=None,
        camera_intrinsics=K, camera_extrinsics=E,
        vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=dev),
        target_point=torch.from_numpy(tps[:1]).float().to(dev),
        prompt=label, prompt_inference=label)
    return example, enc["input_ids"].shape[1]


def resolve_clip(args) -> str:
    """The clip selected by ``--clip`` (index or name)."""
    clips = b2d.list_clips(args.b2d_dir)
    if args.clip.isdigit():
        return clips[int(args.clip)]
    match = [c for c in clips if os.path.basename(c) == args.clip or c == args.clip]
    if not match:
        raise FileNotFoundError(f"clip {args.clip!r} not under {args.b2d_dir}")
    return match[0]


def load_frame(clip: str, idx: int, jpeg_quality: int = 0) -> Tuple[np.ndarray, float, np.ndarray]:
    """(rgb, speed m/s, target points) for one B2D frame."""
    anno = b2d.load_anno(clip, idx)
    rgb = carla_frame(b2d.load_image(clip, "CAM_FRONT", idx, jpeg_quality))
    return rgb, float(anno.get("speed", 0.0)), target_points(clip, idx, anno)


def examples(args, ctx: Ctx) -> Iterator[Tuple[str, object, int]]:
    """Yield (frame tag, DrivingInput, prompt tokens) for the selected clip frames."""
    clip = resolve_clip(args)
    idxs: List[int] = b2d.frame_indices(clip, args.frames, args.stride, args.start)
    if not idxs:
        raise ValueError(f"no frames in {clip} for start={args.start} stride={args.stride}")
    for idx in idxs:
        rgb, speed, tps = load_frame(clip, idx, args.jpeg_quality)
        ex, n_tok = build_example(ctx, rgb, speed, tps)
        yield f"{os.path.basename(clip)}/{idx:05d}", ex, n_tok


def first_example(args, ctx: Ctx) -> Tuple[str, object, int]:
    """The first selected frame only."""
    return next(examples(args, ctx))


def timeit(fn, n: int = 6, warmup: int = 2) -> Tuple[float, float]:
    """(mean ms, sd ms) of ``fn`` over ``n`` synchronised calls after ``warmup``."""
    import torch
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.mean(times)), float(np.std(times))


gpu_mhz = jetson.gpu_mhz
