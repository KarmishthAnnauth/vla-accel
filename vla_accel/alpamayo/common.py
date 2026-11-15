"""Shared helpers for the Alpamayo 1.5 benches: model load, Bench2Drive inputs, stage timing."""

from __future__ import annotations

import os
import time
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from vla_accel.common import b2d, jetson

MODEL_PATH = os.environ.get("ALPAMAYO_MODEL", "/data/alpamayo-1.5")
NUM_FRAMES = 4
NUM_HISTORY = 16
IMAGE_HW = (560, 1008)
CAMERA_SETS = {1: [1], 4: [0, 1, 2, 6]}
# Alpamayo camera index -> Bench2Drive camera (the tele view reuses the front camera).
B2D_CAMERAS = {0: "CAM_FRONT_LEFT", 1: "CAM_FRONT", 2: "CAM_FRONT_RIGHT", 6: "CAM_FRONT"}

gpu_mhz = jetson.gpu_mhz


def load_model(path: str = MODEL_PATH):
    """Alpamayo1_5 in bf16 on the GPU and its processor."""
    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(path, dtype=torch.bfloat16).eval().to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print(f"[load] {time.perf_counter() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on GPU", flush=True)
    return model, processor


def _frame(clip: str, cam: str, k: int) -> torch.Tensor:
    """One camera image as uint8 [3, 560, 1008] on the GPU, bicubic-resized."""
    rgb = b2d.load_image(clip, cam, k)[..., ::-1].copy()
    x = torch.from_numpy(rgb).permute(2, 0, 1).cuda().float()[None]
    return torch.nn.functional.interpolate(
        x, size=IMAGE_HW, mode="bicubic", align_corners=False).clamp(0, 255).to(torch.uint8)[0]


def b2d_payload(clip: str, t: int, n_cams: int = 1) -> dict:
    """Model payload at frame ``t``: 4 frames up to t per camera and the 16-step ego history."""
    cams = CAMERA_SETS[n_cams]
    frames = torch.stack([
        torch.stack([_frame(clip, B2D_CAMERAS[c], k) for k in range(t - NUM_FRAMES + 1, t + 1)])
        for c in cams])
    hist = b2d.to_local(b2d.ego_history(clip, t, NUM_HISTORY), b2d.ego_pose(b2d.load_anno(clip, t)))
    xyz = np.zeros((NUM_HISTORY, 3), dtype=np.float32)
    xyz[:, :2] = hist[:, :2]
    d = hist[:, 2]
    rot = np.zeros((NUM_HISTORY, 3, 3), dtype=np.float32)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = (
        np.cos(d), -np.sin(d), np.sin(d), np.cos(d), 1.0)
    return {
        "image_frames": frames,
        "camera_indices": torch.tensor(cams, dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(xyz)[None, None],
        "ego_history_rot": torch.from_numpy(rot)[None, None],
    }


def scene_frames(clip: str, per_clip: int, stride: int = 0) -> list[int]:
    """Frame indices of a clip: evenly spread (stride 0) or ``stride`` apart after the history."""
    start = NUM_HISTORY + 4
    if stride > 0:
        return b2d.frame_indices(clip, per_clip, stride=stride, start=start)
    n = b2d.num_frames(clip)
    return [int(t) for t in np.linspace(start, n - 2, per_clip).astype(int)]


def b2d_payloads(clips: list[str], per_clip: int, stride: int = 0, n_cams: int = 1) -> list[dict]:
    """Payloads for ``per_clip`` frames of each clip."""
    return [b2d_payload(clip, t, n_cams) for clip in clips for t in scene_frames(clip, per_clip, stride)]


def add_b2d_args(ap, frames: int = 3) -> None:
    """The Bench2Drive input options shared by every script."""
    ap.add_argument("--b2d-dir", default=b2d.B2D_DIR, help="folder of Bench2Drive clips")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip names to use (default: the held-out split, every sixth clip from the third)")
    ap.add_argument("--frames", type=int, default=frames, help="frames per clip")
    ap.add_argument("--stride", type=int, default=0, help="frame spacing (0: spread evenly over the clip)")


def split(args) -> tuple[list[str], list[str]]:
    """(calibration, held-out) clip paths; ``--clips`` overrides the held-out set."""
    calib, held = b2d.split_clips(args.b2d_dir)
    if args.clips:
        held = [os.path.join(args.b2d_dir, c) for c in args.clips]
    return calib, held


def scenes(processor, args, n_cams: int = 1) -> list[dict]:
    """Model inputs for the held-out (or ``--clips``) Bench2Drive frames."""
    _, held = split(args)
    out = [make_inputs(processor, p) for p in b2d_payloads(held, args.frames, args.stride, n_cams)]
    print(f"{len(out)} scenes from Bench2Drive clips: {[os.path.basename(c) for c in held]}", flush=True)
    return out


def make_inputs(processor, payload: dict, nav_text: str | None = None) -> dict:
    """Tokenised model input for a payload, as the rollout expects it."""
    messages = helper.create_message(
        payload["image_frames"].flatten(0, 1),
        camera_indices=payload["camera_indices"],
        num_frames_per_camera=NUM_FRAMES,
        nav_text=nav_text,
    )
    tok = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        device="cuda",
    )
    tok = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in tok.items()}
    return helper.to_device(
        {
            "tokenized_data": tok,
            "ego_history_xyz": payload["ego_history_xyz"],
            "ego_history_rot": payload["ego_history_rot"],
        },
        device="cuda",
    )


class StageTimer:
    """Synchronised wall time of each sub-module call, by patching ``forward``."""

    def __init__(self) -> None:
        self.rec: dict[str, list[float]] = defaultdict(list)
        self._undo: list = []

    def wrap(self, obj, attr: str, name: str) -> None:
        """Time every call of ``obj.attr`` under ``name``."""
        orig = getattr(obj, attr)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            torch.cuda.synchronize()
            self.rec[name].append((time.perf_counter() - t0) * 1e3)
            return out

        setattr(obj, attr, timed)
        self._undo.append((obj, attr, orig))

    def attach(self, model) -> "StageTimer":
        """Wrap the model's stages."""
        self.wrap(model.vlm.model.visual, "forward", "vit")
        self.wrap(model.vlm.model.language_model, "forward", "llm")
        self.wrap(model.vlm.lm_head, "forward", "lm_head")
        self.wrap(model.vlm, "generate", "generate")
        self.wrap(model.expert, "forward", "expert")
        self.wrap(model.diffusion, "sample", "diffusion")
        self.wrap(model.action_space, "action_to_traj", "to_traj")
        return self

    def detach(self) -> None:
        """Restore the original methods."""
        for obj, attr, orig in reversed(self._undo):
            setattr(obj, attr, orig)
        self._undo.clear()

    def reset(self) -> None:
        """Clear the recorded times."""
        self.rec.clear()
