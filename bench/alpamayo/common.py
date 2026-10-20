"""Shared helpers for the Alpamayo 1.5 benches: model load, node-shaped inputs, stage timing.

The inputs reproduce what ``alpamayo_node._run_inference`` feeds the model
(GPU uint8 frames at 560x1008, ``apply_chat_template(device="cuda")``, 16-step
ego history) so that numbers measured here transfer to the node.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

MODEL_PATH = os.environ.get("ALPAMAYO_MODEL", "/media/USER/EXTSSD/alpamayo-1.5")
FRAMES_DIR = os.environ.get(
    "VLA_FRAMES_DIR", "/home/USER/vla_benchmarking/benchmarking/imgdiag"
)
GPU_FREQ = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq"
NUM_FRAMES = 4
NUM_HISTORY = 16
CAMERA_SETS = {1: [1], 4: [0, 1, 2, 6]}


def gpu_mhz() -> int:
    try:
        return int(Path(GPU_FREQ).read_text()) // 1_000_000
    except OSError:
        return -1


def load_model(path: str = MODEL_PATH):
    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(path, dtype=torch.bfloat16).eval().to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print(f"[load] {time.perf_counter() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on GPU", flush=True)
    return model, processor


def make_payload(n_cams: int = 1, variant: int = 0, speed: float = 5.0) -> dict:
    """A node-shaped payload: frames [n_cams, 4, 3, 560, 1008] uint8 on the GPU."""
    files = sorted(Path(FRAMES_DIR).glob("*.png"))
    base = [
        torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1)
        for f in files
    ]
    cams = []
    for c in range(n_cams):
        img = base[(c + variant) % len(base)].cuda().float()[None]
        img = torch.nn.functional.interpolate(
            img, size=(560, 1008), mode="bicubic", align_corners=False
        ).clamp(0, 255).to(torch.uint8)[0]
        cams.append(torch.stack([img.roll(4 * (t + variant), dims=-1) for t in range(NUM_FRAMES)]))
    xyz = torch.zeros(1, 1, NUM_HISTORY, 3)
    xyz[0, 0, :, 0] = (torch.arange(NUM_HISTORY) - (NUM_HISTORY - 1)) * 0.1 * speed
    rot = torch.eye(3).expand(1, 1, NUM_HISTORY, 3, 3).clone()
    return {
        "image_frames": torch.stack(cams),
        "camera_indices": torch.tensor(CAMERA_SETS[n_cams], dtype=torch.int64),
        "ego_history_xyz": xyz,
        "ego_history_rot": rot,
    }


def make_inputs(processor, payload: dict, nav_text: str | None = None) -> dict:
    """``alpamayo_node._run_inference`` up to the model call."""
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
        self.wrap(model.vlm.model.visual, "forward", "vit")
        self.wrap(model.vlm.model.language_model, "forward", "llm")
        self.wrap(model.vlm.lm_head, "forward", "lm_head")
        self.wrap(model.vlm, "generate", "generate")
        self.wrap(model.expert, "forward", "expert")
        self.wrap(model.diffusion, "sample", "diffusion")
        self.wrap(model.action_space, "action_to_traj", "to_traj")
        return self

    def detach(self) -> None:
        for obj, attr, orig in reversed(self._undo):
            setattr(obj, attr, orig)
        self._undo.clear()

    def reset(self) -> None:
        self.rec.clear()


B2D_DIR = os.environ.get("B2D_DIR", "/media/USER/EXTSSD/Bench2Drive-calib")


def b2d_split() -> tuple[list[str], list[str]]:
    """(calibration clips, held-out clips): every sixth clip is held out."""
    import json

    with open(os.path.join(B2D_DIR, "clips.json")) as f:
        clips = sorted(json.load(f))
    held = clips[2::6]
    return [c for c in clips if c not in held], held


def b2d_payload(clip: str, t: int) -> dict:
    """A node-shaped payload at frame ``t`` of a clip: the four front-camera
    frames up to t (10 Hz) and the 16-step ego history from the annotations.
    CARLA is left-handed (y to the right); the model's ego frame is x forward,
    y left, so y and yaw are mirrored."""
    import gzip
    import json

    frames = []
    for k in range(t - NUM_FRAMES + 1, t + 1):
        img = Image.open(os.path.join(B2D_DIR, clip, "camera", "rgb_front", f"{k:05d}.jpg")).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).cuda().float()[None]
        frames.append(torch.nn.functional.interpolate(
            x, size=(560, 1008), mode="bicubic", align_corners=False).clamp(0, 255).to(torch.uint8)[0])
    pos, yaw = [], []
    for k in range(t - NUM_HISTORY + 1, t + 1):
        with gzip.open(os.path.join(B2D_DIR, clip, "anno", f"{k:05d}.json.gz")) as f:
            a = json.load(f)
        ego = next(b for b in a["bounding_boxes"] if b.get("class") == "ego_vehicle")
        pos.append([ego["location"][0], -ego["location"][1]])
        yaw.append(-np.radians(ego["rotation"][2]))
    pos, yaw = np.asarray(pos, dtype=np.float64), np.asarray(yaw)
    c, s = np.cos(yaw[-1]), np.sin(yaw[-1])
    to_local = np.array([[c, s], [-s, c]])
    xyz = np.zeros((NUM_HISTORY, 3), dtype=np.float32)
    xyz[:, :2] = (pos - pos[-1]) @ to_local.T
    rot = np.zeros((NUM_HISTORY, 3, 3), dtype=np.float32)
    d = yaw - yaw[-1]
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = np.cos(d), -np.sin(d), np.sin(d), np.cos(d), 1.0
    return {
        "image_frames": torch.stack(frames)[None],
        "camera_indices": torch.tensor(CAMERA_SETS[1], dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(xyz)[None, None],
        "ego_history_rot": torch.from_numpy(rot)[None, None],
    }


def b2d_payloads(clips: list[str], per_clip: int) -> list[dict]:
    out = []
    for clip in clips:
        n = len(os.listdir(os.path.join(B2D_DIR, clip, "camera", "rgb_front")))
        for t in np.linspace(NUM_HISTORY + 4, n - 2, per_clip).astype(int):
            out.append(b2d_payload(clip, int(t)))
    return out
