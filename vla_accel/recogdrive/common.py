"""Shared helpers for the ReCogDrive benches: agent loading, B2D frame selection, stage timing."""

import os
import sys
import tempfile
import time
from contextlib import contextmanager

import torch

from vla_accel.common import b2d
from vla_accel.recogdrive import inputs

REPO = os.environ.get("RECOGDRIVE_REPO", "/opt/recogdrive")
WEIGHTS = os.environ.get("RECOGDRIVE_WEIGHTS", "/data/weights/ReCogDrive")
VLM = os.environ.get("RECOGDRIVE_VLM", WEIGHTS)
CKPT = os.environ.get("RECOGDRIVE_CKPT", os.path.join(WEIGHTS, "ReCogDrive_Diffusion_Planner_2B_RL.ckpt"))
TRT_DIR = os.environ.get("RECOGDRIVE_TRT_DIR", os.path.join(WEIGHTS, "trt"))
FRAME_DIR = os.environ.get("RECOGDRIVE_FRAME_DIR", os.path.join(tempfile.gettempdir(), "recogdrive_frames"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def build_agent(vlm_path=VLM, checkpoint=CKPT, vlm_size="small"):
    """ReCogDriveAgent as the NAVSIM 2B eval builds it, initialised and in eval mode."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        vlm_path=vlm_path, checkpoint_path=checkpoint, cam_type="single",
        vlm_type="internvl", dit_type="small", sampling_method="ddim",
        cache_mode=False, cache_hidden_state=False, grpo=False, vlm_size=vlm_size)
    agent.initialize()
    agent.eval()
    return agent


def add_b2d_args(p, frames=3, clips=3, held_out=True):
    """Add the Bench2Drive input options to an argparse parser."""
    p.add_argument("--b2d-dir", default=b2d.B2D_DIR)
    p.add_argument("--clip", action="append", default=[], help="clip name or path (repeatable); default: --clips clips")
    p.add_argument("--clips", type=int, default=clips, help="number of clips when --clip is not given")
    p.add_argument("--split", choices=["held", "calib", "all"], default="held" if held_out else "calib")
    p.add_argument("--frames", type=int, default=frames, help="frames per clip")
    p.add_argument("--stride", type=int, default=10, help="frame stride (10 = 1 s)")
    p.add_argument("--start", type=int, default=20, help="first frame (>= 15 gives a full 1.5 s history)")
    p.add_argument("--jpeg-quality", type=int, default=0, help="> 0 re-encodes the camera JPEG")
    p.add_argument("--frame-dir", default=FRAME_DIR, help="where the front frames are written for the agent")
    return p


def select_frames(args, split=None):
    """(clip, frame index) pairs chosen by the B2D options."""
    if args.clip:
        clips = [c if os.path.isdir(c) else os.path.join(args.b2d_dir, c) for c in args.clip]
    else:
        calib, held = b2d.split_clips(args.b2d_dir)
        clips = {"held": held, "calib": calib, "all": calib + held}[split or args.split][:args.clips]
    return [(c, i) for c in clips for i in b2d.frame_indices(c, args.frames, args.stride, args.start)]


def load_inputs(args, split=None):
    """[(name, AgentInput, front RGB)] for the selected B2D frames."""
    out = []
    for clip, idx in select_frames(args, split):
        ai, rgb = inputs.make_input(clip, idx, args.frame_dir, args.jpeg_quality)
        out.append((f"{os.path.basename(os.path.normpath(clip))}/{idx:05d}", ai, rgb))
    if not out:
        raise RuntimeError("no Bench2Drive frames selected")
    return out


class Stages:
    """Wall-clock per stage, with torch.cuda.synchronize() on both sides."""

    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, name):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def wrap(self, obj, attr, name):
        """Time every call of ``obj.attr`` under ``name``; returns the original."""
        fn = getattr(obj, attr)

        def timed(*a, **k):
            with self(name):
                return fn(*a, **k)
        setattr(obj, attr, timed)
        return fn
