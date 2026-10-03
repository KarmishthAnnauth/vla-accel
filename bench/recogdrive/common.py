"""Shared helpers for the ReCogDrive optimisation scripts: build the agent the
way smoke_infer.py does, build AgentInputs, and time stages with CUDA sync."""
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

BENCH = os.environ.get("VLA_BENCH_DIR", "/home/USER/vla_benchmarking/benchmarking")
REPO = os.path.join(BENCH, "recogdrive")
NODE_SRC = os.environ.get("RECOGDRIVE_ROS_SRC", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../ros2_ws/src/recogdrive_ros"))
SSD = os.environ.get("RECOGDRIVE_WEIGHTS", "/media/USER/EXTSSD/ReCogDrive")
VLM = SSD
CKPT = os.path.join(SSD, "ReCogDrive_Diffusion_Planner_2B_RL.ckpt")
for p in (REPO, NODE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_agent(vlm_path=VLM, checkpoint=CKPT, vlm_size="small"):
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


def make_input(image_path, speed=5.0, command="straight", yaw_rate=0.0):
    """AgentInput as the ROS node builds it: 2 s of driving at `speed`."""
    from recogdrive_ros import agent_inputs as ai
    history = ai.PoseHistory()
    yaw = 0.0
    x = y = 0.0
    for k in range(41):
        t = k * 0.05
        history.add(t, x, y, yaw, speed, 0.0)
        x += speed * 0.05 * np.cos(yaw)
        y += speed * 0.05 * np.sin(yaw)
        yaw += yaw_rate * 0.05
    poses, velocities, _ = history.window(2.0)
    cmd = {"left": ai.CMD_LEFT, "straight": ai.CMD_STRAIGHT, "right": ai.CMD_RIGHT}[command]
    return ai.build_agent_input(poses, velocities, np.zeros(2), ai.command_one_hot(cmd), image_path)


def gpu_mhz():
    try:
        return int(open("/sys/class/devfreq/17000000.gpu/cur_freq").read()) // 1000000
    except OSError:
        return -1


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
        fn = getattr(obj, attr)

        def timed(*a, **k):
            with self(name):
                return fn(*a, **k)
        setattr(obj, attr, timed)
        return fn
