#!/usr/bin/env python3
"""Load ReCogDrive exactly as the reference eval does and plan from one image,
without ROS.  The first thing to run when something about the environment or
the weights is in doubt.

  source /media/USER/EXTSSD/ReCogDrive/venv/bin/activate
  PYTHONPATH=$PYTHONPATH:recogdrive_env/stubs ./recogdrive_env/run_capped.sh \\
      python recogdrive_env/smoke_infer.py \\
        --vlm-path <vlm dir> --checkpoint <planner>.ckpt \\
        [--vlm-size small|large] [--image /path/to/front.jpg] [--runs 3] [--speed 5.0]

It makes the three calls run_pdm_score_recogdrive.py makes -- the agent's
constructor with the eval script's arguments, initialize(), and
compute_trajectory(agent_input) -- on an AgentInput built by the ROS node's own
agent_inputs.build_agent_input: the ego driving straight at --speed with the
"go straight" command.  Without --image a black 1920x1080 frame is used, which
only proves the pipeline runs.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p.add_argument("--repo", default=os.path.join(BENCH, "recogdrive"))
p.add_argument("--node-src", default=os.path.join(BENCH, "alpamayo-autoware/src/recogdrive_ros"))
p.add_argument("--vlm-path", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--vlm-size", default="small", choices=["small", "large"])
p.add_argument("--dit-type", default="small")
p.add_argument("--sampling-method", default="ddim")
p.add_argument("--image", default="")
p.add_argument("--runs", type=int, default=3)
p.add_argument("--speed", type=float, default=5.0)
p.add_argument("--command", default="straight", choices=["left", "straight", "right"])
args = p.parse_args()

sys.path.insert(0, args.repo)
sys.path.insert(0, args.node_src)
from recogdrive_ros import agent_inputs as ai

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

t0 = time.time()
agent = ReCogDriveAgent(
    trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
    vlm_path=args.vlm_path, checkpoint_path=args.checkpoint, cam_type="single",
    vlm_type="internvl", dit_type=args.dit_type, sampling_method=args.sampling_method,
    cache_mode=False, cache_hidden_state=False, grpo=False, vlm_size=args.vlm_size)
agent.initialize()
print(f"agent ready in {time.time() - t0:.0f} s; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB; "
      f"backbone {next(agent.backbone.parameters()).dtype}, planner {next(agent.action_head.parameters()).dtype}")

image_path = args.image
if not image_path:
    from PIL import Image
    image_path = os.path.join(tempfile.gettempdir(), "recogdrive_smoke_black.png")
    Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8)).save(image_path)
    print("no --image: planning from a black 1920x1080 frame")

history = ai.PoseHistory()
for k in range(41):
    t = k * 0.05
    history.add(t, args.speed * t, 0.0, 0.0, args.speed, 0.0)
poses, velocities, held = history.window(2.0)
command = {"left": ai.CMD_LEFT, "straight": ai.CMD_STRAIGHT, "right": ai.CMD_RIGHT}[args.command]
agent_input = ai.build_agent_input(poses, velocities, np.zeros(2), ai.command_one_hot(command), image_path)

for i in range(args.runs):
    t1 = time.time()
    trajectory = agent.compute_trajectory(agent_input)
    torch.cuda.synchronize()
    dt = (time.time() - t1) * 1e3
    out = np.asarray(trajectory.poses, dtype=np.float64)
    print(f"run {i}: {dt:.0f} ms, GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    for j, (x, y, h) in enumerate(out):
        print(f"   t+{0.5 * (j + 1):.1f}s  x={x:+7.2f}  y={y:+6.2f}  heading={np.degrees(h):+6.1f} deg")
