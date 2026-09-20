#!/usr/bin/env python3
"""Load MindDrive exactly as team_code/minddrive_b2d_agent.py does and run a few
forwards on a black frame.  Proves the environment (imports, compiled ops, LLM
weights, checkpoint) and reports load time, forward time and peak RSS.

  docker exec <ctr> bash -c 'python3 /benchmarking/minddrive_env/smoke_infer.py [3b|05b] [n_forwards]'
"""
import os, sys, time, resource
import numpy as np
import torch

REPO = "/benchmarking/MindDrive"
sys.path.insert(0, REPO)
sys.path.insert(0, "/benchmarking/alpamayo-autoware/src/minddrive_ros")
os.chdir(REPO)

variant = (sys.argv[1] if len(sys.argv) > 1 else "3b").lower()
n_fwd = int(sys.argv[2]) if len(sys.argv) > 2 else 3
if variant == "3b":
    CFG, CKPT = "adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py", "/models/MindDrive/minddrive_3b_rltrain.pth"
else:
    CFG, CKPT = "adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py", "/models/MindDrive/minddrive_rltrain.pth"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
import mmcv._ext
from minddrive_ros.agent_constants import (
    CAMERA_ORDER, build_agent_results, batch_to_device, custom_wrap_fp16_model)
from team_code.pid_controller_de import PIDController

print(f"[smoke] mmcv from {os.path.dirname(os.path.abspath(sys.modules['mmcv'].__file__))}", flush=True)
t0 = time.time()
cfg = Config.fromfile(CFG)
print(f"[smoke] cfg fp32_infer={cfg.model.get('fp32_infer')} fp16_infer={cfg.model.get('fp16_infer')} "
      f"lm={cfg.model.get('lm_model_type')} llm={cfg.model.get('lm_head')}", flush=True)
model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
print(f"[smoke] build_model {time.time()-t0:.0f}s rss={rss_gb():.1f}GB", flush=True)
t1 = time.time()
ck = load_checkpoint(model, CKPT, map_location="cpu")
print(f"[smoke] load_checkpoint {time.time()-t1:.0f}s rss={rss_gb():.1f}GB", flush=True)
del ck
model.cuda()
model.eval()
print(f"[smoke] cuda+eval {time.time()-t0:.0f}s total, rss={rss_gb():.1f}GB, "
      f"gpu_alloc={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
pipeline = Compose([p for p in cfg.inference_only_pipeline
                    if p["type"] not in ["LoadMultiViewImageFromFilesInCeph"]])
print(f"[smoke] pipeline: {[p['type'] for p in cfg.inference_only_pipeline]}", flush=True)
print(f"[smoke] has planning_memory attr: {hasattr(model, 'planning_memory')}  test_flag={model.test_flag}", flush=True)

pid = PIDController()
dev = torch.device("cuda")
for i in range(n_fwd):
    t2 = time.time()
    black = np.zeros((900, 1600, 3), dtype=np.uint8)
    images = {c: black for c in CAMERA_ORDER}
    can_bus = np.zeros(18)
    can_bus[3:7] = [1.0, 0.0, 0.0, 0.0]
    can_bus[7] = 2.0
    results = build_agent_results(images, can_bus, 0.0, 4, "smoke", i, i / 20.0, get_box_type)
    results = pipeline(results)
    batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
    batch_to_device(batch, dev)
    t3 = time.time()
    with torch.no_grad():
        custom_wrap_fp16_model(model)
        out = model(batch, return_loss=False)
    torch.cuda.synchronize()
    t4 = time.time()
    pb = out[0]["pts_bbox"]
    speed_wp = pb["ego_fut_preds"].cpu().numpy()
    path_wp = pb["pw_ego_fut_pred"].cpu().numpy()
    steer, throttle, brake, meta = pid.control_pid(path_wp, speed_wp, np.float64(2.0), np.array([0.0, 10.0]))
    print(f"[smoke] fwd {i}: prep={1e3*(t3-t2):.0f}ms forward={1e3*(t4-t3):.0f}ms "
          f"speed_wp={speed_wp.shape} path_wp={path_wp.shape} "
          f"speed_value={pb.get('speed_value')} path_value={pb.get('path_value')} "
          f"pid steer={float(steer):+.3f} thr={float(throttle):.3f} brake={float(brake)} "
          f"gpu_max={torch.cuda.max_memory_allocated()/1e9:.1f}GB rss={rss_gb():.1f}GB", flush=True)
    print(f"[smoke]   speed_wp[0..1]={speed_wp[:2].tolist()} path_wp[-1]={path_wp[-1].tolist()}", flush=True)
print("[smoke] OK", flush=True)
