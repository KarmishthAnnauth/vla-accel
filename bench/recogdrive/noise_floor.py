#!/usr/bin/env python3
"""Where does the fast path's deviation come from?  Stage-by-stage against the
reference, and against the reference's own bf16 noise floor: the same LLM
computation done two mathematically equivalent ways with stock transformers
code (left-padded to 2800 = flash varlen kernel; unpadded = flash dense kernel)."""
import glob
import numpy as np
import torch
import common as C

agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
fast = FastReCogDrive(agent, log=print)
bb, vlm, lm = agent.backbone, agent.backbone.model, agent.backbone.model.language_model

seen = {}
orig_lm = lm.forward
def spy_lm(*a, **k):
    seen["embeds"], seen["mask"], seen["pos"] = k["inputs_embeds"].clone(), k["attention_mask"], k["position_ids"]
    return orig_lm(*a, **k)
lm.forward = spy_lm
orig_ga = agent.action_head.get_action
def spy_ga(vl, ai_, *a, **k):
    seen["vl"] = vl.clone(); seen["ai"] = ai_
    return orig_ga(vl, ai_, *a, **k)
agent.action_head.get_action = spy_ga

def rel(a, b):
    a, b = a.float(), b.float()
    row = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (f"rel-L2 {((a - b).norm() / b.norm()).item():.4f}  row-cos min {row.min().item():.5f} "
            f"mean {row.mean().item():.6f}  max|d| {(a - b).abs().max().item():.3f} (|ref| max {b.abs().max().item():.0f})")

frames = sorted(glob.glob("frames/*.bmp"))
for k, (speed, cmd, yaw) in enumerate([(5.0, "straight", 0.0), (12.0, "left", 0.25)]):
    ai = C.make_input(frames[k], speed=speed, command=cmd, yaw_rate=yaw)
    torch.manual_seed(k); ref = np.asarray(agent.compute_trajectory(ai).poses, dtype=np.float64)
    vl_ref, embeds, mask = seen["vl"], seen["embeds"], seen["mask"]
    n_pad = int((mask == 0).sum()); real = embeds[:, n_pad:]
    torch.manual_seed(k); out = np.asarray(fast.compute_trajectory(ai).poses, dtype=np.float64)
    vl_fast = fast.hidden_state().clone()
    print(f"\n=== case {k}: {real.shape[1]} real tokens, {n_pad} pads")
    win = fast._win[0, :real.shape[1] - fast._n_prefix]
    print("LLM input embeddings (image+tail):", "bit-identical" if torch.equal(win, real[0, fast._n_prefix:]) else rel(win, real[0, fast._n_prefix:]))
    with torch.no_grad():
        alt = lm.model(inputs_embeds=real, output_hidden_states=True, return_dict=True).hidden_states[-1]
    print("reference padded vs stock unpadded (noise floor):", rel(alt[0], vl_ref[0, n_pad:]))
    print("fast vs reference padded                        :", rel(vl_fast[0, n_pad:], vl_ref[0, n_pad:]))
    print("fast vs stock unpadded                          :", rel(vl_fast[0, n_pad:], alt[0]))
    print("  prefix rows only, fast vs reference           :", rel(vl_fast[0, n_pad:n_pad + fast._n_prefix], vl_ref[0, n_pad:n_pad + fast._n_prefix]))
    print("  prefix rows only, stock unpadded vs reference :", rel(alt[0, :fast._n_prefix], vl_ref[0, n_pad:n_pad + fast._n_prefix]))
    if n_pad:
        print("  padding rows, fast vs reference               :", rel(vl_fast[0, :n_pad], vl_ref[0, :n_pad]),
              "| reference pad rows all equal:", bool((vl_ref[0, :n_pad] == vl_ref[0, :1]).all()))
    fast._ctx[0, :vl_ref.shape[1]] = vl_ref[0]
    torch.manual_seed(k)
    for i in range(fast._steps + 1):
        fast._noise[i] = torch.randn((1, 8, 3), device="cuda")
    p = fast._planner().float().cpu().numpy()[0]
    print(f"planner alone (reference hidden states in): max|d| {np.abs(p - ref).max():.2e}")
    with torch.no_grad():
        torch.manual_seed(k)
        vl_alt = torch.cat([vl_ref[:, :n_pad], alt.float()], dim=1)
        t_alt = orig_ga(vl_alt, seen["ai"])["pred_traj"].float().cpu().numpy()[0]
    print(f"trajectory: fast vs reference {np.abs(out[:, :2] - ref[:, :2]).max():.4f} m | "
          f"stock-unpadded LLM + reference planner vs reference {np.abs(t_alt[:, :2] - ref[:, :2]).max():.4f} m")
