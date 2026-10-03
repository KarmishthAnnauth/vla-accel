#!/usr/bin/env python3
"""Microbenchmarks that decide which optimisation fits which stage."""
import time
import torch
import common as C

def bench(fn, n=8, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

for dt in (torch.bfloat16, torch.float16, torch.float32):
    a = torch.randn(2800, 1536, device="cuda", dtype=dt); w = torch.randn(1536, 8960, device="cuda", dtype=dt)
    ms = bench(lambda: a @ w, n=20)
    print(f"gemm 2800x1536x8960 {str(dt):16s} {ms:6.2f} ms  {2*2800*1536*8960/ms/1e9:6.1f} TFLOP/s")

agent = C.build_agent()
bb = agent.backbone; vlm = bb.model; lm = vlm.language_model; tok = bb.tokenizer
print("vit dtype", next(vlm.vision_model.parameters()).dtype, "llm attn", type(lm.model.layers[0].self_attn).__name__,
      "vit attn flash", getattr(vlm.vision_model.encoder.layers[0].attn, "use_flash_attn", None))

import navsim.agents.recogdrive.recogdrive_backbone as B
from navsim.agents.recogdrive.utils.conversation import get_conv_template
q = "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):   - t-3: (-7.5, 0.0, 0.0)    - t-2: (-5.0, 0.0, 0.0)    - t-1: (-2.5, 0.0, 0.0)    - t-0: (0.0, 0.0, 0.0)\n3. Active navigation command: [GO STRAIGHT]\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
t = get_conv_template("internvl2_5"); t.system_message = B.system_message
t.append_message(t.roles[0], q); t.append_message(t.roles[1], None)
query = t.get_prompt().replace("<image>", "<img>" + "<IMG_CONTEXT>" * 256 * 9 + "</img>", 1)
ids = tok(query)["input_ids"]
img_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
first = ids.index(img_id); last = len(ids) - 1 - ids[::-1].index(img_id)
print(f"real tokens {len(ids)} (pads {2800-len(ids)}): prefix before image {first}, image {last-first+1}, suffix {len(ids)-last-1}")
print("pad id", tok.pad_token_id, repr(tok.pad_token))

emb = lm.get_input_embeddings()
for n in (64, 512, 1024, 2048, 2695, 2800):
    x = torch.randn(1, n, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ms = bench(lambda: lm.model(inputs_embeds=x, output_hidden_states=True), n=4)
    print(f"llm body {n:5d} tokens: {ms:7.1f} ms")

for n in (1, 2, 9):
    x = torch.randn(n, 3, 448, 448, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ms = bench(lambda: vlm.extract_feature(x), n=4)
    print(f"vit {n} tiles: {ms:7.1f} ms")

ah = agent.action_head
import inspect
for n in (64, 2800):
    vl = torch.randn(1, n, 1536, device="cuda")
    from transformers.feature_extraction_utils import BatchFeature
    ai = BatchFeature({"state": torch.randn(1, 20, device="cuda"), "his_traj": torch.randn(1, 12, device="cuda"), "status_feature": torch.randn(1, 8, device="cuda")})
    with torch.no_grad():
        ms = bench(lambda: ah.get_action(vl, ai), n=4)
    print(f"planner get_action, {n} context tokens: {ms:7.1f} ms")
