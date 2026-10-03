import time, random, sys
import numpy as np
import common as C
from transformers import AutoTokenizer
from PIL import Image
import navsim.agents.recogdrive.recogdrive_backbone as B
from navsim.agents.recogdrive.utils.conversation import get_conv_template
from navsim.agents.recogdrive.utils.utils import format_number
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image, dynamic_preprocess, build_transform

tok = AutoTokenizer.from_pretrained(C.VLM, trust_remote_code=True, use_fast=False)
tok.padding_side = "left"

def question(hist, cmd):
    history_str = ' '.join([f'   - t-{3-j}: ({format_number(hist[j][0])}, {format_number(hist[j][1])}, {format_number(hist[j][2])})' for j in range(4)])
    prompt = ("<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
              "1. Visual perception from front camera view\n"
              f"2. Historical motion context (last 4 timesteps):{history_str}\n"
              f"3. Active navigation command: [{cmd.upper()}]")
    req = ("\nOutput requirements:\n- Predict 8 future trajectory points\n"
           "- Each point format: (x:float, y:float, heading:float)\n"
           "- Use [PT, ...] to encapsulate the trajectory\n"
           "- Maintain numerical precision to 2 decimal places")
    return prompt + req

def query(q, tiles=9):
    t = get_conv_template("internvl2_5"); t.system_message = B.system_message
    t.append_message(t.roles[0], q); t.append_message(t.roles[1], None)
    return t.get_prompt().replace("<image>", "<img>" + "<IMG_CONTEXT>" * 256 * tiles + "</img>", 1)

random.seed(0)
lens = []; mism = 0
head = None
for n in range(300):
    v = random.choice([0, 0.3, 3, 8, 15, 30, 45])
    yr = random.uniform(-0.6, 0.6)
    hist = []
    for j in range(4):
        dt = -(3 - j) * 0.5
        hist.append((v * dt + random.uniform(-.2, .2), random.uniform(-9, 9) * (j < 3) * (v > 0), yr * dt))
    if n == 0: hist = [(0.0, 0.0, 0.0)] * 4
    if n == 1: hist = [(-123.45, -23.45, -3.14)] * 4
    qy = query(question(hist, random.choice(['turn left', 'go straight', 'turn right'])))
    cut = qy.index("</img>") + len("</img>")
    h, tail = qy[:cut], qy[cut:]
    if head is None:
        head = h; head_ids = tok(head)["input_ids"]
    assert h == head
    if n < 40:
        full = tok(qy)["input_ids"]
        if full != head_ids + tok(tail)["input_ids"]: mism += 1
    t0 = time.perf_counter(); tail_ids = tok(tail)["input_ids"]; dt_ms = (time.perf_counter() - t0) * 1e3
    lens.append(len(tail_ids))
print("head tokens", len(head_ids), "| tail tokens: zero-history", lens[0], "worst-case", lens[1], "random min/mean/max", min(lens[2:]), sum(lens[2:]) / len(lens[2:]), max(lens[2:]))
print("real length range", len(head_ids) + min(lens), len(head_ids) + max(lens), "| split-tokenisation mismatches:", mism, "of 40 | tail tokenise", round(dt_ms, 2), "ms")

path = "frames/carla_0.bmp"
def tm(f, n=10):
    f(); t0 = time.perf_counter()
    for _ in range(n): r = f()
    return (time.perf_counter() - t0) / n * 1e3, r
ms, im = tm(lambda: Image.open(path).convert("RGB")); print(f"open+convert {ms:.1f} ms", im.size)
ms, big = tm(lambda: im.resize((1792, 896))); print(f"resize 1792x896 {ms:.1f} ms")
ms, th = tm(lambda: im.resize((448, 448))); print(f"thumbnail {ms:.1f} ms")
ms, tiles = tm(lambda: dynamic_preprocess(im, image_size=448, use_thumbnail=True, max_num=12)); print(f"dynamic_preprocess {ms:.1f} ms, {len(tiles)} tiles")
tf = build_transform(448)
ms, _ = tm(lambda: [tf(t) for t in tiles]); print(f"transform x9 {ms:.1f} ms")
ms, _ = tm(lambda: load_image(path)); print(f"load_image total {ms:.1f} ms")
from concurrent.futures import ThreadPoolExecutor
ex = ThreadPoolExecutor(2)
def par():
    a = ex.submit(im.resize, (1792, 896)); b = ex.submit(im.resize, (448, 448)); return a.result(), b.result()
ms, _ = tm(par); print(f"two resizes in threads {ms:.1f} ms")
ms, _ = tm(lambda: np.asarray(big)); print(f"np.asarray big {ms:.2f} ms")
