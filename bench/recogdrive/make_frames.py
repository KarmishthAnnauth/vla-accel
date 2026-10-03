#!/usr/bin/env python3
"""Make the three test frames the benches use: the CARLA renders in
assets/frames (1024x359), centre-cropped to 16:9 and resized to NAVSIM's
1920x1080, written as frames/carla_{0,1,2}.bmp next to this script.

  python3 make_frames.py [directory with frame_0.png ...]
"""
import os
import sys

from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "../../assets/frames")
os.makedirs(os.path.join(here, "frames"), exist_ok=True)
for i in range(3):
    im = Image.open(os.path.join(src, f"frame_{i}.png")).convert("RGB")
    w, h = im.size
    cw = int(h * 16 / 9)
    im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h)).resize((1920, 1080), Image.BICUBIC)
    im.save(os.path.join(here, f"frames/carla_{i}.bmp"))
    print("wrote", f"frames/carla_{i}.bmp", im.size)
