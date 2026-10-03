import time, torch, numpy as np, common as C
agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
fast = FastReCogDrive(agent, log=print)
ai = C.make_input("frames/carla_0.bmp", speed=8.0)
for _ in range(3): fast.compute_trajectory(ai)
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
print(f"planner graph replay alone      {t(fast._planner):6.2f} ms")
print(f"llm graph replay alone          {t(fast._llm):6.2f} ms")
print(f"vit graph replay alone          {t(fast._vit):6.2f} ms")
hid = fast._llm.out
print(f"context copy (bf16->fp32)       {t(lambda: fast._ctx[0, 304:304+2500].copy_(hid[0, :2500])):6.2f} ms")
def noise():
    for i in range(6): fast._noise[i] = torch.randn((1, 8, 3), device='cuda')
print(f"6x randn                        {t(noise):6.2f} ms")
print(f"out .float().cpu()              {t(lambda: fast._planner.out.float().cpu()):6.2f} ms")
from PIL import Image
feats = lambda: fast._builder.compute_features(ai)
print(f"feature builder                 {t(feats):6.2f} ms")
f = feats()
print(f"question+query+tail (cached)    {t(lambda: fast._tail_ids(fast._query(fast._question(f['history_trajectory'], f['high_command_one_hot'])))):6.2f} ms")
op = lambda: Image.open('frames/carla_0.bmp').convert('RGB')
print(f"PIL open+convert                {t(op):6.2f} ms")
im = op()
print(f"tiles (threaded resizes)        {t(lambda: fast._load_tiles(im)):6.2f} ms")
tiles = fast._load_tiles(im)
print(f"to GPU + normalise              {t(lambda: fast._set_pixels(tiles)):6.2f} ms")
for strips in (1, 2, 4, 6, 8):
    import recogdrive_ros.recogdrive_speedups as S
    S.RESIZE_STRIPS = strips
    print(f"   resize strips={strips}: {t(lambda: fast._load_tiles(im)):6.2f} ms")
