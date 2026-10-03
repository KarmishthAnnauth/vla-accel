import sys, time, numpy as np, torch, common as C
from PIL import Image
agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
ai = C.make_input("frames/carla_0.bmp", speed=8.0)
rgb = np.asarray(Image.open("frames/carla_0.bmp").convert("RGB"))
for eng in [""] + sys.argv[1:]:
    fast = FastReCogDrive(agent, log=lambda m: None, profile=True, vit_engine=eng)
    for _ in range(3): fast.plan(ai, image=rgb)
    ts = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fast.plan(ai, image=rgb); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    v = fast.verify(ai)
    print(f"{C.os.path.basename(eng) or 'PyTorch ViT (exact)':28s} plan {np.mean(ts):6.1f} ms  stages {dict((k, round(x)) for k, x in fast.timing.items())}  verify: hidden max|d| {v['hidden_max_abs']:.3g}, traj max|d| {v['trajectory_max_abs']:.3f}")
    del fast; torch.cuda.empty_cache()
