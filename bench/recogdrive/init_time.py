import time, torch, common as C
agent = C.build_agent()
import recogdrive_ros.recogdrive_speedups as S
orig = {}
def timed(name):
    fn = getattr(S.FastReCogDrive, name)
    def w(self, *a, **k):
        torch.cuda.synchronize(); t0 = time.time(); r = fn(self, *a, **k); torch.cuda.synchronize()
        print(f"  {name:16s} {time.time() - t0:6.1f} s"); return r
    setattr(S.FastReCogDrive, name, w)
for n in ("_init_prompt", "_init_llm", "_init_kernels", "_init_planner", "_init_step"): timed(n)
OS = S._Stage.__init__
def stage_init(self, fn, capture, warmup=2):
    t0 = time.time(); OS(self, fn, capture, warmup); torch.cuda.synchronize(); print(f"  stage {fn.__name__:12s} {time.time() - t0:6.1f} s")
S._Stage.__init__ = stage_init
t0 = time.time(); fast = S.FastReCogDrive(agent, log=lambda m: None); print(f"total {time.time() - t0:.1f} s")
ai = C.make_input("frames/carla_0.bmp", speed=8.0)
print("verify:", fast.verify(ai))
import numpy as np
from PIL import Image
rgb = np.asarray(Image.open("frames/carla_0.bmp").convert("RGB"))
fast.profile = True
for _ in range(3): fast.plan(ai, image=rgb)
ts = []
for _ in range(15):
    torch.cuda.synchronize(); t0 = time.perf_counter(); fast.plan(ai, image=rgb); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
print(f"plan(image=array): mean {np.mean(ts):.1f} ms sd {np.std(ts):.1f}; stages {dict((k, round(v, 1)) for k, v in fast.timing.items())}")
