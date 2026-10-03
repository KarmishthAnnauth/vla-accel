"""The no-padding branch: a prompt longer than the reference's 2800-token
max_length (only reachable with 3-digit history values)."""
import numpy as np, torch, common as C
agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
fast = FastReCogDrive(agent, log=lambda m: None)
for vals in ([-123.45, -23.45, -3.14], [-250.11, 112.72, 2.97], [-12.5, 3.25, 0.4]):
    ai = C.make_input("frames/carla_1.bmp", speed=10.0)
    for e in ai.ego_statuses[:4]:
        e.ego_pose = np.array(vals, dtype=np.float64)
    r = fast.verify(ai, seed=11)
    print(vals, "->", fast._ctx_layout[1], "real tokens,", fast._ctx_layout[0], "pads:", r)
