# SimLingo checkout patch

SimLingo is the one model here whose speedups could not be expressed purely as
post-build transforms — the fast path replaces a method on `DrivingModel` — so
these files drop into a SimLingo checkout:

```
simlingo_training/models/fast_inference.py   ->  <simlingo>/simlingo_training/models/
team_code/inference_client_simlingo.py       ->  <simlingo>/team_code/
0001-nav-planner-no-carla.patch              ->  patch -p0 < ... from <simlingo>/
```

- **`fast_inference.py`** (new file) — `optimize_for_inference(model)` applies
  the LoRA merge, the FlashAttention2 → SDPA swap, the KV cache, the CUDA-graph
  decode step and the driving-head-as-continuation. `verify()` re-checks the
  numerics against the unoptimised model on demand. The ROS node imports it
  lazily and falls back if it raises, so the checkout is still usable without
  it (`fast_inference:=false`).
- **`0001-nav-planner-no-carla.patch`** — guards the `GlobalRoutePlanner`
  import in `team_code/nav_planner.py`. Upstream imports the CARLA agents
  package at module level; there is no aarch64 CARLA wheel, so on the Jetson
  this takes the whole module down. Four lines, and it is what lets the node
  run without the simulator. Shipped as a patch rather than a modified copy so
  that no SimLingo source is redistributed here.
- **`inference_client_simlingo.py`** (new file) — the client side of the
  split-host setup, kept for reference.

Nothing else in the SimLingo checkout is modified.
