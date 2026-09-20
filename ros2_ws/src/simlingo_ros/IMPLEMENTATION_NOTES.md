# SimLingo ROS 2 Integration — Implementation Notes

## Overview

This package (`simlingo_ros`) ports the SimLingo Vision-Language-Action model into the
ROS 2 benchmarking framework previously used for Alpamayo. Nothing in `alpamayo_ros` or
`ros-bridge` was modified.

**Control**: SimLingo drives with its own PID controller (`agent_simlingo.py::control_pid`),
run inside `simlingo_node`, publishing `carla_msgs/CarlaEgoVehicleControl` directly. The
Stanley controller and `carla_ackermann_control` are not in the loop. SimLingo's waypoint
head was trained and tuned against that specific controller, and driving its waypoints with
a differently-tuned Stanley controller produced unstable steering and route crashes.
`control_mode:=trajectory` still selects the old Stanley path — see *Control modes* below.

The companion file on the CARLA machine is:
`carla-host/my_ros2_agent_simlingo.py`

---

## System Architecture

```
CARLA machine (carla-host)                     Inference machine (Orin)
────────────────────────────────────         ─────────────────────────────────────────────
my_ros2_agent_simlingo.py                    simlingo_node
 └─ carla_ros_bridge publishes:               ├─ subscribes: /carla/hero/rgb_0/image
     /carla/hero/rgb_0/image       ────────►  │              /carla/hero/speed
     /carla/hero/speed             ────────►  │              /carla/hero/odometry
     /carla/hero/odometry          ────────►  │              /carla/hero/global_plan
     /carla/hero/global_plan       ────────►  │
                                              ├─ SimLingo control_pid (simlingo_pid.py)
                                              │
                                              └─ publishes:  /carla/hero/vehicle_control_cmd
                                                             /simlingo/predicted_trajectory (debug)
                                                             /simlingo/language_output      (debug)

 leaderboard agent reads:        ◄────────────────────────────────────────────────────────
 /carla/hero/vehicle_control_cmd
```

The trajectory is still published in PID mode so the prediction stays visible in RViz, but
nothing consumes it for control.

### Control modes

| `control_mode` | Control path | Extra nodes needed |
|---|---|---|
| `pid` (default) | `simlingo_node` → `CarlaEgoVehicleControl` on `/carla/hero/vehicle_control_cmd` | none |
| `trajectory` | `simlingo_node` → Trajectory → `stanley_controller_node` → `/carla/hero/ackermann_cmd` | `carla_ackermann_control` |

**In `pid` mode, `carla_ackermann_control` must not be running.** It publishes the same
`/carla/hero/vehicle_control_cmd` topic and the two publishers would fight over the
vehicle. The CARLA-side leaderboard agent must be in direct-control mode (`CONTROL_MODE`),
not `ackermann`.

Both machines share the same ROS 2 DDS domain. No bridge node is needed on the Orin side
because `carla_ros_bridge` (embedded in `my_ros2_agent_simlingo.py` via the leaderboard
framework) already publishes sensor data as standard ROS 2 messages.

---

## File Descriptions

### `my_ros2_agent_simlingo.py`  (carla-host repository)

A leaderboard agent that registers exactly the sensors SimLingo was trained on and nothing
more. The key difference from the original `my_ros2_agent.py` is:

- **Single front camera** (`rgb_0`) at position `(-1.5, 0.0, 2.0)` in vehicle frame, FOV 110°.
  The original agent had three cameras (front-left, front, front-right) which Alpamayo needed
  but SimLingo does not. The position and FOV are taken directly from
  `simlingo/team_code/config_simlingo.py` (`camera_pos_0`, `camera_fov_0`) so the physical
  camera placement at inference matches the placement during training.
- **IMU, GPS, speedometer, odometry** are kept because the ROS 2 topics they generate
  (`/carla/hero/speed`, `/carla/hero/odometry`, `/carla/hero/global_plan`) are consumed by
  `simlingo_node` for target-point computation and speed input.

Control mode is `ackermann` by default (env var `CONTROL_MODE`). The leaderboard agent does
not start `carla_ackermann_control` itself; the node is expected to be running externally
(same assumption as the original agent).

---

### `simlingo_ros/simlingo_node.py`

The main inference node. Structured after `alpamayo_ros/alpamayo_node.py` but adapted for
SimLingo's very different input/output interface.

#### Model loading

SimLingo is not a standard HuggingFace `from_pretrained` model. It is a PyTorch Lightning
checkpoint of a Hydra-configured `DrivingModel` class that wraps InternVL2-1B. Loading
therefore follows the same two-step process as `agent_simlingo.py`:

1. Read `.hydra/config.yaml` — found automatically three directory levels above the `.ckpt`
   file, matching the training output layout (`<session>/<version>/<epoch>/weights.ckpt`).
2. `hydra.utils.instantiate(cfg.model, ...)` to construct the full model graph, then
   `model.load_state_dict(torch.load(ckpt_path))` to load weights.

The base VLM (InternVL2-1B) is loaded via `AutoProcessor.from_pretrained` pointing at the
local cache directory (`{simlingo_path}/pretrained/InternVL2-1B`) which `agent_simlingo.py`
already populates on the CARLA machine. This avoids any network access at runtime.

Model loading and all subsequent inference runs happen inside a single-worker
`ThreadPoolExecutor` so that CUDA handles (cuBLAS, cuDNN) are always created and used in
the same OS thread — a requirement on Jetson iGPU.

#### Image preprocessing

SimLingo was trained on images that passed through a specific preprocessing pipeline.
Replicating it exactly at inference is critical for distribution match:

1. **JPEG encode/decode** (`cv2.imencode` / `cv2.imdecode`): the training dataset was saved
   as JPEG files, introducing compression artefacts. Applying the same codec at inference
   keeps the input distribution consistent.
2. **BGR → RGB**: OpenCV decodes to BGR; the model expects RGB.
3. **Bottom crop** (remove bottom `4.8/16` of height): crops out the car hood, which was
   masked during data collection. The exact fraction comes from `agent_simlingo.py` line 375.
4. **InternVL2 dynamic tiling** (`dynamic_preprocess`, `build_transform`): the vision
   encoder processes images as one or two 448×448 tiles (plus an optional thumbnail). Using
   `max_num=2` and `use_thumbnail=cfg.model.vision_model.use_global_img` matches the
   training data-module configuration read from the hydra config.

The raw image from `carla_ros_bridge` arrives as `sensor_msgs/Image` with RGBA encoding.
The first three channels (RGB) are extracted, then the pipeline above is applied.

#### Target point computation

SimLingo requires an ego-relative 2-D target waypoint `[x, y]` (forward, left) as a
navigation input. During training this comes from the leaderboard's internal `RoutePlanner`
which queries CARLA's map. In the ROS node the equivalent information is available from the
`/carla/hero/global_plan` topic (`carla_msgs/CarlaRoute`), which the leaderboard agent
publishes once at the start of each route with TRANSIENT_LOCAL durability.

The target point is computed each inference cycle as follows:

1. Extract the vehicle's current position and yaw from `/carla/hero/odometry`.
2. Walk the cached route waypoints (world frame) and find the first one that is between
   7.5 m and 50 m ahead — the same distance window used by `RoutePlanner` in
   `nav_planner.py`.
3. Transform to ego frame by applying the inverse of the vehicle's yaw rotation.

Two successive candidates are extracted (current and next target point) to populate both
`<TARGET_POINT>` tokens in the prompt, matching the training convention.

No UKF or GPS conversion is needed here because `/carla/hero/odometry` (published by the
pseudo-sensor) already provides ground-truth world-frame pose, and the route waypoints are
also in world frame. The UKF in `agent_simlingo.py` exists only to filter noisy GPS for
the CARLA leaderboard's coordinate-system quirks, which do not apply in the ROS path.

#### Language prompt and tokenisation

SimLingo uses a standard InternLM2-chat conversation template with the prompt:

```
Current speed: {speed} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.
```

The `<TARGET_POINT>` special tokens are replaced inside the model's `WaypointInputAdaptor`
with learned embeddings conditioned on the coordinate values. The token-to-value mapping is
passed through the `LanguageLabel.placeholder_values` field.

Tokenisation replicates `agent_simlingo.py` exactly:

1. Load `conversation.py` from the cached InternVL2-1B model files and call
   `get_conv_template("internlm2-chat")` to get the prompt wrapper.
2. Replace the `<image>` placeholder with `<img><IMG_CONTEXT> × (num_image_token × num_patches)</img>`.
   `num_image_token` (256 for InternVL2-1B) is computed from the model config at startup.
3. Tokenise with `padding=True`, `add_special_tokens=False`.
4. Wrap in a `LanguageLabel` dataclass with `phrase_ids`, `phrase_valid`, `phrase_mask`,
   and `placeholder_values`.

#### Camera calibration

`DrivingInput` requires intrinsic (K) and extrinsic (4×4) matrices.

- **Intrinsics**: computed analytically from `(W=448, H=448, FOV=110°)` — the tile
  resolution after InternVL2 preprocessing, which is what `agent_simlingo.py` passes
  (line 659: `get_camera_intrinsics(W, H, 110)` where W, H are the tile dimensions).
- **Extrinsics**: hardcoded as `[-1.5, 0.0, 2.0]` translation with identity rotation,
  taken directly from `simlingo_utils.get_camera_extrinsics()`.

#### Model outputs and trajectory conversion

`DrivingModel.forward()` returns:

- `pred_route` — `[1, 20, 2]` ego-relative spatial waypoints (x=forward, y=left), at 0.25 s
  intervals (carla_fps=20, wp_dilation=1, data_save_freq=5 → 4 Hz).
- `pred_speed_wps` — `[1, 10, 2]` ego-relative speed waypoints at the same cadence.
- `language` — generated text output.

The trajectory is converted to `autoware_planning_msgs/Trajectory` in the `base_link` frame.
Longitudinal velocity is derived from the speed waypoints using exactly the same formula as
`control_pid` in `agent_simlingo.py`:

```python
one_second  = carla_fps // (wp_dilation * data_save_freq)  # = 20//(1*5) = 4
half_second = one_second // 2                              # = 2
desired_speed = ||speed_wps[half_second-2] - speed_wps[one_second-2]|| * 2.0
             # = ||speed_wps[0] - speed_wps[2]|| * 2.0
```

This measures the distance the model predicts the vehicle will travel in 0.5 s and doubles
it to get m/s. The resulting scalar is applied uniformly to all 20 trajectory points.

This matches the base implementation's intent: SimLingo's longitudinal control was designed
around a single scalar speed fed into a PID controller (throttle/brake), not a varying speed
profile per waypoint. Applying the same scalar across all trajectory points faithfully
replicates that behaviour within the Autoware trajectory format that the Stanley controller
consumes.

---

### `simlingo_ros/simlingo_pid.py`

SimLingo's own controllers, made importable without CARLA. A verbatim port of:

* `team_code/transfuser_utils.py::PIDController` — longitudinal (throttle/brake)
* `team_code/nav_planner.py::LateralPIDController` — lateral (steering)
* `team_code/agent_simlingo.py::control_pid` + `interpolate_waypoints`

The module imports the upstream classes from `team_code` when they resolve and falls back
to bundled copies otherwise: both upstream modules `import carla` at module scope, which is
not available in the inference container on the Orin. The bundled copies were verified
bit-identical to upstream (zero difference in throttle and steer over randomised inputs),
and the whole `step()` matches `agent_simlingo.control_pid` to within float noise (4e-7,
from the `round(steer, 3)`) over 200 randomised trajectories. Gains come from
`config_simlingo.py::GlobalConfig`; if you change them upstream, change them here.

Two things had to be re-expressed because this node does not run at the simulator's 20 Hz:

* **Stuck/creep recovery** is counted in seconds, not frames — `stuck_threshold` 800 frames
  → 40 s, `creep_duration` 15 frames → 0.75 s.
* **The speed PID's integral window** (`speed_n = 20` samples) covers 1 s at 20 Hz but would
  cover 5 s when stepped at the ~4 Hz inference rate, winding the integral up long after the
  car has reached the target speed. `pid_window_rate_compensation` (default on) rescales the
  window to hold the same ~1 s of history — `n = 4` at 4 Hz. Set it to `false` for the
  literal upstream sample count.

The controller is stepped **once per model prediction**, not once per publish. Its deques
hold the tuned error history, so re-stepping them against a route the model has already
superseded would wind up the integral term — this is the cadence `control_pid` runs at in
`agent_simlingo.py`, which also predicts and controls in the same step.

A watchdog (`control_timeout_sec`, default 1 s) publishes a full brake if inference stops
producing controls. CARLA keeps applying the last `CarlaEgoVehicleControl` it received, so a
hung model would otherwise leave the car rolling on its last throttle value.

---

### `simlingo_ros/stanley_controller_node.py`

Only used in `control_mode:=trajectory`. A direct copy of
`alpamayo_ros/stanley_controller_node.py` with two changes:

1. Default `trajectory_topic` changed from `/alpamayo/predicted_trajectory` to
   `/simlingo/predicted_trajectory`.
2. Added a `control_output_topic` parameter (default `/carla/hero/ackermann_cmd`) so the
   output topic can be configured without modifying the node. This matches what
   `carla_ackermann_control` subscribes to.

The lateral Stanley algorithm, coordinate frame handling, and all tuning parameters are
unchanged.

---

### `launch/simlingo.launch.py`

Launches `simlingo_node` immediately. `stanley_controller_node` is launched only when
`control_mode:=trajectory`, after a 90-second delay. The delay exists because `simlingo_node` blocks during `_setup_model()` (model
instantiation + CUDA warm-up) before the inference timer starts. On Jetson AGX Orin the
first run takes 60–90 seconds; subsequent runs use the local model cache and are faster.
Stanley is lightweight and only needs to start after at least one trajectory has been
published.

Two launch arguments are required:

| Argument | Description |
|---|---|
| `checkpoint_path` | Absolute path to the `.ckpt` file |
| `simlingo_path` | Absolute path to the simlingo repository root |
| `control_mode` | `pid` (default) or `trajectory`. Optional. |

Example:
```bash
ros2 launch simlingo_ros simlingo.launch.py \
  checkpoint_path:=/data/models/simlingo/session_1/version_0/epoch=49/weights.ckpt \
  simlingo_path:=/workspace/simlingo
```

---

## What Was Deliberately Not Changed

- **`alpamayo_ros/`** — untouched. Both packages coexist in the same workspace and can be
  built and launched independently.
- **`ros-bridge/`** — untouched. `carla_ackermann_control` from this package is reused as-is.
- **Stanley controller algorithm** — unchanged, still available via `control_mode:=trajectory`.
  It is no longer the default: its tuning is not SimLingo's, and driving SimLingo's waypoints
  with it produced unstable steering.
- **Control loop frequency** — in `trajectory` mode Stanley still interpolates at 20 Hz. In
  `pid` mode control is emitted once per prediction (~4 Hz), matching the cadence at which
  `agent_simlingo.py` predicts and controls.

---

## Known Limitations and Future Work

- **Single-camera only**: SimLingo was trained on one front camera. Multi-camera support
  would require changes to the model training configuration.
- **No UKF on the Orin side**: target-point computation relies on ground-truth odometry
  from CARLA. For a real vehicle deployment, GPS + IMU fusion would be needed.
- **Single scalar speed across trajectory**: only affects `control_mode:=trajectory`, where
  the base design's single scalar `desired_speed` per inference step is stamped on all 20
  trajectory points. In `pid` mode the controller reads the raw `pred_speed_wps` tensor, so
  nothing is lost to the trajectory encoding or its `min_trajectory_speed_mps` clamp — note
  that clamp (default 0.5 m/s) would otherwise mask the `brake_speed` (0.4 m/s) threshold.
- **Control rate**: `pid` mode emits control at the inference rate (~4 Hz) rather than the
  simulator's 20 Hz. Between predictions CARLA holds the last command. This matches the
  reference agent's one-control-per-prediction structure, but the reference predicts every
  simulator frame, so the vehicle reacts to new information 5× less often here.
- **`conversation.py` dependency**: the InternVL2 prompt template module is loaded
  dynamically from the model cache. If the cache is missing it falls back to
  `huggingface_hub.hf_hub_download`, which requires internet access. Bundling
  `conversation.py` into the package would remove this dependency entirely.
