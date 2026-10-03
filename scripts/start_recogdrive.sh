#!/usr/bin/env bash
#
# start_recogdrive.sh — bring up the whole Orin side of the ReCogDrive eval.
#
#   ./start_recogdrive.sh            launch (sets everything up first, idempotent)
#   ./start_recogdrive.sh --check    preflight only, don't launch
#   ./start_recogdrive.sh --shell    a shell with the node's environment
#   ./start_recogdrive.sh --stop     stop a running launch
#   ./start_recogdrive.sh --rebuild  force a colcon rebuild, then launch
#
# Which weights (default --2b):
#   --2b        the 2B VLM + its planner checkpoint   (agent.vlm_size=small)
#   --8b        the 8B VLM + its planner checkpoint   (agent.vlm_size=large)
# The paths are VLM_2B / CKPT_2B / VLM_8B / CKPT_8B below, relative to the
# ReCogDrive folder on the SSD; override any of them from the environment.
#
# Which inference path (default: the accelerated one, bit-identical VLM, ~0.63 s/frame):
#   --int8      accelerated + the TensorRT INT8 vision encoder ($VIT_INT8_ENGINE):
#               ~0.58 s/frame, plans shift ~0.13 m on average vs the default
#   --reference the unmodified ReCogDriveAgent.compute_trajectory, ~1.33 s/frame
#
# Controller (default --pid):
#   --pid       Bench2Drive PID in the node, publishing CarlaEgoVehicleControl on
#               /carla/hero/vehicle_control_cmd, like the Orion / MindDrive /
#               SimLingo nodes.  carla_ackermann_control must NOT run on carla-host.
#   --stanley   Autoware Trajectory -> Stanley node -> AckermannDrive.  Needs
#               carla_ackermann_control on carla-host.
#
# Anything of the form key:=value is forwarded to `ros2 launch`, e.g.
#   ./start_recogdrive.sh compressed:=false speed_cap_mps:=0
#
# Counterpart of start_minddrive.sh / start_orion.sh.  What differs:
#   1. No container.  The Python environment is a venv on the SSD
#      (recogdrive_env/setup_venv.sh: Jetson torch 2.10 / triton / flash-attn)
#      on top of the host's ROS 2 Humble.
#   2. /benchmarking/recogdrive is used as a checkout, unmodified.
#   3. Weights live on the SSD over a *manual* exfat-FUSE mount that does not
#      survive a reboot (see start_orion.sh). Preflight checks it.
#   4. The launch runs under recogdrive_env/run_capped.sh: a memory cap and a
#      watchdog, so a runaway process is killed instead of freezing the Orin.
#   5. fast_inference (default true) is recogdrive_speedups.FastReCogDrive: the
#      same computation, ~2x faster, checked against the reference at start-up.
set -eo pipefail

NAME=recogdrive_ros                   # systemd user scope of the running launch
BENCH=/home/USER/vla_benchmarking/benchmarking
SSD_MOUNT=${SSD_MOUNT:-/media/USER/EXTSSD}
SSD_DEV=${SSD_DEV:-/dev/sda1}
RECOG_HOST=${RECOG_HOST:-$SSD_MOUNT/ReCogDrive}
VENV=${VENV:-$RECOG_HOST/venv}
CYCLONE_XML=${CYCLONE_XML:-/home/USER/cyclone_zerotier.xml}
ZT_IFACE=ztXXXXXXXXX
DELL_IP=192.0.2.10
MEM_MAX=${MEM_MAX:-24G}               # the node peaks at ~12 GB
MIN_AVAIL_MB=${MIN_AVAIL_MB:-6000}

VLM_2B=${VLM_2B:-.}
CKPT_2B=${CKPT_2B:-ReCogDrive_Diffusion_Planner_2B_RL.ckpt}
VLM_8B=${VLM_8B:-ReCogDrive-VLM-8B}
CKPT_8B=${CKPT_8B:-}
PID_DIR=${PID_DIR:-$BENCH/Orion/team_code}
VIT_INT8_ENGINE=${VIT_INT8_ENGINE:-$RECOG_HOST/trt/vit_qdq_a08_sel.engine}

DOMAIN=${DOMAIN:-0}

REPO="$BENCH/recogdrive"
SRC="$BENCH/alpamayo-autoware"
WS="$BENCH/recogdrive_env/ws"
LOGS="$BENCH/recogdrive_env/logs"
ROS_DISTRO_HOST=humble

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }

VARIANT=2b
INFER=fast
CONTROL=pid
MODE=launch
REBUILD=0
LAUNCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stop|--check|--shell) MODE="${1#--}" ;;
    --rebuild)              REBUILD=1 ;;
    --2b)                   VARIANT=2b ;;
    --8b)                   VARIANT=8b ;;
    --pid)                  CONTROL=pid ;;
    --stanley)              CONTROL=stanley ;;
    --int8)                 INFER=int8 ;;
    --reference)            INFER=reference ;;
    -h|--help)              sed -n '2,44p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *:=*)                   LAUNCH_ARGS+=("$1") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ "$MODE" = "stop" ]; then
  systemctl --user stop "$NAME.scope" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

case "$VARIANT" in
  2b) VLM_REL="$VLM_2B"; CKPT_REL="$CKPT_2B"; VLM_SIZE=small ;;
  8b) VLM_REL="$VLM_8B"; CKPT_REL="$CKPT_8B"; VLM_SIZE=large ;;
esac

step "Preflight ($VARIANT)"

if mountpoint -q "$SSD_MOUNT"; then
  ok "SSD mounted at $SSD_MOUNT ($(df -h "$SSD_MOUNT" | awk 'NR==2{print $4}') free)"
else
  echo "  ${RED}FAIL${RST} $SSD_MOUNT is not mounted" >&2
  echo "       sudo mount -t exfat-fuse -o allow_other,uid=1000,gid=1000,umask=022 \\" >&2
  echo "         $SSD_DEV $SSD_MOUNT" >&2
  exit 1
fi

[ -d "$RECOG_HOST" ] || die "no ReCogDrive folder at $RECOG_HOST (set RECOG_HOST=...)"
[ -x "$VENV/bin/python" ] || die "no venv at $VENV -- build it: $BENCH/recogdrive_env/run_capped.sh $BENCH/recogdrive_env/setup_venv.sh"
ok "venv $VENV"

VLM_HOST="$(cd "$RECOG_HOST/$VLM_REL" 2>/dev/null && pwd)" || die "no VLM directory $RECOG_HOST/$VLM_REL"
for f in config.json tokenizer_config.json; do
  [ -s "$VLM_HOST/$f" ] || die "$f missing in $VLM_HOST -- the VLM will not load (set VLM_${VARIANT^^}=...)"
done
ls "$VLM_HOST"/*.safetensors >/dev/null 2>&1 || die "$VLM_HOST has no safetensors weights"
ok "VLM $VLM_HOST ($(du -ch "$VLM_HOST"/*.safetensors 2>/dev/null | tail -1 | cut -f1))"

[ -n "$CKPT_REL" ] || die "no planner checkpoint configured for $VARIANT: set CKPT_${VARIANT^^}=<path under $RECOG_HOST>"
CKPT_HOST="$RECOG_HOST/$CKPT_REL"
[ -s "$CKPT_HOST" ] || die "planner checkpoint missing: $CKPT_HOST"
ok "planner checkpoint $CKPT_REL ($(( $(stat -c %s "$CKPT_HOST") / 1024 / 1024 )) MiB)"

case "$INFER" in
  fast)      ok "inference: accelerated (fast_inference, exact PyTorch vision encoder)" ;;
  reference) LAUNCH_ARGS=("fast_inference:=false" "${LAUNCH_ARGS[@]}")
             ok "inference: reference path (no acceleration)" ;;
  int8)      [ "$VARIANT" = 2b ] || die "--int8: the engine is for the 2B VLM's vision encoder"
             [ -s "$VIT_INT8_ENGINE" ] || die "--int8: no engine at $VIT_INT8_ENGINE (build: recogdrive_env/opt/vit_int8.py + trt_vit.py)"
             LAUNCH_ARGS=("vit_engine:=$VIT_INT8_ENGINE" "${LAUNCH_ARGS[@]}")
             ok "inference: accelerated + TensorRT INT8 vision encoder ($(basename "$VIT_INT8_ENGINE"))" ;;
esac
LAUNCH_ARGS=("control_mode:=$CONTROL" "pid_controller_dir:=$PID_DIR" "${LAUNCH_ARGS[@]}")
if [ "$CONTROL" = pid ]; then
  [ -f "$PID_DIR/pid_controller.py" ] || die "--pid: no pid_controller.py in $PID_DIR (set PID_DIR=<team_code of an ORION / MindDrive checkout>)"
  ok "control: Bench2Drive PID in the node -> /carla/hero/vehicle_control_cmd"
else ok "control: Stanley node -> /carla/hero/ackermann_cmd"; fi

[ -f "$REPO/navsim/agents/recogdrive/recogdrive_agent.py" ] \
  || die "ReCogDrive checkout missing at $REPO"
ok "checkout $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"

[ -r "$CYCLONE_XML" ] || die "missing $CYCLONE_XML"
ok "cyclonedds config $CYCLONE_XML"

if ip -4 addr show "$ZT_IFACE" 2>/dev/null | grep -q inet; then
  ok "ZT iface $ZT_IFACE $(ip -4 -o addr show "$ZT_IFACE" | awk '{print $4}')"
  if ping -c 1 -W 3 "$DELL_IP" >/dev/null 2>&1; then ok "carla-host $DELL_IP reachable"
  else warn "carla-host $DELL_IP NOT reachable — check ZeroTier"; fi
else
  warn "ZeroTier iface $ZT_IFACE has no IPv4 — no link to carla-host"
fi

step "Clocks"
GPU_MIN=/sys/class/devfreq/17000000.gpu/min_freq
GPU_MAX=/sys/class/devfreq/17000000.gpu/max_freq
if [ -r "$GPU_MIN" ] && [ "$(cat "$GPU_MIN")" = "$(cat "$GPU_MAX")" ]; then
  ok "clocks already pinned (GPU $(( $(cat "$GPU_MAX") / 1000000 )) MHz)"
elif sudo -n jetson_clocks 2>/dev/null; then
  ok "jetson_clocks applied (GPU $(( $(cat "$GPU_MAX") / 1000000 )) MHz)"
else
  warn "clocks NOT pinned — inference will be substantially slower"
  warn "fix with:  sudo jetson_clocks"
fi

[ "$MODE" = "check" ] && { echo; echo "preflight only; not launching."; exit 0; }

set +u
source /opt/ros/$ROS_DISTRO_HOST/setup.bash
source "$VENV/bin/activate"
set -u 2>/dev/null || true
export ROS_DOMAIN_ID="$DOMAIN" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="file://$CYCLONE_XML"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/recogdrive_inductor"
mkdir -p "$LOGS"

step "ROS workspace ($WS)"
PROBE="from carla_msgs.msg import CarlaRoute; from autoware_planning_msgs.msg import Trajectory; import recogdrive_ros.recogdrive_node, recogdrive_ros.recogdrive_speedups"
ws_env(){ set +u; source "$WS/install/setup.bash" 2>/dev/null || true; export PYTHONPATH="${PYTHONPATH:-}:$BENCH/recogdrive_env/stubs"; }
NEED_BUILD=1
if [ "$REBUILD" = 1 ]; then
  echo "  --rebuild: forcing"; rm -rf "$WS"
elif ( ws_env; cd /tmp; python -c "$PROBE" ) >/dev/null 2>&1; then
  NEWER=$(find "$SRC/src/recogdrive_ros/recogdrive_ros" "$SRC/src/recogdrive_ros/launch" \
            -newer "$WS/install/recogdrive_ros/share/recogdrive_ros/package.xml" -name '*.py' 2>/dev/null | head -1)
  if [ -n "$NEWER" ]; then echo "  node sources changed ($NEWER): rebuilding"
  else NEED_BUILD=0; ok "workspace already built"; fi
fi
if [ "$NEED_BUILD" = 1 ]; then
  echo "  building carla_msgs, autoware_planning_msgs, ackermann_msgs, recogdrive_ros (~1 min)…"
  ( cd "$SRC" && CMAKE_COMMAND=/usr/bin/cmake MAKEFLAGS=-j4 "$VENV/bin/python" -m colcon --log-base "$WS/log" build \
        --base-paths src ackermann_msgs --packages-up-to recogdrive_ros \
        --parallel-workers 2 --build-base "$WS/build" --install-base "$WS/install" ) \
      > "$LOGS/colcon_build.log" 2>&1 \
    || die "colcon build failed -- see recogdrive_env/logs/colcon_build.log"
  ( ws_env; cd /tmp; python -c "$PROBE" ) >/dev/null 2>&1 \
    || die "build finished but imports still fail"
  ok "workspace built"
fi
ws_env

step "ReCogDrive imports"
( cd /tmp && python "$BENCH/recogdrive_env/check_imports.py" "$REPO" ) > "$LOGS/check_imports.log" 2>&1 \
  || die "the agent does not import -- see recogdrive_env/logs/check_imports.log"
ok "navsim agent + torch + transformers + flash_attn + triton import"

if [ "$MODE" = "shell" ]; then
  step "Shell"; cd "$REPO"; exec bash --norc -i
fi

step "CARLA side"
OWN_NODES='recogdrive_node|image_decompress|stanley_controller'
PEERS=$(timeout 20 ros2 node list 2>/dev/null | grep -vE "$OWN_NODES" | grep -c . || true)
if [ "${PEERS:-0}" -gt 0 ]; then
  ok "carla-host nodes visible on domain $DOMAIN"
  timeout 15 ros2 node list 2>/dev/null | sed 's/^/       /'
else
  warn "no CARLA nodes on domain $DOMAIN — is the eval running on carla-host?"
  warn "launching anyway; the node will sit waiting for topics."
fi
if [ "$CONTROL" = pid ]; then
  warn "--pid publishes CarlaEgoVehicleControl directly: carla_ackermann_control must NOT run on carla-host."
else
  warn "--stanley: carla_ackermann_control must run on carla-host or the car never moves."
fi

systemctl --user is-active --quiet "$NAME.scope" 2>/dev/null \
  && die "$NAME is already running (./start_recogdrive.sh --stop)"
step "Launch recogdrive.launch.py ($VARIANT, memory cap $MEM_MAX) ${LAUNCH_ARGS[*]}"
exec env UNIT="$NAME" MEM_MAX="$MEM_MAX" MIN_AVAIL_MB="$MIN_AVAIL_MB" \
  "$BENCH/recogdrive_env/run_capped.sh" \
  ros2 launch recogdrive_ros recogdrive.launch.py \
    recogdrive_repo_path:="$REPO" \
    vlm_path:="$VLM_HOST" \
    checkpoint_path:="$CKPT_HOST" \
    vlm_size:=$VLM_SIZE "${LAUNCH_ARGS[@]}"
