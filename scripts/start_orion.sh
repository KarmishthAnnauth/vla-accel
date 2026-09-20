#!/usr/bin/env bash
set -eo pipefail

IMAGE=karmishthannauth/orion_env_ros:v01
NAME=orion_ros
BENCH=/home/USER/vla_benchmarking/benchmarking
SSD_MOUNT=/media/USER/EXTSSD
MODELS_HOST="$SSD_MOUNT/models"
CYCLONE_XML=/home/USER/cyclone_zerotier.xml
ZT_IFACE=ztXXXXXXXXX
DELL_IP=192.0.2.10

DOMAIN=${DOMAIN:-0}

ORION_REPO=/root/Orion
CKPT_CTR=/models/Orion/Orion.pth
CKPT_HOST="$MODELS_HOST/Orion/Orion.pth"
CKPT_SIZE=38459589956
QFORMER_HOST="$MODELS_HOST/Orion/pretrain_qformer"
WS=/opt/orion_ws

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }

in_ctr(){ docker exec "$NAME" bash -c "$1"; }
ENVSETUP='source /opt/ros/humble/setup.bash;
          source '"$WS"'/install/setup.bash 2>/dev/null || true;
          export TORCHINDUCTOR_CACHE_DIR=/benchmarking/.torchinductor_cache TORCHINDUCTOR_FX_GRAPH_CACHE=1;'

LAUNCH_FILE=orion_withpid.launch.py
MODE=launch
REBUILD=0
LAUNCH_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --stop|--check|--shell) MODE="${1#--}" ;;
    --rebuild)              REBUILD=1 ;;
    --pid)                  LAUNCH_FILE=orion_withpid.launch.py ;;
    --stanley)              LAUNCH_FILE=orion.launch.py ;;
    --lite)                 LAUNCH_FILE=orion_lite.launch.py ;;
    -h|--help)              sed -n '2,40p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *:=*)                   LAUNCH_ARGS+=("$1") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ "$MODE" = "stop" ]; then
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

step "Preflight"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (docker pull it)"
ok "image $IMAGE"

if mountpoint -q "$SSD_MOUNT"; then
  ok "EXTSSD mounted ($(df -h "$SSD_MOUNT" | awk 'NR==2{print $4}') free)"
else
  echo "  ${RED}FAIL${RST} $SSD_MOUNT is not mounted" >&2
  echo "       sudo mount -t exfat-fuse -o allow_other,uid=1000,gid=1000,umask=022 \\" >&2
  echo "         /dev/sda1 $SSD_MOUNT" >&2
  exit 1
fi

[ -s "$CKPT_HOST" ] || die "checkpoint missing: $CKPT_HOST"
sz=$(stat -c %s "$CKPT_HOST")
[ "$sz" -eq "$CKPT_SIZE" ] || warn "checkpoint is $sz bytes, expected $CKPT_SIZE (truncated copy?)"
[ "$sz" -eq "$CKPT_SIZE" ] && ok "checkpoint $(( sz/1024/1024/1024 )) GiB"

for f in config.json pytorch_model-00001-of-00002.bin \
         pytorch_model-00002-of-00002.bin tokenizer.model; do
  [ -s "$QFORMER_HOST/$f" ] || die "pretrain_qformer/$f missing — LLM will not load"
done
ok "pretrain_qformer (LLM + tokenizer)"

[ -r "$CYCLONE_XML" ] || die "missing $CYCLONE_XML"
ok "cyclonedds config"

ip -4 addr show "$ZT_IFACE" 2>/dev/null | grep -q inet || die "ZeroTier iface $ZT_IFACE has no IPv4"
ok "ZT iface $ZT_IFACE $(ip -4 -o addr show "$ZT_IFACE" | awk '{print $4}')"

if ping -c 1 -W 3 "$DELL_IP" >/dev/null 2>&1; then ok "carla-host $DELL_IP reachable"
else warn "carla-host $DELL_IP NOT reachable — check ZeroTier"; fi

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

step "Container"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  ok "$NAME already running"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --runtime nvidia --network host --shm-size=8g \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e ROS_DOMAIN_ID="$DOMAIN" \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///cyclone.xml \
    -v "$CYCLONE_XML":/cyclone.xml:ro \
    -v "$BENCH/":/benchmarking \
    -v "$MODELS_HOST":/models:ro \
    --name "$NAME" --entrypoint bash "$IMAGE" -c 'sleep infinity' >/dev/null
  sleep 2
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || die "container failed to start"
  ok "started $NAME"
fi

step "ORION repo (container)"
in_ctr "test -d $ORION_REPO" >/dev/null 2>&1 || die "$ORION_REPO missing from image"
in_ctr "cd $ORION_REPO && python3 -c 'import mmcv._ext'" >/dev/null 2>&1 \
  || die "$ORION_REPO has no compiled mmcv._ext — wrong image?"
ok "mmcv._ext compiled"

in_ctr "test -f $CKPT_CTR && test -d /models/Orion/pretrain_qformer" >/dev/null 2>&1 \
  || die "weights not visible in container at /models/Orion (SSD mount lost?)"
ok "weights visible at /models/Orion"

step "ROS workspace ($WS)"
NEED_BUILD=1
if [ "$REBUILD" = 1 ]; then
  echo "  --rebuild: forcing"
  in_ctr "rm -rf $WS" >/dev/null 2>&1 || true
elif in_ctr "$ENVSETUP python3 -c 'from carla_msgs.msg import CarlaRoute; import orion_ros.orion_node'" >/dev/null 2>&1; then
  NEWER=$(in_ctr "find /benchmarking/alpamayo-autoware/src/orion_ros/orion_ros /benchmarking/alpamayo-autoware/src/orion_ros/launch \
                  -newer $WS/install/orion_ros/share/orion_ros/package.xml -name '*.py' 2>/dev/null | head -1")
  if [ -n "$NEWER" ]; then echo "  node sources changed ($NEWER): rebuilding"
  else NEED_BUILD=0; ok "workspace already built"; fi
fi
if [ "$NEED_BUILD" = 1 ]; then
  echo "  building carla_msgs, autoware_*, ackermann_msgs, orion_ros (~30s)…"
  in_ctr "source /opt/ros/humble/setup.bash;
          cd /benchmarking/alpamayo-autoware &&
          colcon build --base-paths src /benchmarking/alpamayo-autoware/ackermann_msgs \
            --packages-up-to orion_ros \
            --build-base $WS/build --install-base $WS/install" >/dev/null 2>&1 \
    || die "colcon build failed — rerun with: docker exec $NAME bash -c 'cd /benchmarking/alpamayo-autoware && colcon build …'"
  in_ctr "$ENVSETUP python3 -c 'from carla_msgs.msg import CarlaRoute; import orion_ros.orion_node'" >/dev/null 2>&1 \
    || die "build finished but imports still fail"
  ok "workspace built"
fi

if [ "$MODE" = "shell" ]; then
  step "Shell"; exec docker exec -it "$NAME" bash -c "$ENVSETUP cd $ORION_REPO; exec bash"
fi

step "CARLA side"
OWN_NODES='orion_node|orion_withpid_node|orion_lite_node|stanley_controller|image_decompress'
PEERS=$(in_ctr "$ENVSETUP timeout 20 ros2 node list 2>/dev/null" | grep -vE "$OWN_NODES" | grep -c . || true)
if [ "${PEERS:-0}" -gt 0 ]; then
  ok "carla-host nodes visible on domain $DOMAIN"
  in_ctr "$ENVSETUP timeout 15 ros2 node list 2>/dev/null" | sed 's/^/       /'
else
  warn "no CARLA nodes on domain $DOMAIN — is the eval running on carla-host?"
  warn "launching anyway; the node will sit waiting for topics."
  if [ "$LAUNCH_FILE" = "orion.launch.py" ]; then
    warn "--stanley: carla_ackermann_control must run on carla-host or the car never moves."
  else
    warn "--pid publishes CarlaEgoVehicleControl directly; no carla_ackermann_control needed."
  fi
  warn "carla-host must publish all 6 ORION cameras, or the node waits forever."
fi

step "Launch $LAUNCH_FILE (model load ~200 s: 52 GB of weights off the SSD)"
exec docker exec -i "$NAME" bash -c "$ENVSETUP
  exec ros2 launch orion_ros $LAUNCH_FILE ${LAUNCH_ARGS[*]}"
