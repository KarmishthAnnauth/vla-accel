#!/usr/bin/env bash
set -eo pipefail

IMAGE=karmishthannauth/simlingo:humble-cyclonedd
NAME=sim_ros
BENCH=/home/USER/vla_benchmarking/benchmarking
SIMLINGO_HOST="$BENCH/simlingo"
MODELS_HOST=/home/USER/models/simlingo_hf
CYCLONE_XML=/home/USER/cyclone_zerotier.xml
ZT_IFACE=ztXXXXXXXXX
DELL_IP=192.0.2.10

DOMAIN=0

CKPT_CTR=/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
CKPT_HOST="$MODELS_HOST/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
WS=/opt/sim_ws

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }

in_ctr(){ docker exec "$NAME" bash -c "$1"; }
ENVSETUP='source /opt/ros/humble/setup.bash;
          source /opt/ros_ws/install/setup.bash;
          source '"$WS"'/install/setup.bash 2>/dev/null || true;
          export PYTHONPATH="/benchmarking/simlingo:/benchmarking/simlingo/team_code:$PYTHONPATH";'

LAUNCH_ARGS="$*"

case "${1:-}" in
  --stop)
    docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
    exit 0 ;;
esac

step "Preflight"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (docker pull it)"
ok "image $IMAGE"

[ -s "$CKPT_HOST" ] || die "checkpoint missing: $CKPT_HOST"
sz=$(stat -c %s "$CKPT_HOST")
[ "$sz" -eq 2569679322 ] || warn "checkpoint is $sz bytes, expected 2569679322 (truncated download?)"
[ "$sz" -eq 2569679322 ] && ok "checkpoint $(( sz/1024/1024 )) MB"

[ -s "$MODELS_HOST/simlingo/.hydra/config.yaml" ] || die "hydra config missing next to checkpoint"
ok "hydra config"

[ -r "$CYCLONE_XML" ] || die "missing $CYCLONE_XML"
ok "cyclonedds config"

VLM="$SIMLINGO_HOST/pretrained/InternVL2-1B/models--OpenGVLab--InternVL2-1B"
[ -d "$VLM" ] && [ ! -L "$VLM" ] || die "InternVL2 cache missing or is a symlink (must be a real dir): $VLM"
ok "InternVL2 cache"

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
  warn "clocks NOT pinned — inference will be ~3x slower (3.0 s vs 0.9 s per frame)"
  warn "fix with:  sudo jetson_clocks"
fi

[ "${1:-}" = "--check" ] && { echo; echo "preflight only; not launching."; exit 0; }

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
    -v "$MODELS_HOST":/models/simlingo:ro \
    --name "$NAME" --entrypoint bash "$IMAGE" -c 'sleep infinity' >/dev/null
  sleep 2
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || die "container failed to start"
  ok "started $NAME"
fi

step "Python deps"
if in_ctr 'python3 -c "import pytorch_lightning"' >/dev/null 2>&1; then
  ok "pytorch_lightning present"
else
  echo "  installing pytorch_lightning (--no-deps: the image's torch build must not be replaced)"
  in_ctr 'pip install --no-deps pytorch_lightning torchmetrics lightning_utilities' >/dev/null 2>&1 \
    || die "pip install failed"
  in_ctr 'python3 -c "import pytorch_lightning"' >/dev/null 2>&1 || die "pytorch_lightning still missing"
  ok "pytorch_lightning installed"
fi

step "ROS workspace ($WS)"
NEED_BUILD=1
if [ "${1:-}" = "--rebuild" ]; then
  echo "  --rebuild: forcing"
  in_ctr "rm -rf $WS" >/dev/null 2>&1 || true
elif in_ctr "$ENVSETUP python3 -c 'from carla_msgs.msg import CarlaRoute; import simlingo_ros.simlingo_node'" >/dev/null 2>&1; then
  NEED_BUILD=0; ok "workspace already built"
fi
if [ "$NEED_BUILD" = 1 ]; then
  echo "  building carla_msgs, autoware_*, ackermann_msgs, simlingo_ros (~40s)…"
  in_ctr "source /opt/ros/humble/setup.bash; source /opt/ros_ws/install/setup.bash;
          cd /benchmarking/alpamayo-autoware &&
          colcon build --base-paths src /benchmarking/alpamayo-autoware/ackermann_msgs \
            --packages-up-to simlingo_ros \
            --build-base $WS/build --install-base $WS/install" >/dev/null 2>&1 \
    || die "colcon build failed — rerun with: docker exec $NAME bash -c 'cd /benchmarking/alpamayo-autoware && colcon build …'"
  in_ctr "$ENVSETUP python3 -c 'from carla_msgs.msg import CarlaRoute; import simlingo_ros.simlingo_node'" >/dev/null 2>&1 \
    || die "build finished but imports still fail"
  ok "workspace built"
fi

if [ "${1:-}" = "--shell" ]; then
  step "Shell"; exec docker exec -it "$NAME" bash -c "$ENVSETUP exec bash"
fi

step "CARLA side"
PEERS=$(in_ctr "$ENVSETUP timeout 20 ros2 node list 2>/dev/null" | grep -vE 'simlingo_node|stanley_controller' | grep -c . || true)
if [ "${PEERS:-0}" -gt 0 ]; then
  ok "carla-host nodes visible on domain $DOMAIN"
  in_ctr "$ENVSETUP timeout 15 ros2 node list 2>/dev/null" | sed 's/^/       /'
else
  warn "no CARLA nodes on domain $DOMAIN — is the eval running on carla-host?"
  warn "launching anyway; the node will sit waiting for topics."
  warn "reminder: carla_ackermann_control must run on carla-host or the car never moves."
fi

step "Launch (model load takes 60–90s; stanley starts 90s in)"
exec docker exec -i "$NAME" bash -c "$ENVSETUP
  exec ros2 launch simlingo_ros simlingo.launch.py \
    checkpoint_path:=$CKPT_CTR \
    simlingo_path:=/benchmarking/simlingo $LAUNCH_ARGS"
