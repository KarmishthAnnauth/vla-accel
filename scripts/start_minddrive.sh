#!/usr/bin/env bash
set -eo pipefail

IMAGE=karmishthannauth/minddrive_env_ros:v01
ORION_IMAGE=karmishthannauth/orion_env_ros:v01
NAME=minddrive_ros
BENCH=/home/USER/vla_benchmarking/benchmarking
SSD_MOUNT=/media/USER/EXTSSD
MODELS_HOST="$SSD_MOUNT/models"
CYCLONE_XML=/home/USER/cyclone_zerotier.xml
ZT_IFACE=ztXXXXXXXXX
DELL_IP=192.0.2.10

DOMAIN=${DOMAIN:-0}

REPO_HOST="$BENCH/MindDrive"
REPO_CTR=/benchmarking/MindDrive
WS=/opt/minddrive_ws

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "  ${GRN}ok${RST}   $*"; }
warn(){ echo "  ${YEL}warn${RST} $*"; }
die(){  echo "  ${RED}FAIL${RST} $*" >&2; exit 1; }
step(){ echo; echo "==> $*"; }

in_ctr(){ docker exec "$NAME" bash -c "$1"; }
ENVSETUP='source /opt/ros/humble/setup.bash;
          source '"$WS"'/install/setup.bash 2>/dev/null || true;
          export PYTHONPATH=/benchmarking/MindDrive:${PYTHONPATH:-};
          export TORCHINDUCTOR_CACHE_DIR=/benchmarking/.torchinductor_cache TORCHINDUCTOR_FX_GRAPH_CACHE=1;'

FAST_ARGS=(precision:=fp16 merge_lora:=true vit_glue:=true down_proj_t:=true
           vit_weight_t:=true vit_window_nopad:=true map_head_slice:=true compile_targets:=llm,vit
           cuda_graph_vit:=true logits_slice:=true pipeline_prep:=true)

VARIANT=3b
MODE=launch
REBUILD=0
FAST=0
LAUNCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stop|--check|--shell) MODE="${1#--}" ;;
    --rebuild)              REBUILD=1 ;;
    --3b)                   VARIANT=3b ;;
    --05b|--0.5b)           VARIANT=05b ;;
    --fast)                 FAST=1 ;;
    -h|--help)              sed -n '2,45p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *:=*)                   LAUNCH_ARGS+=("$1") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done
if [ "$FAST" = 1 ]; then
  for fa in "${FAST_ARGS[@]}"; do
    key="${fa%%:=*}"; keep=1
    for la in "${LAUNCH_ARGS[@]}"; do [ "${la%%:=*}" = "$key" ] && keep=0; done
    [ "$keep" = 1 ] && LAUNCH_ARGS+=("$fa")
  done
fi

if [ "$MODE" = "stop" ]; then
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

case "$VARIANT" in
  3b)  CKPT_HOST="$MODELS_HOST/MindDrive/minddrive_3b_rltrain.pth"; LLM_DIR="llava-qwen2.5-3b"; CKPT_SIZE=29514202889 ;;
  05b) CKPT_HOST="$MODELS_HOST/MindDrive/minddrive_rltrain.pth";    LLM_DIR="llava-qwen2-0.5b"; CKPT_SIZE=6593355869 ;;
esac

step "Preflight ($VARIANT)"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "image $IMAGE not found -- build it: docker build --network host -t $IMAGE $BENCH/minddrive_env/"
ok "image $IMAGE"

if mountpoint -q "$SSD_MOUNT"; then
  ok "EXTSSD mounted ($(df -h "$SSD_MOUNT" | awk 'NR==2{print $4}') free)"
else
  echo "  ${RED}FAIL${RST} $SSD_MOUNT is not mounted" >&2
  echo "       sudo mount -t exfat-fuse -o allow_other,uid=1000,gid=1000,umask=022 \\" >&2
  echo "         /dev/sdb1 $SSD_MOUNT" >&2
  exit 1
fi

[ -s "$CKPT_HOST" ] || die "checkpoint missing: $CKPT_HOST"
sz=$(stat -c %s "$CKPT_HOST")
[ "$sz" -eq "$CKPT_SIZE" ] || warn "checkpoint is $sz bytes, expected $CKPT_SIZE (truncated copy?)"
[ "$sz" -eq "$CKPT_SIZE" ] && ok "checkpoint $(( sz/1024/1024/1024 )) GiB ($(basename "$CKPT_HOST"))"

for f in config.json tokenizer.json tokenizer_config.json; do
  [ -s "$MODELS_HOST/MindDrive/$LLM_DIR/$f" ] || die "$LLM_DIR/$f missing -- LLM will not load"
done
ls "$MODELS_HOST/MindDrive/$LLM_DIR"/*.safetensors >/dev/null 2>&1 || die "$LLM_DIR has no safetensors shards"
ok "$LLM_DIR (LLM + tokenizer)"

[ -d "$REPO_HOST/mmcv" ] && [ -f "$REPO_HOST/adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py" ] \
  || die "MindDrive checkout missing at $REPO_HOST (rsync it from $MODELS_HOST/MindDrive/MindDrive, no .git)"
[ -L "$REPO_HOST/Bench2DriveZoo" ] || ln -sfn . "$REPO_HOST/Bench2DriveZoo"
mkdir -p "$REPO_HOST/ckpts"
for d in llava-qwen2.5-3b llava-qwen2-0.5b llava-qwen2.5-3b-eva02_petr_proj.pth llava-qwen2-0.5b-eva02_petr_proj.pth; do
  [ "$(readlink "$REPO_HOST/ckpts/$d" 2>/dev/null)" = "/models/MindDrive/$d" ] || ln -sfn "/models/MindDrive/$d" "$REPO_HOST/ckpts/$d"
done
ok "checkout $REPO_HOST (upstream $(cat "$REPO_HOST/.upstream_commit" 2>/dev/null | cut -c1-8 || echo '?'))"

_EXT=mmcv/_ext.cpython-310-aarch64-linux-gnu.so
_SOS=($_EXT mmcv/ops/iou3d_det/iou3d_cuda.cpython-310-aarch64-linux-gnu.so
      mmcv/ops/roiaware_pool3d/roiaware_pool3d_ext.cpython-310-aarch64-linux-gnu.so)
_missing=0
for so in "${_SOS[@]}"; do [ -s "$REPO_HOST/$so" ] || _missing=1; done
if [ "$_missing" = 1 ]; then
  echo "  compiled ops missing: copying from $ORION_IMAGE"
  _tmp=$(docker create "$ORION_IMAGE")
  for so in "${_SOS[@]}"; do docker cp "$_tmp:/root/Orion/$so" "$REPO_HOST/$so"; done
  docker rm "$_tmp" >/dev/null
fi
ok "compiled mmcv ops present"

grep -q "_rl_import_error" "$REPO_HOST/mmcv/runner/iter_based_runner.py" \
  || die "mmcv/runner/iter_based_runner.py is unpatched -- apply minddrive_env/0001-lazy-rl-runner-imports.patch"
grep -q "0002-fp16-qwen-load" "$REPO_HOST/mmcv/utils/misc.py" \
  || die "mmcv/utils/misc.py is unpatched -- apply minddrive_env/0002-fp16-qwen-load.patch"
ok "local patches 0001 (lazy RL imports) and 0002 (fp16 Qwen load) applied"

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

step "MindDrive imports (container)"
in_ctr "$ENVSETUP cd $REPO_CTR && python3 -c 'import mmcv, mmcv._ext, transformers; assert transformers.__version__ == \"4.45.2\", transformers.__version__; import os; assert os.path.dirname(os.path.abspath(mmcv.__file__)) == \"$REPO_CTR/mmcv\", mmcv.__file__'" >/dev/null 2>&1 \
  || die "MindDrive mmcv fork does not import in the container (run --shell and: cd $REPO_CTR && python3 -c 'import mmcv')"
ok "mmcv fork + compiled ops + transformers 4.45.2"
in_ctr "test -f $REPO_CTR/Bench2DriveZoo/ckpts/$LLM_DIR/config.json && test -f /models/MindDrive/$(basename "$CKPT_HOST")" >/dev/null 2>&1 \
  || die "weights not visible in the container (SSD mount lost? ckpts symlinks?)"
ok "weights visible at /models/MindDrive (via ./Bench2DriveZoo/ckpts/$LLM_DIR)"

step "ROS workspace ($WS)"
NEED_BUILD=1
if [ "$REBUILD" = 1 ]; then
  echo "  --rebuild: forcing"
  in_ctr "rm -rf $WS" >/dev/null 2>&1 || true
elif in_ctr "$ENVSETUP python3 -c 'from carla_msgs.msg import CarlaRoute; import minddrive_ros.minddrive_node'" >/dev/null 2>&1; then
  NEWER=$(in_ctr "find /benchmarking/alpamayo-autoware/src/minddrive_ros/minddrive_ros /benchmarking/alpamayo-autoware/src/minddrive_ros/launch \
                  -newer $WS/install/minddrive_ros/share/minddrive_ros/package.xml -name '*.py' 2>/dev/null | head -1")
  if [ -n "$NEWER" ]; then echo "  node sources changed ($NEWER): rebuilding"
  else NEED_BUILD=0; ok "workspace already built"; fi
fi
if [ "$NEED_BUILD" = 1 ]; then
  echo "  building carla_msgs, minddrive_ros (~30s)…"
  in_ctr "source /opt/ros/humble/setup.bash;
          cd /benchmarking/alpamayo-autoware &&
          colcon build --base-paths src \
            --packages-up-to minddrive_ros \
            --build-base $WS/build --install-base $WS/install" > "$BENCH/minddrive_env/logs/colcon_build.log" 2>&1 \
    || die "colcon build failed -- see minddrive_env/logs/colcon_build.log"
  in_ctr "$ENVSETUP python3 -c 'from carla_msgs.msg import CarlaRoute; import minddrive_ros.minddrive_node'" >/dev/null 2>&1 \
    || die "build finished but imports still fail"
  ok "workspace built"
fi

if [ "$MODE" = "shell" ]; then
  step "Shell"; exec docker exec -it "$NAME" bash -c "$ENVSETUP cd $REPO_CTR; exec bash"
fi

step "CARLA side"
OWN_NODES='minddrive_node|image_decompress'
PEERS=$(in_ctr "$ENVSETUP timeout 20 ros2 node list 2>/dev/null" | grep -vE "$OWN_NODES" | grep -c . || true)
if [ "${PEERS:-0}" -gt 0 ]; then
  ok "carla-host nodes visible on domain $DOMAIN"
  in_ctr "$ENVSETUP timeout 15 ros2 node list 2>/dev/null" | sed 's/^/       /'
else
  warn "no CARLA nodes on domain $DOMAIN — is the eval running on carla-host?"
  warn "launching anyway; the node will sit waiting for topics."
  warn "carla-host must publish all 6 cameras (JPEG), or the node waits forever."
fi

step "Launch minddrive.launch.py variant:=$VARIANT ${LAUNCH_ARGS[*]}"
echo "  model load is minutes (the checkpoint comes off the SSD); with compile_targets set, add the compile"
exec docker exec -i "$NAME" bash -c "$ENVSETUP
  exec ros2 launch minddrive_ros minddrive.launch.py variant:=$VARIANT ${LAUNCH_ARGS[*]}"
