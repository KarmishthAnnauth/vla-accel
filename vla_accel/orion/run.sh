#!/usr/bin/env bash
# Run a vla_accel.orion module in the ORION container on a Jetson AGX Orin.
#   ./vla_accel/orion/run.sh bench --variants baseline --profile
#   ./vla_accel/orion/run.sh bench_b2d --setup default --out vla_accel/orion/results/default.npz
#   ./vla_accel/orion/run.sh trtexec --onnx=... --saveEngine=...   (the image's TensorRT 8.6)
set -eo pipefail

IMAGE=${ORION_IMAGE:-karmishthannauth/orion_env_ros:v01}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
WEIGHTS_DIR=${WEIGHTS_DIR:-/path/to/models/Orion}       # Orion.pth + pretrain_qformer/
B2D_DIR=${B2D_DIR:-/path/to/bench2drive-mini}           # extracted Bench2Drive clips
ORION_REPO_HOST=${ORION_REPO_HOST:-}                    # optional: replaces the image's /root/Orion
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/orion_inductor}     # torch.compile cache across runs

[ $# -ge 1 ] || { echo "usage: $0 <module> [args...]  (bench, bench_b2d, int8_sweep, export_vit_onnx, check_vit_engine, trtexec)" >&2; exit 2; }
MODULE=$1; shift

GPU=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
if [ -r "$GPU/min_freq" ] && [ "$(cat "$GPU/min_freq")" != "$(cat "$GPU/max_freq")" ]; then
  echo "WARNING: GPU clock not pinned (sudo nvpmodel -m 0 && sudo jetson_clocks); timings will be noisy." >&2
fi
[ -s "$WEIGHTS_DIR/Orion.pth" ] || echo "WARNING: $WEIGHTS_DIR/Orion.pth not found (set WEIGHTS_DIR)" >&2
[ -d "$B2D_DIR" ] || echo "WARNING: $B2D_DIR not found (set B2D_DIR)" >&2
mkdir -p "$CACHE_DIR"

EXTRA=()
[ -n "$ORION_REPO_HOST" ] && EXTRA+=(-v "$ORION_REPO_HOST":/root/Orion)
TTY=(); [ -t 0 ] && TTY=(-it)

exec docker run --rm "${TTY[@]}" --runtime nvidia --network host --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e PYTHONPATH=/vla-accel \
  -e B2D_DIR=/b2d -e ORION_REPO=/root/Orion -e ORION_CKPT=/models/Orion/Orion.pth \
  -e TORCHINDUCTOR_CACHE_DIR=/cache -e TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
  -v "$REPO":/vla-accel -v "$WEIGHTS_DIR":/models/Orion:ro -v "$B2D_DIR":/b2d:ro -v "$CACHE_DIR":/cache \
  "${EXTRA[@]}" -w /vla-accel --entrypoint bash "$IMAGE" -c '
    if [ ! -e /root/Orion/ckpts/pretrain_qformer ]; then
      mkdir -p /root/Orion/ckpts && ln -sfn /models/Orion/pretrain_qformer /root/Orion/ckpts/pretrain_qformer
    fi
    [ "$0" = trtexec ] && exec /usr/src/tensorrt/bin/trtexec "$@"
    exec python3 -m "vla_accel.orion.$0" "$@"' "$MODULE" "$@"
