#!/usr/bin/env bash
# Run a vla_accel.minddrive module in the MindDrive container on the Orin.
# Usage: vla_accel/minddrive/run.sh bench [args]   |   vla_accel/minddrive/run.sh shell
set -eo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MINDDRIVE_DIR=${MINDDRIVE_DIR:-$HOME/MindDrive}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
B2D_DIR=${B2D_DIR:-/data/bench2drive-mini}
OUT_DIR=${OUT_DIR:-$REPO/vla_accel/minddrive/out}
IMAGE=${IMAGE:-karmishthannauth/minddrive_env:v01}

MOD=${1:?usage: run.sh <module|shell> [args]}; shift
mkdir -p "$OUT_DIR"

DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
if [ -r "$DEVFREQ/min_freq" ] && [ "$(cat "$DEVFREQ/min_freq")" != "$(cat "$DEVFREQ/max_freq")" ]; then
  echo "WARNING: GPU clock not pinned (sudo nvpmodel -m 0 && sudo jetson_clocks); timings will be noisy." >&2
fi

ARGS=(--rm --runtime nvidia --network host --shm-size=8g
      -v "$REPO":/workspace/vla-accel
      -v "$MINDDRIVE_DIR":/workspace/MindDrive
      -v "$MODELS_DIR":/models:ro
      -v "$B2D_DIR":/data/b2d:ro
      -v "$OUT_DIR":/out
      -e B2D_DIR=/data/b2d
      -e PYTHONPATH=/workspace/vla-accel:/workspace/MindDrive
      -e TORCHINDUCTOR_CACHE_DIR=/workspace/MindDrive/.torchinductor_cache
      -e TORCHINDUCTOR_FX_GRAPH_CACHE=1
      -w /workspace/MindDrive)

if [ "$MOD" = shell ]; then
  exec docker run -it "${ARGS[@]}" "$IMAGE" bash
fi
exec docker run -i "${ARGS[@]}" "$IMAGE" python3 -m "vla_accel.minddrive.$MOD" "$@"
