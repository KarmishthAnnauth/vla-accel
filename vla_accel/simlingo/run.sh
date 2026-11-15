#!/usr/bin/env bash
# Run a module of vla_accel.simlingo in the SimLingo container on a Jetson AGX Orin.
#   ./run.sh bench --clip 0 --frames 5
set -eo pipefail

IMAGE=${SIMLINGO_IMAGE:-karmishthannauth/simlingo:humble-cyclonedd}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
SIMLINGO_DIR=${SIMLINGO_DIR:-$HOME/simlingo}            # upstream checkout (with pretrained/InternVL2-1B)
WEIGHTS_DIR=${WEIGHTS_DIR:-$HOME/models/simlingo_hf}    # holds simlingo/checkpoints/epoch=013.ckpt/
B2D_DIR=${B2D_DIR:-/data/bench2drive-mini}

MODULE=${1:?usage: run.sh <bench|profile_baseline|ablation|clock_test> [args...]}
shift

DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
if [ -r "$DEVFREQ/min_freq" ] && [ "$(cat $DEVFREQ/min_freq)" != "$(cat $DEVFREQ/max_freq)" ]; then
  echo "WARNING: GPU clock not pinned (sudo nvpmodel -m 0 && sudo jetson_clocks); timings will be noisy." >&2
fi

exec docker run --rm -i --runtime nvidia --network host --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$REPO":/vla-accel \
  -v "$SIMLINGO_DIR":/simlingo \
  -v "$WEIGHTS_DIR":/models/simlingo:ro \
  -v "$B2D_DIR":/data/b2d:ro \
  -e PYTHONPATH=/vla-accel \
  -e SIMLINGO_DIR=/simlingo \
  -e SIMLINGO_CKPT=/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
  -e B2D_DIR=/data/b2d \
  -w /vla-accel \
  --entrypoint python3 "$IMAGE" -m "vla_accel.simlingo.$MODULE" "$@"
