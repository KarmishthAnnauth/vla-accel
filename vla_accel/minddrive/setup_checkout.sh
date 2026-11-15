#!/usr/bin/env bash
# Prepare a MindDrive checkout for the bench: symlinks, compiled mmcv ops, local patches, container import check.
# Usage: MINDDRIVE_DIR=... MODELS_DIR=... vla_accel/minddrive/setup_checkout.sh
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MINDDRIVE_DIR=${MINDDRIVE_DIR:-$HOME/MindDrive}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
IMAGE=${IMAGE:-karmishthannauth/minddrive_env:v01}
ORION_IMAGE=${ORION_IMAGE:-karmishthannauth/orion_env_ros:v01}
CTR_DIR=/workspace/MindDrive

die(){ echo "FAIL $*" >&2; exit 1; }
ok(){  echo "ok   $*"; }

[ -d "$MINDDRIVE_DIR/mmcv" ] && [ -f "$MINDDRIVE_DIR/adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py" ] \
  || die "MindDrive checkout not found at $MINDDRIVE_DIR"

# Weights are mounted at /models in the container; the config reads ./Bench2DriveZoo/ckpts/...
[ -e "$MINDDRIVE_DIR/Bench2DriveZoo" ] || ln -sfn . "$MINDDRIVE_DIR/Bench2DriveZoo"
mkdir -p "$MINDDRIVE_DIR/ckpts"
for d in llava-qwen2.5-3b llava-qwen2-0.5b llava-qwen2.5-3b-eva02_petr_proj.pth llava-qwen2-0.5b-eva02_petr_proj.pth; do
  ln -sfn "/models/MindDrive/$d" "$MINDDRIVE_DIR/ckpts/$d"
done
for f in minddrive_3b_rltrain.pth llava-qwen2.5-3b/config.json llava-qwen2.5-3b/tokenizer.json; do
  [ -s "$MODELS_DIR/MindDrive/$f" ] || echo "warn $MODELS_DIR/MindDrive/$f missing"
done
ok "Bench2DriveZoo and ckpts symlinks"

# Same mmcv csrc as ORION: reuse the ORION image's compiled ops.
SOS=(mmcv/_ext.cpython-310-aarch64-linux-gnu.so
     mmcv/ops/iou3d_det/iou3d_cuda.cpython-310-aarch64-linux-gnu.so
     mmcv/ops/roiaware_pool3d/roiaware_pool3d_ext.cpython-310-aarch64-linux-gnu.so)
missing=0
for so in "${SOS[@]}"; do [ -s "$MINDDRIVE_DIR/$so" ] || missing=1; done
if [ "$missing" = 1 ]; then
  tmp=$(docker create "$ORION_IMAGE")
  for so in "${SOS[@]}"; do docker cp "$tmp:/root/Orion/$so" "$MINDDRIVE_DIR/$so"; done
  docker rm "$tmp" >/dev/null
fi
ok "compiled mmcv ops"

for p in "$HERE"/docker/0001-lazy-rl-runner-imports.patch "$HERE"/docker/0002-fp16-qwen-load.patch; do
  if (cd "$MINDDRIVE_DIR" && patch -p0 -R --dry-run -s -f < "$p" >/dev/null 2>&1); then
    ok "$(basename "$p") already applied"
  else
    (cd "$MINDDRIVE_DIR" && patch -p0 -s -f < "$p") || die "$(basename "$p") does not apply"
    ok "$(basename "$p") applied"
  fi
done

docker run --rm --runtime nvidia -v "$MINDDRIVE_DIR":$CTR_DIR -w / "$IMAGE" python3 - <<'PY' \
  || die "MindDrive mmcv fork does not import in $IMAGE"
import mmcv, mmcv._ext, peft, torch, transformers
assert mmcv.__file__.startswith("/workspace/MindDrive/"), mmcv.__file__
assert torch.__version__.startswith("2.4.0a0"), torch.__version__
assert transformers.__version__ == "4.45.2", transformers.__version__
print("ok   mmcv", mmcv.__file__, "| torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
PY
