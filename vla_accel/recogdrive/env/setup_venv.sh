#!/usr/bin/env bash
# Build the ReCogDrive venv on a Jetson AGX Orin (JetPack 6, Python 3.10). Idempotent; wheels only.
#   RECOGDRIVE_VENV=/path/venv RECOGDRIVE_REPO=/path/recogdrive ./env/run_capped.sh ./env/setup_venv.sh
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REQ="$(dirname "$HERE")"
REPO="${RECOGDRIVE_REPO:-/opt/recogdrive}"
VENV="${RECOGDRIVE_VENV:-$HOME/venvs/recogdrive}"
PY=/usr/bin/python3.10
CUDA=/usr/local/cuda-12.6
export UV_LINK_MODE=copy MAX_JOBS=1 UV_CONCURRENT_BUILDS=1
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv-recogdrive}

step(){ echo; echo "==> $*"; }

if [ ! -x "$VENV/bin/python" ]; then
  step "Creating venv at $VENV"
  # Built elsewhere and copied without the lib64 symlink, so it also works on exFAT.
  # System site-packages: JetPack's TensorRT Python bindings (trt_vit.py).
  TMP=$(mktemp -d)
  "$PY" -m venv --copies --system-site-packages --without-pip "$TMP/venv"
  mkdir -p "$VENV"
  for d in bin include lib pyvenv.cfg; do cp -r --no-dereference "$TMP/venv/$d" "$VENV/"; done
  sed -i "s#$TMP/venv#$VENV#g" "$VENV"/bin/activate*
  rm -rf "$TMP"
fi
grep -q PYTHONNOUSERSITE "$VENV/bin/activate" \
  || printf '\nexport PYTHONNOUSERSITE=1\n' >> "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
"$VENV/bin/python" -c "import sys; assert sys.prefix != sys.base_prefix, 'not a venv'; print('venv', sys.prefix)"
uvpip(){ uv pip "$1" --python "$VENV/bin/python" "${@:2}"; }

step "torch / torchvision / triton / flash-attn (jetson-ai-lab jp6/cu126)"
uvpip install --no-build -r "$REQ/requirements-jetson.txt"

# The Jetson triton wheel lacks cuda.h and ptxas; torch.compile needs them.
NV="$VENV/lib/python3.10/site-packages/triton/backends/nvidia"
mkdir -p "$NV/include"
cp "$CUDA/include/cuda.h" "$NV/include/cuda.h"
cp "$CUDA/bin/ptxas" "$NV/bin/ptxas"

step "ReCogDrive / InternVL Python packages"
uvpip install --no-deps "antlr4-python3-runtime==4.9.3"
uvpip install --no-build -r "$REQ/requirements.txt" -c "$REQ/requirements-jetson.txt"
step "nuplan-devkit import chain (--no-deps)"
NODEPS=$(mktemp)
grep -v '^nuplan-devkit' "$REQ/requirements-nodeps.txt" > "$NODEPS"
uvpip install --no-deps --no-build -r "$NODEPS"
rm -f "$NODEPS"
uvpip install --no-deps "$(grep '^nuplan-devkit' "$REQ/requirements-nodeps.txt")"

step "Import check"
( cd /tmp && PYTHONPATH="${PYTHONPATH:-}:$HERE/stubs" "$VENV/bin/python" "$HERE/check_imports.py" "$REPO" )

uv cache clean >/dev/null 2>&1 || true
echo
echo "done.  activate with:  source $VENV/bin/activate"
