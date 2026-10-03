#!/usr/bin/env bash
#
# setup_venv.sh — build the ReCogDrive Python environment as a venv on the
# external SSD (no container).
#
#   ./setup_venv.sh              create / complete the venv (idempotent)
#   VENV=/other/path ./setup_venv.sh
#
# Host Python 3.10 (Ubuntu 22.04) with --system-site-packages, so ROS 2 Humble's
# rclpy and message packages from /opt/ros/humble are importable as-is.
#
# The SSD is exfat over FUSE: no symlinks and case-insensitive names.  So:
#   * the venv is made with --copies on the root disk (venv insists on a
#     lib64 -> lib symlink) and moved over without that link, paths rewritten;
#   * uv installs with --link-mode copy and keeps its cache on the root disk
#     (the cache is symlink-based); it is cleaned at the end.
#
# Nothing is compiled: every package must come as a wheel (--no-build), except
# two pure-Python sdists installed on their own: nuplan-devkit (from git) and
# antlr4-python3-runtime.  A source build like the
# flash-attn one is what froze the Orin; if a wheel is missing this fails
# instead.  Run it through the memory guard anyway:
#   ./run_capped.sh ./setup_venv.sh
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${RECOGDRIVE_REPO:-/home/USER/vla_benchmarking/benchmarking/recogdrive}"   # the ReCogDrive checkout
VENV=${VENV:-/media/USER/EXTSSD/ReCogDrive/venv}
PY=/usr/bin/python3.10
CUDA=/usr/local/cuda-12.6
export UV_LINK_MODE=copy MAX_JOBS=1 UV_CONCURRENT_BUILDS=1
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv-recogdrive}

step(){ echo; echo "==> $*"; }

if [ ! -x "$VENV/bin/python" ]; then
  step "Creating venv at $VENV"
  TMP=$(mktemp -d)
  "$PY" -m venv --copies --system-site-packages --without-pip "$TMP/venv"
  mkdir -p "$VENV"
  for d in bin include lib pyvenv.cfg; do cp -r --no-dereference "$TMP/venv/$d" "$VENV/"; done
  sed -i "s#$TMP/venv#$VENV#g" "$VENV"/bin/activate*
  rm -rf "$TMP"
fi
grep -q PYTHONNOUSERSITE "$VENV/bin/activate" \
  || printf '\n# recogdrive: keep ~/.local out of this venv\nexport PYTHONNOUSERSITE=1\n' >> "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
"$VENV/bin/python" -c "import sys; assert sys.prefix != sys.base_prefix, 'not a venv'; print('venv', sys.prefix)"
uvpip(){ uv pip "$1" --python "$VENV/bin/python" "${@:2}"; }

step "torch / torchvision / triton / flash-attn (jetson-ai-lab jp6/cu126)"
uvpip install --no-build -r "$HERE/requirements-jetson.txt"

NV="$VENV/lib/python3.10/site-packages/triton/backends/nvidia"
mkdir -p "$NV/include"
cp "$CUDA/include/cuda.h" "$NV/include/cuda.h"
cp "$CUDA/bin/ptxas" "$NV/bin/ptxas"

step "ReCogDrive / InternVL Python packages"
uvpip install --no-deps "antlr4-python3-runtime==4.9.3"
uvpip install --no-build -r "$HERE/requirements.txt" -c "$HERE/requirements-jetson.txt"
step "nuplan-devkit import chain (--no-deps)"
NODEPS=$(mktemp)
grep -v '^nuplan-devkit' "$HERE/requirements-nodeps.txt" > "$NODEPS"
uvpip install --no-deps --no-build -r "$NODEPS"
rm -f "$NODEPS"
uvpip install --no-deps "$(grep '^nuplan-devkit' "$HERE/requirements-nodeps.txt")"

step "Import check"
( cd /tmp && PYTHONPATH="${PYTHONPATH:-}:$HERE/stubs" "$VENV/bin/python" "$HERE/check_imports.py" "$REPO" )

uv cache clean >/dev/null 2>&1 || true
echo
echo "done.  activate with:  source $VENV/bin/activate"
