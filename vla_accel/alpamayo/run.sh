#!/bin/bash
# Run an Alpamayo 1.5 module in its venv on the Orin: ./run.sh bench --frames 3
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV=${ALPAMAYO_VENV:-/opt/a1_5_venv}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
CUSOLVER=${ALPAMAYO_CUSOLVER:-/usr/local/cuda-12.8/targets/aarch64-linux/lib/libcusolver.so.11.7.2.55}
export ALPAMAYO_MODEL=${ALPAMAYO_MODEL:-/data/alpamayo-1.5}
export ALPAMAYO_TRT_DIR=${ALPAMAYO_TRT_DIR:-/data/alpamayo-1.5-trt}
export B2D_DIR=${B2D_DIR:-/data/bench2drive-mini}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor_alpamayo}
export HF_HUB_OFFLINE=1

[ $# -ge 1 ] || { echo "usage: $0 <module> [args]   (bench, ablation, profile_baseline, trt_build, trt_int8)" >&2; exit 1; }
MOD=${1%.py}; shift

source "${VENV}/bin/activate"
export LD_LIBRARY_PATH=${CUDA_HOME}/targets/aarch64-linux/lib:${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH}
export LD_PRELOAD=${CUSOLVER}${LD_PRELOAD:+:$LD_PRELOAD}
export PYTHONPATH=${REPO}${ALPAMAYO_SRC:+:$ALPAMAYO_SRC}${PYTHONPATH:+:$PYTHONPATH}
cd "${REPO}"
python3 -c "from vla_accel.common import jetson; jetson.warn_if_unpinned()"
exec python3 -m "vla_accel.alpamayo.${MOD}" "$@"
