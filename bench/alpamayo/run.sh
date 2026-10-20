#!/bin/bash
set -e
BENCH_DIR=/home/USER/vla_benchmarking/benchmarking
HERE="$(cd "$(dirname "$0")" && pwd)"
source ${BENCH_DIR}/a1_5_venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=${CUDA_HOME}/targets/aarch64-linux/lib:${CUDA_HOME}/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/local/cuda-12.8/targets/aarch64-linux/lib/libcusolver.so.11.7.2.55${LD_PRELOAD:+:$LD_PRELOAD}
export PYTHONPATH=${HERE}:${PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${BENCH_DIR}/.torchinductor_cache_alpamayo}
export HF_HUB_OFFLINE=1
if [ "$(cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq)" != \
     "$(cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq)" ]; then
    echo "WARNING: GPU clock is not pinned (sudo jetson_clocks); timings will be noisy." >&2
fi
cd "${HERE}"
exec python3 "$@"
