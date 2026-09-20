#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROSBRIDGE_HOST="${ROSBRIDGE_HOST:-192.0.2.10}"
export ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9190}"
export CKPT_PATH="${CKPT_PATH:-/home/USER/models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/team_code:${PYTHONPATH:-}"

cd "$REPO_ROOT"
echo "[orin] rosbridge ws://${ROSBRIDGE_HOST}:${ROSBRIDGE_PORT}"
echo "[orin] ckpt      ${CKPT_PATH}"
exec python3 team_code/inference_client_simlingo.py
