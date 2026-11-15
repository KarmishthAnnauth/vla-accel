#!/usr/bin/env bash
# Run a vla_accel.recogdrive module in the ReCogDrive venv, under the memory guard.
#   ./vla_accel/recogdrive/run.sh bench [args...]      (module name without the package prefix)
set -eo pipefail
[ $# -ge 1 ] || { sed -n '2,3p' "$0" | sed 's/^# \?//'; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export RECOGDRIVE_VENV="${RECOGDRIVE_VENV:-$HOME/venvs/recogdrive}"
export RECOGDRIVE_REPO="${RECOGDRIVE_REPO:-/opt/recogdrive}"
export RECOGDRIVE_WEIGHTS="${RECOGDRIVE_WEIGHTS:-/data/weights/ReCogDrive}"
export B2D_DIR="${B2D_DIR:-/data/bench2drive-mini}"

GOV=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
if [ -r "$GOV/min_freq" ] && [ "$(cat "$GOV/min_freq")" != "$(cat "$GOV/max_freq")" ]; then
  echo "WARNING: GPU clock not pinned (sudo nvpmodel -m 0 && sudo jetson_clocks); timings will be noisy." >&2
fi

source "$RECOGDRIVE_VENV/bin/activate"
# The planner checkpoint is a Lightning pickle, refused by torch >= 2.6's weights_only default.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT:$RECOGDRIVE_REPO:${PYTHONPATH:+$PYTHONPATH:}$HERE/env/stubs"
cd "$ROOT"
MOD="$1"; shift
exec env MEM_MAX=${MEM_MAX:-24G} MIN_AVAIL_MB=${MIN_AVAIL_MB:-8000} \
  "$HERE/env/run_capped.sh" python -W ignore -m "vla_accel.recogdrive.${MOD%.py}" "$@"
