#!/usr/bin/env bash
# Run a bench script in the ReCogDrive venv, under the memory guard.
#   ./run.sh profile_baseline.py [args...]
# VENV: the venv env/recogdrive/setup_venv.sh built.  RECOGDRIVE_ENV_DIR: env/recogdrive.
HERE="$(cd "$(dirname "$0")" && pwd)"
ENVDIR="${RECOGDRIVE_ENV_DIR:-$HERE/../../env/recogdrive}"
source "${VENV:-/media/USER/EXTSSD/ReCogDrive/venv}/bin/activate"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:$ENVDIR/stubs:$HERE"
cd "$HERE"
exec env MEM_MAX=${MEM_MAX:-24G} MIN_AVAIL_MB=${MIN_AVAIL_MB:-8000} "$ENVDIR/run_capped.sh" python -W ignore "$@"
