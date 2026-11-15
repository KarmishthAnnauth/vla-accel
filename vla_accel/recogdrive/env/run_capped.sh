#!/usr/bin/env bash
# Run a heavy job in a memory-capped systemd user scope with a memory/temperature watchdog.
#   ./run_capped.sh <command> [args...]      (MEM_MAX, MIN_AVAIL_MB, MAX_TEMP_C, UNIT)
set -o pipefail
[ $# -ge 1 ] || { sed -n '2,3p' "$0" | sed 's/^# \?//'; exit 2; }

MEM_MAX=${MEM_MAX:-16G}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-8000}
MAX_TEMP_C=${MAX_TEMP_C:-88}
UNIT=${UNIT:-capped-$(date +%Y%m%d-%H%M%S)}
LOGDIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOGDIR"
GLOG="$LOGDIR/guard_$UNIT.log"

watchdog(){
  echo "# $(date '+%F %T') $UNIT: MemoryMax=$MEM_MAX, stop below ${MIN_AVAIL_MB} MB available or above ${MAX_TEMP_C} C: $*" >> "$GLOG"
  while sleep 2; do
    systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null || { [ -n "$started" ] && break; continue; }
    started=1
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    used=$(( $(cat "/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/$UNIT.scope/memory.current" 2>/dev/null || echo 0) / 1048576 ))
    tj=0
    for z in /sys/devices/virtual/thermal/thermal_zone*/temp; do
      t=$(( $(cat "$z" 2>/dev/null || echo 0) / 1000 )); [ "$t" -gt "$tj" ] && tj=$t
    done
    echo "$(date '+%T') job=${used}MB avail=${avail}MB temp=${tj}C" >> "$GLOG"; sync "$GLOG"
    if [ "$avail" -lt "$MIN_AVAIL_MB" ] || [ "$tj" -gt "$MAX_TEMP_C" ]; then
      echo "$(date '+%T') GUARD: stopping $UNIT (avail=${avail}MB temp=${tj}C)" | tee -a "$GLOG" >&2
      systemctl --user stop "$UNIT.scope"; sync "$GLOG"; break
    fi
  done
}

watchdog "$@" &
WD=$!
systemd-run --user --scope --quiet --unit="$UNIT" \
  -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 "$@"
rc=$?
kill "$WD" 2>/dev/null; wait "$WD" 2>/dev/null
[ "$rc" -eq 137 ] && echo "run_capped: $UNIT was killed (memory cap $MEM_MAX or guard) -- see $GLOG" >&2
exit "$rc"
