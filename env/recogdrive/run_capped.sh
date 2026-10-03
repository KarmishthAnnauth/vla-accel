#!/usr/bin/env bash
#
# run_capped.sh — run a heavy job so that it cannot take the Orin down.
#
#   ./run_capped.sh <command> [args...]
#   MEM_MAX=12G MIN_AVAIL_MB=10000 ./run_capped.sh ./setup_venv.sh
#   UNIT=myjob ./run_capped.sh ...     (then: systemctl --user stop myjob.scope)
#
# Two independent guards (the flash-attn build once exhausted memory, froze the
# Orin and powered it off, and over ssh there was no way to turn it back on):
#   1. a cgroup: the job runs in a systemd user scope with MemoryMax=$MEM_MAX
#      and MemorySwapMax=0 (no thrashing into zram).  On overflow the kernel
#      OOM-kills the job only; the rest of the system keeps its memory.
#   2. a watchdog: every 2 s it logs memory / temperature to
#      logs/guard_<unit>.log (synced, so it survives a crash) and stops the
#      scope if system-wide MemAvailable falls below $MIN_AVAIL_MB or the
#      hottest thermal zone exceeds $MAX_TEMP_C.
set -o pipefail
[ $# -ge 1 ] || { sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 2; }

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
