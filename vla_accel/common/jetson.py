"""Jetson Orin helpers: GPU clock readout and a warning when clocks are not pinned."""

from __future__ import annotations

import sys
from pathlib import Path

_DEVFREQ = Path("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu")


def gpu_mhz() -> int:
    """Current GPU clock in MHz, -1 if unreadable."""
    try:
        return int((_DEVFREQ / "cur_freq").read_text()) // 1_000_000
    except OSError:
        return -1


def warn_if_unpinned() -> None:
    """Print a warning unless `sudo jetson_clocks` has pinned the GPU clock."""
    try:
        lo, hi = ((_DEVFREQ / f).read_text().strip() for f in ("min_freq", "max_freq"))
    except OSError:
        return
    if lo != hi:
        print("WARNING: GPU clock not pinned (run `sudo nvpmodel -m 0 && sudo jetson_clocks`); "
              "timings will be noisy.", file=sys.stderr)
