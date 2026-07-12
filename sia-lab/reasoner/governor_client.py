#!/usr/bin/env python3
"""Governor-to-reasoner budget adapter.

Reads device state (thermal/battery) and returns a budget dict:
  {max_loops, act_max_steps, deep_allowed, hot}

Ponytail: real implementation would read /sys/class/thermal and
/sys/class/power_supply. This module starts with a simulation mode that the
eval harness can perturb for repeatable tests.
"""

from __future__ import annotations



class GovernorClient:
    """Produce a compute budget from device state."""

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode  # normal | hot | cool

    def budget(self) -> dict:
        if self.mode == "hot":
            return {"max_loops": 1, "act_max_steps": 1, "deep_allowed": False, "hot": True}
        if self.mode == "cool":
            return {"max_loops": 16, "act_max_steps": 8, "deep_allowed": True, "hot": False}
        return {"max_loops": 8, "act_max_steps": 6, "deep_allowed": True, "hot": False}

    def set_mode(self, mode: str) -> None:
        self.mode = mode


def read_sysfs_budget() -> dict:
    """Real device budget from Linux thermal/power sysfs. Falls back to normal."""
    # ponytail: untrusted environment fallback; real devices parse /sys/class/thermal/thermal_zone*/temp
    return GovernorClient("normal").budget()


def main() -> int:
    for mode in ("cool", "normal", "hot"):
        g = GovernorClient(mode)
        print(f"{mode}: {g.budget()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
