#!/usr/bin/env python3
"""Governor/thermal benchmark stub for SIA edge inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class GovernorBench:
    """Simulate thermal governor behavior under repeated inference calls.

    No actual hardware access; the bench reports whether a naive scheduler
    would throttle before the workload completes. Replace with real sysfs
    reads (/sys/class/thermal, /sys/devices/system/cpu/cpufreq) on device.
    """

    def __init__(
        self,
        trip_temp_c: float = 75.0,
        ambient_c: float = 35.0,
        cooling_rate_c_per_s: float = 2.0,
        heating_rate_c_per_call: float = 3.0,
    ) -> None:
        self.trip_temp_c = trip_temp_c
        self.ambient_c = ambient_c
        self.cooling_rate_c_per_s = cooling_rate_c_per_s
        self.heating_rate_c_per_call = heating_rate_c_per_call

    def simulate(self, calls: int, call_interval_s: float = 0.05) -> dict[str, Any]:
        temp = self.ambient_c
        throttled = 0
        max_temp = temp
        elapsed = 0.0
        for i in range(calls):
            elapsed += call_interval_s
            # natural cooling during interval
            temp = max(
                self.ambient_c,
                temp - self.cooling_rate_c_per_s * call_interval_s,
            )
            # heating from inference
            temp += self.heating_rate_c_per_call
            max_temp = max(max_temp, temp)
            if temp >= self.trip_temp_c:
                throttled += 1
                # governor throttles: pause until under trip
                over = temp - self.trip_temp_c
                pause = over / self.cooling_rate_c_per_s
                elapsed += pause
                temp = max(self.ambient_c, temp - self.cooling_rate_c_per_s * pause)
        return {
            "calls": calls,
            "elapsed_seconds": round(elapsed, 3),
            "max_temp_c": round(max_temp, 2),
            "throttle_events": throttled,
            "tripped": max_temp >= self.trip_temp_c,
        }

    def sweep(
        self,
        call_counts: list[int] | None = None,
        interval_s: float = 0.05,
    ) -> dict[str, Any]:
        if call_counts is None:
            call_counts = [5, 10, 20, 50]
        runs = []
        for n in call_counts:
            runs.append(self.simulate(n, interval_s))
        return {"trip_temp_c": self.trip_temp_c, "runs": runs}


def run_governor(output_path: Path | str | None = None) -> dict[str, Any]:
    bench = GovernorBench()
    result = bench.sweep()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"governor results written to {output_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIA governor/thermal benchmark")
    parser.add_argument("--output", default="sia-lab/eval/outputs/governor.json")
    args = parser.parse_args(argv)
    result = run_governor(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
