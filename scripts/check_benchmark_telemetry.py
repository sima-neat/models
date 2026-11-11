#!/usr/bin/env python3
"""Require a fresh SoM power sample before starting a publishable benchmark.

Usage: check_benchmark_telemetry.py

The benchmark records its own telemetry during the measured window. This
unprivileged preflight neither installs nor changes the Sentinel service.
"""

from __future__ import annotations

import math
import time

TIMEOUT_SECONDS = 10.0
POLL_SECONDS = 0.1


def has_power(telemetry) -> bool:
    power = telemetry.power_som_watts
    return (
        telemetry.status == "ok"
        and 0 < power.samples <= telemetry.samples
        and power.unit == "W"
        and math.isfinite(power.mean)
        and power.mean > 0
    )


def main() -> int:
    import pyneat

    monitor = pyneat.TelemetryMonitor()
    try:
        monitor.start()
        deadline = time.monotonic() + TIMEOUT_SECONDS
        telemetry = monitor.read()
        while not has_power(telemetry) and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            telemetry = monitor.read()
    finally:
        telemetry = monitor.stop()
    if not has_power(telemetry):
        raise SystemExit(
            "Benchmark requires sampled SoM power from Sentinel; "
            f"status={telemetry.status!r}, power samples={telemetry.power_som_watts.samples}. "
            "Check Sentinel before retrying; no model measurement was started."
        )
    print(f"Sentinel preflight: {telemetry.power_som_watts.samples} fresh SoM power sample(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
