import json
import sys

import pyneat

model = pyneat.Model(sys.argv[1])
report = model.benchmark()
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump({
        "core_version": pyneat.__version__,
        "latency_ms": report.latency_ms,
        "fps": report.fps,
        "avg_power_watts": report.avg_power_watts,
        "energy_joules": report.energy_joules,
    }, output, allow_nan=False)
