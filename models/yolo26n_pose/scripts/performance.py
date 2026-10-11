"""Benchmark one compiled MPK with Neat's YOLO26 pose decoder and write its report."""

import argparse
import json

import pyneat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="benchmark report JSON to write")
args = parser.parse_args()

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.YoloV26Pose
options.top_k = 300
model = pyneat.Model(args.mpk, options)
report = model.benchmark()
with open(args.output, "w", encoding="utf-8") as output:
    json.dump({
        "core_version": pyneat.__version__,
        "latency_ms": report.latency_ms,
        "fps": report.fps,
        "avg_power_watts": report.avg_power_watts,
        "energy_joules": report.energy_joules,
    }, output, allow_nan=False)
