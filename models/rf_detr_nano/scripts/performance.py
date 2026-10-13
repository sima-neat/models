"""Benchmark one compiled MPK with Neat Core and write its report."""

import argparse
import json

import pyneat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="benchmark report JSON to write")
args = parser.parse_args()

model = pyneat.Model(args.mpk)
# 1000 samples keep the throughput window long enough for steady FPS and several power readings.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
report = model.benchmark(benchmark_options)
with open(args.output, "w", encoding="utf-8") as output:
    json.dump({
        "core_version": pyneat.__version__,
        "latency_ms": report.latency_ms,
        "fps": report.fps,
        "avg_power_watts": report.avg_power_watts,
        "energy_joules": report.energy_joules,
        "samples": benchmark_options.num_samples,
    }, output, allow_nan=False)
