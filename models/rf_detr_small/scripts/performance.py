"""Benchmark one compiled MPK with Neat Core and save its report."""

import argparse
import json
from pathlib import Path

import pyneat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for benchmark-report.json and benchmark-context.json")
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

model = pyneat.Model(args.mpk)
# 1000 samples size each pass, and each pass runs for at least 60 s, so board telemetry, read every
# 2 s, has about 30 samples.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
model.benchmark(benchmark_options).save(output / "benchmark-report.json")
# What the report does not record: the Core release, the requested run, and the graph that ran.
context = {
    "core_version": pyneat.__version__,
    "num_samples": benchmark_options.num_samples,
    "min_duration_ms": benchmark_options.min_duration_ms,
    "execution_plan": json.loads(model.execution_plan().to_json()),
}
(output / "benchmark-context.json").write_text(json.dumps(context, allow_nan=False), encoding="utf-8")
