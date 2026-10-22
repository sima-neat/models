"""Benchmark one compiled MPK with Neat Core and save its report."""

import argparse

import model_report
import pyneat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()

model = pyneat.Model(args.mpk)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_layer_stats = True
report = model_report.performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload="Model.benchmark with default options",
)
report.save(args.output)
