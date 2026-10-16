"""Benchmark one compiled MPK with Neat's YOLO26 segmentation decoder and save its report."""

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

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.YoloV26Seg
options.top_k = 100
model = pyneat.Model(args.mpk, options)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_layer_stats = True
model.benchmark(benchmark_options).save(output / "benchmark-report.json")
# What the report does not record: the Core release, the requested run, and the graph that ran.
context = {
    "core_version": pyneat.__version__,
    "num_samples": benchmark_options.num_samples,
    "min_duration_ms": benchmark_options.min_duration_ms,
    "include_layer_stats": benchmark_options.include_layer_stats,
    "execution_plan": json.loads(model.execution_plan().to_json()),
    "workload": {
        "selection": "YoloV26Seg decoder, top 100, 160x160 masks",
        "postprocess": model.info().selection.selected_post_kind,
    },
}
(output / "benchmark-context.json").write_text(json.dumps(context, allow_nan=False), encoding="utf-8")
