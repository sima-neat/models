"""Benchmark one compiled MPK with Neat's YOLO26 pose decoder and write its report."""

import argparse
import json

import pyneat


def report_json(value):
    """Return a Core report object as JSON values, so new report fields need no script change."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [report_json(item) for item in value]
    return {
        name: report_json(getattr(value, name))
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="benchmark report JSON to write")
args = parser.parse_args()

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.YoloV26Pose
options.top_k = 100
model = pyneat.Model(args.mpk, options)
# 1000 samples size the latency pass; Core keeps the throughput pass running for at least 30 s,
# so board telemetry has several samples.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
report = model.benchmark(benchmark_options)
with open(args.output, "w", encoding="utf-8") as output:
    json.dump({
        "core_version": pyneat.__version__,
        **report_json(report),
        "samples": benchmark_options.num_samples,
        "execution_plan": json.loads(model.execution_plan().to_json()),
        "workload": {
            "selection": "YoloV26Pose box decoder, top 100",
            "postprocess": model.info().selection.selected_post_kind,
        },
    }, output, allow_nan=False)
