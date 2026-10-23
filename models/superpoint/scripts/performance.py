"""Benchmark one compiled SuperPoint MPK with Neat's SuperPoint decoder and save its report."""

import argparse

import model_report
import pyneat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.SuperPoint
# Magic Leap's demo settings: keypoint threshold 0.015, NMS radius 4, a 4-pixel border.
options.superpoint.profile = pyneat.SuperPointProfile.MagicLeapDemoV1
model = pyneat.Model(args.mpk, options)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_layer_stats = True
postprocess = model.info().selection.selected_post_kind
report = model_report.performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=f"SuperPoint decoder, MagicLeapDemoV1 profile, {postprocess} post-processing",
)
report.save(args.output)
