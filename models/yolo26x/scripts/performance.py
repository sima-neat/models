"""Benchmark one compiled MPK with Neat's YOLO26 box decoder and save its report."""

import argparse

import model_report
import pyneat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.YoloV26
options.top_k = 100
model = pyneat.Model(args.mpk, options)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_mla_layer_stats = True
benchmark_options.include_mla_stage_stats = True
# YOLO26 expects letterboxed input; without this, Neat labels the synthetic
# input stretched and the YOLO26 box decoder rejects it.
benchmark_options.resize_mode = pyneat.ResizeMode.Letterbox
postprocess = model.info().selection.selected_post_kind
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=f"YoloV26 box decoder, top 100, {postprocess} post-processing",
)
report.save(args.output)
