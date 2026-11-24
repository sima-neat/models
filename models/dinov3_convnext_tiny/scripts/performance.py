"""Benchmark one compiled DINOv3 ConvNeXt Tiny MPK with Neat Core on an ImageNet image and save its report."""

import argparse
import random
from pathlib import Path

import model_report
import pyneat
from model_inputs.frame import FRAME_HEIGHT, FRAME_WIDTH, letterbox_to_720p, read_rgb

VALIDATION_SPLIT = "val2012"


def sample_image(dataset: Path) -> Path:
    """One val2012 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / VALIDATION_SPLIT).glob("*.JPEG"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {VALIDATION_SPLIT}/")
    return random.Random(0).choice(images)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
parser.add_argument("--output", required=True, help="directory for model-report.json")
# The benchmark runner passes the package's resources, here only the license. The benchmark reads none of them.
parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
args = parser.parse_args()
dataset_id, dataset_dir = args.dataset.split("=", 1)
image = sample_image(Path(dataset_dir))

options = pyneat.ModelOptions()
# Neat resizes, normalizes, quantizes and tessellates the frame inside the measured pipeline.
options.preprocess.kind = pyneat.InputKind.Image
# Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
options.preprocess.input_max_width = FRAME_WIDTH
options.preprocess.input_max_height = FRAME_HEIGHT
options.preprocess.enable = pyneat.AutoFlag.On
options.preprocess.resize.enable = pyneat.AutoFlag.On
options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
options.preprocess.normalize.enable = pyneat.AutoFlag.On
# Without a preset Neat only scales pixels to [0, 1]; the model expects ImageNet mean and std.
options.preprocess.preset = pyneat.NormalizePreset.ImageNet
model = pyneat.Model(args.mpk, options)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_mla_layer_stats = True
benchmark_options.include_mla_stage_stats = True
frame = letterbox_to_720p(read_rgb(image))
benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=(
        f"{dataset_id} {image.parent.name}/{image.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, "
        "Neat preprocessing stretched to the model input"
    ),
)
report.save(args.output)
