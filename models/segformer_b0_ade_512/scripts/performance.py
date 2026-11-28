"""Benchmark one compiled SegFormer-B0 MPK with Neat Core on an ADE20K image and save its report.

The measured pipeline is what an application runs: Neat preprocessing, the MLA, and Neat's semantic-segmentation
decode, which resizes the class scores bilinearly to the 1280x720 frame and returns one class index per pixel.
"""

import argparse
import random
from pathlib import Path

import model_report
import pyneat
from model_inputs.frame import FRAME_HEIGHT, FRAME_WIDTH, letterbox_to_720p, read_rgb

DATASET = "ade20k-val"
IMAGES = "images/validation"


def sample_image(dataset: Path) -> Path:
    """One validation image, chosen with a fixed seed so every run measures the same input."""
    root = dataset / DATASET if (dataset / DATASET).is_dir() else dataset
    images = sorted((root / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{root} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
parser.add_argument("--output", required=True, help="directory for model-report.json")
parser.add_argument("--resources", type=Path, help="package resource directory (license only)")
args = parser.parse_args()
dataset_id, dataset_dir = args.dataset.split("=", 1)
image = sample_image(Path(dataset_dir))

options = pyneat.ModelOptions()
# Neat stretches the frame to 512x512, normalizes, quantizes and tessellates it inside the measured pipeline.
options.preprocess.kind = pyneat.InputKind.Image
options.preprocess.enable = pyneat.AutoFlag.On
# Size Neat's input buffers for the benchmark frame; otherwise the execution plan shows its 1920x1080 default.
options.preprocess.input_max_width = FRAME_WIDTH
options.preprocess.input_max_height = FRAME_HEIGHT
options.preprocess.resize.enable = pyneat.AutoFlag.On
options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
options.preprocess.normalize.enable = pyneat.AutoFlag.On
# Without a preset Neat only scales pixels to [0, 1]; the model expects ImageNet mean and std.
options.preprocess.preset = pyneat.NormalizePreset.ImageNet
# Neat replaces the logits' dequantize or cast with the class-map decode: it reads the 128x128 scores in place,
# resizes them bilinearly to the frame and returns a 1x720x1280 UINT8 map with one ADE20K class index per pixel.
options.semantic_segmentation.enable = True
options.semantic_segmentation.output_width = FRAME_WIDTH
options.semantic_segmentation.output_height = FRAME_HEIGHT
model = pyneat.Model(args.mpk, options)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_mla_layer_stats = True
benchmark_options.include_mla_stage_stats = True
frame = letterbox_to_720p(read_rgb(image))
benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
postprocess = model.info().selection.selected_post_kind
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=(
        f"{dataset_id} {image.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing stretched to "
        f"512x512, {postprocess} post-processing to a {FRAME_WIDTH}x{FRAME_HEIGHT} class map"
    ),
)
report.save(args.output)
