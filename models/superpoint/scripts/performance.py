"""Benchmark one compiled SuperPoint MPK with Neat's SuperPoint decoder on a COCO image and save its report."""

import argparse
import random
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()
dataset_id, dataset_dir = args.dataset.split("=", 1)
image = sample_image(Path(dataset_dir))

options = pyneat.ModelOptions()
# Neat stretches the frame to the 640x480 input, makes it grayscale and scales it to [0, 1], as SuperPoint takes it,
# then quantizes and tessellates it, inside the measured pipeline.
options.preprocess.kind = pyneat.InputKind.Image
options.preprocess.enable = pyneat.AutoFlag.On
options.preprocess.resize.enable = pyneat.AutoFlag.On
options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
options.preprocess.color_convert.enable = pyneat.AutoFlag.On
options.preprocess.color_convert.input_format = pyneat.PixelFormat.RGB
options.preprocess.color_convert.output_format = pyneat.PixelFormat.GRAY8
options.preprocess.normalize.enable = pyneat.AutoFlag.On
options.decode_type = pyneat.BoxDecodeType.SuperPoint
# Magic Leap's demo settings: keypoint threshold 0.015, NMS radius 4, a 4-pixel border.
options.superpoint.profile = pyneat.SuperPointProfile.MagicLeapDemoV1
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
        f"640x480 grayscale, SuperPoint decoder (MagicLeapDemoV1), {postprocess} post-processing"
    ),
)
report.save(args.output)
