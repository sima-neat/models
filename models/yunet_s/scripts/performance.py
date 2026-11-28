"""Benchmark one compiled YuNet-s MPK with Neat Core on a WIDER FACE image and save its report.

The measured pipeline is what an application runs: Neat letterboxes the frame to 640x640 with zero padding, converts
it to BGR and keeps its pixel values, and the MLA returns the 12 raw head maps. Neat has no face decoder, so the
application decodes boxes and landmarks on the host, outside this measurement.
"""

import argparse
import random
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

DATASET = "wider-face-val"
IMAGES = "WIDER_val/images"
IMAGE_SIZE = 640
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def sample_image(dataset: Path) -> Path:
    """One validation image, chosen with a fixed seed so every run measures the same input."""
    root = dataset / DATASET if (dataset / DATASET).is_dir() else dataset
    images = sorted((root / IMAGES).glob("*/*.jpg"))
    if not images:
        raise SystemExit(f"{root} has no images under {IMAGES}/")
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
parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
parser.add_argument("--output", required=True, help="directory for model-report.json")
parser.add_argument("--resources", type=Path, help="package resource directory (license only)")
args = parser.parse_args()
dataset_id, dataset_dir = args.dataset.split("=", 1)
image = sample_image(Path(dataset_dir))

options = pyneat.ModelOptions()
# Neat letterboxes the frame to the model input, converts it to BGR and quantizes it inside the measured pipeline.
options.preprocess.kind = pyneat.InputKind.Image
options.preprocess.input_max_width = FRAME_WIDTH
options.preprocess.input_max_height = FRAME_HEIGHT
options.preprocess.enable = pyneat.AutoFlag.On
options.preprocess.resize.enable = pyneat.AutoFlag.On
options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
options.preprocess.resize.width = options.preprocess.resize.height = IMAGE_SIZE
# YuNet's padding is zero, as OpenCV's FaceDetectorYN pads its input.
options.preprocess.resize.pad_value = 0
options.preprocess.color_convert.enable = pyneat.AutoFlag.On
options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.BGR
# GenericPreproc normalizes (pixel / 255 - mean) / std; this keeps YuNet's raw pixel domain.
options.preprocess.normalize.enable = pyneat.AutoFlag.On
options.preprocess.normalize.mean = [0.0, 0.0, 0.0]
options.preprocess.normalize.stddev = [1.0 / 255] * 3
options.preprocess.normalize.has_explicit_stats = True
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
        f"{dataset_id} {image.parent.name}/{image.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat letterbox "
        f"to {IMAGE_SIZE}x{IMAGE_SIZE} with zero padding and BGR pixel values, the 12 raw head maps with "
        f"{postprocess} post-processing and no face decoder"
    ),
)
report.save(args.output)
