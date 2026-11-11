"""Benchmark one compiled RAFT-Stereo realtime MPK with Neat Core on a Scene Flow test pair and save its report.

The input is one pair of the Scene Flow finalpass test list, chosen with random.Random(0): its left and right views,
read as RGB and joined side by side into one 1920x540 frame, the model's own input size. Neat's image route stretches
the frame to the model input, which leaves it as it is, and applies the ImageNet mean and std inside the measured
pipeline.
"""

import argparse
import random
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

# Each line names a pair's left view, right view and left-view disparity, relative to the dataset directory.
TEST_LIST = "sceneflow_finalpass_test.txt"


def sample_pair(dataset: Path) -> tuple[str, str]:
    """One test pair's left and right views, chosen with a fixed seed so every run measures the same input."""
    lines = (dataset / TEST_LIST).read_text(encoding="utf-8").splitlines()
    pairs = [line.split()[:2] for line in lines if line.strip()]
    if not pairs:
        raise SystemExit(f"{dataset / TEST_LIST} lists no pairs")
    left, right = random.Random(0).choice(pairs)
    return left, right


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--dataset", required=True, help="scene-flow-val=<extracted directory>")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()
dataset_id, dataset_dir = args.dataset.split("=", 1)
dataset = Path(dataset_dir)
left, right = sample_pair(dataset)
# The pair as one RGB frame: the left view, then the right view.
frame = np.concatenate([read_rgb(dataset / left), read_rgb(dataset / right)], axis=1)

options = pyneat.ModelOptions()
# Neat resizes, normalizes, quantizes and tessellates the frame inside the measured pipeline.
options.preprocess.kind = pyneat.InputKind.Image
# Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
options.preprocess.input_max_width = frame.shape[1]
options.preprocess.input_max_height = frame.shape[0]
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
benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=(
        f"{dataset_id} pair {left} and its right view, side by side in a {frame.shape[1]}x{frame.shape[0]} RGB "
        "frame, Neat preprocessing stretched to the model input"
    ),
)
report.save(args.output)
