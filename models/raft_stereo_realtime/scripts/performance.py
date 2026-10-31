"""Benchmark one compiled RAFT-Stereo realtime MPK with Neat Core on a fixed side-by-side stereo frame and save its
report.

The script builds its input: a seeded, textured 540x1920 RGB frame holding a rectified pair side by side, [left |
right], the size of a Scene Flow pair and of the model input. The right view sees the background 8 pixels and a nearer
block 48 pixels further left. Neat's image route applies the ImageNet mean and std inside the measured pipeline.
"""

import argparse

import model_report
import numpy as np
import pyneat

VIEW_HEIGHT, VIEW_WIDTH = 540, 960
FAR, NEAR = 8, 48  # disparities, in pixels, of the background and of the block


def stereo_frame() -> np.ndarray:
    """The fixed [left | right] RGB frame: a smooth seeded texture, with a block nearer the camera."""
    random = np.random.default_rng(0)
    coarse = random.integers(0, 256, (VIEW_HEIGHT // 8 + 1, (VIEW_WIDTH + FAR) // 8 + 1, 3), dtype=np.uint8)
    texture = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)[:VIEW_HEIGHT]
    block = random.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    # A point at column x of the left view appears at column x - disparity of the right view.
    left = texture[:, :VIEW_WIDTH].copy()
    right = texture[:, FAR : FAR + VIEW_WIDTH].copy()
    left[150:390, 400:720] = block
    right[150:390, 400 - NEAR : 720 - NEAR] = block
    return np.ascontiguousarray(np.concatenate([left, right], axis=1))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()

options = pyneat.ModelOptions()
# Neat normalizes, quantizes and tessellates the frame inside the measured pipeline; at the model's own 540x1920,
# its stretch resize leaves the frame as it is.
options.preprocess.kind = pyneat.InputKind.Image
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
frame = stereo_frame()
benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=(
        f"a fixed {frame.shape[1]}x{frame.shape[0]} RGB side-by-side stereo frame (seeded texture), "
        "Neat preprocessing at the model input"
    ),
)
report.save(args.output)
