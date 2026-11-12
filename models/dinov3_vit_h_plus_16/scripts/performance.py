"""Benchmark one compiled DINOv3 ViT-H+/16 MPK with Neat Core on a fixed 720p frame and save its report.

The model declares no accuracy dataset, so the script builds its own input: a seeded outdoor scene, a sky-to-field
gradient with soft shapes and sensor noise, the same on every run.
"""

import argparse

import model_report
import numpy as np
import pyneat

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720


def scene() -> np.ndarray:
    """A fixed 1280x720 RGB frame: sky above a field, with seeded soft shapes and mild noise."""
    rng = np.random.default_rng(0)
    rows = np.linspace(0.0, 1.0, FRAME_HEIGHT, dtype=np.float32)[:, None, None]
    sky = np.asarray([120, 170, 230], np.float32) * (1.0 - 0.35 * rows)
    field = np.asarray([70, 130, 60], np.float32) * (1.25 - 0.5 * rows)
    frame = np.where(rows < 0.55, sky, field) * np.ones((1, FRAME_WIDTH, 1), np.float32)
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
    for _ in range(12):
        x, y = rng.uniform(0, FRAME_WIDTH), rng.uniform(0, FRAME_HEIGHT)
        radius, colour = rng.uniform(30, 140), rng.uniform(20, 235, 3).astype(np.float32)
        weight = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * radius**2))[..., None]
        frame = frame * (1 - weight) + colour * weight
    frame += rng.normal(0.0, 3.0, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for model-report.json")
# The benchmark runner passes the package's resources, here only the license. The benchmark reads none of them.
parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
args = parser.parse_args()

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
benchmark_options.inputs = [pyneat.Tensor.from_numpy(scene(), copy=True, image_format=pyneat.PixelFormat.RGB)]
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=(
        f"a fixed {FRAME_WIDTH}x{FRAME_HEIGHT} RGB outdoor scene the script builds, "
        "Neat preprocessing stretched to the model input"
    ),
)
report.save(args.output)
