"""Benchmark one compiled SegFormer-B0 MPK with Neat Core on a fixed 720p frame and save its report."""

import argparse

import model_report
import numpy as np
import pyneat

# The model declares no accuracy dataset, so the benchmark input is a fixed, seeded 720p camera-like frame:
# smooth colour gradients with sensor-like noise.
FRAME_WIDTH, FRAME_HEIGHT, SEED = 1280, 720, 0


def synthetic_frame() -> np.ndarray:
    rows = np.linspace(0, 1, FRAME_HEIGHT, dtype=np.float32)[:, None]
    columns = np.linspace(0, 1, FRAME_WIDTH, dtype=np.float32)[None, :]
    channels = [rows * 200 + 30, columns * 180 + 40, (1 - rows) * columns * 220 + 20]
    base = np.stack([np.broadcast_to(channel, (FRAME_HEIGHT, FRAME_WIDTH)) for channel in channels], axis=-1)
    noise = np.random.default_rng(SEED).normal(0, 8, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for model-report.json")
args = parser.parse_args()

options = pyneat.ModelOptions()
# Neat stretches the frame to 512x512, normalizes, quantizes and tessellates it inside the measured pipeline.
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
benchmark_options.inputs = [pyneat.Tensor.from_numpy(synthetic_frame(), copy=True, image_format=pyneat.PixelFormat.RGB)]
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=(
        f"Seeded synthetic {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing stretched to 512x512, "
        "raw 128x128 logits"
    ),
)
report.save(args.output)
