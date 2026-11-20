"""Benchmark one compiled MNIST MPK with Neat Core on an MNIST test digit and save its report.

The benchmark input is one test digit, chosen with random.Random(0), as an RGB image letterboxed into a 1280x720
frame. Neat converts the frame to grayscale, stretches it to the model's 28x28 input and scales it to [0, 1] inside
the measured pipeline, so the throughput is what an application gets from a 720p frame to digit scores.
"""

import argparse
import random
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

SPLIT, IMAGES, COUNT, SIZE = "test", "t10k-images-idx3-ubyte", 10000, 28
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def sample_digit(dataset: Path) -> tuple[int, np.ndarray]:
    """One test digit, chosen with a fixed seed so every run measures the same input: its index and 28x28 pixels."""
    content = (dataset / IMAGES).read_bytes()
    if len(content) != 16 + COUNT * SIZE * SIZE:
        raise SystemExit(f"{dataset / IMAGES} is not the {COUNT} MNIST test images")
    images = np.frombuffer(content, dtype=np.uint8, offset=16).reshape(COUNT, SIZE, SIZE)
    index = random.Random(0).randrange(COUNT)
    return index, images[index]


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def image_options() -> pyneat.ModelOptions:
    """Neat converts the frame to grayscale, stretches it to 28x28 and scales it to [0, 1]; no mean or std preset."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.GRAY8
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="mnist-default=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    index, digit = sample_digit(Path(dataset_dir))

    model = pyneat.Model(args.mpk, image_options())
    # Request 1000 samples and a 60 s minimum measurement duration.
    benchmark_options = pyneat.BenchmarkOptions()
    benchmark_options.num_samples = 1000
    benchmark_options.min_duration_ms = 60000
    benchmark_options.include_mla_layer_stats = True
    benchmark_options.include_mla_stage_stats = True
    frame = letterbox_to_720p(cv2.cvtColor(digit, cv2.COLOR_GRAY2RGB))
    benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
    report = model_report.from_performance(
        model.benchmark(benchmark_options),
        {"model": model},
        workload=(
            f"{dataset_id} {SPLIT} digit {index} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing "
            f"converted to grayscale and stretched to {SIZE}x{SIZE}"
        ),
    )
    report.save(args.output)


if __name__ == "__main__":
    main()
