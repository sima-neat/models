"""Benchmark the complete native SSD detector on one reproducible COCO image."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def read_rgb(path: Path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image):
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    import cv2
    import numpy as np

    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


SCORE_THRESHOLD, IOU_THRESHOLD, TOP_K = 0.2, 0.6, 100


def native_options():
    import pyneat

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = [0.5, 0.5, 0.5]
    options.preprocess.normalize.stddev = [0.5, 0.5, 0.5]
    options.preprocess.normalize.has_explicit_stats = True
    options.decode_type = pyneat.BoxDecodeType.Ssd
    options.num_classes = 91
    options.score_threshold = SCORE_THRESHOLD
    options.nms_iou_threshold = IOU_THRESHOLD
    options.top_k = TOP_K
    options.boxdecode_original_width = 0
    options.boxdecode_original_height = 0
    return options


def benchmark(mpk, dataset_id, dataset_dir, output):
    import model_report
    import pyneat

    image = sample_image(Path(dataset_dir))
    model = pyneat.Model(str(mpk), native_options())
    options = pyneat.BenchmarkOptions()
    options.num_samples = 1000
    options.min_duration_ms = 60000
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    frame = letterbox_to_720p(read_rgb(image))
    options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
    report = model_report.from_performance(
        model.benchmark(options),
        {"model": model},
        workload=(
            f"{dataset_id} {image.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, "
            "native RGB stretch preprocessing with mean/stddev 0.5, native Ssd decoder, "
            f"score threshold {SCORE_THRESHOLD}, NMS IoU {IOU_THRESHOLD}, top {TOP_K}; "
            "complete detector, not raw heads"
        ),
    )
    report.save(output)


def one_argument(values, name):
    if len(values) != 1 or not values[0].strip():
        raise SystemExit(f"{name} requires exactly one nonempty value")
    return values[0]


def parse_dataset(values):
    value = one_argument(values, "--dataset")
    dataset_id, separator, directory = value.partition("=")
    if dataset_id != "coco-2017-val" or not separator or not directory.strip():
        raise SystemExit("--dataset must name coco-2017-val=<directory>")
    return dataset_id, Path(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="compiled model archive")
    parser.add_argument("--dataset", action="append", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", action="append", required=True, help="directory for model-report.json")
    args = parser.parse_args(argv)
    mpk = one_argument(args.mpk, "--mpk")
    dataset_id, dataset_dir = parse_dataset(args.dataset)
    output = one_argument(args.output, "--output")
    # Customer performance policy is distinct from the lower-threshold AP policy.
    benchmark(mpk, dataset_id, dataset_dir, output)


if __name__ == "__main__":
    main()
