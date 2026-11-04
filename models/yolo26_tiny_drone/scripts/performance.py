"""Benchmark Tiny Drone's complete image route on a seeded thermal dataset image.

This speed-only workload includes preprocessing and the one-class decoder.
The separate accuracy stage evaluates the full held-out dataset.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import cv2
import model_report
import numpy as np
import pyneat

FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def sample_image(dataset: Path) -> Path:
    """Choose a verified image from the versioned dataset's stable membership."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema"), manifest.get("schema_version")) != ("sima.anti-uav-evaluation", 1):
        raise ValueError("expected a versioned Anti-UAV evaluation dataset")
    images = manifest.get("images", [])
    if not images or manifest.get("counts", {}).get("images") != len(images):
        raise ValueError("evaluation dataset image membership is empty or incomplete")
    names = [image["file_name"] for image in images]
    if len(names) != len(set(names)):
        raise ValueError("evaluation dataset contains duplicate image paths")
    selected = random.Random(0).choice(sorted(images, key=lambda image: image["file_name"]))
    relative = PurePosixPath(selected["file_name"])
    path = dataset / str(relative)
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("evaluation dataset image path escapes its directory")
    payload = path.read_bytes()
    if len(payload) != selected["size_bytes"] or hashlib.sha256(payload).hexdigest() != selected["sha256"]:
        raise ValueError(f"evaluation dataset image identity differs: {relative}")
    return path


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"could not decode dataset image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Prepare a camera-sized RGB input before the timed model image route."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def model_options():
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # COCO_YOLO supplies centered letterbox with fill114 and RGB / 255; size
    # remains model-derived (1600), not the ordinary YOLO26 640px default.
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.num_classes = 1
    options.score_threshold = 0.05
    options.nms_iou_threshold = 0.60
    options.top_k = 100
    return options


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="dataset-source-id=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args(argv)
    dataset_id, separator, directory = args.dataset.partition("=")
    if not separator or not dataset_id or not directory:
        parser.error("--dataset must be dataset-source-id=<extracted directory>")
    image = sample_image(Path(directory))
    frame = letterbox_to_720p(read_rgb(image))
    model = pyneat.Model(args.mpk, model_options())
    benchmark_options = pyneat.BenchmarkOptions()
    benchmark_options.num_samples = 1000
    benchmark_options.min_duration_ms = 60000
    benchmark_options.include_mla_layer_stats = True
    benchmark_options.include_mla_stage_stats = True
    benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
    postprocess = model.info().selection.selected_post_kind
    report = model_report.from_performance(
        model.benchmark(benchmark_options),
        {"model": model},
        workload=(
            f"{dataset_id} {image.name} in a 1280x720 RGB frame, Neat COCO_YOLO preprocessing, "
            f"YoloV26 one-class decoder, top100 above0.05, NMS0.60, {postprocess} post-processing"
        ),
    )
    report.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
