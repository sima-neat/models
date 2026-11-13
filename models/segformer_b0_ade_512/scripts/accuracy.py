"""Measure SegFormer-B0's semantic segmentation on ADE20K's validation split with its MPK on a board.

Each original-size image enters Neat's preprocessing on the board: centered letterbox to 640x480 with RGB 114 padding
and ImageNet normalization. Scoring maps each original pixel through the unpadded image region into the logits,
using bilinear interpolation with align_corners=False, then takes its class argmax. Label 0 (unlabeled or other) is
ignored and labels 1 to 150 are the classes. mIoU averages each class's intersection over union over the whole split,
including classes present in either the labels or predictions.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from calibration import IMAGE_HEIGHT, IMAGE_WIDTH, PAD_VALUE, letterbox_geometry

DATASET = "ade20k-val"
SPLIT = "validation"
IMAGES = 2000
CLASS_COUNT = 150
LOGITS_HEIGHT, LOGITS_WIDTH = IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4
# Rows of the full-size image interpolated at once, which bounds each scorer's buffer to classes x ROWS x width.
ROWS = 64
TIMEOUT_MS = 120000
QUEUE_DEPTH = 4
# Images decode on worker threads while the MPK runs, and scorers interpolate the logits of earlier images meanwhile;
# Pillow and NumPy release the GIL while they work.
WORKERS, PREFETCH, SCORERS = 4, 8, 6


def _interpolation(source: int, target: int, offset: int, extent: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map original pixel centers into the unpadded region of the quarter-resolution logits."""
    position = np.maximum((offset + (np.arange(target) + 0.5) * extent / target) / 4 - 0.5, 0.0)
    low = np.minimum(np.floor(position).astype(np.int64), source - 1)
    high = np.minimum(low + 1, source - 1)
    return low, high, (position - low).astype(np.float32)


def candidate_classes(logits: np.ndarray) -> np.ndarray:
    """The classes whose interpolated logit can be the largest at some pixel, in ascending order.

    Every interpolated value is a weighted average of the four logits of the cell around it. In a cell, some class
    reaches at least the largest of the classes' smallest corners, and a class whose largest corner is below that
    cannot win there. Interpolating only the other classes therefore gives the same argmax, ties included.
    """
    a, b, c, d = logits[:, :-1, :-1], logits[:, 1:, :-1], logits[:, :-1, 1:], logits[:, 1:, 1:]
    floor = np.minimum(np.minimum(a, b), np.minimum(c, d)).max(axis=0)
    ceiling = np.maximum(np.maximum(a, b), np.maximum(c, d))
    return np.flatnonzero((ceiling >= floor).any(axis=(1, 2)))


def segment(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    """Each original pixel's class, excluding letterbox padding before taking the argmax."""
    if logits.shape != (CLASS_COUNT, LOGITS_HEIGHT, LOGITS_WIDTH) or not np.isfinite(logits).all():
        raise ValueError(f"expected finite {CLASS_COUNT}x{LOGITS_HEIGHT}x{LOGITS_WIDTH} logits, got {logits.shape}")
    candidates = candidate_classes(logits.astype(np.float32, copy=False))
    if candidates.size == 1:
        return np.full((height, width), candidates[0], np.uint8)
    logits = logits[candidates].astype(np.float32, copy=False)
    pad_left, pad_top, scaled_width, scaled_height = letterbox_geometry(width, height)
    left, right, across = _interpolation(LOGITS_WIDTH, width, pad_left, scaled_width)
    columns = logits[:, :, left] * (1 - across) + logits[:, :, right] * across
    top, bottom, down = _interpolation(LOGITS_HEIGHT, height, pad_top, scaled_height)
    classes = np.empty((height, width), np.uint8)
    for start in range(0, height, ROWS):
        rows = slice(start, start + ROWS)
        weight = down[rows][None, :, None]
        block = columns[:, top[rows]] * (1 - weight) + columns[:, bottom[rows]] * weight
        classes[rows] = candidates[block.argmax(axis=0)]
    return classes


def confusion(classes: np.ndarray, label: np.ndarray) -> np.ndarray:
    """The CLASS_COUNT x CLASS_COUNT counts of (true, predicted) classes over the labelled pixels."""
    if classes.shape != label.shape:
        raise ValueError(f"prediction {classes.shape} and label {label.shape} differ in size")
    labelled = label > 0
    truth = label[labelled].astype(np.int64) - 1
    if truth.size and truth.max() >= CLASS_COUNT:
        raise ValueError(f"label values must be 0 to {CLASS_COUNT}")
    pairs = truth * CLASS_COUNT + classes[labelled]
    return np.bincount(pairs, minlength=CLASS_COUNT * CLASS_COUNT).reshape(CLASS_COUNT, CLASS_COUNT)


def scores(counts: np.ndarray) -> dict[str, float]:
    """mIoU over the classes present in the labels or predictions, and the labelled pixels' accuracy."""
    intersection = np.diag(counts).astype(np.float64)
    union = counts.sum(axis=0) + counts.sum(axis=1) - intersection
    present = union > 0
    return {
        "miou": float(np.mean(intersection[present] / union[present])),
        "pixel_accuracy": float(intersection.sum() / counts.sum()),
        "classes": int(present.sum()),
    }


def channels_first(output: np.ndarray) -> np.ndarray:
    """The logits as CHW, from a channels-last or channels-first output with or without its batch axis."""
    logits = np.asarray(output)
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.shape == (LOGITS_HEIGHT, LOGITS_WIDTH, CLASS_COUNT):
        return logits.transpose(2, 0, 1)
    if logits.shape == (CLASS_COUNT, LOGITS_HEIGHT, LOGITS_WIDTH):
        return logits
    raise ValueError(f"unexpected logits shape {np.shape(output)}")


def samples(directory: Path) -> tuple[Path, list[Path]]:
    """The dataset root and its validation images, sorted, each with a label map of the same name."""
    root = directory / DATASET if (directory / DATASET).is_dir() else directory
    images = sorted((root / "images" / SPLIT).glob("*.jpg"))
    labels = sorted((root / "annotations" / SPLIT).glob("*.png"))
    if [image.stem for image in images] != [label.stem for label in labels]:
        raise SystemExit(f"{root} does not pair each validation image with a label map")
    return root, images


def label_map(root: Path, image: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(root / "annotations" / SPLIT / f"{image.stem}.png") as source:
        if source.mode != "L":
            raise ValueError(f"expected an 8-bit label map for {image.name}, got mode {source.mode}")
        return np.asarray(source)


def score_image(root: Path, image: Path, logits: np.ndarray) -> np.ndarray:
    """One image's confusion counts, from its logits in either layout."""
    label = label_map(root, image)
    return confusion(segment(channels_first(logits), *label.shape), label)


def read_rgb(path: Path) -> np.ndarray:
    """The original RGB pixels, never resized before the measured native pipeline."""
    from PIL import Image

    with Image.open(io.BytesIO(path.read_bytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def accumulate(root: Path, images: Sequence[Path], outputs: Iterator[np.ndarray], log_every: int = 250) -> np.ndarray:
    """Score each image's logits on the scorer threads as they arrive, and sum the confusion counts."""
    counts = np.zeros((CLASS_COUNT, CLASS_COUNT), np.int64)
    started = time.monotonic()
    with ThreadPoolExecutor(SCORERS) as scorers:
        pending: deque = deque()
        for count, (image, logits) in enumerate(zip(images, outputs, strict=True), start=1):
            pending.append(scorers.submit(score_image, root, image, logits))
            while len(pending) > 2 * SCORERS:
                counts += pending.popleft().result()
            if count == 1 or count % log_every == 0 or count == len(images):
                elapsed_ms = (time.monotonic() - started) / count * 1000
                print(f"{count}/{len(images)} images, {elapsed_ms:.0f} ms/image", flush=True)
        while pending:
            counts += pending.popleft().result()
    return counts


def parse_mpks(values: list[str]) -> Path:
    mpks = [value.split("=", 1) for value in values]
    if len(mpks) != 1 or len(mpks[0]) != 2 or mpks[0][0] != "model" or not mpks[0][1]:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks[0][1])


def parse_dataset(parser: argparse.ArgumentParser, value: str, limit: int | None) -> tuple[Path, list[Path], int]:
    dataset_id, _, directory = value.partition("=")
    if dataset_id != DATASET or not directory:
        parser.error(f"--dataset must name {DATASET}=<directory>")
    root, available = samples(Path(directory))
    if len(available) != IMAGES:
        parser.error(f"expected all {IMAGES} ADE20K validation images, found {len(available)}")
    if limit is not None and not 1 <= limit <= len(available):
        parser.error(f"--limit must be between 1 and {len(available)}")
    return root, available[:limit], len(available)


def model_logits(path: Path, images: Sequence[Path]) -> Iterator[np.ndarray]:
    """Run each original-size image through Neat's preprocessing and the MPK, yielding a copy of its logits."""
    import pyneat
    from PIL import Image

    sizes = []
    for image in images:
        with Image.open(image) as source:
            sizes.append(source.size)
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = max(width for width, _ in sizes)
    options.preprocess.input_max_height = max(height for _, height in sizes)
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.resize.pad_value = PAD_VALUE
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    model = pyneat.Model(str(path), options)
    runner = None
    try:
        for rgb in prefetched(read_rgb, images):
            tensor = pyneat.Tensor.from_numpy(
                rgb, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
            )
            if runner is None:
                runner = model.build([tensor], run_options=run_options)
            (output,) = runner.run([tensor], timeout_ms=TIMEOUT_MS)
            # The scorers read the logits after the next run reuses the output buffer, so they get a copy.
            yield output.to_numpy(copy=True)
    finally:
        if runner is not None:
            runner.close()


def main(argv: list[str] | None = None) -> int:
    import model_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="package resource directory (license only)")
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    args = parser.parse_args(argv)
    root, images, available = parse_dataset(parser, args.dataset, args.limit)

    started = time.monotonic()
    counts = accumulate(root, images, model_logits(parse_mpks(args.mpk), images))
    seconds = time.monotonic() - started
    measured = scores(counts)
    report = model_report.from_accuracy(
        {
            DATASET: {
                "metrics": {"miou": measured["miou"]},
                "split": SPLIT,
                "evaluated_samples": len(images),
                "available_samples": available,
                "details": {
                    "pixel_accuracy": round(measured["pixel_accuracy"], 4),
                    "classes_present": measured["classes"],
                    "protocol": "Neat preprocessing (640x480 letterbox, RGB 114 padding, ImageNet), single scale, "
                    "logits mapped bilinearly through the unpadded region to the label size",
                    "inference_seconds": round(seconds, 1),
                },
            },
        }
    )
    report.save(args.output)
    print(json.dumps({"miou": round(measured["miou"], 4), "pixel_accuracy": round(measured["pixel_accuracy"], 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
