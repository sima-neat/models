"""Measure Depth Anything V2 Small's relative depth on NYU Depth V2's official test split with its MPK on a board.

The protocol is the one Depth Anything's NYU-D numbers follow (MiDaS): the prediction is interpolated to the ground
truth's 640x480, aligned to it per image by least-squares scale and shift in inverse depth, capped at 10 m, and scored
in depth on every pixel with a measurement, as AbsRel and delta1 averaged over the images.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import HEIGHT, VALIDATION_SPLIT, WIDTH, preprocess_image

DEPTH_DIR = "annotations/depth_test"
# Ground truth is 16-bit millimetres, 0 where the sensor measured nothing, and NYU depth ends at 10 m.
DEPTH_SCALE = 1000.0
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0
DELTA = 1.25
# Images and depth maps load on worker threads while the MPK runs.
WORKERS = 4
PREFETCH = 8


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized and normalized the image; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def resize_align_corners(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize with align_corners=True, as the authors' inference interpolates the prediction."""
    rows = np.linspace(0, image.shape[0] - 1, height)
    columns = np.linspace(0, image.shape[1] - 1, width)
    top = np.floor(rows).astype(np.int64)
    left = np.floor(columns).astype(np.int64)
    bottom = np.minimum(top + 1, image.shape[0] - 1)
    right = np.minimum(left + 1, image.shape[1] - 1)
    down = (rows - top)[:, None]
    across = (columns - left)[None, :]
    upper = image[top][:, left] * (1 - across) + image[top][:, right] * across
    lower = image[bottom][:, left] * (1 - across) + image[bottom][:, right] * across
    return upper * (1 - down) + lower * down


def score(disparity: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """AbsRel and delta1 of a relative disparity map against metric depth, after scale-and-shift alignment."""
    prediction = resize_align_corners(np.asarray(disparity, np.float64), *depth.shape)
    valid = (depth > MIN_DEPTH) & (depth <= MAX_DEPTH)
    if not valid.any():
        raise ValueError("ground truth has no valid depth")
    truth = depth[valid].astype(np.float64)
    # Least squares for scale and shift in inverse depth, then depth capped at MAX_DEPTH.
    design = np.stack([prediction[valid], np.ones(truth.size)], axis=1)
    scale, shift = np.linalg.lstsq(design, 1.0 / truth, rcond=None)[0]
    predicted = 1.0 / np.maximum(scale * prediction[valid] + shift, 1.0 / MAX_DEPTH)
    abs_rel = float(np.mean(np.abs(predicted - truth) / truth))
    delta1 = float(np.mean(np.maximum(predicted / truth, truth / predicted) < DELTA))
    return abs_rel, delta1


def load_sample(root: Path, image: Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    depth_path = root / DEPTH_DIR / image.name.replace("rgb_", "sync_depth_").replace(".jpg", ".png")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError(f"cannot read 16-bit depth {depth_path}")
    return preprocess_image(image.read_bytes()), depth.astype(np.float32) / np.float32(DEPTH_SCALE)


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="nyu-depth-v2-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    root = Path(dataset_dir)

    available = sorted((root / VALIDATION_SPLIT).glob("rgb_*.jpg"))
    if not available:
        raise SystemExit(f"{root} has no images under {VALIDATION_SPLIT}/")
    images = available[: args.limit] if args.limit else available

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    totals = np.zeros(2)
    started = time.monotonic()
    for count, (image, (tensor_data, depth)) in enumerate(
        zip(images, prefetched(lambda path: load_sample(root, path), images)), start=1
    ):
        tensor = pyneat.Tensor.from_numpy(tensor_data, copy=False)
        if runner is None:
            runner = model.build([tensor])
        output = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False)
        if output.size != HEIGHT * WIDTH or not np.isfinite(output).all():
            raise ValueError(f"unexpected depth for {image.name}: shape {output.shape}")
        totals += score(output.reshape(HEIGHT, WIDTH), depth)
        if count == 1 or count % 100 == 0 or count == len(images):
            log(f"{count}/{len(images)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")

    abs_rel, delta1 = totals / len(images)
    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": {"abs_rel": abs_rel, "delta1": delta1},
                "split": VALIDATION_SPLIT,
                "evaluated_samples": len(images),
                "available_samples": len(available),
                "details": {"inference_seconds": round(time.monotonic() - started, 1)},
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
