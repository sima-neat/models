"""Measure SuperPoint's keypoint repeatability and homography accuracy with its MPK on a Modalix board.

The protocol is the SuperPoint paper's HPatches evaluation (Table 4). In each of the 106 hpatches-val sequences,
image 1 is paired with images 2 to 6 and their dataset homographies H_1_k. Every image is resized to the 640x480 input
and each homography rescaled with it. Neat's SuperPoint decoder keeps at most 1000 keypoints per image with Magic
Leap's demo settings, as performance.py does. A keypoint is repeated when the other image has one within 3 pixels of
its projection. The homography is estimated with OpenCV's RANSAC from each image-1 keypoint's nearest neighbour by
descriptor, and is correct when the image corners land within 3 pixels on average.
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
from keypoints import HEIGHT, WIDTH, decode_image, preprocess

SPLIT = "val"
SEQUENCES = 106
TARGETS = range(2, 7)
TOP_K = 1000
PIXEL_TOLERANCE = 3.0
MIN_HOMOGRAPHY_MATCHES = 4
METRICS = ("keypoint_repeatability", "homography_accuracy")
# Sequences decode and resize on worker threads while the MPK runs.
WORKERS = 4
PREFETCH = 8


def tensor_options():
    """PyNeat options for the 640x480 grayscale input this script prepares, decoded with Magic Leap's demo settings."""
    import pyneat

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized the image and made it grayscale in [0, 1]; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    options.decode_type = pyneat.BoxDecodeType.SuperPoint
    # Magic Leap's demo decode: threshold 0.015, NMS radius 4 and a 4-pixel border, keeping the paper's 1000 keypoints.
    options.superpoint.profile = pyneat.SuperPointProfile.MagicLeapDemoV1
    options.top_k = TOP_K
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


def sequence_dirs(dataset: Path) -> list[Path]:
    sequences = sorted(path for path in (dataset / SPLIT).iterdir() if path.is_dir())
    if len(sequences) != SEQUENCES:
        raise SystemExit(f"expected {SEQUENCES} {SPLIT} sequences in {dataset}, found {len(sequences)}")
    return sequences


def load_sequence(directory: Path) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Image 1's model input, and each target's model input with the homography from image 1, in input pixels."""

    def model_input(index: int) -> tuple[np.ndarray, np.ndarray]:
        image = decode_image((directory / f"{index}.ppm").read_bytes())
        height, width = image.shape[:2]
        return preprocess(image), np.diag([WIDTH / width, HEIGHT / height, 1.0])

    reference, reference_scale = model_input(1)
    targets = []
    for index in TARGETS:
        target, scale = model_input(index)
        matrix = np.loadtxt(directory / f"H_1_{index}")
        targets.append((target, scale @ matrix @ np.linalg.inv(reference_scale)))
    return reference, targets


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return projected[:, :2] / projected[:, 2:]


def _inside(points: np.ndarray) -> np.ndarray:
    return (points[:, 0] >= 0) & (points[:, 0] <= WIDTH - 1) & (points[:, 1] >= 0) & (points[:, 1] <= HEIGHT - 1)


def repeatability(keypoints0: np.ndarray, keypoints1: np.ndarray, matrix: np.ndarray) -> float:
    """Share of the keypoints both images can see that have a keypoint of the other image within the tolerance."""
    projected0 = project(keypoints0, matrix)
    projected1 = project(keypoints1, np.linalg.inv(matrix))
    seen0, seen1 = projected0[_inside(projected0)], keypoints1[_inside(projected1)]
    if not len(seen0) or not len(seen1):
        return 0.0
    distances = np.linalg.norm(seen0[:, None] - seen1[None], axis=2)
    repeated = (distances.min(1) <= PIXEL_TOLERANCE).sum() + (distances.min(0) <= PIXEL_TOLERANCE).sum()
    return float(repeated / (len(seen0) + len(seen1)))


def homography_correct(points0: np.ndarray, points1: np.ndarray, matrix: np.ndarray) -> bool:
    """Estimate the homography from matched points with OpenCV's RANSAC and check where it puts the image corners."""
    import cv2

    if len(points0) < MIN_HOMOGRAPHY_MATCHES:
        return False
    # RANSAC samples at random; a fixed seed makes every run score the same.
    cv2.setRNGSeed(0)
    estimate, _ = cv2.findHomography(points0.astype(np.float64), points1.astype(np.float64), cv2.RANSAC)
    if estimate is None:
        return False
    corners = np.array([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]], dtype=np.float64)
    error = np.linalg.norm(project(corners, estimate) - project(corners, matrix), axis=1).mean()
    return bool(error <= PIXEL_TOLERANCE)


def score_pair(features0: tuple, features1: tuple, matrix: np.ndarray) -> dict[str, float]:
    """Score the decoded features of one pair against its homography."""
    keypoints0, _, descriptors0 = features0
    keypoints1, _, descriptors1 = features1
    if len(keypoints1):
        nearest = (descriptors0 @ descriptors1.T).argmax(1)
        correct = homography_correct(keypoints0, keypoints1[nearest], matrix)
    else:
        correct = False
    return {
        "keypoint_repeatability": repeatability(keypoints0, keypoints1, matrix),
        "homography_accuracy": float(correct),
        "keypoints": (len(keypoints0) + len(keypoints1)) / 2,
    }


def evaluate(sequences: Iterator, count: int, features_of: Callable[[np.ndarray], tuple]) -> tuple[dict, float]:
    """Score `count` sequences from load_sequence() with features_of(image) -> decoded features.

    Returns the mean metrics over every pair and the mean keypoint count per image.
    """
    rows = []
    started = time.monotonic()
    for index, (reference, targets) in enumerate(sequences, start=1):
        features0 = features_of(reference)
        rows.extend(score_pair(features0, features_of(target), matrix) for target, matrix in targets)
        if index == 1 or index % 25 == 0 or index == count:
            log(f"{index}/{count} sequences, {(time.monotonic() - started) / index:.1f} s/sequence")
    metrics = {name: float(np.mean([row[name] for row in rows])) for name in METRICS}
    return metrics, round(float(np.mean([row["keypoints"] for row in rows])), 1)


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def report(dataset_id: str, metrics: dict, sequences: int, keypoints: float, seconds: float) -> model_report.Report:
    return model_report.from_accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": SPLIT,
            "evaluated_samples": sequences * len(TARGETS),
            "available_samples": SEQUENCES * len(TARGETS),
            "details": {"keypoints_per_image": keypoints, "seconds": round(seconds, 1)},
        },
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="hpatches-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first sequences only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    sequences = sequence_dirs(Path(dataset_dir))[: args.limit]

    # PyNeat runs only on the board; fp32_baseline_accuracy.py imports this script on a host.
    import pyneat

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None

    def features_of(image: np.ndarray) -> tuple:
        nonlocal runner
        # Neat takes the image as [H, W, 1]; Core 0.5.0 refuses a batch axis on BF16 builds.
        tensor = pyneat.Tensor.from_numpy(np.ascontiguousarray(image[:, :, None]), copy=False)
        if runner is None:
            runner = model.build([tensor])
        features = pyneat.decode_superpoint(runner.run([tensor], timeout_ms=120000))[0]
        return tuple(part.to_numpy(copy=True) for part in (features.keypoints, features.scores, features.descriptors))

    started = time.monotonic()
    metrics, keypoints = evaluate(prefetched(load_sequence, sequences), len(sequences), features_of)
    result = report(dataset_id, metrics, len(sequences), keypoints, time.monotonic() - started)
    result.save(args.output)
    measured = result.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
