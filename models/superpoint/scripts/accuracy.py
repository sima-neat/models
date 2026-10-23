"""Measure SuperPoint's keypoint repeatability, matching accuracy and homography accuracy with its MPK on a Modalix
board.

Each image of a seeded sample of 1000 COCO val2017 images is paired with its own warp by a homography seeded with the
image id. The MPK runs on both images, keypoints.py decodes each with transformers' settings and keeps the best
KEYPOINTS, and the pair is scored against the known homography as in the SuperPoint paper's HPatches evaluation, at
3 pixels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
from keypoints import (
    CELL,
    DESCRIPTOR_SIZE,
    HEIGHT,
    KEYPOINT_CHANNELS,
    WIDTH,
    decode,
    decode_image,
    homography,
    preprocess,
)

IMAGES = "val2017"
AVAILABLE_IMAGES = 5000
# A seeded sample keeps the stage near 7 minutes on a DevKit; the full set takes 35.
SAMPLED_IMAGES = 1000
SAMPLE_SEED = 0
# The SuperPoint paper's HPatches evaluation keeps 1000 keypoints at 480x640.
KEYPOINTS = 1000
PIXEL_TOLERANCE = 3.0
MIN_HOMOGRAPHY_MATCHES = 4
METRICS = ("keypoint_repeatability", "matching_accuracy", "homography_accuracy")
# Image pairs decode and warp on worker threads while the MPK runs.
WORKERS = 4
PREFETCH = 8
GRID = (HEIGHT // CELL, WIDTH // CELL)


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized the image and made it grayscale in [0, 1]; Neat still quantizes and tessellates.
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


def image_pair(content: bytes, image_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two model inputs of one pair and the homography from the first to the second, in input pixels."""
    import cv2

    image = decode_image(content)
    height, width = image.shape[:2]
    warp = homography(image_id, width, height)
    warped = cv2.warpPerspective(image, warp, (width, height))
    resize = np.diag([WIDTH / width, HEIGHT / height, 1.0])
    return preprocess(image), preprocess(warped), resize @ warp @ np.linalg.inv(resize)


def heads(outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the MPK's channels-last keypoint logits and descriptors channels-first, named by their channels."""
    named = {}
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        array = array.reshape(*GRID, array.shape[-1]) if array.size == np.prod(GRID) * array.shape[-1] else array
        if array.shape[:2] != GRID:
            raise ValueError(f"unexpected MPK output: {np.shape(output)}")
        named[array.shape[-1]] = array.transpose(2, 0, 1)
    if sorted(named) != sorted((KEYPOINT_CHANNELS, DESCRIPTOR_SIZE)):
        raise ValueError(f"unexpected MPK outputs: {[np.shape(output) for output in outputs]}")
    if not all(np.isfinite(values).all() for values in named.values()):
        raise ValueError("MPK output contains non-finite values")
    return named[KEYPOINT_CHANNELS], named[DESCRIPTOR_SIZE]


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


def mutual_matches(descriptors0: np.ndarray, descriptors1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of mutual nearest neighbours between two sets of unit descriptors."""
    if not len(descriptors0) or not len(descriptors1):
        return np.zeros(0, int), np.zeros(0, int)
    similarity = descriptors0 @ descriptors1.T
    best0, best1 = similarity.argmax(1), similarity.argmax(0)
    rows = np.flatnonzero(best1[best0] == np.arange(len(best0)))
    return rows, best0[rows]


def homography_correct(points0: np.ndarray, points1: np.ndarray, matrix: np.ndarray) -> bool:
    """Estimate the homography from matched points with OpenCV's RANSAC and check where it puts the image corners."""
    import cv2

    if len(points0) < MIN_HOMOGRAPHY_MATCHES:
        return False
    # RANSAC samples at random; a fixed seed makes every run score the same.
    cv2.setRNGSeed(0)
    estimate, _ = cv2.findHomography(
        points0.astype(np.float64), points1.astype(np.float64), cv2.RANSAC, PIXEL_TOLERANCE
    )
    if estimate is None:
        return False
    corners = np.array([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]], dtype=np.float64)
    return bool(np.linalg.norm(project(corners, estimate) - project(corners, matrix), axis=1).mean() < PIXEL_TOLERANCE)


def score_pair(features0: tuple, features1: tuple, matrix: np.ndarray) -> dict[str, float]:
    """Score the decoded features of one pair against its homography."""
    keypoints0, _, descriptors0 = features0
    keypoints1, _, descriptors1 = features1
    rows, columns = mutual_matches(descriptors0, descriptors1)
    points0, points1 = keypoints0[rows], keypoints1[columns]
    errors = np.linalg.norm(project(points0, matrix) - points1, axis=1)
    return {
        "keypoint_repeatability": repeatability(keypoints0, keypoints1, matrix),
        "matching_accuracy": float((errors <= PIXEL_TOLERANCE).mean()) if len(errors) else 0.0,
        "homography_accuracy": float(homography_correct(points0, points1, matrix)),
        "keypoints": (len(keypoints0) + len(keypoints1)) / 2,
        "matches": float(len(rows)),
    }


def evaluate(pairs: Iterator, count: int, features_of: Callable[[np.ndarray], tuple]) -> tuple[dict, dict]:
    """Score `count` (image0, image1, homography) pairs with features_of(image) -> decoded features."""
    rows = []
    started = time.monotonic()
    for index, (image0, image1, matrix) in enumerate(pairs, start=1):
        rows.append(score_pair(features_of(image0), features_of(image1), matrix))
        if index == 1 or index % 250 == 0 or index == count:
            log(f"{index}/{count} pairs, {(time.monotonic() - started) / index * 1000:.0f} ms/pair")
    metrics = {name: float(np.mean([row[name] for row in rows])) for name in METRICS}
    totals = {name: round(float(np.mean([row[name] for row in rows])), 1) for name in ("keypoints", "matches")}
    return metrics, totals


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
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first sampled images only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    images = sorted((Path(dataset_dir) / IMAGES).glob("*.jpg"))
    if len(images) != AVAILABLE_IMAGES:
        raise SystemExit(f"expected {AVAILABLE_IMAGES} {IMAGES} images, found {len(images)}")
    sampled = sorted(random.Random(SAMPLE_SEED).sample(images, SAMPLED_IMAGES))
    selected = sampled[: args.limit] if args.limit else sampled

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None

    def features_of(image: np.ndarray) -> tuple:
        nonlocal runner
        # Neat takes the image as [H, W, 1]; Core 0.5.0 refuses a batch axis on BF16 builds.
        tensor = pyneat.Tensor.from_numpy(np.ascontiguousarray(image[:, :, None]), copy=False)
        if runner is None:
            runner = model.build([tensor])
        logits, descriptors = heads([output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)])
        return decode(logits, descriptors, limit=KEYPOINTS)

    started = time.monotonic()
    pairs = prefetched(lambda path: image_pair(path.read_bytes(), int(path.stem)), selected)
    metrics, totals = evaluate(pairs, len(selected), features_of)
    report = model_report.accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": IMAGES, "evaluated_samples": len(selected), "available_samples": AVAILABLE_IMAGES,
            "details": {
                "keypoints_per_image": totals["keypoints"],
                "mutual_matches_per_pair": totals["matches"],
                "seconds": round(time.monotonic() - started, 1),
            },
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
