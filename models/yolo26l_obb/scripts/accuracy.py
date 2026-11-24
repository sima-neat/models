"""Measure YOLO26l-obb DOTA-v1.0 oriented-box mAP with its MPK on a Modalix board."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
import shapely
from calibration import IMAGE_SIZE, decode_image, letterbox_geometry, preprocess_tile, tile_windows
from model_report import log

IMAGES = "val/images"
LABELS = "val/labelTxt"
VAL_IMAGES = 458
# DOTA-v1.0 classes in the checkpoint's class order, spelled as in the label files.
CLASSES = (
    "plane ship storage-tank baseball-diamond tennis-court basketball-court ground-track-field harbor "
    "bridge large-vehicle small-vehicle helicopter roundabout soccer-ball-field swimming-pool"
).split()
STRIDES = (8, 16, 32)
# Ultralytics' end-to-end selection and oriented-box validation settings: the best 300 anchor/class pairs scoring
# above 0.01, the confidence its validator sets for the obb task.
TOP_K = 300
SCORE_THRESHOLD = 0.01
# sigmoid(-4.6) is about 0.00995, below SCORE_THRESHOLD, so a logit at or below LOGIT_FLOOR is never kept and the
# decoder skips its sigmoid.
LOGIT_FLOOR = -4.6
# Ultralytics merges the tiles of an image with Fast-NMS on probabilistic IoU at 0.3, one class at a time.
MERGE_IOU = 0.3
# Fast-NMS compares at most this many boxes at a time against every higher-scoring box, to bound memory.
MERGE_BLOCK = 256
# DOTA Task 1 counts a detection as correct when its polygon IoU exceeds 0.5; dota_obb_map averages the same AP
# over IoU thresholds 0.50 to 0.95 in steps of 0.05.
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
METRICS = ("dota_obb_map50", "dota_obb_map")
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes. A decoded DOTA
# image can exceed 150 MB, which bounds the prefetch.
WORKERS = 4
PREFETCH = 4
# The MPK returns bbox_0..2, class_logit_0..2 then angle_0..2 in NHWC. Anchors are the grid-cell centres of each
# level, row-major from stride 8 to stride 32, as Ultralytics' make_anchors lays them out.
GRIDS = [IMAGE_SIZE // stride for stride in STRIDES]
OUTPUT_SHAPES = [(1, size, size, channels) for channels in (4, len(CLASSES), 1) for size in GRIDS]
ANCHORS = np.concatenate(
    [np.stack(np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5), axis=-1).reshape(-1, 2) for size in GRIDS]
)
ANCHOR_STRIDES = np.repeat(STRIDES, [size * size for size in GRIDS])[:, None]


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Indices of the count largest values, largest first, ties in index order, like a stable descending sort."""
    chosen = np.arange(len(values))
    if len(values) > count:
        cutoff = np.partition(values, len(values) - count)[len(values) - count]
        above = np.flatnonzero(values > cutoff)
        chosen = np.sort(np.concatenate([above, np.flatnonzero(values == cutoff)[: count - len(above)]]))
    return chosen[np.argsort(-values[chosen], kind="stable")]


def decode_tile(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Return rows [x, y, w, h, angle, score, class] in input pixels for one tile.

    Selects detections as the exported Ultralytics head does: the TOP_K anchors by best class score, then the
    TOP_K best anchor/class pairs among them that score above SCORE_THRESHOLD. Only logits above LOGIT_FLOOR
    are ranked, which leaves the selection unchanged.
    """
    shapes = [output.shape for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    logits = np.concatenate([output.reshape(-1, len(CLASSES)) for output in outputs[3:6]])
    angles = np.concatenate([output.reshape(-1) for output in outputs[6:]])
    best = logits.max(axis=1)
    # A NaN or +inf logit reaches its anchor's maximum; a -inf one scores 0 and is never kept.
    if not (np.isfinite(distances).all() and np.isfinite(best).all() and np.isfinite(angles).all()):
        raise ValueError("MPK output contains non-finite values")
    live_anchors = np.flatnonzero(best > LOGIT_FLOOR)
    anchors = live_anchors[top_indices(sigmoid(best[live_anchors]), TOP_K)]
    pair_logits = logits[anchors].ravel()
    live_pairs = np.flatnonzero(pair_logits > LOGIT_FLOOR)
    scores = sigmoid(pair_logits[live_pairs])
    picked = top_indices(scores, TOP_K)
    picked = picked[scores[picked] > SCORE_THRESHOLD]
    anchor, label = anchors[live_pairs[picked] // len(CLASSES)], live_pairs[picked] % len(CLASSES)
    # Ultralytics' dist2rbox: box side distances are in grid cells, left and top then right and bottom, in the
    # box's own frame. The centre is half their difference from the anchor, rotated by the raw angle in radians.
    near, far = distances[anchor, :2], distances[anchor, 2:]
    angle = angles[anchor]
    cos, sin = np.cos(angle), np.sin(angle)
    shift_x, shift_y = ((far - near) / 2).T
    centres = ANCHORS[anchor] + np.stack([shift_x * cos - shift_y * sin, shift_x * sin + shift_y * cos], axis=1)
    strides = ANCHOR_STRIDES[anchor]
    return np.column_stack([centres * strides, (near + far) * strides, angle, scores[picked], label])


def image_detections(detections: np.ndarray, window: Sequence[int], width: int, height: int) -> np.ndarray:
    """Map one tile's detections from its letterboxed input onto the image, given the tile's crop size.

    Undoes the letterbox and adds the window offset, as Ultralytics' validator and DOTA merge do; neither clips a
    rotated box.
    """
    gain, _, (left, top_pad, _, _) = letterbox_geometry(width, height)
    mapped = detections.copy()
    mapped[:, :2] = (mapped[:, :2] - [left, top_pad]) / gain + window[:2]
    mapped[:, 2:4] /= gain
    return mapped


def covariance(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a, b, c of each box's Gaussian covariance [[a, c], [c, b]], as Ultralytics' probiou models a box."""
    a, b = boxes[:, 2] ** 2 / 12, boxes[:, 3] ** 2 / 12
    cos, sin = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    return a * cos**2 + b * sin**2, a * sin**2 + b * cos**2, (a - b) * cos * sin


def probiou(first: np.ndarray, second: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Ultralytics' batch_probiou between xywhr boxes: one minus the Hellinger distance of their Gaussians."""
    x1, y1 = first[:, :1], first[:, 1:2]
    x2, y2 = second[:, 0], second[:, 1]
    a1, b1, c1 = (value[:, None] for value in covariance(first))
    a2, b2, c2 = covariance(second)
    a, b, c = a1 + a2, b1 + b2, c1 + c2
    t1 = (a * (y1 - y2) ** 2 + b * (x1 - x2) ** 2) / (a * b - c**2 + eps) * 0.25
    t2 = (c * (x2 - x1) * (y1 - y2)) / (a * b - c**2 + eps) * 0.5
    spread = np.sqrt(np.clip(a1 * b1 - c1**2, 0, None) * np.clip(a2 * b2 - c2**2, 0, None))
    t3 = np.log((a * b - c**2) / (4 * spread + eps) + eps) * 0.5
    distance = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1 - np.sqrt(1 - np.exp(-distance) + eps)


def fast_nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return the indices Ultralytics' Fast-NMS keeps, highest score first.

    A box is dropped when any higher-scoring box overlaps it by MERGE_IOU or more, even one that is dropped
    itself.
    """
    order = np.argsort(-scores, kind="stable")
    boxes = boxes[order]
    keep = np.ones(len(boxes), dtype=bool)
    for start in range(0, len(boxes), MERGE_BLOCK):
        stop = min(start + MERGE_BLOCK, len(boxes))
        higher = np.arange(stop)[:, None] < np.arange(start, stop)
        keep[start:stop] = ~((probiou(boxes[:stop], boxes[start:stop]) >= MERGE_IOU) & higher).any(axis=0)
    return order[keep]


def merge(detections: np.ndarray) -> np.ndarray:
    """Merge one image's tile detections class by class, as Ultralytics' DOTA merge does with its class offset."""
    kept = []
    for label in np.unique(detections[:, 6]):
        group = detections[detections[:, 6] == label]
        kept.append(group[fast_nms(group[:, :5], group[:, 5])])
    return np.concatenate(kept) if kept else detections


def corners(boxes: np.ndarray) -> np.ndarray:
    """Ultralytics' xywhr2xyxyxyxy: the four corners of each rotated box, shape (n, 4, 2)."""
    cos, sin = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    along = np.stack([boxes[:, 2] / 2 * cos, boxes[:, 2] / 2 * sin], axis=1)[:, None]
    across = np.stack([-boxes[:, 3] / 2 * sin, boxes[:, 3] / 2 * cos], axis=1)[:, None]
    signs_along = np.array([1, 1, -1, -1])[None, :, None]
    signs_across = np.array([1, -1, -1, 1])[None, :, None]
    return boxes[:, None, :2] + signs_along * along + signs_across * across


def ground_truth(content: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (n, 4, 2) polygons, class indices and difficult flags of a DOTA label file.

    Lines with fewer than nine fields, such as the imagesource and gsd header, are skipped as the DOTA devkit does.
    """
    rows = [line.split() for line in content.decode().splitlines()]
    rows = [row for row in rows if len(row) >= 9]
    polygons = np.asarray([row[:8] for row in rows], dtype=np.float64).reshape(-1, 4, 2)
    labels = np.asarray([CLASSES.index(row[8]) for row in rows], dtype=np.int64)
    difficult = np.asarray([len(row) > 9 and int(row[9]) != 0 for row in rows], dtype=bool)
    return polygons, labels, difficult


def best_overlaps(detections: np.ndarray, truths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each detection polygon's best polygon IoU with a truth polygon and that truth's index.

    Ties go to the first truth, as np.argmax does in the DOTA devkit. A detection overlapping no truth gets IoU 0
    and index -1.
    """
    best, index = np.zeros(len(detections)), np.full(len(detections), -1)
    if not len(detections) or not len(truths):
        return best, index
    found = shapely.make_valid(shapely.polygons(detections))
    labelled = shapely.make_valid(shapely.polygons(truths))
    pair_found, pair_labelled = shapely.STRtree(labelled).query(found, predicate="intersects")
    overlap = shapely.area(shapely.intersection(found[pair_found], labelled[pair_labelled]))
    union = shapely.area(found)[pair_found] + shapely.area(labelled)[pair_labelled] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    order = np.lexsort((pair_labelled, -iou, pair_found))
    pair_found, pair_labelled, iou = pair_found[order], pair_labelled[order], iou[order]
    first = np.unique(pair_found, return_index=True)[1]
    best[pair_found[first]], index[pair_found[first]] = iou[first], pair_labelled[first]
    return best, index


def voc07_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC 2007 11-point AP, as DOTA Task 1 scores: the mean best precision at recall 0, 0.1, ..., 1 or more."""
    points = [precision[recall >= t].max() if (recall >= t).any() else 0.0 for t in np.arange(0.0, 1.1, 0.1)]
    return float(np.mean(points))


def class_ap(best: np.ndarray, truth: np.ndarray, truth_difficult: np.ndarray, positives: int, threshold: float):
    """DOTA Task 1 AP of one class at one IoU threshold from its detections in descending score order.

    A detection is correct when its best truth is above threshold, not difficult and not already found; it is
    ignored when that truth is difficult, and wrong otherwise.
    """
    above = best > threshold
    difficult = np.zeros_like(above)
    difficult[above] = truth_difficult[truth[above]]
    counted = np.flatnonzero(above & ~difficult)
    correct = np.zeros_like(above)
    correct[counted[np.unique(truth[counted], return_index=True)[1]]] = True
    wrong = ~correct & ~(above & difficult)
    hits, misses = np.cumsum(correct), np.cumsum(wrong)
    return voc07_ap(hits / positives, hits / np.maximum(hits + misses, np.finfo(np.float64).eps))


def score(detections: dict[str, np.ndarray], label_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Return dota_obb_map50 and dota_obb_map over the classes with labelled objects, and each class's AP50."""
    found = {label: [] for label in range(len(CLASSES))}
    difficult_flags, positives, offset = [], np.zeros(len(CLASSES), dtype=np.int64), 0
    for name, boxes in detections.items():
        polygons, labels, difficult = ground_truth((label_dir / f"{Path(name).stem}.txt").read_bytes())
        difficult_flags.append(difficult)
        positives += np.bincount(labels[~difficult], minlength=len(CLASSES))
        for label in range(len(CLASSES)):
            class_detections, class_truths = boxes[boxes[:, 6] == label], np.flatnonzero(labels == label)
            best, index = best_overlaps(corners(class_detections), polygons[class_truths])
            truth = np.full(len(class_detections), -1)
            truth[index >= 0] = class_truths[index[index >= 0]] + offset
            found[label].append(np.column_stack([class_detections[:, 5], best, truth]))
        offset += len(labels)
    truth_difficult = np.concatenate(difficult_flags)
    table = np.zeros((len(CLASSES), len(IOU_THRESHOLDS)))
    scored = [label for label in range(len(CLASSES)) if positives[label]]
    for label in scored:
        rows = np.concatenate(found[label])
        rows = rows[np.argsort(-rows[:, 0], kind="stable")]
        best, truth = rows[:, 1], rows[:, 2].astype(np.int64)
        table[label] = [class_ap(best, truth, truth_difficult, positives[label], t) for t in IOU_THRESHOLDS]
    metrics = {"dota_obb_map50": float(table[scored, 0].mean()), "dota_obb_map": float(table[scored].mean())}
    return metrics, {CLASSES[label]: round(float(table[label, 0]), 4) for label in scored}


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="dota-v1-0-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    model = pyneat.Model(str(parse_mpks(args.mpk)))
    image_dir = dataset / IMAGES
    available = sorted(path.name for path in image_dir.glob("*.jpg"))
    if len(available) != VAL_IMAGES:
        raise SystemExit(f"expected {VAL_IMAGES} DOTA-v1.0 val images, found {len(available)}")
    names = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    runner = None
    detections: dict[str, np.ndarray] = {}
    tiles = 0
    started = time.monotonic()
    images = prefetched(lambda name: decode_image((image_dir / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        height, width = image.shape[:2]
        tile_detections = []
        for window in tile_windows(width, height):
            x0, y0, x1, y1 = window
            tile = image[y0:y1, x0:x1]
            tensor = pyneat.Tensor.from_numpy(preprocess_tile(tile), copy=False)
            if runner is None:
                runner = model.build([tensor])
            outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
            tile_detections.append(image_detections(decode_tile(outputs), window, tile.shape[1], tile.shape[0]))
        tiles += len(tile_detections)
        detections[name] = merge(np.concatenate(tile_detections))
        if count == 1 or count % 50 == 0 or count == len(names):
            elapsed = time.monotonic() - started
            log(f"{count}/{len(names)} images, {tiles} tiles, {elapsed / tiles * 1000:.0f} ms/tile")
    inference_seconds = time.monotonic() - started

    if not any(len(boxes) for boxes in detections.values()):
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    metrics, class_ap50 = score(detections, dataset / LABELS)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {name: metrics[name] for name in METRICS},
            "split": "val", "evaluated_samples": len(names), "available_samples": len(available),
            "details": {
                "tiles": tiles,
                "detections": sum(len(boxes) for boxes in detections.values()),
                "class_ap50": class_ap50,
                "inference_seconds": round(inference_seconds, 1),
                "scoring_seconds": round(scoring_seconds, 1),
            },
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
