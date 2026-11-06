"""Measure YOLO26l-pose COCO keypoint and person-box AP and AR with its MPK on a Modalix board."""

from __future__ import annotations

import argparse
import contextlib
import io
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
from calibration import IMAGE_SIZE, letterbox_geometry, preprocess_image
from faster_coco_eval import COCO, COCOeval_faster

ANNOTATIONS = "annotations/person_keypoints_val2017.json"
IMAGES = "val2017"
KEYPOINTS = 17
STRIDES = (8, 16, 32)
# Ultralytics' COCO-pose validation set: the val2017 images with at least one person whose keypoints are labelled.
SCORED_IMAGES = 2346
# Ultralytics' end-to-end selection and validation settings: the best 300 anchors scoring above 0.001. With one
# class, its second top-k over anchor/class pairs keeps the same anchors.
TOP_K = 300
SCORE_THRESHOLD = 0.001
# Metric name in model.yaml -> (COCOeval_faster iouType, stats_as_dict label). Keypoint scoring keeps the 20
# highest-scoring people per image and has no small-object range.
METRICS = {
    "coco_keypoint_map": ("keypoints", "AP_all"),
    "coco_keypoint_ap50": ("keypoints", "AP_50"),
    "coco_keypoint_ap75": ("keypoints", "AP_75"),
    "coco_keypoint_ap_medium": ("keypoints", "AP_medium"),
    "coco_keypoint_ap_large": ("keypoints", "AP_large"),
    "coco_keypoint_ar": ("keypoints", "AR_all"),
    "coco_keypoint_ar50": ("keypoints", "AR_50"),
    "coco_keypoint_ar75": ("keypoints", "AR_75"),
    "coco_keypoint_ar_medium": ("keypoints", "AR_medium"),
    "coco_keypoint_ar_large": ("keypoints", "AR_large"),
    "coco_person_box_map": ("bbox", "AP_all"),
    "coco_person_box_ap50": ("bbox", "AP_50"),
}
MAX_DETECTIONS = {"keypoints": [20], "bbox": [1, 10, 100]}
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# The MPK returns bbox_0..2, class_logit_0..2 then keypoint_0..2 in NHWC. Anchors are the grid-cell centres of
# each level, row-major from stride 8 to stride 32, as Ultralytics' make_anchors lays them out.
GRIDS = [IMAGE_SIZE // stride for stride in STRIDES]
OUTPUT_SHAPES = [(1, size, size, channels) for channels in (4, 1, KEYPOINTS * 3) for size in GRIDS]
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


def scored_image_ids(ground_truth: COCO) -> list[int]:
    """Return the images with at least one person whose keypoints are labelled."""
    annotations = ground_truth.dataset["annotations"]
    labelled = {annotation["image_id"] for annotation in annotations if annotation["num_keypoints"]}
    if len(labelled) != SCORED_IMAGES:
        raise SystemExit(f"expected {SCORED_IMAGES} images with labelled keypoints, found {len(labelled)}")
    return sorted(labelled)


def image_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map x, y points from the letterboxed input onto the original image and clip them to it, as Ultralytics does."""
    gain, _, (left, top, _, _) = letterbox_geometry(width, height)
    return ((np.asarray(points, dtype=np.float64) - [left, top]) / gain).clip(0, [width, height])


def image_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map x1, y1, x2, y2 boxes from the letterboxed input onto the original image and clip them to it."""
    return image_points(boxes.reshape(-1, 2), width, height).reshape(-1, 4)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))


def top(values: np.ndarray, count: int) -> np.ndarray:
    """Indices of the count largest values, largest first, ties in index order, like a stable descending sort."""
    chosen = np.arange(len(values))
    if len(values) > count:
        cutoff = np.partition(values, len(values) - count)[len(values) - count]
        above = np.flatnonzero(values > cutoff)
        chosen = np.sort(np.concatenate([above, np.flatnonzero(values == cutoff)[: count - len(above)]]))
    return chosen[np.argsort(-values[chosen], kind="stable")]


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int, category: int) -> list[dict]:
    """Return COCO results with a box, 17 keypoints and a score for each selected person in one image.

    Selects people as the exported Ultralytics head does: the TOP_K anchors by person score that score above
    SCORE_THRESHOLD.
    """
    shapes = [output.shape for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    logits = np.concatenate([output.reshape(-1) for output in outputs[3:6]])
    keypoints = np.concatenate([output.reshape(-1, KEYPOINTS, 3) for output in outputs[6:]])
    if not (np.isfinite(distances).all() and np.isfinite(logits).all() and np.isfinite(keypoints).all()):
        raise ValueError("MPK output contains non-finite values")
    scores = sigmoid(logits)
    anchor = top(scores, TOP_K)
    anchor = anchor[scores[anchor] > SCORE_THRESHOLD]
    scores = scores[anchor]
    centres, strides = ANCHORS[anchor], ANCHOR_STRIDES[anchor]
    # Box side distances are in grid cells: left and top from the anchor, then right and bottom.
    corners = np.hstack([centres - distances[anchor, :2], centres + distances[anchor, 2:]])
    x1, y1, x2, y2 = image_boxes(corners * strides, width, height).T
    # Keypoint channels interleave x, y and a visibility logit; x and y are grid-cell offsets from the anchor.
    people = keypoints[anchor]
    xy = image_points(((people[..., :2] + centres[:, None]) * strides[:, None]).reshape(-1, 2), width, height)
    points = np.concatenate([xy.reshape(-1, KEYPOINTS, 2), sigmoid(people[..., 2:])], axis=2)
    points = points.reshape(len(anchor), KEYPOINTS * 3)
    return [
        {
            "image_id": image_id,
            "category_id": category,
            "bbox": [float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])],
            "keypoints": points[i].tolist(),
            "score": float(scores[i]),
        }
        for i in range(len(anchor))
    ]


def score(ground_truth: COCO, results: COCO, iou_type: str, image_ids: list[int]) -> dict[str, float]:
    """Score one iouType and return its declared metrics; COCOeval reports -1 for an empty object-size range."""
    scorer = COCOeval_faster(ground_truth, results, iou_type, print_function=log)
    if list(scorer.params.maxDets) != MAX_DETECTIONS[iou_type]:
        raise ValueError(f"expected {iou_type} maxDets {MAX_DETECTIONS[iou_type]}, got {scorer.params.maxDets}")
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, (kind, label) in METRICS.items() if kind == iou_type}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


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
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    model = pyneat.Model(str(parse_mpks(args.mpk)))
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = scored_image_ids(ground_truth)
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    categories = ground_truth.getCatIds()
    if len(categories) != 1:
        raise SystemExit(f"expected the single COCO person category, found {categories}")

    runner = None
    detections: list[dict] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    images = prefetched(lambda record: preprocess_image((image_dir / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
        detections += decode(outputs, record["id"], record["width"], record["height"], categories[0])
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    inference_seconds = time.monotonic() - started

    if not detections:
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
    stats = score(ground_truth, results, "keypoints", image_ids) | score(ground_truth, results, "bbox", image_ids)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {name: stats[name] for name in METRICS},
            "split": IMAGES, "evaluated_samples": len(image_ids), "available_samples": len(available),
            "details": {
                "detections": len(detections),
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
