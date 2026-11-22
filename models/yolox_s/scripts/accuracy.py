"""Measure YOLOX-s COCO bounding-box AP and AR with its MPK on a Modalix board."""

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

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
CLASSES = 80
STRIDES = (8, 16, 32)
# Match upstream YOLOX postprocess: one best class per cell, objectness * class probability,
# then class-aware NMS at IoU 0.65. Keep low-score detections for COCO evaluation.
SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.65
MAX_DETECTIONS_PER_IMAGE = 1000
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label. With maxDets [1, 10, 100], AR_all,
# AR_second and AR_third are the recall at 1, 10 and 100 detections per image.
METRICS = {
    "coco_map": "AP_all",
    "coco_ap50": "AP_50",
    "coco_ap75": "AP_75",
    "coco_ap_small": "AP_small",
    "coco_ap_medium": "AP_medium",
    "coco_ap_large": "AP_large",
    "coco_ar1": "AR_all",
    "coco_ar10": "AR_second",
    "coco_ar100": "AR_third",
    "coco_ar_small": "AR_small",
    "coco_ar_medium": "AR_medium",
    "coco_ar_large": "AR_large",
}
MAX_DETECTIONS = [1, 10, 100]
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# Each feature level returns raw xywh, objectness logits, and class logits, in that order.
GRIDS = [IMAGE_SIZE // stride for stride in STRIDES]
OUTPUT_SHAPES = [(1, size, size, channels) for size in GRIDS for channels in (4, 1, CLASSES)]
ANCHORS = np.concatenate(
    [np.stack(np.meshgrid(np.arange(size), np.arange(size)), axis=-1).reshape(-1, 2) for size in GRIDS]
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


def image_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map x1, y1, x2, y2 boxes from the letterboxed input onto the original image and clip them to it."""
    gain, _, (left, top, _, _) = letterbox_geometry(width, height)
    mapped = (np.asarray(boxes, dtype=np.float64) - [left, top, left, top]) / gain
    mapped[:, 0::2] = mapped[:, 0::2].clip(0, width)
    mapped[:, 1::2] = mapped[:, 1::2].clip(0, height)
    return mapped


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one x1, y1, x2, y2 box with each of boxes."""
    width = (np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])).clip(0)
    height = (np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])).clip(0)
    overlap = width * height
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return overlap / (area + areas - overlap + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedy NMS like torchvision: indices of the kept boxes, best first, dropping overlaps above IOU_THRESHOLD."""
    order = np.argsort(-scores, kind="stable")
    kept = []
    while len(order) and len(kept) < MAX_DETECTIONS_PER_IMAGE:
        best, order = order[0], order[1:]
        kept.append(best)
        order = order[iou(boxes[best], boxes[order]) <= IOU_THRESHOLD]
    return np.array(kept, dtype=np.int64)


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int, categories: np.ndarray) -> np.ndarray:
    """Decode YOLOX xywh, multiply objectness/class probabilities, and suppress same-class overlaps."""
    shapes = [output.shape for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    if not all(np.isfinite(output).all() for output in outputs):
        raise ValueError("MPK output contains non-finite values")
    regression = np.concatenate([output.reshape(-1, 4) for output in outputs[0::3]]).astype(np.float64)
    objectness = np.concatenate([output.reshape(-1) for output in outputs[1::3]])
    logits = np.concatenate([output.reshape(-1, CLASSES) for output in outputs[2::3]])
    labels = logits.argmax(axis=1)
    scores = sigmoid(objectness.astype(np.float64)) * sigmoid(logits[np.arange(len(labels)), labels].astype(np.float64))
    anchor = np.flatnonzero(scores >= SCORE_THRESHOLD)
    if not len(anchor):
        return np.zeros((0, 7))
    labels, scores = labels[anchor], scores[anchor]
    centres = (regression[anchor, :2] + ANCHORS[anchor]) * ANCHOR_STRIDES[anchor]
    sizes = np.exp(regression[anchor, 2:]) * ANCHOR_STRIDES[anchor]
    corners = np.hstack([centres - sizes / 2, centres + sizes / 2])
    if not np.isfinite(corners).all():
        raise ValueError("decoded boxes contain non-finite values")
    # Suppress each class independently; this avoids coordinate offsets failing for very large boxes.
    kept = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        kept.extend(candidates[nms(corners[candidates], scores[candidates])])
    kept = np.asarray(kept, dtype=np.int64)
    kept = kept[np.argsort(-scores[kept], kind="stable")[:MAX_DETECTIONS_PER_IMAGE]]
    x1, y1, x2, y2 = image_boxes(corners[kept], width, height).T
    return np.stack(
        [np.full(len(kept), image_id), x1, y1, x2 - x1, y2 - y1, scores[kept], categories[labels[kept]]], axis=1
    )


def coco_stats(scorer: COCOeval_faster) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in METRICS.items()}
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
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(str(parse_mpks(args.mpk)), options)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    # YOLOX numbers the 80 COCO classes in category-id order.
    categories = np.array(sorted(ground_truth.getCatIds()))
    if len(categories) != CLASSES:
        raise SystemExit(f"expected {CLASSES} COCO categories, found {len(categories)}")

    runner = None
    rows: list[np.ndarray] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    images = prefetched(lambda record: preprocess_image((image_dir / record["file_name"]).read_bytes()), records)
    try:
        for count, (record, image) in enumerate(zip(records, images), start=1):
            tensor = pyneat.Tensor.from_numpy(image, copy=False)
            if runner is None:
                runner = model.build([tensor])
            outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
            rows.append(decode(outputs, record["id"], record["width"], record["height"], categories))
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    finally:
        if runner is not None:
            runner.close()
    inference_seconds = time.monotonic() - started

    detections = np.concatenate(rows)
    if not len(detections):
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
    scorer = COCOeval_faster(ground_truth, results, "bbox", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    stats = coco_stats(scorer)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "detections": len(detections),
                    "inference_seconds": round(inference_seconds, 1),
                    "scoring_seconds": round(scoring_seconds, 1),
                },
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
