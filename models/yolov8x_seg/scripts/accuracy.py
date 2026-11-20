"""Measure YOLOv8x-seg COCO mask AP and AR and box AP with its MPK on a Modalix board."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from calibration import IMAGE_SIZE, letterbox_geometry, preprocess_image
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# The published metrics are over the full val2017 split.
VAL2017_IMAGES = 5000
CLASSES = 80
MASK_CHANNELS = 32
PROTO_SIZE = IMAGE_SIZE // 4
STRIDES = (8, 16, 32)
# Ultralytics' COCO validation settings: keep every class score above 0.001 (several classes per box), then
# class-aware NMS at IoU 0.7 over at most 30000 candidates, keeping at most 300 detections per image.
SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.7
MAX_CANDIDATES = 30000
MAX_DETECTIONS_PER_IMAGE = 300
# Ultralytics offsets each class's boxes by this many pixels, so one NMS pass never compares two classes.
CLASS_OFFSET = 7680
# sigmoid(-6.91) is about 0.000997, below SCORE_THRESHOLD, so a logit at or below LOGIT_FLOOR is never kept and
# the decoder skips its sigmoid.
LOGIT_FLOOR = -6.91
# Each box edge is a distribution over REG_MAX bins; its expected bin is the edge distance in cells.
REG_MAX = 16
# On the CPU, Ultralytics' crop_mask crops with rounded box bounds, applied as Python slices, for fewer masks than
# this, and with the exact bounds otherwise.
SLICED_CROP_LIMIT = 50
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label, per COCO IoU type. With maxDets [1, 10, 100],
# AR_all, AR_second and AR_third are the recall at 1, 10 and 100 detections per image.
METRICS = {
    "segm": {
        "coco_mask_map": "AP_all",
        "coco_mask_ap50": "AP_50",
        "coco_mask_ap75": "AP_75",
        "coco_mask_ap_small": "AP_small",
        "coco_mask_ap_medium": "AP_medium",
        "coco_mask_ap_large": "AP_large",
        "coco_mask_ar1": "AR_all",
        "coco_mask_ar10": "AR_second",
        "coco_mask_ar100": "AR_third",
        "coco_mask_ar_small": "AR_small",
        "coco_mask_ar_medium": "AR_medium",
        "coco_mask_ar_large": "AR_large",
    },
    "bbox": {"coco_box_map": "AP_all", "coco_box_ap50": "AP_50"},
}
MAX_DETECTIONS = [1, 10, 100]
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# NMS and mask decoding are per-detection Python work; 8 processes keep up with inference on the board's 16 cores.
DECODERS = 8
# The MPK returns bbox_0..2, class_logit_0..2 and mask_coeff_0..2, then mask_proto, all NHWC. Anchors are the
# grid-cell centres of each level, row-major from stride 8 to stride 32, as Ultralytics' make_anchors lays them out.
# Boxes are decoded in float32, like the exported head's arithmetic, so mask crop bounds round as Ultralytics' do.
GRIDS = [IMAGE_SIZE // stride for stride in STRIDES]
OUTPUT_SHAPES = [(1, size, size, channels) for channels in (4 * REG_MAX, CLASSES, MASK_CHANNELS) for size in GRIDS] + [
    (1, PROTO_SIZE, PROTO_SIZE, MASK_CHANNELS)
]
ANCHORS = np.concatenate(
    [np.stack(np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5), axis=-1).reshape(-1, 2) for size in GRIDS]
).astype(np.float32)
ANCHOR_STRIDES = np.repeat(np.float32(STRIDES), [size * size for size in GRIDS])[:, None]
BINS = np.arange(REG_MAX, dtype=np.float32)


def resize_sources(source: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return torch's bilinear resize (align_corners=False) from source to target pixels along one axis.

    For each target pixel: the lower and upper source index and the upper one's weight. The upper index equals the
    lower one where that weight is zero. Positions are computed in float32, as torch does for float32 tensors.
    """
    scale = np.float32(source) / np.float32(target)
    position = scale * (np.arange(target, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
    position = np.maximum(position, np.float32(0))
    lower = np.minimum(np.floor(position).astype(np.int64), source - 1)
    weight = np.clip(position - lower, 0, 1).astype(np.float32)
    upper = np.where(weight > 0, np.minimum(lower + 1, source - 1), lower)
    return lower, upper, weight


# The prototype rows (and columns) that each frame row (and column) samples when upsampled 4x to 640x640.
PROTO_LOWER, PROTO_UPPER, _ = resize_sources(PROTO_SIZE, IMAGE_SIZE)


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


def edge_distances(distributions: np.ndarray) -> np.ndarray:
    """Return the left, top, right and bottom distances in cells: the expected bin of each edge's softmax."""
    logits = distributions.reshape(-1, 4, REG_MAX).astype(np.float32)
    weights = np.exp(logits - logits.max(axis=2, keepdims=True))
    return (weights / weights.sum(axis=2, keepdims=True)) @ BINS


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one x1, y1, x2, y2 box with each of boxes."""
    width = (np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])).clip(0)
    height = (np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])).clip(0)
    overlap = width * height
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return overlap / (area + areas - overlap + 1e-9)


def class_aware(corners: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Offset each class's boxes by CLASS_OFFSET, in float32 as Ultralytics does, so one NMS pass never compares two
    classes. At these offsets float32 rounds a high class's coordinates (to 1/16 pixel from class 69), which decides
    some overlaps near IOU_THRESHOLD the way Ultralytics' NMS does."""
    return corners + (labels * CLASS_OFFSET).astype(np.float32)[:, None]


def nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedy NMS like torchvision: indices of the kept boxes, best first, dropping overlaps above IOU_THRESHOLD."""
    order = np.argsort(-scores, kind="stable")
    kept = []
    while len(order) and len(kept) < MAX_DETECTIONS_PER_IMAGE:
        best, order = order[0], order[1:]
        kept.append(best)
        order = order[iou(boxes[best], boxes[order]) <= IOU_THRESHOLD]
    return np.array(kept, dtype=np.int64)


def select(outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the kept detections' boxes in the 640x640 input, scores, class indices and mask coefficients.

    Selects detections as Ultralytics' COCO validation does: every anchor/class pair scoring above SCORE_THRESHOLD,
    the MAX_CANDIDATES best of them, then class-aware NMS keeping at most MAX_DETECTIONS_PER_IMAGE.
    """
    shapes = [output.shape for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    distributions = np.concatenate([output.reshape(-1, 4 * REG_MAX) for output in outputs[:3]])
    logits = np.concatenate([output.reshape(-1, CLASSES) for output in outputs[3:6]])
    coefficients = np.concatenate([output.reshape(-1, MASK_CHANNELS) for output in outputs[6:9]])
    if not all(np.isfinite(values).all() for values in (distributions, logits, coefficients, outputs[9])):
        raise ValueError("MPK output contains non-finite values")
    anchor, label = np.nonzero(logits > LOGIT_FLOOR)
    scores = sigmoid(logits[anchor, label])
    live = scores > SCORE_THRESHOLD
    anchor, label, scores = anchor[live], label[live], scores[live]
    if len(scores) > MAX_CANDIDATES:
        best = np.argsort(-scores, kind="stable")[:MAX_CANDIDATES]
        anchor, label, scores = anchor[best], label[best], scores[best]
    if not len(scores):
        return np.zeros((0, 4), np.float32), scores, label, np.zeros((0, MASK_CHANNELS), np.float32)
    # Decode each distinct candidate anchor once; several classes can share it.
    unique, inverse = np.unique(anchor, return_inverse=True)
    distances = edge_distances(distributions[unique])[inverse]
    corners = np.hstack([ANCHORS[anchor] - distances[:, :2], ANCHORS[anchor] + distances[:, 2:]])
    corners *= ANCHOR_STRIDES[anchor]
    kept = nms(class_aware(corners, label), scores)
    return corners[kept], scores[kept], label[kept], coefficients[anchor[kept]]


def crop_bounds(boxes: np.ndarray) -> np.ndarray:
    """Return the [x0, y0, x1, y1) pixel range of the 640x640 frame that Ultralytics' crop_mask keeps per box."""
    if len(boxes) < SLICED_CROP_LIMIT:
        # Rounded bounds as Python slice ends: a negative bound counts from the far edge of the frame.
        bounds = np.round(boxes).astype(np.int64)
        bounds = np.where(bounds < 0, IMAGE_SIZE + bounds, bounds)
    else:
        # Pixel index i is kept where x1 <= i < x2.
        bounds = np.ceil(boxes).astype(np.int64)
    return bounds.clip(0, IMAGE_SIZE)


def frame_mask(proto: np.ndarray, coefficients: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Return the binary mask of one detection over frame rows y0:y1 and columns x0:x1.

    Equals Ultralytics' process_mask_native there: the prototype mix, upsampled bilinearly to 640x640, then
    positive. Only the prototype window that the region samples is mixed and upsampled. OpenCV's bilinear resize
    uses torch's pixel centers (align_corners=False), so frame row y is row y - 4 * r0 of the upsampled window.
    """
    r0, r1 = PROTO_LOWER[y0], PROTO_UPPER[y1 - 1] + 1
    c0, c1 = PROTO_LOWER[x0], PROTO_UPPER[x1 - 1] + 1
    mixed = proto[r0:r1, c0:c1] @ coefficients
    upsampled = cv2.resize(mixed, (4 * (c1 - c0), 4 * (r1 - r0)), interpolation=cv2.INTER_LINEAR)
    return upsampled[y0 - 4 * r0 : y1 - 4 * r0, x0 - 4 * c0 : x1 - 4 * c0] > 0


def instance_masks(
    proto: np.ndarray, boxes: np.ndarray, coefficients: np.ndarray, width: int, height: int
) -> list[dict]:
    """Return each detection's mask on the original image as a compressed COCO RLE, as Ultralytics computes it.

    Ultralytics' scale_masks crops the frame by the top padding at the top and bottom and by the left padding at
    the left and right, then resizes the binary mask bilinearly to the image and truncates it to uint8. A pixel
    is therefore set only where every frame pixel it samples with a nonzero weight is set. Each mask is computed
    only over its crop box, outside which it is zero.
    """
    _, _, (left, top, _, _) = letterbox_geometry(width, height)
    row_lower, row_upper, _ = resize_sources(IMAGE_SIZE - 2 * top, height)
    column_lower, column_upper, _ = resize_sources(IMAGE_SIZE - 2 * left, width)
    row_lower, row_upper = row_lower + top, row_upper + top
    column_lower, column_upper = column_lower + left, column_upper + left
    bounds = crop_bounds(boxes)
    # Image rows and columns whose samples all fall inside each crop box; the sources rise monotonically.
    spans = zip(
        np.searchsorted(row_lower, bounds[:, 1]),
        np.searchsorted(row_upper, bounds[:, 3]),
        np.searchsorted(column_lower, bounds[:, 0]),
        np.searchsorted(column_upper, bounds[:, 2]),
    )
    rles = []
    for (x0, y0, x1, y1), mix, (r0, r1, c0, c1) in zip(bounds, coefficients, spans):
        counts = [width * height]
        if r0 < r1 and c0 < c1:
            region = frame_mask(proto, mix, x0, y0, x1, y1)
            rows = region[row_lower[r0:r1] - y0] & region[row_upper[r0:r1] - y0]
            image = rows[:, column_lower[c0:c1] - x0] & rows[:, column_upper[c0:c1] - x0]
            counts = run_lengths(image, r0, c0, height, width)
        rles.append({"size": [height, width], "counts": counts})
    return mask_util.frPyObjects(rles, height, width) if rles else []


def run_lengths(region: np.ndarray, top: int, left: int, height: int, width: int) -> list[int]:
    """Uncompressed COCO RLE counts (column-major, background first) of an image that is zero outside region."""
    padded = np.zeros(region.shape[1] * height + 2, dtype=bool)
    padded[1:-1].reshape(region.shape[1], height)[:, top : top + region.shape[0]] = region.T
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + left * height
    # A run that reaches the last pixel already ends at width * height.
    if not changes.size or changes[-1] < width * height:
        changes = np.append(changes, width * height)
    return np.diff(changes, prepend=0).tolist()


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int, categories: np.ndarray) -> list[dict]:
    """Return the COCO results, each with a box and an RLE mask, for one image."""
    boxes, scores, labels, coefficients = select(outputs)
    masks = instance_masks(outputs[9][0], boxes, coefficients, width, height)
    x1, y1, x2, y2 = image_boxes(boxes, width, height).T
    return [
        {
            "image_id": image_id,
            "category_id": int(category),
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": float(score),
            "segmentation": mask,
        }
        for x, y, w, h, score, category, mask in zip(x1, y1, x2 - x1, y2 - y1, scores, categories[labels], masks)
    ]


def coco_stats(scorer: COCOeval_faster, metrics: dict[str, str]) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in metrics.items()}
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
    available = sorted(ground_truth.getImgIds())
    if len(available) != VAL2017_IMAGES:
        raise SystemExit(f"expected {VAL2017_IMAGES} val2017 images, found {len(available)}")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    # Ultralytics numbers the 80 COCO classes in category-id order.
    categories = np.array(sorted(ground_truth.getCatIds()))
    if len(categories) != CLASSES:
        raise SystemExit(f"expected {CLASSES} COCO categories, found {len(categories)}")

    runner = None
    detections: list[dict] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    images = prefetched(lambda record: preprocess_image((image_dir / record["file_name"]).read_bytes()), records)
    # Worker processes decode earlier images while the board runs the next ones; results keep image order.
    with ProcessPoolExecutor(DECODERS, mp_context=multiprocessing.get_context("spawn")) as decoders:
        pending: deque = deque()
        for count, (record, image) in enumerate(zip(records, images), start=1):
            tensor = pyneat.Tensor.from_numpy(image, copy=False)
            if runner is None:
                runner = model.build([tensor])
            # Copies, because the runner reuses its output buffers while a decoder may still read them.
            outputs = [output.to_numpy(copy=True) for output in runner.run([tensor], timeout_ms=120000)]
            pending.append(
                decoders.submit(decode, outputs, record["id"], record["width"], record["height"], categories)
            )
            if len(pending) > 2 * DECODERS:
                detections.extend(pending.popleft().result())
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
        while pending:
            detections.extend(pending.popleft().result())
    inference_seconds = time.monotonic() - started

    if not detections:
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
    stats = {}
    # Ultralytics scores masks and boxes from one result set, as here.
    for iou_type, metrics in METRICS.items():
        scorer = COCOeval_faster(ground_truth, results, iou_type, print_function=log)
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
        stats.update(coco_stats(scorer, metrics))
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": stats,
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
