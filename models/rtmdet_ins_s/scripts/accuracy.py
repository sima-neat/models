"""Measure RTMDet-Ins S COCO mask AP and AR and box AP with its MPK on a Modalix board.

Detections follow MMDetection 3.3.0's RTMDetInsSepBNHead.predict_by_feat with the model's test configuration:
on each stride, the 1000 best class scores above 0.05; boxes decoded from the prior points and rescaled to the
image; class-aware NMS at IoU 0.6, keeping 100; then each detection's three dynamic 1x1 convolutions over the mask
features, upsampled 8x, resized to the image, and thresholded at a sigmoid of 0.5.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
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
from calibration import IMAGE_SIZE, letterbox_geometry, preprocess_image
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
CLASSES = 80
STRIDES = (8, 16, 32)
LEVELS = [IMAGE_SIZE // stride for stride in STRIDES]
MASK_STRIDE = STRIDES[0]
MASK_SIZE = LEVELS[0]
MASK_FEATURES = 8
# Each prior's kernel: three 1x1 convolutions, 10 -> 8 -> 8 -> 1 channels, as weights then biases.
KERNEL_SPLITS = [(MASK_FEATURES + 2) * MASK_FEATURES, MASK_FEATURES * MASK_FEATURES, MASK_FEATURES,
                 MASK_FEATURES, MASK_FEATURES, 1]
KERNEL_PARAMETERS = sum(KERNEL_SPLITS)
# MMDetection's test configuration for this checkpoint.
SCORE_THRESHOLD = 0.05
TOP_PER_LEVEL = 1000
NMS_IOU = 0.6
MAX_DETECTIONS = 100
# The MPK returns each tensor channels-last: cls_score_0..2, bbox_pred_0..2, kernel_pred_0..2, then mask_feat.
OUTPUT_SHAPES = (
    [(1, size, size, CLASSES) for size in LEVELS]
    + [(1, size, size, 4) for size in LEVELS]
    + [(1, size, size, KERNEL_PARAMETERS) for size in LEVELS]
    + [(1, MASK_SIZE, MASK_SIZE, MASK_FEATURES)]
)
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
COCO_MAX_DETECTIONS = [1, 10, 100]
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# Decoding builds and run-length encodes up to 100 image-size masks per image; processes keep up with the MPK.
DECODERS = 8


def _grid(size: int, stride: int) -> np.ndarray:
    """MMDetection's point priors for one level: (x, y) of each cell's corner, row-major, x fastest."""
    ys, xs = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1) * np.float32(stride)


PRIORS = [_grid(size, stride) for size, stride in zip(LEVELS, STRIDES, strict=True)]
MASK_GRID = _grid(MASK_SIZE, MASK_STRIDE)


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def candidates(
    outputs: Sequence[np.ndarray], resized_width: int, resized_height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the candidates' boxes in the resized image, scores, labels, kernels, and priors (x, y, stride)."""
    boxes, scores, labels, kernels, priors = [], [], [], [], []
    for level, stride in enumerate(STRIDES):
        level_scores = 1.0 / (1.0 + np.exp(-outputs[level].reshape(-1, CLASSES).astype(np.float64)))
        flat = level_scores.ravel()
        valid = np.flatnonzero(flat > SCORE_THRESHOLD)
        order = valid[np.argsort(-flat[valid], kind="stable")][:TOP_PER_LEVEL]
        index, label = order // CLASSES, order % CLASSES
        points = PRIORS[level][index]
        distances = outputs[3 + level].reshape(-1, 4)[index]
        # distance2bbox, clipped to the resized image as MMDetection clips to img_shape.
        level_boxes = np.concatenate([points - distances[:, :2], points + distances[:, 2:]], axis=1)
        level_boxes[:, 0::2] = level_boxes[:, 0::2].clip(0, resized_width)
        level_boxes[:, 1::2] = level_boxes[:, 1::2].clip(0, resized_height)
        boxes.append(level_boxes)
        scores.append(flat[order])
        labels.append(label)
        kernels.append(outputs[6 + level].reshape(-1, KERNEL_PARAMETERS)[index])
        priors.append(np.concatenate([points, np.full((len(index), 1), stride, np.float32)], axis=1))
    return tuple(np.concatenate(values) for values in (boxes, scores, labels, kernels, priors))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """mmcv's NMS: indices kept, best first, dropping a box whose IoU with a kept box exceeds the threshold."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(best)
        width = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        height = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        inter = width * height
        iou = inter / np.maximum(areas[best] + areas[rest] - inter, np.finfo(np.float64).eps)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """mmcv's class-aware NMS: offset each class's boxes apart so boxes of different classes never overlap."""
    if not len(boxes):
        return np.zeros(0, np.int64)
    offsets = labels.astype(np.float64)[:, None] * (boxes.max() + 1)
    return nms(boxes.astype(np.float64) + offsets, scores, NMS_IOU)


def mask_logits(mask_features: np.ndarray, kernels: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Each detection's mask logits on the 80x80 mask grid, from its dynamic convolutions.

    The input is the detection's relative coordinates, (prior - grid point) / (prior stride * 8), then the 8 mask
    features; mask_features is the MPK's (1, 80, 80, 8) output.
    """
    count = len(kernels)
    features = mask_features.reshape(-1, MASK_FEATURES).T.astype(np.float32)
    relative = (priors[:, None, :2] - MASK_GRID[None]) / (priors[:, None, 2:] * 8)
    x = np.concatenate([relative.transpose(0, 2, 1), np.broadcast_to(features, (count, *features.shape))], axis=1)
    w0, w1, w2, b0, b1, b2 = np.split(kernels.astype(np.float32), np.cumsum(KERNEL_SPLITS)[:-1], axis=1)
    x = np.maximum(np.matmul(w0.reshape(count, MASK_FEATURES, -1), x) + b0[:, :, None], 0)
    x = np.maximum(np.matmul(w1.reshape(count, MASK_FEATURES, -1), x) + b1[:, :, None], 0)
    x = np.matmul(w2.reshape(count, 1, -1), x) + b2[:, :, None]
    return x.reshape(count, MASK_SIZE, MASK_SIZE)


def image_masks(logits: np.ndarray, width: int, height: int, scale_x: float, scale_y: float) -> np.ndarray:
    """Upsample the mask logits 8x, resize them to the image and threshold them, as MMDetection does.

    MMDetection resizes the padded 640x640 logits to ceil(640 / scale) and crops the image; it takes the height from
    the width's scale factor and the width from the height's. OpenCV's bilinear resize samples at pixel centres, as
    torch's interpolate does with align_corners=False. Rounding each side separately can leave the rescaled logits a
    pixel short of the image; that edge is background. Returns (height, width, count) booleans.
    """
    planes = np.ascontiguousarray(logits.transpose(1, 2, 0))
    resized = np.zeros((height, width, len(logits)), dtype=bool)
    # OpenCV resizes at most 512 channels at once.
    for start in range(0, len(logits), 512):
        chunk = planes[:, :, start : start + 512]
        upsampled = cv2.resize(chunk, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        size = (math.ceil(IMAGE_SIZE / scale_y), math.ceil(IMAGE_SIZE / scale_x))
        rescaled = cv2.resize(upsampled, size, interpolation=cv2.INTER_LINEAR).reshape(size[1], size[0], -1)
        rows, columns = min(height, size[1]), min(width, size[0])
        resized[:rows, :columns, start : start + 512] = rescaled[:rows, :columns] > 0
    return resized


def select(outputs: Sequence[np.ndarray], width: int, height: int) -> tuple[np.ndarray, ...]:
    """One image's detections after NMS: boxes in the image, scores, labels, kernels and priors."""
    if [tuple(value.shape) for value in outputs] != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {[value.shape for value in outputs]}")
    if not all(np.isfinite(value).all() for value in outputs):
        raise ValueError("MPK output contains non-finite values")
    resized_width, resized_height, scale_x, scale_y = letterbox_geometry(width, height)
    boxes, scores, labels, kernels, priors = candidates(outputs, resized_width, resized_height)
    boxes = boxes / np.asarray([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
    sized = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, labels, kernels, priors = (value[sized] for value in (boxes, scores, labels, kernels, priors))
    keep = batched_nms(boxes, scores, labels)[:MAX_DETECTIONS]
    return tuple(value[keep] for value in (boxes, scores, labels, kernels, priors))


def finish(
    selected: tuple[np.ndarray, ...], mask_features: np.ndarray, image_id: int, width: int, height: int,
    categories: Sequence[int],
) -> dict:
    """Build the selected detections' masks and return the image's COCO box and mask results."""
    boxes, scores, labels, kernels, priors = selected
    results = {"bbox": [], "segm": []}
    if not len(boxes):
        return results
    _, _, scale_x, scale_y = letterbox_geometry(width, height)
    masks = image_masks(mask_logits(mask_features, kernels, priors), width, height, scale_x, scale_y)
    rles = mask_util.encode(np.asfortranarray(masks.astype(np.uint8)))
    for box, score, label, rle in zip(boxes, scores, labels, rles, strict=True):
        common = {"image_id": image_id, "category_id": int(categories[label]), "score": float(score)}
        results["bbox"].append({**common, "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]),
                                                   float(box[3] - box[1])]})
        counts = rle["counts"]
        results["segm"].append({**common, "segmentation": {
            "size": rle["size"], "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
        }})
    return results


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int, categories: Sequence[int]) -> dict:
    """Return one image's COCO box and mask results, each list as COCO's loadRes takes it."""
    return finish(select(outputs, width, height), outputs[9], image_id, width, height, categories)


def evaluate(
    dataset: Path,
    records: list[dict],
    categories: Sequence[int],
    infer: Callable[[np.ndarray], list[np.ndarray]],
) -> dict[str, list[dict]]:
    """Run infer on each image's preprocessed HWC tensor, select its detections, and build their masks on worker
    processes, which receive only the selected kernels and priors and the mask features."""
    detections: dict[str, list[dict]] = {"bbox": [], "segm": []}
    started = time.monotonic()
    images = prefetched(lambda record: preprocess_image((dataset / IMAGES / record["file_name"]).read_bytes()), records)
    with ProcessPoolExecutor(DECODERS, mp_context=multiprocessing.get_context("forkserver")) as decoders:
        pending: deque = deque()

        def collect() -> None:
            for key, values in pending.popleft().result().items():
                detections[key].extend(values)

        for count, (record, image) in enumerate(zip(records, images), start=1):
            outputs = infer(image)
            selected = select(outputs, record["width"], record["height"])
            pending.append(decoders.submit(
                finish, selected, outputs[9], record["id"], record["width"], record["height"], categories
            ))
            while len(pending) > 2 * DECODERS:
                collect()
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
        while pending:
            collect()
    return detections


def coco_stats(scorer: COCOeval_faster, metrics: dict[str, str]) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != COCO_MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {COCO_MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in metrics.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def score(ground_truth: COCO, detections: dict[str, list[dict]], image_ids: list[int]) -> dict[str, float]:
    """Score masks and boxes separately, so each mask's area, not its box's, places it in a COCO size range."""
    stats = {}
    for iou_type, metrics in METRICS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            results = ground_truth.loadRes(detections[iou_type])
        scorer = COCOeval_faster(ground_truth, results, iou_type, print_function=log)
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
        stats.update(coco_stats(scorer, metrics))
    return stats


def parse_mpks(values: list[str]) -> Path:
    """The model's MPK from its --mpk model=<path> argument, as run_model_accuracy.sh passes it."""
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def load_dataset(dataset: Path, limit: int | None) -> tuple[COCO, list[int], list[int], list[dict], list[int]]:
    """The ground truth, every image ID, the evaluated image IDs and records, and the 80 category IDs in order."""
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, limit)) if limit else available
    categories = sorted(ground_truth.getCatIds())
    if len(categories) != CLASSES:
        raise SystemExit(f"expected {CLASSES} COCO categories, found {len(categories)}")
    return ground_truth, available, image_ids, ground_truth.loadImgs(image_ids), categories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)
    ground_truth, available, image_ids, records, categories = load_dataset(dataset, args.limit)

    # Imported here so fp32_baseline_accuracy.py can reuse this script on a host without PyNeat.
    import pyneat

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    model = pyneat.Model(str(parse_mpks(args.mpk)), options)
    runner = None

    def infer(image: np.ndarray) -> list[np.ndarray]:
        nonlocal runner
        # The input stays alive until run() returns, so it is wrapped without a copy.
        tensors = [pyneat.Tensor.from_numpy(image[None], copy=False)]
        if runner is None:
            runner = model.build(tensors)
        # Outputs are views into the MPK's buffers, so copy them before the next image runs.
        return [output.to_numpy(copy=True) for output in runner.run(tensors, timeout_ms=120000)]

    started = time.monotonic()
    detections = evaluate(dataset, records, categories, infer)
    inference_seconds = time.monotonic() - started
    if not detections["segm"]:
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    stats = score(ground_truth, detections, image_ids)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": stats,
            "split": IMAGES, "evaluated_samples": len(image_ids), "available_samples": len(available),
            "details": {
                "detections": len(detections["segm"]),
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
