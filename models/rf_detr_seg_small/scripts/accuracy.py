"""Measure RF-DETR Seg Small COCO mask and box AP with both MPKs on a Modalix board.

The decode follows upstream rfdetr's PostProcess for RFDETRSegSmallConfig: the num_select (100) best
query/class pairs by sigmoid score, boxes scaled to the image, and each pair's mask logits resized
bilinearly to the image and thresholded at zero.
"""

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
from calibration import IMAGE_SIZE, MEAN, QUERY_COUNT, STD, select_top_proposals
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util
from PIL import Image

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
CLASS_COUNT = 91
MASK_SIZE = IMAGE_SIZE // 4
# Upstream's num_select: how many query/class pairs PostProcess keeps per image.
TOP_K = 100
SCORE_THRESHOLD = 0.001
MASK_CHUNK = 32
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
# Images decode on worker threads while the MPKs run; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# Resizing and encoding the masks is CPU work per image; worker processes keep up with the board.
DECODERS = 8
# calibration.preprocess_image's per-channel arithmetic, tabulated for every uint8 value and laid out HWC.
NORMALIZE = ((np.arange(256, dtype=np.float32)[:, None] / 255.0 - MEAN.ravel()) / STD.ravel()).reshape(1, 256, 3)


def preprocess(content: bytes) -> np.ndarray:
    """Return the same values as calibration.preprocess_image, in the backbone's HWC layout."""
    with Image.open(io.BytesIO(content)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return cv2.LUT(np.asarray(resized), NORMALIZE)


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def board_pipeline(mpks: dict[str, Path]) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a function that runs one preprocessed image through both MPKs and returns dets, labels and masks."""
    import pyneat

    def runner(path):
        model = pyneat.Model(str(path))
        built = None

        def run(*inputs):
            nonlocal built
            # Inputs stay alive until run() returns, so NumPy arrays are wrapped without a copy.
            tensors = [
                item if isinstance(item, pyneat.Tensor) else pyneat.Tensor.from_numpy(item, copy=False)
                for item in inputs
            ]
            if built is None:
                built = model.build(tensors)
            return built.run(tensors, timeout_ms=120000)

        return run

    backbone, decoder = runner(mpks["backbone"]), runner(mpks["decoder"])

    def infer(image):
        features, scores, boxes = backbone(image)
        proposals = select_top_proposals(scores.to_numpy(), boxes.to_numpy())
        # The decoder's EV74 input stage needs the features in EV74 memory; only the proposals pass through NumPy.
        outputs = decoder(features.cvu(), proposals)
        # Copies, because the runner reuses its output buffers while a decoder process may still read them.
        dets, labels, masks = (output.to_numpy(copy=True) for output in outputs)
        # The MPK returns the mask logits channels-last: [1, height, width, query].
        return dets, labels, masks[0]

    return infer


def decode(dets: np.ndarray, labels: np.ndarray, masks: np.ndarray, record: dict, valid: np.ndarray) -> list[dict]:
    """Return the COCO results, each with a box and an RLE mask, for the image that record describes.

    masks holds each query's mask logits as [height, width, query].
    """
    width, height = record["width"], record["height"]
    if dets.shape != (1, QUERY_COUNT, 4) or labels.shape != (1, QUERY_COUNT, CLASS_COUNT):
        raise ValueError(f"unexpected decoder outputs: dets={dets.shape}, labels={labels.shape}")
    if masks.shape != (MASK_SIZE, MASK_SIZE, QUERY_COUNT):
        raise ValueError(f"unexpected mask logits: {masks.shape}")
    if not (np.isfinite(dets).all() and np.isfinite(labels).all() and np.isfinite(masks).all()):
        raise ValueError("MPK output contains non-finite values")
    scores = (1.0 / (1.0 + np.exp(-np.clip(labels[0], -80, 80)))).ravel()
    order = np.argsort(-scores, kind="stable")[:TOP_K]
    # RF-DETR keeps the 91 original COCO slots; 11 of them are unused.
    order = order[(scores[order] >= SCORE_THRESHOLD) & valid[order % CLASS_COUNT]]
    if not order.size:
        return []
    queries = order // CLASS_COUNT
    cx, cy, w, h = dets[0, queries].T
    x1, x2 = np.clip((cx - w / 2) * width, 0, width), np.clip((cx + w / 2) * width, 0, width)
    y1, y2 = np.clip((cy - h / 2) * height, 0, height), np.clip((cy + h / 2) * height, 0, height)
    rles = []
    # Resize a few masks at a time, so a large image does not hold every kept mask at once.
    for start in range(0, queries.size, MASK_CHUNK):
        chunk = queries[start : start + MASK_CHUNK]
        # cv2.resize matches upstream's bilinear F.interpolate (align_corners=False) and drops a single channel.
        resized = cv2.resize(np.ascontiguousarray(masks[:, :, chunk]), (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (resized.reshape(height, width, chunk.size) > 0).astype(np.uint8)
        rles.extend(mask_util.encode(np.asfortranarray(binary)))
    return [
        {
            "image_id": record["id"],
            "category_id": int(category),
            "bbox": [float(x), float(y), float(right - x), float(bottom - y)],
            "score": float(score),
            "segmentation": rle,
        }
        for x, y, right, bottom, score, category, rle in zip(
            x1, y1, x2, y2, scores[order], order % CLASS_COUNT, rles, strict=True
        )
    ]


def coco_results(detections: list[dict], iou_type: str) -> list[dict]:
    """Return copies of the results with only the box or only the mask.

    COCO.loadRes takes each result's area from its box when it has one, which would sort masks into the
    small, medium and large ranges by box size.
    """
    dropped = "segmentation" if iou_type == "bbox" else "bbox"
    return [{key: value for key, value in result.items() if key != dropped} for result in detections]


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


def evaluate(
    infer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dataset: Path,
    limit: int | None,
) -> dict:
    """Run infer on every val2017 image and return the dataset's mask and box result for the accuracy report."""
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, limit)) if limit else available
    records = ground_truth.loadImgs(image_ids)
    valid = np.isin(np.arange(CLASS_COUNT), ground_truth.getCatIds())

    detections: list[dict] = []
    started = time.monotonic()
    images = prefetched(lambda record: preprocess((dataset / IMAGES / record["file_name"]).read_bytes()), records)
    # Worker processes decode earlier images while the next ones run; results keep image order.
    with ProcessPoolExecutor(DECODERS, mp_context=multiprocessing.get_context("spawn")) as decoders:
        pending: deque = deque()
        for count, (record, image) in enumerate(zip(records, images), start=1):
            dets, labels, masks = infer(image)
            pending.append(decoders.submit(decode, dets, labels, masks, record, valid))
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
    stats = {}
    for iou_type, metrics in METRICS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            results = ground_truth.loadRes(coco_results(detections, iou_type))
        scorer = COCOeval_faster(ground_truth, results, iou_type, print_function=log)
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
        stats.update(coco_stats(scorer, metrics))
    scoring_seconds = time.monotonic() - started

    return {
        "metrics": stats,
        "split": IMAGES,
        "evaluated_samples": len(image_ids),
        "available_samples": len(available),
        "details": {
            "detections": len(detections),
            "inference_seconds": round(inference_seconds, 1),
            "scoring_seconds": round(scoring_seconds, 1),
        },
    }


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"backbone", "decoder"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and decoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for backbone and decoder")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    result = evaluate(board_pipeline(parse_mpks(args.mpk)), Path(dataset_dir), args.limit)
    model_report.from_accuracy({dataset_id: result}).save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in result["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
