"""Measure RF-DETR Small COCO bounding-box AP and AR with both MPKs on a Modalix board."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from calibration import IMAGE_SIZE, MEAN, QUERY_COUNT, STD, select_top_proposals
from faster_coco_eval import COCO, COCOeval_faster
from model_report import log
from PIL import Image

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
CLASSES = 91
TOP_K = 300
SCORE_THRESHOLD = 0.001
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
# Images decode on worker threads while the MPKs run; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
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


class ModelArtifact:
    def __init__(self, path: Path) -> None:
        self.model = pyneat.Model(str(path))
        self.runner = None

    def __call__(self, *inputs: np.ndarray | pyneat.Tensor) -> list[pyneat.Tensor]:
        # Inputs stay alive until run() returns, so NumPy arrays are wrapped without a copy.
        tensors = [
            item if isinstance(item, pyneat.Tensor) else pyneat.Tensor.from_numpy(item, copy=False) for item in inputs
        ]
        if self.runner is None:
            self.runner = self.model.build(tensors)
        return self.runner.run(tensors, timeout_ms=120000)


def view(tensor: pyneat.Tensor) -> np.ndarray:
    """Read-only view of an output; valid until its MPK runs again."""
    array = tensor.to_numpy(copy=False)
    if array.dtype != np.float32:
        raise ValueError(f"expected a float32 MPK output, got {tensor.dtype}")
    return array


def decode(
    dets: np.ndarray, labels: np.ndarray, image_id: int, width: int, height: int, valid: np.ndarray
) -> np.ndarray:
    """Return COCO result rows [image_id, x, y, w, h, score, category_id] for one image."""
    if dets.shape != (1, QUERY_COUNT, 4) or labels.shape != (1, QUERY_COUNT, CLASSES):
        raise ValueError(f"unexpected decoder outputs: dets={dets.shape}, labels={labels.shape}")
    if not (np.isfinite(dets).all() and np.isfinite(labels).all()):
        raise ValueError("MPK output contains non-finite values")
    scores = (1.0 / (1.0 + np.exp(-np.clip(labels[0], -80, 80)))).ravel()
    order = np.argsort(-scores, kind="stable")[:TOP_K]
    # RF-DETR keeps the 91 original COCO slots; 11 of them are unused.
    order = order[(scores[order] >= SCORE_THRESHOLD) & valid[order % CLASSES]]
    cx, cy, w, h = dets[0, order // CLASSES].T
    x1, x2 = np.clip((cx - w / 2) * width, 0, width), np.clip((cx + w / 2) * width, 0, width)
    y1, y2 = np.clip((cy - h / 2) * height, 0, height), np.clip((cy + h / 2) * height, 0, height)
    keep = (x2 > x1) & (y2 > y1)
    ids = np.full(order.shape, image_id)
    return np.stack([ids, x1, y1, x2 - x1, y2 - y1, scores[order], order % CLASSES], axis=1)[keep]


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


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "decoder"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and decoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for backbone and decoder")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    mpks = parse_mpks(args.mpk)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    valid = np.isin(np.arange(CLASSES), ground_truth.getCatIds())

    backbone, decoder = ModelArtifact(mpks["backbone"]), ModelArtifact(mpks["decoder"])
    rows: list[np.ndarray] = []
    started = time.monotonic()
    images = prefetched(lambda record: preprocess((dataset / IMAGES / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        features, scores, boxes = backbone(image)
        # The decoder's EV74 input stage needs the features in EV74 memory; only the proposals pass through NumPy.
        proposals = select_top_proposals(view(scores), view(boxes))
        dets, labels = (view(output) for output in decoder(features.cvu(), proposals))
        rows.append(decode(dets, labels, record["id"], record["width"], record["height"], valid))
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
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
