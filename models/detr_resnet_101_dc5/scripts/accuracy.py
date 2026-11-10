"""Score one compiled DETR ResNet-101 DC5 MPK on COCO val2017 and save its accuracy report.

Each image is letterboxed into the 1280x720 input and ImageNet-normalized on the CPU, as
calibration.preprocess_image does. Every one of the 100 queries becomes one detection, as DETR's own COCO
evaluation scores them: the class with the highest softmax probability, excluding "no object", and its box
mapped back to the image.
"""

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
from typing import Any

import cv2
import model_report
import numpy as np
from calibration import INPUT_HEIGHT, INPUT_WIDTH, MEAN, STD, letterbox, letterbox_geometry
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
QUERY_COUNT = 100
# DETR keeps the 91 original COCO category slots, so a class index is its COCO category ID.
# The last class means "no object".
CLASSES = 92
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
# Images decode on worker threads while the MPK runs; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# Runs one preprocessed HWC image and returns its (logits, pred_boxes).
Inference = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
# calibration.preprocess_image's per-channel arithmetic, tabulated for every uint8 value and laid out HWC.
NORMALIZE = ((np.arange(256, dtype=np.float32)[:, None] / 255.0 - MEAN) / STD).reshape(1, 256, 3)


def preprocess(content: bytes) -> np.ndarray:
    """Return the same values as calibration.preprocess_image, in the MPK's HWC layout."""
    with Image.open(io.BytesIO(content)) as image:
        letterboxed = letterbox(image.convert("RGB"))
    return cv2.LUT(np.asarray(letterboxed), NORMALIZE)


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def decode(logits: np.ndarray, boxes: np.ndarray, image_id: int, width: int, height: int) -> np.ndarray:
    """Return COCO result rows [image_id, x, y, w, h, score, category_id], one per query."""
    if logits.shape != (1, QUERY_COUNT, CLASSES) or boxes.shape != (1, QUERY_COUNT, 4):
        raise ValueError(f"unexpected MPK outputs: logits={logits.shape}, boxes={boxes.shape}")
    if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
        raise ValueError("MPK output contains non-finite values")
    shifted = np.exp(logits[0] - logits[0].max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    categories = probabilities[:, :-1].argmax(axis=1)
    scores = probabilities[np.arange(QUERY_COUNT), categories]
    # Boxes are (center x, center y, width, height) as fractions of the letterboxed input. Remove the padding
    # and undo the resize to reach image pixels.
    (resized_width, resized_height), (left, top) = letterbox_geometry(width, height)
    scale_x, scale_y = resized_width / width, resized_height / height
    cx, cy, w, h = boxes[0].T
    x1 = ((cx - w / 2) * INPUT_WIDTH - left) / scale_x
    y1 = ((cy - h / 2) * INPUT_HEIGHT - top) / scale_y
    ids = np.full(QUERY_COUNT, image_id)
    return np.stack([ids, x1, y1, w * INPUT_WIDTH / scale_x, h * INPUT_HEIGHT / scale_y, scores, categories], axis=1)


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


def mpk_inference(mpk: Path) -> Inference:
    """Run the compiled model through PyNeat, building its runner from the first image."""
    import pyneat

    model = pyneat.Model(str(mpk))
    runner = None

    def infer(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal runner
        tensors = [pyneat.Tensor.from_numpy(image, copy=False)]
        if runner is None:
            runner = model.build(tensors)
        logits, boxes = (output.to_numpy(copy=False) for output in runner.run(tensors, timeout_ms=120000))
        return logits, boxes

    return infer


def evaluate(dataset: Path, infer: Inference, limit: int | None = None) -> dict[str, Any]:
    """Score infer on COCO val2017, or on a seeded sample of limit images, as model_report expects a dataset."""
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, limit)) if limit else available
    records = ground_truth.loadImgs(image_ids)

    rows: list[np.ndarray] = []
    started = time.monotonic()
    images = prefetched(lambda record: preprocess((dataset / IMAGES / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        logits, boxes = infer(image)
        rows.append(decode(logits, boxes, record["id"], record["width"], record["height"]))
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    inference_seconds = time.monotonic() - started

    detections = np.concatenate(rows)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
    scorer = COCOeval_faster(ground_truth, results, "bbox", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    stats = coco_stats(scorer)
    return {
        "metrics": stats,
        "split": IMAGES,
        "evaluated_samples": len(image_ids),
        "available_samples": len(available),
        "details": {
            "detections": len(detections),
            "inference_seconds": round(inference_seconds, 1),
            "scoring_seconds": round(time.monotonic() - started, 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)

    result = evaluate(Path(dataset_dir), mpk_inference(parse_mpks(args.mpk)), args.limit)
    report = model_report.from_accuracy({dataset_id: result})
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
