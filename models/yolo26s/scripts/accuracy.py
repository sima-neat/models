"""Measure YOLO26s COCO bounding-box AP and AR with its MPK on a Modalix board."""

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
# Ultralytics' end-to-end selection and validation settings: the best 300 anchor/class pairs scoring above 0.001.
TOP_K = 300
SCORE_THRESHOLD = 0.001
# sigmoid(-7) is about 0.00091, below SCORE_THRESHOLD, so a logit at or below LOGIT_FLOOR is never kept and the
# decoder skips its sigmoid.
LOGIT_FLOOR = -7.0
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
# The MPK returns bbox_0..2 then class_logit_0..2 in NHWC. Anchors are the grid-cell centres of each
# level, row-major from stride 8 to stride 32, as Ultralytics' make_anchors lays them out.
GRIDS = [IMAGE_SIZE // stride for stride in STRIDES]
OUTPUT_SHAPES = [(1, size, size, channels) for channels in (4, CLASSES) for size in GRIDS]
ANCHORS = np.concatenate(
    [np.stack(np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5), axis=-1).reshape(-1, 2) for size in GRIDS]
)
ANCHOR_STRIDES = np.repeat(STRIDES, [size * size for size in GRIDS])[:, None]


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already letterboxed and scaled the image; Neat still quantizes and tessellates.
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


def image_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map x1, y1, x2, y2 boxes from the letterboxed input onto the original image and clip them to it."""
    gain, _, (left, top, _, _) = letterbox_geometry(width, height)
    mapped = (np.asarray(boxes, dtype=np.float64) - [left, top, left, top]) / gain
    mapped[:, 0::2] = mapped[:, 0::2].clip(0, width)
    mapped[:, 1::2] = mapped[:, 1::2].clip(0, height)
    return mapped


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


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int, categories: np.ndarray) -> np.ndarray:
    """Return COCO result rows [image_id, x, y, w, h, score, category_id] for one image.

    Selects detections as the exported Ultralytics head does: the TOP_K anchors by best class score, then the
    TOP_K best anchor/class pairs among them that score above SCORE_THRESHOLD. Only logits above LOGIT_FLOOR
    are ranked, which leaves the selection unchanged.
    """
    shapes = [output.shape for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    logits = np.concatenate([output.reshape(-1, CLASSES) for output in outputs[3:]])
    best = logits.max(axis=1)
    # A NaN or +inf logit reaches its anchor's maximum; a -inf one scores 0 and is never kept.
    if not (np.isfinite(distances).all() and np.isfinite(best).all()):
        raise ValueError("MPK output contains non-finite values")
    live_anchors = np.flatnonzero(best > LOGIT_FLOOR)
    anchors = live_anchors[top(sigmoid(best[live_anchors]), TOP_K)]
    pair_logits = logits[anchors].ravel()
    live_pairs = np.flatnonzero(pair_logits > LOGIT_FLOOR)
    scores = sigmoid(pair_logits[live_pairs])
    picked = top(scores, TOP_K)
    picked = picked[scores[picked] > SCORE_THRESHOLD]
    anchor, label = anchors[live_pairs[picked] // CLASSES], live_pairs[picked] % CLASSES
    # Box side distances are in grid cells: left and top from the anchor, then right and bottom.
    corners = np.hstack([ANCHORS[anchor] - distances[anchor, :2], ANCHORS[anchor] + distances[anchor, 2:]])
    x1, y1, x2, y2 = image_boxes(corners * ANCHOR_STRIDES[anchor], width, height).T
    ids = np.full(picked.shape, image_id)
    return np.stack([ids, x1, y1, x2 - x1, y2 - y1, scores[picked], categories[label]], axis=1)


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
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    # Ultralytics numbers the 80 COCO classes in category-id order.
    categories = np.array(sorted(ground_truth.getCatIds()))
    if len(categories) != CLASSES:
        raise SystemExit(f"expected {CLASSES} COCO categories, found {len(categories)}")

    runner = None
    rows: list[np.ndarray] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    images = prefetched(lambda record: preprocess_image((image_dir / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
        rows.append(decode(outputs, record["id"], record["width"], record["height"], categories))
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
