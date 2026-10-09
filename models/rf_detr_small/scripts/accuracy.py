"""Measure RF-DETR Small COCO bounding-box mAP with both MPKs on a Modalix board."""

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

import cv2
import numpy as np
import pyneat
from calibration import MEAN, STD, stable_topk_gather
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
SIZE = 512
QUERIES = 300
CLASSES = 91
TOP_K = 300
SCORE_THRESHOLD = 0.001
# Metric name -> index in COCOeval stats; names match the validation entry in model.yaml.
METRICS = {"coco_map": 0, "coco_ap50": 1, "coco_ap75": 2}
# Images decode on worker threads while the MPKs run; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# calibration.preprocess_image's per-channel arithmetic, tabulated for every uint8 value and laid out HWC.
NORMALIZE = ((np.arange(256, dtype=np.float32)[:, None] / 255.0 - MEAN.ravel()) / STD.ravel()).reshape(1, 256, 3)


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized and normalized the image; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def preprocess(content: bytes) -> np.ndarray:
    """Return the same values as calibration.preprocess_image, in the backbone's HWC layout."""
    with Image.open(io.BytesIO(content)) as image:
        resized = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
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
        self.model = pyneat.Model(str(path), tensor_options())
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
    if dets.shape != (1, QUERIES, 4) or labels.shape != (1, QUERIES, CLASSES):
        raise ValueError(f"unexpected transformer outputs: dets={dets.shape}, labels={labels.shape}")
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


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "transformer"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and transformer=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for backbone and transformer")
    parser.add_argument("--dataset", type=Path, required=True, help="extracted coco-2017/mini directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)

    mpks = parse_mpks(args.mpk)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(args.dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    valid = np.isin(np.arange(CLASSES), ground_truth.getCatIds())

    backbone, transformer = ModelArtifact(mpks["backbone"]), ModelArtifact(mpks["transformer"])
    rows: list[np.ndarray] = []
    started = time.monotonic()
    images = prefetched(lambda record: preprocess((args.dataset / IMAGES / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        features, scores, boxes = backbone(image)
        # The feature tensor goes to the transformer as is; only the proposals pass through NumPy.
        proposals = stable_topk_gather(view(scores), view(boxes), QUERIES)
        dets, labels = (view(output) for output in transformer(features, proposals))
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
    scoring_seconds = time.monotonic() - started

    result = {
        "core_version": pyneat.__version__,
        "dataset": {"split": IMAGES, "evaluated_samples": len(image_ids), "available_samples": len(available)},
        "metrics": {
            name: {"value": float(scorer.stats[index]), "unit": "ratio", "direction": "higher"}
            for name, index in METRICS.items()
        },
        "details": {
            "detections": len(detections),
            "inference_seconds": round(inference_seconds, 1),
            "scoring_seconds": round(scoring_seconds, 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: round(item["value"], 4) for name, item in result["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
