"""Measure D-FINE Nano COCO bounding-box AP and AR with both MPKs on a Modalix board.

The image enters native preprocessing at its original size. Both compiled components run on the board;
Python selects the top 300 queries between them and decodes the decoder's heads into COCO rows.
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

import model_report
import numpy as np
import pyneat
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

QUERY_COUNT = 300
CLASSES = 80
TIMEOUT_MS = 120000
QUEUE_DEPTH = 4
# Images decode on worker threads while the MPKs run; Pillow releases the GIL while it decodes.
WORKERS, PREFETCH = 4, 8
ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
MAX_DETECTIONS = [1, 10, 100]
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


def tensor_options() -> pyneat.ModelOptions:
    """The decoder's: its inputs are tensors, so Neat keeps the MPK's own quantization and layout adapters."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def image_options(max_width: int, max_height: int) -> pyneat.ModelOptions:
    """The backbone's: Neat stretches each original-size image to 640x640 and scales it to [0, 1]."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = [0.0, 0.0, 0.0]
    options.preprocess.normalize.stddev = [1.0, 1.0, 1.0]
    options.preprocess.normalize.has_explicit_stats = True
    options.preprocess.input_max_width = max_width
    options.preprocess.input_max_height = max_height
    # Keep the attention memory the decoder reads where its EV74 input stage can reach it.
    options.advanced_execution.postprocess_target = "EV74"
    return options


def read_rgb(content: bytes) -> np.ndarray:
    """Original RGB pixels, never pre-resized before the measured native pipeline."""
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def select_queries(content, scores, references) -> list[pyneat.Tensor]:
    """Own the gathered rows; leave attention memory on its existing device allocation."""
    meta = content.semantic.preprocess
    content, scores, references = (t.to_numpy(copy=False) for t in (content, scores, references))
    if content.shape != (1, 2000, 128) or scores.shape != (1, 2000) or references.shape != (1, 2000, 4):
        raise ValueError("Unexpected Nano query boundary")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (content, scores, references)):
        raise ValueError("Expected finite float32 query boundary tensors")
    order = np.argsort(-scores, axis=1, kind="stable")[0, :QUERY_COUNT]
    tensors = [
        pyneat.Tensor.from_numpy(
            x[0, order, :].reshape(1, QUERY_COUNT, x.shape[-1]),
            copy=True,
            layout=pyneat.TensorLayout.Unknown,
            memory=pyneat.TensorMemory.EV74,
        )
        for x in (content, references)
    ]
    for tensor in tensors:
        tensor.semantic.preprocess = meta
    return tensors


def view(tensor: pyneat.Tensor) -> np.ndarray:
    """Read-only view of an output; valid until its MPK runs again."""
    array = tensor.to_numpy(copy=False)
    if array.dtype != np.float32:
        raise ValueError(f"expected a float32 MPK output, got {array.dtype}")
    return array


def decode(boxes: np.ndarray, logits: np.ndarray, image_id: int, width: int, height: int, categories) -> np.ndarray:
    """COCO result rows [image_id, x, y, w, h, score, category_id] for one image.

    D-FINE's decode: sigmoid class scores, the 300 best query/class pairs (a query may keep several
    classes), float corner boxes scaled to the original image and clipped to it, no NMS.
    """
    if boxes.shape != (1, QUERY_COUNT, 4) or logits.shape != (1, QUERY_COUNT, CLASSES):
        raise ValueError(f"unexpected decoder outputs: pred_boxes={boxes.shape}, pred_logits={logits.shape}")
    if not (np.isfinite(boxes).all() and np.isfinite(logits).all()):
        raise ValueError("MPK output contains non-finite values")
    scores = (1.0 / (1.0 + np.exp(-np.clip(logits[0], -80, 80)))).ravel()
    order = np.argsort(-scores, kind="stable")[:QUERY_COUNT]
    cx, cy, w, h = boxes[0, order // CLASSES].T
    x1, x2 = np.clip((cx - w / 2) * width, 0, width), np.clip((cx + w / 2) * width, 0, width)
    y1, y2 = np.clip((cy - h / 2) * height, 0, height), np.clip((cy + h / 2) * height, 0, height)
    keep = (x2 > x1) & (y2 > y1)
    ids = np.full(order.shape, image_id)
    rows = np.stack([ids, x1, y1, x2 - x1, y2 - y1, scores[order], categories[order % CLASSES]], axis=1)
    return rows[keep]


def iter_inference(mpks: dict[str, Path], records: list[dict], dataset: Path) -> Iterator[tuple[dict, list]]:
    """Run every image through the backbone, the CPU query selection and the decoder, yielding its heads."""
    options = pyneat.RunOptions()
    options.queue_depth = QUEUE_DEPTH
    options.overflow_policy = pyneat.OverflowPolicy.Block
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    backbone_model = pyneat.Model(
        str(mpks["backbone"]),
        image_options(max(r["width"] for r in records), max(r["height"] for r in records)),
    )
    decoder_model = pyneat.Model(str(mpks["decoder"]), tensor_options())
    images = prefetched(lambda r: read_rgb((dataset / IMAGES / r["file_name"]).read_bytes()), records)
    backbone = decoder = None
    started = time.monotonic()
    try:
        for count, (record, image) in enumerate(zip(records, images, strict=True), 1):
            tensor = pyneat.Tensor.from_numpy(
                image, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
            )
            if backbone is None:
                backbone = backbone_model.build([tensor], run_options=options)
            memory, content, scores, references = backbone.run([tensor], timeout_ms=TIMEOUT_MS)
            inputs = [memory, *select_queries(content, scores, references)]
            if decoder is None:
                decoder = decoder_model.build(inputs, run_options=options)
            outputs = decoder.run(inputs, timeout_ms=TIMEOUT_MS)
            if len(outputs) != 2:
                raise RuntimeError(f"Expected the decoder's pred_boxes and pred_logits, got {len(outputs)} tensors")
            yield record, [view(output) for output in outputs]
            del outputs, inputs, memory, content, scores, references, tensor
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    finally:
        if decoder is not None:
            decoder.close()
        if backbone is not None:
            backbone.close()


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
    if len(values) != 2:
        raise SystemExit("--mpk requires exactly two distinct component bindings")
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "decoder"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and decoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="backbone=<path> and decoder=<path>")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="seeded sample for local checks only; default evaluates all images")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val":
        raise SystemExit("--dataset must name coco-2017-val")
    dataset = Path(dataset_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    if args.limit is not None and not 0 < args.limit <= len(available):
        raise SystemExit("--limit must be positive and no larger than the image count")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    # Class index -> COCO category ID, including the gaps in COCO's numbering.
    categories = np.asarray(sorted(ground_truth.getCatIds()), dtype=np.int64)
    if categories.shape != (CLASSES,) or not (categories > 0).all():
        raise ValueError("Expected the 80 COCO categories")
    rows = []
    started = time.monotonic()
    inference = iter_inference(parse_mpks(args.mpk), records, dataset)
    try:
        for record, (boxes, logits) in inference:
            rows.append(decode(boxes, logits, record["id"], record["width"], record["height"], categories))
    finally:
        inference.close()
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
    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "detections": int(len(detections)),
                    "preprocessing": "original RGB -> native stretch to 640x640, RGB/255",
                    "postprocessing": (
                        "CPU: sigmoid scores, top 300 query/class pairs, float boxes clipped to the image"
                    ),
                    "inference_seconds": round(inference_seconds, 1),
                    "scoring_seconds": round(time.monotonic() - started, 1),
                },
            }
        }
    )
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in stats.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
