"""Measure D-FINE Small COCO accuracy through the same native inference path as performance.py.

The image enters native preprocessing at its original size. Both compiled components and native
DetrFocal BoxDecode execute on the board; Python only gathers proposals and formats COCO rows.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
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

IMAGE_SIZE = 640
QUERY_COUNT = 300
CLASSES = 80
TIMEOUT_MS = 184000
QUEUE_DEPTH = 4
WORKERS, PREFETCH = 4, 8
ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
MAX_DETECTIONS = [1, 10, 100]
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
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The decoder consumes tensor inputs; preserve the MPK's quantization and layout adapters.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def image_options(max_width: int, max_height: int) -> pyneat.ModelOptions:
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
    if content.shape != (1, 8400, 256) or scores.shape != (1, 8400) or references.shape != (1, 8400, 4):
        raise ValueError("Unexpected Small query boundary")
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


def decoder_graph(model: pyneat.Model) -> pyneat.Graph:
    """Decode public FP32 heads; preserve every MPK input/output adapter."""
    route = pyneat.ModelRouteOptions()
    route.include_input, route.include_output = True, False
    graph = model.graph(route)
    graph.add(
        pyneat.nodes.sima_box_decode(
            pyneat.BoxDecodeType.DetrFocal,
            original_width=0,
            original_height=0,  # Per-frame native preprocessing metadata supplies geometry.
            model_width=IMAGE_SIZE,
            model_height=IMAGE_SIZE,
            detection_threshold=0.0,
            nms_iou_threshold=0.0,
            top_k=QUERY_COUNT,
            decode_type_option=pyneat.BoxDecodeTypeOption.InterleavedByHeadLogit,
            source_storage=pyneat.BoxDecodeSourceStorage.DenseHwc,
            detess=False,
            dequant=False,
            resize_mode=pyneat.ResizeMode.Stretch,
        )
    )
    graph.add(pyneat.nodes.output())
    return graph


def native_rows(outputs, record: dict, categories: np.ndarray) -> np.ndarray:
    """COCO [id,x,y,w,h,score,category] rows, retaining all native pairs and empty boxes."""
    if len(outputs) != 1 or outputs[0].semantic.detection is None or outputs[0].semantic.detection.format != "BBOX":
        raise RuntimeError("Expected one native BBOX publication")
    boxes = pyneat.decode_bbox(outputs, strict=True)[0].to_numpy(copy=False)
    if boxes.shape != (QUERY_COUNT, 6) or boxes.dtype != np.float32 or not np.isfinite(boxes).all():
        raise ValueError("Expected 300 finite native XYXY/score/class rows")
    classes = boxes[:, 5]
    if not np.equal(classes, np.floor(classes)).all() or not ((classes >= 0) & (classes < CLASSES)).all():
        raise ValueError("Invalid native class index")
    if not ((boxes[:, 4] >= 0) & (boxes[:, 4] <= 1)).all():
        raise ValueError("Invalid native score")
    # BBOX rounds origin and span separately. Their summed endpoint can be one
    # pixel beyond the clipped float endpoint; preserve it, as the native ABI does.
    limits = np.asarray([record["width"], record["height"]])
    if not ((boxes[:, :2] >= 0) & (boxes[:, :2] <= limits)).all():
        raise ValueError("Native origins exceed original image bounds")
    if not ((boxes[:, 2:4] >= 0) & (boxes[:, 2:4] <= limits + 1)).all():
        raise ValueError("Native endpoints exceed rounded original image bounds")
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    if (widths < 0).any() or (heights < 0).any():
        raise ValueError("Invalid native extent")
    return np.column_stack(
        (
            np.full(QUERY_COUNT, record["id"], dtype=np.int64),
            boxes[:, 0],
            boxes[:, 1],
            widths,
            heights,
            boxes[:, 4],
            categories[classes.astype(np.int64)],
        )
    )


def iter_inference(
    mpks: dict[str, Path], records: list[dict], dataset: Path
) -> Iterator[tuple[dict, list[pyneat.Tensor]]]:
    options = pyneat.RunOptions()
    options.queue_depth = QUEUE_DEPTH
    options.overflow_policy = pyneat.OverflowPolicy.Block
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    backbone_model = pyneat.Model(
        str(mpks["backbone"]),
        image_options(
            max(r["width"] for r in records),
            max(r["height"] for r in records),
        ),
    )
    decoder_model = pyneat.Model(str(mpks["decoder"]), tensor_options())
    graph = decoder_graph(decoder_model)
    images = prefetched(lambda r: read_rgb((dataset / IMAGES / r["file_name"]).read_bytes()), records)
    backbone = decoder = None
    started = time.monotonic()
    try:
        for count, (record, image) in enumerate(zip(records, images, strict=True), 1):
            tensor = pyneat.Tensor.from_numpy(
                image,
                copy=True,
                image_format=pyneat.PixelFormat.RGB,
                memory=pyneat.TensorMemory.EV74,
            )
            if backbone is None:
                backbone = backbone_model.build([tensor], run_options=options)
            memory, content, scores, references = backbone.run([tensor], timeout_ms=TIMEOUT_MS)
            require_original_geometry(memory, record["width"], record["height"])
            inputs = [memory, *select_queries(content, scores, references)]
            if decoder is None:
                decoder = graph.build(inputs, options=options)
            if not decoder.push(inputs):
                raise RuntimeError("Decoder rejected a frame")
            sample = decoder.pull(TIMEOUT_MS)
            if sample is None:
                raise RuntimeError("Decoder/BoxDecode output timed out")
            yield record, sample.tensors
            del sample, inputs, memory, content, scores, references, tensor
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    finally:
        if decoder is not None:
            decoder.close()
        if backbone is not None:
            backbone.close()


def require_original_geometry(memory, width, height):
    """Do not invent or overwrite source geometry on a retained carrier."""
    meta = memory.semantic.preprocess
    if meta is None or (meta.original_width, meta.original_height) != (width, height):
        raise RuntimeError("Native preprocessing did not preserve this frame's original dimensions")
    if (meta.resized_width, meta.resized_height) != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ) or meta.resize_mode != "stretch":
        raise RuntimeError("Native preprocessing geometry differs from the D-FINE stretch contract")
    return meta


def score_coco(ground_truth, detections, image_ids, category_ids):
    """Official COCO tensors, category-partitioned to bound evaluator memory.

    Merge precision/recall bins BEFORE official summarization; never average
    already summarized class APs. This also preserves official -1 missing bins.
    """
    precision, recall, scores = [], [], []
    params = None
    for category in category_ids:
        rows = detections[detections[:, 6] == category]
        result = ground_truth.loadRes(rows)
        scorer = COCOeval_faster(ground_truth, result, "bbox", print_function=log)
        scorer.params.imgIds = image_ids
        scorer.params.catIds = [category]
        scorer.params.maxDets = MAX_DETECTIONS
        scorer.evaluate()
        scorer.accumulate()
        precision.append(scorer.eval["precision"].copy())
        recall.append(scorer.eval["recall"].copy())
        scores.append(scorer.eval["scores"].copy())
        params = copy.deepcopy(scorer.params)
        del scorer, result, rows
        gc.collect()
    merged = COCOeval_faster(ground_truth, None, "bbox", print_function=log)
    params.catIds = list(category_ids)
    merged.params = params
    merged.eval = {
        "precision": np.concatenate(precision, axis=2),
        "recall": np.concatenate(recall, axis=1),
        "scores": np.concatenate(scores, axis=2),
        "params": copy.deepcopy(params),
        "counts": [
            len(params.iouThrs),
            len(params.recThrs),
            len(category_ids),
            len(params.areaRng),
            len(params.maxDets),
        ],
    }
    merged.summarize()
    return merged


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
    categories = np.asarray(sorted(ground_truth.getCatIds()), dtype=np.int64)
    if categories.shape != (CLASSES,) or not (categories > 0).all():
        raise ValueError("Expected the 80 COCO categories")
    rows = []
    started = time.monotonic()
    inference = iter_inference(parse_mpks(args.mpk), records, dataset)
    try:
        for record, outputs in inference:
            rows.append(native_rows(outputs, record, categories))
    finally:
        inference.close()
    inference_seconds = time.monotonic() - started
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        stats = coco_stats(score_coco(ground_truth, np.concatenate(rows), image_ids, categories))
    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "detections": QUERY_COUNT * len(image_ids),
                    "preprocessing": "original RGB -> native stretch640, RGB/255",
                    "postprocessing": "native DetrFocal BoxDecode, clipped/rounded integer BBOX, no extra filters",
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
