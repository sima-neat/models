"""Measure Mask2Former's COCO mask AP and AR with its MPKs on a Modalix board.

Given one --mpk <component>=<path> per component and --dataset coco-2017-val=<extracted directory>, the script
runs each val2017 image through the five components (pipeline.py), and decodes instances
as Mask2Former's instance inference does (pipeline.decode): the top 100 (query, class) pairs by class probability,
each mask upsampled to the image and thresholded at 0, each score the class probability times the mean mask
probability inside the mask. The instances are scored with COCO's mask AP and AR.
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

import cv2
import model_report
import numpy as np
import pyneat
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util
from pipeline import CLASSES, COMPONENTS, IMAGE_SIZE, MASK_SIZE, OUTPUT_SHAPES, QUERIES, preprocess, route

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# The published metrics are over the full val2017 split.
VAL2017_IMAGES = 5000
TIMEOUT_MS = 120000
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label. With maxDets [1, 10, 100], AR_all, AR_second
# and AR_third are the recall at 1, 10 and 100 masks per image.
METRICS = {
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
}
MAX_DETECTIONS = [1, 10, 100]
TOP_K = 100
# Images decode and instances decode on worker threads while the MPKs run; OpenCV releases the GIL.
WORKERS = 4
PREFETCH = 8


def output_layout(class_logits: np.ndarray, mask_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The transformer decoder's outputs as [QUERIES, CLASSES + 1] class logits and [QUERIES, MASK_SIZE,
    MASK_SIZE] mask logits, from NCHW or NHWC."""
    classes = np.asarray(class_logits, np.float32)
    if classes.size != QUERIES * (CLASSES + 1):
        raise ValueError(f"unexpected class output shape {classes.shape}")
    classes = (
        classes.reshape(QUERIES, CLASSES + 1)
        if classes.shape[-1] == CLASSES + 1
        else classes.reshape(CLASSES + 1, QUERIES).T
    )
    masks = np.asarray(mask_logits, np.float32)
    if masks.size != QUERIES * MASK_SIZE**2:
        raise ValueError(f"unexpected mask output shape {masks.shape}")
    if masks.shape[-1] == QUERIES and masks.shape[-2] == MASK_SIZE:
        masks = masks.reshape(MASK_SIZE, MASK_SIZE, QUERIES).transpose(2, 0, 1)
    return np.ascontiguousarray(classes), masks.reshape(QUERIES, MASK_SIZE, MASK_SIZE)


def instances(
    classes: np.ndarray, masks: np.ndarray, original_size: tuple[int, int]
) -> list[tuple[int, float, np.ndarray]]:
    """pipeline.decode with OpenCV's resizes, which are faster on the board's CPU and resize the same way."""
    logits = classes - classes.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = (probabilities / probabilities.sum(axis=1, keepdims=True))[:, :CLASSES]
    flat = probabilities.ravel()
    order = np.argsort(-flat, kind="stable")[:TOP_K]
    height, width = original_size
    found = []
    for index in order:
        query, label = divmod(int(index), CLASSES)
        upsampled = cv2.resize(masks[query], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        logit = cv2.resize(upsampled, (width, height), interpolation=cv2.INTER_LINEAR)
        binary = logit > 0
        mask_score = float((1 / (1 + np.exp(-logit[binary]))).sum() / (binary.sum() + 1e-6))
        found.append((label, float(flat[index]) * mask_score, binary))
    return found


def results(
    image_id: int, category_ids: list[int], classes: np.ndarray, masks: np.ndarray, original_size: tuple[int, int]
) -> list[dict]:
    """One image's COCO results."""
    items = []
    for label, score, binary in instances(classes, masks, original_size):
        rle = mask_util.encode(np.asfortranarray(binary[:, :, None].astype(np.uint8)))[0]
        rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
        items.append({"image_id": image_id, "category_id": category_ids[label], "segmentation": rle, "score": score})
    return items


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def evaluate(
    ground_truth: COCO,
    image_ids: list[int],
    image_dir: Path,
    infer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """Every image's COCO results. infer takes the backbone's HWC input and returns the transformer decoder's
    class and mask logits in any layout output_layout accepts."""
    category_ids = sorted(ground_truth.getCatIds())
    if len(category_ids) != CLASSES:
        raise SystemExit(f"expected {CLASSES} COCO categories, found {len(category_ids)}")
    records = ground_truth.loadImgs(image_ids)
    found: list[dict] = []
    started = time.monotonic()
    load = lambda record: preprocess(read_rgb(image_dir / record["file_name"]))  # noqa: E731
    with ThreadPoolExecutor(WORKERS) as decoders:
        pending: deque = deque()
        for count, (record, image) in enumerate(zip(records, prefetched(load, records)), start=1):
            classes, masks = output_layout(*infer(image))
            size = (record["height"], record["width"])
            pending.append(decoders.submit(results, record["id"], category_ids, classes, masks, size))
            while len(pending) > 2 * WORKERS:
                found.extend(pending.popleft().result())
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
        for future in pending:
            found.extend(future.result())
    return found


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


def score(ground_truth: COCO, found: list[dict], image_ids: list[int]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = ground_truth.loadRes(found)
    scorer = COCOeval_faster(ground_truth, loaded, "segm", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    return coco_stats(scorer)


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: the script prepares the inputs; Neat still quantizes and tessellates them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def hwc(value: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(value), copy=True, layout=pyneat.TensorLayout.HWC)


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """One component's outputs as fresh HWC inputs of the next. Neat 0.6 gives them a leading batch axis, and
    its own views of its outputs are read-only, which from_numpy rejects."""
    arrays = [tensor.to_numpy(copy=True) for tensor in tensors]
    return [hwc(array[0] if array.ndim == 4 else array) for array in arrays]


class Runners:
    """Every component on the board, each runner built from its first real input."""

    def __init__(self, mpks: dict[str, Path]) -> None:
        self.models = {name: pyneat.Model(str(mpks[name]), tensor_options()) for name in COMPONENTS}
        self.runners: dict[str, pyneat.ModelRunner] = {}

    def run(self, component: str, inputs: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
        if component not in self.runners:
            self.runners[component] = self.models[component].build(inputs)
        return self.runners[component].run(inputs, timeout_ms=TIMEOUT_MS)

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        available: dict[str, pyneat.Tensor] = {"image": hwc(image)}
        for component in COMPONENTS[:-1]:
            outputs = handoff(self.run(component, route(component, available)))
            available.update(zip(OUTPUT_SHAPES[component], outputs))
        classes, masks = self.run(COMPONENTS[-1], route(COMPONENTS[-1], available))
        # Copies, because the runner reuses its output buffers while a worker may still read them.
        return classes.to_numpy(copy=True), masks.to_numpy(copy=True)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    runners = Runners(parse_mpks(args.mpk))
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    if len(available) != VAL2017_IMAGES:
        raise SystemExit(f"expected {VAL2017_IMAGES} val2017 images, found {len(available)}")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    started = time.monotonic()
    found = evaluate(ground_truth, image_ids, dataset / IMAGES, runners.infer)
    inference_seconds = time.monotonic() - started
    if not found:
        raise SystemExit("the model found no instances")
    started = time.monotonic()
    stats = score(ground_truth, found, image_ids)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "instances_per_image": TOP_K,
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
