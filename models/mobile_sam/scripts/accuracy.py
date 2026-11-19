"""Measure MobileSAM's COCO mask AP and AR with box prompts, with its MPKs on a Modalix board.

Given one --mpk <component>=<path> per component and --dataset coco-2017-val=<extracted directory>, the script
encodes each val2017 image once, with Neat preparing it as performance.py does, then prompts the decoder with the
ground-truth box of each object that is not a crowd, as segment_anything's predictor does with
multimask_output=False: the box's two corners, and the first of the decoder's masks. Neat's MobileSAM decoder
upsamples that mask to the image as segment_anything does and thresholds it at 0; the mask is scored with COCO's mask
AP and AR under the object's own category, with the decoder's predicted IoU as its score. The boxes come from the
annotations, so the scores measure the masks alone, not detection.
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
from typing import TYPE_CHECKING, Any

import cv2
import model_report
import numpy as np
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util
from pipeline import COMPONENTS, PIXEL_MEAN, PIXEL_STD, PromptEncoding, prompt_inputs

if TYPE_CHECKING:
    import pyneat

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# The published metrics are over the full val2017 split.
VAL2017_IMAGES = 5000
TIMEOUT_MS = 120000
# The largest val2017 image side; Neat sizes the encoder's input buffers for it.
MAX_IMAGE_SIDE = 640
# SAM pads with zeros after normalizing, which is the mean pixel (124, 116, 104); Neat pads one byte for all channels.
PAD_VALUE = 116
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
# Images decode and masks encode on worker threads while the MPKs run; OpenCV releases the GIL.
WORKERS = 4
PREFETCH = 8


def box_prompt(bbox: Sequence[float], original_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The decoder's point inputs for a COCO x, y, width, height box: its top-left and bottom-right corners."""
    x, y, width, height = bbox
    return prompt_inputs([(x, y), (x + width, y + height)], ["box_top_left", "box_bottom_right"], original_size)


def result(annotation: dict[str, Any], mask: np.ndarray, score: float) -> dict:
    """The COCO result for one object: the decoder's first mask on the image, scored by its predicted IoU."""
    rle = mask_util.encode(np.asfortranarray(mask[:, :, None].astype(np.uint8)))[0]
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return {
        "image_id": annotation["image_id"],
        "category_id": annotation["category_id"],
        "segmentation": rle,
        "score": float(score),
    }


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
    encode: Callable[[np.ndarray], Any],
    decode: Callable[[Any, dict[str, np.ndarray]], tuple[np.ndarray, float]],
) -> tuple[list[dict], int]:
    """Every object's COCO result, and the number of decoder runs. encode takes the image's RGB pixels and
    returns its embedding in any form decode takes; decode returns the first mask, as the image's binary pixels,
    and its predicted IoU.

    Images run grouped by size, because the decoder's masks have the image's size and Neat builds a decoder runner
    for each size. The order changes no score."""
    records = sorted(ground_truth.loadImgs(image_ids), key=lambda record: (record["height"], record["width"]))
    results: list[dict] = []
    started, prompts = time.monotonic(), 0
    load = lambda record: read_rgb(image_dir / record["file_name"])  # noqa: E731
    with ThreadPoolExecutor(WORKERS) as encoders:
        pending: deque = deque()
        for count, (record, image) in enumerate(zip(records, prefetched(load, records)), start=1):
            size = (record["height"], record["width"])
            embedding = encode(image)
            for annotation in ground_truth.loadAnns(ground_truth.getAnnIds(imgIds=record["id"], iscrowd=False)):
                mask, score = decode(embedding, box_prompt(annotation["bbox"], size))
                pending.append(encoders.submit(result, annotation, mask, score))
                prompts += 1
            while len(pending) > 4 * WORKERS:
                results.append(pending.popleft().result())
            if count == 1 or count % 250 == 0 or count == len(records):
                log(
                    f"{count}/{len(records)} images, {prompts} prompts, "
                    f"{(time.monotonic() - started) / count * 1000:.0f} ms/image"
                )
        results.extend(future.result() for future in pending)
    return results, prompts


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


def score(ground_truth: COCO, results: list[dict], image_ids: list[int]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = ground_truth.loadRes(results)
    scorer = COCOeval_faster(ground_truth, loaded, "segm", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    return coco_stats(scorer)


def image_options() -> pyneat.ModelOptions:
    """The image encoder's, as performance.image_options explains: Neat resizes the longer side to 1024 with
    antialiased bilinear resampling, places the image top-left, pads and normalizes it as SAM does."""
    import pyneat

    options = pyneat.ModelOptions()
    preprocess = options.preprocess
    preprocess.kind = pyneat.InputKind.Image
    preprocess.input_max_width = preprocess.input_max_height = MAX_IMAGE_SIDE
    preprocess.enable = pyneat.AutoFlag.On
    preprocess.resize.enable = pyneat.AutoFlag.On
    preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    preprocess.resize.placement = pyneat.LetterboxPlacement.TopLeft
    preprocess.resize.pad_value = PAD_VALUE
    preprocess.resize.interpolation = pyneat.ResizeInterpolation.BilinearAntialiased
    preprocess.rescale_factor = 1.0
    preprocess.normalize.enable = pyneat.AutoFlag.On
    preprocess.normalize.mean = [float(value) for value in PIXEL_MEAN]
    preprocess.normalize.stddev = [float(value) for value in PIXEL_STD]
    preprocess.normalize.has_explicit_stats = True
    preprocess.floating_conversion = pyneat.PreprocFloatingConversion.NearestEven
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The prompt decoder's: the script prepares the prompt's tensors; Neat still casts them."""
    import pyneat

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def decoder_options() -> pyneat.ModelOptions:
    """The prompt decoder's, with Neat's MobileSAM decoder returning mask 0, as for a box prompt."""
    import pyneat

    options = tensor_options()
    options.decode_type = pyneat.BoxDecodeType.MobileSam
    options.multimask_output = False
    return options


def rgb(image: np.ndarray) -> pyneat.Tensor:
    """An RGB image as the encoder's input, in the memory Neat's preprocessing reads; from_numpy refuses a
    read-only array even when it copies."""
    import pyneat

    return pyneat.Tensor.from_numpy(
        np.require(image, requirements="CW"), copy=True, image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def hwc(value: np.ndarray) -> pyneat.Tensor:
    """An HWC array, or an NCHW batch of one, as the HWC tensor Neat takes, made writable for from_numpy."""
    import pyneat

    if value.ndim == 4:
        value = value[0].transpose(1, 2, 0)
    return pyneat.Tensor.from_numpy(np.require(value, requirements="CW"), copy=True, layout=pyneat.TensorLayout.HWC)


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """The encoder's outputs copied to EV74 memory for the decoder, as performance.handoff explains; the copy
    keeps the image geometry the decoder's mask resize reads, and outlives the encoder's next run."""
    return [tensor.cvu() for tensor in tensors]


class Runners:
    """The encoder and reusable decoder, each runner built from its first real input. The encoder's runner takes
    every image size; the decoder's masks have the image's size, so a new size gets a new decoder runner."""

    def __init__(self, mpks: dict[str, Path], resources: Path | None = None) -> None:
        import pyneat

        self.prompt_encoding = PromptEncoding.from_package(mpks["prompt_decoder"], resources)
        self.models = {
            "image_encoder": pyneat.Model(str(mpks["image_encoder"]), image_options()),
            "prompt_decoder": pyneat.Model(str(mpks["prompt_decoder"]), decoder_options()),
        }
        self.runners: dict[str, pyneat.ModelRunner] = {}
        self.size: tuple[int, int] | None = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        close_error = None
        for runner in reversed(list(self.runners.values())):
            try:
                runner.close()
            except BaseException as failure:
                close_error = close_error or failure
                log(f"runner cleanup failed: {failure}")
        if error is None and close_error is not None:
            raise close_error

    def run(self, component: str, inputs: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
        if component not in self.runners:
            self.runners[component] = self.models[component].build(inputs)
        return self.runners[component].run(inputs, timeout_ms=TIMEOUT_MS)

    def encode(self, image: np.ndarray) -> list[pyneat.Tensor]:
        if image.shape[:2] != self.size and "prompt_decoder" in self.runners:
            self.runners.pop("prompt_decoder").close()
        self.size = image.shape[:2]
        return handoff(self.run("image_encoder", [rgb(image)]))

    def decode(self, embedding: list[pyneat.Tensor], prompt: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        import pyneat

        inputs = [*embedding, hwc(self.prompt_encoding(prompt["point_coords"])), hwc(prompt["point_types"])]
        (decoded,) = pyneat.decode_segmentation(self.run("prompt_decoder", inputs))
        # Copies, because the runner reuses its output buffers while a worker may still read them.
        boxes, masks = decoded.boxes.to_numpy(copy=True), decoded.masks.to_numpy(copy=True)
        if masks.shape != (1, *self.size):
            raise ValueError(f"expected one {self.size} mask, got {masks.shape}")
        return masks[0] > 0, float(boxes[0, 4])


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
    parser.add_argument("--resources", type=Path, help="package root holding prompt-encoding.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    if len(available) != VAL2017_IMAGES:
        raise SystemExit(f"expected {VAL2017_IMAGES} val2017 images, found {len(available)}")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    started = time.monotonic()
    with Runners(parse_mpks(args.mpk), args.resources) as runners:
        results, prompts = evaluate(ground_truth, image_ids, dataset / IMAGES, runners.encode, runners.decode)
    inference_seconds = time.monotonic() - started
    if not results:
        raise SystemExit("the images have no objects to prompt")
    started = time.monotonic()
    stats = score(ground_truth, results, image_ids)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "prompts": prompts,
                    "prompt": "ground-truth box of each non-crowd object",
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
