"""Score the complete YOLO-World package on official COCO 2017 validation.

The BF16 text encoder creates the 80 category vectors once; every image runs
through the compiled mixed detector. Raw sigmoid scores use the same multi-label
DFL decode and class-aware NMS as the FP32 baseline. The performance script
separately exercises native YOLOv8 BoxDecode at the application threshold."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import tarfile
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
from calibration import IMAGE_SIZE, letterbox_geometry, preprocess_image
from faster_coco_eval import COCO, COCOeval_faster

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
CLASSES = 80
COMPONENTS = ("text_encoder", "detector")
STRIDES = (8, 16, 32)
# Ultralytics' COCO validation settings: keep every class score above 0.001 (several classes per box), then
# class-aware NMS at IoU 0.7 over at most 30000 candidates, keeping at most 300 detections per image.
SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.7
MAX_CANDIDATES = 30000
MAX_DETECTIONS_PER_IMAGE = 300
# Ultralytics offsets each class's boxes by this many pixels, so one NMS pass never compares two classes.
CLASS_OFFSET = 7680
# Each box edge is a distribution over REG_MAX bins; its expected bin is the edge distance in cells.
REG_MAX = 16
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
# The MPK returns bbox_dfl_s8/16/32 then class_prob_s8/16/32 in NHWC. Anchors are the grid-cell centres of each
# level, row-major from stride 8 to stride 32, as Ultralytics' make_anchors lays them out.
GRIDS = [IMAGE_SIZE // stride for stride in STRIDES]
OUTPUT_SHAPES = [(1, size, size, channels) for channels in (4 * REG_MAX, CLASSES) for size in GRIDS]
ANCHORS = np.concatenate(
    [np.stack(np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5), axis=-1).reshape(-1, 2) for size in GRIDS]
)
ANCHOR_STRIDES = np.repeat(STRIDES, [size * size for size in GRIDS])[:, None]


def parse_mpks(values):
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("use --mpk text_encoder=<path> --mpk detector=<path>")
    result = dict(pairs)
    if len(pairs) != len(result) or set(result) != set(COMPONENTS):
        raise ValueError("expected exactly text_encoder and detector")
    return {name: Path(path) for name, path in result.items()}


def options(decode=False):
    import pyneat

    value = pyneat.ModelOptions()
    value.preprocess.kind = pyneat.InputKind.Tensor
    if decode:
        value.decode_type = pyneat.BoxDecodeType.YoloV8
        value.decode_type_option = pyneat.BoxDecodeTypeOption.GroupedByRoleProbability
        value.num_classes = 80
        value.score_threshold = .25
        value.nms_iou_threshold = .7
        value.top_k = 10
        value.boxdecode_original_width = value.boxdecode_original_height = 640
        value.boxdecode_resize_mode = pyneat.ResizeMode.Letterbox
    return value


class Component:
    def __init__(self, archive, decode=False):
        import pyneat

        with tarfile.open(archive, "r:gz") as source:
            (name,) = [n for n in source.getnames() if n.endswith("_mpk.json")]
            self.names = [item["name"] for item in json.load(source.extractfile(name))["input_nodes"]]
        self.model = pyneat.Model(str(archive), options(decode))
        self.specs = self.model.input_specs()
        self.runner = None

    def tensors(self, inputs):
        import pyneat

        if set(inputs) != set(self.names):
            raise ValueError(f"expected inputs {self.names}, got {list(inputs)}")
        result = []
        for name, spec in zip(self.names, self.specs, strict=True):
            value = np.asarray(inputs[name], np.float32)
            shape = tuple(spec.shape)
            if value.shape != shape:
                if value.shape[0] != 1 or value.shape[1:] != shape:
                    raise ValueError(f"{name}: expected {shape}, got {value.shape}")
                value = value[0]
            result.append(pyneat.Tensor.from_numpy(np.ascontiguousarray(value), copy=True,
                          layout=pyneat.TensorLayout.HWC if value.ndim in (3, 4) else pyneat.TensorLayout.Unknown))
        return result

    def run(self, tensors):
        if self.runner is None:
            self.runner = self.model.build(tensors)
        output = self.runner.run(tensors, timeout_ms=120000)
        if not output:
            raise RuntimeError("model produced no output")
        return [value.to_numpy(copy=True) for value in output]

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.runner = None


def prompt_assets(archive, resources=None):
    from tokenizers import Tokenizer

    # Compile staging nests each component; the installed package puts MPKs at its root.
    archive = Path(archive)
    beside_mpks = archive.parent.parent if archive.parent.name == "text_encoder" else archive.parent
    # The benchmark runner passes the pinned resources, which hold only the licenses; the assets are then beside the MPKs.
    root = Path(resources) if resources is not None and (Path(resources) / "prompt_assets").is_dir() else beside_mpks
    assets = root / "prompt_assets"
    table = np.load(assets / "token_embedding_table.npy", allow_pickle=False)
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    if table.shape != (49408, 512) or table.dtype != np.float32:
        raise ValueError("unexpected CLIP lookup table")
    tokenizer.no_truncation()
    tokenizer.enable_padding(length=77, direction="right", pad_id=49407, pad_token="<|endoftext|>")
    return tokenizer, table


def prompt_input(prompt, tokenizer, table):
    (encoded,) = tokenizer.encode_batch([prompt])
    if len(encoded.ids) != 77:
        raise ValueError("prompt exceeds CLIP's 77-token context")
    ids = np.asarray(encoded.ids, np.int64)
    end = int(ids.argmax())
    if ids[end] != 49407:
        raise ValueError("unexpected CLIP end-of-text token")
    selector = np.zeros((1, 1, 96, 1), np.float32)
    selector[0, 0, end, 0] = 1
    return {"token_embeddings": table[ids][None, None], "eos_selector": selector}


def encode_classes(archive, prompts, resources=None):
    if not 1 <= len(prompts) <= 80 or any(not prompt.strip() for prompt in prompts):
        raise ValueError("provide 1..80 nonempty prompts")
    tokenizer, table = prompt_assets(archive, resources)
    encoder = Component(archive)
    embeddings, mask = np.zeros((1, 80, 512), np.float32), np.zeros((1, 80), np.float32)
    try:
        for i, prompt in enumerate(prompts):
            (result,) = encoder.run(encoder.tensors(prompt_input(prompt, tokenizer, table)))
            vector = np.asarray(result, np.float32).reshape(512)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or abs(norm - 1) > .02:
                raise ValueError(f"invalid CLIP result for {prompt!r}")
            embeddings[0, i], mask[0, i] = vector / norm, 1
    finally:
        encoder.close()
    return embeddings, mask


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


def edge_distances(distributions: np.ndarray) -> np.ndarray:
    """Return the left, top, right and bottom distances in cells: the expected bin of each edge's softmax."""
    logits = distributions.reshape(-1, 4, REG_MAX).astype(np.float64)
    weights = np.exp(logits - logits.max(axis=2, keepdims=True))
    return (weights / weights.sum(axis=2, keepdims=True)) @ np.arange(REG_MAX, dtype=np.float64)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one x1, y1, x2, y2 box with each of boxes."""
    width = (np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])).clip(0)
    height = (np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])).clip(0)
    overlap = width * height
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return overlap / (area + areas - overlap + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedy NMS like torchvision: indices of the kept boxes, best first, dropping overlaps above IOU_THRESHOLD."""
    order = np.argsort(-scores, kind="stable")
    kept = []
    while len(order) and len(kept) < MAX_DETECTIONS_PER_IMAGE:
        best, order = order[0], order[1:]
        kept.append(best)
        order = order[iou(boxes[best], boxes[order]) <= IOU_THRESHOLD]
    return np.array(kept, dtype=np.int64)


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int, categories: np.ndarray) -> np.ndarray:
    """Return COCO result rows [image_id, x, y, w, h, score, category_id] for one image.

    Selects detections as Ultralytics' COCO validation does: every anchor/class pair scoring above SCORE_THRESHOLD,
    the MAX_CANDIDATES best of them, then class-aware NMS keeping at most MAX_DETECTIONS_PER_IMAGE.
    """
    shapes = [output.shape for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    distributions = np.concatenate([output.reshape(-1, 4 * REG_MAX) for output in outputs[:3]])
    probabilities = np.concatenate([output.reshape(-1, CLASSES) for output in outputs[3:]])
    if not (np.isfinite(distributions).all() and np.isfinite(probabilities).all()):
        raise ValueError("MPK output contains non-finite values")
    if probabilities.min() < -0.01 or probabilities.max() > 1.01:
        raise ValueError("expected sigmoid class probabilities, not logits")
    anchor, label = np.nonzero(probabilities > SCORE_THRESHOLD)
    scores = probabilities[anchor, label]
    if len(scores) > MAX_CANDIDATES:
        best = np.argsort(-scores, kind="stable")[:MAX_CANDIDATES]
        anchor, label, scores = anchor[best], label[best], scores[best]
    if not len(scores):
        return np.zeros((0, 7))
    # Decode each distinct candidate anchor once; several classes can share it.
    unique, inverse = np.unique(anchor, return_inverse=True)
    distances = edge_distances(distributions[unique])[inverse]
    corners = np.hstack([ANCHORS[anchor] - distances[:, :2], ANCHORS[anchor] + distances[:, 2:]])
    corners *= ANCHOR_STRIDES[anchor]
    kept = nms(corners + (label * CLASS_OFFSET)[:, None], scores)
    x1, y1, x2, y2 = image_boxes(corners[kept], width, height).T
    ids = np.full(kept.shape, image_id)
    return np.stack([ids, x1, y1, x2 - x1, y2 - y1, scores[kept], categories[label[kept]]], axis=1)


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


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=<path>; text_encoder and detector")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="package root; prompt_assets default to the MPK location")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    mpks = parse_mpks(args.mpk)
    model = Component(mpks["detector"])
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    # Ultralytics numbers the 80 COCO classes in category-id order.
    categories = np.array(sorted(ground_truth.getCatIds()))
    if len(categories) != CLASSES:
        raise SystemExit(f"expected {CLASSES} COCO categories, found {len(categories)}")

    category_records = sorted(ground_truth.dataset["categories"], key=lambda row: row["id"])
    embeddings, mask = encode_classes(mpks["text_encoder"], [row["name"] for row in category_records], args.resources)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "prompts_bf16.npz", text_embeddings=embeddings, prompt_mask=mask, category_ids=categories)
    rows: list[np.ndarray] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    images = prefetched(lambda record: preprocess_image((image_dir / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        tensors = model.tensors({"images": image[None], "text_embeddings": embeddings, "prompt_mask": mask})
        try:
            outputs = model.run(tensors)
        except Exception:
            log(f"Detector failed at image {count}/{len(records)}: {record['file_name']} (COCO id {record['id']})")
            with contextlib.suppress(Exception):
                model.close()
            raise
        rows.append(decode(outputs, record["id"], record["width"], record["height"], categories))
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    model.close()
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
