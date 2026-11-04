"""Measure Tiny Drone accuracy through Neat preprocessing and its public box decoder.

Evaluate the registered whole-sequence validation/test selections, not the
calibration images. Images enter at their original size as RGB UInt8. Neat
performs EV74 preprocessing, model execution and A65 BoxDecode; its published
boxes already use original-image coordinates. Do not normalize, sigmoid, apply
NMS or invert the letterbox a second time here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import model_report
import numpy as np
import pyneat
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

MODEL_SIZE = 1600
SCORE_THRESHOLD = 0.001
NMS_IOU = 0.60
TOP_K = 300
MAX_DETECTIONS = [1, 10, 100, 300]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def model_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.resize.pad_value = 114
    options.preprocess.resize.scaling_type = "BILINEAR"
    # Dimensions and quantization remain model-derived. COCO_YOLO means RGB /255,
    # with mean0/std1, not an additional division by255 in this script.
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.processcvu.pre_run_target = "EV74"
    options.processcvu.post_run_target = "A65"
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.decode_type_option = pyneat.BoxDecodeTypeOption.GroupedByRoleLogit
    options.num_classes = 1
    options.score_threshold = SCORE_THRESHOLD
    options.nms_iou_threshold = NMS_IOU
    options.top_k = TOP_K
    return options


def dataset_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe dataset path: {relative}")
    return resolved


def validate_dataset_identity(manifest: dict[str, Any], source_id: str | None = None) -> None:
    if (manifest.get("schema"), manifest.get("schema_version")) != ("sima.anti-uav-evaluation", 1):
        raise ValueError("unsupported Anti-UAV evaluation manifest")
    if manifest.get("dataset_id") != "anti-uav-track3" or manifest.get("split") not in {"val", "test"}:
        raise ValueError("accuracy requires a registered held-out Anti-UAV validation/test split")
    if manifest.get("variant") != f"sequence-safe-{manifest['split']}-v1":
        raise ValueError("dataset variant does not match its declared held-out split")
    expected = f"anti-uav-{manifest['split']}"
    if source_id is not None and source_id != expected:
        raise ValueError(f"dataset source ID {source_id!r} does not match its manifest: {expected}")


def load_dataset(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the registered manifest and its exact COCO annotations before device use."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validate_dataset_identity(manifest)
    rows = manifest.get("images", [])
    if not rows or len(rows) != manifest.get("counts", {}).get("images"):
        raise ValueError("dataset manifest has missing images or inconsistent image counts")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError("dataset contains duplicate image identities or paths")
    payload = dataset_file(root, manifest["annotation_file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["annotation_sha256"]:
        raise ValueError("COCO annotation checksum does not match the dataset manifest")
    annotations = json.loads(payload)
    if annotations.get("categories") != [{"id": 1, "name": "drone"}]:
        raise ValueError("Anti-UAV annotations must map category1 to drone")
    images = annotations.get("images", [])
    if len(images) != len(rows) or {image["id"] for image in images} != set(by_id):
        raise ValueError("COCO annotations and manifest have different image coverage")
    for image in images:
        row = by_id[image["id"]]
        if any(image.get(field) != row[field] for field in ("file_name", "width", "height")):
            raise ValueError(f"COCO image metadata differs from the manifest: {image['id']}")
        if image["width"] <= 0 or image["height"] <= 0:
            raise ValueError(f"invalid image dimensions: {image['id']}")
    boxes = annotations.get("annotations", [])
    if len(boxes) != manifest["counts"]["annotations"] or len({box["id"] for box in boxes}) != len(boxes):
        raise ValueError("COCO annotations have duplicate identities or inconsistent counts")
    for box in boxes:
        bbox = box.get("bbox", [])
        if (
            box.get("image_id") not in by_id
            or box.get("category_id") != 1
            or box.get("iscrowd", 0) != 0
            or len(bbox) != 4
            or not all(math.isfinite(value) for value in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
            or not math.isfinite(box.get("area", math.nan))
            or box["area"] != bbox[2] * bbox[3]
        ):
            raise ValueError(f"invalid COCO ground-truth box: {box.get('id')}")
    counts = Counter(box["image_id"] for box in boxes)
    if any(counts[row["id"]] != row["annotation_count"] for row in rows):
        raise ValueError("per-image annotation counts differ from the manifest")
    return manifest, annotations


def read_rgb(root: Path, row: dict[str, Any]) -> np.ndarray:
    payload = dataset_file(root, row["file_name"]).read_bytes()
    if len(payload) != row["size_bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"image checksum/size mismatch: {row['file_name']}")
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"image geometry differs from its manifest: {row['file_name']}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def prefetched(root: Path, rows: Sequence[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """Bounded, ordered JPEG decoding while the existing device route executes."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = deque((row, pool.submit(read_rgb, root, row)) for row in rows[:4])
        for row in rows[4:]:
            previous, image = pending.popleft()
            yield previous, image.result()
            pending.append((row, pool.submit(read_rgb, root, row)))
        while pending:
            row, image = pending.popleft()
            yield row, image.result()


def prediction_rows(decoded: Any, image_id: int, width: int, height: int) -> np.ndarray:
    """COCO rows from public boxes, retaining empty images and zero-area detections."""
    if len(decoded.boxes) > TOP_K:
        raise ValueError("BoxDecode published more detections than configured")
    records = []
    previous_score = math.inf
    for box in decoded.boxes:
        values = (box.x1, box.y1, box.x2, box.y2, box.score)
        if not all(math.isfinite(value) for value in values) or box.class_id != 0:
            raise ValueError("BoxDecode returned a non-finite value or a non-drone class")
        if not (0 <= box.x1 <= box.x2 <= width and 0 <= box.y1 <= box.y2 <= height):
            raise ValueError("BoxDecode returned geometry outside the original image")
        if not float(np.float32(SCORE_THRESHOLD)) <= box.score <= 1 or box.score > previous_score:
            raise ValueError("BoxDecode returned an invalid score or score order")
        previous_score = box.score
        records.append((image_id, box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1, box.score, 1))
    return np.asarray(records, dtype=np.float64).reshape(-1, 7)


def coco_metrics(scorer: Any) -> dict[str, float]:
    """Use named area/cap axes, not summarize()'s positional maxDets assumptions."""
    metrics = {}
    for area_index, area in enumerate(scorer.params.areaRngLbl):
        cap = scorer.params.maxDets.index(100)
        values = scorer.eval["precision"][:, :, :, area_index, cap]
        valid = values[values >= 0]
        if not valid.size:
            continue  # No GT in this optional area range; never publish -1 as a ratio.
        suffix = "" if area == "all" else f"_{area}"
        metrics["coco_map" if area == "all" else f"coco_ap{suffix}"] = float(valid.mean())
        if area == "all":
            for threshold, name in ((0.5, "coco_ap50"), (0.75, "coco_ap75")):
                selected = values[np.isclose(scorer.params.iouThrs, threshold)]
                metrics[name] = float(selected[selected >= 0].mean())
        for maximum in [1, 10, 100] if area == "all" else [100]:
            recall = scorer.eval["recall"][:, :, area_index, scorer.params.maxDets.index(maximum)]
            metrics[f"coco_ar{maximum}" if area == "all" else f"coco_ar{suffix}"] = float(recall[recall >= 0].mean())
    if not {"coco_map", "coco_ap50"} <= metrics.keys():
        raise ValueError("the evaluated selection has no ground-truth objects to score")
    return metrics


def score_detections(
    annotations: dict[str, Any], detections: np.ndarray, image_ids: list[int]
) -> tuple[dict[str, float], dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = annotations
        ground_truth.createIndex()
        if len(detections):
            predicted = ground_truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {
                "images": annotations["images"],
                "categories": annotations["categories"],
                "annotations": [],
            }
            predicted.createIndex()
    scorer = COCOeval_faster(ground_truth, predicted, "bbox", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.params.catIds = [1]
    scorer.params.maxDets = MAX_DETECTIONS
    scorer.evaluate()
    scorer.accumulate()
    metrics = coco_metrics(scorer)
    cap = scorer.params.maxDets.index(TOP_K)
    precision = scorer.eval["precision"][:, :, :, 0, cap]
    recall = scorer.eval["recall"][:, :, 0, cap]
    supplemental = {
        "ap50_95_max300": float(precision[precision >= 0].mean()),
        "ap50_max300": float(precision[0][precision[0] >= 0].mean()),
        "ar50_95_max300": float(recall[recall >= 0].mean()),
        "recall50_max300": float(recall[0][recall[0] >= 0].mean()),
    }
    # A separate single-IoU/area/cap evaluation makes matching identities
    # unambiguous; the multi-threshold summary's merged matches cannot do that.
    matching = COCOeval_faster(ground_truth, predicted, "bbox", print_function=log)
    matching.params.imgIds = image_ids
    matching.params.catIds = [1]
    matching.params.iouThrs = np.array([0.5])
    matching.params.areaRng = [[0, 1e10]]
    matching.params.areaRngLbl = ["all"]
    matching.params.maxDets = [TOP_K]
    matching.evaluate()
    matching.accumulate()
    matches = {int(pair.split("_", 1)[1]) for pair in matching.eval["matched"]}
    selected = set(image_ids)
    dimensions = {image["id"]: (image["width"], image["height"]) for image in annotations["images"]}
    tiny = set()
    for box in annotations["annotations"]:
        if box["image_id"] in selected:
            gain = min(MODEL_SIZE / side for side in dimensions[box["image_id"]])
            if box["area"] * gain * gain < 256:
                tiny.add(box["id"])
    supplemental.update(
        tiny_ground_truth=len(tiny),
        tiny_matched50=len(tiny & matches),
        tiny_recall50_max300=len(tiny & matches) / len(tiny) if tiny else None,
        tiny_definition="GT area <256 square pixels at 1600 letterbox scale; COCO one-to-one IoU50 matches, max300",
    )
    return metrics, supplemental


def evaluate_dataset(model: Any, root: Path, limit: int | None = None) -> dict[str, Any]:
    manifest, annotations = load_dataset(root)
    rows = manifest["images"]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = sorted(random.Random(0).sample(rows, min(limit, len(rows))), key=lambda row: row["id"])
    runner = None
    predictions = []
    started = time.monotonic()
    try:
        for count, (row, image) in enumerate(prefetched(root, rows), start=1):
            tensor = pyneat.Tensor.from_numpy(image, copy=False, image_format=pyneat.PixelFormat.RGB)
            if runner is None:
                runner = model.build([tensor])
            outputs = runner.run([tensor], timeout_ms=120000)
            if len(outputs) != 1:
                raise ValueError("expected one published BoxDecode output")
            decoded = pyneat.detections.decode_bbox_tensor(outputs[0], row["width"], row["height"], TOP_K, True)
            predictions.append(prediction_rows(decoded, row["id"], row["width"], row["height"]))
            if count == 1 or count % 250 == 0 or count == len(rows):
                log(f"{manifest['split']}: {count}/{len(rows)} images")
    finally:
        if runner is not None:
            runner.close()
    inference_seconds = time.monotonic() - started
    detections = np.concatenate(predictions)
    started = time.monotonic()
    metrics, supplemental = score_detections(annotations, detections, [row["id"] for row in rows])
    return {
        "metrics": metrics,
        "split": manifest["split"],
        "evaluated_samples": len(rows),
        "available_samples": len(manifest["images"]),
        "details": {
            "detections": len(detections),
            "inference_seconds": round(inference_seconds, 3),
            "scoring_seconds": round(time.monotonic() - started, 3),
            "evaluator": {
                "name": "faster-coco-eval",
                "version": version("faster-coco-eval"),
                "numpy_version": np.__version__,
                "pillow_version": version("Pillow"),
            },
            "selection": {"seed": 0, "requested_limit": limit, "full_split": len(rows) == len(manifest["images"])},
            "protocol": {
                "input": "native-size RGB UInt8",
                "model_size": MODEL_SIZE,
                "letterbox_fill": 114,
                "preprocessing": "EV74 bilinear COCO_YOLO; RGB/255, model-owned quantization/layout",
                "decode": "A65 YoloV26 GroupedByRoleLogit; public rounded original-image BBOX",
                "score_threshold": SCORE_THRESHOLD,
                "nms_iou": NMS_IOU,
                "top_k": TOP_K,
                "standard_metrics_max_detections": 100,
                "class_mapping": {"0": "drone/category1"},
            },
            "dataset": {
                "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                "available_counts": manifest["counts"],
                **{
                    key: manifest[key]
                    for key in (
                        "variant",
                        "source_archive_sha256",
                        "source_split_manifest_sha256",
                        "annotation_sha256",
                        "canonicalization_policy",
                        "limitations",
                    )
                },
            },
            "supplemental": supplemental,
        },
    }


def parse_named(values: list[str], label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{label} requires unique nonempty name=path arguments")
        result[name] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<compiled archive>")
    parser.add_argument("--dataset", action="append", required=True, help="<dataset source id>=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="seeded per-dataset sample limit, for local screening only")
    args = parser.parse_args(argv)
    mpks = parse_named(args.mpk, "--mpk")
    if set(mpks) != {"model"}:
        parser.error("--mpk must name exactly model=<path>")
    datasets = parse_named(args.dataset, "--dataset")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for name, directory in datasets.items():
        validate_dataset_identity(json.loads((directory / "manifest.json").read_text(encoding="utf-8")), name)
    model = pyneat.Model(str(mpks["model"]), model_options())
    report = model_report.from_accuracy(
        {name: evaluate_dataset(model, directory, args.limit) for name, directory in datasets.items()}
    )
    report.save(args.output)
    log(json.dumps({name: result["metrics"] for name, result in report.document["datasets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
