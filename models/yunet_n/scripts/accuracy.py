"""Measure YuNet-n face-detection AP on the WIDER FACE validation split with its MPK on a Modalix board.

Each image is letterboxed on the host as calibration.py does (centered, zero-padded, BGR pixel values). The MPK returns
the 12 raw head maps; they are decoded as OpenCV's FaceDetectorYN decodes them, kept by NMS at the thresholds OpenCV
Zoo evaluates YuNet with, mapped back through the letterbox, and scored with the official WIDER FACE evaluation on
the easy, medium and hard subsets.

The evaluation is ported from OpenCV Zoo's tools/eval/datasets/widerface.py (opencv/opencv_zoo at
48b10fa505aeb51fbd127fc60a5c660040e823b7, Apache-2.0), a Python port of the official MATLAB evaluation. It keeps that
code's conventions, so OpenCV Zoo's published YuNet scores reproduce:
- every score is rescaled to [0, 1] by the minimum and maximum over the whole set;
- an image without detections contributes one placeholder detection, [10, 10, 20, 20] with score 0.002;
- in each image, every detection in descending score order takes the ground-truth face it overlaps most; at IoU 0.5 or
  more it recalls that face, or is ignored when the face is outside the subset scored;
- precision and recall are sampled at 1000 score thresholds, and AP is the area under the VOC precision envelope.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from calibration import HEADS, IMAGE_SIZE, OUTPUTS, STRIDES, decode_image, letterbox, letterbox_geometry

DATASET = "wider-face-val"
IMAGES = "WIDER_val/images"
GROUND_TRUTH = "eval_tools/ground_truth"
GRIDS = {stride: IMAGE_SIZE // stride for stride in STRIDES}
# OpenCV Zoo's YuNet evaluation settings: score threshold, NMS IoU and the most detections kept per image.
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
TOP_K = 5000
# Images decode and letterbox on worker threads while the MPK runs; OpenCV releases the GIL meanwhile.
WORKERS = 4
PREFETCH = 8
# The official WIDER FACE evaluation's subsets, match IoU, score thresholds and placeholder detection.
SETTINGS = ("easy", "medium", "hard")
IOU_THRESHOLD = 0.5
THRESHOLDS = 1000
PLACEHOLDER = np.array([[10.0, 10.0, 20.0, 20.0, 0.002]])


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def head_maps(outputs: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """The 12 head maps by name, each flattened to [H*W, C] in row-major cell order; Neat returns them channels-last."""
    if len(outputs) != len(OUTPUTS):
        raise RuntimeError(f"expected {len(OUTPUTS)} head maps, got {len(outputs)}")
    maps = {}
    for name, output in zip(OUTPUTS, outputs, strict=True):
        head, stride = name.split("_")
        grid, channels = GRIDS[int(stride)], HEADS[head]
        array = np.asarray(output, dtype=np.float32)
        if array.size != grid * grid * channels or array.shape[-1] != channels:
            raise RuntimeError(f"unexpected {name} shape {array.shape}")
        maps[name] = array.reshape(grid * grid, channels)
    return maps


def decode(maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's box [x, y, w, h], score and five [x, y] landmarks in input pixels, as FaceDetectorYN decodes them."""
    boxes, scores, landmarks = [], [], []
    for stride in STRIDES:
        grid = GRIDS[stride]
        cells = np.arange(grid * grid)
        column, row = (cells % grid).astype(np.float32), (cells // grid).astype(np.float32)
        # The face score is the geometric mean of the clipped class and objectness probabilities.
        cls = np.clip(maps[f"cls_{stride}"][:, 0], 0, 1)
        obj = np.clip(maps[f"obj_{stride}"][:, 0], 0, 1)
        offsets, points = maps[f"bbox_{stride}"], maps[f"kps_{stride}"].reshape(-1, 5, 2)
        width, height = np.exp(offsets[:, 2]) * stride, np.exp(offsets[:, 3]) * stride
        centre_x, centre_y = (column + offsets[:, 0]) * stride, (row + offsets[:, 1]) * stride
        boxes.append(np.stack([centre_x - width / 2, centre_y - height / 2, width, height], axis=1))
        scores.append(np.sqrt(cls * obj))
        landmarks.append((points + np.stack([column, row], axis=1)[:, None, :]) * stride)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(landmarks)


def detect(maps: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    """The image's face detections as [x, y, w, h, score] rows in image pixels, after NMS, best first."""
    import cv2

    boxes, scores, _ = decode(maps)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), SCORE_THRESHOLD, NMS_THRESHOLD, top_k=TOP_K)
    keep = np.asarray(keep, dtype=int).reshape(-1)
    gain, _, (left, top, _, _) = letterbox_geometry(width, height)
    found = boxes[keep].astype(np.float64)
    found[:, 0] = (found[:, 0] - left) / gain
    found[:, 1] = (found[:, 1] - top) / gain
    found[:, 2:4] /= gain
    # OpenCV Zoo rounds boxes to 0.1 pixel and scores to three decimals before scoring.
    return np.concatenate([np.round(found, 1), np.round(scores[keep], 3)[:, None]], axis=1)


def load_image(root: Path, item: tuple[str, str]) -> tuple[np.ndarray, int, int]:
    """The letterboxed model input for one validation image, with the image's width and height."""
    image = decode_image((root / IMAGES / item[0] / f"{item[1]}.jpg").read_bytes())
    height, width = image.shape[:2]
    return letterbox(image).astype(np.float32), width, height


def parse_dataset(parser: argparse.ArgumentParser, value: str) -> Path:
    dataset_id, _, directory = value.partition("=")
    if dataset_id != DATASET or not directory:
        parser.error(f"--dataset must name {DATASET}=<extracted directory>")
    # The archive extracts to one wider-face-val/ folder; take that folder or the directory holding it.
    root = Path(directory) / DATASET if (Path(directory) / DATASET).is_dir() else Path(directory)
    for required in (IMAGES, GROUND_TRUTH):
        if not (root / required).is_dir():
            parser.error(f"{root} has no {required}/")
    return root


def load_ground_truth(root: Path) -> list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]]:
    """Each validation image as (event, image name, face boxes [x, y, w, h], 1-based indices each subset scores)."""
    from scipy.io import loadmat

    directory = root / GROUND_TRUTH
    faces = loadmat(str(directory / "wider_face_val.mat"))
    subsets = {setting: loadmat(str(directory / f"wider_{setting}_val.mat"))["gt_list"] for setting in SETTINGS}
    images = []
    for index, event in enumerate(faces["event_list"]):
        boxes = faces["face_bbx_list"][index][0]
        for position, item in enumerate(faces["file_list"][index][0]):
            keep = {setting: subsets[setting][index][0][position][0].astype(int).ravel() for setting in SETTINGS}
            images.append((str(event[0][0]), str(item[0][0]), boxes[position][0].astype(float), keep))
    return images


def _overlaps(boxes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """IoU of [x1, y1, x2, y2] boxes against faces, counting pixels inclusively as the official evaluation does."""
    top_left = np.maximum(boxes[:, np.newaxis, 0:2], faces[:, 0:2])
    bottom_right = np.minimum(boxes[:, np.newaxis, 2:4], faces[:, 2:4])
    inner = np.prod(bottom_right - top_left + 1, axis=2) * (top_left < bottom_right).all(axis=2)
    box_area = np.prod(boxes[:, 2:4] - boxes[:, 0:2] + 1, axis=1)
    face_area = np.prod(faces[:, 2:4] - faces[:, 0:2] + 1, axis=1)
    return inner / (box_area[:, np.newaxis] + face_area - inner)


def _match(detections: np.ndarray, faces: np.ndarray, ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per detection, the faces recalled so far, and whether it is a counted proposal (1) or ignored (-1)."""
    boxes, targets = detections[:, :4].copy(), faces.copy()
    boxes[:, 2:4] += boxes[:, 0:2]
    targets[:, 2:4] += targets[:, 0:2]
    overlaps = _overlaps(boxes, targets)
    recalled_so_far = np.zeros(len(detections))
    recalled = np.zeros(len(faces))
    proposals = np.ones(len(detections))
    for row in range(len(detections)):
        best = overlaps[row].argmax()
        if overlaps[row, best] >= IOU_THRESHOLD:
            if ignore[best] == 0:
                recalled[best] = -1
                proposals[row] = -1
            elif recalled[best] == 0:
                recalled[best] = 1
        recalled_so_far[row] = np.count_nonzero(recalled == 1)
    return recalled_so_far, proposals


def _image_curve(detections: np.ndarray, proposals: np.ndarray, recalled_so_far: np.ndarray) -> np.ndarray:
    """Per score threshold: the counted proposals and the recalled faces among detections at or above it."""
    curve = np.zeros((THRESHOLDS, 2))
    for step in range(THRESHOLDS):
        above = np.where(detections[:, 4] >= 1 - (step + 1) / THRESHOLDS)[0]
        if len(above):
            last = above[-1]
            curve[step] = np.count_nonzero(proposals[: last + 1] == 1), recalled_so_far[last]
    return curve


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def evaluate(
    detections: Mapping[tuple[str, str], np.ndarray],
    ground_truth: list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]],
) -> dict[str, float]:
    """AP per subset for detections keyed by (event, image name), each an array of [x, y, w, h, score] rows."""
    rows = {}
    for event, name, _, _ in ground_truth:
        found = np.asarray(detections.get((event, name), np.zeros((0, 5))), dtype=float).reshape(-1, 5)
        found = found if len(found) else PLACEHOLDER.copy()
        rows[event, name] = found[np.argsort(-found[:, 4], kind="stable")]
    scores = np.concatenate([found[:, 4] for found in rows.values()])
    low, spread = scores.min(), scores.max() - scores.min()
    for found in rows.values():
        found[:, 4] = (found[:, 4] - low) / spread
    aps = {}
    for setting in SETTINGS:
        curve, faces_scored = np.zeros((THRESHOLDS, 2)), 0
        for event, name, faces, keep in ground_truth:
            faces_scored += len(keep[setting])
            found = rows[event, name]
            if not len(faces):
                continue
            ignore = np.zeros(len(faces))
            if len(keep[setting]):
                ignore[keep[setting] - 1] = 1
            recalled_so_far, proposals = _match(found, faces, ignore)
            curve += _image_curve(found, proposals, recalled_so_far)
        aps[setting] = _voc_ap(curve[:, 1] / faces_scored, curve[:, 1] / curve[:, 0])
    return aps


def parse_mpks(values: list[str]) -> Path:
    paths = dict(value.split("=", 1) for value in values)
    if set(paths) != {"model"}:
        raise SystemExit("pass the model component's MPK as --mpk model=<path>")
    return Path(paths["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def model_detections(mpk: Path, root: Path, items: Sequence[tuple[str, str]]) -> Iterator[np.ndarray]:
    """Run each letterboxed image through the MPK and yield its detections."""
    import pyneat

    options = pyneat.ModelOptions()
    # The host letterboxes and keeps BGR pixel values; Neat only quantizes and tessellates.
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(str(mpk), options)
    runner = None
    try:
        for image, width, height in prefetched(lambda item: load_image(root, item), items):
            tensor = pyneat.Tensor.from_numpy(image, copy=False)
            if runner is None:
                runner = model.build([tensor])
            outputs = [output.to_numpy(copy=True) for output in runner.run([tensor], timeout_ms=120000)]
            yield detect(head_maps(outputs), width, height)
    finally:
        if runner is not None:
            runner.close()


def score(
    root: Path, items: Sequence[tuple[str, str]], detections: Iterator[np.ndarray], ground_truth: list
) -> tuple[dict[str, float], int, float]:
    """WIDER FACE AP per subset over the given images, the detection count and the inference time."""
    found, started = {}, time.monotonic()
    for count, (item, rows) in enumerate(zip(items, detections, strict=True), start=1):
        found[item] = rows
        if count == 1 or count % 500 == 0 or count == len(items):
            log(f"{count}/{len(items)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    seconds = time.monotonic() - started
    selected = set(items)
    truth = [entry for entry in ground_truth if (entry[0], entry[1]) in selected]
    aps = evaluate(found, truth)
    metrics = {f"widerface_ap_{setting}": value for setting, value in aps.items()}
    return metrics, sum(len(rows) for rows in found.values()), seconds


def main(argv: list[str] | None = None) -> int:
    import model_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="package resource directory (license only)")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    root = parse_dataset(parser, args.dataset)
    ground_truth = load_ground_truth(root)
    available = [(event, name) for event, name, _, _ in ground_truth]
    items = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    metrics, detections, seconds = score(root, items, model_detections(parse_mpks(args.mpk), root, items), ground_truth)
    report = model_report.from_accuracy(
        {
            DATASET: {
                "metrics": metrics,
                "split": "validation",
                "evaluated_samples": len(items),
                "available_samples": len(available),
                "details": {
                    "detections": detections,
                    "protocol": "640x640 centered zero-padded letterbox, BGR pixel values; FaceDetectorYN decode; "
                    f"score threshold {SCORE_THRESHOLD}, NMS IoU {NMS_THRESHOLD}, top {TOP_K}",
                    "inference_seconds": round(seconds, 1),
                },
            },
        }
    )
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
