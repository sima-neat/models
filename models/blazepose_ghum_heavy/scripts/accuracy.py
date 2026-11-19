"""Measure BlazePose GHUM Heavy 3D's COCO keypoint AP and AR with its MPK on a Modalix board.

The model is MediaPipe's landmark stage: it finds one person's landmarks in a crop around that person. Each
non-crowd val2017 person with labelled keypoints is cropped from its ground-truth box. MediaPipe builds its
square from two keypoints of its person detector and rotates it upright
(mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt and pose_landmark_by_roi_cpu.pbtxt, tag v0.10.26);
this crop keeps its 1.25 scale and border but is not rotated: a square centred on the box, its side 1.25 times
the box's longer side, pixels outside the image
repeating the image's edge (ImageToTensorCalculator's default BORDER_REPLICATE), resized bilinearly to 256x256
and scaled to [0, 1]. The 17 COCO keypoints are BlazePose landmarks COCO_LANDMARKS (PoseLandmark in
mediapipe/python/solutions/pose.py), mapped from the crop back to the image. Each keypoint carries
sigmoid(visibility logit), MediaPipe's landmark visibility; the person scores Identity_1, the pose-presence
probability MediaPipe thresholds at 0.5. COCOeval_faster scores the keypoints with yolo26n_pose's settings.

Unlike MediaPipe, the crop is not rotated to upright the person, since the box gives no rotation, and the
landmarks are not refined with the heatmap, an output the package does not keep. MediaPipe's published scores
are on its own single-person Yoga, Dance and HIIT sets; the ground-truth boxes here leave out person detection.
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
from calibration import INPUT_SIZE, crop, crop_square, decode_image
from faster_coco_eval import COCO, COCOeval_faster

DATASET = "coco-2017-val"
ANNOTATIONS = "annotations/person_keypoints_val2017.json"
IMAGES = "val2017"
# Ultralytics' COCO-pose validation set: the val2017 images with at least one person whose keypoints are labelled.
SCORED_IMAGES = 2346
# Identity holds 39 records of x, y, z, visibility logit and presence logit; Identity_1 one probability;
# Identity_4 39 world x, y, z records.
RECORDS = 39
OUTPUT_SIZES = [RECORDS * 5, 1, RECORDS * 3]
# BlazePose's nose, left and right eye, ear, shoulder, elbow, wrist, hip, knee and ankle, in COCO's keypoint order.
COCO_LANDMARKS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# Metric name in model.yaml -> COCOeval_faster stats_as_dict label. Keypoint scoring keeps the 20 highest-scoring
# people per image and has no small-object range.
METRICS = {
    "coco_keypoint_map": "AP_all",
    "coco_keypoint_ap50": "AP_50",
    "coco_keypoint_ap75": "AP_75",
    "coco_keypoint_ap_medium": "AP_medium",
    "coco_keypoint_ap_large": "AP_large",
    "coco_keypoint_ar": "AR_all",
    "coco_keypoint_ar50": "AR_50",
    "coco_keypoint_ar75": "AR_75",
    "coco_keypoint_ar_medium": "AR_medium",
    "coco_keypoint_ar_large": "AR_large",
}
MAX_DETECTIONS = [20]
# Images decode and crop on worker threads while the model runs; OpenCV releases the GIL while it decodes and warps.
WORKERS = 4
PREFETCH = 8

Infer = Callable[[np.ndarray], Sequence[np.ndarray]]


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def load_ground_truth(dataset: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(dataset / ANNOTATIONS))


def scored_image_ids(ground_truth: COCO) -> list[int]:
    """Return the images with at least one person whose keypoints are labelled."""
    annotations = ground_truth.dataset["annotations"]
    labelled = {annotation["image_id"] for annotation in annotations if annotation["num_keypoints"]}
    if len(labelled) != SCORED_IMAGES:
        raise SystemExit(f"expected {SCORED_IMAGES} images with labelled keypoints, found {len(labelled)}")
    return sorted(labelled)


def sampled(image_ids: list[int], limit: int | None) -> list[int]:
    """Every image, or a seeded sample of limit images for a quick local run."""
    return sorted(random.Random(0).sample(image_ids, limit)) if limit else image_ids


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))


def keypoints(outputs: Sequence[np.ndarray], left: float, top: float, side: float) -> tuple[np.ndarray, float]:
    """The 17 COCO keypoints as image x, y and visibility, and the presence score, from one crop's outputs."""
    if [output.size for output in outputs] != OUTPUT_SIZES:
        raise ValueError(f"unexpected model outputs: {[output.shape for output in outputs]}")
    if not all(np.isfinite(output).all() for output in outputs):
        raise ValueError("model output contains non-finite values")
    records = outputs[0].reshape(RECORDS, 5)[COCO_LANDMARKS].astype(np.float64)
    points = np.array([left, top]) + records[:, :2] * (side / INPUT_SIZE)
    return np.column_stack([points, sigmoid(records[:, 3])]), float(outputs[1].reshape(()))


def people(annotations: list[dict], image_ids: list[int]) -> dict[int, list[dict]]:
    """The scored people of each image: non-crowd person annotations with labelled keypoints."""
    scored: dict[int, list[dict]] = {image_id: [] for image_id in image_ids}
    for annotation in annotations:
        if annotation["image_id"] in scored and annotation["num_keypoints"] and not annotation["iscrowd"]:
            scored[annotation["image_id"]].append(annotation)
    return scored


def read_crops(path: Path, persons: list[dict]) -> list[tuple[float, float, float, np.ndarray]]:
    """Each person's square and its model input, from one image read as RGB without EXIF rotation, as COCO is."""
    image = decode_image(path.read_bytes())
    return [(*square, crop(image, *square)) for square in (crop_square(person["bbox"]) for person in persons)]


def score(ground_truth: COCO, detections: list[dict], image_ids: list[int]) -> dict[str, float]:
    """Score the keypoints and return the declared metrics; COCOeval reports -1 for an empty object-size range."""
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
    scorer = COCOeval_faster(ground_truth, results, "keypoints", print_function=log)
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected keypoints maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    stats = {name: float(scorer.stats_as_dict[label]) for name, label in METRICS.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth people to score for: {', '.join(empty)}")
    return stats


def evaluate(dataset: Path, ground_truth: COCO, image_ids: list[int], infer: Infer) -> tuple[dict[str, float], list]:
    """Score infer, which maps one 256x256 crop to the model's three outputs, on every scored person of the images;
    return the metrics and the COCO results."""
    coco = ground_truth.dataset
    persons = people(coco["annotations"], image_ids)
    files = {record["id"]: record["file_name"] for record in coco["images"]}
    category = [record["id"] for record in coco["categories"]]
    if len(category) != 1:
        raise SystemExit(f"expected the single COCO person category, found {category}")
    detections: list[dict] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    crops = prefetched(lambda image_id: read_crops(image_dir / files[image_id], persons[image_id]), image_ids)
    for count, (image_id, squares) in enumerate(zip(image_ids, crops), start=1):
        for left, top, side, pixels in squares:
            points, presence = keypoints(infer(pixels), left, top, side)
            detections.append({
                "image_id": image_id, "category_id": category[0],
                "keypoints": points.reshape(-1).tolist(), "score": presence,
            })
        if count == 1 or count % 250 == 0 or count == len(image_ids):
            log(f"{count}/{len(image_ids)} images, {len(detections)} people, "
                f"{(time.monotonic() - started) / len(detections) * 1000:.0f} ms/person")
    return score(ground_truth, detections, image_ids), detections


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    dataset = Path(directory)
    ground_truth = load_ground_truth(dataset)
    available = scored_image_ids(ground_truth)
    image_ids = sampled(available, args.limit)

    import pyneat

    # The crops arrive as model input, float32 in [0, 1]; with default options Neat only quantizes and tessellates.
    model = pyneat.Model(str(parse_mpks(args.mpk)))
    runner = None

    def infer(pixels: np.ndarray) -> list[np.ndarray]:
        nonlocal runner
        tensor = pyneat.Tensor.from_numpy(pixels, copy=False)
        if runner is None:
            runner = model.build([tensor])
        return [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]

    started = time.monotonic()
    metrics, detections = evaluate(dataset, ground_truth, image_ids, infer)
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": IMAGES, "evaluated_samples": len(image_ids), "available_samples": len(available),
            "details": {"people": len(detections), "inference_seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
