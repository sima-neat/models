"""Measure LightGlue's relative pose AUC on yfcc100m-keypoints val with its MPK on a Modalix board.

Each val pair's SuperPoint keypoints run through the matcher, and the CPU extracts LightGlue's matches. The relative
pose comes from the matched keypoints, in each image's original pixels: RANSAC on the essential matrix with a 1-pixel
threshold, then the decomposition that keeps the most matches in front of both cameras. A pair's pose error is the
larger of the rotation and translation angle errors, infinite without a pose, and pose_auc_N is the area under the
recall of the pose error up to N degrees, as SuperGlue's paper evaluates YFCC100M. The matcher takes 512 real keypoints
per image, so pairs with an image of fewer are left out: one val image has 511, in 22 of the 4,000 pairs.
fp32_baseline_accuracy.py scores the FP32 ONNX with this script's evaluate().
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
from calibration import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_SHAPES,
    Pair,
    dataset_root,
    features,
    matcher_inputs,
    matches,
    pairs_file,
    read_pairs,
    split_of,
    usable_pairs,
)

THRESHOLDS = (5, 10, 20)
RANSAC_PIXELS = 1.0
RANSAC_CONFIDENCE = 0.99999
# The fewest matches an essential matrix is estimated from.
MIN_MATCHES = 5

# A pairs line: both image names, both rotations, both 3x3 intrinsics and the 4x4 pose from image 0 to image 1.
PAIR_FIELDS = 38

# The matcher's assignment [i, j] for a pair's channels-last inputs, from the MPK or the FP32 ONNX.
AssignmentOf = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class Image:
    """An image's original size and the quarter turns its SuperPoint input was rotated by."""

    width: int
    height: int
    rotation: int


@dataclass(frozen=True)
class GroundTruth:
    """A pair's ground truth: each image's intrinsics, at its original size, and the pose from image 0 to image 1."""

    intrinsics0: np.ndarray
    intrinsics1: np.ndarray
    pose: np.ndarray


def read_images(root: Path) -> dict[str, Image]:
    images = {}
    for line in (root / "annotations" / "images.txt").read_text(encoding="utf-8").splitlines():
        name, width, height, rotation, _ = line.split()
        images[name] = Image(int(width), int(height), int(rotation) % 4)
    return images


def read_ground_truth(root: Path) -> dict[Pair, GroundTruth]:
    truths = {}
    for line in pairs_file(root).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != PAIR_FIELDS:
            raise ValueError(f"a pairs line has {len(fields)} fields, expected {PAIR_FIELDS}: {line[:80]}")
        # images.txt holds each image's rotation; the ground truth is the unrotated images'.
        values = np.asarray(fields[4:], np.float64)
        truths[Pair(fields[0], fields[1])] = GroundTruth(
            values[:9].reshape(3, 3), values[9:18].reshape(3, 3), values[18:].reshape(4, 4))
    return truths


def original_pixels(keypoints: np.ndarray, image: Image) -> np.ndarray:
    """Keypoints in pixels of SuperPoint's 640x480 input, in pixels of the original image. The input is the image
    stretched to 640x480, or for an odd number of quarter turns to 480x640, then turned as numpy's rot90 turns it; the
    turns are undone before the stretch."""
    x, y = keypoints[:, 0], keypoints[:, 1]
    width, height = (IMAGE_HEIGHT, IMAGE_WIDTH) if image.rotation % 2 else (IMAGE_WIDTH, IMAGE_HEIGHT)
    if image.rotation == 1:
        x, y = width - 1 - y, x
    elif image.rotation == 2:
        x, y = width - 1 - x, height - 1 - y
    elif image.rotation == 3:
        x, y = y, height - 1 - x
    return np.stack([x * image.width / width, y * image.height / height], 1)


def ransac_threshold(truth: GroundTruth, image0: Image, image1: Image) -> float:
    """RANSAC_PIXELS pixels of SuperPoint's input, in normalized image coordinates: the pixel over the focal lengths
    of both images' intrinsics, scaled as the images were stretched to the input."""
    focals = []
    for intrinsics, image in ((truth.intrinsics0, image0), (truth.intrinsics1, image1)):
        width, height = (IMAGE_HEIGHT, IMAGE_WIDTH) if image.rotation % 2 else (IMAGE_WIDTH, IMAGE_HEIGHT)
        focals += [intrinsics[0, 0] * width / image.width, intrinsics[1, 1] * height / image.height]
    return RANSAC_PIXELS / float(np.mean(focals))


def estimate_pose(points0: np.ndarray, points1: np.ndarray, intrinsics0: np.ndarray, intrinsics1: np.ndarray,
                  threshold: float) -> tuple[np.ndarray, np.ndarray] | None:
    """The rotation and unit translation from image 0 to image 1, from matched points in the original images' pixels,
    or None when the matches give no pose. The RANSAC threshold is in normalized image coordinates."""
    if len(points0) < MIN_MATCHES:
        return None
    normalized0 = (points0 - intrinsics0[[0, 1], [2, 2]]) / intrinsics0[[0, 1], [0, 1]]
    normalized1 = (points1 - intrinsics1[[0, 1], [2, 2]]) / intrinsics1[[0, 1], [0, 1]]
    essential, inliers = cv2.findEssentialMat(normalized0, normalized1, np.eye(3), method=cv2.RANSAC,
                                              prob=RANSAC_CONFIDENCE, threshold=threshold)
    if essential is None:
        return None
    best, pose = 0, None
    # findEssentialMat may return several solutions stacked as 3x3 blocks.
    for candidate in np.split(essential, len(essential) // 3):
        count, rotation, translation, _ = cv2.recoverPose(candidate, normalized0, normalized1, np.eye(3), 1e9,
                                                         mask=inliers.copy())[:4]
        if count > best:
            best, pose = count, (rotation, translation[:, 0])
    return pose


def angle_degrees(cosine: float) -> float:
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pose_error(truth: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> float:
    """The larger of the rotation error and the translation direction error, in degrees. An essential matrix fixes
    the translation only up to sign, so a translation error above 90 degrees counts from the opposite direction."""
    true_rotation, true_translation = truth[:3, :3], truth[:3, 3]
    rotation_error = angle_degrees((np.trace(true_rotation.T @ rotation) - 1) / 2)
    norm = np.linalg.norm(translation) * np.linalg.norm(true_translation)
    translation_error = angle_degrees(np.dot(translation, true_translation) / norm)
    return max(rotation_error, min(translation_error, 180 - translation_error))


def pose_auc(errors: list[float], thresholds: tuple[int, ...] = THRESHOLDS) -> list[float]:
    """The area under the recall of the pose errors up to each threshold, normalized to [0, 1]."""
    errors = np.sort(np.asarray(errors, np.float64))
    recall = np.arange(1, len(errors) + 1) / len(errors)
    errors, recall = np.concatenate([[0.0], errors]), np.concatenate([[0.0], recall])
    areas = []
    for threshold in thresholds:
        last = int(np.searchsorted(errors, threshold))
        curve = np.concatenate([recall[:last], [recall[last - 1]]])
        steps = np.concatenate([errors[:last], [threshold]])
        areas.append(float(np.sum((steps[1:] - steps[:-1]) * (curve[1:] + curve[:-1]) / 2) / threshold))
    return areas


def pair_error(root: Path, images: dict[str, Image], pair: Pair, truth: GroundTruth,
               assignment_of: AssignmentOf) -> tuple[float, int]:
    """One pair's pose error and match count."""
    image0, image1 = features(root, pair.name0), features(root, pair.name1)
    matched = matches(assignment_of(matcher_inputs(image0, image1)))
    first, second = images[pair.name0], images[pair.name1]
    points0 = original_pixels(image0[0][matched[:, 0]], first)
    points1 = original_pixels(image1[0][matched[:, 1]], second)
    pose = estimate_pose(points0, points1, truth.intrinsics0, truth.intrinsics1, ransac_threshold(truth, first, second))
    return (np.inf if pose is None else pose_error(truth.pose, *pose)), len(matched)


def evaluate(root: Path, pairs: Sequence[Pair], assignment_of: AssignmentOf) -> tuple[dict[str, float], float]:
    """The pairs' pose AUC metrics and mean match count, with the assignments assignment_of returns."""
    images, truths = read_images(root), read_ground_truth(root)
    errors, match_counts = [], []
    started = time.monotonic()
    for count, pair in enumerate(pairs, start=1):
        error, matched = pair_error(root, images, pair, truths[pair], assignment_of)
        errors.append(error)
        match_counts.append(matched)
        if count == 1 or count % 500 == 0 or count == len(pairs):
            log(f"{count}/{len(pairs)} pairs, {(time.monotonic() - started) / count * 1000:.0f} ms/pair")
    metrics = {f"pose_auc_{threshold}": auc for threshold, auc in zip(THRESHOLDS, pose_auc(errors))}
    return metrics, float(np.mean(match_counts))


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
    parser.add_argument("--dataset", required=True, help="yfcc100m-keypoints-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first pairs only")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    root = dataset_root(Path(dataset_dir))
    available, usable = read_pairs(root), usable_pairs(root)
    pairs = usable[: args.limit] if args.limit else usable

    # PyNeat runs on the board only; fp32_baseline_accuracy.py imports this script on a host.
    import pyneat

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script encodes the keypoints; Neat only quantizes and tessellates the inputs.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(str(parse_mpks(args.mpk)), options)
    runner = None

    def assignment_of(inputs: dict[str, np.ndarray]) -> np.ndarray:
        """The MPK's assignment, building the runner on the first pair, with the inputs in EV74 memory."""
        nonlocal runner
        tensors = [
            pyneat.Tensor.from_numpy(inputs[name], copy=True, memory=pyneat.TensorMemory.EV74) for name in INPUT_SHAPES
        ]
        if runner is None:
            runner = model.build(tensors)
        (assignment,) = runner.run(tensors, timeout_ms=120000)
        return assignment.to_numpy(copy=False)

    started = time.monotonic()
    metrics, mean_matches = evaluate(root, pairs, assignment_of)
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": split_of(root), "evaluated_samples": len(pairs), "available_samples": len(available),
            "details": {"inference_seconds": round(time.monotonic() - started, 1),
                        "mean_matches": round(mean_matches, 1)},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
