"""Measure LightGlue's relative pose AUC on yfcc100m-keypoints val with its MPK on a Modalix board.

Each val pair's SuperPoint keypoints run through the matcher, and the CPU extracts LightGlue's matches. The relative
pose comes from the matched keypoints, in each image's original pixels: RANSAC on the essential matrix with a 1-pixel
threshold, then the decomposition that keeps the most matches in front of both cameras. A pair's pose error is the
larger of the rotation and translation angle errors, infinite without a pose, and pose_auc_N is the area under the
recall of the pose error up to N degrees, as SuperGlue's paper evaluates YFCC100M.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from calibration import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_SHAPES,
    Image,
    Pair,
    dataset_root,
    features,
    matcher_inputs,
    matches,
    read_images,
    read_pairs,
    split_of,
)

THRESHOLDS = (5, 10, 20)
RANSAC_PIXELS = 1.0
RANSAC_CONFIDENCE = 0.99999
# The fewest matches an essential matrix is estimated from.
MIN_MATCHES = 5


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script normalizes the keypoints; Neat only quantizes and tessellates the inputs.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def input_tensors(inputs: dict[str, np.ndarray]) -> list[pyneat.Tensor]:
    """The matcher's inputs as Neat tensors in EV74 memory. Neat reads a [1, 512, C] input as height 1, width 512 and
    C channels, so each array goes in as [1, 1, 512, C]."""
    return [
        pyneat.Tensor.from_numpy(inputs[name].reshape(1, 1, *shape[1:]), copy=True, memory=pyneat.TensorMemory.EV74)
        for name, shape in INPUT_SHAPES.items()
    ]


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


def ransac_threshold(pair: Pair, image0: Image, image1: Image) -> float:
    """RANSAC_PIXELS pixels of SuperPoint's input, in normalized image coordinates: the pixel over the focal lengths
    of both images' intrinsics, scaled as the images were stretched to the input."""
    focals = []
    for intrinsics, image in ((pair.intrinsics0, image0), (pair.intrinsics1, image1)):
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


def pair_error(root: Path, images: dict[str, Image], pair: Pair, runner: pyneat.ModelRunner | None,
               model: pyneat.Model) -> tuple[float, int, pyneat.ModelRunner]:
    """One pair's pose error and match count, building the runner on the first pair."""
    image0, image1 = features(root, pair.name0), features(root, pair.name1)
    tensors = input_tensors(matcher_inputs(image0, image1))
    if runner is None:
        runner = model.build(tensors)
    (assignment,) = runner.run(tensors, timeout_ms=120000)
    matched = matches(assignment.to_numpy(copy=False), (len(image0[0]), len(image1[0])))
    first, second = images[pair.name0], images[pair.name1]
    points0, points1 = original_pixels(image0[0][matched[:, 0]], first), original_pixels(image1[0][matched[:, 1]], second)
    pose = estimate_pose(points0, points1, pair.intrinsics0, pair.intrinsics1, ransac_threshold(pair, first, second))
    return (np.inf if pose is None else pose_error(pair.pose, *pose)), len(matched), runner


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
    parser.add_argument("--limit", type=int, help="measure the first pairs only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    root = dataset_root(Path(dataset_dir))
    images, available = read_images(root), read_pairs(root)
    pairs = available[: args.limit] if args.limit else available

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    errors, match_counts = [], []
    started = time.monotonic()
    for count, pair in enumerate(pairs, start=1):
        error, matched, runner = pair_error(root, images, pair, runner, model)
        errors.append(error)
        match_counts.append(matched)
        if count == 1 or count % 500 == 0 or count == len(pairs):
            log(f"{count}/{len(pairs)} pairs, {(time.monotonic() - started) / count * 1000:.0f} ms/pair")

    aucs = pose_auc(errors)
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {f"pose_auc_{threshold}": auc for threshold, auc in zip(THRESHOLDS, aucs)},
            "split": split_of(root), "evaluated_samples": len(pairs), "available_samples": len(available),
            "details": {"inference_seconds": round(time.monotonic() - started, 1),
                        "mean_matches": round(float(np.mean(match_counts)), 1)},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
