"""Measure SuperGlue Outdoor's relative pose accuracy on the YFCC100M test pairs with its MPK on a Modalix board.

This is the outdoor evaluation of SuperGlue's README (match_pairs.py --eval on assets/yfcc_test_pairs_with_gt.txt):
4000 image pairs of four YFCC100M scenes. yfcc100m-keypoints val holds each image's SuperPoint FP32 keypoints at the
README's settings, but KEYPOINTS of them instead of up to 2048, the fixed input of the compiled matcher; yfcc100m val
holds the images and each pair's ground-truth intrinsics and relative pose, which the script maps onto the resized,
rotated images as match_pairs.py does. For each pair the script runs the MPK, extracts mutual matches after the Sinkhorn
iterations, estimates the relative pose from them and scores it as match_pairs.py does: the area under the
pose-error curve at 5, 10 and 20 degrees, match precision and matching score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import model_report
import numpy as np
import pyneat
from matching import MATCH_THRESHOLD, extract_matches, log_sinkhorn, normalize_keypoints

# Inputs: SuperPoint FP32 keypoints of every val image. Ground truth: the val pairs file's intrinsics and poses.
KEYPOINTS_DATASET = "yfcc100m-keypoints-val"
PAIRS_DATASET = "yfcc100m-val"
PAIRS_FILE = "yfcc_test_pairs_with_gt.txt"
SPLIT = "test"
KEYPOINTS = 512
DESCRIPTOR_SIZE = 256
PAIRS = 4000
KEYPOINT_ARRAYS = ("original_sizes", "rotations", "image_sizes", "keypoints", "scores", "descriptors")
# match_pairs.py's evaluation settings: 20 Sinkhorn iterations, a 1-pixel RANSAC threshold on the resized images,
# and a match counts as correct when its epipolar error is below 5e-4.
SINKHORN_ITERATIONS = 20
POSE_PIXELS = 1.0
RANSAC_CONFIDENCE = 0.99999
EPIPOLAR_THRESHOLD = 5e-4
AUC_DEGREES = (5, 10, 20)
METRICS = ("pose_auc_5", "pose_auc_10", "pose_auc_20", "match_precision", "matching_score")
# The MPK takes channels-last tensors in this order and returns the couplings channels-last, [1, 1, i, j]; Neat
# Core 0.5.0 drops the unit axis, [1, i, j].
INPUTS = {
    "keypoints0": (1, 1, KEYPOINTS, 2),
    "scores0": (1, 1, KEYPOINTS, 1),
    "descriptors0": (1, 1, KEYPOINTS, DESCRIPTOR_SIZE),
    "keypoints1": (1, 1, KEYPOINTS, 2),
    "scores1": (1, 1, KEYPOINTS, 1),
    "descriptors1": (1, 1, KEYPOINTS, DESCRIPTOR_SIZE),
}
OUTPUT_SHAPES = ((1, 1, KEYPOINTS + 1, KEYPOINTS + 1), (1, KEYPOINTS + 1, KEYPOINTS + 1))


def load_keypoints(directory: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """The keypoint dataset's image names and arrays, checked against its manifest."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = (manifest.get("id"), manifest.get("variant"), manifest.get("keypoints"))
    if identity != ("yfcc100m-keypoints", "val", KEYPOINTS):
        raise SystemExit(f"{directory} is not the {KEYPOINTS_DATASET} dataset with {KEYPOINTS} keypoints per image")
    names = (directory / "images.txt").read_text(encoding="utf-8").split()
    data = {}
    for name in KEYPOINT_ARRAYS:
        path = directory / f"{name}.npy"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
            raise SystemExit(f"{path.name} does not match the manifest")
        data[name] = np.load(path, allow_pickle=False)
    images = len(names)
    expected = {
        "original_sizes": (images, 2), "rotations": (images,), "image_sizes": (images, 2),
        "keypoints": (images, KEYPOINTS, 2), "scores": (images, KEYPOINTS),
        "descriptors": (images, KEYPOINTS, DESCRIPTOR_SIZE),
    }
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise SystemExit(f"{name} has shape {data[name].shape}, expected {shape}")
    return names, data


def rotate_intrinsics(K: np.ndarray, width: int, height: int, rotation: int) -> np.ndarray:
    """SuperGlue's utils.rotate_intrinsics for an image of (width, height) after the rotation."""
    w, h = (height, width) if rotation % 2 else (width, height)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if rotation == 1:
        return np.array([[fy, 0.0, cy], [0.0, fx, w - 1 - cx], [0.0, 0.0, 1.0]])
    if rotation == 2:
        return np.array([[fx, 0.0, w - 1 - cx], [0.0, fy, h - 1 - cy], [0.0, 0.0, 1.0]])
    return np.array([[fy, 0.0, h - 1 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]])


def rotate_pose_inplane(i_T_w: np.ndarray, rotation: int) -> np.ndarray:
    """SuperGlue's utils.rotate_pose_inplane, float32 rotation matrices included."""
    matrices = [
        np.array([[np.cos(r), -np.sin(r), 0.0, 0.0], [np.sin(r), np.cos(r), 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(matrices[rotation], i_T_w)


def image_intrinsics(K: np.ndarray, data: dict[str, np.ndarray], image: int) -> np.ndarray:
    """An original image's intrinsics mapped onto the resized, rotated image its keypoints refer to, as
    match_pairs.py's scale_intrinsics and rotate_intrinsics do."""
    width, height = data["image_sizes"][image]
    rotation = int(data["rotations"][image])
    resized = (height, width) if rotation % 2 else (width, height)
    sx, sy = data["original_sizes"][image] / np.array(resized, dtype=np.float64)
    if rotation % 2:
        sx, sy = sy, sx
    K = np.diag([1 / sx, 1 / sy, 1.0]) @ K
    return rotate_intrinsics(K, width, height, rotation) if rotation else K


def load_dataset(keypoints_dir: Path, pairs_dir: Path) -> dict[str, np.ndarray]:
    """The keypoints of every pair's two images, and each pair's intrinsics and pose on those images."""
    names, data = load_keypoints(keypoints_dir)
    position = {name: index for index, name in enumerate(names)}
    lines = [line.split() for line in (pairs_dir / PAIRS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != PAIRS or any(len(line) != 38 for line in lines):
        raise SystemExit(f"{PAIRS_FILE} must hold {PAIRS} pairs of 38 fields")
    pair_images, intrinsics, poses = [], [], []
    for line in lines:
        if line[0] not in position or line[1] not in position:
            raise SystemExit(f"the keypoint dataset has no keypoints for {line[0]} or {line[1]}")
        images = (position[line[0]], position[line[1]])
        for image, rotation in zip(images, (int(line[2]), int(line[3]))):
            if data["rotations"][image] != rotation:
                detected = data["rotations"][image]
                raise SystemExit(f"{names[image]} was detected with rotation {detected}, not {rotation}")
        pose = np.array(line[22:], dtype=np.float64).reshape(4, 4)
        cameras, camera_T_world = [], []
        for image, values, start in ((images[0], line[4:13], np.eye(4)), (images[1], line[13:22], pose)):
            cameras.append(image_intrinsics(np.array(values, dtype=np.float64).reshape(3, 3), data, image))
            rotation = int(data["rotations"][image])
            camera_T_world.append(rotate_pose_inplane(start, rotation) if rotation else start)
        if data["rotations"][images[0]] or data["rotations"][images[1]]:
            pose = camera_T_world[1] @ np.linalg.inv(camera_T_world[0])
        pair_images.append(images)
        intrinsics.append(cameras)
        poses.append(pose)
    data["pair_images"] = np.array(pair_images, dtype=np.int32)
    data["intrinsics"] = np.array(intrinsics, dtype=np.float64)
    data["poses"] = np.array(poses, dtype=np.float64)
    return data


def matcher_inputs(data: dict[str, np.ndarray], pair: int) -> list[np.ndarray]:
    """One pair's MPK inputs, channels-last in the declared order, each image's keypoints normalized by its size."""
    values = {}
    for side, image in enumerate(data["pair_images"][pair]):
        width, height = data["image_sizes"][image]
        values[f"keypoints{side}"] = normalize_keypoints(data["keypoints"][image], width, height)
        values[f"scores{side}"] = data["scores"][image]
        values[f"descriptors{side}"] = data["descriptors"][image]
    return [np.ascontiguousarray(values[name], dtype=np.float32).reshape(shape) for name, shape in INPUTS.items()]


def couplings_matrix(output: np.ndarray) -> np.ndarray:
    """The MPK's channels-last couplings as the matrix with image-0 keypoints as rows."""
    if output.shape not in OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK output: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("MPK output contains non-finite values")
    return output.reshape(KEYPOINTS + 1, KEYPOINTS + 1)


def _normalize(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    return (points - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]


def estimate_pose(points0: np.ndarray, points1: np.ndarray, K0: np.ndarray, K1: np.ndarray):
    """SuperGlue's utils.estimate_pose: the essential matrix by RANSAC, then the pose with the most inliers."""
    import cv2

    if len(points0) < 5:
        return None
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    points0, points1 = _normalize(points0, K0), _normalize(points1, K1)
    # RANSAC samples at random; a fixed seed makes every run score the same.
    cv2.setRNGSeed(0)
    E, mask = cv2.findEssentialMat(
        points0, points1, np.eye(3), threshold=POSE_PIXELS / f_mean, prob=RANSAC_CONFIDENCE, method=cv2.RANSAC
    )
    if E is None:
        return None
    best, pose = 0, None
    for candidate in np.split(E, len(E) / 3):
        inliers, R, t, _ = cv2.recoverPose(candidate, points0, points1, np.eye(3), 1e9, mask=mask)
        if inliers > best:
            best, pose = inliers, (R, t[:, 0])
    return pose


def epipolar_errors(points0: np.ndarray, points1: np.ndarray, T_0to1: np.ndarray, K0: np.ndarray,
                    K1: np.ndarray) -> np.ndarray:
    """SuperGlue's utils.compute_epipolar_error: the symmetric epipolar distance in normalized coordinates."""
    points0 = np.hstack([_normalize(points0, K0), np.ones((len(points0), 1))])
    points1 = np.hstack([_normalize(points1, K1), np.ones((len(points1), 1))])
    t0, t1, t2 = T_0to1[:3, 3]
    E = np.array([[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]]) @ T_0to1[:3, :3]
    Ep0, Etp1 = points0 @ E.T, points1 @ E
    p1Ep0 = np.sum(points1 * Ep0, -1)
    return p1Ep0**2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))


def pose_error(T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """SuperGlue's compute_pose_error, reduced as match_pairs.py does: the larger of the two angular errors."""
    t_gt = T_0to1[:3, 3]
    error_t = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / (np.linalg.norm(t) * np.linalg.norm(t_gt)), -1.0, 1.0)))
    error_t = min(error_t, 180 - error_t)
    cos = np.clip((np.trace(T_0to1[:3, :3].T @ R) - 1) / 2, -1.0, 1.0)
    return float(max(error_t, np.rad2deg(np.abs(np.arccos(cos)))))


def pose_auc(errors: Sequence[float], thresholds: Sequence[float] = AUC_DEGREES) -> list[float]:
    """SuperGlue's utils.pose_auc: the area under the recall curve of the pose error up to each threshold."""
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    recall = np.r_[0.0, (np.arange(len(errors)) + 1) / len(errors)]
    errors = np.r_[0.0, errors]
    aucs = []
    for threshold in thresholds:
        last = np.searchsorted(errors, threshold)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], threshold]
        # The trapezoid rule, as np.trapz computes it.
        aucs.append(float(np.sum((e[1:] - e[:-1]) * (r[1:] + r[:-1]) / 2) / threshold))
    return aucs


def score_pair(couplings: np.ndarray, data: dict[str, np.ndarray], pair: int) -> dict[str, float]:
    """Match one pair from its [N+1, N+1] couplings and score the matches and the pose as match_pairs.py does."""
    matches, _ = extract_matches(log_sinkhorn(couplings, SINKHORN_ITERATIONS), MATCH_THRESHOLD)
    image0, image1 = data["pair_images"][pair]
    K0, K1 = data["intrinsics"][pair]
    T_0to1 = data["poses"][pair]
    valid = matches >= 0
    points0 = data["keypoints"][image0][valid].astype(np.float64)
    points1 = data["keypoints"][image1][matches[valid]].astype(np.float64)
    correct = epipolar_errors(points0, points1, T_0to1, K0, K1) < EPIPOLAR_THRESHOLD
    pose = estimate_pose(points0, points1, K0, K1)
    return {
        "pose_error": np.inf if pose is None else pose_error(T_0to1, *pose),
        "precision": float(correct.mean()) if len(correct) else 0.0,
        "matching_score": float(correct.sum()) / KEYPOINTS,
        "matches": float(valid.sum()),
    }


def evaluate(data: dict[str, np.ndarray], pairs: Sequence[int],
             couplings_of: Callable[[list[np.ndarray]], np.ndarray]) -> tuple[dict[str, float], dict[str, float]]:
    """Score the pairs with couplings_of(inputs), which returns the [N+1, N+1] couplings; return metrics and totals."""
    rows = []
    started = time.monotonic()
    for count, pair in enumerate(pairs, start=1):
        rows.append(score_pair(couplings_of(matcher_inputs(data, pair)), data, pair))
        if count == 1 or count % 500 == 0 or count == len(pairs):
            log(f"{count}/{len(pairs)} pairs, {(time.monotonic() - started) / count * 1000:.0f} ms/pair")
    aucs = pose_auc([row["pose_error"] for row in rows])
    metrics = {f"pose_auc_{degrees}": auc for degrees, auc in zip(AUC_DEGREES, aucs)}
    metrics["match_precision"] = float(np.mean([row["precision"] for row in rows]))
    metrics["matching_score"] = float(np.mean([row["matching_score"] for row in rows]))
    return metrics, {"matches": round(float(np.mean([row["matches"] for row in rows])), 1)}


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
    parser.add_argument(
        "--dataset", action="append", required=True,
        help=f"{KEYPOINTS_DATASET}=<extracted directory> and {PAIRS_DATASET}=<extracted directory>",
    )
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first pairs only (local runs only)")
    args = parser.parse_args(argv)
    datasets = dict(value.split("=", 1) for value in args.dataset)
    if set(datasets) != {KEYPOINTS_DATASET, PAIRS_DATASET}:
        raise SystemExit(f"--dataset must name {KEYPOINTS_DATASET} and {PAIRS_DATASET}, got {sorted(datasets)}")
    data = load_dataset(Path(datasets[KEYPOINTS_DATASET]), Path(datasets[PAIRS_DATASET]))
    pairs = range(min(args.limit, PAIRS) if args.limit else PAIRS)

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The inputs are keypoint features, not images: Neat only quantizes and tessellates them.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(str(parse_mpks(args.mpk)), options)
    runner = None

    def couplings_of(inputs: list[np.ndarray]) -> np.ndarray:
        nonlocal runner
        tensors = [pyneat.Tensor.from_numpy(value, copy=False) for value in inputs]
        if runner is None:
            runner = model.build(tensors)
        (output,) = runner.run(tensors, timeout_ms=120000)
        return couplings_matrix(output.to_numpy(copy=True))

    started = time.monotonic()
    metrics, totals = evaluate(data, pairs, couplings_of)
    # One evaluation needs both datasets, and the registry expects a result per pinned dataset: both carry it.
    result = {
        "metrics": metrics,
        "split": SPLIT, "evaluated_samples": len(pairs), "available_samples": PAIRS,
        "details": {
            "inputs": KEYPOINTS_DATASET, "ground_truth": PAIRS_DATASET,
            "matches_per_pair": totals["matches"], "seconds": round(time.monotonic() - started, 1),
        },
    }
    report = model_report.from_accuracy({KEYPOINTS_DATASET: result, PAIRS_DATASET: result})
    report.save(args.output)
    measured = report.document["datasets"][PAIRS_DATASET]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
