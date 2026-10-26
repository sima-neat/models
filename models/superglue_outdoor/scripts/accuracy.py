"""Measure SuperGlue Outdoor's relative pose accuracy on the YFCC100M test pairs with its MPK on a Modalix board.

This is the outdoor evaluation of SuperGlue's README (match_pairs.py --eval on assets/yfcc_test_pairs_with_gt.txt):
4000 image pairs of four YFCC100M scenes with ground-truth intrinsics and relative poses. The dataset holds each
image's SuperPoint FP32 features at the README's settings, but KEYPOINTS of them instead of up to 2048, the fixed
input of the compiled matcher. For each pair the script runs the MPK, extracts mutual matches after the Sinkhorn
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

DATASET_ID = "yfcc100m"
SPLIT = "test"
KEYPOINTS = 512
DESCRIPTOR_SIZE = 256
PAIRS = 4000
ARRAYS = ("image_sizes", "keypoints", "scores", "descriptors", "pair_images", "intrinsics", "poses")
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


def load_dataset(directory: Path) -> dict[str, np.ndarray]:
    """Load the dataset's arrays after checking them against its manifest."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("id") != DATASET_ID or manifest.get("keypoints") != KEYPOINTS or manifest.get("pairs") != PAIRS:
        raise SystemExit(f"{directory} is not a {DATASET_ID} dataset of {PAIRS} pairs with {KEYPOINTS} keypoints")
    data = {}
    for name in ARRAYS:
        path = directory / f"{name}.npy"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
            raise SystemExit(f"{path.name} does not match the manifest")
        data[name] = np.load(path, allow_pickle=False)
    images = len(data["image_sizes"])
    expected = {
        "image_sizes": (images, 2), "keypoints": (images, KEYPOINTS, 2), "scores": (images, KEYPOINTS),
        "descriptors": (images, KEYPOINTS, DESCRIPTOR_SIZE), "pair_images": (PAIRS, 2),
        "intrinsics": (PAIRS, 2, 3, 3), "poses": (PAIRS, 4, 4),
    }
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise SystemExit(f"{name} has shape {data[name].shape}, expected {shape}")
    if data["pair_images"].min() < 0 or data["pair_images"].max() >= images:
        raise SystemExit("pair_images refers to images the dataset does not hold")
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
    parser.add_argument("--dataset", required=True, help="yfcc100m-superpoint-test-512=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first pairs only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    data = load_dataset(Path(dataset_dir))
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
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": SPLIT, "evaluated_samples": len(pairs), "available_samples": PAIRS,
            "details": {"matches_per_pair": totals["matches"], "seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
