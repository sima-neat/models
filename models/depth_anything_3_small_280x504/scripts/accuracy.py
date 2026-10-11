"""Measure Depth Anything 3 Small AbsRel and delta1 on held-out KITTI drives with its MPK."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyneat
from calibration import CALIBRATION_DRIVES, HEIGHT, WIDTH, cover_geometry, frame_key, preprocess_image

VIEWS = 2
# kitti/default holds 652 test frames from 28 drives; calibration draws from 7 of them, the other 21 are scored.
FRAMES = 506
# The depth PNGs hold meters x 256; KITTI depth evaluation keeps depths in (1e-3, 80) m.
DEPTH_SCALE = 256.0
MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0
# The aligned disparity is kept at or above 1 / MAX_ALIGNED_DEPTH so that no prediction becomes infinite.
MAX_ALIGNED_DEPTH = 1000.0
DELTA = 1.25
# Published metrics and whether a lower or higher value is better.
DIRECTIONS = {"abs_rel": "lower", "delta1": "higher"}
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized, cropped and normalized the images; Neat still converts and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def view_pairs(dataset: Path) -> list[tuple[Path, Path]]:
    """Return each held-out frame as view 0 with the nearest other frame of its drive as view 1.

    Of two equally near frames, the later one is view 1. Pairs are in drive and frame order.
    """
    frames = sorted((dataset / "test" / "images").glob("*.png"), key=lambda path: frame_key(path.name))
    drives: dict[str, list[tuple[int, Path]]] = {}
    for path in frames:
        drive, index = frame_key(path.name)
        drives.setdefault(drive, []).append((index, path))
    pairs = []
    for drive, items in drives.items():
        if drive in CALIBRATION_DRIVES:
            continue
        for index, path in items:
            _, _, neighbor = min((abs(frame - index), -frame, other) for frame, other in items if frame != index)
            pairs.append((path, neighbor))
    return pairs


def read_ground_truth(path: Path) -> np.ndarray:
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(f"expected a 16-bit depth PNG: {path}")
    return raw.astype(np.float64) / DEPTH_SCALE


def load(pair: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference, other = pair
    truth = read_ground_truth(reference.parents[1] / "depth" / reference.name)
    return preprocess_image(reference.read_bytes()), preprocess_image(other.read_bytes()), truth


def paired(depth: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair each valid ground-truth pixel with the network pixel that covers it.

    Returns the network depths and the ground-truth depths in meters. Ground-truth pixels that the center
    crop removed are not scored.
    """
    rows, columns = truth.shape
    _, (resized_width, resized_height), (left, top) = cover_geometry(columns, rows)
    ys, xs = np.nonzero((truth > MIN_DEPTH) & (truth < MAX_DEPTH))
    u = np.floor((xs + 0.5) * (resized_width / columns) - left).astype(int)
    v = np.floor((ys + 0.5) * (resized_height / rows) - top).astype(int)
    keep = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    return depth[v[keep], u[keep]].astype(np.float64), truth[ys[keep], xs[keep]]


def score(depth: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return one image's AbsRel and delta1 after aligning the predicted disparity to the ground truth.

    The network's depth has no metric scale, so a least-squares scale and shift maps its inverse onto the
    inverse ground truth (the Depth Anything V1/V2 protocol) before scoring in depth.
    """
    pixels, meters = paired(depth, truth)
    if not len(meters):
        raise ValueError("no valid ground-truth pixel inside the network view")
    # A zero network depth would give an infinite disparity.
    disparity = 1.0 / np.maximum(pixels, 1e-6)
    (scale, shift), *_ = np.linalg.lstsq(np.stack([disparity, np.ones_like(disparity)], 1), 1.0 / meters, rcond=None)
    prediction = 1.0 / np.maximum(scale * disparity + shift, 1.0 / MAX_ALIGNED_DEPTH)
    ratio = np.maximum(prediction / meters, meters / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(prediction - meters) / meters)),
        "delta1": float(np.mean(ratio < DELTA)),
        "pixels": len(meters),
    }


def reference_depth(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Return view 0's depth map from the MPK's depth, depth_conf and pose_enc outputs.

    The MPK returns depth with the views as the last axis.
    """
    sizes = [output.size for output in outputs]
    if sizes != [HEIGHT * WIDTH * VIEWS, HEIGHT * WIDTH * VIEWS, VIEWS * 9]:
        raise ValueError(f"unexpected MPK outputs: {[output.shape for output in outputs]}")
    depth = outputs[0].reshape(HEIGHT, WIDTH, VIEWS)[..., 0]
    if not np.isfinite(depth).all():
        raise ValueError("MPK depth contains non-finite values")
    return depth


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
    parser.add_argument("--dataset", type=Path, required=True, help="extracted kitti/default directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="measure a seeded sample of frames (local runs only)")
    args = parser.parse_args(argv)

    available = view_pairs(args.dataset)
    if len(available) != FRAMES:
        raise SystemExit(f"expected {FRAMES} held-out KITTI test frames, found {len(available)}")
    selected = sorted(random.Random(0).sample(range(FRAMES), args.limit)) if args.limit else range(FRAMES)
    pairs = [available[index] for index in selected]

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    rows: list[dict[str, float]] = []
    started = time.monotonic()
    for count, (*views, truth) in enumerate(prefetched(load, pairs), start=1):
        # An HxWx3 array has no unambiguous layout of its own.
        tensors = [pyneat.Tensor.from_numpy(view, copy=False, layout=pyneat.TensorLayout.HWC) for view in views]
        if runner is None:
            runner = model.build(tensors)
        outputs = [output.to_numpy(copy=False) for output in runner.run(tensors, timeout_ms=120000)]
        rows.append(score(reference_depth(outputs), truth))
        if count == 1 or count % 100 == 0 or count == len(pairs):
            log(f"{count}/{len(pairs)} frames, {(time.monotonic() - started) / count * 1000:.0f} ms/frame")
    inference_seconds = time.monotonic() - started

    # Each metric is the mean of per-image values.
    stats = {name: float(np.mean([row[name] for row in rows])) for name in DIRECTIONS}
    result = {
        "core_version": pyneat.__version__,
        "dataset": {"split": "test", "evaluated_samples": len(pairs), "available_samples": FRAMES},
        "metrics": {
            name: {"value": value, "unit": "ratio", "direction": DIRECTIONS[name]} for name, value in stats.items()
        },
        "details": {
            "scored_pixels": int(sum(row["pixels"] for row in rows)),
            "inference_seconds": round(inference_seconds, 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: round(item["value"], 4) for name, item in result["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
