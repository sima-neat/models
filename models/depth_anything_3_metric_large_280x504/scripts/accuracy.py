"""Measure Depth Anything 3 Metric Large AbsRel and delta1 on the NYU Depth V2 Eigen test set with its MPK."""

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
from calibration import HEIGHT, WIDTH, cover_geometry, preprocess_image

# nyu-depth-v2/default splits the 654-frame Eigen test set into val, cal and test; train is not scored.
SPLITS = ("val", "cal", "test")
FRAMES = 654
# sync_depth PNGs hold millimeters; the Eigen protocol keeps depths in (1e-3, 10) m inside the Eigen crop.
DEPTH_SCALE = 1000.0
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0
EIGEN_CROP = (slice(45, 471), slice(41, 601))
GROUND_TRUTH_SHAPE = (480, 640)
# Upstream converts the metric head's output to meters as focal * depth / 300, with the focal length in
# input pixels. NYU's color camera has fx 518.8579 and fy 519.4696 at 640x480; resizing scales it, and
# cropping keeps it.
NYU_FOCAL = (518.8579 + 519.4696) / 2
CANONICAL_FOCAL = 300.0
DELTA = 1.25
# Published metrics and whether a lower or higher value is better.
DIRECTIONS = {"abs_rel": "lower", "delta1": "higher"}
# Images decode on worker threads while the MPK runs; OpenCV releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized, cropped and normalized the image; Neat still converts and tessellates.
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


def frames(dataset: Path) -> list[Path]:
    """Return the Eigen test RGB frames in frame-number order."""
    paths = [path for split in SPLITS for path in (dataset / split).glob("rgb_*.jpg")]
    return sorted(paths, key=lambda path: int(path.stem.removeprefix("rgb_")))


def ground_truth_path(frame: Path) -> Path:
    return frame.with_name(frame.stem.replace("rgb_", "sync_depth_", 1) + ".png")


def read_ground_truth(path: Path) -> np.ndarray:
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.shape != GROUND_TRUTH_SHAPE:
        raise ValueError(f"expected a 16-bit {GROUND_TRUTH_SHAPE} depth PNG: {path}")
    return raw.astype(np.float64) / DEPTH_SCALE


def load(frame: Path) -> tuple[np.ndarray, np.ndarray]:
    return preprocess_image(frame.read_bytes()), read_ground_truth(ground_truth_path(frame))


def paired(depth: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Pair each valid ground-truth pixel in the Eigen crop with the network pixel that covers it.

    Returns the network depths, the ground-truth depths in meters, and the resize scale. Ground-truth pixels
    that the center crop removed are not scored.
    """
    rows, columns = truth.shape
    scale, (resized_width, resized_height), (left, top) = cover_geometry(columns, rows)
    valid = (truth > MIN_DEPTH) & (truth < MAX_DEPTH)
    crop = np.zeros_like(valid)
    crop[EIGEN_CROP] = True
    ys, xs = np.nonzero(valid & crop)
    u = np.floor((xs + 0.5) * (resized_width / columns) - left).astype(int)
    v = np.floor((ys + 0.5) * (resized_height / rows) - top).astype(int)
    keep = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    return depth[v[keep], u[keep]].astype(np.float64), truth[ys[keep], xs[keep]], scale


def score(depth: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return one image's AbsRel and delta1 for the network's canonical depth against meters."""
    pixels, meters, scale = paired(depth, truth)
    if not len(meters):
        raise ValueError("no valid ground-truth pixel inside the network view")
    prediction = np.maximum(NYU_FOCAL * scale * pixels / CANONICAL_FOCAL, 1e-6)
    ratio = np.maximum(prediction / meters, meters / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(prediction - meters) / meters)),
        "delta1": float(np.mean(ratio < DELTA)),
        "pixels": len(meters),
    }


def depth_output(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Return the depth map from the MPK's depth and sky outputs."""
    if len(outputs) != 2 or any(output.size != HEIGHT * WIDTH for output in outputs):
        raise ValueError(f"unexpected MPK outputs: {[output.shape for output in outputs]}")
    depth = outputs[0].reshape(HEIGHT, WIDTH)
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
    parser.add_argument("--dataset", type=Path, required=True, help="extracted nyu-depth-v2/default directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="measure a seeded sample of frames (local runs only)")
    args = parser.parse_args(argv)

    available = frames(args.dataset)
    if len(available) != FRAMES:
        raise SystemExit(f"expected {FRAMES} Eigen test frames in {', '.join(SPLITS)}, found {len(available)}")
    selected = sorted(random.Random(0).sample(range(FRAMES), args.limit)) if args.limit else range(FRAMES)
    images = [available[index] for index in selected]

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    rows: list[dict[str, float]] = []
    started = time.monotonic()
    for count, (image, truth) in enumerate(prefetched(load, images), start=1):
        # An HxWx3 array has no unambiguous layout of its own.
        tensor = pyneat.Tensor.from_numpy(image, copy=False, layout=pyneat.TensorLayout.HWC)
        if runner is None:
            runner = model.build([tensor])
        outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
        rows.append(score(depth_output(outputs), truth))
        if count == 1 or count % 100 == 0 or count == len(images):
            log(f"{count}/{len(images)} frames, {(time.monotonic() - started) / count * 1000:.0f} ms/frame")
    inference_seconds = time.monotonic() - started

    # Each metric is the mean of per-image values, as in the Eigen protocol.
    stats = {name: float(np.mean([row[name] for row in rows])) for name in DIRECTIONS}
    result = {
        "core_version": pyneat.__version__,
        "dataset": {"split": "+".join(SPLITS), "evaluated_samples": len(images), "available_samples": FRAMES},
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
