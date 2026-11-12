"""Score all 654 NYU test images, with per-image inverse-depth scale/shift alignment.

The registered archive stores the official test set across val, cal and test;
train is excluded. Ground truth is millimeter PNG. Only valid Eigen-crop pixels
retained by the cover resize and center crop are scored, using the containing
network pixel. This crop-specific relative-depth protocol is not a metric-depth
score or directly comparable to full-image published results.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from preprocess import HEIGHT, WIDTH, cover_geometry, preprocess_image

DATASET = "nyu-depth-v2-default"
SPLITS = ("val", "cal", "test")
FRAMES = 654
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


def frames(dataset: Path) -> list[Path]:
    paths = sorted(
        (path for split in SPLITS for path in (dataset / split).glob("rgb_*.jpg")),
        key=lambda path: int(path.stem.removeprefix("rgb_")),
    )
    if len(paths) != FRAMES or len({path.name for path in paths}) != FRAMES:
        raise ValueError(f"expected {FRAMES} distinct NYU test images in {SPLITS}, found {len(paths)}")
    return paths


def selected_frames(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    if not 1 <= limit <= len(paths):
        raise ValueError(f"--limit must be between 1 and {len(paths)}")
    return [paths[index] for index in sorted(random.Random(0).sample(range(len(paths)), limit))]


def read_ground_truth(frame: Path) -> np.ndarray:
    import cv2

    path = frame.with_name(frame.stem.replace("rgb_", "sync_depth_", 1) + ".png")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.shape != (480, 640):
        raise ValueError(f"expected a 480x640 uint16 depth PNG: {path}")
    return raw.astype(np.float64) / 1000


def score(depth: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if depth.shape != (HEIGHT, WIDTH) or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("expected a finite, positive depth map at the model resolution")
    if truth.shape != (480, 640):
        raise ValueError("expected NYU's 480x640 ground truth")
    _, (rw, rh), (left, top) = cover_geometry(640, 480)
    crop = np.zeros(truth.shape, dtype=bool)
    crop[45:471, 41:601] = True
    ys, xs = np.nonzero(crop & (truth > MIN_DEPTH) & (truth < MAX_DEPTH) & np.isfinite(truth))
    u = np.floor((xs + 0.5) * (rw / 640) - left).astype(int)
    v = np.floor((ys + 0.5) * (rh / 480) - top).astype(int)
    keep = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    meters = truth[ys[keep], xs[keep]]
    if not meters.size:
        raise ValueError("no valid ground-truth pixels inside the retained image")
    disparity = 1 / depth[v[keep], u[keep]].astype(np.float64)
    design = np.stack([disparity, np.ones_like(disparity)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, 1 / meters, rcond=None)
    prediction = 1 / np.maximum(scale * disparity + shift, 1 / MAX_DEPTH)
    ratio = np.maximum(prediction / meters, meters / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(prediction - meters) / meters)),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def depth_output(outputs: list[np.ndarray]) -> np.ndarray:
    if len(outputs) != 2 or any(output.size != HEIGHT * WIDTH for output in outputs):
        raise ValueError(f"expected depth and depth_conf maps, found {[output.shape for output in outputs]}")
    if not all(np.isfinite(output).all() for output in outputs):
        raise ValueError("model outputs contain non-finite values")
    return outputs[0].reshape(HEIGHT, WIDTH)


def evaluate(images: list[Path], infer: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    rows = []
    started = time.monotonic()
    for count, frame in enumerate(images, start=1):
        rows.append(score(infer(preprocess_image(frame.read_bytes())), read_ground_truth(frame)))
        if count == 1 or count % 100 == 0 or count == len(images):
            print(f"{count}/{len(images)} images; {(time.monotonic() - started) / count:.2f} s/image", flush=True)
    return {name: float(np.mean([row[name] for row in rows])) for name in ("abs_rel", "delta1")}


def main() -> None:
    import model_report
    import pyneat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<archive>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<directory>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="seeded image sample, for local runs only")
    args = parser.parse_args()
    mpks = [value.split("=", 1) for value in args.mpk]
    if len(mpks) != 1 or len(mpks[0]) != 2 or mpks[0][0] != "model":
        parser.error("--mpk must name exactly model=<archive>")
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        parser.error(f"--dataset must name {DATASET}")
    available = frames(Path(directory))
    images = selected_frames(available, args.limit)
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(mpks[0][1], options)
    runner = None

    def infer(image):
        nonlocal runner
        tensor = pyneat.Tensor.from_numpy(image, copy=False, layout=pyneat.TensorLayout.HWC)
        if runner is None:
            runner = model.build([tensor])
        outputs = [value.to_numpy(copy=False) for value in runner.run([tensor], timeout_ms=120000)]
        return depth_output(outputs)

    metrics = evaluate(images, infer)
    model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": metrics,
                "split": "+".join(SPLITS),
                "evaluated_samples": len(images),
                "available_samples": len(available),
                "details": {
                    "alignment": "per-image least-squares scale and shift in inverse depth",
                    "crop": "Eigen crop intersected with the retained network field of view",
                },
            }
        }
    ).save(args.output)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
