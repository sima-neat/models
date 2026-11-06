"""Measure RAFT-Stereo realtime's disparity error on the Scene Flow finalpass test pairs with its MPK on a Modalix
board.

Each pair the test list names runs through the MPK as one side-by-side frame, prepared as calibration prepares it: both
views ImageNet-normalized by scene_flow.preprocess_image, the left view beside the right one. The left view's
disparity is scored against the pair's true disparity with OpenStereo's per-image metrics:

- a pixel counts when its true disparity is above 0 and under 192 (scene_flow.MAX_DISPARITY), the bound of
  RAFT-Stereo's own Scene Flow evaluation;
- EPE is the mean absolute disparity error over a pair's counted pixels, in pixels (scene_flow.end_point_error);
- D1-all is the percentage of a pair's counted pixels whose error exceeds both 3 px and 5% of the true disparity;
- each metric is the mean of its per-pair values, and a pair without a counted pixel scores 0 for both.
"""

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

import model_report
import numpy as np
import pyneat
from scene_flow import HEIGHT, MAX_DISPARITY, WIDTH, Pair, end_point_error, preprocess_image, read_pfm

# Each line names a pair's left view, right view and left-view disparity, relative to the dataset directory.
TEST_LIST = "sceneflow_finalpass_test.txt"
SPLIT = "test"
# D1 counts a pixel whose error exceeds both D1_PIXELS and D1_FRACTION of its true disparity.
D1_PIXELS, D1_FRACTION = 3.0, 0.05
# Neat returns the left view's disparity NHWC, one channel.
OUTPUT_SHAPE = (1, HEIGHT, WIDTH, 1)
# Pairs decode and normalize on worker threads while the MPK runs; Pillow and NumPy release the GIL for most of it.
WORKERS = 4
PREFETCH = 8


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def listed_pairs(dataset: Path) -> list[Pair]:
    """The (left, right, disparity) paths of every pair the test list names, in list order."""
    lines = (dataset / TEST_LIST).read_text(encoding="utf-8").splitlines()
    pairs = [tuple(line.split()) for line in lines if line.strip()]
    if not pairs or any(len(pair) != 3 for pair in pairs):
        raise SystemExit(f"{dataset / TEST_LIST} must name a left view, a right view and a disparity per line")
    return pairs


def read_pair(dataset: Path, pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    """One pair's model input, the [left | right] side-by-side frame, and its left view's true disparity."""
    left, right, disparity = (dataset / name for name in pair)
    frame = np.concatenate([preprocess_image(left.read_bytes()), preprocess_image(right.read_bytes())], axis=1)
    return frame, read_pfm(disparity.read_bytes())


def d1_all(disparity: np.ndarray, truth: np.ndarray) -> float:
    """The percentage of the counted pixels whose error exceeds both 3 px and 5% of the true disparity, and 0 for a
    pair without a counted pixel."""
    counted = (truth > 0) & (truth < MAX_DISPARITY)
    if not counted.any():
        return 0.0
    error = np.abs(disparity.reshape(truth.shape) - truth)[counted]
    return float(100 * np.mean((error > D1_PIXELS) & (error / truth[counted] > D1_FRACTION)))


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
    parser.add_argument("--dataset", required=True, help="scene-flow-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of pairs (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    available = listed_pairs(dataset)
    if args.limit:
        pairs = [available[index] for index in sorted(random.Random(0).sample(range(len(available)), args.limit))]
    else:
        pairs = available

    model = pyneat.Model(str(parse_mpks(args.mpk)))
    runner = None
    epe, d1 = [], []
    started = time.monotonic()
    frames = prefetched(lambda pair: read_pair(dataset, pair), pairs)
    for count, (pair, (frame, truth)) in enumerate(zip(pairs, frames), start=1):
        tensor = pyneat.Tensor.from_numpy(frame, copy=False)
        if runner is None:
            runner = model.build([tensor])
        disparity = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False)
        if disparity.shape != OUTPUT_SHAPE or not np.isfinite(disparity).all():
            raise ValueError(f"unexpected disparity for {pair[0]}: shape {disparity.shape}")
        epe.append(end_point_error(disparity, truth))
        d1.append(d1_all(disparity, truth))
        if count == 1 or count % 250 == 0 or count == len(pairs):
            log(f"{count}/{len(pairs)} pairs, {(time.monotonic() - started) / count * 1000:.0f} ms/pair")

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {"epe": float(np.mean(epe)), "d1_all": float(np.mean(d1))},
            "split": SPLIT, "evaluated_samples": len(pairs), "available_samples": len(available),
            "details": {"inference_seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
