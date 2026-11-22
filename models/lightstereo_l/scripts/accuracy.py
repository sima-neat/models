"""Measure LightStereo-L's disparity error on the Scene Flow finalpass test pairs with its MPK on a Modalix board.

Each pair the test list names runs through the MPK as one side-by-side frame, prepared as OpenStereo's evaluation
prepares it: both views read as RGB, scaled to [0, 1] and ImageNet-normalized by scene_flow.preprocess_image, the left
view beside the right one. The MPK pads each view to 544x960 itself, as OpenStereo's RightTopPad does, and returns
the 540x960 left view's disparity. It is scored against the pair's true disparity with OpenStereo's per-image
metrics (stereo/evaluation/metric_per_image.py):

- a pixel counts when its true disparity is above 0 and under 192 (scene_flow.MAX_DISPARITY), the config's
  EVALUATOR MAX_DISP;
- EPE is the mean absolute disparity error over a pair's counted pixels, in pixels (scene_flow.end_point_error);
- D1-all is the percentage of a pair's counted pixels whose error exceeds both 3 px and 5% of the true disparity;
- each metric is the mean of its per-pair values, and a pair without a counted pixel scores 0 for both.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
from model_report import log
from scene_flow import HEIGHT, MAX_DISPARITY, WIDTH, Pair, end_point_error, preprocess_image, read_pfm

DATASET = "scene-flow-val"
# Each line names a pair's left view, right view and left-view disparity, relative to the dataset directory.
TEST_LIST = "sceneflow_finalpass_test.txt"
SPLIT = "test"
# D1 counts a pixel whose error exceeds both D1_PIXELS and D1_FRACTION of its true disparity.
D1_PIXELS, D1_FRACTION = 3.0, 0.05
# Pairs decode and normalize on worker threads while the model runs; Pillow and NumPy release the GIL for most of it.
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


def sampled_pairs(pairs: list[Pair], limit: int | None) -> list[Pair]:
    """Every pair, or a seeded sample of limit pairs in list order for a quick local run."""
    if not limit:
        return pairs
    return [pairs[index] for index in sorted(random.Random(0).sample(range(len(pairs)), limit))]


def read_pair(dataset: Path, pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    """One pair's model input, the [left | right] side-by-side HWC frame, and its left view's true disparity."""
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


def evaluate(dataset: Path, pairs: list[Pair], disparity_of: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    """Score disparity_of, which maps one HWC frame to its 540x960 disparity, on the pairs: mean EPE and D1-all."""
    epe, d1 = [], []
    started = time.monotonic()
    frames = prefetched(lambda pair: read_pair(dataset, pair), pairs)
    for count, (pair, (frame, truth)) in enumerate(zip(pairs, frames), start=1):
        disparity = disparity_of(frame)
        if disparity.size != HEIGHT * WIDTH or not np.isfinite(disparity).all():
            raise ValueError(f"unexpected disparity for {pair[0]}: shape {disparity.shape}")
        epe.append(end_point_error(disparity, truth))
        d1.append(d1_all(disparity, truth))
        if count == 1 or count % 250 == 0 or count == len(pairs):
            log(f"{count}/{len(pairs)} pairs, {(time.monotonic() - started) / count * 1000:.0f} ms/pair")
    return {"epe": float(np.mean(epe)), "d1_all": float(np.mean(d1))}


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of pairs")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    dataset = Path(directory)
    available = listed_pairs(dataset)
    pairs = sampled_pairs(available, args.limit)

    import pyneat

    options = pyneat.ModelOptions()
    # The frames arrive normalized, as model input: Neat only quantizes and tessellates them.
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(str(parse_mpks(args.mpk)), options)
    runner = None

    def disparity_of(frame: np.ndarray) -> np.ndarray:
        nonlocal runner
        tensor = pyneat.Tensor.from_numpy(frame, copy=False)
        if runner is None:
            runner = model.build([tensor])
        return runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False)

    started = time.monotonic()
    metrics = evaluate(dataset, pairs, disparity_of)
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": SPLIT, "evaluated_samples": len(pairs), "available_samples": len(available),
            "details": {"inference_seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
