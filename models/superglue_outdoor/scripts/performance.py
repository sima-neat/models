"""Benchmark SuperGlue with Neat Core on one keypoint pair, including its CPU matching, and save its report.

SuperGlue takes keypoints, not images. With --dataset yfcc100m-keypoints-val=<extracted directory> the benchmark
repeats two random.Random(0) images of that dataset with 512 keypoints each, the SuperPoint FP32 keypoints
accuracy.py matches; without it, a fixed seeded pair of 512 keypoints. Each measured pair runs the MPK and then
the application's CPU steps, the 20 Sinkhorn iterations and the mutual-match extraction (matching.py), as
accuracy.py does. Core's own benchmark of the MPK supplies only the MLA layer and stage stats.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from matching import MATCH_THRESHOLD, extract_matches, log_sinkhorn, normalize_keypoints

DATASET = "yfcc100m-keypoints-val"
IMAGES_FILE = "annotations/images.txt"
KEYPOINTS, DESCRIPTOR_SIZE = 512, 256
# The MPK's channels-last inputs, in its order.
INPUTS = ("keypoints0", "scores0", "descriptors0", "keypoints1", "scores1", "descriptors1")
# SuperPoint's input size, which the keypoints' pixel coordinates refer to.
FIXED_WIDTH, FIXED_HEIGHT = 640, 480
# As accuracy.py and SuperGlue's match_pairs.py evaluation.
SINKHORN_ITERATIONS = 20
WARMUP_PAIRS = 10
MEASURED_PAIRS = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000


def dataset_images(dataset: Path) -> tuple[str, list]:
    """Two random.Random(0) images with 512 stored keypoints each: their names and each one's
    (keypoints, scores, descriptors, size)."""
    rows = [row.split() for row in (dataset / IMAGES_FILE).read_text(encoding="utf-8").splitlines()]
    full = [row[0] for row in rows if int(row[4]) == KEYPOINTS]
    names = random.Random(0).sample(full, 2)
    images = []
    for name in names:
        stem = dataset / "val" / str(Path(name).with_suffix(""))
        arrays = [np.load(stem / f"{key}.npy", allow_pickle=False) for key in ("keypoints", "scores", "descriptors")]
        images.append((*arrays, (FIXED_WIDTH, FIXED_HEIGHT)))
    return f"{DATASET} {names[0]} and {names[1]}", images


def fixed_images() -> tuple[str, list]:
    """Two seeded sets of 512 keypoints in a 1600x1200 image, with scores and unit descriptors."""
    rng = np.random.RandomState(0)
    images = []
    for _ in range(2):
        keypoints = rng.uniform((0, 0), (FIXED_WIDTH, FIXED_HEIGHT), (KEYPOINTS, 2))
        descriptors = rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_SIZE))
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        images.append((keypoints, rng.uniform(0.005, 1, KEYPOINTS), descriptors, (FIXED_WIDTH, FIXED_HEIGHT)))
    return f"a seeded pair of {FIXED_WIDTH}x{FIXED_HEIGHT} images", images


def pair_tensors(images: list) -> list[np.ndarray]:
    """The pair as the MPK's channels-last inputs, each image's keypoints normalized by its size."""
    values = {}
    for side, (keypoints, scores, descriptors, (width, height)) in enumerate(images):
        values[f"keypoints{side}"] = normalize_keypoints(keypoints, width, height).reshape(1, 1, KEYPOINTS, 2)
        values[f"scores{side}"] = np.asarray(scores, np.float32).reshape(1, 1, KEYPOINTS, 1)
        values[f"descriptors{side}"] = np.asarray(descriptors, np.float32).reshape(1, 1, KEYPOINTS, DESCRIPTOR_SIZE)
    return [np.ascontiguousarray(values[name]) for name in INPUTS]


def tensor_options() -> pyneat.ModelOptions:
    """The inputs are keypoint features, not images: Neat only quantizes and tessellates them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def mla_stats(mpk: str) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for the MPK. Their throughput is the MPK's, not the model's, so Core's
    default run serves."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    report = pyneat.Model(mpk).benchmark(options)
    layer_stats, stage_stats = report.mla_layer_stats, report.mla_stage_stats
    layers = [
        {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
        for layer in layer_stats.layers
    ]
    stages = [{"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stage_stats.stages]
    return {"status": layer_stats.status, "layers": layers}, {"status": stage_stats.status, "stages": stages}


def match_pair(runner, tensors: list) -> np.ndarray:
    """The MPK's couplings, then the CPU's Sinkhorn iterations and mutual-match extraction: the pair's matches."""
    (output,) = runner.run(tensors, timeout_ms=TIMEOUT_MS)
    couplings = output.to_numpy(copy=False).reshape(KEYPOINTS + 1, KEYPOINTS + 1)
    matches, _ = extract_matches(log_sinkhorn(couplings, SINKHORN_ITERATIONS), MATCH_THRESHOLD)
    return matches


def measure(model: pyneat.Model, tensors: list) -> tuple[list[float], float, pyneat.Telemetry]:
    """Seconds per pair after the warm-up, the measured span's length, and board telemetry over it."""
    runner = model.build(tensors)
    try:
        for _ in range(WARMUP_PAIRS):
            match_pair(runner, tensors)
        monitor = pyneat.TelemetryMonitor()
        monitor.start()
        seconds: list[float] = []
        started = time.perf_counter()
        while len(seconds) < MEASURED_PAIRS or time.perf_counter() - started < MIN_DURATION_S:
            begun = time.perf_counter()
            match_pair(runner, tensors)
            seconds.append(time.perf_counter() - begun)
        span = time.perf_counter() - started
        telemetry = monitor.stop()
    finally:
        runner.close()
    return seconds, span, telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    if args.dataset:
        dataset_id, dataset_dir = args.dataset.split("=", 1)
        if dataset_id != DATASET:
            raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
        workload, images = dataset_images(Path(dataset_dir))
    else:
        workload, images = fixed_images()

    layer_stats, stage_stats = mla_stats(args.mpk)
    model = pyneat.Model(args.mpk, tensor_options())
    tensors = [pyneat.Tensor.from_numpy(value, copy=True) for value in pair_tensors(images)]
    seconds, span, telemetry = measure(model, tensors)
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(seconds) * 1000)
    benchmark.latency_p50_ms = float(np.percentile(seconds, 50) * 1000)
    benchmark.latency_p95_ms = float(np.percentile(seconds, 95) * 1000)
    benchmark.latency_p99_ms = float(np.percentile(seconds, 99) * 1000)
    benchmark.fps = len(seconds) / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = len(seconds)
    report = model_report.from_performance(
        benchmark,
        {"model": model},
        workload=(
            f"{workload}, {KEYPOINTS} keypoints each, on the tensor route, then {SINKHORN_ITERATIONS} Sinkhorn "
            "iterations and mutual-match extraction on the CPU, one pair at a time"
        ),
        telemetry=telemetry,
        mla_layer_stats={"model": layer_stats},
        mla_stage_stats={"model": stage_stats},
    )
    report.save(args.output)
    print(f"{benchmark.fps:.1f} pairs/s over {len(seconds)} pairs; latency {benchmark.latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
