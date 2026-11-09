"""Benchmark LightGlue with Neat Core on one keypoint pair, including its CPU steps, and save its report.

LightGlue takes keypoints, not images. Given --dataset yfcc100m-keypoints-val=<extracted directory>, the benchmark
repeats one random.Random(0) val pair whose images both have 512 keypoints, the SuperPoint keypoints accuracy.py
matches. Each measured pair runs the application's steps, as accuracy.py does: the CPU computes both images' keypoint
encodings and copies the inputs to EV74 memory, the MPK computes the assignment, and the CPU extracts the mutual
matches, the pair's result. Latency runs one pair at a time and reports the MPK's and the CPU's share. The throughput
pass runs the runner asynchronously, as SuperGlue's does: one thread encodes and pushes each pair, one pulls each
pair's assignment, and MATCH_WORKERS threads extract the matches, so the MPK works on one pair while the CPU encodes
the next and extracts earlier ones. Core's own benchmark of the MPK supplies only the MLA layer and stage stats.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import INPUT_SHAPES, KEYPOINTS, Pair, dataset_root, features, matcher_inputs, matches, usable_pairs

WARMUP_PAIRS = 10
MEASURED_PAIRS = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Pairs the runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
# Threads extracting matches on the CPU in the throughput pass, one per core.
MATCH_WORKERS = os.cpu_count() or 1

# One image's 512 keypoints [512, 2] in pixels of the 640x480 input and its descriptors [512, 256].
ImageFeatures = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LatencyResult:
    """Seconds per pair, one pair at a time, and the MPK's part of each."""

    pair_s: list[float]
    model_s: list[float]


@dataclass(frozen=True)
class ThroughputResult:
    pairs: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.pairs / self.seconds


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script encodes the keypoints; Neat only quantizes and tessellates the inputs.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def input_tensors(inputs: dict[str, np.ndarray]) -> list[pyneat.Tensor]:
    """The matcher's inputs as Neat tensors in EV74 memory."""
    return [pyneat.Tensor.from_numpy(inputs[name], copy=True, memory=pyneat.TensorMemory.EV74) for name in INPUT_SHAPES]


def pair_tensors(images: tuple[ImageFeatures, ImageFeatures]) -> list[pyneat.Tensor]:
    """The CPU step before the MPK: both images' keypoint encodings, and the inputs copied to EV74 memory."""
    return input_tensors(matcher_inputs(*images))


def pair_matches(assignment: pyneat.Tensor) -> np.ndarray:
    """The CPU step after the MPK: the mutual matches in the pair's assignment."""
    return matches(assignment.to_numpy(copy=False))


def benchmark_pair(dataset: Path) -> tuple[Pair, tuple[ImageFeatures, ImageFeatures]]:
    """One val pair whose images both have 512 keypoints, chosen with a fixed seed so every run measures the same
    input, and both images' keypoints and descriptors."""
    root = dataset_root(dataset)
    pair = random.Random(0).choice(sorted(usable_pairs(root), key=lambda item: (item.name0, item.name1)))
    return pair, (features(root, pair.name0), features(root, pair.name1))


def mla_stats(mpk: str, tensors: list[pyneat.Tensor]) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for the MPK on the pair. Their throughput is the MPK's, not the model's, so
    Core's default run length serves."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    options.inputs = tensors
    report = pyneat.Model(mpk, tensor_options()).benchmark(options)
    layer_stats, stage_stats = report.mla_layer_stats, report.mla_stage_stats
    layers = [
        {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
        for layer in layer_stats.layers
    ]
    stages = [{"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stage_stats.stages]
    return {"status": layer_stats.status, "layers": layers}, {"status": stage_stats.status, "stages": stages}


def build_runner(model: pyneat.Model, tensors: list[pyneat.Tensor]) -> pyneat.ModelRunner:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    return model.build(tensors, run_options=run_options)


def match_pair(runner: pyneat.ModelRunner, images: tuple[ImageFeatures, ImageFeatures]) -> np.ndarray:
    """One pair's application steps: encoding, the MPK and match extraction."""
    (assignment,) = runner.run(pair_tensors(images), timeout_ms=TIMEOUT_MS)
    return pair_matches(assignment)


def measure_latency(runner: pyneat.ModelRunner, images: tuple[ImageFeatures, ImageFeatures]) -> LatencyResult:
    """Each pair one at a time after the warm-up, timing the whole pair and its MPK run."""
    for _ in range(WARMUP_PAIRS):
        match_pair(runner, images)
    pair_s: list[float] = []
    model_s: list[float] = []
    started = time.perf_counter()
    while len(pair_s) < MEASURED_PAIRS or time.perf_counter() - started < MIN_DURATION_S:
        begun = time.perf_counter()
        tensors = pair_tensors(images)
        model_begun = time.perf_counter()
        (assignment,) = runner.run(tensors, timeout_ms=TIMEOUT_MS)
        model_s.append(time.perf_counter() - model_begun)
        pair_matches(assignment)
        pair_s.append(time.perf_counter() - begun)
    return LatencyResult(pair_s, model_s)


def measure_throughput(runner: pyneat.ModelRunner, images: tuple[ImageFeatures, ImageFeatures]) -> ThroughputResult:
    """Matched pairs per second and board telemetry with the MPK and the CPU busy at once. One thread encodes and
    pushes each pair, this one pulls each pair's assignment and hands it to MATCH_WORKERS extraction threads. The
    measured span starts after WARMUP_PAIRS matched pairs and lasts at least MEASURED_PAIRS pairs and MIN_DURATION_S
    seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()

    def push_pairs() -> None:
        try:
            while not measured.is_set():
                if not runner.push(pair_tensors(images)) and not measured.is_set():
                    raise RuntimeError("the runner rejected a pair")
        finally:
            runner.close_input()

    pending: collections.deque[Future] = collections.deque()
    matched, started = 0, 0.0
    result: ThroughputResult | None = None
    with ThreadPoolExecutor(MATCH_WORKERS + 1) as pool:
        pusher = pool.submit(push_pairs)
        try:
            while outputs := runner.pull(TIMEOUT_MS).tensors:
                assignment = outputs[0].to_numpy(copy=True)
                # Return the output buffer to the runner before the next pull.
                del outputs
                pending.append(pool.submit(matches, assignment))
                # Count each pair once 2 * MATCH_WORKERS later pairs queue behind it, so the workers stay busy.
                if len(pending) <= 2 * MATCH_WORKERS:
                    continue
                pending.popleft().result()
                matched += 1
                if matched == WARMUP_PAIRS:
                    monitor.start()
                    started = time.perf_counter()
                elif matched > WARMUP_PAIRS and result is None:
                    pairs, seconds = matched - WARMUP_PAIRS, time.perf_counter() - started
                    if pairs >= MEASURED_PAIRS and seconds >= MIN_DURATION_S:
                        result = ThroughputResult(pairs, seconds, monitor.stop())
                        measured.set()
        finally:
            measured.set()
            # Wakes a push waiting on a full queue, so the pusher ends even when a pull failed.
            runner.close_input()
            # A failure inside the measured span leaves the monitor running.
            if monitor.running():
                monitor.stop()
        pusher.result()
        for match in pending:
            match.result()
    if result is None:
        raise RuntimeError(f"no pair came out within {TIMEOUT_MS} ms before the measured span ended")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="yfcc100m-keypoints-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    pair, images = benchmark_pair(Path(dataset_dir))

    layer_stats, stage_stats = mla_stats(args.mpk, pair_tensors(images))
    model = pyneat.Model(args.mpk, tensor_options())
    runner = build_runner(model, pair_tensors(images))
    try:
        latency = measure_latency(runner, images)
        throughput = measure_throughput(runner, images)
    finally:
        runner.close()
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latency.pair_s) * 1000)
    benchmark.latency_p50_ms = float(np.percentile(latency.pair_s, 50) * 1000)
    benchmark.latency_p95_ms = float(np.percentile(latency.pair_s, 95) * 1000)
    benchmark.latency_p99_ms = float(np.percentile(latency.pair_s, 99) * 1000)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.pairs
    report = model_report.from_performance(
        benchmark,
        {"model": model},
        workload=(
            f"{dataset_id} pair {pair.name0} and {pair.name1}, {KEYPOINTS} SuperPoint keypoints per image: keypoint "
            f"encoding on the CPU, the matcher on the tensor route, then mutual-match extraction on the CPU; latency "
            f"one pair at a time, throughput with the runner's async push and pull and {MATCH_WORKERS} CPU "
            f"extraction threads"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats={"model": layer_stats},
        mla_stage_stats={"model": stage_stats},
    )
    report.save(args.output)
    model_ms = float(np.mean(latency.model_s) * 1000)
    print(
        f"{throughput.fps:.1f} pairs/s over {throughput.pairs} pairs; latency {benchmark.latency_ms:.2f} ms per pair: "
        f"MPK {model_ms:.2f} ms, CPU encoding and extraction {benchmark.latency_ms - model_ms:.2f} ms"
    )


if __name__ == "__main__":
    main()
