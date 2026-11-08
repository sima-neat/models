"""Benchmark SuperGlue with Neat Core on one keypoint pair, including its CPU matching, and save its report.

SuperGlue takes keypoints, not images. Given --dataset yfcc100m-keypoints-val=<extracted directory>, the benchmark
repeats one random.Random(0) pair of annotations/pairs_val.txt whose images both hold 512 keypoints, the SuperPoint
FP32 keypoints accuracy.py matches. Each measured pair runs the MPK
and then the application's CPU steps, the 20 Sinkhorn iterations and the mutual-match extraction, as accuracy.py
does: SuperGlue's own model ends with them, so a pair's matches are its result. Latency runs one pair at a time and
reports the MPK's and the CPU's share. The CPU steps take longer than the MPK, so the throughput pass matches pairs in
NEAT_WORKERS processes, each with its own runner and an equal share of the CPU cores, as an application with the
DevKit's cores would. Core's own benchmark of the MPK supplies only the MLA layer and stage stats.
"""

from __future__ import annotations

import argparse
import collections
import gc
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

DATASET = "yfcc100m-keypoints-val"
IMAGES_FILE = "annotations/images.txt"
PAIRS_FILE = "annotations/pairs_val.txt"
KEYPOINTS, DESCRIPTOR_SIZE = 512, 256
# The MPK's channels-last inputs, in its order.
INPUTS = ("keypoints0", "scores0", "descriptors0", "keypoints1", "scores1", "descriptors1")
# SuperPoint's input size, which the keypoints' pixel coordinates refer to.
INPUT_WIDTH, INPUT_HEIGHT = 640, 480
# As accuracy.py and SuperGlue's match_pairs.py evaluation.
SINKHORN_ITERATIONS = 20
MATCH_THRESHOLD = 0.2
WARMUP_PAIRS = 10
MEASURED_PAIRS = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Throughput-pass processes, each running the MPK and the CPU matching. Neat Core starts at most four accelerator
# clients at once.
NEAT_WORKERS = 4
# Seconds between the workers' starts: runners built at the same moment can find the accelerator busy.
NEAT_START_STAGGER_S = 2.0


@dataclass(frozen=True)
class ImageKeypoints:
    """One image's SuperPoint output, in pixels of the 640 x 480 input."""

    keypoints: np.ndarray  # [KEYPOINTS, 2]
    scores: np.ndarray  # [KEYPOINTS]
    descriptors: np.ndarray  # [KEYPOINTS, DESCRIPTOR_SIZE]


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


def dataset_pair(dataset: Path) -> tuple[str, list[ImageKeypoints]]:
    """One random.Random(0) pair of pairs_val.txt whose images both hold 512 stored keypoints, with its name."""
    counts = {
        row.split()[0]: int(row.split()[4]) for row in (dataset / IMAGES_FILE).read_text(encoding="utf-8").splitlines()
    }
    lines = (dataset / PAIRS_FILE).read_text(encoding="utf-8").splitlines()
    pairs = [line.split()[:2] for line in lines if line.strip()]
    full = [names for names in pairs if all(counts.get(name) == KEYPOINTS for name in names)]
    if not full:
        raise SystemExit(f"{PAIRS_FILE} lists no pair whose images both hold {KEYPOINTS} keypoints")
    names = random.Random(0).choice(full)
    images = []
    for name in names:
        stem = dataset / "val" / str(Path(name).with_suffix(""))
        arrays = [np.load(stem / f"{key}.npy", allow_pickle=False) for key in ("keypoints", "scores", "descriptors")]
        images.append(ImageKeypoints(*arrays))
    return f"the {DATASET} pair {names[0]} and {names[1]}", images


def normalize_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map pixel (x, y) keypoints to SuperGlue's frame: centred on the image, divided by 0.7 times its longer side."""
    size = np.array([width, height], dtype=np.float64)
    return ((np.asarray(keypoints, dtype=np.float64) - size / 2) / (size.max() * 0.7)).astype(np.float32)


def pair_tensors(images: list[ImageKeypoints]) -> list[np.ndarray]:
    """The pair as the MPK's channels-last inputs, each image's keypoints normalized by the input size."""
    values = {}
    for side, image in enumerate(images):
        keypoints = normalize_keypoints(image.keypoints, INPUT_WIDTH, INPUT_HEIGHT)
        values[f"keypoints{side}"] = keypoints.reshape(1, 1, KEYPOINTS, 2)
        values[f"scores{side}"] = np.asarray(image.scores, np.float32).reshape(1, 1, KEYPOINTS, 1)
        descriptors = np.asarray(image.descriptors, np.float32)
        values[f"descriptors{side}"] = descriptors.reshape(1, 1, KEYPOINTS, DESCRIPTOR_SIZE)
    return [np.ascontiguousarray(values[name]) for name in INPUTS]


def log_sinkhorn(couplings: np.ndarray, iterations: int) -> np.ndarray:
    """Return the log optimal-transport matrix of [M+1, N+1] couplings, as SuperGlue's log_optimal_transport."""
    rows, columns = couplings.shape[-2] - 1, couplings.shape[-1] - 1
    normalization = -np.log(rows + columns)
    log_mu = np.concatenate([np.full(rows, normalization), [np.log(columns) + normalization]])
    log_nu = np.concatenate([np.full(columns, normalization), [np.log(rows) + normalization]])
    cost = np.asarray(couplings, dtype=np.float64).reshape(rows + 1, columns + 1)
    u, v = np.zeros_like(log_mu), np.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - _logsumexp(cost + v[None, :], axis=1)
        v = log_nu - _logsumexp(cost + u[:, None], axis=0)
    return cost + u[:, None] + v[None, :] - normalization


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    top = values.max(axis=axis, keepdims=True)
    return (top + np.log(np.exp(values - top).sum(axis=axis, keepdims=True))).squeeze(axis)


def extract_matches(log_transport: np.ndarray, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return, for each image-0 keypoint, its mutual nearest image-1 keypoint (-1 when unmatched) and match score."""
    inner = log_transport[:-1, :-1]
    best0, best1 = inner.argmax(1), inner.argmax(0)
    mutual = best1[best0] == np.arange(len(best0))
    scores = np.where(mutual, np.exp(inner.max(1)), 0.0)
    valid = mutual & (scores > threshold)
    return np.where(valid, best0, -1), np.where(valid, scores, 0.0)


def sinkhorn_matches(couplings: np.ndarray, iterations: int, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """extract_matches(log_sinkhorn(couplings, iterations), threshold), computed in float32 without the log matrix.

    The same iterations in the linear domain: one exponential, shifted by the maximum to stay finite, then
    matrix-vector products, at the float32 precision SuperGlue's PyTorch model runs them in. log is increasing, so
    each row's best column is the argmax of kernel * v and each column's best row the argmax of kernel * u; the score
    is exp(log transport) = kernel * u * v * (M + N). A row or column that underflows falls back to log space.
    """
    rows, columns = couplings.shape[-2] - 1, couplings.shape[-1] - 1
    scale = rows + columns
    mu = np.full(rows + 1, 1.0 / scale, dtype=np.float32)
    mu[-1] = columns / scale
    nu = np.full(columns + 1, 1.0 / scale, dtype=np.float32)
    nu[-1] = rows / scale
    cost = np.asarray(couplings, dtype=np.float32).reshape(rows + 1, columns + 1)
    kernel = cv2.exp(cost - cost.max())
    kernel_t = np.ascontiguousarray(kernel.T)
    u, v = np.ones(rows + 1, dtype=np.float32), np.ones(columns + 1, dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(iterations):
            u = mu / (kernel @ v)
            v = nu / (kernel_t @ u)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(u > 0) and np.all(v > 0)):
        return extract_matches(log_sinkhorn(couplings, iterations), threshold)
    by_row = kernel[:-1, :-1] * v[None, :-1]
    best0 = by_row.argmax(1)
    best1 = (kernel_t[:-1, :-1] * u[None, :-1]).argmax(1)
    mutual = best1[best0] == np.arange(rows)
    scores = np.where(mutual, by_row[np.arange(rows), best0] * u[:-1] * scale, 0.0)
    valid = mutual & (scores > threshold)
    return np.where(valid, best0, -1), np.where(valid, scores, 0.0)


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


def run_mpk(runner, tensors: list) -> np.ndarray:
    """The MPK's [N+1, N+1] couplings for the pair."""
    (output,) = runner.run(tensors, timeout_ms=TIMEOUT_MS)
    return output.to_numpy(copy=False).reshape(KEYPOINTS + 1, KEYPOINTS + 1)


def match_pair(runner, tensors: list) -> np.ndarray:
    """The MPK's couplings, then the CPU's Sinkhorn iterations and mutual-match extraction: the pair's matches."""
    matches, _ = sinkhorn_matches(run_mpk(runner, tensors), SINKHORN_ITERATIONS, MATCH_THRESHOLD)
    return matches


def measure_latency(model: pyneat.Model, tensors: list) -> LatencyResult:
    """Each pair one at a time after the warm-up, timing the whole pair and its MPK run."""
    runner = model.build(tensors)
    try:
        for _ in range(WARMUP_PAIRS):
            match_pair(runner, tensors)
        pair_s: list[float] = []
        model_s: list[float] = []
        started = time.perf_counter()
        while len(pair_s) < MEASURED_PAIRS or time.perf_counter() - started < MIN_DURATION_S:
            begun = time.perf_counter()
            couplings = run_mpk(runner, tensors)
            model_s.append(time.perf_counter() - begun)
            sinkhorn_matches(couplings, SINKHORN_ITERATIONS, MATCH_THRESHOLD)
            pair_s.append(time.perf_counter() - begun)
    finally:
        runner.close()
    return LatencyResult(pair_s, model_s)


def _start_worker(mpk: str, arrays: list[np.ndarray], threads: int, started, ready) -> None:
    """Pool initializer of a throughput worker: its own model and runner, built one worker at a time."""
    global _worker_runner, _worker_tensors
    # The parent's runner set this for Neat's GStreamer runtime; Neat refuses to start with it inherited.
    os.environ.pop("GST_PLUGIN_SYSTEM_PATH_1_0", None)
    cv2.setNumThreads(threads)
    with started.get_lock():
        order = started.value
        started.value += 1
    time.sleep(order * NEAT_START_STAGGER_S)
    _worker_tensors = [pyneat.Tensor.from_numpy(value, copy=True) for value in arrays]
    _worker_runner = pyneat.Model(mpk, tensor_options()).build(_worker_tensors)
    match_pair(_worker_runner, _worker_tensors)
    with ready.get_lock():
        ready.value += 1


def _match_once(_: int) -> int:
    return int((match_pair(_worker_runner, _worker_tensors) >= 0).sum())


def measure_throughput(mpk: str, arrays: list[np.ndarray]) -> ThroughputResult:
    """Pairs matched in NEAT_WORKERS processes, each keeping at most two pairs queued. The measured span starts after
    WARMUP_PAIRS pairs and lasts at least MEASURED_PAIRS pairs and MIN_DURATION_S seconds."""
    # OpenBLAS, OpenMP and OpenCV start one thread per core in every process by default; each worker takes its share
    # instead. The spawned workers import numpy after this, so their thread pools read it.
    threads = max(1, (os.cpu_count() or 1) // NEAT_WORKERS)
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(threads)
    monitor = pyneat.TelemetryMonitor()
    context = multiprocessing.get_context("spawn")
    ready = context.Value("i", 0)
    pairs, started = 0, 0.0
    initargs = (mpk, arrays, threads, context.Value("i", 0), ready)
    with context.Pool(NEAT_WORKERS, _start_worker, initargs) as pool:
        # Stream pairs only once every worker has built its runner.
        deadline = time.monotonic() + 120 + NEAT_WORKERS * NEAT_START_STAGGER_S
        while ready.value < NEAT_WORKERS:
            if time.monotonic() > deadline:
                raise RuntimeError(f"only {ready.value} of {NEAT_WORKERS} workers started")
            time.sleep(0.5)
        ahead = collections.deque()
        submitted = 0
        while True:
            ahead.append(pool.apply_async(_match_once, (submitted,)))
            submitted += 1
            if len(ahead) <= 2 * NEAT_WORKERS:
                continue
            ahead.popleft().get()
            pairs += 1
            if pairs == WARMUP_PAIRS:
                monitor.start()
                started = time.perf_counter()
            elif pairs > WARMUP_PAIRS:
                measured, seconds = pairs - WARMUP_PAIRS, time.perf_counter() - started
                if measured >= MEASURED_PAIRS and seconds >= MIN_DURATION_S:
                    pool.terminate()
                    return ThroughputResult(measured, seconds, monitor.stop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    workload, images = dataset_pair(Path(dataset_dir))

    layer_stats, stage_stats = mla_stats(args.mpk)
    arrays = pair_tensors(images)
    model = pyneat.Model(args.mpk, tensor_options())
    latency = measure_latency(model, [pyneat.Tensor.from_numpy(value, copy=True) for value in arrays])
    # The throughput workers need every accelerator client Neat allows; reload the model for the report after.
    del model
    gc.collect()
    throughput = measure_throughput(args.mpk, arrays)
    model = pyneat.Model(args.mpk, tensor_options())
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
            f"{workload}, {KEYPOINTS} keypoints each, on the tensor route, then {SINKHORN_ITERATIONS} Sinkhorn "
            f"iterations and mutual-match extraction on the CPU; latency one pair at a time, throughput with "
            f"{NEAT_WORKERS} processes each running the MPK and the CPU matching"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats={"model": layer_stats},
        mla_stage_stats={"model": stage_stats},
    )
    report.save(args.output)
    model_ms = float(np.mean(latency.model_s) * 1000)
    print(
        f"{throughput.fps:.1f} pairs/s over {throughput.pairs} pairs; latency {benchmark.latency_ms:.2f} ms per pair: "
        f"MPK {model_ms:.2f} ms, CPU matching {benchmark.latency_ms - model_ms:.2f} ms"
    )


if __name__ == "__main__":
    main()
