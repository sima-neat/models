"""Benchmark FresUNet with Neat Core on one OSCD test city, including its CPU decode, and save its report.

FresUNet takes a 26-channel multispectral tensor, not an image. Given --dataset oscd-default=<extracted directory>,
the benchmark repeats one random.Random(0) city of test.txt, prepared once as accuracy.py prepares it (oscd.py: both
dates' 13 bands normalized, resized to 544x544 and stacked) and passed on the tensor route. Each measured sample runs
the MPK and then the CPU step that turns its two-class output into the change map, the argmax accuracy.py scores.
Latency runs one sample at a time and reports the MPK's and the CPU's share. The throughput pass runs the runner
asynchronously, as SuperGlue's does: one thread pushes the sample, this one pulls each output, and DECODE_WORKERS
threads decode, so the MPK works on one sample while the CPU decodes earlier ones. Core's own benchmark of the MPK
supplies only the MLA layer and stage stats.
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
from oscd import cities, city_input, dataset_root, prediction

DATASET = "oscd-default"
SPLIT = "test"
WARMUP_SAMPLES = 10
MEASURED_SAMPLES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Samples the runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
# Threads decoding outputs on the CPU in the throughput pass, one per core.
DECODE_WORKERS = os.cpu_count() or 1


@dataclass(frozen=True)
class LatencyResult:
    """Seconds per sample, one sample at a time, and the MPK's part of each."""

    sample_s: list[float]
    model_s: list[float]


@dataclass(frozen=True)
class ThroughputResult:
    samples: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.samples / self.seconds


def dataset_sample(dataset: Path) -> tuple[str, np.ndarray]:
    """The workload's name and the [544, 544, 26] input of one random.Random(0) test city."""
    root = dataset_root(dataset)
    city = random.Random(0).choice(cities(root, SPLIT))
    return f"{DATASET} {SPLIT} city {city}", city_input(root, city)


def tensor_options() -> pyneat.ModelOptions:
    """The input is a prepared 26-channel tensor: Neat only quantizes and tessellates it."""
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
    """The MPK's two-class output for the sample."""
    (output,) = runner.run(tensors, timeout_ms=TIMEOUT_MS)
    return output.to_numpy(copy=False)


def build_runner(model: pyneat.Model, tensors: list) -> pyneat.ModelRunner:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    return model.build(tensors, run_options=run_options)


def measure_latency(runner: pyneat.ModelRunner, tensors: list) -> LatencyResult:
    """Each sample one at a time after the warm-up, timing the whole sample and its MPK run."""
    for _ in range(WARMUP_SAMPLES):
        prediction(run_mpk(runner, tensors))
    sample_s: list[float] = []
    model_s: list[float] = []
    started = time.perf_counter()
    while len(sample_s) < MEASURED_SAMPLES or time.perf_counter() - started < MIN_DURATION_S:
        begun = time.perf_counter()
        output = run_mpk(runner, tensors)
        model_s.append(time.perf_counter() - begun)
        prediction(output)
        sample_s.append(time.perf_counter() - begun)
    return LatencyResult(sample_s, model_s)


def measure_throughput(runner: pyneat.ModelRunner, tensors: list) -> ThroughputResult:
    """Decoded samples per second and board telemetry with the MPK and the CPU busy at once. One thread pushes the
    sample, this one pulls each output and hands it to DECODE_WORKERS decoding threads. The measured span starts after
    WARMUP_SAMPLES decoded samples and lasts at least MEASURED_SAMPLES samples and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()

    def push_samples() -> None:
        try:
            while not measured.is_set():
                if not runner.push(tensors) and not measured.is_set():
                    raise RuntimeError("the runner rejected a sample")
        finally:
            runner.close_input()

    pending: collections.deque[Future] = collections.deque()
    decoded, started = 0, 0.0
    result: ThroughputResult | None = None
    with ThreadPoolExecutor(DECODE_WORKERS + 1) as pool:
        pusher = pool.submit(push_samples)
        try:
            while outputs := runner.pull(TIMEOUT_MS).tensors:
                output = outputs[0].to_numpy(copy=True)
                # Return the output buffer to the runner before the next pull.
                del outputs
                pending.append(pool.submit(prediction, output))
                # Count each sample once 2 * DECODE_WORKERS later samples queue behind it, so the workers stay busy.
                if len(pending) <= 2 * DECODE_WORKERS:
                    continue
                pending.popleft().result()
                decoded += 1
                if decoded == WARMUP_SAMPLES:
                    monitor.start()
                    started = time.perf_counter()
                elif decoded > WARMUP_SAMPLES and result is None:
                    samples, seconds = decoded - WARMUP_SAMPLES, time.perf_counter() - started
                    if samples >= MEASURED_SAMPLES and seconds >= MIN_DURATION_S:
                        result = ThroughputResult(samples, seconds, monitor.stop())
                        measured.set()
        finally:
            measured.set()
            # Wakes a push waiting on a full queue, so the pusher ends even when a pull failed.
            runner.close_input()
            # A failure inside the measured span leaves the monitor running.
            if monitor.running():
                monitor.stop()
        pusher.result()
        for decode in pending:
            decode.result()
    if result is None:
        raise RuntimeError(f"no sample came out within {TIMEOUT_MS} ms before the measured span ended")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    workload, sample = dataset_sample(Path(dataset_dir))

    layer_stats, stage_stats = mla_stats(args.mpk)
    tensors = [pyneat.Tensor.from_numpy(sample, copy=True, layout=pyneat.TensorLayout.HWC)]
    model = pyneat.Model(args.mpk, tensor_options())
    runner = build_runner(model, tensors)
    try:
        latency = measure_latency(runner, tensors)
        throughput = measure_throughput(runner, tensors)
    finally:
        runner.close()
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latency.sample_s) * 1000)
    benchmark.latency_p50_ms = float(np.percentile(latency.sample_s, 50) * 1000)
    benchmark.latency_p95_ms = float(np.percentile(latency.sample_s, 95) * 1000)
    benchmark.latency_p99_ms = float(np.percentile(latency.sample_s, 99) * 1000)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.samples
    report = model_report.from_performance(
        benchmark,
        {"model": model},
        workload=(
            f"{workload}, a 544x544x26 tensor on the tensor route, then the change-map argmax on the CPU; latency one "
            f"sample at a time, throughput with the runner's async push and pull and {DECODE_WORKERS} CPU decoding "
            f"threads"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats={"model": layer_stats},
        mla_stage_stats={"model": stage_stats},
    )
    report.save(args.output)
    model_ms = float(np.mean(latency.model_s) * 1000)
    print(
        f"{throughput.fps:.1f} samples/s over {throughput.samples} samples; latency {benchmark.latency_ms:.2f} ms per "
        f"sample: MPK {model_ms:.2f} ms, CPU decode {benchmark.latency_ms - model_ms:.2f} ms"
    )


if __name__ == "__main__":
    main()
