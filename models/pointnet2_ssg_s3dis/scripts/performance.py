"""Benchmark PointNet++ SSG with Neat Core and save its report.

Given one --mpk <component>=<path> per component, in model.yaml order, the script measures the whole model on one
4096-point block: farthest point sampling, ball queries and 3-NN weights on the CPU (the block's plan), then the
eight MPKs with the CPU grouping and interpolation between them. Latency runs one block at a time. The CPU steps
cost far more time than the MPKs take, so the throughput pass streams blocks as an application with the DevKit's
16 cores would: PLAN_WORKERS processes compute the plans of upcoming blocks, and NEAT_WORKERS processes, each with
its own eight MPK runners, group, interpolate and run the MPKs. It also records each MPK's MLA layer and stage
stats. With --dataset s3dis-val=<extracted directory> the input is the first 1 m tile block of one
random.Random(0) room, built exactly as accuracy.py builds it; without it, pointcloud.synthetic_block(), a fixed
seeded block. Either way every run measures the same points.

    PYTHONPATH=tools python performance.py --mpk sa1=<archive> ... --mpk fp1=<archive> \
        [--dataset s3dis-val=<extracted directory>] --output <directory>
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from pointcloud import COMPONENTS, DATASET, POINTS, SEED, blocks, planned, segment, synthetic_block

WARMUP_BLOCKS = 10
MEASURED_BLOCKS = 200
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Throughput-pass processes: block plans, and segment() with the MPKs, which together keep the DevKit's 16 cores
# busy. Each Neat worker holds eight one-buffer runners; four workers with Neat's default four buffers ran out of
# the DevKit's contiguous (CMA) memory. On a DevKit, 3 + 12 reach 24.3 blocks/s; 4 + 12, 23.8; 2 + 13, 22.3.
PLAN_WORKERS = 12
NEAT_WORKERS = 3
# Thread counts of OpenMP, OpenBLAS and OpenCV, one per worker process.
WORKER_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENCV_FOR_THREADS_NUM")
# Seconds between the Neat workers' starts: runners built at the same moment can find the accelerator busy.
NEAT_START_STAGGER_S = 2.0


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: its input is grouped or interpolated point features, not an image."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for one MPK. Their throughput is not the model's, so Core's default run
    serves."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    report = pyneat.Model(str(mpk)).benchmark(options)
    layer_stats, stage_stats = report.mla_layer_stats, report.mla_stage_stats
    layers = [
        {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
        for layer in layer_stats.layers
    ]
    stages = [{"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stage_stats.stages]
    return {"status": layer_stats.status, "layers": layers}, {"status": stage_stats.status, "stages": stages}


def run_options() -> pyneat.RunOptions:
    """Each runner runs one block at a time, so one buffer of each kind suffices; the default four per runner, times
    eight runners per Neat worker, is what exhausted the DevKit's contiguous (CMA) memory."""
    options = pyneat.RunOptions()
    options.queue_depth = 1
    options.advanced.sync_num_buffers_override = 1
    return options


def pipeline(models: dict[str, pyneat.Model]):
    """run(component, input) for pointcloud.segment(..., channels_last=True): each MPK takes its input as the one
    [H, W, C] tensor segment() builds."""
    runners = {}

    def run(component: str, value: np.ndarray) -> np.ndarray:
        tensor = pyneat.Tensor.from_numpy(value, copy=False)
        if component not in runners:
            runners[component] = models[component].build([tensor], run_options=run_options())
        (output,) = runners[component].run([tensor], timeout_ms=TIMEOUT_MS)
        return output.to_numpy(copy=True)

    return run, runners


def measure_latency(run, block: np.ndarray) -> list[float]:
    """Seconds per block, each block's plan and MPKs run in turn with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_BLOCKS):
        segment(block, run, channels_last=True)
    seconds: list[float] = []
    started = time.perf_counter()
    while len(seconds) < MEASURED_BLOCKS or time.perf_counter() - started < MIN_DURATION_S:
        begun = time.perf_counter()
        logits = segment(block, run, channels_last=True)
        seconds.append(time.perf_counter() - begun)
    if not np.isfinite(logits).all():
        raise RuntimeError("the pipeline produced non-finite logits")
    return seconds


def _start_neat_worker(mpks: dict[str, Path], block: np.ndarray, started, ready) -> None:
    """Pool initializer of a Neat worker: its own models and runners, built one worker at a time."""
    global _worker_block, _worker_run
    # The parent's runners set this for Neat's GStreamer runtime; Neat refuses to start with it inherited.
    os.environ.pop("GST_PLUGIN_SYSTEM_PATH_1_0", None)
    with started.get_lock():
        order = started.value
        started.value += 1
    time.sleep(order * NEAT_START_STAGGER_S)
    _worker_block = block
    _worker_run, _ = pipeline({name: pyneat.Model(str(mpk), tensor_options()) for name, mpk in mpks.items()})
    segment(block, _worker_run, channels_last=True)
    with ready.get_lock():
        ready.value += 1


def _segment_planned(indices: dict) -> bool:
    return bool(np.isfinite(segment(_worker_block, _worker_run, indices, channels_last=True)).all())


def measure_throughput(mpks: dict[str, Path], block: np.ndarray) -> tuple[int, float, pyneat.Telemetry]:
    """Blocks, seconds and board telemetry with blocks streamed through PLAN_WORKERS plan processes and NEAT_WORKERS
    MPK processes, each keeping at most twice its count of blocks queued. The measured span starts after
    WARMUP_BLOCKS blocks and lasts at least MEASURED_BLOCKS blocks and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    context = multiprocessing.get_context("spawn")
    # Every worker is one of many processes on 16 cores; library threads would only contend. Spawned workers read
    # these when they import NumPy and OpenCV, so the latency pass keeps its threads.
    for name in WORKER_THREAD_VARIABLES:
        os.environ[name] = "1"
    blocks, started = 0, 0.0
    ready = context.Value("i", 0)
    with context.Pool(NEAT_WORKERS, _start_neat_worker, (mpks, block, context.Value("i", 0), ready)) as neat:
        # Stream blocks only once every worker has built its runners.
        deadline = time.monotonic() + 120 + NEAT_WORKERS * NEAT_START_STAGGER_S
        while ready.value < NEAT_WORKERS:
            if time.monotonic() > deadline:
                raise RuntimeError(f"only {ready.value} of {NEAT_WORKERS} Neat workers started")
            time.sleep(0.5)
        ahead = collections.deque()
        for _, _, indices in planned(((block, None) for _ in itertools.count()), PLAN_WORKERS):
            ahead.append(neat.apply_async(_segment_planned, (indices,)))
            if len(ahead) <= 2 * NEAT_WORKERS:
                continue
            if not ahead.popleft().get():
                raise RuntimeError("the pipeline produced non-finite logits")
            blocks += 1
            if blocks == WARMUP_BLOCKS:
                monitor.start()
                started = time.perf_counter()
            elif blocks > WARMUP_BLOCKS:
                measured, seconds = blocks - WARMUP_BLOCKS, time.perf_counter() - started
                if measured >= MEASURED_BLOCKS and seconds >= MIN_DURATION_S:
                    neat.terminate()
                    return measured, seconds, monitor.stop()
    raise RuntimeError("the block stream stopped before the measured span ended")


def dataset_block(directory: Path) -> tuple[str, np.ndarray]:
    """The first tile block of one random.Random(0) room of s3dis val, as accuracy.py's tile protocol builds it."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    room = random.Random(0).choice(manifest["rooms"])
    points = {key: np.load(directory / "val" / room / f"{key}.npy", allow_pickle=False) for key in ("xyz", "rgb")}
    features, _ = next(blocks(points, np.random.RandomState(SEED), "tiles"))
    return f"{DATASET} {room}, its first 1 m tile block", features


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or list(mpks) != list(COMPONENTS):
        raise SystemExit(f"--mpk must name {', '.join(COMPONENTS)} in that order")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args()
    mpks = parse_mpks(args.mpk)
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpk), tensor_options()) for name, mpk in mpks.items()}
    if args.dataset:
        dataset_id, directory = args.dataset.split("=", 1)
        if dataset_id != DATASET:
            raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
        source, block = dataset_block(Path(directory))
    else:
        source, block = f"a seeded synthetic {POINTS}-point block", synthetic_block()
    run, runners = pipeline(models)
    try:
        seconds = measure_latency(run, block)
    finally:
        # The throughput pass's workers need every runner Neat Core allows.
        for runner in runners.values():
            runner.close()
    frames, span, telemetry = measure_throughput(mpks, block)

    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(seconds) * 1000)
    benchmark.latency_p50_ms = float(np.percentile(seconds, 50) * 1000)
    benchmark.latency_p95_ms = float(np.percentile(seconds, 95) * 1000)
    benchmark.latency_p99_ms = float(np.percentile(seconds, 99) * 1000)
    benchmark.fps = frames / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"{source}: CPU sampling, ball query and 3-NN weights, then the eight "
            f"MPKs with CPU grouping and interpolation between them; throughput with {PLAN_WORKERS} processes "
            f"computing upcoming blocks' sampling, ball queries and 3-NN and {NEAT_WORKERS} processes running "
            "the grouping, interpolation and MPKs"
        ),
        telemetry=telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(Path(args.output))
    print(f"{benchmark.fps:.2f} blocks/s over {frames} blocks; latency {benchmark.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
