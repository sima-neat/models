"""Benchmark RF-DETR Medium with Neat Core and save its report.

Given --mpk backbone=<path> --mpk decoder=<path>, the script measures the whole model as it runs: the
backbone, the top 300 proposals on the CPU, and the decoder, each stage working on a different frame at the
same time. It also records each MPK's MLA layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import model_report
import numpy as np
import pyneat

IMAGE_SIZE = 576
QUERY_COUNT = 300
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The input is already a resized, normalized image; Neat still quantizes and tessellates it.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def select_top_proposals(scores: pyneat.Tensor, boxes: pyneat.Tensor) -> pyneat.Tensor:
    """calibration.select_top_proposals, without the Pillow import this stage cannot load."""
    scores, boxes = scores.to_numpy(), boxes.to_numpy()
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("proposal scores and boxes must be finite")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :QUERY_COUNT]
    proposals = np.take_along_axis(boxes, order[..., None], axis=1)
    # The decoder may read the proposals after the backbone outputs are released, so they are copied.
    return pyneat.Tensor.from_numpy(proposals, copy=True, memory=pyneat.TensorMemory.EV74)


def percentile_ms(seconds: list[float], percentile: float) -> float:
    return float(np.percentile(seconds, percentile) * 1000)


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for one MPK, in the shape Core saves them. Their throughput is not the
    model's, so Core's default run serves."""
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


def synthetic_image() -> pyneat.Tensor:
    """One random image, placed once in the memory the backbone reads, reused for every frame."""
    pixels = np.random.default_rng(0).standard_normal((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    return pyneat.Tensor.from_numpy(pixels, copy=True, memory=pyneat.TensorMemory.EV74)


def build_runners(models: dict[str, pyneat.Model], image: pyneat.Tensor) -> tuple[pyneat.ModelRunner, ...]:
    """A runner for each model. The decoder's is built from one real backbone output."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    backbone = models["backbone"].build([image], run_options=run_options)
    features, scores, boxes = backbone.run([image], timeout_ms=TIMEOUT_MS)
    decoder = models["decoder"].build([features, select_top_proposals(scores, boxes)], run_options=run_options)
    return backbone, decoder


def infer_one(backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor) -> None:
    features, scores, boxes = backbone.run([image], timeout_ms=TIMEOUT_MS)
    decoder.run([features, select_top_proposals(scores, boxes)], timeout_ms=TIMEOUT_MS)


def measure_latency(backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor) -> list[float]:
    """Seconds per frame through all three stages with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(backbone, decoder, image)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(backbone, decoder, image)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the decoder output, how long they took, and the
    board telemetry sampled over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"the {name} rejected a frame")


def drain(runner: pyneat.ModelRunner, name: str, pushed: dict[str, int]) -> Iterator[list[pyneat.Tensor]]:
    """Each output until the runner's end of stream. Fewer outputs than frames pushed means a stall.

    Core's pull returns no tensors both at end of stream and on a timeout, so the count tells them apart.
    """
    pulled = 0
    while tensors := runner.pull(TIMEOUT_MS).tensors:
        pulled += 1
        yield tensors
        # Return the outputs to the runner before waiting for the next frame.
        del tensors
    if pulled != pushed[name]:
        raise RuntimeError(f"the {name} stalled: {pulled} of {pushed[name]} frames came out within {TIMEOUT_MS} ms")


def run_together(*stages: Callable[[], None], runners: tuple[pyneat.ModelRunner, ...]) -> None:
    """Run each stage on its own thread. The first failure ends every runner's input, which unblocks a stage
    waiting to push, and is raised once all stages stop. Each stage waits in Core with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            for runner in runners:
                runner.close_input()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def measure_throughput(backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"backbone": 0, "decoder": 0}

    def push_inputs() -> None:
        try:
            while not measured.is_set():
                push(backbone, [image], "backbone")
                pushed["backbone"] += 1
        finally:
            # End of stream follows the last frame through the backbone, then the decoder.
            backbone.close_input()

    def forward_proposals() -> None:
        try:
            for features, scores, boxes in drain(backbone, "backbone", pushed):
                proposals = select_top_proposals(scores, boxes)
                push(decoder, [features, proposals], "decoder")
                pushed["decoder"] += 1
                # The decoder holds its inputs now, so the backbone gets its output buffers back.
                del features, scores, boxes, proposals
        finally:
            decoder.close_input()

    def pull_outputs() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(decoder, "decoder", pushed):
            # Return the decoder's output buffers before the next pull.
            del tensors
            outputs += 1
            if outputs == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_FRAMES and not measured.is_set():
                frames, seconds = outputs - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(frames, seconds, monitor.stop())
                    measured.set()
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    try:
        run_together(push_inputs, forward_proposals, pull_outputs, runners=(backbone, decoder))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        backbone.close()
        decoder.close()
    return throughput


def write_report(
    output: Path,
    latencies: list[float],
    throughput: Throughput,
    models: dict[str, pyneat.Model],
    mla_layer_stats: dict[str, dict[str, object]],
    mla_stage_stats: dict[str, dict[str, object]],
) -> None:
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = percentile_ms(latencies, 50)
    benchmark.latency_p95_ms = percentile_ms(latencies, 95)
    benchmark.latency_p99_ms = percentile_ms(latencies, 99)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"Backbone, top {QUERY_COUNT} proposals on the CPU, and decoder, pipelined {QUEUE_DEPTH} frames deep "
            f"on a {IMAGE_SIZE}x{IMAGE_SIZE} FLOAT32 tensor"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.1f} FPS over {throughput.frames} frames in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms:.2f} ms over {len(latencies)} frames"
    )


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpk), tensor_options()) for name, mpk in mpks.items()}
    image = synthetic_image()
    backbone, decoder = build_runners(models, image)
    latencies = measure_latency(backbone, decoder, image)
    throughput = measure_throughput(backbone, decoder, image)
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"backbone", "decoder"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and decoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
