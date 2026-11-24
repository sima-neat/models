"""Benchmark D-FINE Nano with Neat Core and save its report.

Given --mpk backbone=<path> --mpk decoder=<path> and --dataset coco-2017-val=<extracted directory>, the
script measures the whole model as it runs on one COCO image in a 720p frame: Neat's preprocessing and the
backbone, the top 300 queries on the CPU, and the decoder, each stage working on a different frame at the
same time. It also records each MPK's MLA layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import model_report
import numpy as np
import pyneat
from model_inputs.frame import FRAME_HEIGHT, FRAME_WIDTH, letterbox_to_720p, read_rgb

IMAGE_SIZE = 640
QUERY_COUNT = 300
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"


def image_options() -> pyneat.ModelOptions:
    """The backbone's: Neat stretches, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    # Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # D-FINE expects RGB scaled to [0, 1], with no ImageNet mean or std.
    options.preprocess.normalize.mean = [0.0, 0.0, 0.0]
    options.preprocess.normalize.stddev = [1.0, 1.0, 1.0]
    options.preprocess.normalize.has_explicit_stats = True
    # Keep the attention memory the decoder reads where its EV74 input stage can reach it.
    options.advanced_execution.postprocess_target = "EV74"
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The decoder's: its inputs are the backbone's features and the selected queries, not an image."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def select_queries(content: pyneat.Tensor, scores: pyneat.Tensor, references: pyneat.Tensor) -> list[pyneat.Tensor]:
    """Own the two gathered arrays; attention memory is forwarded unchanged."""
    preprocess_meta = content.semantic.preprocess
    content, scores, references = (x.to_numpy(copy=False) for x in (content, scores, references))
    if content.shape != (1, 2000, 128) or scores.shape != (1, 2000) or references.shape != (1, 2000, 4):
        raise ValueError("Unexpected Nano query boundary")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (content, scores, references)):
        raise ValueError("Expected finite float32 query boundary")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :QUERY_COUNT]
    gathered = [
        pyneat.Tensor.from_numpy(
            x[0, order[0], :].reshape(1, QUERY_COUNT, x.shape[-1]),
            copy=True,
            layout=pyneat.TensorLayout.Unknown,
            memory=pyneat.TensorMemory.EV74,
        )
        for x in (content, references)
    ]
    for tensor in gathered:
        tensor.semantic.preprocess = preprocess_meta
    return gathered


def require_heads(outputs: list[pyneat.Tensor]) -> None:
    """Count a frame only once the decoder published both of its public heads."""
    if len(outputs) != 2:
        raise RuntimeError(f"Expected the decoder's pred_boxes and pred_logits, got {len(outputs)} tensors")


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


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the backbone reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def build_runners(models: dict[str, pyneat.Model], image: pyneat.Tensor) -> tuple[pyneat.ModelRunner, ...]:
    """A runner for each model. The decoder's is built from one real backbone output."""
    options = pyneat.RunOptions()
    options.queue_depth = QUEUE_DEPTH
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    options.overflow_policy = pyneat.OverflowPolicy.Block
    backbone = models["backbone"].build([image], run_options=options)
    try:
        memory, content, scores, references = backbone.run([image], timeout_ms=TIMEOUT_MS)
        decoder = models["decoder"].build([memory, *select_queries(content, scores, references)], run_options=options)
        return backbone, decoder
    except BaseException:
        backbone.close()
        raise


def infer_one(backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor) -> None:
    memory, content, scores, references = backbone.run([image], timeout_ms=TIMEOUT_MS)
    require_heads(decoder.run([memory, *select_queries(content, scores, references)], timeout_ms=TIMEOUT_MS))


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
    while True:
        sample = runner.pull(TIMEOUT_MS)
        if sample is None or not sample.tensors:
            break
        tensors = sample.tensors
        pulled += 1
        yield tensors
        # Release both references before waiting for the next frame.
        del tensors, sample
    if pulled != pushed[name]:
        raise RuntimeError(f"the {name} stalled: {pulled} of {pushed[name]} frames came out within {TIMEOUT_MS} ms")


def run_together(*stages: Callable[[], None], runners: tuple[pyneat.ModelRunner, ...]) -> None:
    """Run each stage on its own thread. The first failure ends every runner's input, which unblocks a stage
    waiting to push, and is raised once all stages stop. Each stage waits in Core with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:  # noqa: BLE001 - release every stage, then rethrow.
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

    def forward_queries() -> None:
        try:
            for memory, content, scores, references in drain(backbone, "backbone", pushed):
                selected = select_queries(content, scores, references)
                push(decoder, [memory, *selected], "decoder")
                pushed["decoder"] += 1
                # The decoder holds its inputs now, so the backbone gets its output buffers back.
                del memory, content, scores, references, selected
        finally:
            decoder.close_input()

    def pull_outputs() -> None:
        nonlocal throughput
        started = 0.0
        for outputs, tensors in enumerate(drain(decoder, "decoder", pushed), start=1):
            require_heads(tensors)
            # Return the decoder's output buffers before the next pull.
            del tensors
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
        run_together(push_inputs, forward_queries, pull_outputs, runners=(backbone, decoder))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
    return throughput


def write_report(
    output: Path,
    image: str,
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
            f"{image} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing stretched to "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}, backbone, top {QUERY_COUNT} queries on the CPU, and decoder, pipelined "
            f"{QUEUE_DEPTH} frames deep"
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


def benchmark_pipeline(mpks: dict[str, Path], dataset_id: str, dataset: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        "backbone": pyneat.Model(str(mpks["backbone"]), image_options()),
        "decoder": pyneat.Model(str(mpks["decoder"]), tensor_options()),
    }
    path = sample_image(dataset)
    image = frame_tensor(path)
    backbone, decoder = build_runners(models, image)
    try:
        latencies = measure_latency(backbone, decoder, image)
        throughput = measure_throughput(backbone, decoder, image)
    finally:
        decoder.close()
        backbone.close()
    write_report(output, f"{dataset_id} {path.name}", latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"backbone", "decoder"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and decoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_pipeline(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
