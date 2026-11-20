"""Benchmark V-JEPA 2.1 ViT-B 16x256 with Neat Core and save its report.

Given one --mpk <component>=<path> per component (encoder_0, encoder_1, encoder_2), the script measures the whole
encoder on one fixed clip, each component working on a different clip at the same time. The model declares no
accuracy dataset, so the clip is built here: 16 frames of a seeded 720p camera-like scene (smooth colour gradients
with sensor-like noise) panning 8 pixels a frame. Each frame is resized, centre-cropped and normalized on the CPU
once, and the frames are stacked into the encoder's tubelet layout (pipeline.py); Neat quantizes and tessellates the
clip inside the measured pipeline. The script also records each MPK's MLA layer and stage stats and its execution
plan.

    PYTHONPATH=tools python performance.py --mpk encoder_0=<archive> --mpk encoder_1=<archive> \
        --mpk encoder_2=<archive> --output <directory>
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
from pipeline import COMPONENTS, FRAME_SIZE, FRAMES, preprocess_frame, stack_clip

WARMUP_CLIPS = 20
MEASURED_CLIPS = 300
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Clips each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
FRAME_WIDTH, FRAME_HEIGHT, SEED = 1280, 720, 0
PAN_PIXELS = 8


def synthetic_frame() -> np.ndarray:
    rows = np.linspace(0, 1, FRAME_HEIGHT, dtype=np.float32)[:, None]
    columns = np.linspace(0, 1, FRAME_WIDTH, dtype=np.float32)[None, :]
    channels = [rows * 200 + 30, columns * 180 + 40, (1 - rows) * columns * 220 + 20]
    base = np.stack([np.broadcast_to(channel, (FRAME_HEIGHT, FRAME_WIDTH)) for channel in channels], axis=-1)
    noise = np.random.default_rng(SEED).normal(0, 8, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def synthetic_clip() -> np.ndarray:
    """The encoder's NCHW input for FRAMES frames of the scene, panning PAN_PIXELS a frame."""
    scene = synthetic_frame()
    frames = [preprocess_frame(np.roll(scene, -index * PAN_PIXELS, axis=1)) for index in range(FRAMES)]
    return stack_clip(np.stack(frames))


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: the script prepares the clip; Neat still quantizes and tessellates it."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def hwc(value: np.ndarray) -> pyneat.Tensor:
    """An NCHW batch of one as the HWC tensor Neat takes."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(value[0].transpose(1, 2, 0)), copy=True, layout=pyneat.TensorLayout.HWC
    )


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """One component's outputs as the next component's inputs: fresh CPU [H, W, C] tensors holding a copy, the form
    hwc() gives the first input. Neat Core builds a runner's input stage from the tensors it receives; an output
    passed on as is, or with Neat 0.6's leading batch axis, has media caps the next model's input stage rejects
    ([misconfig.media_caps]). The layout is stated: Neat reads an unlabelled 3D array as CHW."""
    arrays = [tensor.to_numpy(copy=True) for tensor in tensors]
    return [
        pyneat.Tensor.from_numpy(
            np.ascontiguousarray(array[0] if array.ndim == 4 else array), copy=True, layout=pyneat.TensorLayout.HWC
        )
        for array in arrays
    ]


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


def build_runners(models: dict[str, pyneat.Model], clip: pyneat.Tensor) -> list[pyneat.ModelRunner]:
    """A runner for each component, in COMPONENTS order, each built from one real output of the one before."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runners, tensors = [], [clip]
    for component in COMPONENTS:
        try:
            runners.append(models[component].build(tensors, run_options=run_options))
        except pyneat.NeatError as error:
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise RuntimeError(f"building the {component} runner on input shapes {shapes} failed") from error
        tensors = handoff(runners[-1].run(tensors, timeout_ms=TIMEOUT_MS))
    return runners


def infer_one(runners: list[pyneat.ModelRunner], clip: pyneat.Tensor) -> None:
    tensors = [clip]
    for runner in runners:
        tensors = handoff(runner.run(tensors, timeout_ms=TIMEOUT_MS))


def measure_latency(runners: list[pyneat.ModelRunner], clip: pyneat.Tensor) -> list[float]:
    """Seconds per clip through every component with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_CLIPS):
        infer_one(runners, clip)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_CLIPS or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(runners, clip)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the clips the last component output, how long they took, and the
    board telemetry sampled over the same span."""

    clips: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.clips / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"{name} rejected a clip")


def drain(runner: pyneat.ModelRunner, name: str, pushed: dict[str, int]) -> Iterator[list[pyneat.Tensor]]:
    """Each output until the runner's end of stream. Fewer outputs than clips pushed means a stall.

    Core's pull returns no tensors both at end of stream and on a timeout, so the count tells them apart.
    """
    pulled = 0
    while tensors := runner.pull(TIMEOUT_MS).tensors:
        pulled += 1
        yield tensors
        # Return the outputs to the runner before waiting for the next clip.
        del tensors
    if pulled != pushed[name]:
        raise RuntimeError(f"{name} stalled: {pulled} of {pushed[name]} clips came out within {TIMEOUT_MS} ms")


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


def measure_throughput(runners: list[pyneat.ModelRunner], clip: pyneat.Tensor) -> Throughput:
    """Clips per second and board telemetry with every component busy, each on a different clip. The measured span
    starts after WARMUP_CLIPS outputs and lasts at least MEASURED_CLIPS clips and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = dict.fromkeys(COMPONENTS, 0)

    def push_clips() -> None:
        try:
            while not measured.is_set():
                push(runners[0], [clip], COMPONENTS[0])
                pushed[COMPONENTS[0]] += 1
        finally:
            # End of stream follows the last clip through every component.
            runners[0].close_input()

    def forward(index: int) -> Callable[[], None]:
        source, target = COMPONENTS[index], COMPONENTS[index + 1]

        def forward_tokens() -> None:
            try:
                for tensors in drain(runners[index], source, pushed):
                    push(runners[index + 1], handoff(tensors), target)
                    pushed[target] += 1
                    # The next runner holds its inputs now, so this one gets its output buffers back.
                    del tensors
            finally:
                runners[index + 1].close_input()

        forward_tokens.__name__ = f"forward_{source}"
        return forward_tokens

    def pull_features() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(runners[-1], COMPONENTS[-1], pushed):
            # Return the last component's output buffers before the next pull.
            del tensors
            outputs += 1
            if outputs == WARMUP_CLIPS:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_CLIPS and not measured.is_set():
                clips, seconds = outputs - WARMUP_CLIPS, time.perf_counter() - started
                if clips >= MEASURED_CLIPS and seconds >= MIN_DURATION_S:
                    throughput = Throughput(clips, seconds, monitor.stop())
                    measured.set()
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    stages = [push_clips, *(forward(index) for index in range(len(runners) - 1)), pull_features]
    try:
        run_together(*stages, runners=tuple(runners))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        for runner in runners:
            runner.close()
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
    benchmark.throughput_samples = throughput.clips
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"A fixed seeded {FRAMES}-frame clip of a {FRAME_WIDTH}x{FRAME_HEIGHT} scene panning {PAN_PIXELS} pixels "
            f"a frame, each frame resized and centre-cropped to {FRAME_SIZE}x{FRAME_SIZE} and normalized on the CPU "
            f"once; the three encoder parts, pipelined {QUEUE_DEPTH} clips deep"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.1f} clips/s over {throughput.clips} clips in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms:.2f} ms over {len(latencies)} clips"
    )


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpks[name]), tensor_options()) for name in COMPONENTS}
    clip = hwc(synthetic_clip())
    runners = build_runners(models, clip)
    latencies = measure_latency(runners, clip)
    throughput = measure_throughput(runners, clip)
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
