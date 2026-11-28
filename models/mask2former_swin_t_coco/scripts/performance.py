"""Benchmark Mask2Former Swin-T with Neat Core and save its report.

Given one --mpk <component>=<path> per component (backbone_0, backbone_1, pixel_decoder_0 to pixel_decoder_12,
transformer_decoder) and --dataset coco-2017-val=<extracted directory>, the script measures the whole model as it
runs on one COCO image in a 720p frame: Neat's preprocessing and the sixteen components, each working on a different
frame at the same time, up to the class and mask logits. Each component input takes the earlier component output
of the same name; an output a later component needs (two of the backbone's feature maps) is copied out of the
runner's buffers and carried along with its frame. It also records each MPK's MLA
layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

PIXEL_LAYERS = 6
COMPONENTS = (
    "backbone_0",
    "backbone_1",
    *(f"pixel_decoder_{index}" for index in range(2 * PIXEL_LAYERS + 1)),
    "transformer_decoder",
)
# Each component's inputs and outputs by name, as in pipeline.py: each deformable layer runs as two components, the
# first returning its heads' share of the output projection (partial), the second completing the layer (tokens).
INPUTS = {
    "backbone_0": ("image",),
    "backbone_1": ("stage3_input",),
    "pixel_decoder_0": ("features1", "features2", "features3"),
    **{f"pixel_decoder_{2 * layer}": (f"tokens{layer - 1}",) for layer in range(1, PIXEL_LAYERS)},
    **{
        f"pixel_decoder_{2 * layer + 1}": ("embedded" if layer == 0 else f"tokens{layer - 1}", f"partial{layer}")
        for layer in range(PIXEL_LAYERS)
    },
    f"pixel_decoder_{2 * PIXEL_LAYERS}": (f"tokens{PIXEL_LAYERS - 1}", "features0"),
    "transformer_decoder": ("mask_features", f"tokens{PIXEL_LAYERS - 1}"),
}
OUTPUTS = {
    "backbone_0": ("features0", "features1", "stage3_input"),
    "backbone_1": ("features2", "features3"),
    "pixel_decoder_0": ("embedded", "partial0"),
    **{f"pixel_decoder_{2 * layer}": (f"partial{layer}",) for layer in range(1, PIXEL_LAYERS)},
    **{f"pixel_decoder_{2 * layer + 1}": (f"tokens{layer}",) for layer in range(PIXEL_LAYERS)},
    f"pixel_decoder_{2 * PIXEL_LAYERS}": ("mask_features",),
    "transformer_decoder": ("class_logits", "mask_logits"),
}
INPUTS = {component: INPUTS[component] for component in COMPONENTS}
OUTPUTS = {component: OUTPUTS[component] for component in COMPONENTS}
IMAGE_SIZE = 384
WARMUP_FRAMES = 20
MEASURED_FRAMES = 300
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


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
    # Without a preset Neat only scales pixels to [0, 1]; the model expects ImageNet mean and std.
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The decoders': their inputs are the previous component's outputs."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


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


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the backbone reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def needed_after(index: int) -> set[str]:
    """The tensor names a component after COMPONENTS[index] reads."""
    return {name for component in COMPONENTS[index + 1 :] for name in INPUTS[component]}


def carried(tensor: pyneat.Tensor) -> pyneat.Tensor:
    """A copy of an output a later component reads, so the runner can reuse its buffer meanwhile."""
    return pyneat.Tensor.from_numpy(tensor.to_numpy(copy=True), copy=True, memory=pyneat.TensorMemory.EV74)


def step(index: int, available: dict[str, pyneat.Tensor], outputs: list[pyneat.Tensor]) -> dict[str, pyneat.Tensor]:
    """What the components after COMPONENTS[index] read, given its outputs and what it was carried. The next
    component's inputs are passed as they are; the rest are copied."""
    available = {**available, **dict(zip(OUTPUTS[COMPONENTS[index]], outputs))}
    following = set(INPUTS[COMPONENTS[index + 1]]) if index + 1 < len(COMPONENTS) else set()
    return {
        name: tensor if name in following and name not in needed_after(index + 1) else carried(tensor)
        for name, tensor in available.items()
        if name in needed_after(index)
    }


def build_runners(models: dict[str, pyneat.Model], image: pyneat.Tensor) -> tuple[pyneat.ModelRunner, ...]:
    """A runner for each model, each built from one real input."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runners, available = [], {"image": image}
    for index, name in enumerate(COMPONENTS):
        inputs = [available[key] for key in INPUTS[name]]
        runner = models[name].build(inputs, run_options=run_options)
        runners.append(runner)
        available = step(index, available, runner.run(inputs, timeout_ms=TIMEOUT_MS))
    return tuple(runners)


def infer_one(runners: tuple[pyneat.ModelRunner, ...], image: pyneat.Tensor) -> None:
    available = {"image": image}
    for index, (name, runner) in enumerate(zip(COMPONENTS, runners)):
        available = step(index, available, runner.run([available[key] for key in INPUTS[name]], timeout_ms=TIMEOUT_MS))


def measure_latency(runners: tuple[pyneat.ModelRunner, ...], image: pyneat.Tensor) -> list[float]:
    """Seconds per frame through every component with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(runners, image)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(runners, image)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the transformer decoder output, how long they took,
    and the board telemetry sampled over the same span."""

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


def measure_throughput(runners: tuple[pyneat.ModelRunner, ...], image: pyneat.Tensor) -> Throughput:
    """Frames per second and board telemetry with every component busy, each on a different frame. The measured
    span starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = dict.fromkeys(COMPONENTS, 0)

    def push_inputs() -> None:
        try:
            while not measured.is_set():
                push(runners[0], [image], COMPONENTS[0])
                pushed[COMPONENTS[0]] += 1
        finally:
            # End of stream follows the last frame through each component in turn.
            runners[0].close_input()

    # What each component's frames carry for later components, in frame order.
    carries = [deque() for _ in COMPONENTS]

    def forwarder(index: int) -> Callable[[], None]:
        source, target = COMPONENTS[index], COMPONENTS[index + 1]

        def forward() -> None:
            try:
                for tensors in drain(runners[index], source, pushed):
                    available = step(index, carries[index].popleft() if index else {}, tensors)
                    push(
                        runners[index + 1],
                        [
                            available.pop(key) if key not in needed_after(index + 1) else available[key]
                            for key in INPUTS[target]
                        ],
                        target,
                    )
                    carries[index + 1].append(available)
                    pushed[target] += 1
                    # The next component holds its inputs now, so this one gets its output buffers back.
                    del tensors, available
            finally:
                runners[index + 1].close_input()

        forward.__name__ = f"forward_{source}"
        return forward

    def pull_outputs() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(runners[-1], COMPONENTS[-1], pushed):
            # Return the transformer decoder's output buffers before the next pull.
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
        run_together(push_inputs, *(forwarder(i) for i in range(len(COMPONENTS) - 1)), pull_outputs, runners=runners)
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        for runner in runners:
            runner.close()
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
            f"{IMAGE_SIZE}x{IMAGE_SIZE}, the backbone (two components), the pixel decoder (two components per "
            f"deformable layer and one for the 1/4 level) and the transformer decoder up to the class and mask "
            f"logits, pipelined {QUEUE_DEPTH} frames deep"
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
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        name: pyneat.Model(str(mpks[name]), image_options() if name == COMPONENTS[0] else tensor_options())
        for name in COMPONENTS
    }
    path = sample_image(dataset)
    image = frame_tensor(path)
    runners = build_runners(models, image)
    latencies = measure_latency(runners, image)
    throughput = measure_throughput(runners, image)
    write_report(output, f"{dataset_id} {path.name}", latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(mpks[name]) for name in COMPONENTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_pipeline(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
