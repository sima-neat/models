"""Benchmark MobileSAM from a real 720p RGB frame to one final binary mask.

The timed path includes Neat's preprocessing of the frame (SAM's resize of the longer side to 1024, normalization
and right/bottom padding), the whole image encoder, the centre-click prompt, its decoder and Neat's MobileSAM decoder,
which selects the highest-IoU mask among masks 1-3, resizes it to the frame and thresholds it.
Image-file decoding and creation of the fixed camera frame are outside timing, as in the other image benchmarks.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import model_report
import numpy as np
import pyneat
from pipeline import COMPONENTS, PIXEL_MEAN, PIXEL_STD, PromptEncoding, prompt_inputs

WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
# The image source is deterministic; every frame still performs the full SAM image-to-mask work.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114
DATASET = "coco-2017-val"
# SAM pads with zeros after normalizing, which is the mean pixel (124, 116, 104); Neat pads one byte for all channels.
PAD_VALUE = 116


def sample_image(dataset: Path) -> Path:
    images = sorted(path for path in (dataset / "val2017").glob("*.jpg") if not path.name.startswith("."))
    if not images:
        raise SystemExit(f"{dataset} has no val2017 JPEG images")
    return random.Random(0).choice(images)


def read_frame(path: Path) -> np.ndarray:
    """A decoded image as a fixed 720p RGB camera frame; source preparation is outside timing."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def image_options(width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> pyneat.ModelOptions:
    """The image encoder's: Neat prepares the RGB image as SAM does, inside the pipeline.

    The longer side becomes 1024 with antialiased bilinear resampling of the unscaled pixels, the image sits in
    the top-left corner of the 1024 x 1024 input, and SAM's mean and standard deviation normalize it in
    float32, rounded to the encoder's bfloat16 to nearest even as the package's own cast does. Images up to width x
    height fit Neat's input buffers.
    """
    options = pyneat.ModelOptions()
    preprocess = options.preprocess
    preprocess.kind = pyneat.InputKind.Image
    preprocess.input_max_width, preprocess.input_max_height = width, height
    preprocess.enable = pyneat.AutoFlag.On
    preprocess.resize.enable = pyneat.AutoFlag.On
    preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    preprocess.resize.placement = pyneat.LetterboxPlacement.TopLeft
    preprocess.resize.pad_value = PAD_VALUE
    preprocess.resize.interpolation = pyneat.ResizeInterpolation.BilinearAntialiased
    preprocess.rescale_factor = 1.0
    preprocess.normalize.enable = pyneat.AutoFlag.On
    preprocess.normalize.mean = [float(value) for value in PIXEL_MEAN]
    preprocess.normalize.stddev = [float(value) for value in PIXEL_STD]
    preprocess.normalize.has_explicit_stats = True
    preprocess.floating_conversion = pyneat.PreprocFloatingConversion.NearestEven
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The prompt decoder's inputs are the image embedding and the prompt's tensors; Core casts them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def decoder_options() -> pyneat.ModelOptions:
    """The prompt decoder's, with Neat's MobileSAM decoder: the highest-IoU mask among masks 1-3, for one click."""
    options = tensor_options()
    options.decode_type = pyneat.BoxDecodeType.MobileSam
    options.multimask_output = True
    return options


def hwc(value: np.ndarray) -> pyneat.Tensor:
    """A prepared HWC array, or an NCHW batch of one, as an owned tensor for Core.

    Tensor.from_numpy refuses a read-only array even when it copies, so the array is made writable first.
    """
    if value.ndim == 4:
        value = value[0].transpose(1, 2, 0)
    return pyneat.Tensor.from_numpy(np.require(value, requirements="CW"), copy=True, layout=pyneat.TensorLayout.HWC)


def image_tensor(frame: np.ndarray) -> pyneat.Tensor:
    """The camera frame as the encoder's RGB input, in the memory Neat's preprocessing reads; made once."""
    return pyneat.Tensor.from_numpy(
        np.require(frame, requirements="CW"), copy=True, image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def planned(model: pyneat.Model, inputs: list[pyneat.Tensor]) -> SimpleNamespace:
    """The model as model_report reads it, its execution plan drawn from real inputs: Neat plans a Segment Anything
    decoder only from inputs, because its masks take the image's size from them."""
    return SimpleNamespace(execution_plan=lambda: model.execution_plan(inputs))


def prepare_prompt(frame: np.ndarray, encoding: PromptEncoding) -> list[pyneat.Tensor]:
    height, width = frame.shape[:2]
    arrays = prompt_inputs([(width / 2, height / 2)], ["foreground"], (height, width))
    return [hwc(encoding(arrays["point_coords"])), hwc(arrays["point_types"])]


def finish(tensors: list[pyneat.Tensor], frame: np.ndarray) -> np.ndarray:
    """Read Neat's decoded mask while its tensor is retained; return an independently owned frame-sized mask."""
    if len(tensors) != 1:
        raise ValueError(f"expected one decoded mask, got {len(tensors)} outputs")
    (result,) = pyneat.decode_segmentation(list(tensors))
    masks = result.masks.to_numpy(copy=False)
    if masks.shape != (1, *frame.shape[:2]):
        raise ValueError(f"expected one {frame.shape[:2]} mask, got {masks.shape}")
    return masks[0] > 0


def close_runners(runners: list[pyneat.ModelRunner]) -> None:
    """Attempt every close without hiding an active build/inference exception."""
    active_error = sys.exc_info()[1]
    close_error = None
    for runner in reversed(runners):
        try:
            runner.close()
        except BaseException as error:
            close_error = close_error or error
            print(f"runner cleanup failed: {error}", file=sys.stderr)
    if active_error is None and close_error is not None:
        raise close_error


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """The encoder's outputs, copied to EV74 memory for the decoder's first stage.

    The copy keeps the image geometry Neat's preprocessing recorded, which the decoder's mask resize reads.
    """
    return [tensor.cvu() for tensor in tensors]


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


def stage_inputs(component: str, tensors: list[pyneat.Tensor], prompt: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """A component's inputs from the previous component's outputs; the decoder also takes the prompt."""
    return [*tensors, *prompt] if component == "prompt_decoder" else tensors


def build_runners(
    models: dict[str, pyneat.Model], image: pyneat.Tensor, prompt: list[pyneat.Tensor]
) -> list[pyneat.ModelRunner]:
    """A runner for each component, in COMPONENTS order, each built from one real output of the one before."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runners, tensors = [], [image]
    try:
        for index, component in enumerate(COMPONENTS):
            inputs = stage_inputs(component, tensors, prompt)
            try:
                runners.append(models[component].build(inputs, run_options=run_options))
            except pyneat.NeatError as error:
                shapes = [tuple(tensor.shape) for tensor in inputs]
                raise RuntimeError(f"building the {component} runner on input shapes {shapes} failed") from error
            outputs = runners[-1].run(inputs, timeout_ms=TIMEOUT_MS)
            if index + 1 < len(COMPONENTS):
                tensors = handoff(outputs)
        return runners
    except BaseException:
        close_runners(runners)
        raise


def infer_one(
    runners: list[pyneat.ModelRunner], image: pyneat.Tensor, frame: np.ndarray, encoding: PromptEncoding
) -> np.ndarray:
    tensors = [image]
    prompt = prepare_prompt(frame, encoding)
    for index, (component, runner) in enumerate(zip(COMPONENTS, runners)):
        outputs = runner.run(stage_inputs(component, tensors, prompt), timeout_ms=TIMEOUT_MS)
        if index + 1 < len(COMPONENTS):
            tensors = handoff(outputs)
    return finish(outputs, frame)


def measure_latency(
    runners: list[pyneat.ModelRunner], image: pyneat.Tensor, frame: np.ndarray, encoding: PromptEncoding
) -> list[float]:
    """Seconds from RGB input through final mask, with one frame in flight, after warmup."""
    for _ in range(WARMUP_FRAMES):
        infer_one(runners, image, frame, encoding)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(runners, image, frame, encoding)
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


def run_together(
    *stages: Callable[[], None], runners: tuple[pyneat.ModelRunner, ...], cancel: Callable[[], None]
) -> None:
    """Run stages with bounded queues; cancel and join every started thread before propagating a failure."""
    errors: list[BaseException] = []

    def close_inputs() -> None:
        cancel()
        for runner in runners:
            try:
                runner.close_input()
            except BaseException as cleanup_error:
                print(f"closing runner input failed: {cleanup_error}", file=sys.stderr)

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            close_inputs()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
        for thread in started:
            thread.join()
        if errors:
            raise errors[0]
    finally:
        primary_error = sys.exc_info()[1]
        if primary_error is not None:
            close_inputs()
        join_error = None
        # A start or join can fail before the usual drain; attempt every started join before releasing the runners.
        for thread in started:
            try:
                thread.join()
            except BaseException as error:
                close_inputs()
                join_error = join_error or error
                print(f"joining pipeline thread failed: {error}", file=sys.stderr)
        if primary_error is None and join_error is not None:
            raise join_error


def measure_throughput(
    runners: list[pyneat.ModelRunner], image: pyneat.Tensor, frame: np.ndarray, encoding: PromptEncoding
) -> Throughput:
    """Frames per second and board telemetry with every component busy, each on a different frame. The measured
    span starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = dict.fromkeys(COMPONENTS, 0)

    def push_images() -> None:
        try:
            while not measured.is_set():
                push(runners[0], [image], COMPONENTS[0])
                pushed[COMPONENTS[0]] += 1
        finally:
            # End of stream follows the last frame through every component.
            runners[0].close_input()

    def forward(index: int) -> Callable[[], None]:
        source, target = COMPONENTS[index], COMPONENTS[index + 1]

        def forward_outputs() -> None:
            try:
                for tensors in drain(runners[index], source, pushed):
                    prompt = prepare_prompt(frame, encoding) if target == "prompt_decoder" else []
                    push(runners[index + 1], stage_inputs(target, handoff(tensors), prompt), target)
                    pushed[target] += 1
                    # The next runner holds its inputs now, so this one gets its output buffers back.
                    del tensors
            finally:
                runners[index + 1].close_input()

        forward_outputs.__name__ = f"forward_{source}"
        return forward_outputs

    def pull_masks() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(runners[-1], COMPONENTS[-1], pushed):
            finish(tensors, frame)
            # The independently owned mask is complete before counting its frame and releasing the outputs.
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

    stages = [push_images, *(forward(index) for index in range(len(runners) - 1)), pull_masks]
    try:
        run_together(*stages, runners=tuple(runners), cancel=measured.set)
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
    return throughput


def write_report(
    output: Path,
    latencies: list[float],
    throughput: Throughput,
    models: dict[str, pyneat.Model],
    mla_layer_stats: dict[str, dict[str, object]],
    mla_stage_stats: dict[str, dict[str, object]],
    source: Path,
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
            f"{DATASET} val2017/{source.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame; "
            "Neat preprocessing (top-left letterbox to 1024x1024, antialiased bilinear, SAM normalization), "
            "the whole image encoder, centre foreground click, "
            "CPU FP32 prompt Fourier encoding, prompt decoder and Neat's MobileSAM decoder (best-IoU candidate "
            "mask 1-3 resized to the frame and thresholded at zero); "
            f"pipelined {QUEUE_DEPTH} frames deep"
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


def benchmark_pipeline(
    mpks: dict[str, Path], dataset: Path, output: Path, resources: Path | None = None
) -> None:
    # Load package-bound CPU weights before constructing any model, profiling or building a native runner.
    encoding = PromptEncoding.from_package(mpks["prompt_decoder"], resources)
    models = {
        "image_encoder": pyneat.Model(str(mpks["image_encoder"]), image_options()),
        "prompt_decoder": pyneat.Model(str(mpks["prompt_decoder"]), decoder_options()),
    }
    mla_layer_stats, mla_stage_stats = {}, {}
    # Separate idle diagnostic runs; their layer times are not a loaded-pipeline stage decomposition.
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    source = sample_image(dataset)
    frame = read_frame(source)
    image = image_tensor(frame)
    runners = build_runners(models, image, prepare_prompt(frame, encoding))
    try:
        decoder_inputs = stage_inputs(
            "prompt_decoder", handoff(runners[0].run([image], timeout_ms=TIMEOUT_MS)), prepare_prompt(frame, encoding)
        )
        latencies = measure_latency(runners, image, frame, encoding)
        throughput = measure_throughput(runners, image, frame, encoding)
        reported = {**models, "prompt_decoder": planned(models["prompt_decoder"], decoder_inputs)}
        write_report(output, latencies, throughput, reported, mla_layer_stats, mla_stage_stats, source)
    finally:
        close_runners(runners)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="package root holding prompt-encoding.json")
    args = parser.parse_args()
    dataset_id, separator, directory = args.dataset.partition("=")
    if dataset_id != DATASET or not separator or not directory.strip():
        parser.error(f"--dataset must name {DATASET}=<extracted directory>")
    benchmark_pipeline(parse_mpks(args.mpk), Path(directory), Path(args.output), args.resources)


if __name__ == "__main__":
    main()
