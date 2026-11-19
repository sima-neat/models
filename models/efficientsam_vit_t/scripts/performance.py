"""Benchmark EfficientSAM ViT-T with Neat Core and save its report.

Given --mpk image_encoder=<path> --mpk prompt_decoder=<path>, the script measures the whole model on one image and
one box prompt per frame: Neat's preprocessing and the image encoder on a fixed 720p frame, the CPU normalizing the
box prompt's points, the prompt decoder on the encoder's embedding with those points, and the CPU upsampling the
best of the decoder's masks to the frame, each stage working on a different frame at the same time. The model
declares no accuracy dataset, so the script builds its frame: a seeded outdoor scene with one object, which the
prompt's box surrounds. Before measuring, it checks that the compiled models' mask covers that object, which also
checks the encoder's MLA code, whose compile skips AFE's simulator check. The script also records each MPK's MLA
layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

IMAGE_SIZE = 1024
MAX_POINTS = 6
POINT_TYPES = ("padding", "negative", "positive", "box_top_left", "box_bottom_right")
# A points row: x, y, the one-hot point type and a 0.
POINT_COLUMNS = 8
# Neat returns low_res_masks as 1x256x256x3: height, width, and a channel per candidate mask.
MASK_SHAPE = (256, 256, 3)
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
# The object in the benchmark frame, and the box prompt around it: (x0, y0, x1, y1) in frame pixels.
OBJECT_BOX = (430.0, 210.0, 850.0, 560.0)
# The least intersection over union with the object that the compiled models' mask must reach.
MIN_OBJECT_IOU = 0.5


def image_options() -> pyneat.ModelOptions:
    """The image encoder's: Neat stretches the frame to 1024x1024 and scales it to [0, 1] inside the pipeline; the
    encoder subtracts ImageNet's mean and divides by its std itself."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    # Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    # Without a preset or explicit statistics Neat only scales pixels to [0, 1].
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The prompt decoder's: its inputs are the image embedding and the prompt's point tensors."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def synthetic_frame() -> np.ndarray:
    """A fixed 1280x720 RGB frame: sky above a field, seeded soft shapes and mild noise, and one solid object inside
    OBJECT_BOX for the prompt to mark."""
    rng = np.random.default_rng(0)
    rows = np.linspace(0.0, 1.0, FRAME_HEIGHT, dtype=np.float32)[:, None, None]
    sky = np.asarray([120, 170, 230], np.float32) * (1.0 - 0.35 * rows)
    field = np.asarray([70, 130, 60], np.float32) * (1.25 - 0.5 * rows)
    frame = np.where(rows < 0.55, sky, field) * np.ones((1, FRAME_WIDTH, 1), np.float32)
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
    for _ in range(10):
        x, y = rng.uniform(0, FRAME_WIDTH), rng.uniform(0, FRAME_HEIGHT)
        radius, colour = rng.uniform(30, 120), rng.uniform(20, 235, 3).astype(np.float32)
        weight = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * radius**2))[..., None]
        frame = frame * (1 - weight) + colour * weight
    frame[object_mask()] = np.asarray([200, 60, 40], np.float32)
    frame += rng.normal(0.0, 3.0, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def object_mask() -> np.ndarray:
    """The pixels of the frame's object: an ellipse 20 pixels inside OBJECT_BOX."""
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
    x0, y0, x1, y1 = OBJECT_BOX
    return (
        ((xs - (x0 + x1) / 2) / ((x1 - x0) / 2 - 20)) ** 2 + ((ys - (y0 + y1) / 2) / ((y1 - y0) / 2 - 20)) ** 2
    ) <= 1


def prompt_array(points: list[tuple[float, float]], types: list[str], size: tuple[int, int]) -> np.ndarray:
    """The prompt decoder's points [1,6,8] for points in the image's pixels: per row x, y, the one-hot type and a 0.

    A point (x, y) of a height x width image becomes ((x * 1024 / width + 0.5) / 1024,
    (y * 1024 / height + 0.5) / 1024);
    unused rows are padding at coordinate -1, as EfficientSAM pads its prompts.
    """
    height, width = size
    rows = np.zeros((1, MAX_POINTS, POINT_COLUMNS), np.float32)
    rows[0, :, :2] = -0.5 / IMAGE_SIZE
    rows[0, :, 2] = 1.0
    for row, ((x, y), kind) in enumerate(zip(points, types)):
        rows[0, row, :2] = ((x * IMAGE_SIZE / width + 0.5) / IMAGE_SIZE, (y * IMAGE_SIZE / height + 0.5) / IMAGE_SIZE)
        rows[0, row, 2] = 0.0
        rows[0, row, 2 + POINT_TYPES.index(kind)] = 1.0
    return rows


def prompt_tensors() -> list[pyneat.Tensor]:
    """The CPU step before the prompt decoder, run for every frame: the box prompt around the frame's object,
    normalized into the decoder's points and placed in the memory the decoder reads.

    Neat sizes the points input as height 1, width 6 and depth 8, so the tensor declares HWC axes; untagged, a 3-D
    tensor counts as CHW.
    """
    x0, y0, x1, y1 = OBJECT_BOX
    array = prompt_array([(x0, y0), (x1, y1)], ["box_top_left", "box_bottom_right"], (FRAME_HEIGHT, FRAME_WIDTH))
    return [pyneat.Tensor.from_numpy(array, copy=True, memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC)]


def frame_mask(outputs: list[pyneat.Tensor]) -> np.ndarray:
    """The CPU step: the mask with the highest predicted IoU, upsampled to the frame and thresholded at 0."""
    arrays = [output.to_numpy(copy=False) for output in outputs]
    (masks,) = [array for array in arrays if array.shape[-3:] == MASK_SHAPE]
    (scores,) = [array.reshape(-1) for array in arrays if array.size == MASK_SHAPE[-1]]
    best = np.ascontiguousarray(masks.reshape(MASK_SHAPE)[:, :, int(np.argmax(scores))])
    return cv2.resize(best, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR) > 0


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


def decoder_embeddings(embeddings: pyneat.Tensor) -> pyneat.Tensor:
    """The image encoder's output, copied to EV74 memory for the prompt decoder.

    Neat returns the encoder's output as a CPU tensor, and the decoder's first stage runs on the EV74. Neat Core develop
    builds since 27 October refuse to bind that tensor to it ("EV74 requires admitted CMA or contiguous camera memory"),
    so each frame's embedding is copied to EV74 memory, where the prompt is created too.
    """
    return embeddings.cvu()


def build_runners(models: dict[str, pyneat.Model],
                  image: pyneat.Tensor) -> tuple[pyneat.ModelRunner, pyneat.ModelRunner]:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    image_encoder = models["image_encoder"].build([image], run_options=run_options)
    (embeddings,) = image_encoder.run([image], timeout_ms=TIMEOUT_MS)
    prompt_decoder = models["prompt_decoder"].build([decoder_embeddings(embeddings), *prompt_tensors()],
                                                    run_options=run_options)
    return image_encoder, prompt_decoder


def check_mask(image_encoder: pyneat.ModelRunner, prompt_decoder: pyneat.ModelRunner, image: pyneat.Tensor) -> float:
    """The intersection over union of the compiled models' mask with the frame's object. Below MIN_OBJECT_IOU the
    models do not segment what the benchmark feeds them, so the script stops."""
    (embeddings,) = image_encoder.run([image], timeout_ms=TIMEOUT_MS)
    mask = frame_mask(prompt_decoder.run([decoder_embeddings(embeddings), *prompt_tensors()], timeout_ms=TIMEOUT_MS))
    truth = object_mask()
    iou = float(np.sum(mask & truth) / np.sum(mask | truth))
    if iou < MIN_OBJECT_IOU:
        raise RuntimeError(
            f"the compiled models' mask covers the frame's object with IoU {iou:.3f}, below {MIN_OBJECT_IOU}"
        )
    return iou


def infer_one(image_encoder: pyneat.ModelRunner, prompt_decoder: pyneat.ModelRunner, image: pyneat.Tensor) -> None:
    (embeddings,) = image_encoder.run([image], timeout_ms=TIMEOUT_MS)
    frame_mask(prompt_decoder.run([decoder_embeddings(embeddings), *prompt_tensors()], timeout_ms=TIMEOUT_MS))


def measure_latency(image_encoder: pyneat.ModelRunner, prompt_decoder: pyneat.ModelRunner,
                    image: pyneat.Tensor) -> list[float]:
    """Seconds per frame through all four stages with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(image_encoder, prompt_decoder, image)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(image_encoder, prompt_decoder, image)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the CPU step finished, how long they took, and the
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


def measure_throughput(image_encoder: pyneat.ModelRunner, prompt_decoder: pyneat.ModelRunner,
                       image: pyneat.Tensor) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"image encoder": 0, "prompt decoder": 0}

    def push_frames() -> None:
        try:
            while not measured.is_set():
                push(image_encoder, [image], "image encoder")
                pushed["image encoder"] += 1
        finally:
            # End of stream follows the last frame through the image encoder, then the prompt decoder.
            image_encoder.close_input()

    def forward_embeddings() -> None:
        try:
            for (embeddings,) in drain(image_encoder, "image encoder", pushed):
                push(prompt_decoder, [decoder_embeddings(embeddings), *prompt_tensors()], "prompt decoder")
                pushed["prompt decoder"] += 1
                # The decoder holds a copy, so the image encoder gets its output buffer back.
                del embeddings
        finally:
            prompt_decoder.close_input()

    def build_masks() -> None:
        nonlocal throughput
        frames, started = 0, 0.0
        for outputs in drain(prompt_decoder, "prompt decoder", pushed):
            frame_mask(outputs)
            # Return the decoder's output buffers before the next pull.
            del outputs
            frames += 1
            if frames == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif frames > WARMUP_FRAMES and not measured.is_set():
                count, seconds = frames - WARMUP_FRAMES, time.perf_counter() - started
                if count >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(count, seconds, monitor.stop())
                    measured.set()
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    try:
        run_together(push_frames, forward_embeddings, build_masks, runners=(image_encoder, prompt_decoder))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        image_encoder.close()
        prompt_decoder.close()
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
            f"a fixed {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame the script builds, Neat preprocessing stretched to "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}, image encoder, a fixed box prompt's points normalized on the CPU, prompt "
            f"decoder, and the best mask upsampled to the frame on the CPU, pipelined {QUEUE_DEPTH} frames deep"
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
    models = {
        "image_encoder": pyneat.Model(str(mpks["image_encoder"]), image_options()),
        "prompt_decoder": pyneat.Model(str(mpks["prompt_decoder"]), tensor_options()),
    }
    image = pyneat.Tensor.from_numpy(synthetic_frame(), copy=True, image_format=pyneat.PixelFormat.RGB,
                                     memory=pyneat.TensorMemory.EV74)
    image_encoder, prompt_decoder = build_runners(models, image)
    print(f"Mask IoU with the frame's object: {check_mask(image_encoder, prompt_decoder, image):.3f}")
    latencies = measure_latency(image_encoder, prompt_decoder, image)
    throughput = measure_throughput(image_encoder, prompt_decoder, image)
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"image_encoder", "prompt_decoder"}:
        raise SystemExit("--mpk must name exactly image_encoder=<path> and prompt_decoder=<path>")
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
