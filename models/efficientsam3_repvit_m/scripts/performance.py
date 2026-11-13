"""Benchmark EfficientSAM3 RepViT-M with Neat Core and save its report.

Given --mpk text_encoder=<path> --mpk image_model=<path> and --dataset coco-2017-val=<extracted directory>, the
script measures the whole model on one image and prompt per frame: the text encoder on the fixed prompt "person",
the packing of its features and token mask into the image model's text input on the CPU, the stretch and
normalization of one COCO image in a 720p frame on the CPU, and the image model, each stage working on a different
frame at the same time. Running the text encoder for every frame measures an application that changes its prompt
with each image; one with a fixed prompt runs it once. The script also records each MPK's MLA layer and stage stats
and its execution plan.

The fixed prompt's token IDs are prepared before timing, as in CLIP's benchmark. Accuracy and applications use
the tokenizer shipped with the package.

The image model runs on the tensor route, as in accuracy.py: Neat builds its image route only in front of an MPK
with one input, and the image model also takes the packed text.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

IMAGE_SIZE = 1008
CONTEXT, VOCABULARY, FEATURES = 16, 49408, 256
PROMPT = "person"
PROMPT_TOKENS = (49406, 2533, 49407)  # CLIP start token, "person", end token.
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def tensor_options() -> pyneat.ModelOptions:
    """Both models': the script prepares every input, and Neat only quantizes and tessellates."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def ev74_tensor(array: np.ndarray) -> pyneat.Tensor:
    # from_numpy guesses CHW when the first dimension is 1, 3 or 4; the token inputs are HWC.
    return pyneat.Tensor.from_numpy(array, copy=True, memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC)


def prompt_tensor(prompt_tokens: Sequence[int]) -> pyneat.Tensor:
    """The text encoder input [1, 16, 49408]: row i is one-hot for token i; padding rows are zero."""
    one_hot = np.zeros((1, CONTEXT, VOCABULARY), np.float32)
    one_hot[0, np.arange(len(prompt_tokens)), prompt_tokens] = 1
    return ev74_tensor(one_hot)


def pack_text(features: pyneat.Tensor, prompt_tokens: Sequence[int]) -> pyneat.Tensor:
    """The CPU step: the image model's text input [1, 17, 256], rows 0-15 the text encoder's features and row 16
    1 for each prompt token, start and end included."""
    text = np.zeros((1, CONTEXT + 1, FEATURES), np.float32)
    text[0, :CONTEXT] = features.to_numpy(copy=False).reshape(CONTEXT, FEATURES)
    text[0, CONTEXT, : len(prompt_tokens)] = 1
    return ev74_tensor(text)


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
    """The image's 720p frame, placed once in device memory as a camera frame arrives, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def pixels(frame: pyneat.Tensor) -> pyneat.Tensor:
    """The CPU step: the image model's pixel input [1008, 1008, 3], the frame stretched and scaled to [-1, 1] as
    accuracy.py prepares each image."""
    image = cv2.resize(frame.to_numpy(copy=False), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return ev74_tensor(image.astype(np.float32) / np.float32(127.5) - np.float32(1))


def build_runners(
    models: dict[str, pyneat.Model], prompt: pyneat.Tensor, frame: pyneat.Tensor, prompt_tokens: Sequence[int]
) -> tuple[pyneat.ModelRunner, pyneat.ModelRunner]:
    """A runner for each model. The image model's is built from one real text encoder output."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    text_encoder = models["text_encoder"].build([prompt], run_options=run_options)
    (features,) = text_encoder.run([prompt], timeout_ms=TIMEOUT_MS)
    image_model = models["image_model"].build(
        [pixels(frame), pack_text(features, prompt_tokens)], run_options=run_options
    )
    return text_encoder, image_model


def infer_one(
    text_encoder: pyneat.ModelRunner,
    image_model: pyneat.ModelRunner,
    prompt: pyneat.Tensor,
    frame: pyneat.Tensor,
    prompt_tokens: Sequence[int],
) -> None:
    (features,) = text_encoder.run([prompt], timeout_ms=TIMEOUT_MS)
    image_model.run([pixels(frame), pack_text(features, prompt_tokens)], timeout_ms=TIMEOUT_MS)


def measure_latency(
    text_encoder: pyneat.ModelRunner,
    image_model: pyneat.ModelRunner,
    prompt: pyneat.Tensor,
    frame: pyneat.Tensor,
    prompt_tokens: Sequence[int],
) -> list[float]:
    """Seconds per frame through all three stages with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(text_encoder, image_model, prompt, frame, prompt_tokens)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(text_encoder, image_model, prompt, frame, prompt_tokens)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the image model output, how long they took, and the
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


def measure_throughput(
    text_encoder: pyneat.ModelRunner,
    image_model: pyneat.ModelRunner,
    prompt: pyneat.Tensor,
    frame: pyneat.Tensor,
    prompt_tokens: Sequence[int],
) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"text encoder": 0, "image model": 0}

    def push_prompts() -> None:
        try:
            while not measured.is_set():
                push(text_encoder, [prompt], "text encoder")
                pushed["text encoder"] += 1
        finally:
            # End of stream follows the last frame through the text encoder, then the image model.
            text_encoder.close_input()

    def forward_text() -> None:
        try:
            for (features,) in drain(text_encoder, "text encoder", pushed):
                push(image_model, [pixels(frame), pack_text(features, prompt_tokens)], "image model")
                pushed["image model"] += 1
                # The packed text is a copy, so the text encoder gets its output buffer back.
                del features
        finally:
            image_model.close_input()

    def pull_outputs() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(image_model, "image model", pushed):
            # Return the image model's output buffers before the next pull.
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
        run_together(push_prompts, forward_text, pull_outputs, runners=(text_encoder, image_model))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        text_encoder.close()
        image_model.close()
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
            f'text encoder on the prompt "{PROMPT}", text packing on the CPU, and {image} in a {FRAME_WIDTH}x'
            f"{FRAME_HEIGHT} RGB frame, stretched to {IMAGE_SIZE}x{IMAGE_SIZE} and normalized on the CPU, and the "
            f"image model, pipelined {QUEUE_DEPTH} frames deep"
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
    prompt_tokens = PROMPT_TOKENS
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        "text_encoder": pyneat.Model(str(mpks["text_encoder"]), tensor_options()),
        "image_model": pyneat.Model(str(mpks["image_model"]), tensor_options()),
    }
    path = sample_image(dataset)
    prompt, frame = prompt_tensor(prompt_tokens), frame_tensor(path)
    text_encoder, image_model = build_runners(models, prompt, frame, prompt_tokens)
    latencies = measure_latency(text_encoder, image_model, prompt, frame, prompt_tokens)
    throughput = measure_throughput(text_encoder, image_model, prompt, frame, prompt_tokens)
    write_report(
        output, f"{dataset_id} {IMAGES}/{path.name}", latencies, throughput, models, mla_layer_stats, mla_stage_stats
    )


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"text_encoder", "image_model"}:
        raise SystemExit("--mpk must name exactly text_encoder=<path> and image_model=<path>")
    # In run order, so the report lists the components as model.yaml does.
    return {name: Path(mpks[name]) for name in ("text_encoder", "image_model")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="optional package resources; the benchmark prompt is prepared")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_pipeline(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
