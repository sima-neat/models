"""Benchmark SigLIP 2 ViT-B/16 with Neat Core and save its report.

Given --mpk image_encoder=<path> --mpk text_encoder=<path> and --dataset imagenet-2012-val=<extracted directory>,
the script measures the whole model on one image-text pair per frame: Neat's preprocessing and the image encoder on
one ImageNet image in a 720p frame, the text encoder on a 64-token prompt, and their cosine similarity on the CPU,
each stage working on a different frame at the same time. The package carries no tokenizer, so the prompt is a fixed
tensor of token embeddings; the text encoder's time does not depend on their values. The script also records each
MPK's MLA layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import queue
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

IMAGE_SIZE = 224
TEXT_SHAPE = (1, 64, 768)
EMBED_DIM = 768
# The image mean and std, from the checkpoint's preprocessor_config.json.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
VALIDATION_SPLIT = "val2012"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def image_options() -> pyneat.ModelOptions:
    """The image encoder's: Neat stretches, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # Without explicit statistics Neat only scales pixels to [0, 1]; the model expects mean and std 0.5.
    options.preprocess.normalize.mean = MEAN
    options.preprocess.normalize.stddev = STD
    options.preprocess.normalize.has_explicit_stats = True
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The text encoder's: its input is the prompt's token embeddings."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def image_embedding(outputs: list[pyneat.Tensor]) -> np.ndarray:
    """The image encoder's embedding, told apart from its 1,000 logits by its size, copied off the runner."""
    (embedding,) = [array for array in (output.to_numpy().reshape(-1) for output in outputs) if array.size == EMBED_DIM]
    return embedding


def similarity(image: np.ndarray, outputs: list[pyneat.Tensor]) -> float:
    """The cosine similarity of the image and the prompt's pooled embedding: the CPU step."""
    (text_embeds,) = outputs
    text = text_embeds.to_numpy(copy=False).reshape(EMBED_DIM)
    score = float(np.dot(image, text) / (np.linalg.norm(image) * np.linalg.norm(text)))
    if not np.isfinite(score):
        raise ValueError("the image-text similarity is not finite")
    return score


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
    """One val2012 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / VALIDATION_SPLIT).glob("*.JPEG"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {VALIDATION_SPLIT}/")
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
    """The image's 720p frame, placed once in the memory the image encoder reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def prompt_tensor() -> pyneat.Tensor:
    """A fixed 64-token prompt of token-embedding scale, reused for every frame."""
    embeds = np.random.default_rng(0).normal(0.0, 0.02, TEXT_SHAPE).astype(np.float32)
    return pyneat.Tensor.from_numpy(embeds, copy=True, memory=pyneat.TensorMemory.EV74)


def build_runners(models: dict[str, pyneat.Model], image: pyneat.Tensor,
                  prompt: pyneat.Tensor) -> tuple[pyneat.ModelRunner, pyneat.ModelRunner]:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    image_encoder = models["image_encoder"].build([image], run_options=run_options)
    text_encoder = models["text_encoder"].build([prompt], run_options=run_options)
    return image_encoder, text_encoder


def infer_one(image_encoder: pyneat.ModelRunner, text_encoder: pyneat.ModelRunner, image: pyneat.Tensor,
              prompt: pyneat.Tensor) -> None:
    embedding = image_embedding(image_encoder.run([image], timeout_ms=TIMEOUT_MS))
    similarity(embedding, text_encoder.run([prompt], timeout_ms=TIMEOUT_MS))


def measure_latency(image_encoder: pyneat.ModelRunner, text_encoder: pyneat.ModelRunner, image: pyneat.Tensor,
                    prompt: pyneat.Tensor) -> list[float]:
    """Seconds per frame through all three stages with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(image_encoder, text_encoder, image, prompt)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(image_encoder, text_encoder, image, prompt)
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


def measure_throughput(image_encoder: pyneat.ModelRunner, text_encoder: pyneat.ModelRunner, image: pyneat.Tensor,
                       prompt: pyneat.Tensor) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"image encoder": 0, "text encoder": 0}
    embeddings: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()

    def push_frames() -> None:
        try:
            while not measured.is_set():
                push(image_encoder, [image], "image encoder")
                pushed["image encoder"] += 1
        finally:
            # End of stream follows the last frame through the image encoder, then the text encoder.
            image_encoder.close_input()

    def forward_prompts() -> None:
        try:
            for outputs in drain(image_encoder, "image encoder", pushed):
                embeddings.put(image_embedding(outputs))
                # The embedding is copied, so the image encoder gets its output buffers back.
                del outputs
                push(text_encoder, [prompt], "text encoder")
                pushed["text encoder"] += 1
        finally:
            text_encoder.close_input()

    def score_pairs() -> None:
        nonlocal throughput
        frames, started = 0, 0.0
        for outputs in drain(text_encoder, "text encoder", pushed):
            similarity(embeddings.get(), outputs)
            # Return the text encoder's output buffers before the next pull.
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
        run_together(push_frames, forward_prompts, score_pairs, runners=(image_encoder, text_encoder))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        image_encoder.close()
        text_encoder.close()
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
            f"{IMAGE_SIZE}x{IMAGE_SIZE}, image encoder, text encoder on a fixed {TEXT_SHAPE[1]}-token prompt, and "
            f"cosine similarity on the CPU, pipelined {QUEUE_DEPTH} frames deep"
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
        "image_encoder": pyneat.Model(str(mpks["image_encoder"]), image_options()),
        "text_encoder": pyneat.Model(str(mpks["text_encoder"]), tensor_options()),
    }
    path = sample_image(dataset)
    image, prompt = frame_tensor(path), prompt_tensor()
    image_encoder, text_encoder = build_runners(models, image, prompt)
    latencies = measure_latency(image_encoder, text_encoder, image, prompt)
    throughput = measure_throughput(image_encoder, text_encoder, image, prompt)
    write_report(output, f"{dataset_id} {VALIDATION_SPLIT}/{path.name}", latencies, throughput, models,
                 mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"image_encoder", "text_encoder"}:
        raise SystemExit("--mpk must name exactly image_encoder=<path> and text_encoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_pipeline(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
