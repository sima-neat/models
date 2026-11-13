"""Benchmark EdgeTAM with Neat Core as it tracks one object through a video, and save its report.

Given --mpk <component>=<path> for image_encoder, memory_attention, mask_decoder and memory_encoder, the script
measures one tracked frame: Neat's preprocessing and the image encoder, the memory attention over the memory bank,
the mask decoder, the mask choice and upsampling on the CPU, and the memory encoder, whose memory joins the bank.
The input is the first frame of one seeded DAVIS 2017 val sequence from --dataset davis-2017-val=<directory>,
letterboxed into a 720p frame. The script clicks an annotated object on it and tracks the object through repeats of
that frame.

Latency runs one frame at a time and reports each step's share. In the throughput pass the image encoder works ahead
on the next frames while the tracker finishes the current one. A frame's memory attention needs the previous frame's
memory, so the other three components run in order. The script also records each MPK's MLA layer and stage stats
and its execution plan.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

COMPONENTS = ["image_encoder", "memory_attention", "mask_decoder", "memory_encoder"]
# surgery.py's interfaces, channels last as the MPKs take and return them.
IMAGE_SIZE = 1024
EMBEDDING = (1, 64, 64, 256)
HIGH_RES = ((1, 256, 256, 32), (1, 128, 128, 64))
MASKS, MASK_SIZE = 4, 256
HIDDEN, MEMORY_DIM = 256, 64
MEMORY_TOKENS, MEMORY_SLOTS, OBJECT_POINTERS = 512, 7, 16
PROMPT_POINTS = 3
POINT_TYPES = ["padding", "background", "foreground", "box_top_left", "box_bottom_right"]
# The tracker's mask logits for a frame where the object score says the object is absent.
NO_OBJECT_LOGIT = -1024.0
# A 16-pointer bank is full after this many frames, so the measured frames all attend to a full bank.
WARMUP_FRAMES = 20
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames the image encoder may work ahead of the tracker; more adds latency without adding throughput.
QUEUE_DEPTH = 2
# The benchmark input is a 720p camera frame.
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
FILL = 114
SPLIT = "val"
# One thread per CPU core turns its share of the upsampled mask's rows into probabilities.
CPU_CORES = 4
MASK_THREADS = concurrent.futures.ThreadPoolExecutor(CPU_CORES)


def sample_image(dataset: Path) -> Path:
    """The first frame of one val sequence, chosen with a fixed seed so every run measures the same input."""
    sequences = (dataset / f"{SPLIT}.txt").read_text(encoding="utf-8").split()
    frames = sorted((dataset / SPLIT / random.Random(0).choice(sequences)).glob("*.jpg"))
    if not frames:
        raise SystemExit(f"{dataset} has no frames under {SPLIT}/")
    return frames[0]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(width: int, height: int) -> tuple[float, int, int]:
    """The scale and the left and top padding that centre a width x height image in the 1280x720 frame."""
    scale = min(FRAME_WIDTH / width, FRAME_HEIGHT / height)
    return scale, (FRAME_WIDTH - round(width * scale)) // 2, (FRAME_HEIGHT - round(height * scale)) // 2


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale, left, top = letterbox(image.shape[1], image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def object_click(image: Path) -> tuple[float, float]:
    """The first frame's click, as (x, y) frame pixels: the annotated pixel farthest from the background."""
    annotation = image.parents[2] / "annotations" / image.parent.name / f"{image.stem}.png"
    objects = cv2.imread(str(annotation), cv2.IMREAD_GRAYSCALE)
    if objects is None or not objects.any():
        raise SystemExit(f"{annotation} annotates no object")
    # The mask is padded so an object touching the frame's edge still counts the edge as background.
    depth = cv2.distanceTransform(cv2.copyMakeBorder((objects > 0).astype(np.uint8), 1, 1, 1, 1, 0), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(depth[1:-1, 1:-1]), objects.shape)
    scale, left, top = letterbox(objects.shape[1], objects.shape[0])
    return left + (x + 0.5) * scale, top + (y + 0.5) * scale


def image_options() -> pyneat.ModelOptions:
    """The image encoder's: Neat stretches, normalizes, quantizes and tessellates the frame inside the pipeline."""
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
    # The output stage runs on the EV74 and writes memory the other components' input stages read, so the
    # embeddings and features pass on without a copy. The CPU output stage writes memory only the CPU reads.
    options.processcvu.post_run_target = "EV74"
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The other components': their inputs are features, memories and prompts, which Neat only converts."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def decoder_options() -> pyneat.ModelOptions:
    """The mask decoder's: its input stage runs on the CPU, which reads the memory attention's output where that
    component's CPU output stage wrote it, and the image encoder's features where the EV74 wrote them."""
    options = tensor_options()
    options.processcvu.pre_run_target = "A65"
    return options


MODEL_OPTIONS = {
    "image_encoder": image_options,
    "memory_attention": tensor_options,
    "mask_decoder": decoder_options,
    "memory_encoder": tensor_options,
}


def cpu_tensor(array: np.ndarray) -> pyneat.Tensor:
    """A float32 array the CPU made, in the memory a component's input stage reads. Its shape is already the
    MPK's, so it carries no layout: Neat would read a 3D array as channels first and reject it."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(array, np.float32),
        copy=True,
        memory=pyneat.TensorMemory.EV74,
        layout=pyneat.TensorLayout.Unknown,
    )


def prompt_arrays(points: list[tuple[float, float]], types: list[str]) -> list[np.ndarray]:
    """The mask decoder's point_coords and point_types for (x, y) frame pixels; unused slots are padding.

    Coordinates are scaled to the 1024x1024 model input, shifted to pixel centres and normalized to [-1, 1], as
    calibration.Prompt does.
    """
    coords = np.zeros((1, 1, PROMPT_POINTS, 2), np.float32)
    one_hot = np.zeros((1, 1, PROMPT_POINTS, len(POINT_TYPES)), np.float32)
    one_hot[..., POINT_TYPES.index("padding")] = 1
    for slot, ((x, y), kind) in enumerate(zip(points, types, strict=True)):
        model_pixel = np.asarray([x * IMAGE_SIZE / FRAME_WIDTH, y * IMAGE_SIZE / FRAME_HEIGHT], np.float32)
        coords[0, 0, slot] = (model_pixel + 0.5) / IMAGE_SIZE * 2 - 1
        one_hot[0, 0, slot] = np.eye(len(POINT_TYPES), dtype=np.float32)[POINT_TYPES.index(kind)]
    return [coords, one_hot]


def choose_mask(
    masks: np.ndarray, iou_scores: np.ndarray, object_score: float, pointers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The tracker's [256, 256] mask logits and object pointer from the decoder's four masks.

    A click and a tracked frame keep the best of masks 1-3 by predicted IoU, as calibration.select_mask does; a
    frame whose object score is not positive has no object.
    """
    best = 1 + int(np.argmax(iou_scores[1:]))
    mask = masks[..., best]
    if object_score <= 0:
        mask = np.full_like(mask, NO_OBJECT_LOGIT)
    return mask, pointers[best]


def to_probabilities(logits: np.ndarray) -> None:
    """Replace logit rows with 1 / (1 + exp(-logit)) in place.

    OpenCV's exp takes a third of numpy's time on the board. It overflows to infinity for the no-object logit,
    which gives the probability 0 it should.
    """
    np.negative(logits, out=logits)
    cv2.exp(logits, dst=logits)
    logits += 1
    np.reciprocal(logits, out=logits)


def memory_mask(mask: np.ndarray, prompted: bool) -> np.ndarray:
    """The memory encoder's [1, 1024, 1024, 1] mask: the logits upsampled bilinearly, thresholded on the prompted
    frame and as probabilities on a tracked one."""
    values = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    if prompted:
        values = (values > 0).astype(np.float32)
    else:
        # OpenCV and numpy release the GIL, so the row blocks run on separate cores.
        list(MASK_THREADS.map(to_probabilities, np.array_split(values, CPU_CORES)))
    return values.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


@dataclass
class MemoryBank:
    """The frames the tracker remembers: the prompted frame, then the latest frames, newest first."""

    memories: collections.deque = field(default_factory=lambda: collections.deque(maxlen=MEMORY_SLOTS - 1))
    pointers: collections.deque = field(default_factory=lambda: collections.deque(maxlen=OBJECT_POINTERS - 1))
    prompted_memory: np.ndarray | None = None
    prompted_pointer: np.ndarray | None = None

    def add(self, memory: np.ndarray, pointer: np.ndarray) -> None:
        if self.prompted_memory is None:
            self.prompted_memory, self.prompted_pointer = memory, pointer
        else:
            self.memories.appendleft(memory)
            self.pointers.appendleft(pointer)

    def arrays(self) -> list[np.ndarray]:
        """The memory attention's memories, memory_mask, object_pointers and object_pointer_mask. An unused slot
        stays zero with a mask of 0."""
        slots = [self.prompted_memory, *self.memories]
        memories = np.zeros((1, MEMORY_SLOTS * MEMORY_TOKENS, MEMORY_DIM), np.float32)
        memories[0, : len(slots) * MEMORY_TOKENS] = np.concatenate(slots)
        used_memories = np.zeros((1, MEMORY_SLOTS), np.float32)
        used_memories[0, : len(slots)] = 1
        kept = [self.prompted_pointer, *self.pointers]
        pointers = np.zeros((1, OBJECT_POINTERS, HIDDEN), np.float32)
        pointers[0, : len(kept)] = np.stack(kept)
        used_pointers = np.zeros((1, OBJECT_POINTERS), np.float32)
        used_pointers[0, : len(kept)] = 1
        return [memories, used_memories, pointers, used_pointers]


@dataclass
class Tracker:
    """One object's tracking state on the board, with the seconds each step of every frame took."""

    runners: dict[str, pyneat.ModelRunner]
    bank: MemoryBank = field(default_factory=MemoryBank)
    seconds: dict[str, list[float]] = field(default_factory=lambda: collections.defaultdict(list))

    @contextmanager
    def timed(self, step: str) -> Iterator[None]:
        started = time.perf_counter()
        yield
        self.seconds[step].append(time.perf_counter() - started)

    def run(self, component: str, inputs: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
        with self.timed(component):
            return self.runners[component].run(inputs, timeout_ms=TIMEOUT_MS)

    def encode(self, frame: pyneat.Tensor) -> list[pyneat.Tensor]:
        return self.run("image_encoder", [frame])

    def track(self, encoded: list[pyneat.Tensor], prompt: list[np.ndarray] | None = None) -> np.ndarray:
        """Segment one frame from its image encoder outputs, prompted on the first frame and tracked on later ones,
        and add its memory to the bank; return the mask logits."""
        prompted = prompt is not None
        # Each component reads the previous one's outputs where they are (MODEL_OPTIONS), so none is copied.
        embeddings, *high_res = encoded
        conditioned = embeddings
        if not prompted:
            with self.timed("cpu"):
                bank = [cpu_tensor(array) for array in self.bank.arrays()]
            (conditioned,) = self.run("memory_attention", [embeddings, *bank])
        with self.timed("cpu"):
            points = [cpu_tensor(array) for array in (prompt if prompted else prompt_arrays([], []))]
        decoded = self.run("mask_decoder", [conditioned, *high_res, *points])
        with self.timed("cpu"):
            masks, iou_scores, object_score, pointers = (tensor.to_numpy(copy=False) for tensor in decoded)
            mask, pointer = choose_mask(
                masks.reshape(MASK_SIZE, MASK_SIZE, MASKS),
                iou_scores.reshape(MASKS),
                float(object_score.reshape(())),
                pointers.reshape(MASKS, HIDDEN),
            )
            mask, pointer = mask.copy(), pointer.copy()
            # Return the decoder's output buffers before the memory encoder runs.
            del masks, iou_scores, object_score, pointers, decoded
            remembered = cpu_tensor(memory_mask(mask, prompted))
        (memory,) = self.run("memory_encoder", [embeddings, remembered])
        with self.timed("cpu"):
            self.bank.add(memory.to_numpy(copy=True).reshape(MEMORY_TOKENS, MEMORY_DIM), pointer)
        return mask


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


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the image encoder reads and reused for every frame."""
    return pyneat.Tensor.from_numpy(
        letterbox_to_720p(read_rgb(path)),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def build_runners(models: dict[str, pyneat.Model], frame: pyneat.Tensor) -> dict[str, pyneat.ModelRunner]:
    """A runner for each model. Only the image encoder's may queue frames; the others are built from inputs of
    their shapes."""
    queued = pyneat.RunOptions()
    queued.queue_depth = QUEUE_DEPTH
    queued.overflow_policy = pyneat.OverflowPolicy.Block
    embeddings = cpu_tensor(np.zeros(EMBEDDING))
    high_res = [cpu_tensor(np.zeros(shape)) for shape in HIGH_RES]
    bank = MemoryBank()
    bank.add(np.zeros((MEMORY_TOKENS, MEMORY_DIM), np.float32), np.zeros(HIDDEN, np.float32))
    prompt = [cpu_tensor(array) for array in prompt_arrays([], [])]
    mask = cpu_tensor(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1)))
    return {
        "image_encoder": models["image_encoder"].build([frame], run_options=queued),
        "memory_attention": models["memory_attention"].build(
            [embeddings, *(cpu_tensor(array) for array in bank.arrays())]
        ),
        "mask_decoder": models["mask_decoder"].build([embeddings, *high_res, *prompt]),
        "memory_encoder": models["memory_encoder"].build([embeddings, mask]),
    }


def measure_latency(tracker: Tracker, frame: pyneat.Tensor) -> list[float]:
    """Seconds per tracked frame through every step with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        tracker.track(tracker.encode(frame))
    tracker.seconds.clear()
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        tracker.track(tracker.encode(frame))
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames tracked, how long they took, and the board telemetry
    sampled over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def run_together(*stages: Callable[[], None], runner: pyneat.ModelRunner) -> None:
    """Run each stage on its own thread. The first failure ends the runner's input, which unblocks a stage waiting
    to push, and is raised once all stages stop. Each stage waits in Core with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            runner.close_input()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def measure_throughput(tracker: Tracker, frame: pyneat.Tensor) -> Throughput:
    """Tracked frames per second and board telemetry with the image encoder working ahead of the tracker. The
    measured span starts after WARMUP_FRAMES frames and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S
    seconds."""
    encoder = tracker.runners["image_encoder"]
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = 0

    def push_frames() -> None:
        nonlocal pushed
        try:
            while not measured.is_set():
                if not encoder.push([frame]):
                    raise RuntimeError("the image encoder rejected a frame")
                pushed += 1
        finally:
            # End of stream follows the last frame through the image encoder.
            encoder.close_input()

    def track_frames() -> None:
        nonlocal throughput
        tracked, started = 0, 0.0
        # Core's pull returns no tensors both at end of stream and on a timeout; the count below tells them apart.
        while encoded := encoder.pull(TIMEOUT_MS).tensors:
            tracker.track(encoded)
            # Return the image encoder's output buffers before the next pull.
            del encoded
            tracked += 1
            if tracked == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif tracked > WARMUP_FRAMES and not measured.is_set():
                frames, seconds = tracked - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(frames, seconds, monitor.stop())
                    measured.set()
        if tracked != pushed:
            raise RuntimeError(f"the image encoder stalled: {tracked} of {pushed} frames came out")

    try:
        run_together(push_frames, track_frames, runner=encoder)
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
    return throughput


def write_report(
    output: Path,
    frame_name: str,
    latencies: list[float],
    throughput: Throughput,
    models: dict[str, pyneat.Model],
    mla_layer_stats: dict[str, dict[str, object]],
    mla_stage_stats: dict[str, dict[str, object]],
) -> None:
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = float(np.percentile(latencies, 50) * 1000)
    benchmark.latency_p95_ms = float(np.percentile(latencies, 95) * 1000)
    benchmark.latency_p99_ms = float(np.percentile(latencies, 99) * 1000)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"tracked frames after one click on an annotated object, on {frame_name} letterboxed to "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} RGB: Neat preprocessing stretched to {IMAGE_SIZE}x{IMAGE_SIZE}, image "
            f"encoder, memory attention over {MEMORY_SLOTS} memories and {OBJECT_POINTERS} object pointers, mask "
            f"decoder, mask choice and upsampling on the CPU, and memory encoder; latency one frame at a time, "
            f"throughput with the image encoder up to {QUEUE_DEPTH} frames ahead"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)


def benchmark_tracking(mpks: dict[str, Path], dataset_id: str, dataset: Path, output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpks[name]), MODEL_OPTIONS[name]()) for name in COMPONENTS}
    image = sample_image(dataset)
    frame = frame_tensor(image)
    tracker = Tracker(build_runners(models, frame))
    try:
        tracker.track(tracker.encode(frame), prompt_arrays([object_click(image)], ["foreground"]))
        latencies = measure_latency(tracker, frame)
        # The CPU works several times in a frame, so each step's share is its total over the measured frames.
        steps_ms = {step: sum(seconds) / len(latencies) * 1000 for step, seconds in tracker.seconds.items()}
        throughput = measure_throughput(tracker, frame)
    finally:
        for runner in tracker.runners.values():
            runner.close()
    frame_name = f"{dataset_id} {image.parent.name}/{image.name}"
    write_report(output, frame_name, latencies, throughput, models, mla_layer_stats, mla_stage_stats)
    shares = ", ".join(f"{step} {steps_ms[step]:.2f} ms" for step in [*COMPONENTS, "cpu"])
    print(
        f"{throughput.fps:.2f} FPS over {throughput.frames} frames in {throughput.seconds:.1f} s; "
        f"latency {np.mean(latencies) * 1000:.2f} ms over {len(latencies)} frames: {shares}"
    )


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly <component>=<path> for {', '.join(COMPONENTS)}")
    return {name: Path(mpks[name]) for name in COMPONENTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="davis-2017-val=<extracted davis directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    # The benchmark runner passes the package's resources, here only the license. The benchmark reads none of them.
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_tracking(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
