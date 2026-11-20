"""Benchmark DiT-XL/2 256 with Neat Core and save its report.

Given one --mpk <component>=<path> per component (transformer_0 to transformer_3, vae_decoder), the script measures
the whole model generating one image: a seeded latent, 50 DDIM steps that each run the four transformer parts for
the class and again for the null class, with classifier-free guidance and the DDIM update on the CPU, then the VAE
decoder. One inference is one generated 256x256 image, so throughput is in images per second. The script also
records each MPK's MLA layer and stage stats and its execution plan.

The transformer parts run as a pipeline: each part's runner has its own thread, which pulls its output, copies it
once into the next part's input and pushes it on, so the MLA does not wait for the CPU between calls. A step's class
and null-class passes travel through the parts back to back, and the CPU guides and steps the latent when both
arrive. The latency pass generates one image at a time; the throughput pass keeps THROUGHPUT_IMAGES independent
images in flight, so one image's CPU work overlaps another's MLA work. Every image is the same as sequential calls
would give.

The input is fixed: class 207 (golden retriever), seed 0, guidance scale 4, as the model's README describes. The
model takes no image, so it has no accuracy dataset to draw one from.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import model_report
import numpy as np
import pyneat
from dit import COMPONENTS, LATENT_CHANNELS, LATENT_SIZE, TRANSFORMER, class_one_hot, timestep_features

STEPS = 50
GUIDANCE_SCALE = 4.0
CLASS_LABEL = 207
SEED = 0
# DiT's training schedule: 1000 linear betas from 1e-4 to 0.02.
TRAIN_TIMESTEPS, BETA_START, BETA_END = 1000, 1e-4, 0.02
WARMUP_IMAGES = 1
# Each image takes about 9 s, so a pass is bounded by its duration rather than its count.
MEASURED_IMAGES = 4
MIN_DURATION_S = 60
# Images in flight during the throughput pass: two keep the MLA busy; more only lengthen each image.
THROUGHPUT_IMAGES = 2
# Jobs each runner may queue: both guidance passes of every image in flight.
QUEUE_DEPTH = 4
TIMEOUT_MS = 60000
PREDICTION_CHANNELS = 8


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: the script prepares their inputs; Neat still quantizes and tessellates them."""
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


def hwc_array(value: np.ndarray) -> pyneat.Tensor:
    """An [H, W, C] array as Neat's input tensor."""
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(value), copy=True, layout=pyneat.TensorLayout.HWC)


def handoff(tensor: pyneat.Tensor) -> pyneat.Tensor:
    """A transformer part's output as the next part's input, copied once out of the runner's buffer. An output
    passed on as is has media caps the next model's input stage rejects; the fresh tensor wraps the copy without a
    second one. The layout is stated: Neat reads an unlabelled 3D array as CHW."""
    array = tensor.to_numpy(copy=True)
    array = np.ascontiguousarray(array[0] if array.ndim == 4 else array)
    return pyneat.Tensor.from_numpy(array, copy=False, layout=pyneat.TensorLayout.HWC)


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


@dataclass(frozen=True)
class Schedule:
    """The DDIM timesteps, from 999 to 0, and the cumulative alphas of DiT's training schedule."""

    timesteps: np.ndarray
    alphas: np.ndarray

    @classmethod
    def ddim(cls, steps: int) -> Schedule:
        betas = np.linspace(BETA_START, BETA_END, TRAIN_TIMESTEPS, dtype=np.float64)
        timesteps = np.linspace(TRAIN_TIMESTEPS - 1, 0, steps).round().astype(int)
        return cls(timesteps, np.cumprod(1 - betas))

    def step(self, index: int, latent: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """The deterministic DDIM update from timesteps[index] to the next timestep, or to the clean latent:
        sqrt(a') * (x - sqrt(1 - a) * n) / sqrt(a) + sqrt(1 - a') * n, as one scale of x plus one of n."""
        alpha = self.alphas[self.timesteps[index]]
        following = self.alphas[self.timesteps[index + 1]] if index + 1 < len(self.timesteps) else 1.0
        latent_scale = np.float32(np.sqrt(following / alpha))
        noise_scale = np.float32(np.sqrt(1 - following) - np.sqrt(following * (1 - alpha) / alpha))
        stepped = latent * latent_scale
        stepped += noise_scale * noise
        return stepped.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Conditioning:
    """Each step's timestep features and the class one-hot, as Neat tensors for both guidance slots."""

    timesteps: list[tuple[pyneat.Tensor, pyneat.Tensor]]
    classes: tuple[pyneat.Tensor, pyneat.Tensor]

    @classmethod
    def of(cls, schedule: Schedule, label: int) -> Conditioning:
        one_hot = class_one_hot(label)
        steps = []
        for timestep in schedule.timesteps:
            features = timestep_features(float(timestep))
            steps.append((hwc(features[0:1]), hwc(features[1:2])))
        return cls(steps, (hwc(one_hot[0:1]), hwc(one_hot[1:2])))


def initial_latent() -> np.ndarray:
    shape = (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    return np.random.RandomState(SEED).standard_normal(shape).astype(np.float32)


def guided(conditional: np.ndarray, unconditional: np.ndarray) -> np.ndarray:
    """Classifier-free-guided noise from the two passes' [H, W, 8] predictions; noise is channels 0-3."""
    noise = unconditional[..., :LATENT_CHANNELS]
    return noise + GUIDANCE_SCALE * (conditional[..., :LATENT_CHANNELS] - noise)


def build_runners(models: dict[str, pyneat.Model], conditioning: Conditioning) -> dict[str, pyneat.ModelRunner]:
    """A queued runner for each component, each transformer part built from one real output of the part before."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runners: dict[str, pyneat.ModelRunner] = {}
    latent = hwc(initial_latent())
    value = latent
    for part in TRANSFORMER:
        inputs = [value, conditioning.timesteps[0][0], conditioning.classes[0]]
        runners[part] = models[part].build(inputs, run_options=run_options)
        value = handoff(runners[part].run(inputs, timeout_ms=TIMEOUT_MS)[0])
    runners["vae_decoder"] = models["vae_decoder"].build([latent], run_options=run_options)
    return runners


@dataclass
class Generation:
    """One image's denoising state: its [H, W, 4] latent, its step, and the passes of that step returned so far."""

    latent: np.ndarray
    index: int = 0
    predictions: dict[int, np.ndarray] = field(default_factory=dict)


class Pipeline:
    """The transformer parts as a pipeline: one thread per part forwards each job to the next part, and the last
    part's predictions return to the caller's thread, which guides, steps and decodes. Jobs are (image, step, slot);
    each runner keeps them in order, so a FIFO of job metadata travels beside each runner's queue."""

    def __init__(self, runners: dict[str, pyneat.ModelRunner], schedule: Schedule, conditioning: Conditioning):
        self.parts = [runners[part] for part in TRANSFORMER]
        self.decoder = runners["vae_decoder"]
        self.schedule, self.conditioning = schedule, conditioning
        self.jobs: list[queue.Queue] = [queue.Queue() for _ in self.parts]
        self.returned: queue.Queue = queue.Queue()
        self.errors: list[BaseException] = []
        self.threads = [threading.Thread(target=self._forward, args=(stage,), daemon=True)
                        for stage in range(len(self.parts))]
        for thread in self.threads:
            thread.start()

    def _push(self, stage: int, value: pyneat.Tensor, job: tuple[int, int, int]) -> None:
        _, index, slot = job
        self.jobs[stage].put(job)
        inputs = [value, self.conditioning.timesteps[index][slot], self.conditioning.classes[slot]]
        if not self.parts[stage].push(inputs):
            raise RuntimeError(f"{TRANSFORMER[stage]} rejected a job")

    def _forward(self, stage: int) -> None:
        try:
            while (job := self.jobs[stage].get()) is not None:
                tensors = self.parts[stage].pull(TIMEOUT_MS).tensors
                if not tensors:
                    raise RuntimeError(f"{TRANSFORMER[stage]} returned nothing within {TIMEOUT_MS} ms")
                if stage + 1 < len(self.parts):
                    self._push(stage + 1, handoff(tensors[0]), job)
                else:
                    prediction = tensors[0].to_numpy(copy=True).reshape(LATENT_SIZE, LATENT_SIZE, PREDICTION_CHANNELS)
                    self.returned.put((job[0], job[2], prediction))
                # Return the runner's output buffers before the next pull.
                del tensors
        except BaseException as error:
            self.errors.append(error)
            self.returned.put(None)

    def _start_step(self, image: int, generation: Generation) -> None:
        for slot in range(2):
            self._push(0, hwc_array(generation.latent), (image, generation.index, slot))

    def generate(self, count: int, in_flight: int) -> tuple[list[float], np.ndarray]:
        """count images with up to in_flight at a time. Returns each image's seconds from start to decoded image,
        and the last image, [1, 3, 256, 256] in [-1, 1]."""
        start_latent = initial_latent()[0].transpose(1, 2, 0).copy()
        active: dict[int, Generation] = {}
        started_at: dict[int, float] = {}
        seconds: list[float] = []
        image = np.empty(0)
        next_image = 0

        def start() -> None:
            nonlocal next_image
            active[next_image] = Generation(start_latent.copy())
            started_at[next_image] = time.perf_counter()
            self._start_step(next_image, active[next_image])
            next_image += 1

        for _ in range(min(in_flight, count)):
            start()
        while len(seconds) < count:
            item = self.returned.get(timeout=TIMEOUT_MS / 1000)
            if item is None:
                raise self.errors[0]
            number, slot, prediction = item
            generation = active[number]
            generation.predictions[slot] = prediction
            if len(generation.predictions) < 2:
                continue
            noise = guided(generation.predictions[0], generation.predictions[1])
            generation.latent = self.schedule.step(generation.index, generation.latent, noise)
            generation.predictions, generation.index = {}, generation.index + 1
            if generation.index < len(self.schedule.timesteps):
                self._start_step(number, generation)
                continue
            # The decoder divides by the VAE's latent scale itself.
            decoded = self.decoder.run([hwc_array(generation.latent)], timeout_ms=TIMEOUT_MS)[0]
            array = decoded.to_numpy(copy=True)
            image = np.ascontiguousarray((array if array.ndim == 4 else array[None]).transpose(0, 3, 1, 2))
            seconds.append(time.perf_counter() - started_at.pop(number))
            del active[number]
            if next_image < count:
                start()
        return seconds, image

    def close(self) -> None:
        self.jobs[0].put(None)
        for stage, thread in enumerate(self.threads):
            thread.join(timeout=TIMEOUT_MS / 1000)
            if stage + 1 < len(self.jobs):
                self.jobs[stage + 1].put(None)


def timed_images(pipeline: Pipeline, in_flight: int) -> tuple[list[float], float, np.ndarray]:
    """Images in batches of in_flight until at least MEASURED_IMAGES and MIN_DURATION_S: each image's seconds, the
    pass's wall-clock seconds, and the last image."""
    seconds: list[float] = []
    image = np.empty(0)
    started = time.perf_counter()
    while len(seconds) < MEASURED_IMAGES or time.perf_counter() - started < MIN_DURATION_S:
        batch, image = pipeline.generate(2 * in_flight, in_flight)
        seconds += batch
    return seconds, time.perf_counter() - started, image


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the images generated, how long they took, and the board telemetry
    sampled over the same span."""

    images: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.images / self.seconds


def measure_throughput(pipeline: Pipeline) -> Throughput:
    """Images per second and board telemetry, with THROUGHPUT_IMAGES images in flight."""
    monitor = pyneat.TelemetryMonitor()
    monitor.start()
    try:
        seconds, elapsed, image = timed_images(pipeline, THROUGHPUT_IMAGES)
        check_image(image)
        return Throughput(len(seconds), elapsed, monitor.stop())
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()


def check_image(image: np.ndarray) -> None:
    """A generated image is finite and spans the decoder's range, so a broken build does not pass as fast."""
    if image.shape != (1, 3, 256, 256) or not np.isfinite(image).all() or float(image.std()) < 0.05:
        raise RuntimeError(f"the decoder produced no image: shape {image.shape}, std {float(image.std()):.4f}")


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
    benchmark.throughput_samples = throughput.images
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"One 256x256 image per inference: class {CLASS_LABEL}, seed {SEED}, {STEPS} DDIM steps, each the four "
            f"transformer parts for the class and the null class with guidance {GUIDANCE_SCALE:g} and the DDIM update "
            f"on the CPU, then the VAE decoder; the parts pipelined, {THROUGHPUT_IMAGES} images in flight"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.4f} images/s over {throughput.images} images in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms / 1000:.2f} s over {len(latencies)} images"
    )


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpks[name]), tensor_options()) for name in COMPONENTS}
    schedule = Schedule.ddim(STEPS)
    conditioning = Conditioning.of(schedule, CLASS_LABEL)
    runners = build_runners(models, conditioning)
    pipeline = Pipeline(runners, schedule, conditioning)
    try:
        _, image = pipeline.generate(WARMUP_IMAGES, 1)
        check_image(image)
        latencies, _, image = timed_images(pipeline, 1)
        check_image(image)
        throughput = measure_throughput(pipeline)
    finally:
        pipeline.close()
        for runner in runners.values():
            runner.close()
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", help="unused: the model takes no image, so its input is fixed")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
