"""Benchmark Supertonic 3 with Neat Core and save its report.

Given --mpk vector_estimator=<path> --mpk vocoder=<path> and --resources <path>, the script measures a sentence
at the longest length the model takes: the duration predictor and text encoder on the CPU, eight denoising steps
of the vector estimator, with the classifier-free guidance and Euler update on the CPU between them, then the
vocoder. A batch-2 vector estimator runs both guidance branches of a step in one call; a batch-1 one runs them as
two calls. The latency pass runs one sentence at a time. A sentence's steps depend on each other, but sentences
do not, so the throughput pass keeps several in flight with one thread per stage: the CPU front end, the vector
estimator with the update between its steps, and the vocoder. Throughput is sentences per second. It also
records each MPK's MLA layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import model_report
import numpy as np
import onnxruntime as ort
import pyneat

# A fixed sentence as long as the text limit allows, in its language tag.
SENTENCE = (
    "<en>The morning train to the coast leaves at seven, stops at every small town along the river, and reaches "
    "the harbor just before noon, when the fishing boats return and the market opens.</en>"
)
VOICE = "F1"
STEPS = 8
LATENT_LENGTH = 192
# Public HWC shapes of the vector estimator's inputs, in compiled order, and of its output.
VECTOR_INPUTS = {
    "noisy_latent": (1, LATENT_LENGTH, 144),
    "text_emb": (1, LATENT_LENGTH, 256),
    "style_ttl": (1, 50, 256),
    "style_key": (1, 50, 256),
    "latent_mask": (1, LATENT_LENGTH, 1),
    "text_mask": (1, 1, LATENT_LENGTH),
    "time_sinusoidal": (1, 1, 64),
    "rope_tables": (1, LATENT_LENGTH, 128),
}
LATENT = (1, LATENT_LENGTH, 144)
WARMUP_SENTENCES = 10
MEASURED_SENTENCES = 200
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Sentences the throughput pass keeps in flight; each has one step on the vector estimator at a time.
IN_FLIGHT = 4


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


def tensor(value: np.ndarray) -> pyneat.Tensor:
    # Neat uses HWC as the axis semantic for both HWC and batched NHWC tensors.
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(value, dtype=np.float32), copy=True,
                                    layout=pyneat.TensorLayout.HWC)


def first_output(result: object) -> np.ndarray:
    tensors = result if isinstance(result, (list, tuple)) else getattr(result, "tensors", None) or [result.tensor]
    return np.asarray(tensors[0].to_numpy(copy=False), dtype=np.float32)


def build_runner(model: pyneat.Model, seeds: list[pyneat.Tensor], queue_depth: int) -> pyneat.ModelRunner:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = queue_depth
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.Owned
    return model.build(seeds, run_options=run_options)


def cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


@dataclass
class InFlight:
    """One sentence on the vector estimator: its latent, the step it is on, and the velocities pulled so far."""

    text_emb: np.ndarray
    latent: np.ndarray
    step: int = 0
    velocities: list[np.ndarray] = field(default_factory=list)


class Pipeline:
    """The fixed text through the CPU graphs, then seeded synthetic accelerator inputs, on one runner per MPK."""

    def __init__(self, models: dict[str, pyneat.Model], resources: Path) -> None:
        rng = np.random.default_rng(1101)
        self.batch = int(models["vector_estimator"].compiled_batch_size())
        if self.batch not in (1, 2):
            raise SystemExit(f"the vector estimator must be compiled at batch 1 or 2, not {self.batch}")
        self.calls_per_step = 1 if self.batch == 2 else 2
        # Branch 0 is conditional, branch 1 unconditional: they differ in text and style.
        self.branches = [
            {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in VECTOR_INPUTS.items()}
            for _ in range(2)
        ]
        for branch in self.branches:
            branch["latent_mask"].fill(1.0)
            branch["text_mask"].fill(1.0)
        supertone = resources / "supertonic-3"
        indexer = json.loads((supertone / "onnx/unicode_indexer.json").read_text(encoding="utf-8"))
        voice = json.loads((supertone / f"voice_styles/{VOICE}.json").read_text(encoding="utf-8"))
        style_ttl, style_dp = (
            np.asarray(voice[name]["data"], dtype=np.float32).reshape(voice[name]["dims"])
            for name in ("style_ttl", "style_dp")
        )
        text_ids = np.asarray([[indexer[ord(character)] for character in SENTENCE]], dtype=np.int64)
        if text_ids.shape[1] > LATENT_LENGTH:
            raise SystemExit(f"the sentence has {text_ids.shape[1]} characters; the limit is {LATENT_LENGTH}")
        text_mask = np.ones((1, 1, text_ids.shape[1]), dtype=np.float32)
        self.duration_inputs = {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask}
        self.text_inputs = {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask}
        self.duration = cpu_session(supertone / "onnx/duration_predictor.onnx")
        self.text = cpu_session(supertone / "onnx/text_encoder.onnx")
        self.branches[0]["style_ttl"] = style_ttl
        self.latent = rng.standard_normal(LATENT, dtype=np.float32)
        # Every in-flight sentence may have one step's calls queued, so a push never waits on a full queue.
        depth = IN_FLIGHT * self.calls_per_step
        seeds = self.vector_inputs(self.branches[0]["text_emb"], self.latent, 0)[0]
        self.vector = build_runner(models["vector_estimator"], seeds, depth)
        self.vocoder = build_runner(models["vocoder"], [tensor(self.latent)], IN_FLIGHT)

    def front_end(self) -> np.ndarray:
        """The CPU graphs, returning the padded text embedding. The compiled length is fixed, so the predicted
        duration changes no later work."""
        self.duration.run(None, self.duration_inputs)
        text_emb = self.text.run(None, self.text_inputs)[0].transpose(0, 2, 1)
        padded = np.zeros(VECTOR_INPUTS["text_emb"], dtype=np.float32)
        padded[:, :text_emb.shape[1]] = text_emb
        return padded

    def vector_inputs(self, text_emb: np.ndarray, latent: np.ndarray, step: int) -> list[list[pyneat.Tensor]]:
        """The inputs of each call of this step: one call with both branches stacked, or one per branch."""
        time_input = np.full(VECTOR_INPUTS["time_sinusoidal"], step / STEPS, np.float32)
        calls = []
        for index, branch in enumerate(self.branches):
            values = {**branch, "noisy_latent": latent, "time_sinusoidal": time_input}
            if index == 0:
                values["text_emb"] = text_emb
            calls.append([values[name] for name in VECTOR_INPUTS])
        if self.batch == 2:
            calls = [[np.stack(pair) for pair in zip(*calls, strict=True)]]
        return [[tensor(value) for value in call] for call in calls]

    @staticmethod
    def update(latent: np.ndarray, velocities: list[np.ndarray]) -> np.ndarray:
        conditional, unconditional = velocities
        return (latent + (4.0 * conditional - 3.0 * unconditional) / STEPS).astype(np.float32)

    def run(self) -> None:
        """One sentence with nothing else in flight."""
        text_emb = self.front_end()
        latent = self.latent
        for step in range(STEPS):
            velocities = []
            for inputs in self.vector_inputs(text_emb, latent, step):
                result = self.vector.run(inputs, timeout_ms=TIMEOUT_MS)
                velocities.extend(first_output(result).reshape(-1, *LATENT))
            latent = self.update(latent, velocities)
        self.vocoder.run([tensor(latent)], timeout_ms=TIMEOUT_MS)

    def close(self) -> None:
        self.vector.close()
        self.vocoder.close()


def measure_latency(pipeline: Pipeline) -> list[float]:
    """Seconds per sentence with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_SENTENCES):
        pipeline.run()
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_SENTENCES or time.perf_counter() < ends:
        started = time.perf_counter()
        pipeline.run()
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the sentences the vocoder output, how long they took, and the
    board telemetry sampled over the same span."""

    sentences: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def rate(self) -> float:
        return self.sentences / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"the {name} rejected an input")


def pull(runner: pyneat.ModelRunner, name: str) -> np.ndarray:
    tensors = runner.pull(TIMEOUT_MS).tensors
    if not tensors:
        raise RuntimeError(f"the {name} returned nothing within {TIMEOUT_MS} ms")
    return np.asarray(tensors[0].to_numpy(copy=False), dtype=np.float32)


def run_together(*stages: Callable[[], None], stop: threading.Event, runners: tuple[pyneat.ModelRunner, ...]) -> None:
    """Run each stage on its own thread. The first failure stops every stage and ends every runner's input,
    and is raised once all stages stop. Each stage waits in Core or on a queue with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            stop.set()
            for runner in runners:
                runner.close_input()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def measure_throughput(pipeline: Pipeline) -> Throughput:
    """Sentences per second and board telemetry with up to IN_FLIGHT sentences in flight. The measured span starts
    after WARMUP_SENTENCES outputs and lasts at least MEASURED_SENTENCES sentences and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    stop = threading.Event()
    # Text embeddings from the CPU front end, then None once the front end stops.
    ready: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=IN_FLIGHT)
    finished = {"vocoder_pushed": 0}
    throughput: Throughput | None = None

    def front_end() -> None:
        try:
            while not stop.is_set():
                text_emb = pipeline.front_end()
                while not stop.is_set():
                    try:
                        ready.put(text_emb, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            # A failed denoise stage reads no more, so the end marker must not wait for room forever.
            try:
                ready.put(None, timeout=TIMEOUT_MS / 1000)
            except queue.Full:
                pass

    def denoise() -> None:
        # Calls come back in the order they were pushed, so each pulled output belongs to the oldest call.
        pending: deque[InFlight] = deque()
        in_flight, accepting = 0, True

        def start(sentence: InFlight) -> None:
            for inputs in pipeline.vector_inputs(sentence.text_emb, sentence.latent, sentence.step):
                push(pipeline.vector, inputs, "vector estimator")
                pending.append(sentence)

        try:
            while accepting or in_flight:
                while accepting and in_flight < IN_FLIGHT:
                    try:
                        text_emb = ready.get(block=in_flight == 0, timeout=None if in_flight == 0 else 0)
                    except queue.Empty:
                        break
                    if text_emb is None:
                        accepting = False
                        break
                    start(InFlight(text_emb, pipeline.latent))
                    in_flight += 1
                if not in_flight:
                    continue
                sentence = pending.popleft()
                sentence.velocities.extend(pull(pipeline.vector, "vector estimator").reshape(-1, *LATENT))
                if len(sentence.velocities) < 2:
                    continue
                sentence.latent = pipeline.update(sentence.latent, sentence.velocities)
                sentence.velocities, sentence.step = [], sentence.step + 1
                if sentence.step < STEPS:
                    start(sentence)
                else:
                    push(pipeline.vocoder, [tensor(sentence.latent)], "vocoder")
                    finished["vocoder_pushed"] += 1
                    in_flight -= 1
        finally:
            pipeline.vocoder.close_input()

    def vocode() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        while tensors := pipeline.vocoder.pull(TIMEOUT_MS).tensors:
            del tensors
            outputs += 1
            if outputs == WARMUP_SENTENCES:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_SENTENCES and throughput is None:
                sentences, seconds = outputs - WARMUP_SENTENCES, time.perf_counter() - started
                if sentences >= MEASURED_SENTENCES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(sentences, seconds, monitor.stop())
                    stop.set()
        if outputs != finished["vocoder_pushed"]:
            raise RuntimeError(f"the vocoder stalled: {outputs} of {finished['vocoder_pushed']} sentences came out")
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    try:
        run_together(front_end, denoise, vocode, stop=stop, runners=(pipeline.vector, pipeline.vocoder))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
    return throughput


def benchmark_pipeline(mpks: dict[str, Path], resources: Path, output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpk)) for name, mpk in mpks.items()}
    pipeline = Pipeline(models, resources)
    try:
        latencies = measure_latency(pipeline)
        throughput = measure_throughput(pipeline)
    finally:
        pipeline.close()
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = percentile_ms(latencies, 50)
    benchmark.latency_p95_ms = percentile_ms(latencies, 95)
    benchmark.latency_p99_ms = percentile_ms(latencies, 99)
    benchmark.fps = throughput.rate
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.sentences
    calls = "one batch-2 call" if pipeline.batch == 2 else "two batch-1 calls"
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"A {LATENT_LENGTH}-frame sentence: the duration predictor and text encoder on the CPU, {STEPS} "
            f"vector-estimator steps of {calls} each with guidance and the Euler update on the CPU, then the "
            f"vocoder. Latency is one sentence at a time; throughput, in sentences per second, keeps up to "
            f"{IN_FLIGHT} sentences in flight with one thread per stage"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(f"{benchmark.fps:.2f} sentences/s over {throughput.sentences} sentences in {throughput.seconds:.1f} s; "
          f"latency {benchmark.latency_ms:.1f} ms")


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or list(mpks) != ["vector_estimator", "vocoder"]:
        raise SystemExit("--mpk must name vector_estimator=<path> then vocoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--resources", required=True, help="directory holding the package's supertonic-3 folder")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.resources), Path(args.output))


if __name__ == "__main__":
    main()
