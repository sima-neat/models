"""Benchmark Supertonic 3 with Neat Core and save its report.

Given --mpk vector_estimator=<path> --mpk vocoder=<path>, the script measures what the accelerator does for
one sentence at the longest length the model takes: eight denoising steps of the vector estimator, with the
classifier-free guidance and Euler update on the CPU between them, then the vocoder. A batch-2 vector
estimator runs both guidance branches of a step in one call; a batch-1 one runs them as two calls. Each
sentence depends on its own previous step, so sentences run one after another and throughput is sentences
per second. It also records each MPK's MLA layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat

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


def build_runner(model: pyneat.Model, seeds: list[np.ndarray]) -> pyneat.ModelRunner:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = 2
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.Owned
    return model.build([tensor(seed) for seed in seeds], run_options=run_options)


class Sentence:
    """One full-length sentence's accelerator work, with seeded synthetic inputs."""

    def __init__(self, models: dict[str, pyneat.Model]) -> None:
        rng = np.random.default_rng(1101)
        self.batch = int(models["vector_estimator"].compiled_batch_size())
        if self.batch not in (1, 2):
            raise SystemExit(f"the vector estimator must be compiled at batch 1 or 2, not {self.batch}")
        # Branch 0 is conditional, branch 1 unconditional: they differ in text and style.
        self.branches = [
            {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in VECTOR_INPUTS.items()}
            for _ in range(2)
        ]
        for branch in self.branches:
            branch["latent_mask"].fill(1.0)
            branch["text_mask"].fill(1.0)
        self.latent = rng.standard_normal(LATENT, dtype=np.float32)
        self.vector = build_runner(models["vector_estimator"], self._vector_inputs(self.latent, 0)[0])
        self.vocoder = build_runner(models["vocoder"], [self.latent])

    def _vector_inputs(self, latent: np.ndarray, step: int) -> list[list[np.ndarray]]:
        """The inputs of each call of this step: one call with both branches stacked, or one per branch."""
        calls = []
        for branch in self.branches:
            branch["noisy_latent"] = latent
            branch["time_sinusoidal"] = np.full(VECTOR_INPUTS["time_sinusoidal"], step / STEPS, np.float32)
            calls.append([branch[name] for name in VECTOR_INPUTS])
        if self.batch == 2:
            return [[np.stack(pair) for pair in zip(*calls, strict=True)]]
        return calls

    def run(self) -> None:
        latent = self.latent
        for step in range(STEPS):
            velocities = []
            for inputs in self._vector_inputs(latent, step):
                result = self.vector.run([tensor(value) for value in inputs], timeout_ms=TIMEOUT_MS)
                velocities.extend(first_output(result).reshape(-1, *LATENT))
            conditional, unconditional = velocities
            latent = (latent + (4.0 * conditional - 3.0 * unconditional) / STEPS).astype(np.float32)
        self.vocoder.run([tensor(latent)], timeout_ms=TIMEOUT_MS)

    def close(self) -> None:
        self.vector.close()
        self.vocoder.close()


def measure(sentence: Sentence) -> tuple[list[float], float, pyneat.Telemetry]:
    """Seconds per sentence, and the span and telemetry of the measured sentences, after the warm-up."""
    for _ in range(WARMUP_SENTENCES):
        sentence.run()
    monitor = pyneat.TelemetryMonitor()
    monitor.start()
    try:
        seconds: list[float] = []
        started = time.perf_counter()
        while len(seconds) < MEASURED_SENTENCES or time.perf_counter() - started < MIN_DURATION_S:
            begun = time.perf_counter()
            sentence.run()
            seconds.append(time.perf_counter() - begun)
        span = time.perf_counter() - started
    finally:
        telemetry = monitor.stop()
    return seconds, span, telemetry


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpk)) for name, mpk in mpks.items()}
    sentence = Sentence(models)
    try:
        latencies, span, telemetry = measure(sentence)
    finally:
        sentence.close()
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = percentile_ms(latencies, 50)
    benchmark.latency_p95_ms = percentile_ms(latencies, 95)
    benchmark.latency_p99_ms = percentile_ms(latencies, 99)
    benchmark.fps = len(latencies) / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = len(latencies)
    calls = "one batch-2 call" if sentence.batch == 2 else "two batch-1 calls"
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"One {LATENT_LENGTH}-frame sentence: {STEPS} vector-estimator steps of {calls} each with guidance and "
            "the Euler update on the CPU, then the vocoder; throughput is sentences per second"
        ),
        telemetry=telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(f"{benchmark.fps:.2f} sentences/s over {len(latencies)} sentences in {span:.1f} s; "
          f"latency {benchmark.latency_ms:.1f} ms")


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or list(mpks) != ["vector_estimator", "vocoder"]:
        raise SystemExit("--mpk must name vector_estimator=<path> then vocoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
