"""Public-Neat LeWAM execution and whole-replan measurement, not campaign FPS.

This standalone board script has no model_registry or YAML dependency. Inputs
are a Dataset Registry request pack: real RGB224 observation/goal images, the
original noise, source-generated time/horizon features and explicit history padding.
Image normalization is timed on Neat's image route. Camera acquisition, image
resizing before RGB224, and physics are excluded and named in the report.
It remains undeclared in model.yaml until this input protocol is qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from pathlib import Path

import numpy as np

COMPONENTS = ("front", "body", "head", "policy")
WARMUPS, SAMPLES, MIN_SECONDS, TIMEOUT_MS = 100, 1000, 60.0, 10000
ACTION_MEAN = np.array([-0.007812565192580223, 0.006860686466097832], np.float32)
ACTION_STD = np.array([0.2084674835205078, 0.20674866437911987], np.float32)
MASK_SENTINEL = np.float32(-(2**24))
OUTPUTS = {
    "front": (1, 64, 51, 51),
    "body": (1, 512, 7, 7),
    "head": (1, 192, 1, 1),
    "policy": (1, 192, 1, 10),
}


def attention_mask(padded: bool) -> np.ndarray:
    """Original full19 topology, transposed to the packed key-first ABI."""
    attends = np.zeros((19, 19), np.bool_)
    attends[:2, :2] = True
    attends[2, :3] = True
    attends[2, 4:9] = True
    attends[3, :2] = True
    attends[3, 3] = True
    for row in range(4, 9):
        attends[row, :2] = True
        attends[row, 4 : row + 1] = True
    for row in range(9, 19):
        attends[row, :2] = True
        attends[row, 3] = True
        attends[row, 9 : row + 1] = True
    if padded:
        attends[:, 0] = False
    return np.where(attends.T, np.float32(0), np.float32(-np.inf)).astype(np.float32)


def finite(value: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite float32 {shape}")
    return value


def history_padding(value) -> np.ndarray:
    pad = np.asarray(value)
    if pad.shape != (2,) or pad.dtype != np.bool_ or pad[1]:
        raise ValueError("Current observation must be valid; only the older history may be padded")
    return pad


def validate_request(request: dict) -> dict[str, np.ndarray]:
    """Normalize admitted arrays before any native runner work; never coerce dtypes."""
    if set(request) != {"images", "history_pad", "noise", "time_features", "horizon_features"}:
        raise ValueError("Request must contain exactly images, history_pad, noise and time/horizon features")
    pad = history_padding(request["history_pad"])
    images = np.asarray(request["images"])
    if images.shape != (int((~pad).sum()) + 1, 224, 224, 3) or images.dtype != np.uint8:
        raise ValueError("Images must be valid history observations followed by goal, RGB224 uint8")
    return {
        "images": images,
        "history_pad": pad,
        "noise": finite(request["noise"], (10, 2), "noise"),
        "time_features": finite(request["time_features"], (8, 192), "time features"),
        "horizon_features": finite(request["horizon_features"], (192,), "horizon features"),
    }


def integrate(run_policy, history: np.ndarray, goal: np.ndarray, request: dict) -> np.ndarray:
    """Eight own-output updates; no fused multiply-add and no shortened horizon."""
    history = finite(history, (1, 192, 1, 2), "history")
    goal = finite(goal, (1, 192, 1, 1), "goal")
    value = finite(request["noise"], (10, 2), "noise").T[None, :, None, :].copy()
    time_features = finite(request["time_features"], (8, 192), "time features")
    horizon = finite(request["horizon_features"], (192,), "horizon features")
    mask = attention_mask(bool(history_padding(request["history_pad"])[0]))
    packed = np.zeros((1, 192, 1, 34), np.float32)
    packed[:, :, :, :2] = history
    packed[:, :, :, 2:3] = goal
    packed[0, :, 0, 14] = horizon
    packed[0, :19, 0, 15:34] = np.where(np.isneginf(mask), MASK_SENTINEL, mask)
    for step in range(8):
        packed[:, :2, :, 3:13] = value
        packed[0, :, 0, 13] = time_features[step]
        result = finite(run_policy(packed), OUTPUTS["policy"], "policy velocity")
        if np.any(result[:, 2:, :, :]):
            raise ValueError("Policy returned nonzero padded channels")
        delta = np.divide(result[:, :2], np.float32(8), dtype=np.float32)
        value = np.add(value, delta, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite Euler feedback")
    actions = value[0, :, 0, :].T
    return np.add(np.multiply(actions, ACTION_STD, dtype=np.float32), ACTION_MEAN, dtype=np.float32)


def tensor_options(pn):
    options = pn.ModelOptions()
    options.preprocess.kind = pn.InputKind.Tensor
    options.preprocess.resize.enable = pn.AutoFlag.Off
    options.preprocess.color_convert.enable = pn.AutoFlag.Off
    options.preprocess.normalize.enable = pn.AutoFlag.Off
    return options


def image_options(pn):
    options = pn.ModelOptions()
    options.preprocess.kind = pn.InputKind.Image
    options.preprocess.enable = pn.AutoFlag.On
    options.preprocess.resize.enable = pn.AutoFlag.Off
    options.preprocess.normalize.enable = pn.AutoFlag.On
    options.preprocess.preset = pn.NormalizePreset.ImageNet
    return options


def as_nchw(tensor, shape: tuple) -> np.ndarray:
    value = tensor.to_numpy()
    nhwc = (shape[0], shape[2], shape[3], shape[1])
    if value.shape == nhwc and value.shape != shape:
        value = value.transpose(0, 3, 1, 2)
    return finite(value, shape, "public model output").copy()


class Policy:
    """Four public ModelRunners, one in-flight request, no cross-request cache.

    A timed-out/faulted call poisons this instance; it is never silently retried.
    All data is copied into owned Neat tensors, and outputs stay alive until the
    dependent synchronous call completes. Raw physical tile buffers are not
    reinterpreted as flat tensors. This baseline is not the private fast path.
    """

    def __init__(self, mpks: dict[str, Path], pn=None):
        if pn is None:
            import pyneat as pn
        if set(mpks) != set(COMPONENTS):
            raise ValueError("Four component MPKs are required")
        self.pn, self.runners, self.failed, self.closed = pn, {}, False, False
        self._request_lock = threading.Lock()
        self.models = {
            name: pn.Model(str(mpks[name]), image_options(pn) if name == "front" else tensor_options(pn))
            for name in COMPONENTS
        }

    def _call(self, name, tensor):
        if self.failed or self.closed:
            raise RuntimeError("Runner is failed or closed")
        try:
            if name not in self.runners:
                options = self.pn.RunOptions()
                options.queue_depth = 1
                options.overflow_policy = self.pn.OverflowPolicy.Block
                self.runners[name] = self.models[name].build([tensor], run_options=options)
            output = self.runners[name].run([tensor], timeout_ms=TIMEOUT_MS)
            if len(output) != 1:
                raise RuntimeError(f"{name}: expected one completed output")
            return output[0]
        except BaseException:
            self.failed = True
            raise

    def _policy(self, packed):
        tensor = self.pn.Tensor.from_numpy(
            np.ascontiguousarray(packed.transpose(0, 2, 3, 1)),
            copy=True,
            layout=self.pn.TensorLayout.NHWC,
            memory=self.pn.TensorMemory.EV74,
        )
        return as_nchw(self._call("policy", tensor), OUTPUTS["policy"])

    def replan(self, request: dict) -> np.ndarray:
        request = validate_request(request)
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Only one replan may be in flight")
        try:
            return self._replan(request)
        finally:
            self._request_lock.release()

    def _replan(self, request: dict) -> np.ndarray:
        if self.failed or self.closed:
            raise RuntimeError("Runner is failed or closed")
        try:
            latents = []
            for image in request["images"]:
                value = self.pn.Tensor.from_numpy(
                    image, copy=True, image_format=self.pn.PixelFormat.RGB, memory=self.pn.TensorMemory.EV74
                )
                for name in COMPONENTS[:3]:
                    value = self._call(name, value)
                latents.append(as_nchw(value, OUTPUTS["head"]))
            history = np.zeros((1, 192, 1, 2), np.float32)
            for index, latent in zip(np.flatnonzero(~request["history_pad"]), latents[:-1], strict=True):
                history[:, :, :, index : index + 1] = latent
            return integrate(self._policy, history, latents[-1], request)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Cannot close while a replan is in flight")
        try:
            self.closed = True
            errors = []
            # Retain failed closes for an explicit retry; never imply retirement.
            for name, runner in reversed(list(self.runners.items())):
                try:
                    runner.close()
                    del self.runners[name]
                except Exception as error:
                    errors.append(error)
            if errors:
                raise RuntimeError("One or more Neat runners failed to close") from errors[0]
        finally:
            self._request_lock.release()


def parse_mpks(values: list[str]) -> dict[str, Path]:
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(values):
        raise ValueError("Malformed or duplicate --mpk")
    paths = {name: Path(value) for name, value in pairs}
    if set(paths) != set(COMPONENTS) or any(
        not x.is_file() or not x.name.endswith("_mpk.tar.gz") for x in paths.values()
    ):
        raise ValueError("Exactly four existing standard component MPKs are required")
    return paths


def load_request(dataset: Path) -> tuple[dict, str]:
    manifest = json.loads((dataset / "requests.json").read_text())
    if manifest.get("schema") != "lewam.requests.v1" or manifest.get("source") != "source_trajectories":
        raise ValueError("A registered real-source RGB224 request pack is required")
    rows = manifest["requests"]
    if not rows or len({x["file"] for x in rows}) != len(rows):
        raise ValueError("Empty or duplicate request manifest")
    row = random.Random(0).choice(sorted(rows, key=lambda x: x["file"]))
    path = dataset / row["file"]
    if path.is_symlink() or path.parent.resolve() != dataset.resolve() or path.suffix != ".npz":
        raise ValueError("Unsafe request archive path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Request bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        request = dict(archive)
    return validate_request(request), path.name


def measure(infer, *, samples=SAMPLES, seconds=MIN_SECONDS, warmups=WARMUPS, clock=time.perf_counter):
    if samples < 1 or seconds < 0 or warmups < 0:
        raise ValueError("Invalid measurement window")
    for _ in range(warmups):
        infer()
    times = []
    start = clock()
    while len(times) < samples or clock() - start < seconds:
        before = clock()
        finite(infer(), (10, 2), "completed actions")
        times.append(clock() - before)
    elapsed = clock() - start
    if elapsed <= 0:
        raise ValueError("Nonpositive elapsed time")
    return times, elapsed


def mla_stats(mpk: Path) -> tuple[dict, dict]:
    """One MPK's idle profiling pass; these timings are not whole-replan FPS."""
    import pyneat

    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    stats = pyneat.Model(str(mpk)).benchmark(options)
    layers = {
        "status": stats.mla_layer_stats.status,
        "layers": [
            {"stage_index": x.stage_index, "name": x.name, "latency_us": x.latency_us}
            for x in stats.mla_layer_stats.layers
        ],
    }
    stages = {
        "status": stats.mla_stage_stats.status,
        "stages": [{"stage_index": x.stage_index, "latency_us": x.latency_us} for x in stats.mla_stage_stats.stages],
    }
    return layers, stages


def write_report(
    output: Path,
    models: dict,
    latencies: list[float],
    *,
    samples: int,
    duration: float,
    workload: str,
    telemetry,
    mla_layer_stats: dict,
    mla_stage_stats: dict,
) -> None:
    import model_report
    import pyneat

    result = pyneat.BenchmarkReport()
    result.latency_ms = float(np.mean(latencies) * 1000)
    result.latency_p50_ms, result.latency_p95_ms, result.latency_p99_ms = (
        float(np.percentile(latencies, q) * 1000) for q in (50, 95, 99)
    )
    result.fps = samples / duration
    result.throughput_duration_ms, result.throughput_samples = duration * 1000, samples
    model_report.from_performance(
        result,
        models,
        workload=workload,
        telemetry=telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    ).save(output)


def benchmark(mpks: dict, dataset_id: str, dataset: Path, output: Path) -> None:
    import pyneat

    request, filename = load_request(dataset)
    policy = Policy(mpks, pyneat)
    monitor = pyneat.TelemetryMonitor()
    try:
        latencies, _ = measure(lambda: policy.replan(request))
        for _ in range(WARMUPS):
            policy.replan(request)
        monitor.start()
        try:
            throughputs, duration = measure(lambda: policy.replan(request), warmups=0)
        finally:
            telemetry = monitor.stop()
    finally:
        policy.close()
    # Profiling is a separate idle pass after every inference runner closed.
    layers, stages = {}, {}
    for name, path in mpks.items():
        layers[name], stages[name] = mla_stats(path)
    write_report(
        output,
        policy.models,
        latencies,
        samples=len(throughputs),
        duration=duration,
        workload=f"{dataset_id}/{filename}; RGB224 + timed Neat ImageNet normalization; uncached images; "
        "front/body/head; eight own-feedback policy calls; ten actions; one replan in flight; "
        "excludes acquisition, upstream resize and physics; not campaign image-ready FPS",
        telemetry=telemetry,
        mla_layer_stats=layers,
        mla_stage_stats=stages,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True)
    parser.add_argument("--dataset", required=True, help="registered-source-id=extracted directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_id, directory = args.dataset.split("=", 1)
    benchmark(parse_mpks(args.mpk), dataset_id, Path(directory), args.output)


if __name__ == "__main__":
    main()
