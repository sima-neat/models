"""Independent host oracles for LeWAM boundaries; no compiler/board qualification."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "front": ((1, 3, 224, 224), (1, 64, 51, 51)),
    "body": ((1, 64, 51, 51), (1, 512, 7, 7)),
    "head": ((1, 512, 7, 7), (1, 192, 1, 1)),
    "policy": ((1, 192, 1, 34), (1, 192, 1, 10)),
}
# Allowed key bitsets for each of the source policy's nineteen query tokens.
# This fixed table deliberately does not call the implementation's mask builder.
ALLOWED_KEYS = (
    3,
    3,
    503,
    11,
    19,
    51,
    115,
    243,
    499,
    523,
    1547,
    3595,
    7691,
    15883,
    32267,
    65035,
    130571,
    261643,
    523787,
)
ACTION_MEAN = np.array([-0.007812565192580223, 0.006860686466097832], np.float32)
ACTION_STD = np.array([0.2084674835205078, 0.20674866437911987], np.float32)


@pytest.fixture
def model_dir() -> Path:
    # Reuse the repository's load_script fixture while this model remains staged.
    return ROOT / "contrib/lewam_pusht/model"


@pytest.fixture
def request_data() -> dict:
    return {
        "images": np.zeros((2, 224, 224, 3), np.uint8),
        "history_pad": np.array([True, False]),
        "noise": np.arange(20, dtype=np.float32).reshape(10, 2) / 32,
        "time_features": np.arange(8 * 192, dtype=np.float32).reshape(8, 192),
        "horizon_features": np.arange(192, dtype=np.float32) + 100,
    }


def test_export_and_runtime_interfaces_match_an_independent_contract(load_script):
    validation, runtime = load_script("validation"), load_script("performance")
    assert validation.SHAPES == SHAPES
    assert runtime.OUTPUTS == {name: shapes[1] for name, shapes in SHAPES.items()}
    assert tuple(validation.COMPONENTS) == tuple(runtime.COMPONENTS) == tuple(SHAPES)


@pytest.mark.parametrize("padded", [True, False])
def test_every_packed_token_and_action_normalization(load_script, request_data, padded):
    runtime = load_script("performance")
    request_data["history_pad"] = np.array([padded, False])
    history = np.arange(384, dtype=np.float32).reshape(1, 192, 1, 2)
    if padded:
        history[..., 0] = 0
    goal = np.arange(192, dtype=np.float32).reshape(1, 192, 1, 1) + 400
    packed = np.zeros((1, 192, 1, 34), np.float32)
    packed[..., :2], packed[..., 2:3] = history, goal
    packed[0, :, 0, 14] = request_data["horizon_features"]
    for query, keys in enumerate(ALLOWED_KEYS):
        for key in range(19):
            allowed = bool(keys & (1 << key)) and not (padded and key == 0)
            packed[0, key, 0, 15 + query] = 0 if allowed else -(2**24)
    calls = 0

    def velocity(actual):
        nonlocal calls
        packed[0, :2, 0, 3:13] = (request_data["noise"] + np.float32(calls / 4)).T
        packed[0, :, 0, 13] = request_data["time_features"][calls]
        np.testing.assert_array_equal(actual, packed)
        calls += 1
        result = np.zeros((1, 192, 1, 10), np.float32)
        result[:, :2] = 2
        return result

    actions = runtime.integrate(velocity, history, goal, request_data)
    assert calls == 8
    np.testing.assert_array_equal(runtime.ACTION_MEAN, ACTION_MEAN)
    np.testing.assert_array_equal(runtime.ACTION_STD, ACTION_STD)
    expected = np.add(np.multiply(request_data["noise"] + np.float32(2), ACTION_STD), ACTION_MEAN)
    np.testing.assert_array_equal(actions, expected)


def test_actual_models_receive_image_and_tensor_options(load_script, request_data):
    runtime = load_script("performance")
    received = {}

    def model(path, options):
        received[Path(path).name] = options.preprocess
        return SimpleNamespace()

    pn = SimpleNamespace(
        Model=model,
        ModelOptions=lambda: SimpleNamespace(
            preprocess=SimpleNamespace(
                resize=SimpleNamespace(), normalize=SimpleNamespace(), color_convert=SimpleNamespace()
            )
        ),
        InputKind=SimpleNamespace(Image="image", Tensor="tensor"),
        AutoFlag=SimpleNamespace(On="on", Off="off"),
        NormalizePreset=SimpleNamespace(ImageNet="imagenet"),
    )
    policy = runtime.Policy({name: Path(name) for name in SHAPES}, pn)
    assert set(received) == set(SHAPES)
    image = received["front"]
    assert (image.kind, image.enable, image.resize.enable, image.normalize.enable, image.preset) == (
        "image",
        "on",
        "off",
        "on",
        "imagenet",
    )
    for name in ("body", "head", "policy"):
        tensor = received[name]
        assert (tensor.kind, tensor.resize.enable, tensor.color_convert.enable, tensor.normalize.enable) == (
            "tensor",
            "off",
            "off",
            "off",
        )
    request_data["history_pad"] = [True, False]
    policy._replan = lambda normalized: normalized["history_pad"]
    normalized = policy.replan(request_data)
    assert isinstance(normalized, np.ndarray) and normalized.dtype == np.bool_
    np.testing.assert_array_equal(normalized, [True, False])
    policy.close()


@pytest.mark.parametrize("failure", [None, "quantize", "compile"])
def test_actual_compiler_call_preserves_bf16_and_failure_contract(load_script, monkeypatch, tmp_path, failure):
    compiler = load_script("compile")
    calls, settings = {}, []
    values = np.arange(2 * 34 * 192, dtype=np.float32).reshape(2, 1, 34, 192)

    class Config:
        def __getattr__(self, name):
            def configure(value):
                settings.append((name, value))
                return self

            return configure

    config = Config()

    def quantize(**kwargs):
        calls["quantize"] = kwargs
        if failure == "quantize":
            raise SystemExit(0)
        return SimpleNamespace(compile=compile_model)

    def compile_model(**kwargs):
        calls["compile"] = kwargs
        if failure == "compile":
            raise SystemExit(0)

    def read_calibration(directory, component, count):
        assert (directory, component, count) == (tmp_path / "calibration", "policy", 2)
        return values

    def onnx_source(path, **kwargs):
        calls["source"] = (path, kwargs)
        return "verified-source"

    def load_model(source, **kwargs):
        calls["load"] = (source, kwargs)
        return SimpleNamespace(quantize=quantize)

    modules = {
        "afe.apis.defines": dict(
            BiasCorrectionType=SimpleNamespace(NONE="none"),
            HistogramMSEMethod=lambda bins: ("mse", bins),
            InputName=lambda name: ("input", name),
            RequantizationMode=SimpleNamespace(sima="sima"),
            bfloat16_scheme=lambda: "bf16",
            default_quantization=config,
            gen2_target="gen2",
        ),
        "afe.apis.loaded_net": {"load_model": load_model},
        "afe.core.utils": {"convert_data_generator_to_iterable": lambda data: ("iterable", data)},
        "afe.load.importers.general_importer": {"onnx_source": onnx_source},
        "sima_utils.data.data_generator": {"DataGenerator": lambda data: data},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(compiler, "read_calibration", read_calibration)
    arguments = (
        tmp_path / "policy.onnx",
        {"input": "policy_input", "input_shape": [1, 192, 1, 34]},
        tmp_path / "calibration",
        tmp_path / "output",
        "lewam_pusht_w16a16_policy",
        {"precision": "w16a16", "calibration_samples": 2, "calibration_bins": 2048},
    )
    if failure:
        with pytest.raises(RuntimeError, match="Compiler terminated"):
            compiler.compile_component(*arguments)
    else:
        compiler.compile_component(*arguments)
    assert calls["source"] == (
        str(arguments[0]),
        {"shape_dict": {"policy_input": (1, 192, 1, 34)}, "layout": "NCHW", "flexible_batch_size": True},
    )
    assert calls["load"] == ("verified-source", {"target": "gen2"})
    assert settings == [
        ("with_activation_quantization", "bf16"),
        ("with_weight_quantization", "bf16"),
        ("with_requantization_mode", "sima"),
        ("with_calibration", ("mse", 2048)),
        ("with_bias_correction", "none"),
        ("with_channel_equalization", False),
        ("with_smooth_quant", False),
        ("with_per_token_quant", False),
    ]
    quantization = calls["quantize"]
    assert quantization["quantization_config"] is config
    assert quantization["model_name"] == "lewam_pusht_w16a16_policy"
    kind, data = quantization["calibration_data"]
    assert kind == "iterable" and list(data) == [("input", "policy_input")]
    assert data[("input", "policy_input")] is values
    if failure != "quantize":
        assert calls["compile"]["output_path"] == str(tmp_path / "output")
        assert calls["compile"]["layer_norm_use_fp32_intermediates"] is False
        assert calls["compile"]["rms_norm_use_fp32_intermediates"] is False
        assert calls["compile"]["deployable"] is True
    else:
        assert "compile" not in calls


@pytest.mark.parametrize("invalid", [None, "schema", "path", "hash", "shape"])
def test_request_loader_checks_real_files(load_script, request_data, tmp_path, invalid):
    runtime = load_script("performance")
    if invalid == "shape":
        request_data["noise"] = request_data["noise"][:5]
    path = tmp_path / "request_data.npz"
    np.savez_compressed(path, **request_data)
    row = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"schema": "lewam.requests.v1", "source": "source_trajectories", "requests": [row]}
    if invalid == "schema":
        manifest["schema"] = "unknown"
    elif invalid == "path":
        row["file"] = "../request_data.npz"
    elif invalid == "hash":
        row["sha256"] = "0" * 64
    (tmp_path / "requests.json").write_text(json.dumps(manifest))
    if invalid:
        with pytest.raises(ValueError):
            runtime.load_request(tmp_path)
    else:
        actual, name = runtime.load_request(tmp_path)
        assert name == "request_data.npz" and set(actual) == set(request_data)
        for key, value in request_data.items():
            np.testing.assert_array_equal(actual[key], value)


def episode_arrays() -> dict:
    images = np.zeros((150, 224, 224, 3), np.uint8)
    return {
        "initial_states": np.zeros((150, 7), np.float64),
        "goal_states": np.zeros((150, 7), np.float64),
        "initial_images": images + np.uint8(200),
        "goal_images": np.broadcast_to(np.arange(150, dtype=np.uint8)[:, None, None, None], images.shape),
        "time_features": np.arange(8 * 192, dtype=np.float32).reshape(8, 192),
        "horizon_features": np.broadcast_to(np.arange(10, dtype=np.float32)[:, None], (10, 192)),
    }


@pytest.mark.parametrize("invalid", [None, "selection", "initial_images", "hash", "environment"])
def test_episode_loader_checks_frozen_identity_and_initial_images(load_script, tmp_path, monkeypatch, invalid):
    accuracy = load_script("accuracy")
    shutil.copyfile(ROOT / "tests/fixtures/lewam_pusht/selection.json", tmp_path / "selection.json")
    rows = json.loads((tmp_path / "selection.json").read_text())["episodes"]
    rows = [{**row, "reset_seed": index} for index, row in enumerate(rows)]
    arrays = episode_arrays()
    if invalid == "initial_images":
        arrays.pop("initial_images")
    path = tmp_path / "episodes.npz"
    np.savez_compressed(path, **arrays)
    manifest = {
        "schema": "lewam.episodes.v1",
        "held_out": False,
        "goal_offset_steps": 25,
        "eval_budget": 50,
        "exec_actions": 5,
        "episodes": rows,
        "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "environment": dict.fromkeys(("stable-worldmodel", "pymunk", "pygame", "torch"), "fixture"),
    }
    if invalid == "selection":
        rows[0]["start"] += 1
    elif invalid == "hash":
        manifest["arrays_sha256"] = "0" * 64
    monkeypatch.setattr(
        accuracy.importlib.metadata, "version", lambda _: "wrong" if invalid == "environment" else "fixture"
    )
    (tmp_path / "episodes.json").write_text(json.dumps(manifest))
    if invalid:
        with pytest.raises(ValueError):
            accuracy.load_episodes(tmp_path)
    else:
        actual, loaded = accuracy.load_episodes(tmp_path)
        assert actual == manifest and set(loaded) == set(arrays)
        for key in arrays:
            np.testing.assert_array_equal(loaded[key], arrays[key])


def test_evaluation_preserves_late_history_active_rng_and_budget(load_script):
    accuracy = load_script("accuracy")
    arrays, environments, draws, calls = episode_arrays(), [], [], []
    rows = [
        {"episode": i, "start": 0, "seed": seed, "reset_seed": 1000 + i}
        for i, seed in enumerate([42] * 50 + [0] * 50 + [1] * 50)
    ]
    assigned_noise = {}

    class Environment:
        def __init__(self):
            self.index, self.steps, self.closed, self.renders = len(environments), 0, False, []
            environments.append(self)

        def reset(self, *, seed):
            assert seed == 1000 + self.index

        def _set_state(self, state):
            np.testing.assert_array_equal(state, arrays["initial_states"][self.index])

        def _set_goal_state(self, goal):
            np.testing.assert_array_equal(goal, arrays["goal_states"][self.index])

        def render(self):
            self.renders.append(self.steps)
            return np.full((224, 224, 3), self.steps, np.uint8)

        def step(self, action):
            value = self.index * 100 + self.steps
            np.testing.assert_array_equal(action, [value, -value])
            self.steps += 1
            reached = (self.index % 50, self.steps) in {(0, 2), (1, 7)}
            state = np.zeros(7)
            state[0] = 0 if reached else 30
            return {"state": state}, 0, reached, False, {}

        def close(self):
            self.closed = True

    def noise_factory(seed):
        group = [42, 0, 1].index(seed)
        cursor, replan = 0, 0

        def noise(count):
            nonlocal cursor, replan
            active = list(range(min(replan, 2), 50))
            assert count == len(active)
            values = np.arange(cursor, cursor + count, dtype=np.float32) + seed * 1000
            for index, value in zip(active, values, strict=True):
                assigned_noise[(group * 50 + index, replan * 5)] = value
            draws.append((seed, count))
            cursor, replan = cursor + count, replan + 1
            return np.broadcast_to(values[:, None, None], (count, 10, 2))

        return noise

    def replan(request_data):
        index = int(request_data["images"][-1, 0, 0, 0])
        step = environments[index].steps
        assert step % 5 == 0
        current = 200 if step == 0 else step
        expected = ([200 if step == 5 else step - 5] if step else []) + [current, index]
        assert request_data["images"][:, 0, 0, 0].tolist() == expected
        np.testing.assert_array_equal(request_data["history_pad"], [step == 0, False])
        np.testing.assert_array_equal(request_data["horizon_features"], arrays["horizon_features"][step // 5])
        np.testing.assert_array_equal(request_data["time_features"], arrays["time_features"])
        np.testing.assert_array_equal(request_data["noise"], np.full((10, 2), assigned_noise[index, step], np.float32))
        calls.append((index, step))
        values = index * 100 + step + np.arange(10, dtype=np.float32)
        return np.stack((values, -values), axis=1)

    results = accuracy.evaluate(SimpleNamespace(replan=replan), rows, arrays, Environment, noise_factory)
    assert draws == [(seed, count) for seed in (42, 0, 1) for count in (50, 49, *([48] * 8))]
    assert len(calls) == 3 * (50 + 49 + 8 * 48)
    assert [(row["success"], row["steps"]) for row in results] == ([(True, 2), (True, 7)] + [(False, 50)] * 48) * 3
    assert all(env.closed and 0 not in env.renders for env in environments)


def test_benchmark_reports_completed_replans_and_profiles_only_after_close(
    load_script, request_data, monkeypatch, tmp_path
):
    runtime = load_script("performance")
    state = SimpleNamespace(closed=False, measured=0, saved=None)
    models = {name: object() for name in SHAPES}
    policy = SimpleNamespace(
        models=models, replan=lambda _: np.zeros((10, 2), np.float32), close=lambda: setattr(state, "closed", True)
    )

    def measure(infer, **kwargs):
        state.measured += 1
        for _ in range(2 if state.measured == 1 else 4):
            infer()
        return ([0.1, 0.2], 0.3) if state.measured == 1 else ([0.5] * 4, 2.0)

    def stats(options):
        assert state.closed and options.include_mla_layer_stats and options.include_mla_stage_stats
        return SimpleNamespace(
            mla_layer_stats=SimpleNamespace(status="ok", layers=[]),
            mla_stage_stats=SimpleNamespace(status="ok", stages=[]),
        )

    def report(benchmark, actual_models, **kwargs):
        assert actual_models is models and kwargs["telemetry"] == "telemetry"
        assert benchmark.fps == 2 and benchmark.throughput_samples == 4
        assert benchmark.throughput_duration_ms == 2000
        assert benchmark.latency_ms == pytest.approx(150)
        assert set(kwargs["mla_layer_stats"]) == set(kwargs["mla_stage_stats"]) == set(SHAPES)
        return SimpleNamespace(save=lambda path: setattr(state, "saved", path))

    pn = SimpleNamespace(
        BenchmarkOptions=SimpleNamespace,
        BenchmarkReport=SimpleNamespace,
        TelemetryMonitor=lambda: SimpleNamespace(start=lambda: None, stop=lambda: "telemetry"),
        Model=lambda _: SimpleNamespace(benchmark=stats),
    )
    monkeypatch.setitem(sys.modules, "pyneat", pn)
    monkeypatch.setitem(sys.modules, "model_report", SimpleNamespace(from_performance=report))
    monkeypatch.setattr(runtime, "load_request", lambda _: (request_data, "fixture.npz"))
    monkeypatch.setattr(runtime, "Policy", lambda *args: policy)
    monkeypatch.setattr(runtime, "measure", measure)
    monkeypatch.setattr(runtime, "WARMUPS", 2)
    runtime.benchmark({name: Path(name) for name in SHAPES}, "pusht-reproduction", tmp_path, tmp_path / "report")
    assert state.measured == 2 and state.saved == tmp_path / "report"
