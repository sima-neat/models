"""Host assembly tests; native execution remains a board qualification gate."""

import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def performance_script(load_script):
    return load_script("performance")


def test_native_normalization_declares_rgb_over_255_and_the_benchmark_frame(performance_script, monkeypatch):
    options = types.SimpleNamespace(
        advanced_execution=types.SimpleNamespace(),
        preprocess=types.SimpleNamespace(normalize=types.SimpleNamespace(), resize=types.SimpleNamespace()),
    )
    monkeypatch.setattr(performance_script.pyneat, "ModelOptions", lambda: options, raising=False)
    monkeypatch.setattr(performance_script.pyneat, "InputKind", types.SimpleNamespace(Image="Image"), raising=False)
    monkeypatch.setattr(performance_script.pyneat, "AutoFlag", types.SimpleNamespace(On="On"), raising=False)
    monkeypatch.setattr(
        performance_script.pyneat, "ResizeMode", types.SimpleNamespace(Stretch="Stretch"), raising=False
    )
    actual = performance_script.image_options()
    assert actual is options
    assert actual.advanced_execution.postprocess_target == "EV74"
    assert actual.preprocess.kind == "Image"
    assert actual.preprocess.resize.mode == "Stretch"
    assert (actual.preprocess.input_max_width, actual.preprocess.input_max_height) == (1280, 720)
    norm = actual.preprocess.normalize
    assert norm.enable == "On" and norm.has_explicit_stats is True
    assert norm.mean == [0.0, 0.0, 0.0] and norm.stddev == [1.0, 1.0, 1.0]


def test_decoder_takes_tensors_without_image_preprocessing(performance_script, monkeypatch):
    options = types.SimpleNamespace(
        preprocess=types.SimpleNamespace(
            resize=types.SimpleNamespace(), color_convert=types.SimpleNamespace(), normalize=types.SimpleNamespace()
        )
    )
    monkeypatch.setattr(performance_script.pyneat, "ModelOptions", lambda: options, raising=False)
    monkeypatch.setattr(performance_script.pyneat, "InputKind", types.SimpleNamespace(Tensor="Tensor"), raising=False)
    monkeypatch.setattr(performance_script.pyneat, "AutoFlag", types.SimpleNamespace(Off="Off"), raising=False)
    actual = performance_script.tensor_options()
    assert actual.preprocess.kind == "Tensor"
    assert (actual.preprocess.resize.enable, actual.preprocess.normalize.enable) == ("Off", "Off")


def test_drain_checks_eos_counts_and_accepts_optional_sample(performance_script):
    class Run:
        def __init__(self):
            self.outputs = iter([types.SimpleNamespace(tensors=[1]), None])

        def pull(self, timeout):
            return next(self.outputs)

    assert list(performance_script.drain(Run(), "decoder", {"decoder": 1})) == [[1]]
    with pytest.raises(RuntimeError, match="1 of 2"):
        list(performance_script.drain(Run(), "decoder", {"decoder": 2}))


def test_query_bridge_owns_gathers_not_attention_memory(performance_script, monkeypatch):
    monkeypatch.setattr(
        performance_script.pyneat, "TensorLayout", types.SimpleNamespace(Unknown="Unknown"), raising=False
    )

    def tensor_of(array):
        return types.SimpleNamespace(
            to_numpy=lambda **kwargs: array,
            semantic=types.SimpleNamespace(preprocess="same-frame-meta"),
        )

    content = np.arange(2000 * 128, dtype=np.float32).reshape(1, 2000, 128)
    scores = np.arange(2000, dtype=np.float32)[None]
    refs = np.arange(2000 * 4, dtype=np.float32).reshape(1, 2000, 4)
    copies = []

    def from_numpy(array, **options):
        assert options == {"copy": True, "memory": "EV74", "layout": "Unknown"}
        copies.append(array.copy())
        return types.SimpleNamespace(array=copies[-1], semantic=types.SimpleNamespace())

    monkeypatch.setattr(
        performance_script.pyneat, "Tensor", types.SimpleNamespace(from_numpy=from_numpy), raising=False
    )
    monkeypatch.setattr(performance_script.pyneat, "TensorMemory", types.SimpleNamespace(EV74="EV74"), raising=False)
    selected = performance_script.select_queries(*map(tensor_of, (content, scores, refs)))
    np.testing.assert_array_equal(selected[0].array, content[:, 1999:1699:-1])
    np.testing.assert_array_equal(selected[1].array, refs[:, 1999:1699:-1])
    assert all(not np.shares_memory(x.array, y) for x, y in zip(selected, (content, refs), strict=True))
    assert all(x.semantic.preprocess == "same-frame-meta" for x in selected)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_a_frame_counts_only_with_both_decoder_heads(performance_script, count):
    outputs = [object()] * count
    if count == 2:
        performance_script.require_heads(outputs)
    else:
        with pytest.raises(RuntimeError, match="pred_boxes and pred_logits"):
            performance_script.require_heads(outputs)


def test_sequential_inference_forwards_attention_memory_to_the_decoder(performance_script, monkeypatch):
    memory, content, scores, references, selected = (object() for _ in range(5))
    backbone = types.SimpleNamespace(run=lambda inputs, timeout_ms: [memory, content, scores, references])
    ran = []

    def run(inputs, timeout_ms):
        assert inputs[0] is memory and inputs[1] is selected
        ran.append(timeout_ms)
        return ["boxes", "logits"]

    def select(c, s, r):
        assert (c, s, r) == (content, scores, references)
        return [selected]

    monkeypatch.setattr(performance_script, "select_queries", select)
    performance_script.infer_one(backbone, types.SimpleNamespace(run=run), object())
    assert ran == [performance_script.TIMEOUT_MS]


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["backbone", "decoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id


def test_failed_output_setup_never_builds_runners(performance_script, monkeypatch):
    called = []
    monkeypatch.setattr(performance_script, "mla_stats", lambda *args: called.append("hardware"))
    monkeypatch.setattr(performance_script, "build_runners", lambda *args: called.append("build"))
    with pytest.raises(OSError):
        performance_script.benchmark_pipeline({}, "coco-2017-val", Path("/unused"), Path("/dev/null/not-a-directory"))
    assert called == []


@pytest.mark.parametrize("failure", ["seed", "decoder_build"])
def test_partial_runner_setup_closes_the_created_backbone(performance_script, monkeypatch, failure):
    closed, error = [], ValueError(failure)

    def seed(*args, **kwargs):
        if failure == "seed":
            raise error
        return tuple(object() for _ in range(4))

    def build_decoder(*args, **kwargs):
        raise error

    backbone = types.SimpleNamespace(run=seed, close=lambda: closed.append("backbone"))
    models = {
        "backbone": types.SimpleNamespace(build=lambda *args, **kwargs: backbone),
        "decoder": types.SimpleNamespace(build=build_decoder),
    }
    monkeypatch.setattr(performance_script.pyneat, "RunOptions", types.SimpleNamespace, raising=False)
    monkeypatch.setattr(
        performance_script.pyneat, "OutputMemory", types.SimpleNamespace(ZeroCopy="zero_copy"), raising=False
    )
    monkeypatch.setattr(
        performance_script.pyneat, "OverflowPolicy", types.SimpleNamespace(Block="block"), raising=False
    )
    monkeypatch.setattr(performance_script, "select_queries", lambda *args: [object(), object()])
    with pytest.raises(ValueError) as caught:
        performance_script.build_runners(models, object())
    assert caught.value is error
    assert closed == ["backbone"]


@pytest.mark.parametrize("failure", [None, "latency", "throughput"])
def test_benchmark_owns_runner_cleanup_once_on_success_and_failure(performance_script, monkeypatch, tmp_path, failure):
    closed, reported, error = [], [], ValueError(failure)
    backbone = types.SimpleNamespace(close=lambda: closed.append("backbone"))
    decoder = types.SimpleNamespace(close=lambda: closed.append("decoder"))
    monkeypatch.setattr(performance_script, "mla_stats", lambda path: ({}, {}))
    monkeypatch.setattr(performance_script.pyneat, "Model", lambda *args: object(), raising=False)
    monkeypatch.setattr(performance_script, "image_options", lambda: None)
    monkeypatch.setattr(performance_script, "tensor_options", lambda: None)
    monkeypatch.setattr(performance_script, "sample_image", lambda path: Path("frame.jpg"))
    monkeypatch.setattr(performance_script, "frame_tensor", lambda path: object())
    monkeypatch.setattr(performance_script, "build_runners", lambda *args: (backbone, decoder))

    def measure(stage, result):
        def run(*args):
            if failure == stage:
                raise error
            return result

        return run

    monkeypatch.setattr(performance_script, "measure_latency", measure("latency", [0.1]))
    monkeypatch.setattr(performance_script, "measure_throughput", measure("throughput", object()))
    monkeypatch.setattr(performance_script, "write_report", lambda *args: reported.append(True))
    args = ({"backbone": Path("backbone"), "decoder": Path("decoder")}, "coco-2017-val", tmp_path, tmp_path)
    if failure:
        with pytest.raises(ValueError) as caught:
            performance_script.benchmark_pipeline(*args)
        assert caught.value is error
    else:
        performance_script.benchmark_pipeline(*args)
    assert closed == ["decoder", "backbone"]
    assert reported == ([] if failure else [True])


def test_throughput_failure_owns_monitor_cleanup_but_not_runner_lifetime(performance_script, monkeypatch):
    cleaned, error = [], ValueError("forwarding")
    monitor = types.SimpleNamespace(running=lambda: True, stop=lambda: cleaned.append("monitor"))
    monkeypatch.setattr(performance_script.pyneat, "TelemetryMonitor", lambda: monitor, raising=False)

    def fail(*stages, **kwargs):
        raise error

    monkeypatch.setattr(performance_script, "run_together", fail)
    runner = types.SimpleNamespace(close=lambda: cleaned.append("runner"))
    with pytest.raises(ValueError) as caught:
        performance_script.measure_throughput(runner, runner, object())
    assert caught.value is error
    assert cleaned == ["monitor"]


def test_forwarding_failure_closes_both_inputs_and_joins_every_stage(performance_script):
    ready = threading.Barrier(3, timeout=2)
    backbone_closed, decoder_closed = threading.Event(), threading.Event()
    finished, closed, error = [], [], ValueError("forwarding")

    def close(name, event):
        closed.append(name)
        event.set()

    runners = (
        types.SimpleNamespace(close_input=lambda: close("backbone", backbone_closed)),
        types.SimpleNamespace(close_input=lambda: close("decoder", decoder_closed)),
    )

    def push_inputs():
        ready.wait()
        assert backbone_closed.wait(2)
        finished.append("push")

    def forward_queries():
        ready.wait()
        raise error

    def pull_outputs():
        ready.wait()
        assert decoder_closed.wait(2)
        finished.append("pull")

    with pytest.raises(ValueError) as caught:
        performance_script.run_together(push_inputs, forward_queries, pull_outputs, runners=runners)
    assert caught.value is error
    assert sorted(finished) == ["pull", "push"]
    assert closed == ["backbone", "decoder"]
    assert not any(
        thread.is_alive() and thread.name in {"push_inputs", "forward_queries", "pull_outputs"}
        for thread in threading.enumerate()
    )
