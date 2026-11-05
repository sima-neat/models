"""Host assembly tests; native execution remains a board qualification gate."""

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def p(load_script):
    return load_script("performance")


def test_complete_model_ingress_then_explicit_raw_head_decode(p, monkeypatch):
    calls = []

    class Route:
        include_input = False
        include_output = True

    class Graph:
        def add(self, node):
            calls.append(node)

    class Model:
        def graph(self, route):
            assert route.include_input is True and route.include_output is False
            return Graph()

    model = Model()

    def boxdecode(bound, **options):
        assert bound == "DetrFocal"
        assert options == {
            "original_width": 1280,
            "original_height": 720,
            "model_width": 640,
            "model_height": 640,
            "detection_threshold": 0.0,
            "nms_iou_threshold": 0.0,
            "top_k": 300,
            "decode_type_option": "InterleavedByHeadLogit",
            "source_storage": "DenseHwc",
            "detess": False,
            "dequant": False,
            "resize_mode": "Stretch",
        }
        return "native_boxdecode"

    monkeypatch.setattr(p.pyneat, "ModelRouteOptions", Route, raising=False)
    monkeypatch.setattr(
        p.pyneat,
        "BoxDecodeType",
        types.SimpleNamespace(DetrFocal="DetrFocal"),
        raising=False,
    )
    monkeypatch.setattr(p.pyneat, "ResizeMode", types.SimpleNamespace(Stretch="Stretch"), raising=False)
    monkeypatch.setattr(
        p.pyneat,
        "BoxDecodeTypeOption",
        types.SimpleNamespace(InterleavedByHeadLogit="InterleavedByHeadLogit"),
        raising=False,
    )
    monkeypatch.setattr(p.pyneat, "BoxDecodeSourceStorage", types.SimpleNamespace(DenseHwc="DenseHwc"), raising=False)
    monkeypatch.setattr(
        p.pyneat,
        "nodes",
        types.SimpleNamespace(sima_box_decode=boxdecode, output=lambda: "output"),
        raising=False,
    )
    p.decoder_graph(model)
    assert calls == ["native_boxdecode", "output"]


def tensor(format):
    return types.SimpleNamespace(semantic=types.SimpleNamespace(detection=types.SimpleNamespace(format=format)))


def test_completion_requires_native_bbox_and_parses_inside_boundary(p, monkeypatch):
    outputs = [tensor("BBOX")]
    called = []

    def decode(values, **options):
        assert values is outputs and options == {"strict": True}
        called.append(True)
        return [types.SimpleNamespace(to_numpy=lambda: np.zeros((300, 6), np.float32))]

    monkeypatch.setattr(p.pyneat, "decode_bbox", decode, raising=False)
    p.require_native_boxes(outputs)
    assert called == [True]


@pytest.mark.parametrize("outputs", [[], [tensor("RAW")], [tensor("BBOX"), tensor("BBOX")]])
def test_raw_or_missing_results_cannot_count_as_native(p, outputs):
    with pytest.raises(RuntimeError):
        p.require_native_boxes(outputs)


def test_dropped_pairs_rejected(p, monkeypatch):
    monkeypatch.setattr(
        p.pyneat,
        "decode_bbox",
        lambda *a, **k: [types.SimpleNamespace(to_numpy=lambda: np.zeros((299, 6)))],
        raising=False,
    )
    with pytest.raises(RuntimeError, match="299 of 300"):
        p.require_native_boxes([tensor("BBOX")])


def test_drain_checks_eos_counts_and_accepts_optional_sample(p):
    class Run:
        def __init__(self):
            self.outputs = iter([types.SimpleNamespace(tensors=[1]), None])

        def pull(self, timeout):
            return next(self.outputs)

    assert list(p.drain(Run(), "decoder", {"decoder": 1})) == [[1]]
    with pytest.raises(RuntimeError, match="1 of 2"):
        list(p.drain(Run(), "decoder", {"decoder": 2}))


def test_query_bridge_owns_gathers_not_attention_memory(p, monkeypatch):
    monkeypatch.setattr(p.pyneat, "TensorLayout", types.SimpleNamespace(Unknown="Unknown"), raising=False)

    def tensor_of(array):
        return types.SimpleNamespace(
            to_numpy=lambda **kwargs: array,
            semantic=types.SimpleNamespace(preprocess="same-frame-meta"),
        )

    content = np.arange(8400 * 256, dtype=np.float32).reshape(1, 8400, 256)
    scores = np.arange(8400, dtype=np.float32)[None]
    refs = np.arange(8400 * 4, dtype=np.float32).reshape(1, 8400, 4)
    copies = []

    def from_numpy(array, **options):
        assert options == {"copy": True, "memory": "EV74", "layout": "Unknown"}
        copies.append(array.copy())
        return types.SimpleNamespace(array=copies[-1], semantic=types.SimpleNamespace())

    monkeypatch.setattr(p.pyneat, "Tensor", types.SimpleNamespace(from_numpy=from_numpy), raising=False)
    monkeypatch.setattr(p.pyneat, "TensorMemory", types.SimpleNamespace(EV74="EV74"), raising=False)
    selected = p.select_queries(*map(tensor_of, (content, scores, refs)))
    np.testing.assert_array_equal(selected[0].array, content[:, 8399:8099:-1])
    np.testing.assert_array_equal(selected[1].array, refs[:, 8399:8099:-1])
    assert all(not np.shares_memory(x.array, y) for x, y in zip(selected, (content, refs), strict=True))

    assert all(x.semantic.preprocess == "same-frame-meta" for x in selected)


def test_native_normalization_declares_rgb_over_255(p, monkeypatch):
    options = types.SimpleNamespace(
        advanced_execution=types.SimpleNamespace(),
        preprocess=types.SimpleNamespace(normalize=types.SimpleNamespace(), resize=types.SimpleNamespace()),
    )
    monkeypatch.setattr(p.pyneat, "ModelOptions", lambda: options, raising=False)
    monkeypatch.setattr(p.pyneat, "InputKind", types.SimpleNamespace(Image="Image"), raising=False)
    monkeypatch.setattr(p.pyneat, "AutoFlag", types.SimpleNamespace(On="On"), raising=False)
    monkeypatch.setattr(p.pyneat, "ResizeMode", types.SimpleNamespace(Stretch="Stretch"), raising=False)
    actual = p.image_options()
    assert actual is options
    assert actual.advanced_execution.postprocess_target == "EV74"
    assert actual.preprocess.kind == "Image"
    assert (actual.preprocess.input_max_width, actual.preprocess.input_max_height) == (1280, 720)
    assert actual.preprocess.resize.mode == "Stretch"
    norm = actual.preprocess.normalize
    assert norm.enable == "On" and norm.has_explicit_stats is True
    assert norm.mean == [0.0, 0.0, 0.0] and norm.stddev == [1.0, 1.0, 1.0]


def test_sequential_inference_forwards_attention_without_conversion(p, monkeypatch):
    memory, content, scores, references, selected = (object() for _ in range(5))
    backbone = types.SimpleNamespace(run=lambda inputs, timeout_ms: [memory, content, scores, references])
    decoder = object()

    def select(c, s, r):
        assert (c, s, r) == (content, scores, references)
        return [selected]

    def invoke(run, inputs):
        assert run is decoder and inputs[0] is memory and inputs[1] is selected

    monkeypatch.setattr(p, "select_queries", select)
    monkeypatch.setattr(p, "invoke_decoder", invoke)
    p.infer_one(backbone, decoder, object())


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


def test_failed_output_setup_never_builds_runners(load_script, monkeypatch):
    p = load_script("performance")
    called = []
    monkeypatch.setattr(p, "mla_stats", lambda *args: called.append("hardware"))
    monkeypatch.setattr(p, "build_runners", lambda *args: called.append("build"))
    with pytest.raises(OSError):
        p.benchmark_pipeline({}, "coco-2017-val", Path("/unused"), Path("/dev/null/not-a-directory"))
    assert called == []


@pytest.mark.parametrize("failure", ["seed", "decoder_build"])
def test_partial_runner_setup_closes_the_created_backbone(p, monkeypatch, failure):
    closed, error = [], ValueError(failure)

    def seed(*args, **kwargs):
        if failure == "seed":
            raise error
        return tuple(object() for _ in range(4))

    def build_decoder(*args, **kwargs):
        raise error

    backbone = types.SimpleNamespace(run=seed, close=lambda: closed.append("backbone"))
    models = {"backbone": types.SimpleNamespace(build=lambda *args, **kwargs: backbone), "decoder": object()}
    monkeypatch.setattr(p.pyneat, "RunOptions", types.SimpleNamespace, raising=False)
    monkeypatch.setattr(p.pyneat, "OutputMemory", types.SimpleNamespace(ZeroCopy="zero_copy"), raising=False)
    monkeypatch.setattr(p.pyneat, "OverflowPolicy", types.SimpleNamespace(Block="block"), raising=False)
    monkeypatch.setattr(p, "select_queries", lambda *args: [object(), object()])
    monkeypatch.setattr(p, "decoder_graph", lambda model: types.SimpleNamespace(build=build_decoder))
    with pytest.raises(ValueError) as caught:
        p.build_runners(models, object())
    assert caught.value is error
    assert closed == ["backbone"]


@pytest.mark.parametrize("failure", [None, "latency", "throughput"])
def test_benchmark_owns_runner_cleanup_once_on_success_and_failure(p, monkeypatch, tmp_path, failure):
    closed, reported, error = [], [], ValueError(failure)
    backbone = types.SimpleNamespace(close=lambda: closed.append("backbone"))
    decoder = types.SimpleNamespace(close=lambda: closed.append("decoder"))
    monkeypatch.setattr(p, "mla_stats", lambda path: ({}, {}))
    monkeypatch.setattr(
        p.pyneat, "Model", lambda *args: types.SimpleNamespace(execution_plan=lambda *args: object()), raising=False
    )
    monkeypatch.setattr(p, "image_options", lambda: None)
    monkeypatch.setattr(p, "tensor_options", lambda: None)
    monkeypatch.setattr(p, "sample_image", lambda path: Path("frame.jpg"))
    monkeypatch.setattr(p, "frame_tensor", lambda path: object())
    monkeypatch.setattr(p, "build_runners", lambda *args: (backbone, decoder))

    def measure(stage, result):
        def run(*args):
            if failure == stage:
                raise error
            return result

        return run

    monkeypatch.setattr(p, "measure_latency", measure("latency", [0.1]))
    monkeypatch.setattr(p, "measure_throughput", measure("throughput", object()))
    monkeypatch.setattr(p, "write_report", lambda *args: reported.append(True))
    args = ({"backbone": Path("backbone"), "decoder": Path("decoder")}, "coco-2017-val", tmp_path, tmp_path)
    if failure:
        with pytest.raises(ValueError) as caught:
            p.benchmark_pipeline(*args)
        assert caught.value is error
    else:
        p.benchmark_pipeline(*args)
    assert closed == ["decoder", "backbone"]
    assert reported == ([] if failure else [True])


def test_throughput_failure_owns_monitor_cleanup_but_not_runner_lifetime(p, monkeypatch):
    cleaned, error = [], ValueError("forwarding")
    monitor = types.SimpleNamespace(running=lambda: True, stop=lambda: cleaned.append("monitor"))
    monkeypatch.setattr(p.pyneat, "TelemetryMonitor", lambda: monitor, raising=False)

    def fail(*stages, **kwargs):
        raise error

    monkeypatch.setattr(p, "run_together", fail)
    runner = types.SimpleNamespace(close=lambda: cleaned.append("runner"))
    with pytest.raises(ValueError) as caught:
        p.measure_throughput(runner, runner, object())
    assert caught.value is error
    assert cleaned == ["monitor"]


def test_forwarding_failure_closes_both_inputs_and_joins_every_stage(p):
    import threading

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

    def forward_proposals():
        ready.wait()
        raise error

    def pull_outputs():
        ready.wait()
        assert decoder_closed.wait(2)
        finished.append("pull")

    with pytest.raises(ValueError) as caught:
        p.run_together(push_inputs, forward_proposals, pull_outputs, runners=runners)
    assert caught.value is error
    assert sorted(finished) == ["pull", "push"]
    assert closed == ["backbone", "decoder"]
    assert not any(
        thread.is_alive() and thread.name in {"push_inputs", "forward_proposals", "pull_outputs"}
        for thread in threading.enumerate()
    )


def test_report_contains_complete_latency_power_and_telemetry(performance):
    metrics = performance["metrics"]
    for key in (
        "throughput_fps",
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "avg_power_watts",
        "peak_power_watts",
        "avg_mla_power_watts",
        "peak_mla_power_watts",
        "energy_per_inference_joules",
    ):
        assert np.isfinite(metrics[key]) and metrics[key] > 0, key
    assert metrics["latency_p50_ms"] <= metrics["latency_p95_ms"] <= metrics["latency_p99_ms"]
    assert metrics["avg_power_watts"] <= metrics["peak_power_watts"]
    assert metrics["energy_per_inference_joules"] == pytest.approx(
        metrics["avg_power_watts"] / metrics["throughput_fps"]
    )
    assert performance["telemetry"]["duration_ms"] >= 60000
    assert performance["telemetry"]["samples"] > 1
    assert "1280x720 RGB" in performance["workload"]
    assert "native DetrFocal BoxDecode" in performance["workload"]
    assert "pipelined 4 frames deep" in performance["workload"]


def test_plans_record_preprocessing_and_all_public_boundary_shapes(performance):
    backbone = performance["components"]["backbone"]["execution_plan"]
    assert backbone["edges"][0]["tensors"] == [{"name": "input_image", "shape": [1, 720, 1280, 3], "dtype": "uint8"}]
    preproc = next(node for node in backbone["nodes"] if node["kind"] == "Preproc")
    assert preproc["runs_on"] == "EV74"
    settings = preproc["settings"]
    assert (settings["input_img_type"], settings["output_img_type"]) == ("RGB", "RGB")
    assert (settings["scaled_width"], settings["scaled_height"], settings["scaling_type"]) == (640, 640, "BILINEAR")
    assert settings["normalize"] is True and settings["aspect_ratio"] is False
    assert settings["channel_mean"] == [0.0, 0.0, 0.0]
    assert settings["channel_stddev"] == [1.0, 1.0, 1.0]
    outputs = backbone["edges"][-1]["tensors"]
    assert [tensor["shape"] for tensor in outputs] == [[1, 8400, 256], [1, 8400, 256], [1, 8400], [1, 8400, 4]]
    assert all(tensor["dtype"] == "float32" for tensor in outputs)
    decoder = performance["components"]["decoder"]["execution_plan"]
    assert sorted(tensor["shape"] for tensor in decoder["edges"][0]["tensors"]) == sorted(
        [[1, 8400, 256], [1, 300, 256], [1, 300, 4]]
    )
    assert [tensor["shape"] for tensor in decoder["edges"][-1]["tensors"]] == [[1, 300, 4], [1, 300, 80]]
