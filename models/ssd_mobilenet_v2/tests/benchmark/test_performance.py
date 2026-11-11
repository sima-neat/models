from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import model_report
import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def assert_native_plan(plan):
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    nodes = {node["id"]: node for node in plan["nodes"]}
    assert len(nodes) == len(plan["nodes"]), "execution-plan node IDs must be unique"
    edges = plan["edges"]
    assert edges and all(edge["from"] in nodes and edge["to"] in nodes for edge in edges)
    inputs = {key for key, node in nodes.items() if node["kind"] == "Input"}
    preproc = {key for key, node in nodes.items() if node["kind"] == "Preproc"}
    mla = {key for key, node in nodes.items() if node["runs_on"] == "MLA"}
    outputs = {key for key, node in nodes.items() if node["kind"] == "Output"}
    assert inputs and preproc and mla and outputs, "complete image preprocessing/model/output route required"
    assert all(nodes[key]["runs_on"] in {"A65", "EV74"} for key in preproc)
    boxes = [node for node in nodes.values() if node["kind"] == "SimaBoxDecode"]
    assert len(boxes) == 1
    (box,) = boxes
    assert box["runs_on"] == "A65"
    # Current Core exports decoder identity/topology, but not decoder knobs.
    # Knobs are independently asserted on the public ModelOptions below.
    settings = box.get("settings")
    if settings is not None:
        assert settings["decode_type"] == "ssd"
        assert settings["num_classes"] == 91
        assert settings["top_k"] == 100
        assert float(np.float32(settings["detection_threshold"])) == float(np.float32(0.2))
        assert float(np.float32(settings["nms_iou_threshold"])) == float(np.float32(0.6))

    def reachable(start, removed=()):
        seen, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in seen or node in removed:
                continue
            seen.add(node)
            pending.extend(edge["to"] for edge in edges if edge["from"] == node)
        return seen

    assert any(mla & reachable(key) for key in preproc), "preprocessing must precede the model"
    assert all(box["id"] in reachable(key) for key in mla), "MLA output must reach SSD"
    assert outputs <= reachable(box["id"])
    for key in inputs:
        assert outputs <= reachable(key), "input must reach every detector output"
        assert not outputs & reachable(key, preproc), "image output bypasses preprocessing"
        assert not outputs & reachable(key, mla), "image output bypasses MLA"
        assert not outputs & reachable(key, {box["id"]}), "raw-head output bypasses SSD"


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path, build_profile: str):
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_complete_native_execution_plan(performance):
    (component,) = performance["components"].values()
    assert_native_plan(component["execution_plan"])


def test_performance_records_latency_power_and_thermal_measurements(performance):
    metrics = performance["metrics"]
    required = (
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_fps",
        "avg_power_watts",
        "peak_power_watts",
        "energy_per_inference_joules",
        "avg_mla_power_watts",
        "peak_mla_power_watts",
        "avg_die_temperature_c",
        "peak_die_temperature_c",
        "avg_board_temperature_c",
        "peak_board_temperature_c",
    )
    assert all(name in metrics and math.isfinite(metrics[name]) for name in required)
    assert all(metrics[name] > 0 for name in required if name.startswith("latency_") or "power_watts" in name)
    assert metrics["latency_p50_ms"] <= metrics["latency_p95_ms"] <= metrics["latency_p99_ms"]
    assert metrics["avg_power_watts"] <= metrics["peak_power_watts"]
    assert metrics["avg_die_temperature_c"] <= metrics["peak_die_temperature_c"]
    assert metrics["avg_board_temperature_c"] <= metrics["peak_board_temperature_c"]


def test_execution_plan_takes_the_benchmark_frame(performance):
    (component,) = performance["components"].values()
    (frame,) = component["execution_plan"]["edges"][0]["tensors"]
    assert frame["shape"] == [1, 720, 1280, 3]


def test_mla_stats_collected(performance):
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"])
    assert stage_stats["stages"]


def test_performance_explicit_native_customer_policy(load_script):
    script = load_script("performance")
    options = script.native_options()
    assert options.decode_type == sys.modules["pyneat"].BoxDecodeType.Ssd
    assert options.num_classes == 91
    assert (options.score_threshold, options.nms_iou_threshold, options.top_k) == (0.2, 0.6, 100)
    assert options.preprocess.kind == sys.modules["pyneat"].InputKind.Image
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (1280, 720)
    assert options.preprocess.resize.mode == sys.modules["pyneat"].ResizeMode.Stretch
    assert options.preprocess.normalize.mean == [0.5] * 3
    assert options.preprocess.normalize.stddev == [0.5] * 3
    assert (options.boxdecode_original_width, options.boxdecode_original_height) == (0, 0)


def test_performance_uses_seeded_real_720p_input_and_whole_model_report(load_script, monkeypatch, tmp_path):
    script = load_script("performance")
    folder = tmp_path / "val2017"
    folder.mkdir()
    for name in ["c.jpg", "a.jpg", "b.jpg"]:
        (folder / name).touch()
    assert script.sample_image(tmp_path).name == "b.jpg"
    monkeypatch.setattr(script, "read_rgb", lambda _: np.zeros((40, 40, 3), np.uint8))
    imported, benchmarks, reports, saved = [], [], [], []

    def tensor(frame, **kw):
        imported.append((frame.copy(), kw))
        return "image-tensor"

    monkeypatch.setattr(sys.modules["pyneat"].Tensor, "from_numpy", tensor)
    model = SimpleNamespace(benchmark=lambda options: benchmarks.append(options) or "actual-core-report")
    monkeypatch.setattr(sys.modules["pyneat"], "Model", lambda path, options: model)

    def report(core_report, models, **kw):
        reports.append((core_report, models, kw))
        return SimpleNamespace(save=lambda path: saved.append(path))

    monkeypatch.setattr(model_report, "from_performance", report)
    script.benchmark("model.tar.gz", "coco-2017-val", tmp_path, tmp_path / "out")
    (options,) = benchmarks
    assert options.inputs == ["image-tensor"]
    assert (options.num_samples, options.min_duration_ms) == (1000, 60000)
    assert options.include_mla_layer_stats is True and options.include_mla_stage_stats is True
    image, kwargs = imported[0]
    assert image.shape == (720, 1280, 3) and image.dtype == np.uint8
    assert (image[:, :280] == 114).all() and (image[:, 280:1000] == 0).all()
    assert kwargs == dict(copy=True, image_format=sys.modules["pyneat"].PixelFormat.RGB)
    assert reports[0][:2] == ("actual-core-report", {"model": model})
    assert "top 100" in reports[0][2]["workload"] and "complete detector" in reports[0][2]["workload"]
    assert saved == [tmp_path / "out"]


def test_performance_is_standalone_without_registry_or_sibling_imports(model_dir):
    tree = ast.parse((model_dir / "scripts/performance.py").read_text())
    imports = {
        name.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    }
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {"__future__", "argparse", "random", "pathlib", "cv2", "model_report", "numpy", "pyneat"}


def _plan(preproc_backend="EV74", fused=True):
    # This helper exercises topology; the stage-output test checks actual frame geometry.
    nodes = [
        dict(id="in", kind="Input", runs_on="A65"),
        dict(id="pre", kind="Preproc", runs_on=preproc_backend, kernel="preproc_quant_tess" if fused else "preproc"),
        dict(id="mla", kind="ModelFragment", runs_on="MLA"),
        dict(
            id="ssd",
            kind="SimaBoxDecode",
            runs_on="A65",
            settings=dict(
                decode_type="ssd",
                num_classes=91,
                top_k=100,
                detection_threshold=0.2,
                nms_iou_threshold=0.6,
            ),
        ),
        dict(id="out", kind="Output", runs_on="A65"),
    ]
    chain = ["in", "pre", "mla", "ssd", "out"]
    if not fused:
        nodes.append(dict(id="quant", kind="QuantTess", runs_on=preproc_backend))
        chain.insert(2, "quant")
    return {
        "schema": "sima.neat.execution_plan",
        "schema_version": 1,
        "nodes": nodes,
        "edges": [{"from": left, "to": right} for left, right in zip(chain, chain[1:])],
    }


@pytest.mark.parametrize("backend", ["A65", "EV74"])
@pytest.mark.parametrize("fused", [True, False])
def test_native_plan_accepts_auto_selected_preprocessing_and_fusion(backend, fused):
    assert_native_plan(_plan(backend, fused))


@pytest.mark.parametrize("edge", [("in", "mla"), ("in", "ssd"), ("mla", "out")])
def test_native_plan_rejects_preprocessing_model_and_decoder_bypasses(edge):
    plan = _plan()
    plan["edges"].append({"from": edge[0], "to": edge[1]})
    with pytest.raises(AssertionError, match="bypasses"):
        assert_native_plan(plan)


@pytest.mark.parametrize("kind", ["Input", "Preproc", "SimaBoxDecode", "Output"])
def test_native_plan_rejects_missing_semantic_stages(kind):
    plan = _plan()
    plan["nodes"] = [node for node in plan["nodes"] if node["kind"] != kind]
    with pytest.raises(AssertionError):
        assert_native_plan(plan)


def test_native_plan_rejects_duplicate_ids_and_unknown_edges():
    plan = _plan()
    plan["nodes"].append(plan["nodes"][0].copy())
    with pytest.raises(AssertionError, match="unique"):
        assert_native_plan(plan)
    plan = _plan()
    plan["edges"].append({"from": "unknown", "to": "out"})
    with pytest.raises(AssertionError):
        assert_native_plan(plan)


def test_native_plan_check_rejects_raw_heads_and_wrong_policy():
    assert_native_plan(_plan())
    plan = _plan()
    plan["edges"].append({"from": "mla", "to": "out"})
    with pytest.raises(AssertionError, match="bypasses"):
        assert_native_plan(plan)
    for key, value in [
        ("decode_type", "yolov8"),
        ("decode_type", "ssd-multiclass-v1"),
        ("num_classes", 90),
        ("top_k", 10),
        ("detection_threshold", 0.25),
        ("nms_iou_threshold", 0.55),
    ]:
        plan = _plan()
        next(node for node in plan["nodes"] if node["id"] == "ssd")["settings"][key] = value
        with pytest.raises(AssertionError):
            assert_native_plan(plan)


@pytest.mark.parametrize("value", ["", "=", "coco-2017-val=", "coco-2017-val=   ", "other=/data", "coco-2017-val"])
def test_dataset_argument_is_one_named_nonempty_source(load_script, value):
    script = load_script("performance")
    with pytest.raises(SystemExit, match="--dataset"):
        script.parse_dataset([value])


def test_dataset_argument_rejects_repeats_instead_of_taking_the_last(load_script):
    script = load_script("performance")
    with pytest.raises(SystemExit, match="--dataset"):
        script.parse_dataset(["coco-2017-val=/first", "coco-2017-val=/second"])
    assert script.parse_dataset(["coco-2017-val=/data"]) == ("coco-2017-val", Path("/data"))


@pytest.mark.parametrize("values", [[], [""], ["   "], ["/a", "/b"]])
def test_output_argument_rejects_empty_or_duplicate_values(load_script, values):
    script = load_script("performance")
    with pytest.raises(SystemExit, match="--output"):
        script.one_argument(values, "--output")


@pytest.mark.parametrize("flag,value", [("--dataset", "coco-2017-val=/again"), ("--output", "/again")])
def test_actual_parser_rejects_duplicate_options_before_model(load_script, monkeypatch, flag, value):
    script = load_script("performance")
    monkeypatch.setattr(script, "benchmark", lambda *args: pytest.fail("invalid options must not benchmark"))
    mpk = "/model.tar.gz"
    with pytest.raises(SystemExit, match=flag):
        script.main(["--mpk", mpk, "--dataset", "coco-2017-val=/data", "--output", "/out", flag, value])


def test_native_plan_accepts_current_core_export_without_decoder_settings():
    plan = _plan()
    next(node for node in plan["nodes"] if node["id"] == "ssd").pop("settings")
    assert_native_plan(plan)


def test_performance_records_customer_workload_policy(performance):
    workload = performance["workload"]
    assert all(value in workload for value in ("1280x720", "native Ssd decoder", "threshold 0.2", "IoU 0.6", "top 100"))
