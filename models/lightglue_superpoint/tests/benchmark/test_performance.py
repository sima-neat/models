from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

# Ground truth for a pairs line: two identity intrinsics and an identity pose.
IDENTITY = " ".join(["1 0 0 0 1 0 0 0 1"] * 2 + ["1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"])


def _write_val(root: Path, counts: dict[str, int]) -> None:
    """A yfcc100m-keypoints val directory: each image's keypoints and descriptors, and pairs of consecutive images."""
    names = sorted(counts)
    for name in names:
        directory = root / "val" / Path(name).with_suffix("")
        directory.mkdir(parents=True)
        np.save(directory / "keypoints.npy", np.zeros((counts[name], 2), np.float32))
        np.save(directory / "descriptors.npy", np.zeros((counts[name], 256), np.float32))
    (root / "annotations").mkdir()
    rows = [f"{name} 1280 960 0 {counts[name]}" for name in names]
    (root / "annotations" / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    pairs = [f"{first} {second} 0 0 {IDENTITY}" for first, second in zip(names, names[1:])]
    (root / "annotations" / "pairs_val.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")


@pytest.mark.parametrize("script_name", ["performance", "accuracy"])
def test_inputs_go_to_neat_as_one_row_of_512_keypoints_in_ev74_memory(
    script_name: str, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script(script_name)
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: (array, options))
    inputs = {name: np.zeros(shape, np.float32) for name, shape in script.INPUT_SHAPES.items()}

    tensors = script.input_tensors(inputs)

    assert [array.shape for array, _ in tensors] == [(1, 1, 512, 128), (1, 1, 512, 256)] * 2
    assert all(options == {"copy": True, "memory": script.pyneat.TensorMemory.EV74} for _, options in tensors)


def test_the_benchmark_measures_one_fixed_val_pair(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")
    _write_val(tmp_path / "yfcc100m-keypoints-val", {f"s/{index}.jpg": 512 for index in range(4)} | {"s/4.jpg": 300})

    first, second = script.benchmark_pair(tmp_path), script.benchmark_pair(tmp_path)

    assert (first[0].name0, first[0].name1) == (second[0].name0, second[0].name1)
    assert "s/4.jpg" not in (first[0].name0, first[0].name1)
    assert list(first[1]) == list(script.INPUT_SHAPES)


def test_the_script_times_the_pair_on_the_tensor_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("performance")
    _write_val(tmp_path / "yfcc100m-keypoints-val", {f"s/{index}.jpg": 512 for index in range(4)})
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: (array, options))
    saved = SimpleNamespace(calls=[])
    monkeypatch.setattr(script.model_report, "from_performance", lambda *args, **kwargs: (
        saved.calls.append((args, kwargs)) or SimpleNamespace(save=lambda output: saved.calls.append(output))
    ))
    dataset = f"yfcc100m-keypoints-val={tmp_path}"
    monkeypatch.setattr(sys, "argv", ["performance.py", "--mpk", "m.tar.gz", "--dataset", dataset, "--output", "out"])

    script.main()

    (mpk, options), _ = script.pyneat.Model.call_args
    assert mpk == "m.tar.gz"
    assert options.preprocess.kind == script.pyneat.InputKind.Tensor
    preprocess = options.preprocess
    switches = (preprocess.resize.enable, preprocess.color_convert.enable, preprocess.normalize.enable)
    assert switches == (script.pyneat.AutoFlag.Off,) * 3
    (request,), _ = script.pyneat.Model.return_value.benchmark.call_args
    timing = (request.num_samples, request.min_duration_ms, request.include_mla_layer_stats)
    assert timing + (request.include_mla_stage_stats,) == (1000, 60000, True, True)
    pair, inputs = script.benchmark_pair(tmp_path)
    assert [array.shape for array, _ in request.inputs] == [(1, 1, 512, 128), (1, 1, 512, 256)] * 2
    np.testing.assert_array_equal(request.inputs[0][0].reshape(1, 512, 128), inputs["encoding0"])
    ((_, models), kwargs), output = saved.calls
    assert kwargs["workload"].startswith(f"yfcc100m-keypoints-val pair {pair.name0} and {pair.name1}")
    assert (models, output) == ({"model": script.pyneat.Model.return_value}, "out")


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"
