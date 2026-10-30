from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


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
    root = tmp_path / "yfcc100m-keypoints-val"
    names = [f"s/{index}.jpg" for index in range(4)]
    for name in names:
        directory = root / "val" / Path(name).with_suffix("")
        directory.mkdir(parents=True)
        np.save(directory / "keypoints.npy", np.zeros((512, 2), np.float32))
        np.save(directory / "descriptors.npy", np.zeros((512, 256), np.float32))
    (root / "annotations").mkdir()
    ground_truth = " ".join(["1 0 0 0 1 0 0 0 1"] * 2 + ["1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"])
    pairs = [f"{first} {second} 0 0 {ground_truth}" for first, second in zip(names, names[1:])]
    (root / "annotations" / "pairs_val.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")

    first, second = script.benchmark_pair(tmp_path), script.benchmark_pair(tmp_path)

    assert (first[0].name0, first[0].name1) == (second[0].name0, second[0].name1)
    assert list(first[1]) == list(script.INPUT_SHAPES)


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
