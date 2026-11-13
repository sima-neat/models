from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["text_encoder", "image_model"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_image_model_execution_plan_takes_the_prepared_pixels(performance: dict[str, Any]) -> None:
    # The image model runs on the tensor route, so its plan starts at the CPU-prepared 1008x1008 pixels.
    edges = performance["components"]["image_model"]["execution_plan"]["edges"]
    assert [1008, 1008, 3] in [tensor["shape"][-3:] for edge in edges for tensor in edge["tensors"]]


def test_the_cpu_step_prepares_the_frame_as_accuracy_does(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script, accuracy = load_script("performance"), load_script("accuracy")
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: array)
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    # accuracy.py reads BGR files; the frame is RGB.
    assert script.cv2.imwrite(str(path), script.cv2.cvtColor(frame, script.cv2.COLOR_RGB2BGR))

    prepared = script.pixels(SimpleNamespace(to_numpy=lambda copy: frame))

    assert (prepared.shape, prepared.dtype) == ((1008, 1008, 3), np.float32)
    np.testing.assert_array_equal(prepared, accuracy.preprocess_image(path))


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id


def test_the_cpu_step_packs_the_prompt_as_calibration_does(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script, calibration = load_script("performance"), load_script("calibration")
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: array)
    features = np.random.default_rng(0).random((1, 16, 256), dtype=np.float32)
    prompt = script.PROMPT_TOKENS
    assert list(prompt) == calibration.tokens(script.PROMPT)

    np.testing.assert_array_equal(script.prompt_tensor(prompt), calibration.one_hot(prompt))
    packed = script.pack_text(SimpleNamespace(to_numpy=lambda copy: features), prompt)
    np.testing.assert_array_equal(packed, calibration.packed(features, prompt))
