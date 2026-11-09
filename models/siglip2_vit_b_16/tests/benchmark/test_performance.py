from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.array.copy() if copy else self.array


def test_mpk_arguments_name_exactly_both_encoders(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")

    mpks = script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz", "text_encoder=/data/text_mpk.tar.gz"])
    assert list(mpks) == ["image_encoder", "text_encoder"]
    with pytest.raises(SystemExit, match="exactly image_encoder=<path> and text_encoder=<path>"):
        script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz"])


def test_the_cpu_looks_the_check_prompt_up_for_the_text_encoder(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")
    prompt = script.load_prompt()

    embeds = script.token_embeddings(prompt)
    assert embeds.shape == (1, 64, 768)
    for position, token in enumerate(prompt.token_ids):
        np.testing.assert_array_equal(embeds[0, position], prompt.rows[token])


def test_the_cpu_step_scores_the_image_against_the_pooled_text(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")
    image = np.zeros(768, np.float32)
    image[0] = 2.0
    text = np.zeros((1, 768), np.float32)
    text[0, :2] = [1.0, 1.0]

    embedding = script.image_embedding([_Tensor(np.zeros((1, 1000), np.float32)), _Tensor(image[None])])
    np.testing.assert_array_equal(embedding, image)
    assert script.similarity(embedding, [_Tensor(text)]) == pytest.approx(np.sqrt(0.5))


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["image_encoder", "text_encoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_image_encoder_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    (frame,) = performance["components"]["image_encoder"]["execution_plan"]["edges"][0]["tensors"]
    assert frame["shape"] == [1, 720, 1280, 3]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id
