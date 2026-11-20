from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from model_registry.model_tests import measured_shortfalls


def test_ddim_schedule_runs_from_noise_to_the_clean_latent(load_script: Callable[[str], ModuleType]) -> None:
    performance = load_script("performance")
    schedule = performance.Schedule.ddim(performance.STEPS)
    assert (schedule.timesteps[0], schedule.timesteps[-1], len(schedule.timesteps)) == (999, 0, 50)
    clean = np.random.RandomState(1).standard_normal((1, 4, 32, 32)).astype(np.float32)
    noise = np.random.RandomState(2).standard_normal(clean.shape).astype(np.float32)
    # A latent noised to the first timestep, with the noise predicted exactly, steps to the next timestep's latent
    # and, from the last timestep, to the clean latent itself.
    for index in (0, len(schedule.timesteps) - 1):
        alpha = schedule.alphas[schedule.timesteps[index]]
        noisy = (np.sqrt(alpha) * clean + np.sqrt(1 - alpha) * noise).astype(np.float32)
        following = schedule.alphas[schedule.timesteps[index + 1]] if index == 0 else 1.0
        expected = np.sqrt(following) * clean + np.sqrt(1 - following) * noise
        np.testing.assert_allclose(schedule.step(index, noisy, noise), expected, atol=1e-4)


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == [
        "transformer_0", "transformer_1", "transformer_2", "transformer_3", "vae_decoder"]
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
