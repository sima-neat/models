from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def _superpoint_dataset(root: Path) -> Path:
    """A yfcc100m-keypoints val directory of three images at the 640 x 480 input's centre, one with 100 keypoints,
    and two listed pairs, one of them holding the short image."""
    rows = []
    for index, count in enumerate((512, 100, 512)):
        stem = root / "val" / "a" / str(index)
        stem.mkdir(parents=True)
        np.save(stem / "keypoints.npy", np.tile(np.array([320, 240], np.float32), (count, 1)))
        np.save(stem / "scores.npy", np.full(count, 0.5, np.float32))
        np.save(stem / "descriptors.npy", np.full((count, 256), 1 / 16, np.float32))
        rows.append(f"a/{index}.jpg 1280 960 0 {count}")
    (root / "annotations").mkdir()
    (root / "annotations" / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "annotations" / "pairs_val.txt").write_text("a/0.jpg a/1.jpg 0 0\na/0.jpg a/2.jpg 0 0\n", encoding="utf-8")
    return root


def test_the_benchmark_pair_is_a_listed_pair_of_full_images(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    performance = load_script("performance")

    workload, images = performance.dataset_pair(_superpoint_dataset(tmp_path / "dataset"))
    tensors = performance.pair_tensors(images)

    # The pair holding the 100-keypoint image is never picked.
    assert workload == "the yfcc100m-keypoints-val pair a/0.jpg and a/2.jpg"
    assert [tensor.shape for tensor in tensors] == [(1, 1, 512, 2), (1, 1, 512, 1), (1, 1, 512, 256)] * 2
    # The keypoints sit at the input's centre, which SuperGlue's frame maps to the origin.
    assert all((tensor == 0).all() for tensor in tensors[0::3])


def test_the_fixed_pair_fills_the_input(load_script: Callable[[str], ModuleType]) -> None:
    performance = load_script("performance")

    _, images = performance.fixed_pair()

    for image in images:
        assert image.keypoints.shape == (512, 2) and (image.keypoints < [640, 480]).all()
        np.testing.assert_allclose(np.linalg.norm(image.descriptors, axis=1), 1)


def test_the_model_takes_tensors_without_image_preprocessing(load_script: Callable[[str], ModuleType]) -> None:
    performance = load_script("performance")

    options = performance.tensor_options()

    assert options.preprocess.kind is performance.pyneat.InputKind.Tensor
    off = performance.pyneat.AutoFlag.Off
    assert (options.preprocess.resize.enable, options.preprocess.normalize.enable) == (off, off)
    assert options.preprocess.color_convert.enable is off


def test_mla_stats_requests_layer_and_stage_stats(load_script: Callable[[str], ModuleType]) -> None:
    performance = load_script("performance")

    performance.mla_stats("model.tar.gz")

    options = performance.pyneat.BenchmarkOptions.return_value
    assert (options.include_mla_layer_stats, options.include_mla_stage_stats) == (True, True)
    performance.pyneat.Model.return_value.benchmark.assert_called_once_with(options)


def test_every_timed_pair_runs_the_cpu_matching(load_script: Callable[[str], ModuleType]) -> None:
    performance = load_script("performance")

    class Runner:
        """Returns couplings in Neat's channels-last [1, 1, i, j] shape, each keypoint scoring itself highest."""

        def run(self, tensors: list, timeout_ms: int) -> list:
            couplings = np.full((513, 513), -5.0, np.float32)
            np.fill_diagonal(couplings, 5.0)
            return [types.SimpleNamespace(to_numpy=lambda copy: couplings.reshape(1, 1, 513, 513))]

    assert performance.match_pair(Runner(), []).tolist() == list(range(512))


@pytest.mark.parametrize("spread", [3.0, 12.0, 400.0])
def test_the_benchmark_matches_as_accuracy_does(spread: float, load_script: Callable[[str], ModuleType]) -> None:
    """performance.py times its own copy of the CPU matching; it must give accuracy.py's matches."""
    performance, accuracy = load_script("performance"), load_script("accuracy")
    rng = np.random.RandomState(7)
    couplings = (rng.normal(0, 1, (513, 513)) * spread / 10).astype(np.float32)
    couplings[np.arange(300), rng.permutation(512)[:300]] += spread

    matches, scores = performance.sinkhorn_matches(couplings, 20, 0.2)
    expected = accuracy.sinkhorn_matches(couplings, 20, 0.2)

    np.testing.assert_array_equal(matches, expected[0])
    np.testing.assert_array_equal(scores, expected[1])
    keypoints = rng.uniform((0, 0), (640, 480), (512, 2))
    np.testing.assert_array_equal(
        performance.normalize_keypoints(keypoints, 640, 480), accuracy.normalize_keypoints(keypoints, 640, 480)
    )


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
