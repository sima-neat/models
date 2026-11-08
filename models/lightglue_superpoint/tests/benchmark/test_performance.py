from __future__ import annotations

import queue
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


def _images() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Two images' 512 keypoints at the 640x480 input's centre and one pixel to its right, and their descriptors."""
    return tuple((np.tile(np.float32([320 + side, 240]), (512, 1)), np.full((512, 256), 1 / 16, np.float32))
                 for side in (0, 1))


def _assignment(values: np.ndarray) -> SimpleNamespace:
    """A Neat output tensor holding the assignment in Neat's [1, 1, i, j] shape."""
    return SimpleNamespace(to_numpy=lambda copy: values.reshape(1, 1, 512, 512))


def _diagonal() -> np.ndarray:
    """An assignment in which each keypoint matches the keypoint of the same index."""
    values = np.full((512, 512), 1e-4, np.float32)
    np.fill_diagonal(values, 0.9)
    return values


@pytest.fixture
def script(load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """script.py, with each Neat tensor standing for its array and options."""
    script = load_script("performance")
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: (array, options))
    return script


def test_inputs_go_to_neat_as_one_row_of_512_keypoints_in_ev74_memory(script: ModuleType) -> None:
    inputs = {name: np.zeros(shape, np.float32) for name, shape in script.INPUT_SHAPES.items()}

    tensors = script.input_tensors(inputs)

    assert [array.shape for array, _ in tensors] == [(1, 1, 512, 128), (1, 1, 512, 256)] * 2
    assert all(options == {"copy": True, "memory": script.pyneat.TensorMemory.EV74} for _, options in tensors)


def test_the_benchmark_measures_one_fixed_val_pair(tmp_path: Path, script: ModuleType) -> None:
    _write_val(tmp_path / "yfcc100m-keypoints-val", {f"s/{index}.jpg": 512 for index in range(4)} | {"s/4.jpg": 300})

    (first, images), (second, _) = script.benchmark_pair(tmp_path), script.benchmark_pair(tmp_path)

    assert (first.name0, first.name1) == (second.name0, second.name1)
    assert "s/4.jpg" not in (first.name0, first.name1)
    assert [array.shape for image in images for array in image] == [(512, 2), (512, 256)] * 2


def test_the_model_takes_tensors_without_image_preprocessing(script: ModuleType) -> None:
    options = script.tensor_options()

    assert options.preprocess.kind is script.pyneat.InputKind.Tensor
    off = script.pyneat.AutoFlag.Off
    assert (options.preprocess.resize.enable, options.preprocess.normalize.enable) == (off, off)
    assert options.preprocess.color_convert.enable is off


def test_mla_stats_come_from_cores_benchmark_of_the_pair(script: ModuleType) -> None:
    tensors = script.pair_tensors(_images())

    script.mla_stats("model.tar.gz", tensors)

    options = script.pyneat.BenchmarkOptions.return_value
    assert (options.include_mla_layer_stats, options.include_mla_stage_stats) == (True, True)
    assert options.inputs == tensors
    script.pyneat.Model.return_value.benchmark.assert_called_once_with(options)


def test_every_timed_pair_encodes_the_keypoints_and_extracts_the_matches(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {"WARMUP_PAIRS": 1, "MEASURED_PAIRS": 3, "MIN_DURATION_S": 0}.items():
        monkeypatch.setattr(script, name, value)
    images = _images()
    expected = script.matcher_inputs(*images)
    runs: list[list] = []

    class Runner:
        def run(self, tensors: list, timeout_ms: int) -> list:
            runs.append(tensors)
            return [_assignment(_diagonal())]

    assert script.match_pair(Runner(), images).tolist() == [[index, index] for index in range(512)]
    latency = script.measure_latency(Runner(), images)

    # Each run gets the pair's encodings, computed again for it.
    for tensors in runs:
        for (array, _), name in zip(tensors, script.INPUT_SHAPES):
            np.testing.assert_array_equal(array.reshape(expected[name].shape), expected[name])
    assert len(runs) == 1 + 1 + 3
    assert len(latency.pair_s) == len(latency.model_s) == 3
    assert all(model <= pair for model, pair in zip(latency.model_s, latency.pair_s))


def test_throughput_runs_the_runner_async_and_matches_every_pulled_pair(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {"WARMUP_PAIRS": 2, "MEASURED_PAIRS": 5, "MIN_DURATION_S": 0, "MATCH_WORKERS": 2}.items():
        monkeypatch.setattr(script, name, value)
    encoded: list[int] = []
    monkeypatch.setattr(script, "pair_tensors", lambda images: encoded.append(1) or ["tensor"])
    matched: list[int] = []
    monkeypatch.setattr(script, "matches", lambda assignment: matched.append(1))

    class Runner:
        """An async runner holding at most four pairs: push blocks when full, pull ends after close_input."""

        def __init__(self) -> None:
            self.outputs: queue.Queue = queue.Queue(maxsize=4)
            self.pushed = 0

        def push(self, tensors: list) -> bool:
            assert tensors == ["tensor"]
            self.outputs.put([_assignment(np.zeros((512, 512), np.float32))])
            self.pushed += 1
            return True

        def close_input(self) -> None:
            self.outputs.put([])

        def pull(self, timeout_ms: int) -> SimpleNamespace:
            return SimpleNamespace(tensors=self.outputs.get(timeout=timeout_ms / 1000))

    runner = Runner()
    result = script.measure_throughput(runner, _images())

    assert result.pairs >= 5
    # Every pushed pair was encoded first, and every pair the pull loop drains is matched on the CPU.
    assert len(encoded) == runner.pushed == len(matched) >= result.pairs + 2


def test_the_report_holds_the_whole_pairs_timing_and_names_its_cpu_steps(
    tmp_path: Path, script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_val(tmp_path / "yfcc100m-keypoints-val", {f"s/{index}.jpg": 512 for index in range(4)})
    monkeypatch.setattr(script, "mla_stats", lambda mpk, tensors: ("layers", "stages"))
    monkeypatch.setattr(script, "measure_latency", lambda runner, images: script.LatencyResult(
        [0.010, 0.012, 0.014], [0.008, 0.009, 0.010]))
    telemetry = SimpleNamespace()
    monkeypatch.setattr(script, "measure_throughput", lambda runner, images: script.ThroughputResult(
        500, 4.0, telemetry))
    saved: list = []
    monkeypatch.setattr(script.model_report, "from_performance", lambda benchmark, models, **options: (
        saved.append((benchmark, models, options)) or SimpleNamespace(save=saved.append)
    ))
    dataset = f"yfcc100m-keypoints-val={tmp_path}"
    monkeypatch.setattr("sys.argv", ["script.py", "--mpk", "m.tar.gz", "--dataset", dataset, "--output", "out"])

    script.main()

    (benchmark, models, options), output = saved
    assert (benchmark.latency_ms, benchmark.fps, benchmark.throughput_samples) == pytest.approx((12.0, 125.0, 500))
    assert (options["mla_layer_stats"], options["mla_stage_stats"]) == ({"model": "layers"}, {"model": "stages"})
    assert options["telemetry"] is telemetry
    pair, _ = script.benchmark_pair(tmp_path)
    assert options["workload"].startswith(f"yfcc100m-keypoints-val pair {pair.name0} and {pair.name1}")
    assert "keypoint encoding on the CPU" in options["workload"]
    assert "mutual-match extraction on the CPU" in options["workload"]
    assert (models, output) == ({"model": script.pyneat.Model.return_value}, "out")
    script.pyneat.Model.return_value.build.return_value.close.assert_called_once_with()


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
