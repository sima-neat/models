from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls
from PIL import Image


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.parse_mpks(["model=/data/raft_stereo_realtime_mpk.tar.gz"]) == Path(
        "/data/raft_stereo_realtime_mpk.tar.gz"
    )
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        accuracy.parse_mpks(["left=/data/raft_stereo_realtime_mpk.tar.gz"])


def test_d1_all_counts_errors_over_both_3_px_and_5_percent(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    truth = np.array([[100.0, 100.0, 40.0, 40.0, 191.0, 0.0, -3.0, 192.0]], np.float32)
    disparity = truth + np.array([[-4.8, 6.0, 3.0, 3.5, 20.0, 9.0, 9.0, 20.0]], np.float32)

    # Of the five pixels between 0 and 192, 6 px off 100, 3.5 px off 40 and 20 px off 191 exceed both bounds; 4.8 px
    # under 100 is not over 5% of 100, only of the 95.2 px prediction, and 3 px off 40 is not over 3 px. The 0, -3 and
    # 192 px pixels are not counted, though 20 px off 192 would exceed both bounds too.
    assert accuracy.d1_all(disparity, truth) == pytest.approx(60.0)
    # A pair with no counted pixel scores 0, not NaN.
    assert accuracy.d1_all(disparity, np.zeros_like(truth)) == 0.0


def _png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (960, 540), (value, value, value)).save(path)


def _pfm(path: Path, disparity: np.ndarray) -> None:
    """A little-endian one-channel PFM, which stores its rows from the bottom."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Pf\n960 540\n-1.0\n" + np.flipud(disparity).astype("<f4").tobytes())


def _neat(disparities: list[np.ndarray], frames: list[np.ndarray]) -> types.SimpleNamespace:
    """PyNeat for the script: its runner records each input frame and returns the next disparity, NHWC as Neat does."""
    outputs = iter(disparities)

    class Runner:
        def run(self, tensors: list[np.ndarray], timeout_ms: int) -> list[types.SimpleNamespace]:
            (frame,) = tensors
            frames.append(frame)
            disparity = next(outputs).reshape(1, 540, 960, 1)
            return [types.SimpleNamespace(to_numpy=lambda copy=False: disparity)]

    class Model:
        def __init__(self, path: str) -> None:
            assert path == "raft_stereo_realtime_mpk.tar.gz"

        def build(self, tensors: list[np.ndarray]) -> Runner:
            return Runner()

    return types.SimpleNamespace(
        __version__="3.0.0", build_info=lambda: {"version": "3.0.0"},
        Model=Model, Tensor=types.SimpleNamespace(from_numpy=lambda array, copy: array),
    )


def test_each_pair_runs_side_by_side_and_scores_average_over_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    dataset = tmp_path / "scene_flow"
    # The first pair's truth counts only 10 pixels, 4 px off; the other two count every pixel, 1 px off.
    sparse = np.zeros((540, 960), np.float32)
    sparse[:2, :5] = 50.0
    dense = np.full((540, 960), 50.0, np.float32)
    lines = []
    for index, (left, right, truth) in enumerate([(60, 90, sparse), (30, 120, dense), (150, 210, dense)]):
        views = [f"FlyingThings3D/frames_finalpass/TEST/A/000{index}/{view}/0006.png" for view in ("left", "right")]
        disparity = f"FlyingThings3D/disparity/TEST/A/000{index}/left/0006.pfm"
        _png(dataset / views[0], left)
        _png(dataset / views[1], right)
        _pfm(dataset / disparity, truth)
        lines.append(f"{views[0]} {views[1]} {disparity}\n")
    (dataset / accuracy.TEST_LIST).write_text("".join(lines), encoding="utf-8")
    frames: list[np.ndarray] = []
    neat = _neat([sparse + 4, dense + 1, dense + 1], frames)
    # The script runs the MPK through its pyneat; model_report reads the runtime's version from the same module.
    monkeypatch.setattr(accuracy, "pyneat", neat)
    monkeypatch.setitem(sys.modules, "pyneat", neat)

    accuracy.main([
        "--mpk", "model=raft_stereo_realtime_mpk.tar.gz", "--dataset", f"scene-flow-val={dataset}",
        "--output", str(tmp_path / "output"),
    ])

    # The left view fills the first 960 columns of the frame and the right view the rest, each ImageNet-normalized.
    assert [(frame.shape, frame.dtype) for frame in frames] == [((540, 1920, 3), np.float32)] * 3
    scene_flow = load_script("scene_flow")
    np.testing.assert_allclose(frames[0][0, 959], (60 / 255 - scene_flow.MEAN) / scene_flow.STD, rtol=1e-6)
    np.testing.assert_allclose(frames[0][0, 960], (90 / 255 - scene_flow.MEAN) / scene_flow.STD, rtol=1e-6)
    report = json.loads((tmp_path / "output/model-report.json").read_text(encoding="utf-8"))
    result = report["datasets"]["scene-flow-val"]
    assert (result["split"], result["evaluated_samples"], result["available_samples"]) == ("test", 3, 3)
    # Each pair scores alone: EPE 4, 1 and 1 px, D1-all 100% (4 px is over 3 px and 5% of 50), 0% and 0%. Their mean
    # is 2 px and 33.3%; their median would be 1 px and 0%, and pooling the pixels about 1 px and 0%.
    assert result["metrics"] == pytest.approx({"epe": 2.0, "d1_all": 100 / 3})


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
