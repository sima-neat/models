"""Checks of script.py's dataset loading, block protocols and scoring."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

S3DIS_ORDER = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


@pytest.fixture
def script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def _room(size_x: float, size_y: float, points: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    xyz = (rng.uniform(0, 1, (points, 3)) * [size_x, size_y, 3]).astype(np.float32)
    return {"xyz": xyz, "rgb": rng.randint(0, 256, (points, 3)).astype(np.uint8),
            "label": rng.randint(0, 13, points).astype(np.uint8)}


def _write_dataset(directory: Path, rooms: dict[str, dict[str, np.ndarray]], classes=S3DIS_ORDER, **manifest) -> Path:
    """An s3dis val archive's layout: val/<area>/<room>/{xyz,rgb}.npy and annotations/<area>/<room>.npy."""
    directory.mkdir()
    for name, room in rooms.items():
        (directory / "val" / name).mkdir(parents=True)
        np.save(directory / "val" / name / "xyz.npy", room["xyz"])
        np.save(directory / "val" / name / "rgb.npy", room["rgb"])
        (directory / "annotations" / name).parent.mkdir(parents=True, exist_ok=True)
        np.save(directory / "annotations" / f"{name}.npy", room["label"])
    (directory / "annotations" / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    content = {"id": "s3dis", "variant": "val", "rooms": list(rooms)}
    (directory / "manifest.json").write_text(json.dumps({**content, **manifest}), encoding="utf-8")
    return directory


def test_labels_are_renumbered_to_the_models_class_order(tmp_path: Path, script: ModuleType) -> None:
    room = _room(2, 2, 100)
    room["label"] = np.array([S3DIS_ORDER.index(name) for name in ("table", "chair", "sofa", "bookcase")] * 25,
                             np.uint8)
    (name, loaded), = script.load_rooms(_write_dataset(tmp_path / "s3dis-val", {"Area_5/office_1": room}))
    assert name == "Area_5/office_1"
    assert [script.CLASS_NAMES[i] for i in loaded["label"][:4]] == ["table", "chair", "sofa", "bookcase"]


@pytest.mark.parametrize("case", ["area", "classes", "variant"])
def test_other_datasets_are_refused(tmp_path: Path, case: str, script: ModuleType) -> None:
    room = "Area_1/office_1" if case == "area" else "Area_5/office_1"
    classes = S3DIS_ORDER[:12] if case == "classes" else S3DIS_ORDER
    changes = {"variant": "cal"} if case == "variant" else {}
    directory = _write_dataset(tmp_path / "s3dis-val", {room: _room(2, 2, 10)}, classes, **changes)
    with pytest.raises(SystemExit):
        script.load_rooms(directory)


def test_tiles_give_one_block_per_occupied_metre(script: ModuleType) -> None:
    room = _room(2.5, 1.5, 20000)
    blocks = list(script.blocks(room, np.random.RandomState(0), "tiles"))
    # Columns start every metre and the last one ends at the room's edge: 3 x 2 of them.
    assert len(blocks) == 6
    for features, points in blocks:
        assert features.shape == (script.POINTS, script.FEATURES) and features.dtype == np.float32
        assert len(np.unique(points)) == script.POINTS  # enough points: no repeats
        # x, y centred on the 1 m column; rgb in [0, 1]; normalized xyz in [0, 1].
        assert np.abs(features[:, :2]).max() <= 0.5 + script.blocks.__globals__["PADDING"] + 1e-6
        assert 0 <= features[:, 3:].min() and features[:, 3:].max() <= 1


def test_small_columns_repeat_points_to_fill_a_block(script: ModuleType) -> None:
    (features, points), = list(script.blocks(_room(0.9, 0.9, 100), np.random.RandomState(0), "tiles"))
    assert len(points) == script.POINTS and set(points.tolist()) == set(range(100))


def test_whole_protocol_covers_every_point(script: ModuleType) -> None:
    room = _room(1.6, 1.2, 9000)
    points = np.concatenate([p for _, p in script.blocks(room, np.random.RandomState(0), "whole")])
    assert set(points.tolist()) == set(range(9000))
    assert len(points) % script.POINTS == 0


def test_scores_follow_their_definitions(script: ModuleType) -> None:
    confusion = np.zeros((13, 13), np.int64)
    confusion[0, 0], confusion[0, 1], confusion[1, 1] = 3, 1, 4
    result = script.scores(confusion)
    # IoU: class 0 is 3 / (4 + 3 - 3), class 1 is 4 / (4 + 5 - 4); absent classes are left out.
    assert result["miou"] == pytest.approx((3 / 4 + 4 / 5) / 2)
    assert result["overall_accuracy"] == pytest.approx(7 / 8)
    assert result["mean_class_accuracy"] == pytest.approx((3 / 4 + 1) / 2)


def test_evaluate_scores_only_the_points_a_block_holds(script: ModuleType) -> None:
    room = _room(2, 2, 20000)
    room["label"][:] = 2  # every point is a wall

    def walls(features, indices):
        logits = np.zeros((len(features), 13), np.float32)
        logits[:, 2] = 1
        return logits

    metrics, per_class, count = script.evaluate([("office_1", room)], walls)
    assert count == 4 and metrics["overall_accuracy"] == 1.0 and per_class[2] == 1.0


def test_plan_workers_give_the_serial_votes(script: ModuleType, model_dir: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    # Spawned workers import pointcloud by name, as they do beside accuracy.py on the DevKit.
    monkeypatch.syspath_prepend(str(model_dir / "scripts"))
    monkeypatch.delitem(sys.modules, "pointcloud", raising=False)
    # planned() is pointcloud's; give it the plan of the module the workers import.
    monkeypatch.setitem(script.planned.__globals__, "plan", importlib.import_module("pointcloud").plan)
    room = _room(2.5, 1.5, 20000)

    def by_plan(features, indices):
        # Class of each point: which centre of the last level is nearest, so the votes depend on the plan.
        logits = np.zeros((len(features), 13), np.float32)
        logits[np.arange(len(features)), indices["interpolation"][0][0][:, 0] % 13] = 1
        return logits

    serial = script.evaluate([("office_1", room)], by_plan)
    assert script.evaluate([("office_1", room)], by_plan, workers=2) == serial


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
