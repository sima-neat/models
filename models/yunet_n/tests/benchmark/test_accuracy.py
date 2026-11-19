from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def _empty_maps(accuracy: ModuleType) -> dict[str, np.ndarray]:
    """Head maps with no face anywhere: zero scores and zero offsets."""
    return {
        name: np.zeros((accuracy.GRIDS[int(name.split("_")[1])] ** 2, accuracy.HEADS[name.split("_")[0]]), np.float32)
        for name in accuracy.OUTPUTS
    }


def test_decode_places_a_cell_as_facedetectoryn_does(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    maps = _empty_maps(accuracy)
    # Stride 16, row 3, column 5: cell 3 * 40 + 5.
    cell = 3 * 40 + 5
    maps["cls_16"][cell], maps["obj_16"][cell] = 0.81, 0.64
    maps["bbox_16"][cell] = [0.5, 0.25, np.log(2.0), np.log(3.0)]
    maps["kps_16"][cell] = [0.5, 0.5, 1, 1, 2, 2, 3, 3, -1, -1]

    boxes, scores, landmarks = accuracy.decode(maps)

    index = 6400 + cell
    # Centre (5.5, 3.25) cells, size 2 x 3 cells, at 16 pixels per cell.
    np.testing.assert_allclose(boxes[index], [5.5 * 16 - 16, 3.25 * 16 - 24, 32, 48], rtol=1e-6)
    assert scores[index] == pytest.approx(np.sqrt(0.81 * 0.64))
    np.testing.assert_allclose(landmarks[index], [[88, 56], [96, 64], [112, 80], [128, 96], [64, 32]])
    assert scores.shape == (6400 + 1600 + 400,) and np.count_nonzero(scores) == 1


def test_detections_return_to_image_pixels_through_the_letterbox(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    maps = _empty_maps(accuracy)
    cell = 20 * 80 + 40
    maps["cls_8"][cell] = maps["obj_8"][cell] = 0.9
    maps["bbox_8"][cell] = [0.0, 0.0, np.log(4.0), np.log(4.0)]

    # A 1024x768 image fits the input at gain 0.625 with 80 rows of padding above it.
    (row,) = accuracy.detect(maps, 1024, 768)

    np.testing.assert_allclose(row, [(320 - 16) / 0.625, (160 - 16 - 80) / 0.625, 51.2, 51.2, 0.9], atol=1e-9)


def test_head_maps_reject_an_output_of_the_wrong_size(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [np.zeros((1, accuracy.GRIDS[int(name.split("_")[1])], accuracy.GRIDS[int(name.split("_")[1])],
                         accuracy.HEADS[name.split("_")[0]]), np.float32) for name in accuracy.OUTPUTS]

    assert accuracy.head_maps(outputs)["kps_32"].shape == (400, 10)
    outputs[-1] = outputs[-1][..., :8]
    with pytest.raises(RuntimeError, match="unexpected kps_32 shape"):
        accuracy.head_maps(outputs)


def test_dataset_is_the_extracted_folder_or_the_directory_holding_it(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    root = tmp_path / "wider-face-val"
    for required in (accuracy.IMAGES, accuracy.GROUND_TRUTH):
        (root / required).mkdir(parents=True)
    parser = argparse.ArgumentParser()

    assert accuracy.parse_dataset(parser, f"wider-face-val={tmp_path}") == root
    assert accuracy.parse_dataset(parser, f"wider-face-val={root}") == root
    with pytest.raises(SystemExit):
        accuracy.parse_dataset(parser, f"wider-face-val={tmp_path / 'missing'}")


def _truth() -> list:
    """Two images: three faces the hard subset scores (one also easy), and one image without faces."""
    faces = np.array([[10, 10, 40, 40], [100, 100, 20, 20], [200, 50, 30, 30]], float)
    keep = {"easy": np.array([1]), "medium": np.array([1, 2]), "hard": np.array([1, 2, 3])}
    empty = {setting: np.zeros(0, int) for setting in ("easy", "medium", "hard")}
    return [("0--Parade", "a", faces, keep), ("0--Parade", "b", np.zeros((0, 4)), empty)]


def test_wider_face_ap_rewards_found_faces_and_ignores_unscored_ones(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    truth = _truth()
    perfect = {("0--Parade", "a"): np.array([[10, 10, 40, 40, 0.9], [100, 100, 20, 20, 0.8], [200, 50, 30, 30, 0.7]])}
    missing = {("0--Parade", "a"): np.array([[10, 10, 40, 40, 0.9], [100, 100, 20, 20, 0.8]])}

    assert accuracy.evaluate(perfect, truth) == pytest.approx({"easy": 1.0, "medium": 1.0, "hard": 1.0})
    found = accuracy.evaluate(missing, truth)
    # The missed third face caps the hard subset's recall at 2 of 3; the other subsets never score it.
    assert (found["easy"], found["medium"]) == pytest.approx((1.0, 1.0))
    assert found["hard"] == pytest.approx(2 / 3)


def test_fp32_baseline_feeds_the_onnx_the_letterboxed_nchw_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    pytest.importorskip("onnxruntime")
    baseline = load_script("fp32_baseline_accuracy")
    received = []

    class Session:
        def __init__(self, path, providers):
            pass

        def get_inputs(self):
            return [type("Value", (), {"name": "input"})]

        def get_outputs(self):
            return [type("Value", (), {"name": name}) for name in baseline.accuracy.OUTPUTS]

        def run(self, names, feeds):
            received.append(feeds["input"])
            return [np.zeros((1, *head.shape), np.float32) for head in _empty_maps(baseline.accuracy).values()]

    monkeypatch.setattr("onnxruntime.InferenceSession", Session)
    image = np.full((640, 640, 3), 7.0, np.float32)
    monkeypatch.setattr(baseline.accuracy, "load_image", lambda root, item: (image, 1024, 768))

    (detections,) = baseline.onnx_detections(tmp_path / "model.onnx", tmp_path, [("0--Parade", "a")])

    np.testing.assert_array_equal(received[0], image.transpose(2, 0, 1)[None])
    assert detections.shape == (0, 5)


def test_fp32_baseline_scores_with_the_accuracy_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    truth = _truth()
    rows = [np.array([[10, 10, 40, 40, 0.9], [100, 100, 20, 20, 0.8]]), np.zeros((0, 5))]
    monkeypatch.setattr(baseline.accuracy, "parse_dataset", lambda parser, value: tmp_path)
    monkeypatch.setattr(baseline.accuracy, "load_ground_truth", lambda root: truth)
    monkeypatch.setattr(baseline, "onnx_detections", lambda onnx, root, items: iter(rows))
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "model.onnx", "--dataset", "wider-face-val=/data", "--output", str(output)])

    assert json.loads(output.read_text()) == {
        "dataset": "wider-face-val",
        "images": 2,
        "metrics": {"widerface_ap_easy": 1.0, "widerface_ap_medium": 1.0, "widerface_ap_hard": 0.6667},
    }


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
