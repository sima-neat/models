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

# MediaPipe's PoseLandmark order (mediapipe/python/solutions/pose.py) and COCO's keypoint order.
MEDIAPIPE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle",
]


def _outputs(records: np.ndarray | None = None, presence: float = 0.9) -> list[np.ndarray]:
    """The model's three outputs as the board returns them: landmark records, presence and world landmarks."""
    landmarks = np.zeros((39, 5), np.float32) if records is None else records
    return [landmarks.reshape(1, 195), np.full((1, 1), presence, np.float32), np.zeros((1, 117), np.float32)]


def test_coco_keypoints_are_the_same_named_mediapipe_landmarks(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.COCO_LANDMARKS == [MEDIAPIPE_LANDMARKS.index(name) for name in COCO_KEYPOINTS]


def test_keypoints_map_the_coco_landmarks_from_the_crop_to_the_image(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    records = np.zeros((39, 5), np.float32)
    # Landmark i sits at (i, 2i) in the 256 px input with visibility logit 2; records 33-38 are auxiliary.
    records[:, 0] = np.arange(39)
    records[:, 1] = 2 * np.arange(39)
    records[:, 3] = 2.0
    records[33:, :2] = 1000.0

    points, presence = accuracy.keypoints(_outputs(records, presence=0.8), 70.0, 40.0, 512.0)

    # The 512 px square starts at (70, 40), so an input pixel is 2 image pixels: the nose (landmark 0) is at
    # (70, 40) and the right ankle (landmark 28) at (70 + 2 * 28, 40 + 4 * 28).
    assert points.shape == (17, 3)
    np.testing.assert_allclose(points[0], [70, 40, 1 / (1 + np.exp(-2.0))], rtol=1e-6)
    np.testing.assert_allclose(points[16, :2], [126, 152])
    landmarks = np.array(accuracy.COCO_LANDMARKS)
    np.testing.assert_allclose(points[:, :2], np.column_stack([70 + 2 * landmarks, 40 + 4 * landmarks]))
    # Identity_1 is already a probability and scores the person as it is.
    assert presence == pytest.approx(0.8)


def test_keypoints_read_the_outputs_by_size_in_any_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [output.reshape(1, 1, 1, -1) for output in _outputs()]

    points, _ = accuracy.keypoints(outputs, 0.0, 0.0, 256.0)

    assert points.shape == (17, 3)


@pytest.mark.parametrize(("output", "value"), [(0, np.nan), (0, np.inf), (1, np.nan), (2, -np.inf)])
def test_keypoints_reject_non_finite_outputs(
    output: int, value: float, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    outputs = _outputs()
    outputs[output].reshape(-1)[0] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.keypoints(outputs, 0.0, 0.0, 256.0)


@pytest.mark.parametrize("order", [[2, 1, 0], [0, 1], [1, 0, 2]])
def test_keypoints_reject_other_outputs(order: list[int], load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _outputs()

    with pytest.raises(ValueError, match="unexpected model outputs"):
        accuracy.keypoints([outputs[index] for index in order], 0.0, 0.0, 256.0)


def test_scored_people_are_non_crowd_with_labelled_keypoints(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    annotations = [
        {"id": 1, "image_id": 5, "num_keypoints": 3, "iscrowd": 0},
        {"id": 2, "image_id": 5, "num_keypoints": 0, "iscrowd": 0},
        {"id": 3, "image_id": 5, "num_keypoints": 0, "iscrowd": 1},
        {"id": 4, "image_id": 2, "num_keypoints": 17, "iscrowd": 0},
        {"id": 5, "image_id": 9, "num_keypoints": 4, "iscrowd": 0},
    ]

    scored = accuracy.people(annotations, [2, 5])

    assert {image: [person["id"] for person in persons] for image, persons in scored.items()} == {2: [4], 5: [1]}


def test_scored_images_are_those_with_labelled_keypoints(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    ground_truth = types.SimpleNamespace(dataset={"annotations": [
        {"image_id": 5, "num_keypoints": 3}, {"image_id": 5, "num_keypoints": 0},
        {"image_id": 2, "num_keypoints": 17}, {"image_id": 9, "num_keypoints": 0},
    ]})
    monkeypatch.setattr(accuracy, "SCORED_IMAGES", 2)

    assert accuracy.scored_image_ids(ground_truth) == [2, 5]
    monkeypatch.setattr(accuracy, "SCORED_IMAGES", 2346)
    with pytest.raises(SystemExit, match="expected 2346 images"):
        accuracy.scored_image_ids(ground_truth)


# One 640x480 grey image (value 51) with a 60x120 person at (100, 80), a 40x40 one at (600, 440) whose square runs
# past the image's right and bottom edges, a crowd and an unlabelled person; a second image without people is not
# scored.
PEOPLE = [
    {"id": 11, "image_id": 1, "iscrowd": 0, "num_keypoints": 2, "bbox": [100.0, 80.0, 60.0, 120.0]},
    {"id": 12, "image_id": 1, "iscrowd": 0, "num_keypoints": 1, "bbox": [600.0, 440.0, 40.0, 40.0]},
    {"id": 13, "image_id": 1, "iscrowd": 1, "num_keypoints": 0, "bbox": [0.0, 0.0, 50.0, 50.0]},
    {"id": 14, "image_id": 1, "iscrowd": 0, "num_keypoints": 0, "bbox": [300.0, 0.0, 20.0, 50.0]},
]


def _coco(directory: Path) -> types.SimpleNamespace:
    # Imported here because the DevKit's test environment has no OpenCV, and the stage test below needs none.
    import cv2

    (directory / "val2017").mkdir(parents=True)
    cv2.imwrite(str(directory / "val2017/000000000001.png"), np.full((480, 640, 3), 51, np.uint8))
    return types.SimpleNamespace(dataset={
        "images": [{"id": 1, "file_name": "000000000001.png"}, {"id": 2, "file_name": "000000000002.jpg"}],
        "annotations": PEOPLE,
        "categories": [{"id": 1, "name": "person"}],
    })


def _model(crops: list[np.ndarray]) -> Callable[[np.ndarray], list[np.ndarray]]:
    """A model that records each crop and returns every landmark at the crop centre, visibility logit 0."""
    presence = iter([0.75, 0.5])

    def infer(pixels: np.ndarray) -> list[np.ndarray]:
        crops.append(pixels)
        records = np.zeros((39, 5), np.float32)
        records[:, :2] = 128.0
        return _outputs(records, presence=next(presence))

    return infer


def _scored(monkeypatch: pytest.MonkeyPatch, accuracy: ModuleType, ground_truth: types.SimpleNamespace) -> list:
    """Replace COCO scoring, which the board's faster-coco-eval does, and record what it receives."""
    calls: list = []

    def score(truth: Any, detections: list[dict], image_ids: list[int]) -> dict[str, float]:
        calls.append((truth, detections, image_ids))
        return dict.fromkeys(accuracy.METRICS, 0.5)

    monkeypatch.setattr(accuracy, "load_ground_truth", lambda dataset: ground_truth)
    monkeypatch.setattr(accuracy, "SCORED_IMAGES", 1)
    monkeypatch.setattr(accuracy, "score", score)
    return calls


def _assert_people_cropped_and_mapped_back(crops: list[np.ndarray], calls: list) -> None:
    # Only the two labelled non-crowd people run, each as a 256x256 crop of the grey image in [0, 1]; past the
    # image's edges, the second person's 50 px square repeats the grey.
    assert [(pixels.shape, pixels.dtype) for pixels in crops] == [((256, 256, 3), np.float32)] * 2
    for pixels in crops:
        np.testing.assert_allclose(pixels, 51 / 255, atol=1e-6)
    ((_, detections, image_ids),) = calls
    assert image_ids == [1]
    # A landmark at the crop centre maps to the box centre: (130, 140) and (620, 460).
    assert [(person["image_id"], person["category_id"], person["score"]) for person in detections] == [
        (1, 1, 0.75), (1, 1, 0.5)
    ]
    for person, centre in zip(detections, [(130, 140), (620, 460)]):
        np.testing.assert_allclose(np.reshape(person["keypoints"], (17, 3)), [[*centre, 0.5]] * 17)


def test_accuracy_crops_each_person_for_the_mpk_and_reports_coco_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    calls = _scored(monkeypatch, accuracy, _coco(tmp_path / "coco"))
    crops: list[np.ndarray] = []
    infer = _model(crops)

    class Runner:
        def run(self, tensors: list[np.ndarray], timeout_ms: int) -> list[types.SimpleNamespace]:
            (pixels,) = tensors
            return [types.SimpleNamespace(to_numpy=lambda copy, output=output: output) for output in infer(pixels)]

    class Model:
        def __init__(self, path: str) -> None:
            # The crops arrive as model input, so the script keeps Neat's default options.
            assert path == "blazepose_ghum_heavy_mpk.tar.gz"

        def build(self, tensors: list[np.ndarray]) -> Runner:
            return Runner()

    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        __version__="3.0.0", build_info=lambda: {"version": "3.0.0"},
        Model=Model, Tensor=types.SimpleNamespace(from_numpy=lambda array, copy: array),
    ))

    accuracy.main([
        "--mpk", "model=blazepose_ghum_heavy_mpk.tar.gz", "--dataset", f"coco-2017-val={tmp_path / 'coco'}",
        "--output", str(tmp_path / "output"),
    ])

    _assert_people_cropped_and_mapped_back(crops, calls)
    report = json.loads((tmp_path / "output/model-report.json").read_text(encoding="utf-8"))
    result = report["datasets"]["coco-2017-val"]
    assert (result["split"], result["evaluated_samples"], result["available_samples"]) == ("val2017", 1, 1)
    assert result["details"]["people"] == 2
    assert result["metrics"] == dict.fromkeys(accuracy.METRICS, 0.5)


def test_fp32_baseline_scores_the_onnx_as_accuracy_scores_the_mpk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    accuracy = baseline.accuracy
    calls = _scored(monkeypatch, accuracy, _coco(tmp_path / "coco"))
    crops: list[np.ndarray] = []
    infer = _model(crops)
    requests: list[tuple[list[str], tuple[int, ...]]] = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            assert path == "model.onnx"

        def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            (pixels,) = inputs.values()
            requests.append((names, pixels.shape))
            return infer(pixels[0])

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=Session))
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "model.onnx", "--dataset", f"coco-2017-val={tmp_path / 'coco'}", "--output", str(output)])

    # The ONNX takes the crop channels-last with a batch axis, input_1 [1, 256, 256, 3], and returns the three
    # outputs the package keeps, in its order.
    assert requests == [(["Identity", "Identity_1", "Identity_4"], (1, 256, 256, 3))] * 2
    _assert_people_cropped_and_mapped_back(crops, calls)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result == {"images": 1, "people": 2, "metrics": dict.fromkeys(accuracy.METRICS, 0.5)}


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
