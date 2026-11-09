from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

# Ground truth for a pairs line: two identity intrinsics and an identity pose.
IDENTITY = " ".join(["1 0 0 0 1 0 0 0 1"] * 2 + ["1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"])


def _write_val(root: Path, images: int) -> Path:
    """A yfcc100m-keypoints val directory of 512-keypoint images, spread over the 640x480 input, and pairs of
    consecutive images."""
    names = [f"s/{index}.jpg" for index in range(images)]
    keypoints = np.stack(np.meshgrid(np.linspace(20, 620, 32), np.linspace(20, 460, 16)), -1).reshape(512, 2)
    for name in names:
        directory = root / "val" / Path(name).with_suffix("")
        directory.mkdir(parents=True)
        np.save(directory / "keypoints.npy", keypoints.astype(np.float32))
        np.save(directory / "descriptors.npy", np.full((512, 256), 1 / 16, np.float32))
    (root / "annotations").mkdir()
    rows = [f"{name} 1280 960 0 512" for name in names]
    (root / "annotations" / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    pairs = [f"{first} {second} 0 0 {IDENTITY}" for first, second in zip(names, names[1:])]
    (root / "annotations" / "pairs_val.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    return root


def _diagonal() -> np.ndarray:
    """An assignment in which each keypoint matches the keypoint of the same index."""
    values = np.full((1, 512, 512), 1e-4, np.float32)
    values[0, np.arange(512), np.arange(512)] = 0.9
    return values


def test_ground_truth_and_images_read_every_value(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    root = _write_val(tmp_path / "yfcc100m-keypoints-val", 2)
    pairs_file = root / "annotations" / "pairs_val.txt"

    ((pair, truth),) = script.read_ground_truth(root).items()
    assert (pair.name0, pair.name1) == ("s/0.jpg", "s/1.jpg")
    np.testing.assert_array_equal(truth.intrinsics0, np.eye(3))
    np.testing.assert_array_equal(truth.pose, np.eye(4))
    assert script.read_images(root)["s/0.jpg"] == script.Image(1280, 960, 0)
    pairs_file.write_text(pairs_file.read_text(encoding="utf-8").strip() + " 7", encoding="utf-8")
    with pytest.raises(ValueError, match="39 fields"):
        script.read_ground_truth(root)


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/lightglue_mpk.tar.gz"]) == Path("/data/lightglue_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["matcher=/data/lightglue_mpk.tar.gz"])


def test_pose_auc_integrates_the_recall_of_the_pose_errors(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    # Sorted errors 1, 2, 15 and a failure: recall 1/4, 1/2, 3/4, and 1 only at infinity.
    np.testing.assert_allclose(script.pose_auc([1.0, 2.0, np.inf, 15.0]), [0.4, 0.45, 0.61875])
    assert script.pose_auc([np.inf, np.inf]) == [0.0, 0.0, 0.0]


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def _rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def test_pose_error_takes_the_larger_angle_and_either_translation_sign(
    load_script: Callable[[str], ModuleType],
) -> None:
    script = load_script("accuracy")
    truth = np.eye(4)
    truth[:3, 3] = [2.0, 0.0, 0.0]

    assert script.pose_error(truth, np.eye(3), np.asarray([1.0, 0.0, 0.0])) == pytest.approx(0, abs=1e-6)
    assert script.pose_error(truth, np.eye(3), np.asarray([-1.0, 0.0, 0.0])) == pytest.approx(0, abs=1e-6)
    assert script.pose_error(truth, _rotation_z(10), np.asarray([1.0, 0.0, 0.0])) == pytest.approx(10)
    assert script.pose_error(truth, np.eye(3), np.asarray([1.0, 1.0, 0.0])) == pytest.approx(45)


def test_the_pose_estimate_recovers_a_known_relative_pose(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    rng = np.random.default_rng(0)
    intrinsics = np.asarray([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    rotation, translation = _rotation_z(5) @ _rotation_x(5), np.asarray([0.4, 0.1, 0.05])
    points = rng.uniform([-2, -2, 4], [2, 2, 8], (200, 3))
    projected0 = points @ intrinsics.T
    projected1 = (points @ rotation.T + translation) @ intrinsics.T
    pixels0, pixels1 = projected0[:, :2] / projected0[:, 2:], projected1[:, :2] / projected1[:, 2:]
    truth = np.eye(4)
    truth[:3, :3], truth[:3, 3] = rotation, translation

    estimate = script.estimate_pose(pixels0, pixels1, intrinsics, intrinsics, 1 / 500)

    assert estimate is not None and script.pose_error(truth, *estimate) < 1
    assert script.estimate_pose(pixels0[:4], pixels1[:4], intrinsics, intrinsics, 1 / 500) is None


def test_the_ransac_threshold_is_a_pixel_of_the_stretched_input(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    intrinsics = np.asarray([[1000.0, 0, 640], [0, 1000.0, 480], [0, 0, 1]])
    truth = script.GroundTruth(intrinsics, intrinsics, np.eye(4))
    # A 1280x960 image stretches to 640x480 at half its size, so its focal lengths halve; a turned 960x1280 image
    # stretches to 480x640, also at half its size.
    upright, turned = script.Image(1280, 960, 0), script.Image(960, 1280, 1)

    assert script.ransac_threshold(truth, upright, upright) == pytest.approx(1 / 500)
    assert script.ransac_threshold(truth, upright, turned) == pytest.approx(1 / 500)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_keypoints_return_to_the_original_image_pixels(rotation: int, load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    # The stretched image before its turns, each pixel holding its own (x, y), turned as the dataset turns it.
    width, height = (480, 640) if rotation % 2 else (640, 480)
    ys, xs = np.mgrid[0:height, 0:width]
    turned = np.rot90(np.stack([xs, ys], -1), k=rotation)
    assert turned.shape[:2] == (480, 640)
    points = np.asarray([[0, 0], [639, 0], [17, 400], [600, 479]])  # (x, y) in SuperPoint's turned input
    stretched = turned[points[:, 1], points[:, 0]]
    image = script.Image(width=width * 2, height=height * 3, rotation=rotation)

    np.testing.assert_allclose(script.original_pixels(points.astype(np.float64), image), stretched * [2, 3])


def test_the_mpk_takes_each_input_as_one_row_in_ev74_memory_and_scores_every_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    _write_val(tmp_path / "yfcc100m-keypoints-val", 3)
    pyneat = sys.modules["pyneat"]
    monkeypatch.setattr(pyneat.Tensor, "from_numpy", lambda array, **options: (array, options))
    runner = pyneat.Model.return_value.build.return_value
    runner.run.side_effect = lambda tensors, timeout_ms: [SimpleNamespace(to_numpy=lambda copy: _diagonal())]
    reports: list[dict] = []
    monkeypatch.setattr(script.model_report, "from_accuracy", lambda datasets: (
        reports.append(datasets) or SimpleNamespace(save=lambda output: None, document={"datasets": datasets})
    ))
    dataset = f"yfcc100m-keypoints-val={tmp_path}"

    assert script.main(["--mpk", "model=m.tar.gz", "--dataset", dataset, "--output", "out"]) == 0

    # The runner is built once, on the first pair, and runs both pairs.
    pyneat.Model.return_value.build.assert_called_once()
    assert runner.run.call_count == 2
    tensors = runner.run.call_args.args[0]
    assert [array.shape for array, _ in tensors] == [(1, 1, 512, 128), (1, 1, 512, 256)] * 2
    assert all(options == {"copy": True, "memory": pyneat.TensorMemory.EV74} for _, options in tensors)
    ((dataset_id, result),) = reports[0].items()
    assert dataset_id == "yfcc100m-keypoints-val"
    assert set(result["metrics"]) == {"pose_auc_5", "pose_auc_10", "pose_auc_20"}
    assert (result["evaluated_samples"], result["details"]["mean_matches"]) == (2, 512)


def test_the_fp32_baseline_scores_the_onnx_with_the_accuracy_scripts_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    _write_val(tmp_path / "yfcc100m-keypoints-val", 3)
    feeds: list[dict[str, np.ndarray]] = []
    names = ["encoding0", "descriptors0", "encoding1", "descriptors1"]

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name) for name in names]

        def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            feeds.append(inputs)
            return [_diagonal().reshape(1, 512, 1, 512)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    output = tmp_path / "fp32-baseline.json"
    arguments = ["--onnx", "model.onnx", "--dataset", f"yfcc100m-keypoints-val={tmp_path}", "--output", str(output)]

    assert baseline.main(arguments) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert (result["dataset"], result["pairs"], result["mean_matches"]) == ("yfcc100m-keypoints-val", 2, 512)
    assert set(result["metrics"]) == {"pose_auc_5", "pose_auc_10", "pose_auc_20"}
    # The ONNX gets the matcher's inputs channels-first, not channels-last as the MPK does.
    assert [{name: value.shape for name, value in inputs.items()} for inputs in feeds] == [
        {"encoding0": (1, 128, 1, 512), "descriptors0": (1, 256, 1, 512), "encoding1": (1, 128, 1, 512),
         "descriptors1": (1, 256, 1, 512)}] * 2
    names.reverse()
    with pytest.raises(SystemExit, match="expected"):
        baseline.main(arguments)


def test_the_fp32_baseline_reads_the_onnxs_assignment_with_image_0_keypoints_as_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    _write_val(tmp_path / "yfcc100m-keypoints-val", 2)
    assignment = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name) for name in baseline.INPUT_SHAPES]

        def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            # The ONNX writes entry (i, j) at [0, j, 0, i].
            return [assignment.T[None, :, None]]

    returned: list[np.ndarray] = []

    def evaluate(root: Path, pairs: list, assignment_of: Callable) -> tuple[dict[str, float], float]:
        returned.append(assignment_of({name: np.zeros(shape, np.float32)
                                       for name, shape in baseline.INPUT_SHAPES.items()}))
        return {}, 0.0

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    monkeypatch.setattr(baseline.accuracy, "evaluate", evaluate)
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "model.onnx", "--dataset", f"x={tmp_path}", "--output", str(output)])

    np.testing.assert_array_equal(returned, [assignment])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
