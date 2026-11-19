from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def _ramp_logits(accuracy: ModuleType) -> np.ndarray:
    """Class 1 wins on the left half of the logits and class 0 on the right; the rest never win."""
    logits = np.full((accuracy.CLASS_COUNT, accuracy.LOGITS_SIZE, accuracy.LOGITS_SIZE), -1e3, np.float32)
    column = np.arange(accuracy.LOGITS_SIZE, dtype=np.float32)
    logits[0] = column
    logits[1] = accuracy.LOGITS_SIZE - 1 - column
    return logits


def test_segment_interpolates_the_logits_before_the_argmax(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    classes = accuracy.segment(_ramp_logits(accuracy), 40, 256)

    # At twice the width, output column i samples logit column (i + 0.5) / 2 - 0.5; the classes tie at logit column
    # 63.5, which output column 127.5 samples.
    assert classes.shape == (40, 256) and classes.dtype == np.uint8
    assert (classes[:, :128] == 1).all() and (classes[:, 128:] == 0).all()


def _every_class_argmax(accuracy: ModuleType, logits: np.ndarray, height: int, width: int) -> np.ndarray:
    """The argmax of all 150 classes interpolated to height x width, without pruning any class."""
    left, right, across = accuracy._interpolation(accuracy.LOGITS_SIZE, width)
    top, bottom, down = accuracy._interpolation(accuracy.LOGITS_SIZE, height)
    columns = logits[:, :, left] * (1 - across) + logits[:, :, right] * across
    down = down[None, :, None]
    return (columns[:, top] * (1 - down) + columns[:, bottom] * down).argmax(axis=0)


def test_segment_interpolates_only_classes_that_can_win_and_gets_the_same_argmax(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(0)
    size = (accuracy.CLASS_COUNT, accuracy.LOGITS_SIZE, accuracy.LOGITS_SIZE)
    noisy = rng.standard_normal(size).astype(np.float32)
    # Like a trained model's: four classes own regions, and the classes absent from the image sit far below them.
    peaked = rng.standard_normal(size).astype(np.float32) - 20
    regions = rng.integers(0, 4, (accuracy.LOGITS_SIZE // 16, accuracy.LOGITS_SIZE // 16)).repeat(16, 0).repeat(16, 1)
    owners = [3, 40, 77, 149]
    for index, owner in enumerate(owners):
        peaked[owner] = rng.standard_normal(size[1:]) + 10 * (regions == index)

    assert len(accuracy.candidate_classes(noisy)) == accuracy.CLASS_COUNT
    assert list(accuracy.candidate_classes(peaked)) == owners
    for logits in (noisy, peaked):
        for height, width in ((70, 90), (300, 201)):
            np.testing.assert_array_equal(
                accuracy.segment(logits, height, width), _every_class_argmax(accuracy, logits, height, width)
            )


def test_segment_rejects_unexpected_logits(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    logits = _ramp_logits(accuracy)

    with pytest.raises(ValueError, match="logits"):
        accuracy.segment(logits[:, :64], 10, 10)
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        accuracy.segment(logits, 10, 10)


def test_miou_ignores_unlabelled_pixels_and_averages_the_present_classes(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    # Label 0 is unlabelled; labels 1 and 2 are classes 0 and 1.
    label = np.array([[0, 1, 1], [2, 2, 1]], np.uint8)
    classes = np.array([[7, 0, 1], [1, 1, 0]], np.uint8)

    counts = accuracy.confusion(classes, label)
    measured = accuracy.scores(counts)

    # Class 0: 2 hits of 3 labelled, plus no false hits (union 3). Class 1: 2 hits of 2, plus 1 false hit (union 3).
    assert counts.sum() == 5
    assert measured["miou"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert measured["pixel_accuracy"] == pytest.approx(4 / 5)
    assert measured["classes"] == 2


def test_confusion_rejects_mismatched_or_out_of_range_labels(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    with pytest.raises(ValueError, match="differ in size"):
        accuracy.confusion(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8))
    with pytest.raises(ValueError, match="label values"):
        accuracy.confusion(np.zeros((1, 1), np.uint8), np.full((1, 1), 151, np.uint8))


def test_logits_come_back_channels_first_from_either_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    logits = _ramp_logits(accuracy)

    np.testing.assert_array_equal(accuracy.channels_first(logits[None]), logits)
    np.testing.assert_array_equal(accuracy.channels_first(logits.transpose(1, 2, 0)[None]), logits)
    with pytest.raises(ValueError, match="unexpected logits shape"):
        accuracy.channels_first(logits.reshape(-1))


def test_validation_images_pair_each_image_with_its_label_map(
    load_script: Callable[[str], ModuleType], tmp_path: Path
) -> None:
    # The DevKit's test environment has no Pillow; the scoring tests above do not need it.
    image = pytest.importorskip("PIL.Image")
    accuracy = load_script("accuracy")
    root = tmp_path / accuracy.DATASET
    for folder, suffix in (("images", "jpg"), ("annotations", "png")):
        (root / folder / accuracy.SPLIT).mkdir(parents=True)
        for name in ("ADE_val_00000002", "ADE_val_00000001"):
            image.new("L", (4, 3)).save(root / folder / accuracy.SPLIT / f"{name}.{suffix}")

    # The archive's top folder or the folder itself.
    for directory in (tmp_path, root):
        found_root, images = accuracy.validation_images(directory)
        assert found_root == root
        assert [image.stem for image in images] == ["ADE_val_00000001", "ADE_val_00000002"]
    assert accuracy.label_map(root, images[0]).shape == (3, 4)
    (root / "annotations" / accuracy.SPLIT / "ADE_val_00000002.png").unlink()
    with pytest.raises(SystemExit, match="label map"):
        accuracy.validation_images(tmp_path)


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.parse_mpks(["model=/data/fcn_hrnet_w18_mpk.tar.gz"]) == Path("/data/fcn_hrnet_w18_mpk.tar.gz")
    with pytest.raises(SystemExit, match="model=<path>"):
        accuracy.parse_mpks(["model=/a", "model=/b"])


def test_fp32_baseline_cuts_the_conversion_and_feeds_it_the_preprocessed_nchw_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    # The DevKit runs these tests in PyNeat, which has none of these packages; the host checks run this one.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pil = pytest.importorskip("PIL.Image")
    baseline = load_script("fp32_baseline_accuracy")
    calibration = load_script("calibration")
    # The baseline imports surgery.py when it runs, from its own folder; load_script imports only while loading.
    monkeypatch.setitem(sys.modules, "surgery", load_script("surgery"))
    helper, numpy_helper = onnx.helper, onnx.numpy_helper
    size = calibration.IMAGE_SIZE
    # Shaped like the conversion: a stride-4 Conv that sums each 4x4 patch into all 150 classes, then the resize to
    # the input that surgery.py cuts.
    weight = numpy_helper.from_array(np.ones((150, 3, 4, 4), np.float32), "weight")
    scales = numpy_helper.from_array(np.array([1, 1, 4, 4], np.float32), "scales")
    resize = {"mode": "linear", "coordinate_transformation_mode": "half_pixel"}
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["image", "weight"], ["head"], strides=[4, 4]),
            helper.make_node("Resize", ["head", "", "scales"], ["logits"], **resize),
        ],
        "converted",
        [helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 150, size, size])],
        [weight, scales],
    )
    model = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), model)
    image = tmp_path / "image.jpg"
    pil.new("RGB", (64, 48), (200, 30, 90)).save(image, quality=100)

    (logits,) = baseline.onnx_logits(model, [image])

    pixels = calibration.preprocess_image(image.read_bytes()).transpose(2, 0, 1)[None]
    patches = pixels.reshape(1, 3, size // 4, 4, size // 4, 4).sum(axis=(1, 3, 5))
    expected = np.broadcast_to(patches[:, None], (1, 150, size // 4, size // 4))
    np.testing.assert_allclose(logits, expected, rtol=1e-5, atol=1e-3)


def test_fp32_baseline_scores_with_the_accuracy_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    root, images = Path("/data/ade20k-val"), [Path("/data/ade20k-val/images/validation/ADE_val_00000001.jpg")]
    counts = np.zeros((baseline.accuracy.CLASS_COUNT,) * 2, np.int64)
    counts[0, 0], counts[1, 1], counts[1, 0] = 6, 3, 1
    calls = []

    def accumulate(dataset_root: Path, items: list[Path], outputs: Any) -> np.ndarray:
        calls.append((dataset_root, items, list(outputs)))
        return counts

    monkeypatch.setattr(baseline.accuracy, "parse_dataset", lambda parser, value, limit: (root, images, 2000))
    monkeypatch.setattr(baseline.accuracy, "accumulate", accumulate)
    monkeypatch.setattr(baseline, "onnx_logits", lambda onnx, items: iter([f"logits from {onnx.name}"]))
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "model.onnx", "--dataset", "ade20k-val=/data", "--output", str(output)])

    measured = baseline.accuracy.scores(counts)
    assert calls == [(root, images, ["logits from model.onnx"])]
    assert json.loads(output.read_text()) == {
        "dataset": "ade20k-val",
        "images": 1,
        "metrics": {"miou": round(measured["miou"], 4), "pixel_accuracy": round(measured["pixel_accuracy"], 4)},
    }


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
    result = accuracy["datasets"]["ade20k-val"]
    assert result["evaluated_samples"] == result["available_samples"] == 2000
