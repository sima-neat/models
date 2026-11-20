from __future__ import annotations

import io
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    variant = recipe._variant(recipe._metadata(), profile_id)

    assert variant["parameters"]["precision"] == profile_id
    assert variant["parameters"]["input_shape"] == [1, 1, 28, 28]
    if profile_id == "w8a8":
        assert variant["parameters"]["calibration_samples"] == 500


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="exactly one build profile named 'w8a16'"):
        recipe._variant(recipe._metadata(), "w8a16")


@pytest.mark.parametrize(("field", "value", "message"), [
    ("precision", "w8a16", "only the declared Modalix BF16 and INT8 variants"),
    ("input_shape", [1, 1, 32, 32], "unsupported MNIST input shape"),
])
def test_compile_rejects_an_undeclared_precision_or_shape_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    field: str,
    value: object,
    message: str,
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["components"][0]["parameters"][field] = value
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match=message):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile["id"], None)
    assert not (tmp_path / "model").exists()


def test_int8_requires_the_calibration_dataset(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="requires --dataset"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w8a8", None)
    assert not (tmp_path / "model").exists()


def test_compile_main_rejects_an_invocation_named_otherwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(sys, "argv", [
        "compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"),
        "--build-profile", "w16a16", "--compile-invocation", "model",
    ])

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err


# calibration.py, the MNIST reader compile.py and accuracy.py share.


def _idx(images: np.ndarray, labels: np.ndarray) -> tuple[bytes, bytes]:
    image_header = np.array([0x803, len(images), 28, 28], dtype=">u4").tobytes()
    label_header = np.array([0x801, len(labels)], dtype=">u4").tobytes()
    return image_header + images.astype(np.uint8).tobytes(), label_header + labels.astype(np.uint8).tobytes()


@pytest.fixture
def calibration(load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_script("calibration")
    # Small splits keep the fixtures tiny; the reader checks every count against SPLITS.
    monkeypatch.setitem(module.SPLITS, "train", ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 6))
    monkeypatch.setitem(module.SPLITS, "test", ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 4))
    return module


def _write(directory: Path, split: str, images: np.ndarray, labels: np.ndarray, calibration: ModuleType) -> None:
    image_bytes, label_bytes = _idx(images, labels)
    image_name, label_name, _ = calibration.SPLITS[split]
    (directory / image_name).write_bytes(image_bytes)
    (directory / label_name).write_bytes(label_bytes)


def test_read_split_returns_images_and_labels_from_a_directory_or_the_archive(
    calibration: ModuleType, tmp_path: Path
) -> None:
    images = np.arange(4 * 28 * 28).reshape(4, 28, 28) % 256
    labels = np.array([7, 2, 1, 0])
    _write(tmp_path, "test", images, labels, calibration)

    read_images, read_labels = calibration.read_split(tmp_path, "test")
    np.testing.assert_array_equal(read_images, images)
    np.testing.assert_array_equal(read_labels, labels)

    archive = tmp_path / "mnist.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in calibration.SPLITS["test"][:2]:
            content = (tmp_path / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    np.testing.assert_array_equal(calibration.read_split(archive, "test")[1], labels)


def test_read_split_rejects_a_wrong_count_or_label(calibration: ModuleType, tmp_path: Path) -> None:
    _write(tmp_path, "test", np.zeros((3, 28, 28)), np.zeros(3), calibration)
    with pytest.raises(ValueError, match="is not 4 28x28 MNIST images"):
        calibration.read_split(tmp_path, "test")
    _write(tmp_path, "test", np.zeros((4, 28, 28)), np.array([1, 2, 3, 10]), calibration)
    with pytest.raises(ValueError, match="label outside 0-9"):
        calibration.read_split(tmp_path, "test")


def test_preprocessing_scales_to_unit_range_in_nhwc(calibration: ModuleType) -> None:
    result = calibration.preprocess(np.array([[[0, 255] * 14] * 28], dtype=np.uint8))
    assert result.shape == (1, 28, 28, 1) and result.dtype == np.float32
    assert (result.min(), result.max()) == (0.0, 1.0)


def test_calibration_takes_the_first_training_images(calibration: ModuleType, tmp_path: Path) -> None:
    images = np.stack([np.full((28, 28), value) for value in range(6)])
    _write(tmp_path, "train", images, np.arange(6), calibration)

    samples = calibration.calibration_samples(tmp_path, 3)
    assert samples.shape == (3, 28, 28, 1)
    np.testing.assert_allclose(samples[:, 0, 0, 0], np.arange(3) / 255, rtol=1e-6)
    with pytest.raises(ValueError, match="calibration requires 7 images"):
        calibration.calibration_samples(tmp_path, 7)
