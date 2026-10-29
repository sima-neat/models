from __future__ import annotations

import io
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _image(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    import cv2

    image = np.zeros((height, width, 3), np.uint8)
    image[:] = colour[::-1]  # OpenCV writes BGR.
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_resizes_to_518x686_and_normalizes_with_imagenet_statistics(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")

    result = calibration.preprocess_image(_image(640, 480))

    assert (calibration.HEIGHT, calibration.WIDTH) == (518, 686)
    assert calibration.HEIGHT % 14 == 0 and calibration.WIDTH % 14 == 0
    assert result.shape == (518, 686, 3) and result.dtype == np.float32
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[259, 343], red, atol=1e-5)


def test_calibration_takes_the_first_cal_images_from_the_archive(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(
        tmp_path / "cal.tgz",
        {
            "nyu_depth_v2/cal/rgb_00009.jpg": _image(64, 48),
            "nyu_depth_v2/cal/rgb_00002.jpg": _image(64, 48),
            "nyu_depth_v2/annotations/depth_cal/sync_depth_00002.png": b"depth",
            "nyu_depth_v2/cal.txt": b"00009\n00002\n",
        },
    )

    assert calibration.split_names(archive, "cal") == [
        "nyu_depth_v2/cal/rgb_00002.jpg",
        "nyu_depth_v2/cal/rgb_00009.jpg",
    ]
    assert calibration.calibration_samples(archive, 1).shape == (1, 518, 686, 3)
    with pytest.raises(ValueError, match="cannot take 3 calibration images from 2"):
        calibration.calibration_samples(archive, 3)


def test_fidelity_selection_is_the_declared_seeded_sample(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"nyu_depth_v2/test/rgb_{index:05d}.jpg" for index in range(10)]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 1, 10)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe._variant(recipe._metadata(), profile_id)["parameters"]

    assert parameters["precision"] == profile_id
    assert parameters["input_shape"] == [1, 3, 518, 686]
    if profile_id == "w8a8":
        assert parameters["calibration_samples"] == 50


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()

    assert metadata["numerical_fidelity"]["outputs"] == ["predicted_depth"]
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._variant(metadata, "w32a32")


def test_compile_rejects_another_input_shape_before_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["components"][0]["parameters"]["input_shape"] = [1, 3, 518, 518]
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match="unsupported Depth Anything V2 input shape"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile["id"], None)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_unknown_invocation_or_missing_fidelity_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"), "--build-profile", "w16a16"]
        + arguments,
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
