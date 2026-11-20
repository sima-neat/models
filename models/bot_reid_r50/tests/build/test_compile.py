from __future__ import annotations

import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image


def _jpeg(width: int, height: int, colour: tuple[int, int, int] = (200, 40, 10)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _mpk(path: Path, processors: list[str]) -> Path:
    content = json.dumps({"plugins": [{"processor": processor} for processor in processors]}).encode()
    return _write_archive(path, {"model_mpk.json": content})


def test_preprocessing_keeps_rgb_pixel_values_at_the_network_input(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    result = calibration.preprocess_image(_jpeg(64, 160))

    assert result.shape == (calibration.HEIGHT, calibration.WIDTH, 3) == (256, 128, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[128, 64], [200, 40, 10], atol=3)
    # Neat scales by 1/255 and these statistics undo it, so the network sees the same [0, 255] values.
    np.testing.assert_allclose((np.float32(200) / 255 - calibration.MEAN) / calibration.STD, [200] * 3, rtol=1e-6)


def test_identities_come_from_market_1501_names(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.identity("query/0001_c1s1_001051_00.jpg") == (1, 0)
    assert calibration.identity("bounding_box_test/0000_c6s1_000776_03.jpg") == (0, 5)
    assert calibration.identity("bounding_box_test/-1_c3s1_000001_01.jpg") is None
    with pytest.raises(ValueError, match="naming"):
        calibration.identity("query/Thumbs.jpg")


def test_calibration_samples_seeded_training_images_without_junk(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(
        tmp_path / "market.tar.gz",
        {
            "bounding_box_train/0002_c1s1_000451_03.jpg": _jpeg(64, 128),
            "bounding_box_train/0007_c2s3_070952_01.jpg": _jpeg(64, 128),
            "bounding_box_test/-1_c1s1_000401_03.jpg": _jpeg(64, 128),
            "query/0001_c1s1_001051_00.jpg": _jpeg(64, 128),
        },
    )

    samples = calibration.calibration_samples(archive, 2)

    assert samples.shape == (2, 256, 128, 3)
    np.testing.assert_array_equal(samples, calibration.calibration_samples(archive, 2))
    with pytest.raises(ValueError, match="calibration requires 3 images"):
        calibration.calibration_samples(archive, 3)
    with pytest.raises(ValueError, match="no images under bounding_box_test/"):
        calibration.split_names(archive, calibration.GALLERY)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8", "w8a16"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    try:
        parameters = recipe._parameters(profile_id)
    except ValueError:
        pytest.skip(f"model.yaml does not declare {profile_id}")

    assert parameters["precision"] == profile_id
    if profile_id == "w8a8":
        assert parameters["calibration_samples"] == 50


@pytest.mark.parametrize(
    ("precision", "calibrates", "per_token"), [("w8a8", True, False), ("w8a16", False, True), ("w16a16", False, False)]
)
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.CalibrationMethod.from_str.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration
    assert arguments.get("per_token_quant", False) is per_token


def test_audit_requires_one_mla_segment_and_no_host_code(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", ["EV74", "MLA", "EV74"]))
    for name, processors in [("two", ["MLA", "MLA"]), ("host", ["MLA", "A65"]), ("none", ["EV74"])]:
        with pytest.raises(RuntimeError, match="plugins run on"):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", processors))


def test_int8_build_requires_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: {"precision": "w8a8", "calibration_samples": 50})

    with pytest.raises(ValueError, match="requires --dataset"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w8a8", None)


def test_compile_rejects_an_undeclared_precision_before_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: {"precision": "fp32"})

    with pytest.raises(ValueError, match="only the declared BF16 and INT8 build profiles"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16", None)
    assert not (tmp_path / "model").exists()


def test_compile_main_rejects_an_unknown_invocation(
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
