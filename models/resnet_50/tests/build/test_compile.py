from __future__ import annotations

import io
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]


def _png(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="PNG")
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_uses_imagenet_resize_and_normalization(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.SHORTEST_EDGE == 256
    assert calibration.CROP == 224
    assert calibration.RESCALE == pytest.approx(1 / 255)
    np.testing.assert_allclose(calibration.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(calibration.STD, [0.229, 0.224, 0.225])


def test_preprocessing_resizes_the_shortest_edge_then_crops_the_centre(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, resizes to 256x512; the centre crop keeps 112 rows of each.
    image = Image.new("RGB", (128, 256), (255, 0, 0))
    image.paste((0, 0, 255), (0, 128, 128, 256))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[100, 112], red, atol=1e-5)
    np.testing.assert_allclose(result[124, 112], blue, atol=1e-5)


def test_calibration_stacks_every_cal_image_from_the_archive(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(
        tmp_path / "mini.tar.gz",
        {
            "cal/b.webp": _png(300, 260),
            "cal/a.webp.bak": _png(260, 300),
            "cal/notes.txt": b"not an image",
            "val/n01440764/c.webp": _png(256, 256),
        },
    )

    samples = calibration.calibration_samples(archive, 2)

    assert samples.shape == (2, 224, 224, 3)
    assert samples.dtype == np.float32
    with pytest.raises(ValueError, match="calibration requires 3 images"):
        calibration.calibration_samples(archive, 3)


def test_split_names_read_an_extracted_directory_and_require_the_split(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    (tmp_path / "val" / "n01").mkdir(parents=True)
    (tmp_path / "val" / "n01" / "x.webp").write_bytes(_png(256, 256))

    assert calibration.split_names(tmp_path, "val") == ["val/n01/x.webp"]
    with pytest.raises(ValueError, match="no images under cal/"):
        calibration.split_names(tmp_path, "cal")


def test_validation_labels_follow_the_sorted_synsets(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    names = [f"val/n{index:08d}/image.webp" for index in reversed(range(calibration.CLASSES))]

    labels = calibration.validation_labels(names)

    assert labels["val/n00000000/image.webp"] == 0
    assert labels[f"val/n{calibration.CLASSES - 1:08d}/image.webp"] == calibration.CLASSES - 1
    with pytest.raises(ValueError, match="expected 1000 validation classes, found 999"):
        calibration.validation_labels(names[1:])


def test_read_images_reports_a_missing_image(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(tmp_path / "mini.tar.gz", {"val/n01/a.webp": b"a"})

    assert calibration.read_images(archive, ["val/n01/a.webp"]) == {"val/n01/a.webp": b"a"}
    with pytest.raises(ValueError, match="missing 1 requested images"):
        calibration.read_images(archive, ["val/n01/a.webp", "val/n01/b.webp"])


def test_fidelity_selection_is_the_declared_seeded_sample(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    names = [f"val/n01/{index:02d}.webp" for index in range(10)]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 1, 10)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8", "w8a16"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe._variant(recipe._metadata(), profile_id)["parameters"]

    assert parameters["precision"] == profile_id
    if profile_id == "w8a8":
        assert parameters["calibration_samples"] == 500


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()

    assert metadata["numerical_fidelity"]["outputs"] == ["logits"]
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._variant(metadata, "w32a32")


def test_compile_rejects_an_undeclared_precision_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["components"][0]["parameters"]["precision"] = "fp32"
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match="only the declared Modalix BF16 and INT8 variants"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile["id"], None)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize(("profile_id", "per_token"), [("w16a16", False), ("w8a8", False), ("w8a16", True)])
def test_only_w8a16_quantizes_activations_per_token(
    profile_id: str,
    per_token: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    loaded = MagicMock()
    monkeypatch.setattr(recipe, "onnx_source", MagicMock())
    monkeypatch.setattr(recipe, "load_model", MagicMock(return_value=loaded))
    monkeypatch.setattr(recipe, "AfeModel", MagicMock())
    monkeypatch.setattr(recipe, "calibration_samples", lambda dataset, count: np.zeros((1, 224, 224, 3), np.float32))
    monkeypatch.setattr(recipe, "QuantizationParams", dict)

    with pytest.raises(RuntimeError, match="expected exactly one MPK archive"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile_id, tmp_path / "calibration")

    assert loaded.quantize.call_args.args[1]["per_token_quant"] is per_token


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
