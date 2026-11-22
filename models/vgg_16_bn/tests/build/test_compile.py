from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from model_registry.errors import ValidationError
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]


def _png(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="PNG")
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_follows_the_checkpoints_config(load_script: Callable[[str], ModuleType]) -> None:
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


def test_preprocessing_rounds_the_crop_offset_like_torchvision(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A 256x259 image keeps its size and has 35 spare rows. torchvision's CenterCrop starts at
    # round(35 / 2) = 18, not 17, so the 18 green rows on top stay out of the crop.
    image = Image.new("RGB", (256, 259), (255, 0, 0))
    image.paste((0, 255, 0), (0, 0, 256, 18))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[0, 112], red, atol=1e-5)


def test_calibration_samples_the_requested_cal_images_from_the_archive(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(
        tmp_path / "cal.tar.gz",
        {
            "imagenet/cal2012/n01440764_1.JPEG": _png(300, 260),
            "imagenet/cal2012/n01443537_1.JPEG": _png(260, 300),
            "imagenet/cal2012.txt": b"n01440764_1.JPEG\nn01443537_1.JPEG\n",
            "imagenet/val2012/ILSVRC2012_val_00000001.JPEG": _png(256, 256),
        },
    )

    samples = calibration.calibration_samples(archive, 1)

    assert samples.shape == (1, 224, 224, 3)
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, calibration.calibration_samples(archive, 1))
    with pytest.raises(ValueError, match="calibration requires 3 images"):
        calibration.calibration_samples(archive, 3)


def test_split_names_read_an_extracted_directory_and_require_the_split(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    (tmp_path / "val2012").mkdir()
    (tmp_path / "val2012" / "x.JPEG").write_bytes(_png(256, 256))

    assert calibration.split_names(tmp_path, "val2012") == ["val2012/x.JPEG"]
    with pytest.raises(ValueError, match="no images under cal2012/"):
        calibration.split_names(tmp_path, "cal2012")


def test_validation_labels_come_from_the_labels_file(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    categories = [{"id": index, "wnid": f"n{index:08d}", "name": f"class {index}"} for index in range(1000)]
    labels = {
        "categories": categories,
        "images": [
            {"id": 7, "file_name": "ILSVRC2012_val_00000007.JPEG", "width": 500, "height": 375},
            {"id": 2, "file_name": "ILSVRC2012_val_00000002.JPEG", "width": 500, "height": 375},
        ],
        "annotations": [{"id": 7, "image_id": 7, "category_id": 999}, {"id": 2, "image_id": 2, "category_id": 0}],
    }
    path = tmp_path / "annotations" / "labels_val2012.json"
    path.parent.mkdir()
    path.write_text(json.dumps(labels), encoding="utf-8")

    assert calibration.validation_labels(tmp_path) == {
        "val2012/ILSVRC2012_val_00000002.JPEG": 0,
        "val2012/ILSVRC2012_val_00000007.JPEG": 999,
    }
    path.write_text(json.dumps({**labels, "categories": categories[1:]}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 1000 validation classes, found 999"):
        calibration.validation_labels(tmp_path)


def test_read_images_reports_a_missing_image(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(tmp_path / "cal.tar.gz", {"imagenet/cal2012/a.JPEG": b"a"})

    assert calibration.read_images(archive, ["imagenet/cal2012/a.JPEG"]) == {"imagenet/cal2012/a.JPEG": b"a"}
    with pytest.raises(ValueError, match="missing 1 requested images"):
        calibration.read_images(archive, ["imagenet/cal2012/a.JPEG", "imagenet/cal2012/b.JPEG"])


def test_fidelity_selection_is_the_declared_seeded_sample(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    names = [f"cal2012/n01440764_{index:02d}.JPEG" for index in range(10)]
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
    parameters = recipe.component_parameters(recipe.MODEL_DIR, profile_id, recipe.COMPONENT_ID)

    assert parameters["precision"] == profile_id


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        recipe.component_parameters(recipe.MODEL_DIR, "w32a32", recipe.COMPONENT_ID)


def test_numerical_fidelity_compares_the_logits(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    assert recipe.load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]["outputs"] == ["logits"]


@pytest.mark.parametrize(("precision", "calibrates"), [("w8a8", True), ("w16a16", False)])
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.CalibrationMethod.from_str.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration


def test_quantization_rejects_an_undeclared_precision(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="unsupported precision 'w8a16'"):
        recipe._quantization("w8a16")


def test_bf16_builds_quantize_without_calibration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(
        recipe,
        "component_parameters",
        lambda model_dir, profile_id, component_id: {
            "precision": "w16a16",
            "input_name": "pixel_values",
            "input_shape": [1, 3, 224, 224],
        },
    )

    with pytest.raises(RuntimeError, match="expected exactly one MPK archive"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16", None)
    assert recipe.load_model.return_value.quantize.call_args.args[0] is None


def test_compile_rejects_an_undeclared_precision_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": "fp32"}
    monkeypatch.setattr(recipe, "component_parameters", lambda model_dir, profile_id, component_id: parameters)

    with pytest.raises(ValueError, match="only the declared BF16 and INT8 build profiles"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16", None)
    assert not (tmp_path / "model").exists()


def test_fidelity_compares_the_quantized_logits_with_the_onnx_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    fidelity = recipe.load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]
    images = {"cal2012/a.JPEG": 1.0, "cal2012/b.JPEG": 2.0}
    onnx_input_shapes = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def run(self, outputs: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            onnx_input_shapes.append(feeds["pixel_values"].shape)
            return [np.full((1, 1000), feeds["pixel_values"][0, 0, 0, 0], np.float32)]

    candidate = MagicMock()
    # This candidate returns the logits as [1, 1, 1, 1000], which the recipe reshapes to the source's [1, 1000].
    # Each image's last logit comes back 1 too high.
    candidate.execute.side_effect = [
        [np.append(np.full(999, value), value + 1).reshape(1, 1, 1, 1000)] for value in images.values()
    ]
    monkeypatch.setattr(recipe, "fidelity_names", lambda dataset, seed, offset, samples: list(images))
    monkeypatch.setattr(recipe, "read_images", lambda dataset, names: dict(images))
    monkeypatch.setattr(recipe, "preprocess_image", lambda value: np.full((224, 224, 3), value, np.float32))
    monkeypatch.setattr(recipe, "ort", MagicMock(InferenceSession=Session))

    report = recipe.measure_fidelity(tmp_path / "model.onnx", candidate, tmp_path / "cal", fidelity, "w8a8")

    assert onnx_input_shapes == [(1, 3, 224, 224)] * 2
    assert (report["model_id"], report["build_profile_id"], report["samples"]) == ("vgg_16_bn", "w8a8", 2)
    (metric,) = report["metrics"]
    assert metric["name"] == "logits"
    assert metric["max_absolute_error"] == 1.0
    assert metric["mean_absolute_error"] == pytest.approx(1 / 1000)


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
