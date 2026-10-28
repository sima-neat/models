from __future__ import annotations

import io
import json
import random
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

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
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _mpk(path: Path, plugins: list[dict], inputs: list[dict], version: str) -> Path:
    manifest = {"plugins": plugins, "input_nodes": inputs, "model_sdk_version": version}
    return _write_archive(path, {"model_mpk.json": json.dumps(manifest).encode(), "model.elf": b"elf"})


def test_preprocessing_follows_the_checkpoints_processor(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    processor = {"size": {"height": 224, "width": 224}, "resample": 2, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    assert calibration.SIZE == processor["size"]["height"] == processor["size"]["width"]
    assert calibration.RESCALE == pytest.approx(1 / 255)
    np.testing.assert_allclose(calibration.MEAN, processor["image_mean"])
    np.testing.assert_allclose(calibration.STD, processor["image_std"])


def test_preprocessing_resizes_straight_to_the_model_input(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, squeezes to 224x224 without a crop: 112 rows of each.
    image = Image.new("RGB", (128, 256), (255, 0, 0))
    image.paste((0, 0, 255), (0, 128, 128, 256))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    result = calibration.preprocess_image(stream.getvalue())
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    for row, colour in ((100, [1, 0, 0]), (124, [0, 0, 1])):
        expected = (np.asarray(colour, np.float32) - calibration.MEAN) / calibration.STD
        np.testing.assert_allclose(result[row, 112], expected, atol=1e-5)


def test_selection_is_seeded_and_reads_an_archive(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    names = [f"imagenet/cal2012/n01440764_{index:02d}.JPEG" for index in range(10)]
    archive = _write_archive(tmp_path / "cal.tar.gz", {name: _png(260, 300) for name in names})
    expected = sorted(names)
    random.Random(42).shuffle(expected)
    calibration_names = calibration.select_names(archive, 42, 0, 4)
    fidelity_names = calibration.select_names(archive, 42, 4, 5)
    assert (calibration_names, fidelity_names) == (expected[:4], expected[4:9])
    assert calibration.load_images(archive, calibration_names).shape == (4, 224, 224, 3)
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.select_names(archive, 42, 1, 10)


def test_split_names_read_an_extracted_directory_and_require_the_split(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
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


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_both_encoders_of_each_build_profile(
    load_script: Callable[[str], ModuleType], profile_id: str
) -> None:
    recipe = load_script("compile")
    parameters = recipe._component_parameters(recipe.load_metadata(MODEL_DIR), profile_id)
    assert list(parameters) == list(recipe.COMPONENT_IDS) == ["image_encoder", "text_encoder"]
    assert {item["precision"] for item in parameters.values()} == {profile_id}
    # Both precisions calibrate both encoders: AFE needs calibrated ranges even at BF16.
    assert all("calibration_bins" in item for item in parameters.values())
    assert recipe._precision_label(parameters) == profile_id.upper()


def test_compile_rejects_an_undeclared_build_profile_or_mixed_precisions(
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    with pytest.raises(ValueError, match="no build profile 'w32a32'"):
        recipe._component_parameters(recipe.load_metadata(MODEL_DIR), "w32a32")
    with pytest.raises(ValueError, match="one precision"):
        recipe._precision_label({"image_encoder": {"precision": "w8a8"}, "text_encoder": {"precision": "w16a16"}})


def test_fidelity_compares_every_output_after_the_calibration_images(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe.load_metadata(MODEL_DIR)
    fidelity = metadata["numerical_fidelity"]
    assert fidelity["outputs"] == ["image_embeds", "logits", "text_embeds"]
    counts = [
        int(binding["parameters"]["calibration_samples"])
        for profile in metadata["build_profiles"]
        for binding in profile["components"]
        if binding["id"] == "image_encoder"
    ]
    assert len(counts) == len(metadata["build_profiles"]) and fidelity["selection"]["offset"] >= max(counts)
    assert metadata["calibrations"][0]["dataset"] == fidelity["dataset"]


def test_fidelity_error_sums_match_the_metrics_over_all_samples(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    rng = np.random.default_rng(0)
    expected, actual = rng.normal(size=(2, 3, 5)), rng.normal(size=(2, 3, 5))
    error = recipe._OutputError("logits")
    for want, got in zip(expected, actual):
        error.add(want, got)
    summary = error.summary()
    difference = actual - expected
    assert summary["name"] == "logits"
    assert summary["cosine_similarity"] == pytest.approx(
        np.dot(expected.ravel(), actual.ravel()) / (np.linalg.norm(expected) * np.linalg.norm(actual))
    )
    assert summary["normalized_rmse"] == pytest.approx(np.sqrt(np.mean(difference**2) / np.mean(expected**2)))
    assert summary["mean_absolute_error"] == pytest.approx(np.mean(np.abs(difference)))
    assert summary["max_absolute_error"] == pytest.approx(np.max(np.abs(difference)))
    with pytest.raises(RuntimeError, match="non-finite"):
        error.add(np.array([np.nan]), np.array([0.0]))


def test_afe_layout_converts_only_rank_four_tensors(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    image, prompt = np.zeros((1, 3, 224, 224), np.float32), np.zeros((1, 64, 768), np.float32)
    assert recipe._afe_layout(image).shape == (1, 224, 224, 3)
    assert recipe._afe_layout(prompt).shape == (1, 64, 768)


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_input(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    plugins = [{"processor": "EV74"}, {"processor": "MLA"}, {"processor": "EV74"}]
    inputs = [{"name": "token_embeds", "size": 64 * 768 * 4}]
    version = recipe.COMPILER_VERSION
    assert recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", plugins, inputs, version), "text_encoder")["mla_plugins"] == 1
    for name, arguments, message in (
        ("cpu", ([*plugins, {"processor": "A65"}], inputs, version), "host=1"),
        ("input", (plugins, [{"name": "token_embeds", "size": 4}], version), "input interface"),
        ("version", (plugins, inputs, "0.0.0"), "compiler mismatch"),
    ):
        with pytest.raises(RuntimeError, match=message):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", *arguments), "text_encoder")


def test_compile_main_rejects_an_invocation_not_named_after_the_build_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    arguments = ["compile.py", "--model", str(tmp_path / "model.onnx"), "--output-dir", str(tmp_path / "out"),
                 "--fidelity-dataset", str(tmp_path), "--build-profile", "w16a16", "--compile-invocation", "w8a8"]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit):
        recipe.main()
    assert not (tmp_path / "out").exists()


def test_build_holds_both_encoders_and_bf16_matches_the_onnx(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [item["component_id"] for item in outputs["outputs"]] == ["image_encoder", "text_encoder"]
    for item in outputs["outputs"]:
        recipe.audit_mpk(compiled / item["mpk_path"], item["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    assert [metric["name"] for metric in fidelity["metrics"]] == ["image_embeds", "logits", "text_embeds"]
    if build_profile == "w16a16":
        # BF16 keeps every output within rounding of the ONNX; a broken operator shows up as a low cosine.
        assert all(metric["cosine_similarity"] > 0.99 for metric in fidelity["metrics"]), fidelity["metrics"]
