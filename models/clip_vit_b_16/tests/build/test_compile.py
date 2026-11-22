from __future__ import annotations

import io
import json
import random
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]
# W8A16 CI on 50 ImageNet images and 64 prompts measured 0.99303 image,
# 0.99899 text cosine. Keep margin for compiler rounding.
FIDELITY_MIN_COSINE = {
    "w16a16": {"image_embeds": 0.99, "text_embeds": 0.99},
    "w8a16": {"image_embeds": 0.99, "text_embeds": 0.995},
}


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


def _mpk(path: Path, plugins: list[dict], inputs: list[dict]) -> Path:
    manifest = {"plugins": plugins, "input_nodes": inputs}
    return _write_archive(path, {"model_mpk.json": json.dumps(manifest).encode(), "model.elf": b"elf"})


def test_preprocessing_follows_the_checkpoints_processor(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    processor = {
        "crop_size": 224,
        "size": 224,
        "resample": 3,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }
    assert calibration.SIZE == processor["size"] == processor["crop_size"]
    assert calibration.RESCALE == pytest.approx(1 / 255)
    np.testing.assert_allclose(calibration.MEAN, processor["image_mean"])
    np.testing.assert_allclose(calibration.STD, processor["image_std"])


def test_preprocessing_resizes_the_short_side_and_crops_the_center(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A 300x200 landscape image in red, green and blue thirds. The short side scales to 224, so the image becomes
    # 336x224, and the centered crop keeps columns 56 to 279: red, green and blue span 56, 112 and 56 of them.
    image = Image.new("RGB", (300, 200), (255, 0, 0))
    image.paste((0, 255, 0), (100, 0, 200, 200))
    image.paste((0, 0, 255), (200, 0, 300, 200))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    result = calibration.preprocess_image(stream.getvalue())
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    for column, colour in ((20, [1, 0, 0]), (112, [0, 1, 0]), (204, [0, 0, 1])):
        expected = (np.asarray(colour, np.float32) - calibration.MEAN) / calibration.STD
        np.testing.assert_allclose(result[112, column], expected, atol=1e-5)


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


def test_compile_reads_both_encoders_of_each_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe.load_metadata(MODEL_DIR)
    assert metadata["build_profiles"]
    for profile in metadata["build_profiles"]:
        parameters = recipe._component_parameters(metadata, profile["id"])
        assert list(parameters) == list(recipe.COMPONENT_IDS) == ["image_encoder", "text_encoder"]
        assert {item["precision"] for item in parameters.values()} == {profile["id"]}
        assert all("calibration_bins" not in item and "calibration_samples" not in item for item in parameters.values())
        assert recipe._precision_label(parameters) == profile["id"].upper()


@pytest.mark.parametrize("precision", ["w8a16", "w16a16"])
def test_the_quantization_recipe_follows_the_precision(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    for name in ("quantization_scheme", "bfloat16_scheme", "SkipCalibration", "QuantizationParams"):
        monkeypatch.setattr(recipe, name, MagicMock(name=name))
    recipe._quantization_config({"precision": precision})
    if precision == "w8a16":
        # BF16 activations, INT8 weights per channel.
        assert recipe.quantization_scheme.call_args_list == [call(asymmetric=False, per_channel=True, bits=8)]
        recipe.bfloat16_scheme.assert_called_once_with()
    else:
        recipe.quantization_scheme.assert_not_called()
        recipe.bfloat16_scheme.assert_called_once_with()
    # Only w8a16 scales activations per token.
    options = recipe.QuantizationParams.call_args.kwargs
    assert options["per_token_quant"] == (precision == "w8a16")
    assert options["requantization_mode"] == recipe.RequantizationMode.sima
    recipe.SkipCalibration.assert_called_once_with()
    assert options["calibration_method"] is recipe.SkipCalibration.return_value
    with pytest.raises(ValueError, match="unsupported precision"):
        recipe._quantization_config({"precision": "w4a8"})


def test_compile_rejects_an_undeclared_build_profile_or_mixed_precisions(
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    with pytest.raises(ValueError, match="no build profile 'w32a32'"):
        recipe._component_parameters(recipe.load_metadata(MODEL_DIR), "w32a32")
    with pytest.raises(ValueError, match="one precision"):
        recipe._precision_label({"image_encoder": {"precision": "w8a8"}, "text_encoder": {"precision": "w16a16"}})


def test_fidelity_checks_every_output_without_calibration(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe.load_metadata(MODEL_DIR)
    fidelity = metadata["numerical_fidelity"]
    assert fidelity["outputs"] == ["image_embeds", "text_embeds"]
    assert fidelity["selection"]["offset"] == 0
    assert not metadata.get("calibrations")


def test_quantization_supplies_no_calibration_data(monkeypatch, load_script):
    recipe = load_script("compile")
    loaded = MagicMock()
    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: loaded)
    recipe._quantize_component(Path("encoder.onnx"), "clip", "text_encoder", {"precision": "w8a16"})
    assert loaded.quantize.call_args.kwargs["calibration_data"] is None


def test_fidelity_error_sums_match_the_metrics_over_all_samples(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    rng = np.random.default_rng(0)
    expected, actual = rng.normal(size=(2, 3, 5)), rng.normal(size=(2, 3, 5))
    error = recipe._OutputError("image_embeds")
    for want, got in zip(expected, actual):
        error.add(want, got)
    summary = error.summary()
    difference = actual - expected
    assert summary["name"] == "image_embeds"
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
    image, prompt = np.zeros((1, 3, 224, 224), np.float32), np.zeros((1, 77, 512), np.float32)
    assert recipe._afe_layout(image).shape == (1, 224, 224, 3)
    assert recipe._afe_layout(prompt).shape == (1, 77, 512)


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_input(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    plugins = [{"processor": "EV74"}, {"processor": "MLA"}, {"processor": "EV74"}]
    inputs = [{"name": "token_embeds", "size": 77 * 512 * 4}]
    assert recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", plugins, inputs), "text_encoder")["mla_plugins"] == 1
    for name, arguments, message in (
        ("cpu", ([*plugins, {"processor": "A65"}], inputs), "host=1"),
        ("input", (plugins, [{"name": "token_embeds", "size": 4}]), "input interface"),
    ):
        with pytest.raises(RuntimeError, match=message):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", *arguments), "text_encoder")


def test_compile_main_rejects_an_invocation_not_named_after_the_build_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    arguments = [
        "compile.py",
        "--model",
        str(tmp_path / "weights.pth"),
        "--output-dir",
        str(tmp_path / "out"),
        "--fidelity-dataset",
        str(tmp_path),
        "--build-profile",
        "w16a16",
        "--compile-invocation",
        "w8a8",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit) as stopped:
        recipe.main()
    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _check_fidelity(fidelity: dict, build_profile: str) -> None:
    floors = FIDELITY_MIN_COSINE[build_profile]
    assert [metric["name"] for metric in fidelity["metrics"]] == list(floors)
    for metric in fidelity["metrics"]:
        assert metric["cosine_similarity"] >= floors[metric["name"]], metric


@pytest.mark.parametrize("profile", ["w16a16", "w8a16"])
@pytest.mark.parametrize("output", ["image_embeds", "text_embeds"])
def test_fidelity_gate_rejects_drift_in_each_output(profile: str, output: str) -> None:
    fidelity = {"metrics": [{"name": name, "cosine_similarity": 1.0} for name in FIDELITY_MIN_COSINE[profile]]}
    _check_fidelity(fidelity, profile)
    next(metric for metric in fidelity["metrics"] if metric["name"] == output)["cosine_similarity"] = (
        FIDELITY_MIN_COSINE[profile][output] - 0.0001
    )
    with pytest.raises(AssertionError, match=output):
        _check_fidelity(fidelity, profile)


def test_build_holds_both_encoders_and_matches_the_onnx(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [item["component_id"] for item in outputs["outputs"]] == ["image_encoder", "text_encoder"]
    for item in outputs["outputs"]:
        recipe.audit_mpk(compiled / item["mpk_path"], item["component_id"])
    from model_registry.runtime_files import read_runtime_files

    assert [item["path"] for item in read_runtime_files(compiled)] == [
        "text_encoder/tokenizer.json", "text_encoder/token_embeddings.npy",
    ]
    table = np.load(compiled / "text_encoder/token_embeddings.npy", mmap_mode="r", allow_pickle=False)
    assert table.shape == (49408, recipe.COMPONENT_GRAPHS["text_encoder"]["inputs"]["token_embeds"][-1])
    assert table.dtype == np.float32
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    _check_fidelity(fidelity, build_profile)


ROOT = MODEL_DIR.parents[1]


@pytest.fixture
def exporter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("surgery")


def test_prompt_tables_are_openais_imagenet_classes_and_templates(exporter: ModuleType):
    assert len(exporter.CLASS_NAMES) == exporter.CLASS_COUNT == 1000
    assert (exporter.CLASS_NAMES[0], exporter.CLASS_NAMES[-1]) == ("tench", "toilet paper")
    assert len(exporter.TEMPLATES) == 80
    assert all(template.count("{}") == 1 for template in exporter.TEMPLATES)


def _text_probe() -> dict[str, np.ndarray]:
    with np.load(ROOT / "models/clip_vit_b_16/scripts/text_probe.npz") as saved:
        return {name: saved[name] for name in saved.files}


def test_the_text_probe_holds_the_exported_prompts(exporter: ModuleType):
    assert "a photo of a {}." in exporter.TEMPLATES
    assert _text_probe()["prompts"].tolist() == list(exporter.TEXT_PROBE_PROMPTS)


def _interface_model(exporter: ModuleType, case: str) -> onnx.ModelProto:
    helper = onnx.helper

    def value(name: str) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, exporter.INTERFACE[name])

    inputs = [value(name) for name in exporter.INPUTS]
    outputs = [value(name) for name in exporter.OUTPUTS]
    if case == "order":
        outputs[:2] = outputs[1::-1]
    elif case == "dtype":
        outputs[-1].type.tensor_type.elem_type = onnx.TensorProto.INT64
    elif case == "shape":
        outputs[0].type.tensor_type.shape.dim[0].dim_value = 2
    nodes = []
    for output in outputs:
        dtype = output.type.tensor_type.elem_type
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        array = np.zeros(shape, dtype=np.int64 if dtype == onnx.TensorProto.INT64 else np.float32)
        nodes.append(helper.make_node("Constant", [], [output.name], value=numpy_helper.from_array(array)))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, exporter: ModuleType):
    path = tmp_path / "model.onnx"
    onnx.save(_interface_model(exporter, case), path)
    if case == "valid":
        exporter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            exporter.validate_export(path)
