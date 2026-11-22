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
import onnx
import pytest
from model_registry.errors import ValidationError
from onnx import helper, numpy_helper
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]
# A small stand-in for the export: the same node names and block structure at 8 channels in 2 heads.
CHANNELS, HEADS, TOKENS = 8, 2, 196


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


def test_preprocessing_follows_the_checkpoints_processor(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.SIZE == 224
    assert calibration.RESCALE == pytest.approx(1 / 255)
    np.testing.assert_allclose(calibration.MEAN, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(calibration.STD, [0.5, 0.5, 0.5])


def test_preprocessing_resizes_straight_to_the_model_input(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, squeezes to 224x224 without a crop: 112 rows of each.
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


def test_fidelity_selection_reads_the_cal_images_from_an_archive(
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
    expected = ["imagenet/cal2012/n01440764_1.JPEG", "imagenet/cal2012/n01443537_1.JPEG"]
    random.Random(42).shuffle(expected)

    names = calibration.fidelity_names(archive, 42, 0, 2)

    assert names == expected
    assert set(calibration.read_images(archive, names)) == set(names)


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


def _export_graph(head_scale: float = 8**-0.5) -> onnx.ModelProto:
    """The export's structure at a small width: patch embedding, twelve attention blocks, and the class-token head.

    Each block projects query, key and value separately and scales the query and key heads by head_scale each.
    """
    rng = np.random.default_rng(0)
    head_size = CHANNELS // HEADS
    nodes, initializers = [], []

    def constant(name: str, array: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def node(op_type: str, inputs: list[str], output: str, **attributes) -> str:
        nodes.append(helper.make_node(op_type, inputs, [output], name=output.removesuffix("_output_0"), **attributes))
        return output

    patch = constant("patch.weight", rng.normal(0, 0.05, (CHANNELS, 3, 16, 16)).astype(np.float32))
    nodes.append(
        helper.make_node("Constant", [], ["patch_shape"], value=numpy_helper.from_array(np.int64([1, CHANNELS, -1])))
    )
    patches = node("Conv", ["pixel_values", patch], "/vit/embeddings/Conv_output_0", strides=[16, 16])
    patches = node("Reshape", [patches, "patch_shape"], "/vit/embeddings/Reshape_output_0")
    tokens = node("Transpose", [patches], "/vit/embeddings/Transpose_output_0", perm=[0, 2, 1])
    heads_shape = constant("heads_shape", np.int64([1, TOKENS, HEADS, head_size]))
    tokens_shape = constant("tokens_shape", np.int64([1, TOKENS, CHANNELS]))
    scale = constant("head_scale", np.float32([head_scale]))
    for layer in range(12):
        prefix = f"/vit/encoder/layer.{layer}/attention/attention"
        projected = {}
        for projection in ("query", "key", "value"):
            weight = constant(
                f"{prefix}/{projection}.weight", rng.normal(0, 0.3, (CHANNELS, CHANNELS)).astype(np.float32)
            )
            bias = constant(f"{prefix}/{projection}.bias", rng.normal(0, 0.1, CHANNELS).astype(np.float32))
            output = node("MatMul", [tokens, weight], f"{prefix}/{projection}/MatMul_output_0")
            projected[projection] = node("Add", [bias, output], f"{prefix}/{projection}/Add_output_0")
        query = node("Reshape", [projected["query"], heads_shape], f"{prefix}/Reshape_2_output_0")
        query = node("Transpose", [query], f"{prefix}/Transpose_1_output_0", perm=[0, 2, 1, 3])
        key = node("Reshape", [projected["key"], heads_shape], f"{prefix}/Reshape_output_0")
        key = node("Transpose", [key], f"{prefix}/Transpose_2_output_0", perm=[0, 2, 3, 1])
        value = node("Reshape", [projected["value"], heads_shape], f"{prefix}/Reshape_1_output_0")
        value = node("Transpose", [value], f"{prefix}/Transpose_output_0", perm=[0, 2, 1, 3])
        query = node("Mul", [query, scale], f"{prefix}/Mul_output_0")
        key = node("Mul", [key, scale], f"{prefix}/Mul_1_output_0")
        scores = node(
            "Softmax", [node("MatMul", [query, key], f"{prefix}/MatMul_output_0")], f"{prefix}/Softmax_output_0"
        )
        context = node("MatMul", [scores, value], f"{prefix}/MatMul_1_output_0")
        context = node("Transpose", [context], f"{prefix}/Transpose_3_output_0", perm=[0, 2, 1, 3])
        context = node("Reshape", [context, tokens_shape], f"{prefix}/Reshape_3_output_0")
        tokens = node("Add", [context, tokens], f"/vit/encoder/layer.{layer}/Add_output_0")
    class_token = node("Gather", [tokens, constant("class_index", np.int64(0))], "/Gather_output_0", axis=1)
    classifier = constant("classifier.weight", rng.normal(0, 0.3, (CHANNELS, 1000)).astype(np.float32))
    node("MatMul", [class_token, classifier], "logits")
    graph = helper.make_graph(
        nodes,
        "vit_b_16_fixture",
        [helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, ["batch_size", 3, 224, 224])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["batch_size", 1000])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7)


def test_surgery_matches_the_export(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    import onnxruntime

    surgery = load_script("surgery")
    source = tmp_path / "model.onnx"
    onnx.save(_export_graph(), source)

    fused = surgery.prepare_model(source, tmp_path / "out")

    pixels = np.random.default_rng(1).uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)

    def logits(path: Path) -> np.ndarray:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return session.run(None, {"pixel_values": pixels})[0]

    np.testing.assert_allclose(logits(fused), logits(source), rtol=1e-5, atol=1e-5)


def test_surgery_fuses_each_blocks_projections_and_scale(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "model.onnx"
    export = _export_graph()
    onnx.save(export, source)

    fused = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    operations = [node.op_type for node in fused.graph.node]
    assert operations.count("Split") == 12
    assert "Mul" not in operations
    # Two attention MatMuls and one projection per block, then the classifier.
    assert operations.count("MatMul") == 12 * 3 + 1
    source_weights = {value.name: numpy_helper.to_array(value) for value in export.graph.initializer}
    weights = {value.name: numpy_helper.to_array(value) for value in fused.graph.initializer}
    prefix = "/vit/encoder/layer.3/attention/attention"
    expected = np.concatenate(
        [
            source_weights[f"{prefix}/query.weight"] / 8,
            source_weights[f"{prefix}/key.weight"],
            source_weights[f"{prefix}/value.weight"],
        ],
        axis=1,
    )
    np.testing.assert_array_equal(weights[f"{prefix}/qkv/weight"], expected)


def test_surgery_rejects_an_attention_scale_it_cannot_fold_exactly(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "model.onnx"
    onnx.save(_export_graph(head_scale=0.5), source)

    with pytest.raises(ValueError, match="layer 0 scales attention by 0.25"):
        surgery.prepare_model(source, tmp_path / "out")


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a16"])
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


@pytest.mark.parametrize(("precision", "per_token"), [("w8a16", True), ("w16a16", False)])
def test_quantization_keeps_bf16_activations_without_calibration(
    load_script: Callable[[str], ModuleType], precision: str, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision)

    arguments = recipe.QuantizationParams.call_args.kwargs
    assert arguments["calibration_method"] is recipe.SkipCalibration.return_value
    assert arguments["activation_quantization_scheme"] is recipe.bfloat16_scheme.return_value
    assert arguments.get("per_token_quant", False) is per_token


def test_quantization_rejects_an_undeclared_precision(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="unsupported precision 'w8a8'"):
        recipe._quantization("w8a8")


def test_bf16_builds_quantize_the_prepared_graph_without_calibration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "fused.onnx")

    with pytest.raises(RuntimeError, match="expected exactly one MPK archive"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16")
    assert recipe.onnx_source.call_args.args[0].endswith("fused.onnx")
    assert recipe.load_model.return_value.quantize.call_args.args[0] is None


def test_compile_rejects_an_undeclared_precision_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": "fp32"}
    monkeypatch.setattr(recipe, "component_parameters", lambda model_dir, profile_id, component_id: parameters)

    with pytest.raises(ValueError, match="only the declared w16a16 and w8a16 build profiles"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16")
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

    report = recipe.measure_fidelity(tmp_path / "model.onnx", candidate, tmp_path / "cal", fidelity, "w8a16")

    assert onnx_input_shapes == [(1, 3, 224, 224)] * 2
    assert (report["model_id"], report["build_profile_id"], report["samples"]) == ("vit_b_16", "w8a16", 2)
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
