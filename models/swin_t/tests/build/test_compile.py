from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from PIL import Image


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


def _pooled_head() -> onnx.ModelProto:
    """The export's head: final tokens, a token-major transpose, a 1-D GlobalAveragePool and a classifier."""
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.standard_normal((4, 768)).astype(np.float32), "classifier.weight")
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["tokens"], ["channels"], perm=[0, 2, 1], name="/swin/Transpose"),
            helper.make_node("GlobalAveragePool", ["channels"], ["pooled"], name="/swin/pooler/GlobalAveragePool"),
            helper.make_node("Flatten", ["pooled"], ["features"], axis=1, name="/swin/Flatten"),
            helper.make_node("Gemm", ["features", "classifier.weight"], ["logits"], transB=1, name="/classifier/Gemm"),
        ],
        "head",
        [helper.make_tensor_value_info("tokens", TensorProto.FLOAT, [1, 49, 768])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
        [weight],
    )
    return onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))


def test_preprocessing_follows_the_swin_authors_evaluation_transform(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.SHORTEST_EDGE == int(256 / 224 * 224)
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


def test_pooling_as_an_image_keeps_the_classifier_output(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = _pooled_head()
    tokens = np.random.default_rng(1).standard_normal((1, 49, 768)).astype(np.float32)
    expected = ReferenceEvaluator(source).run(None, {"tokens": tokens})[0]

    rewritten = surgery.pool_as_image(onnx.ModelProto.FromString(source.SerializeToString()))

    onnx.checker.check_model(rewritten)
    (pool,) = [node for node in rewritten.graph.node if node.op_type == "GlobalAveragePool"]
    shapes = surgery._shapes(rewritten)
    assert shapes[pool.input[0]] == [1, 768, 7, 7]
    np.testing.assert_allclose(ReferenceEvaluator(rewritten).run(None, {"tokens": tokens})[0], expected, rtol=1e-5)


def test_pooling_as_an_image_rejects_an_unexpected_head(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    model = _pooled_head()
    model.graph.node[0].attribute[0].ints[:] = [0, 1, 2]

    with pytest.raises(ValueError, match="token transpose"):
        surgery.pool_as_image(model)


def _model(nodes, inputs, outputs, constants) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "pattern",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in inputs.items()],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs],
        [numpy_helper.from_array(np.asarray(value), name) for name, value in constants.items()],
    )
    return onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))


def _assert_rewrite_keeps_outputs(surgery: ModuleType, rewrite, model: onnx.ModelProto) -> onnx.ModelProto:
    rng = np.random.default_rng(2)
    feeds = {
        value.name: rng.standard_normal([dim.dim_value for dim in value.type.tensor_type.shape.dim]).astype(np.float32)
        for value in model.graph.input
    }
    expected = ReferenceEvaluator(model).run(None, feeds)
    rewritten = rewrite(onnx.ModelProto.FromString(model.SerializeToString()))
    onnx.checker.check_model(rewritten)
    for actual, wanted in zip(ReferenceEvaluator(rewritten).run(None, feeds), expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-5, atol=1e-5)
    return rewritten


def _all_shapes(model: onnx.ModelProto) -> list[list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    return [[dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in inferred.graph.value_info]


def test_windows_partition_and_merge_in_four_dimensions(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    # One block on a 14x14 grid of 8 channels: four 7x7 windows, and their merge back into the grid.
    model = _model(
        [
            helper.make_node("Reshape", ["grid", "split"], ["parts"]),
            helper.make_node("Transpose", ["parts"], ["swapped"], perm=[0, 1, 3, 2, 4, 5]),
            helper.make_node("Reshape", ["swapped", "flat"], ["windows"]),
            helper.make_node("Reshape", ["windows", "unflat"], ["window_parts"]),
            helper.make_node("Transpose", ["window_parts"], ["unswapped"], perm=[0, 1, 3, 2, 4, 5]),
            helper.make_node("Reshape", ["unswapped", "back"], ["merged"]),
        ],
        {"grid": [1, 14, 14, 8]},
        ["windows", "merged"],
        {
            "split": np.asarray([1, 2, 7, 2, 7, 8], np.int64),
            "flat": np.asarray([4, 49, 8], np.int64),
            "unflat": np.asarray([1, 2, 2, 7, 7, 8], np.int64),
            "back": np.asarray([1, 14, 14, 8], np.int64),
        },
    )

    rewritten = _assert_rewrite_keeps_outputs(surgery, surgery.windows_in_four_dimensions, model)

    assert max(len(shape) for shape in _all_shapes(rewritten)) == 4


def test_shift_masks_merge_into_the_position_bias(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "SHIFTED_BLOCKS", 1)
    rng = np.random.default_rng(3)
    model = _model(
        [
            helper.make_node("Add", ["scores", "bias"], ["biased"]),
            helper.make_node("Reshape", ["biased", "split"], ["by_window"]),
            helper.make_node("Add", ["by_window", "mask"], ["masked"]),
            helper.make_node("Reshape", ["masked", "back"], ["merged"]),
            helper.make_node("Softmax", ["merged"], ["probabilities"], axis=-1),
        ],
        {"scores": [4, 2, 49, 49]},
        ["probabilities"],
        {
            "bias": rng.standard_normal((1, 2, 49, 49)).astype(np.float32),
            "mask": np.where(rng.random((1, 4, 1, 49, 49)) < 0.3, -100.0, 0.0).astype(np.float32),
            "split": np.asarray([1, 4, 2, 49, 49], np.int64),
            "back": np.asarray([4, 2, 49, 49], np.int64),
        },
    )

    rewritten = _assert_rewrite_keeps_outputs(surgery, surgery.merge_shift_masks, model)

    assert [node.op_type for node in rewritten.graph.node] == ["Add", "Softmax"]


def test_score_scale_folds_into_the_query_projection(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    rng = np.random.default_rng(4)
    heads = [
        node
        for index, (name, perm) in enumerate((("q", [0, 2, 1, 3]), ("k", [0, 2, 3, 1]), ("v", [0, 2, 1, 3])))
        for node in (
            helper.make_node("Add", [f"{name}_bias", f"{name}_raw"], [f"{name}_flat"]),
            helper.make_node("Reshape", [f"{name}_flat", "heads"], [f"{name}_heads"]),
            helper.make_node("Transpose", [f"{name}_heads"], [name], perm=perm),
        )
    ]
    model = _model(
        [
            helper.make_node("MatMul", ["tokens", "qkv_weight"], ["qkv"]),
            helper.make_node("Split", ["qkv", "widths"], ["q_raw", "k_raw", "v_raw"], axis=-1),
            *heads,
            helper.make_node("MatMul", ["q", "k"], ["scores"]),
            helper.make_node("Div", ["scores", "root"], ["scaled"]),
            helper.make_node("Add", ["scaled", "bias"], ["biased"]),
            helper.make_node("Softmax", ["biased"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", "v"], ["attended"]),
        ],
        {"tokens": [4, 49, 32]},
        ["attended"],
        {
            "qkv_weight": rng.standard_normal((32, 96)).astype(np.float32),
            "widths": np.asarray([32, 32, 32], np.int64),
            # Heads of 16 channels scale scores by 1/4; a power of two moves through the projection exactly,
            # so the comparison does not depend on how a NumPy build rounds.
            "heads": np.asarray([4, 49, 2, 16], np.int64),
            "root": np.asarray(4.0, np.float32),
            "bias": rng.standard_normal((1, 2, 49, 49)).astype(np.float32),
            **{f"{name}_bias": rng.standard_normal(32).astype(np.float32) for name in "qkv"},
        },
    )

    rewritten = _assert_rewrite_keeps_outputs(surgery, surgery.fold_score_scale, model)

    assert "Div" not in {node.op_type for node in rewritten.graph.node}


def test_gelu_constants_fold_into_the_mlp_weights(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    rng = np.random.default_rng(5)
    model = _model(
        [
            helper.make_node("MatMul", ["tokens", "fc1"], ["projected"]),
            helper.make_node("Add", ["fc1_bias", "projected"], ["hidden"]),
            helper.make_node("Div", ["hidden", "root_two"], ["scaled"]),
            helper.make_node("Erf", ["scaled"], ["erf"]),
            helper.make_node("Add", ["erf", "one"], ["shifted"]),
            helper.make_node("Mul", ["hidden", "shifted"], ["gated"]),
            helper.make_node("Mul", ["gated", "half"], ["gelu"]),
            helper.make_node("MatMul", ["gelu", "fc2"], ["output"]),
        ],
        {"tokens": [1, 10, 8]},
        ["output"],
        {
            "fc1": rng.standard_normal((8, 32)).astype(np.float32),
            "fc1_bias": rng.standard_normal(32).astype(np.float32),
            "fc2": rng.standard_normal((32, 8)).astype(np.float32),
            "root_two": np.asarray(np.sqrt(2), np.float32),
            "one": np.asarray(1.0, np.float32),
            "half": np.asarray(0.5, np.float32),
        },
    )

    rewritten = _assert_rewrite_keeps_outputs(surgery, surgery.fold_gelu_constants, model)

    assert [node.op_type for node in rewritten.graph.node] == ["MatMul", "Add", "Erf", "Add", "Mul", "MatMul"]


def test_window_attention_runs_in_head_window_order(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "WINDOWED_BLOCKS", 1)
    rng = np.random.default_rng(6)
    model = _model(
        [
            helper.make_node("Transpose", ["q_heads"], ["q"], perm=[0, 2, 1, 3]),
            helper.make_node("Transpose", ["k_heads"], ["k"], perm=[0, 2, 3, 1]),
            helper.make_node("Transpose", ["v_heads"], ["v"], perm=[0, 2, 1, 3]),
            helper.make_node("MatMul", ["q", "k"], ["scores"]),
            helper.make_node("Add", ["scores", "bias"], ["biased"]),
            helper.make_node("Softmax", ["biased"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", "v"], ["attended"]),
            helper.make_node("Transpose", ["attended"], ["merged"], perm=[0, 2, 1, 3]),
        ],
        {"q_heads": [4, 49, 2, 8], "k_heads": [4, 49, 2, 8], "v_heads": [4, 49, 2, 8]},
        ["merged"],
        # A shifted block's bias: the position bias with each window's mask already added.
        {"bias": rng.standard_normal((4, 2, 49, 49)).astype(np.float32)},
    )

    rewritten = _assert_rewrite_keeps_outputs(surgery, surgery.heads_before_windows, model)

    (product,) = [node for node in rewritten.graph.node if node.output[0] == "scores"]
    assert surgery._shapes(rewritten)[product.output[0]] == [2, 4, 49, 49]


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8", "w8a16"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe._variant(recipe._metadata(), profile_id)["parameters"]

    assert parameters["precision"] == profile_id
    assert parameters["input_shape"] == [1, 3, 224, 224]
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
