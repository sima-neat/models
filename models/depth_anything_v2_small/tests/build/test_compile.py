from __future__ import annotations

import io
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import onnx
import pytest
from model_registry.errors import ValidationError
from onnx import helper, numpy_helper


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


def test_the_benchmarks_read_the_directory_the_runners_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """The runners look through the archive's one top-level directory, so the split sits directly under --dataset."""
    arguments = ["--mpk", "model=m.tar.gz", "--dataset", f"nyu-depth-v2-val={tmp_path}", "--output", str(tmp_path)]
    message = f"{tmp_path} has no images under test/"

    with pytest.raises(SystemExit, match=message):
        load_script("accuracy").main(arguments)
    # performance.py parses its arguments and runs as it loads.
    monkeypatch.setattr("sys.argv", ["performance.py", *arguments])
    with pytest.raises(SystemExit, match=message):
        load_script("performance")


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8_mixed", "w8a16"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe.component_parameters(recipe.MODEL_DIR, profile_id, recipe.COMPONENT_ID)

    assert parameters["precision"] == profile_id.removesuffix("_mixed")
    assert parameters["input_shape"] == [1, 3, 518, 686]
    if profile_id == "w8a8_mixed":
        assert parameters["calibration_samples"] == 10
        assert parameters["bf16_regions"] == ["normalization_residual", "attention"]


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        recipe.component_parameters(recipe.MODEL_DIR, "w32a32", recipe.COMPONENT_ID)


def test_compile_rejects_another_input_shape_before_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    parameters = recipe.component_parameters(recipe.MODEL_DIR, "w16a16", recipe.COMPONENT_ID)
    parameters["input_shape"] = [1, 3, 518, 518]
    monkeypatch.setattr(recipe, "component_parameters", lambda model_dir, profile_id, component_id: parameters)

    with pytest.raises(ValueError, match="unsupported Depth Anything V2 input shape"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16", None)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8_mixed", "w8a16"])
def test_only_int8_activations_need_calibration(load_script, monkeypatch, profile_id):
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "QuantizationParams", SimpleNamespace)
    monkeypatch.setattr(recipe, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(recipe, "quantization_scheme", lambda **kwargs: kwargs)
    monkeypatch.setattr(recipe, "SkipCalibration", lambda: "skip")
    monkeypatch.setattr(recipe, "CalibrationMethod", SimpleNamespace(from_str=lambda name: name))
    precision = recipe.component_parameters(recipe.MODEL_DIR, profile_id, recipe.COMPONENT_ID)["precision"]

    config = recipe.quantization(precision)

    assert config.calibration_method == ("mse" if profile_id == "w8a8_mixed" else "skip")
    assert config.per_token_quant == (profile_id == "w8a16")
    assert config.activation_quantization_scheme == (
        {"asymmetric": True, "per_channel": False, "bits": 8} if profile_id == "w8a8_mixed" else "bf16"
    )
    assert config.weight_quantization_scheme == (
        "bf16" if profile_id == "w16a16" else {"asymmetric": False, "per_channel": True, "bits": 8}
    )


@pytest.mark.parametrize(
    ("profile", "parameters", "message"),
    [
        ("w8a8_mixed", {"precision": "w8a8"}, "mixed token"),
        ("w8a8", {"precision": "w8a8", "bf16_regions": ["attention"]}, "mixed token"),
        ("w16a16_mixed", {"precision": "w16a16", "bf16_regions": ["attention"]}, "W8A8 base precision"),
    ],
)
def test_mixed_precision_requires_a_matching_profile(load_script, monkeypatch, tmp_path, profile, parameters, message):
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "component_parameters", lambda model_dir, profile_id, component_id: parameters)
    with pytest.raises(ValueError, match=message):
        recipe.compile_model(tmp_path / "source.onnx", tmp_path / "model", profile, None)


def test_fidelity_samples_are_the_declared_selection_preprocessed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    selections = []

    def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
        selections.append((dataset, seed, offset, samples))
        return ["first", "second"]

    monkeypatch.setattr(recipe, "fidelity_names", fidelity_names)
    monkeypatch.setattr(recipe, "read_images", lambda dataset, names: {name: name.encode() for name in names})
    monkeypatch.setattr(recipe, "preprocess_image", lambda content: content.decode().upper())

    samples = recipe.fidelity_samples(tmp_path / "dataset")

    selection = recipe.load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]["selection"]
    assert selections == [(tmp_path / "dataset", selection["seed"], selection["offset"], selection["samples"])]
    assert samples == ["FIRST", "SECOND"]


def test_precision_hints_end_at_the_first_compute_reader(load_script, monkeypatch):
    precision = load_script("precision")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["input", "input"], ["residual"], name="residual"),
                helper.make_node("Identity", ["residual"], ["layout"], name="layout"),
                helper.make_node("Relu", ["layout"], ["output"], name="head"),
            ],
            "boundary",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
        )
    )
    original = [node.SerializeToString() for node in model.graph.node]
    monkeypatch.setattr(precision, "regions", lambda _: {"normalization_residual": {"residual"}})

    precision.annotate_bf16_regions(model, ["normalization_residual"])

    onnx.checker.check_model(model)
    hints = {node.output[0]: node for node in model.graph.node if node.domain == "ai.sima"}
    assert {name: helper.get_attribute_value(node.attribute[0]) for name, node in hints.items()} == {
        "residual": b"bfloat16",
        "output": b"int8",
    }
    restored = []
    for node in model.graph.node:
        if node.domain != "ai.sima":
            node.output[:] = [value.removesuffix("/unannotated") for value in node.output]
            restored.append(node.SerializeToString())
    assert restored == original


@pytest.mark.parametrize("regions", [["unknown"], ["attention", "attention"]])
def test_precision_hints_reject_unknown_or_duplicate_regions(load_script, regions):
    precision = load_script("precision")
    with pytest.raises(ValueError, match="distinct BF16 regions"):
        precision.annotate_bf16_regions(onnx.ModelProto(), regions)


@pytest.mark.parametrize("form", ["gemm", "matmul_bias_first", "matmul_bias_last"])
def test_batched_attention_projections_preserve_outputs(load_script, form):
    import onnxruntime as ort

    surgery = load_script("surgery")
    nodes, initializers, outputs = [], [], []
    rng = np.random.default_rng(42)
    initializers.append(numpy_helper.from_array(rng.normal(0, 0.02, (384, 384)).astype(np.float32), "weight"))
    initializers.extend(
        [
            numpy_helper.from_array(np.array([1814, 384], np.int64), "flat_shape"),
            numpy_helper.from_array(np.array([1, 1814, 384], np.int64), "shape"),
        ]
    )
    for index in range(12):
        name = f"block{index}"
        bias = name + ".attn.proj.bias"
        initializers.append(numpy_helper.from_array(rng.normal(0, 0.02, 384).astype(np.float32), bias))
        if form == "gemm":
            nodes.extend(
                [
                    helper.make_node("Reshape", ["input", "flat_shape"], [name + "/flat"], name=name + "/flatten"),
                    helper.make_node(
                        "Gemm",
                        [name + "/flat", "weight", bias],
                        [name + "/projected"],
                        name=name + "/Gemm",
                        alpha=1.0,
                        beta=1.0,
                        transA=0,
                        transB=0,
                    ),
                    helper.make_node("Reshape", [name + "/projected", "shape"], [name], name=name + "/restore"),
                ]
            )
        else:
            inputs = [bias, name + "/projected"] if form == "matmul_bias_first" else [name + "/projected", bias]
            nodes.extend(
                [
                    helper.make_node("MatMul", ["input", "weight"], [name + "/projected"], name=name + "/MatMul"),
                    helper.make_node("Add", inputs, [name], name=name + "/Add"),
                ]
            )
        outputs.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1814, 384]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "attention",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1814, 384])],
            outputs,
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 14)],
        ir_version=8,
    )
    model = onnx.shape_inference.infer_shapes(model)
    original = model.SerializeToString()
    rewritten = surgery.batched_attention_projections(model)
    onnx.checker.check_model(rewritten)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    image = rng.normal(size=(1, 1814, 384)).astype(np.float32)
    expected = ort.InferenceSession(original, options).run(None, {"input": image})
    actual = ort.InferenceSession(rewritten.SerializeToString(), options).run(None, {"input": image})
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)


def test_unfused_learned_attention_projections_stay_int8(load_script):
    precision = load_script("precision")
    nodes = [
        helper.make_node("Div", ["input", "input"], [f"norm{index}"], name=f"/norm/Div{index}") for index in range(106)
    ]
    for index in range(12):
        prefix = f"/blocks.{index}/attn/"
        for name, operation, inputs in [
            ("Mul", "Mul", ["input", "scale"]),
            ("MatMul", "MatMul", ["q", "k"]),
            ("Softmax", "Softmax", ["scores"]),
            ("MatMul_1", "MatMul", ["probabilities", "v"]),
            ("qkv/MatMul", "MatMul", ["input", "weights"]),
            ("proj/MatMul", "MatMul", ["input", "weights"]),
        ]:
            nodes.append(helper.make_node(operation, inputs, [prefix + name], name=prefix + name))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "regions",
            [],
            [],
            [
                numpy_helper.from_array(np.eye(384, dtype=np.float32), "weights"),
            ],
        )
    )

    selected = precision.regions(model)["attention"]

    assert len(selected) == 48
    assert not any("/qkv/" in name or "/proj/" in name for name in selected)


def test_compiled_package_keeps_the_full_resolution_depth_map(compiled, model_dir, build_profile, load_script):
    archive = compiled / "model" / f"{model_dir.name}_{build_profile}_mpk.tar.gz"
    load_script("compile").audit_mpk(archive)


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
