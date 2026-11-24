from __future__ import annotations

import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
from model_registry.dependencies import onnx_opset_version
from model_registry.errors import ValidationError
from model_registry.model import load_model as load_metadata
from PIL import Image

OPSET = onnx_opset_version()
PROFILES = ("w16a16", "w8a16", "w8a8_mixed")


@pytest.fixture
def builder(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("surgery")


def _png(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
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


def test_preprocessing_resizes_to_640x480_and_applies_imagenet_normalization(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # A 320x320 image, red above blue, stretches to 640x480 without cropping: rows 0-239 red, 240-479 blue.
    image = Image.new("RGB", (320, 320), (255, 0, 0))
    image.paste((0, 0, 255), (0, 160, 320, 320))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert result.shape == (480, 640, 3)
    assert result.dtype == np.float32
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[0, 0], red, atol=1e-5)
    np.testing.assert_allclose(result[230, 639], red, atol=1e-5)
    np.testing.assert_allclose(result[250, 0], blue, atol=1e-5)
    np.testing.assert_allclose(result[479, 639], blue, atol=1e-5)


def test_selection_is_a_seeded_shuffle_of_the_pittsburgh_images_only(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # pitts30k/cal keeps its images under pitts30k/images/<tar>/, beside the split, the list and the README.
    images = {
        f"pitts30k/images/00{index % 2}/{index:06d}_pitch1_yaw1.jpg": _png(8, 8, (index, 0, 0)) for index in range(6)
    }
    archive = _write_archive(
        tmp_path / "cal.tgz",
        {
            **images,
            "pitts30k/pitts30k_train.mat": b"spec",
            "pitts30k/cal.txt": b"list",
            "pitts30k/README.txt": b"readme",
        },
    )

    shuffled = calibration.select_images(archive, 42, 0, 6)

    assert sorted(shuffled) == sorted(images)
    assert calibration.select_images(archive, 42, 2, 3) == shuffled[2:5]
    assert calibration.load_images(archive, shuffled[:2]).shape == (2, 480, 640, 3)
    with pytest.raises(ValueError, match="has 6 Pittsburgh images, but 7 are required"):
        calibration.select_images(archive, 42, 4, 3)


def test_calibration_images_precede_the_fidelity_selection(
    tmp_path: Path, model_dir: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    root = tmp_path / "cal"
    (root / "pitts30k" / "images" / "000").mkdir(parents=True)
    for index in range(80):
        (root / "pitts30k" / "images" / "000" / f"{index:06d}_pitch1_yaw1.jpg").write_bytes(b"")
    metadata = load_metadata(model_dir)
    selection = metadata["numerical_fidelity"]["selection"]
    fidelity = recipe.select_images(root, selection["seed"], selection["offset"], selection["samples"])

    for profile in PROFILES:
        count = recipe.component_parameters(model_dir, profile, recipe.COMPONENT_ID)["calibration_samples"]
        calibration = recipe.calibration_names(root, selection, count)
        # model.yaml calibrates every build on 20 images and compares 5 more, starting at offset 50.
        assert (len(calibration), len(fidelity)) == (20, 5)
        assert not set(calibration) & set(fidelity)
    with pytest.raises(ValueError, match="must begin after the calibration images"):
        recipe.calibration_names(root, {**selection, "offset": 49}, 50)


@pytest.mark.parametrize(("precision", "per_token"), [("w8a8", True), ("w8a16", True), ("w16a16", False)])
def test_every_build_calibrates_and_only_bf16_weights_skip_per_token_quantization(
    precision: str, per_token: bool, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    recipe.quantization(precision, 2048)

    arguments = recipe.QuantizationParams.call_args.kwargs
    # The VLAD norm's Sqrt is an INT8 lookup table even in the BF16 builds, so every build calibrates its range.
    recipe.HistogramMSEMethod.assert_called_once_with(2048)
    assert arguments["calibration_method"] is recipe.HistogramMSEMethod.return_value
    assert arguments["activation_quantization_scheme"] is recipe.bfloat16_scheme.return_value
    assert arguments.get("per_token_quant", False) is per_token
    with pytest.raises(ValueError, match="unsupported precision"):
        recipe.quantization("w4a8", 2048)


@pytest.mark.parametrize(
    ("profile", "parameters", "message"),
    [
        ("w8a8", {"precision": "w8a8"}, "only a W8A8 build, runs bf16_regions"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["encoder"]}, "only a W8A8 build, runs bf16_regions"),
        ("w8a16_mixed", {"precision": "w8a16", "bf16_regions": ["pooling"]}, "only a W8A8 build, runs bf16_regions"),
        ("w8a8_pooling", {"precision": "w8a8", "bf16_regions": ["pooling"]}, "must have the mixed token"),
        ("w8a16_mixed", {"precision": "w8a16"}, "must have the mixed token"),
    ],
)
def test_only_a_mixed_w8a8_profile_runs_its_pooling_bf16(
    profile: str, parameters: dict[str, Any], message: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    assert recipe.bf16_regions("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["pooling"]}) == ["pooling"]
    assert recipe.bf16_regions("w8a16", {"precision": "w8a16"}) == []
    with pytest.raises(ValueError, match=message):
        recipe.bf16_regions(profile, parameters)


@pytest.mark.parametrize(("int8_convolutions", "honored"), [(12, True), (0, False), (13, False)])
def test_a_mixed_build_fails_unless_afe_runs_exactly_the_marked_convolutions_int8(
    int8_convolutions: int, honored: bool, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    class ConvAddActivationOp:
        pass

    def node(precision: Any) -> MagicMock:
        item = MagicMock()
        item.ir.operation = ConvAddActivationOp()
        item.ir.quant_config.quantization_precision.value = precision
        return item

    quantized = MagicMock()
    int8, bf16 = recipe.QuantizationPrecision.INT_8, recipe.QuantizationPrecision.BFLOAT_16_INT8_WEIGHTS
    quantized._net.iter_nodes_recursive.return_value = [node(int8)] * int8_convolutions + [node(bf16)] * 3

    if honored:
        recipe.require_bf16_pooling(quantized)
    else:
        # Too few means AFE ignored the marks; too many means INT8 reached conv5_3 or the pooling.
        with pytest.raises(RuntimeError, match="expected the encoder's 12 convolutions before conv5_3 in INT8"):
            recipe.require_bf16_pooling(quantized)


def test_compile_reads_the_declared_build_parameters(model_dir: Path, load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = load_metadata(model_dir)
    component_id = recipe.COMPONENT_ID
    calibration = {"calibration_bins": 2048, "calibration_samples": 20}

    assert recipe.component_parameters(model_dir, "w16a16", component_id) == {"precision": "w16a16", **calibration}
    assert recipe.component_parameters(model_dir, "w8a16", component_id) == {"precision": "w8a16", **calibration}
    assert recipe.component_parameters(model_dir, "w8a8_mixed", component_id) == {
        "precision": "w8a8",
        "bf16_regions": ["pooling"],
        **calibration,
    }
    assert metadata["numerical_fidelity"]["outputs"] == [recipe.OUTPUT]
    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        recipe.component_parameters(model_dir, "w32a32", component_id)


def _mpk_manifest() -> dict[str, Any]:
    return {
        # One float32 480x640 RGB image in, one float32 4096-value descriptor out.
        "input_nodes": [{"name": "image", "size": 3 * 480 * 640 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": 4096 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict[str, Any]) -> None:
    _write_archive(path, {"model.elf": b"elf", "netvlad_mpk.json": json.dumps(manifest).encode()})


@pytest.mark.parametrize("case", ["valid", "two_mla", "host", "input", "output", "mla_output"])
def test_mpk_audit_requires_one_mla_segment_and_the_descriptor_interface(
    tmp_path: Path,
    case: str,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    manifest = _mpk_manifest()
    plugins = manifest["plugins"]
    if case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    elif case == "input":
        manifest["input_nodes"][0]["size"] = 3 * 224 * 224 * 4
    elif case == "output":
        plugins[-1]["output_nodes"] = [{"size": 512 * 4}]
    elif case == "mla_output":
        # The descriptor must come back through the float32 conversion, not straight from the MLA.
        plugins[1]["output_nodes"] = plugins.pop()["output_nodes"]
    archive = tmp_path / "netvlad_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        recipe.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(archive)


def test_compile_reports_the_compilers_silent_exit_as_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    loaded = MagicMock()
    # AFE raised a bare SystemExit when it could not segment NetVLAD's 32768-wide L2 normalization at INT8.
    loaded.quantize.return_value.compile.side_effect = SystemExit()
    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: loaded)
    parameters = {"precision": "w16a16", "calibration_bins": 2048, "calibration_samples": 1}

    with pytest.raises(RuntimeError, match="failed to compile"):
        recipe.compile_model(
            tmp_path / "model.onnx", tmp_path / "out", "w16a16", parameters, np.zeros((1, 480, 640, 3))
        )


def test_fidelity_compares_the_quantized_descriptors_with_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    fidelity = load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]
    pixel_values = (1.0, 2.0)
    images = np.stack([np.full((480, 640, 3), value) for value in pixel_values])
    reference_inputs = []

    def reference(image: np.ndarray) -> np.ndarray:
        reference_inputs.append(image.shape)
        return np.full(4096, image[0, 0, 0, 0], np.float64)

    candidate = MagicMock()
    # Each image's last descriptor value comes back 1 too high.
    candidate.execute.side_effect = [
        [np.append(np.full(4095, value), value + 1).reshape(1, 4096)] for value in pixel_values
    ]
    monkeypatch.setattr(recipe, "select_images", lambda dataset, seed, offset, samples: ["a.jpg", "b.jpg"])
    monkeypatch.setattr(recipe, "load_images", lambda dataset, names: images)

    report = recipe.measure_fidelity(reference, candidate, tmp_path / "cal", fidelity, "w8a8_mixed")

    assert reference_inputs == [(1, 3, 480, 640)] * 2
    assert (report["model_id"], report["build_profile_id"], report["samples"]) == ("netvlad", "w8a8_mixed", 2)
    (metric,) = report["metrics"]
    assert metric["name"] == "descriptor"
    assert metric["max_absolute_error"] == 1.0
    assert metric["mean_absolute_error"] == pytest.approx(1 / 4096)


@pytest.mark.parametrize(
    ("model", "arguments", "message"),
    [
        ("weights.pth", ["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        ("model.onnx", ["--compile-invocation", "w16a16"], "--model must be the verified weights.pth"),
        (
            "weights.pth",
            ["--compile-invocation", "w16a16"],
            "every build calibrates; pass the dataset-registry --dataset",
        ),
        ("weights.pth", ["--compile-invocation", "w16a16", "--dataset", "{cal}"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_a_wrong_invocation_model_or_missing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    model: str,
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    (tmp_path / model).touch()
    (tmp_path / "cal").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(tmp_path / model),
            "--output-dir",
            str(tmp_path / "out"),
            "--build-profile",
            "w16a16",
        ]
        + [argument.format(cal=tmp_path / "cal") for argument in arguments],
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


def test_reference_aggregates_soft_assigned_residuals_per_cluster(builder: ModuleType) -> None:
    # Two clusters over three-channel features with an identity WPCA, so the descriptor is the normalized VLAD.
    centroids = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assignment = np.zeros((2, 3, 1, 1))
    identity, no_bias = np.eye(6), np.zeros(6)
    # One cell holding 3 * e_0 normalizes to e_0; equal logits split it evenly, leaving residual e_0 - c_k per cluster.
    one_cell = np.asarray([3.0, 0.0, 0.0]).reshape(3, 1, 1)
    expected = np.asarray([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]) / np.sqrt(2) / np.sqrt(2)

    descriptor = builder.reference_descriptor(one_cell, centroids, assignment, identity, no_bias)

    np.testing.assert_allclose(descriptor, expected.ravel())
    # A strong assignment to cluster 0 moves the cell's weight there; cluster 1's residual shrinks to zero.
    assignment[0, 0] = 100.0
    descriptor = builder.reference_descriptor(one_cell, centroids, assignment, identity, no_bias)
    np.testing.assert_allclose(descriptor, np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2), atol=1e-12)


def test_descriptor_comparison_rejects_shape_nan_and_drift(builder: ModuleType) -> None:
    expected = builder._l2_normalize(np.random.RandomState(1).normal(0, 1, builder.DESCRIPTOR_DIM), 0)
    actual = expected[None].astype(np.float32)

    assert builder.compare_descriptors(expected, actual) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unexpected or non-finite"):
        builder.compare_descriptors(expected, actual[:, :512])
    with pytest.raises(ValueError, match="Unexpected or non-finite"):
        builder.compare_descriptors(expected, np.where(np.arange(actual.size) == 7, np.nan, actual))
    drifted = actual.copy()
    drifted[0, 0] += 1e-3
    with pytest.raises(AssertionError, match="descriptor"):
        builder.compare_descriptors(expected, drifted)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape", "unfolded"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, builder: ModuleType) -> None:
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in builder.INTERFACE.items()
    ]
    if case == "order":
        values.reverse()
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[1].type.tensor_type.shape.dim[1].dim_value = 512
    zeros = onnx.numpy_helper.from_array(np.zeros((1, 4096), np.float32), "zeros")
    if case == "unfolded":
        node = helper.make_node("Constant", [], [values[1].name], value=zeros)
    else:
        node = helper.make_node("Identity", ["zeros"], [values[1].name])
    graph = helper.make_graph([node], "fixture", values[:1], values[1:], [zeros])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET + 1 if case == "opset" else OPSET)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        builder.validate_export(path)
    else:
        with pytest.raises(ValueError):
            builder.validate_export(path)


def test_encoder_keeps_torchvision_vgg16_layer_indices(builder: ModuleType) -> None:
    torch = pytest.importorskip("torch")
    encoder = builder.build_encoder()

    # torchvision's vgg16().features[:29]: 13 convolutions, 4 max pools, a ReLU after every convolution but conv5_3.
    convolutions = [index for index, layer in enumerate(encoder) if isinstance(layer, torch.nn.Conv2d)]
    pools = [index for index, layer in enumerate(encoder) if isinstance(layer, torch.nn.MaxPool2d)]
    assert len(encoder) == 29 and (convolutions[-1], pools) == (28, [4, 9, 16, 23])
    assert len(convolutions) == 13 and encoder[28].out_channels == 512


def test_encoder_matches_torchvision_vgg16_features_layer_by_layer(builder: ModuleType) -> None:
    pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    expected = torchvision.models.vgg16(weights=None).features[:29]

    # repr lists each layer's kernel, stride, padding and channels, so equal reprs mean the same layers.
    assert [repr(layer) for layer in builder.build_encoder()] == [repr(layer) for layer in expected]


def test_int8_marks_cover_the_encoder_up_to_conv5_3_and_leave_its_normalization_unsplit(
    tmp_path: Path, builder: ModuleType
) -> None:
    helper = onnx.helper
    conv5_3 = "/encoder/encoder.28/Conv_output_0"
    # A stand-in for the export: two encoder operations, then the per-cell normalization reading conv5_3.
    nodes = [
        helper.make_node("Relu", ["image"], ["/encoder/encoder.0/Relu_output_0"], name="/encoder/encoder.0/Relu"),
        helper.make_node("Relu", ["/encoder/encoder.0/Relu_output_0"], [conv5_3], name="/encoder/encoder.28/Conv"),
        helper.make_node("ReduceL2", [conv5_3], ["norm"], name="/ReduceL2", axes=[1], keepdims=1),
        helper.make_node("Clip", ["norm", "eps"], ["clipped"], name="/Clip"),
        helper.make_node("Div", [conv5_3, "clipped"], ["descriptor"], name="/Div"),
    ]
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 2]) for name in ("image", "descriptor")]
    eps = onnx.numpy_helper.from_array(np.asarray(1e-12, np.float32), "eps")
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:], [eps])
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)]), path)

    assert builder.annotate_int8_encoder(path) == {"int8": 1, "bfloat16": 1}

    marks = {
        node.name.removesuffix("/AnnotatePrecision"): node.attribute[0].s.decode()
        for node in onnx.load(path).graph.node
        if node.op_type == "AnnotatePrecision"
    }
    # No mark splits the normalization, so AFE can fuse it into one L2Norm that inherits conv5_3's BF16.
    assert marks == {"/encoder/encoder.0/Relu": "int8", "/encoder/encoder.28/Conv": "bfloat16"}


def test_compiled_mpk_has_one_mla_segment_and_the_descriptor_interface(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    recipe.audit_mpk(compiled / "model" / f"netvlad_{build_profile}_mpk.tar.gz")


def test_compiled_build_reports_descriptor_fidelity(compiled: Path, build_profile: str) -> None:
    report = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))

    assert (report["build_profile_id"], report["samples"]) == (build_profile, 5)
    (metric,) = report["metrics"]
    assert metric["name"] == "descriptor"
    assert np.isfinite([metric["cosine_similarity"], metric["normalized_rmse"]]).all()
