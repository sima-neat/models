"""Checks of the conversion's source, its reference export interface, and the converted weights and reference."""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def interface_fixture(converter: ModuleType, case: str) -> onnx.ModelProto:
    """A graph with the declared interface, or one broken in the named way; each output is a constant."""
    helper = onnx.helper
    shapes = {name: list(shape) for name, shape in converter.INTERFACE.items()}
    names = list(shapes)
    if case == "names":
        names[2] = "last_hidden_state"
    elif case == "shape":
        shapes["patch_embeddings"][1] = converter.PATCHES + 1
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in zip(names, shapes.values())
    ]
    if case == "dtype":
        values[1].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [value.name],
            value=helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(shape).ravel()),
        )
        for value, shape in zip(values[1:], list(shapes.values())[1:])
    ]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])


def test_the_interface_is_the_one_surgery_exports(
    converter: ModuleType, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")

    assert converter.INTERFACE == surgery.INTERFACE
    assert (converter.HIDDEN_SIZE, converter.PATCHES) == (surgery.HIDDEN_SIZE, surgery.PATCHES)


@pytest.mark.parametrize("case", ["valid", "opset", "names", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, converter: ModuleType) -> None:
    path = tmp_path / "reference.onnx"
    onnx.save(interface_fixture(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType) -> None:
    model = load_metadata(MODEL_DIR)
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source({"type": "file", "uri": "https://example.com/model.pth"})


@pytest.mark.parametrize("case", ["valid", "architecture", "classifier", "features"])
def test_config_must_name_the_architecture_without_a_classifier(
    tmp_path: Path, case: str, converter: ModuleType
) -> None:
    config = {"architecture": converter.ARCHITECTURE, "num_classes": 0, "num_features": converter.HIDDEN_SIZE}
    if case == "architecture":
        config["architecture"] = "convnext_nano"
    elif case == "classifier":
        config["num_classes"] = 1000
    elif case == "features":
        config["num_features"] = 2 * converter.HIDDEN_SIZE
    (tmp_path / "config.json").write_text(json.dumps(config))
    if case == "valid":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="config.json declares"):
            converter.validate_config(tmp_path)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "different"])
def test_output_comparison_rejects_shape_nan_and_excess_error(case: str, converter: ModuleType) -> None:
    expected = [np.ones(converter.INTERFACE[name], dtype=np.float32) for name in converter.OUTPUTS]
    actual = [value + converter.TOLERANCE / 2 for value in expected]
    if case == "shape":
        actual[1] = actual[1][:, 1:]
    elif case == "nan":
        actual[0][0, 0] = np.nan
    elif case == "different":
        actual[1][0, 3, 7] += 10 * converter.TOLERANCE
    if case == "valid":
        converter.compare_outputs(expected, actual)
    else:
        with pytest.raises((ValueError, AssertionError)):
            converter.compare_outputs(expected, actual)


def test_converted_weights_are_the_timm_convnext_state_dict(
    converted: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """The conversion stage's real output: a float32 state dict that surgery.py's network loads strictly."""
    torch = pytest.importorskip("torch")
    surgery = load_script("surgery")
    weights = torch.load(converted, weights_only=True)

    assert all(value.dtype == torch.float32 for value in weights.values())
    surgery.build_network().load_state_dict(weights, strict=True)


def test_converted_reference_returns_finite_class_and_patch_embeddings(converted: Path, converter: ModuleType) -> None:
    import onnxruntime as ort

    _, reference = converter.conversion_outputs(MODEL_DIR)
    path = converted.with_name(reference)
    converter.validate_export(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).normal(size=converter.INTERFACE["pixel_values"]).astype(np.float32)
    class_embedding, patch_embeddings = session.run(list(converter.OUTPUTS), {"pixel_values": image})
    assert np.isfinite(class_embedding).all() and np.isfinite(patch_embeddings).all()
    # The reference keeps the exact GELU.
    assert "Erf" in {node.op_type for node in onnx.load(path).graph.node}
    # Different patches of a varied image get different embeddings.
    assert np.std(patch_embeddings[0], axis=0).mean() > 0
