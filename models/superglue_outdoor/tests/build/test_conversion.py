"""Checks of the conversion's interface, its CPU post-processing reference and the 4D rewrite of the matcher."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_keypoints_normalize_to_superglue_frame(converter: ModuleType) -> None:
    # A 640x480 image: the centre maps to 0 and the longer side spans +-1/(2*0.7).
    normalized = converter.normalize_keypoints([[320, 240], [640, 240], [0, 0]], 640, 480)

    np.testing.assert_allclose(normalized, [[0, 0], [0.5 / 0.7, 0], [-0.5 / 0.7, -0.375 / 0.7]], rtol=1e-6)


def test_sinkhorn_and_extraction_find_clear_matches(converter: ModuleType) -> None:
    size = 6
    couplings = np.full((size + 1, size + 1), -2.0)
    permutation = np.array([3, 0, 5, 1, 4, 2])
    couplings[np.arange(size), permutation] = 8.0
    couplings[4] = -2.0  # keypoint 4 has no partner: its mass goes to the dustbin
    couplings[:, -1] = couplings[-1] = 1.0
    log_transport = converter.log_sinkhorn(couplings[None])

    matches, scores = converter.extract_matches(log_transport)
    # Extraction compares keypoints only, so the two leftovers still pair up, with a near-zero score...
    assert matches.tolist() == [3, 0, 5, 1, 4, 2]
    assert scores[4] < 0.1 and (np.delete(scores, 4) > 0.9).all()
    # ...and a match threshold drops them.
    assert converter.extract_matches(log_transport, threshold=0.2)[0].tolist() == [3, 0, 5, 1, -1, 2]


def test_extraction_requires_mutual_best_matches(converter: ModuleType) -> None:
    log_transport = np.log(np.array([[0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))

    matches, _ = converter.extract_matches(log_transport)

    # Keypoint 0 prefers 0, but image-1 keypoint 0 prefers keypoint 1: only 1 -> 0 is mutual.
    assert matches.tolist() == [-1, 0]


def test_exported_layout_round_trips(converter: ModuleType) -> None:
    pair = converter.synthetic_pair(7)
    inputs = converter.to_inputs(pair)

    assert {name: list(value.shape) for name, value in inputs.items()} == converter.INPUTS
    np.testing.assert_array_equal(inputs["descriptors0"][0, :, 0, :].T, pair["descriptors0"])
    np.testing.assert_array_equal(inputs["scores1"][0, 0, 0], pair["scores1"])
    couplings = np.arange((converter.KEYPOINTS + 1) ** 2, dtype=np.float32).reshape(converter.OUTPUT["couplings"])
    matrix = converter.couplings_matrix(couplings)
    # couplings[0, j, 0, i] is row i (image 0), column j (image 1).
    assert matrix[2, 5] == couplings[0, 5, 0, 2]


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, converter: ModuleType) -> None:
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    if case == "order":
        values[0], values[1] = values[1], values[0]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[-1].type.tensor_type.shape.dim[1].dim_value = 100
    inputs, outputs = values[:-1], values[-1:]
    output_shape = converter.OUTPUT["couplings"]
    node = helper.make_node(
        "Constant",
        [],
        [outputs[0].name],
        value=helper.make_tensor("value", onnx.TensorProto.FLOAT, output_shape, np.zeros(int(np.prod(output_shape)))),
    )
    graph = helper.make_graph([node], "fixture", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/superglue_outdoor")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="model.safetensors"):
        converter.validate_source({"uri": "hf://models/magic-leap-community/superglue_indoor@main/model.bin"})


def test_initializer_sharing_merges_identical_weights(converter: ModuleType) -> None:
    helper = onnx.helper
    weight = onnx.numpy_helper.from_array(np.ones((2, 2), np.float32), "w0")
    copy = onnx.numpy_helper.from_array(np.ones((2, 2), np.float32), "w1")
    nodes = [helper.make_node("MatMul", ["x", "w0"], ["y0"]), helper.make_node("MatMul", ["x", "w1"], ["y1"])]
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2, 2]) for name in ("x", "y0", "y1")]
    model = helper.make_model(helper.make_graph(nodes, "fixture", values[:1], values[1:], [weight, copy]))

    shared = converter.share_initializers(model)

    assert [item.name for item in shared.graph.initializer] == ["w0"]
    assert [node.input[1] for node in shared.graph.node] == ["w0", "w0"]


def test_mla_layout_matcher_equals_the_transformers_layers(converter: ModuleType) -> None:
    """The 4D rewrite with random weights must give the transformers layers' couplings."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.SuperGlueForKeypointMatching(transformers.SuperGlueConfig(**converter.CONFIG)).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean.uniform_(-0.2, 0.2)
            module.running_var.uniform_(0.5, 1.5)
    pair = converter.synthetic_pair(11)
    with torch.inference_mode():
        reference = converter.reference_matcher(model)
        expected = reference(*(torch.from_numpy(pair[name])[None] for name in converter.INPUTS))
        actual = converter.build_matcher(model)(
            *(torch.from_numpy(value) for value in converter.to_inputs(pair).values())
        )

    np.testing.assert_allclose(converter.couplings_matrix(actual.numpy()), expected[0].numpy(), atol=1e-4)


def test_converted_model_has_the_declared_interface(converted: Path, converter: ModuleType) -> None:
    """The conversion stage's real output: opset 17, the 4D inputs and the couplings output, self-contained."""
    converter.validate_export(converted)
