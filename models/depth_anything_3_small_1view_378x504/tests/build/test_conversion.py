"""Conversion exports the original forward pass; compiler rewrites belong to surgery.py."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from model_registry.model import load_model
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def test_source_has_exact_weights_config_and_upstream_code(model_dir, load_script):
    converter = load_script("convert")
    source = load_model(model_dir)["conversion"]["default"]["source"]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="file set"):
        converter.validate_source({**source, "files": source["files"][:-1]})


def test_cartesian_product_export_preserves_pair_order(load_script, tmp_path):
    torch = pytest.importorskip("torch")
    converter = load_script("convert")

    class Cartesian(torch.nn.Module):
        def forward(self, rows, columns):
            return torch.cartesian_prod(rows, columns)

    rows, columns = torch.tensor([2, 5, 9]), torch.tensor([4, 8])
    path = tmp_path / "cartesian.onnx"
    torch.onnx.register_custom_op_symbolic("aten::cartesian_prod", converter.cartesian_product, 17)
    try:
        torch.onnx.export(
            Cartesian(),
            (rows, columns),
            str(path),
            opset_version=17,
            input_names=["rows", "columns"],
            output_names=["pairs"],
            dynamo=False,
        )
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::cartesian_prod", 17)
    (actual,) = ReferenceEvaluator(onnx.load(path)).run(None, {"rows": rows.numpy(), "columns": columns.numpy()})
    np.testing.assert_array_equal(actual, [[2, 4], [2, 8], [5, 4], [5, 8], [9, 4], [9, 8]])


def test_conversion_exports_original_static_onnx(converted: Path, load_script):
    assert converted.name == "model.onnx"
    load_script("convert").validate_export(converted)
    model = onnx.load(converted)
    # The source retains upstream camera assignment, packed QKV and the learned two-channel projection.
    assert sum(node.op_type == "ScatterND" for node in model.graph.node) == 1
    assert (
        sum(
            node.op_type == "Transpose" and helper.get_attribute_value(node.attribute[0]) == [2, 0, 3, 1, 4]
            for node in model.graph.node
        )
        == 12
    )
    assert sum(numpy_helper.to_array(value).shape == (2, 32, 1, 1) for value in model.graph.initializer) == 1
