"""Fold static shapes, preserve batched attention projections and apply the mixed-precision regions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from precision import annotate_bf16_regions

INPUT = "pixel_values"
OUTPUT = "predicted_depth"
INPUT_SHAPE = [1, 3, 518, 686]
OUTPUT_SHAPE = [1, 518, 686]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unexpected Depth Anything V2 graph: {message}")


def _shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    return {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}


def fold(source: Path) -> onnx.ModelProto:
    """Fix the export's dynamic shapes to one 518x686 image and fold every shape computation."""
    from onnxsim import simplify

    model, valid = simplify(onnx.load(source), overwrite_input_shapes={INPUT: INPUT_SHAPE})
    _require(valid, "onnxsim could not validate the folded graph")
    model = onnx.shape_inference.infer_shapes(model)
    shapes = _shapes(model)
    _require(shapes.get(INPUT) == INPUT_SHAPE and shapes.get(OUTPUT) == OUTPUT_SHAPE, "input or output interface")
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Shape", "ScatterND", "Range", "Where"})
    _require(not unfolded, f"shape computations survived folding: {unfolded}")
    return model


def batched_attention_projections(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace flattened Gemm projections with equivalent NLC MatMul + bias.

    The frontend can otherwise fold the attention's token axis into its constant projection weights.
    This is the graph form used when qualifying the mixed-precision recipe.
    """
    producers = {value: node for node in model.graph.node for value in node.output}
    users = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            users[value].append(node)
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    shapes = _shapes(model)
    replacements, removed = {}, set()
    for node in model.graph.node:
        if node.op_type != "Gemm" or ".attn.proj.bias" not in node.input[-1]:
            continue
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        _require(attributes == {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}, "attention Gemm")
        before = producers[node.input[0]]
        after = users[node.output[0]]
        _require(len(after) == 1, "shared flattened attention projection")
        after = after[0]
        _require(before.op_type == after.op_type == "Reshape", "attention projection reshapes")
        _require(len(users[before.output[0]]) == 1, "shared flattened attention input")
        _require(constants[node.input[1]].shape == (384, 384), "attention projection weights")
        _require(shapes[after.output[0]] == [1, 1814, 384], "attention projection output")
        prefix = node.name + "/batched"
        shape, weight = prefix + "/shape", prefix + "/weight"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.array([1, 1814, 384], np.int64), shape),
                numpy_helper.from_array(constants[node.input[1]][None], weight),
            ]
        )
        replacements[before.name] = [
            helper.make_node("Reshape", [before.input[0], shape], [prefix + "/input"], name=prefix + "/Reshape"),
            helper.make_node("MatMul", [prefix + "/input", weight], [prefix + "/result"], name=prefix + "/MatMul"),
            helper.make_node("Add", [prefix + "/result", node.input[2]], list(after.output), name=prefix + "/Add"),
        ]
        removed.update((before.name, node.name, after.name))
    _require(len(replacements) == 12, "expected twelve attention output projections")
    nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.extend(replacements[node.name])
        elif node.name not in removed:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def prepare_model(source: Path, output: Path, bf16_regions: list[str] | None = None) -> Path:
    """Write the graph the recipe imports."""
    model = fold(source)
    if bf16_regions:
        model = batched_attention_projections(model)
        annotate_bf16_regions(model, bf16_regions)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return output


def max_difference(source: Path, prepared: Path, image: np.ndarray) -> float:
    """Largest depth difference between the source and the prepared graph on one NCHW image."""
    import onnxruntime as ort

    outputs = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run([OUTPUT], {INPUT: image})[0]
        for path in (source, prepared)
    ]
    return float(np.max(np.abs(outputs[0] - outputs[1])))
