"""Fuse each ViT-B/16 block's query, key and value projections into one, with the attention scale in its weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

INPUT = "pixel_values"
OUTPUT = "logits"
LAYERS = 12
HEAD_SIZE = 64
# The export scales query and key by HEAD_SIZE ** -0.25 each. Their product, 1/8, is a power of two, so it moves into
# the query weights and bias without rounding them.
ATTENTION_SCALE = np.float32(HEAD_SIZE**-0.5)
PROJECTIONS = ("query", "key", "value")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Unsupported ViT-B/16 graph: {message}")


def _shapes(values) -> dict[str, list[int]]:
    return {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim] for value in values}


def _fuse_attention_projections(model: onnx.ModelProto, layer: int) -> None:
    """Replace the block's three projections and two scale multiplies with one projection and a Split."""
    prefix = f"/vit/encoder/layer.{layer}/attention/attention"
    nodes = {node.name: node for node in model.graph.node}
    weights = {value.name: value for value in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    matmuls = [nodes.get(f"{prefix}/{projection}/MatMul") for projection in PROJECTIONS]
    adds = [nodes.get(f"{prefix}/{projection}/Add") for projection in PROJECTIONS]
    _require(all(matmuls) and all(adds), f"layer {layer} lacks a query, key or value MatMul and Add")
    _require(len({matmul.input[0] for matmul in matmuls}) == 1, f"layer {layer} projections read different inputs")
    weight_arrays, bias_arrays = [], []
    for matmul, add in zip(matmuls, adds):
        _require(matmul.input[1] in weights, f"{matmul.name} has no constant weight")
        (bias,) = [name for name in add.input if name != matmul.output[0]]
        _require(bias in weights, f"{add.name} has no constant bias")
        weight_arrays.append(numpy_helper.to_array(weights[matmul.input[1]]))
        bias_arrays.append(numpy_helper.to_array(weights[bias]))
    width = weight_arrays[0].shape[1]
    _require(all(w.shape == weight_arrays[0].shape for w in weight_arrays), f"layer {layer} projections differ")

    def only_consumer(value: str, op_type: str) -> onnx.NodeProto:
        readers = consumers.get(value, [])
        _require(len(readers) == 1 and readers[0].op_type == op_type, f"{value} does not feed one {op_type}")
        return readers[0]

    # Each scale Mul follows the head split of the query or the key: Add -> Reshape -> Transpose -> Mul.
    scales = []
    for add in adds[:2]:
        transpose = only_consumer(only_consumer(add.output[0], "Reshape").output[0], "Transpose")
        scale = only_consumer(transpose.output[0], "Mul")
        factor = [name for name in scale.input if name != transpose.output[0]]
        _require(len(factor) == 1 and factor[0] in weights, f"{scale.name} has no constant factor")
        scales.append((scale, transpose.output[0], float(numpy_helper.to_array(weights[factor[0]]).squeeze())))
    product = scales[0][2] * scales[1][2]
    _require(np.isclose(product, ATTENTION_SCALE, rtol=1e-6), f"layer {layer} scales attention by {product}")

    weight_arrays[0] = weight_arrays[0] * ATTENTION_SCALE
    bias_arrays[0] = bias_arrays[0] * ATTENTION_SCALE
    names = {kind: f"{prefix}/qkv/{kind}" for kind in ("weight", "bias", "sizes", "MatMul", "Add", "Split")}
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.concatenate(weight_arrays, axis=1), names["weight"]),
            numpy_helper.from_array(np.concatenate(bias_arrays), names["bias"]),
            numpy_helper.from_array(np.asarray([width] * len(PROJECTIONS), np.int64), names["sizes"]),
        ]
    )
    fused = [
        onnx.helper.make_node(
            "MatMul", [matmuls[0].input[0], names["weight"]], [f"{names['MatMul']}_output_0"], name=names["MatMul"]
        ),
        onnx.helper.make_node(
            "Add", [f"{names['MatMul']}_output_0", names["bias"]], [f"{names['Add']}_output_0"], name=names["Add"]
        ),
        # The Split keeps each projection's original output name, so the head reshapes read it unchanged.
        onnx.helper.make_node(
            "Split",
            [f"{names['Add']}_output_0", names["sizes"]],
            [add.output[0] for add in adds],
            name=names["Split"],
            axis=-1,
        ),
    ]

    for scale, unscaled, _ in scales:
        for consumer in consumers[scale.output[0]]:
            consumer.input[:] = [unscaled if name == scale.output[0] else name for name in consumer.input]
    removed = {node.name for node in [*matmuls, *adds]} | {scale.name for scale, _, _ in scales}
    rebuilt = []
    for node in model.graph.node:
        if node.name == matmuls[0].name:
            rebuilt.extend(fused)
        if node.name not in removed:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)


def _prune_initializers(model: onnx.ModelProto) -> None:
    """Drop the separate projection weights and scale constants the fused blocks no longer read."""
    used = {name for node in model.graph.node for name in node.input}
    kept = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def prepare_model(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _require([value.name for value in model.graph.input] == [INPUT], f"inputs {_shapes(model.graph.input)}")
    _require([value.name for value in model.graph.output] == [OUTPUT], f"outputs {_shapes(model.graph.output)}")

    # No onnxsim fold first: the attention blocks hold no shape arithmetic, and the Model Compiler's onnxsim 0.4 folds
    # only with its optimizer on, which also rewrites the projections matched here. The compiler takes the export's
    # remaining batch-size arithmetic as it is.
    for layer in range(LAYERS):
        _fuse_attention_projections(model, layer)
    _prune_initializers(model)

    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    path = output_dir / "vit_b_16_fused_qkv.onnx"
    onnx.save(model, path)
    return path
