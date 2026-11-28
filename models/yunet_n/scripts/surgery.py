"""Expose YuNet's 12 raw head maps: cut each output before upstream's flattening Transpose and Reshape.

Upstream ends every head with Conv -> Transpose(0, 2, 3, 1) -> Reshape([1, -1, C]), and the cls and obj heads with a
Sigmoid after that. The cut graph ends each head at its Conv (bbox, kps) or at a Sigmoid of it (cls, obj), in NCHW.
The MLA returns these maps channels-last, so reshaping a returned map to [1, H*W, C] gives back the upstream output
exactly: the Sigmoid is elementwise, and the Transpose and Reshape only move values.
"""

from pathlib import Path

import onnx
from calibration import HEADS, IMAGE_SIZE, OUTPUTS, STRIDES
from onnx import helper

INPUT = "input"
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
SOURCE_OUTPUT_SHAPES = {
    f"{head}_{stride}": [1, (IMAGE_SIZE // stride) ** 2, channels]
    for head, channels in HEADS.items()
    for stride in STRIDES
}
OUTPUT_SHAPES = {
    f"{head}_{stride}": [1, channels, IMAGE_SIZE // stride, IMAGE_SIZE // stride]
    for head, channels in HEADS.items()
    for stride in STRIDES
}


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported YuNet graph: {message}")


def _shapes(values):
    return {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in values}


def _tail(producers, name):
    """The nodes from an output back to its head Conv, output first, and that Conv."""
    chain, node = [], producers[name]
    while node.op_type != "Conv":
        chain.append(node)
        node = producers[node.input[0]]
    return chain, node


def prepare_model(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {value.name for value in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializers]
    _require(_shapes(inputs) == {INPUT: INPUT_SHAPE}, f"inputs {_shapes(inputs)}")
    _require(_shapes(graph.output) == SOURCE_OUTPUT_SHAPES, f"outputs {_shapes(graph.output)}")
    producers = {name: node for node in graph.node for name in node.output}
    removed, sigmoids = set(), []
    for name in OUTPUTS:
        chain, conv = _tail(producers, name)
        scored = name.split("_")[0] in ("cls", "obj")
        expected = ["Sigmoid", "Reshape", "Transpose"] if scored else ["Reshape", "Transpose"]
        _require([node.op_type for node in chain] == expected, f"{name} ends with {[n.op_type for n in chain]}")
        transpose = chain[-1]
        perm = [attribute.ints for attribute in transpose.attribute if attribute.name == "perm"]
        _require(perm == [[0, 2, 3, 1]], f"{name}'s Transpose permutes {perm}")
        consumers = [node for node in graph.node if conv.output[0] in node.input]
        _require(consumers == [transpose], f"{name}'s head Conv feeds more than its Transpose")
        removed.update(id(node) for node in chain)
        if scored:
            sigmoids.append(helper.make_node("Sigmoid", [conv.output[0]], [name], name=f"{name}_sigmoid"))
        else:
            conv.output[0] = name
    kept = [node for node in graph.node if id(node) not in removed]
    del graph.node[:]
    graph.node.extend(kept + sigmoids)
    del graph.output[:]
    graph.output.extend(
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in OUTPUT_SHAPES.items()
    )
    used = {name for node in graph.node for name in node.input}
    kept_initializers = [value for value in graph.initializer if value.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    _require(_shapes(model.graph.output) == OUTPUT_SHAPES, "unexpected raw-head shapes")
    _require(
        {node.op_type for node in model.graph.node} <= {"Conv", "Relu", "MaxPool", "Resize", "Add", "Sigmoid"},
        "operators outside the cut backbone and heads remain",
    )
    output = output_dir / "yunet_heads.onnx"
    onnx.save(model, output)
    return output
