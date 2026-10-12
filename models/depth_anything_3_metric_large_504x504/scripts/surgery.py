"""Rewrite the folded Depth Anything 3 Metric Large export into MLA-friendly data movement.

The rewrites change only how values move, never what they are:

- Each attention block reshapes its fused QKV projection to 5-D, transposes it, and gathers Q, K and V
  on axis 0. A channel Split and 4-D Reshape/Transpose per part select the same values.
- Reshapes and Squeezes that keep a tensor's shape, Casts to the same type, and reshape or transpose
  chains collapse into one operation.
- Each DPT tap reshapes a block output to [1, 1, T, C], applies LayerNorm, drops the class token, and
  reshapes to [1, T - 1, C]. Dropping the class token first and normalizing the 3-D tokens gives the
  same values, because LayerNorm normalizes each token on its own. Otherwise the block output feeds a
  4-D and a 3-D LayerNorm, and the compiler leaves one of them as a channel variance that the MLA
  cannot run.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT = "image"
INPUT_SHAPE = [1, 3, 504, 504]
OUTPUT_SHAPES = {"depth": [1, 1, 504, 504], "sky": [1, 1, 504, 504]}
OUTPUTS = list(OUTPUT_SHAPES)
BLOCKS = 24
TAPS = 4


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported Depth Anything 3 graph: {message}")


def _shape(value) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _index(model):
    shapes = {value.name: _shape(value) for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    shapes.update({name: list(value.shape) for name, value in constants.items()})
    producers, users = {}, defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            if name:
                users[name].append(node)
    # A graph output is used outside the graph and must keep its producer.
    for value in model.graph.output:
        users[value.name].append(None)
    return shapes, constants, producers, users


def _validate_interface(model):
    inputs = {value.name: _shape(value) for value in model.graph.input}
    outputs = {value.name: _shape(value) for value in model.graph.output}
    _require(inputs == {INPUT: INPUT_SHAPE} and outputs == OUTPUT_SHAPES, f"interface {inputs} -> {outputs}")


def _prune(model):
    """Drop nodes and initializers that no longer reach a graph output."""
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [value for value in model.graph.initializer if value.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def _finish(model):
    _prune(model)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def _constant(model, name, value):
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
    return name


def _replace_nodes(model, added, removed):
    """Insert each replacement where the node it is keyed by stood, and drop the removed nodes."""
    nodes = []
    for node in model.graph.node:
        nodes.extend(added.get(node.name, []))
        if node.name not in removed:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def _split_qkv(model, expected):
    """Reshape [B, N, 3, h, d] -> Transpose(2, 0, 3, 1, 4) -> Gather 0/1/2 becomes Split + 4-D moves."""
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes, constants, producers, users = _index(model)
    added, removed = {}, set()
    for transpose in model.graph.node:
        if transpose.op_type != "Transpose" or _attributes(transpose).get("perm") != [2, 0, 3, 1, 4]:
            continue
        reshape = producers.get(transpose.input[0])
        _require(reshape is not None and reshape.op_type == "Reshape", f"QKV reshape before {transpose.name}")
        _require(users[reshape.output[0]] == [transpose], f"private QKV reshape {reshape.name}")
        batch, tokens, parts, heads, width = constants[reshape.input[1]].tolist()
        source = reshape.input[0]
        _require(parts == 3 and shapes[source] == [batch, tokens, 3 * heads * width], f"QKV packing at {reshape.name}")
        gathers = {}
        for gather in users[transpose.output[0]]:
            _require(
                gather is not None and gather.op_type == "Gather" and _attributes(gather).get("axis", 0) == 0,
                f"axis-0 Gather after {transpose.name}",
            )
            index = constants.get(gather.input[1])
            _require(index is not None and index.shape == (), f"scalar Gather index at {gather.name}")
            gathers[int(index)] = gather
        _require(sorted(gathers) == [0, 1, 2], f"Q, K and V Gathers after {transpose.name}")
        prefix = transpose.name.rsplit("/", 1)[0] + "/qkv_split"
        names = [f"{prefix}/{part}" for part in "qkv"]
        sizes = _constant(model, f"{prefix}/sizes", np.array([heads * width] * 3, np.int64))
        head_shape = _constant(model, f"{prefix}/head_shape", np.array([batch, tokens, heads, width], np.int64))
        nodes = [helper.make_node("Split", [source, sizes], names, name=f"{prefix}/Split", axis=2)]
        for index, name in enumerate(names):
            nodes.append(helper.make_node("Reshape", [name, head_shape], [f"{name}/heads"], name=f"{name}/Reshape"))
            nodes.append(
                helper.make_node(
                    "Transpose",
                    [f"{name}/heads"],
                    [gathers[index].output[0]],
                    name=f"{name}/Transpose",
                    perm=[0, 2, 1, 3],
                )
            )
        added[reshape.name] = nodes
        removed.update({reshape.name, transpose.name, *(gather.name for gather in gathers.values())})
    _require(len(added) == expected, f"expected {expected} QKV blocks, found {len(added)}")
    _replace_nodes(model, added, removed)
    return _finish(model)


def _bypass(model, node, producers, users, outputs) -> bool:
    """Remove a node whose output equals its first input; a graph output keeps its name."""
    source, result = node.input[0], node.output[0]
    if result in outputs:
        producer = producers.get(source)
        if producer is None or len(users[source]) != 1:
            return False
        producer.output[list(producer.output).index(source)] = result
    else:
        for user in users[result]:
            for index, name in enumerate(user.input):
                if name == result:
                    user.input[index] = source
    model.graph.node.remove(node)
    return True


def _collapse_data_movement(model):
    """Collapse reshape and transpose chains, then remove identity Reshapes, Squeezes and Casts."""
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph
    values = (*graph.input, *graph.value_info, *graph.output)
    # No rewrite here changes what a value name holds, so one shape inference serves them all.
    shapes = {value.name: _shape(value) for value in values}
    floats = {value.name for value in values if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT}
    producers = {}
    for node in graph.node:
        producer = producers.get(node.input[0]) if node.input else None
        if (
            node.op_type == "Reshape"
            and producer is not None
            and producer.op_type in ("Reshape", "Squeeze", "Unsqueeze")
        ):
            # A reshape reads only the row-major element order, which the previous reshape kept.
            node.input[0] = producer.input[0]
        elif node.op_type == "Transpose" and producer is not None and producer.op_type == "Transpose":
            first, second = _attributes(producer)["perm"], _attributes(node)["perm"]
            node.input[0] = producer.input[0]
            permutation = next(item for item in node.attribute if item.name == "perm")
            del permutation.ints[:]
            permutation.ints.extend(first[axis] for axis in second)
        for name in node.output:
            producers[name] = node
    _prune(model)
    outputs = {value.name for value in graph.output}
    while True:
        _, _, producers, users = _index(model)
        for node in graph.node:
            identity = (
                node.op_type == "Cast" and _attributes(node)["to"] == onnx.TensorProto.FLOAT and node.input[0] in floats
            )
            identity |= (
                node.op_type in ("Reshape", "Squeeze", "Unsqueeze")
                and shapes.get(node.input[0]) is not None
                and shapes.get(node.input[0]) == shapes.get(node.output[0])
            )
            if identity and _bypass(model, node, producers, users, outputs):
                break
        else:
            return _finish(model)


def _slice_before_tap_norms(model, expected):
    """Reshape [1, T, C] -> [1, 1, T, C] -> LayerNorm -> Slice tokens 1: -> Reshape [1, T - 1, C]
    becomes Slice tokens 1: -> LayerNorm."""
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes, constants, producers, users = _index(model)
    added, removed = {}, set()
    for norm in model.graph.node:
        if norm.op_type != "LayerNormalization" or len(shapes[norm.input[0]]) != 4:
            continue
        expand = producers.get(norm.input[0])
        _require(expand is not None and expand.op_type == "Reshape", f"tap reshape before {norm.name}")
        source = expand.input[0]
        _require(len(shapes[source]) == 3, f"3-D token tensor before {expand.name}")
        batch, tokens, channels = shapes[source]
        _require(
            shapes[norm.input[0]] == [batch, 1, tokens, channels] and users[expand.output[0]] == [norm],
            f"private [1, 1, T, C] tap reshape before {norm.name}",
        )
        _require(_attributes(norm).get("axis") == -1, f"channel LayerNorm at {norm.name}")
        cut = users[norm.output[0]][0] if len(users[norm.output[0]]) == 1 else None
        _require(
            cut is not None and cut.op_type == "Slice" and len(cut.input) == 5, f"private token Slice after {norm.name}"
        )
        starts, ends, axes, steps = (constants.get(name) for name in cut.input[1:])
        _require(all(value is not None for value in (starts, ends, axes, steps)), f"constant Slice at {cut.name}")
        _require(
            starts.tolist() == [1] and axes.tolist() == [2] and steps.tolist() == [1] and ends.tolist()[0] >= tokens,
            f"class-token Slice at {cut.name}",
        )
        flatten = users[cut.output[0]][0] if len(users[cut.output[0]]) == 1 else None
        _require(
            flatten is not None
            and flatten.op_type == "Reshape"
            and shapes[flatten.output[0]] == [batch, tokens - 1, channels],
            f"3-D tap reshape after {cut.name}",
        )
        prefix = norm.name.rsplit("/", 1)[0] + "/tokens"
        bounds = [
            _constant(model, f"{prefix}/{name}", np.array([value], np.int64))
            for name, value in (("starts", 1), ("ends", tokens), ("axes", 1), ("steps", 1))
        ]
        added[expand.name] = [
            helper.make_node("Slice", [source, *bounds], [f"{prefix}/patches"], name=f"{prefix}/Slice"),
            helper.make_node(
                "LayerNormalization",
                [f"{prefix}/patches", *norm.input[1:]],
                [flatten.output[0]],
                name=norm.name,
                **_attributes(norm),
            ),
        ]
        removed.update({expand.name, norm.name, cut.name, flatten.name})
    _require(len(added) == expected, f"expected {expected} DPT taps, found {len(added)}")
    _replace_nodes(model, added, removed)
    return _finish(model)


def prepare_model(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _validate_interface(model)
    # convert.py folds the export with onnxsim; the matching relies on constant shapes and indices.
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Constant", "Shape"})
    _require(not unfolded, f"source still has {unfolded} nodes; fold the source with onnxsim first")
    # The rewrites replace and remove nodes by name.
    names = [node.name for node in model.graph.node]
    _require(all(names) and len(set(names)) == len(names), "every node needs a unique name")
    model = _split_qkv(model, BLOCKS)
    model = _collapse_data_movement(model)
    model = _slice_before_tap_norms(model, TAPS)
    _validate_interface(model)
    left = sorted({node.op_type for node in model.graph.node} & {"Gather", "Cast", "Squeeze", "Unsqueeze"})
    _require(not left, f"leftover data movement {left}")
    shapes, _, _, _ = _index(model)
    _require(all(len(shape) <= 4 for shape in shapes.values()), "every tensor must be at most 4-D")
    _require(
        all(len(shapes[node.input[0]]) == 3 for node in model.graph.node if node.op_type == "LayerNormalization"),
        "every LayerNorm must normalize a 3-D token tensor",
    )
    path = output_dir / "depth_anything_3_metric_large.onnx"
    onnx.save(model, path)
    return path
