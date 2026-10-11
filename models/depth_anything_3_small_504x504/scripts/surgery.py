"""Rewrite the folded two-view Depth Anything 3 Small export into MLA-friendly data movement.

The export runs the two views as a batch of two through the patch embedding, the per-view transformer
blocks, the DualDPT head and the camera decoder, and as one [1, 2 x 1297, 384] sequence through the
global blocks. The MLA expects batch one everywhere. The rewrites change only how values move, never
what they are:

- Each attention block reshapes its fused QKV projection to 5-D, transposes it, and gathers Q, K and V
  on axis 0. A channel Split and 4-D Reshape/Transpose per part select the same values.
- Every operation on the view batch is cloned once per view with batch one. The clones share weights
  and take their own view's slice of view-indexed constants, such as the camera tokens and RoPE
  tables. Where the export stacked the views into [1, 2, ...], the per-view results are concatenated
  on axis 1 instead, so the global blocks and the public outputs keep their shapes.
- The head reads the depth and confidence channels from its NCHW output instead of through an NHWC
  transpose.
- Reshapes and Squeezes that keep a tensor's shape, Casts to the same type, and reshape or transpose
  chains collapse into one operation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT_SHAPES = {"image0": [1, 3, 504, 504], "image1": [1, 3, 504, 504]}
OUTPUT_SHAPES = {"depth": [1, 2, 504, 504], "depth_conf": [1, 2, 504, 504], "pose_enc": [1, 2, 9]}
INPUTS = list(INPUT_SHAPES)
OUTPUTS = list(OUTPUT_SHAPES)
VIEWS = 2
BLOCKS = 12
# Operations that compute each view on its own, given the axis checks in _require_per_view.
VIEW_OPS = {
    "Add",
    "Cast",
    "Concat",
    "Conv",
    "ConvTranspose",
    "Div",
    "Erf",
    "Exp",
    "Gather",
    "Gemm",
    "LayerNormalization",
    "MatMul",
    "Mul",
    "Neg",
    "Relu",
    "Reshape",
    "Resize",
    "Slice",
    "Softmax",
    "Split",
    "Squeeze",
    "Transpose",
}
# Weights of these operations may have VIEWS rows by coincidence; they apply to each view whole.
WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "LayerNormalization", "MatMul"}


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
    _require(inputs == INPUT_SHAPES and outputs == OUTPUT_SHAPES, f"interface {inputs} -> {outputs}")


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


def _require_per_view(node, shapes, constants):
    """Require that an operation on the view batch never mixes or reorders the view axis."""
    attributes = _attributes(node)
    rank = len(shapes[node.input[0]])
    kind = node.op_type
    _require(kind in VIEW_OPS, f"unexpected view-batched {kind} {node.name}")
    if kind == "Transpose":
        _require(attributes["perm"][0] == 0, f"view axis moves at {node.name}")
    elif kind in ("Concat", "Gather", "Softmax", "LayerNormalization", "Split"):
        default = {"Concat": 0, "Gather": 0, "Softmax": -1, "LayerNormalization": -1, "Split": 0}[kind]
        _require(attributes.get("axis", default) % rank != 0, f"view-axis {kind} at {node.name}")
    elif kind in ("Slice", "Squeeze"):
        position = 3 if kind == "Slice" else 1
        axes = constants.get(node.input[position]) if len(node.input) > position else None
        _require(
            axes is not None and all(axis % rank != 0 for axis in axes.tolist()), f"view-axis {kind} at {node.name}"
        )
    elif kind == "Gemm":
        _require(rank == 2 and not attributes.get("transA", 0), f"Gemm rows must be views at {node.name}")
    elif kind == "Resize":
        sizes = constants.get(node.input[3]) if len(node.input) > 3 else None
        _require(sizes is not None and int(sizes[0]) == VIEWS, f"Resize must use output sizes at {node.name}")


def _unbatch_views(model):
    """Clone each operation on the view batch once per view with batch one."""
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes, constants, producers, users = _index(model)
    outputs = {value.name for value in model.graph.output}

    def batched(name):
        return name in producers and len(shapes.get(name, [])) >= 2 and shapes[name][0] == VIEWS

    region = {node.name for node in model.graph.node if any(batched(name) for name in node.output)}
    # A [1, VIEWS, ...] value that concatenates per-view [1, 1, ...] parts, mapped to those parts.
    stacked = {}
    for node in model.graph.node:
        if node.name in region:
            _require(all(batched(name) for name in node.output), f"partly view-batched outputs at {node.name}")
        elif node.op_type == "Concat" and _attributes(node)["axis"] == 1 and len(node.input) == VIEWS:
            if all(shapes[name][:2] == [1, 1] for name in node.input):
                stacked[node.output[0]] = list(node.input)
    for name in {name for node in model.graph.node for name in node.output if batched(name)}:
        _require(name not in outputs, f"view-batched public output {name}")
        for user in users[name]:
            if user.name not in region:
                _require(
                    user.op_type == "Reshape" and shapes[user.output[0]] == [1, VIEWS, *shapes[name][1:]],
                    f"{user.name} leaves the view batch without stacking",
                )

    initializers, view_slices, input_slices, nodes = [], {}, {}, []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
        return name

    def view_constant(name, view):
        """One view's slice of a view-indexed constant; identical slices are shared."""
        if name not in view_slices:
            parts = [constants[name][index : index + 1] for index in range(VIEWS)]
            if all(np.array_equal(parts[0], part) for part in parts):
                view_slices[name] = [constant(f"{name}/views", parts[0])] * VIEWS
            else:
                view_slices[name] = [constant(f"{name}/view{index}", part) for index, part in enumerate(parts)]
        return view_slices[name][view]

    def view_input(name, view):
        """One view of a [1, VIEWS, ...] value: its concatenated part, or a Slice on axis 1."""
        if name in stacked:
            return stacked[name][view]
        if (name, view) not in input_slices:
            part = f"{name}/view{view}"
            bounds = [
                constant(f"{part}/{bound}", np.array([value], np.int64))
                for bound, value in (("starts", view), ("ends", view + 1), ("axes", 1))
            ]
            nodes.append(helper.make_node("Slice", [name, *bounds], [part], name=f"{part}/Slice"))
            input_slices[name, view] = part
        return input_slices[name, view]

    for node in model.graph.node:
        if node.name in region:
            sources = [name for name in node.input if name and name not in constants]
            if all(batched(name) for name in sources):
                _require_per_view(node, shapes, constants)
            else:
                _require(
                    node.op_type == "Reshape" and len(sources) == 1 and shapes[sources[0]][:2] == [1, VIEWS],
                    f"{node.name} enters the view batch from an unstacked value",
                )
            for view in range(VIEWS):
                clone = onnx.NodeProto()
                clone.CopyFrom(node)
                clone.name = f"{node.name}/view{view}"
                del clone.output[:]
                clone.output.extend(f"{name}/view{view}" for name in node.output)
                for position, name in enumerate(node.input):
                    if not name:
                        continue
                    if batched(name):
                        clone.input[position] = f"{name}/view{view}"
                    elif name not in constants:
                        clone.input[position] = view_input(name, view)
                    elif node.op_type == "Reshape" and position == 1:
                        target = [1, *shapes[node.output[0]][1:]]
                        clone.input[position] = constant(f"{clone.name}/shape", np.array(target, np.int64))
                    elif node.op_type == "Resize" and position == 3:
                        target = [1, *constants[name][1:]]
                        clone.input[position] = constant(f"{clone.name}/sizes", np.array(target, np.int64))
                    elif node.op_type in ("Add", "Mul", "Div", "Concat"):
                        value = constants[name]
                        rank = len(shapes[node.output[0]])
                        if value.ndim == rank and value.shape[0] == VIEWS:
                            clone.input[position] = view_constant(name, view)
                        else:
                            _require(value.ndim < rank or value.shape[0] == 1, f"constant {name} at {node.name}")
                    else:
                        value = constants[name]
                        _require(
                            value.ndim < 2 or value.shape[0] != VIEWS or node.op_type in WEIGHTED_OPS,
                            f"view-indexed constant {name} at {node.name}",
                        )
                nodes.append(clone)
        elif any(batched(name) for name in node.input):
            # A [VIEWS, ...] -> [1, VIEWS, ...] Reshape: concatenate the per-view results on axis 1 instead.
            source, rest = node.input[0], shapes[node.input[0]][1:]
            parts = []
            for view in range(VIEWS):
                part = f"{node.output[0]}/view{view}"
                shape = constant(f"{part}/shape", np.array([1, 1, *rest], np.int64))
                nodes.append(
                    helper.make_node("Reshape", [f"{source}/view{view}", shape], [part], name=f"{node.name}/view{view}")
                )
                parts.append(part)
            nodes.append(helper.make_node("Concat", parts, [node.output[0]], name=f"{node.name}/stack", axis=1))
            stacked[node.output[0]] = parts
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    model = _finish(model)
    left = [value.name for value in model.graph.value_info if _shape(value)[:1] == [VIEWS]]
    _require(not left, f"view-batched values remain: {left[:5]}")
    return model


def _select_head_channels(model):
    """Exp(Slice or Gather of an NHWC Transpose) becomes Exp(Slice) of the NCHW head output, per view."""
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes, constants, _, users = _index(model)

    def only_user(node, kind):
        following = users[node.output[0]]
        _require(
            len(following) == 1 and following[0] is not None and following[0].op_type == kind,
            f"expected private {node.name} -> {kind}",
        )
        return following[0]

    def channel(source, index, name):
        bounds = [
            _constant(model, f"{name}/{bound}", np.array([value], np.int64))
            for bound, value in (("starts", index), ("ends", index + 1), ("axes", 1))
        ]
        return helper.make_node("Slice", [source, *bounds], [name], name=f"{name}/Slice")

    added, removed = {}, set()
    for transpose in model.graph.node:
        if transpose.op_type != "Transpose" or _attributes(transpose)["perm"] != [0, 2, 3, 1]:
            continue
        source = transpose.input[0]
        # The head's two channels are depth and confidence.
        if shapes[source][:2] != [1, 2] or "/head/" not in transpose.name:
            continue
        readers = users[transpose.output[0]]
        kinds = {node.op_type: node for node in readers if node is not None}
        _require(
            len(readers) == 2 and sorted(kinds) == ["Gather", "Slice"],
            f"depth and confidence readers of {transpose.name}",
        )
        depth, confidence = kinds["Slice"], kinds["Gather"]
        steps = [1] if len(depth.input) < 5 else constants[depth.input[4]].tolist()
        _require(
            [constants[name].tolist() for name in depth.input[1:4]] == [[0], [-1], [3]] and steps == [1],
            f"depth channel at {depth.name}",
        )
        _require(
            _attributes(confidence).get("axis") == 3 and constants[confidence.input[1]].tolist() == -1,
            f"confidence channel at {confidence.name}",
        )
        depth_exp = only_user(depth, "Exp")
        squeeze = only_user(depth_exp, "Squeeze")
        _require(constants[squeeze.input[1]].tolist() == [3], f"channel Squeeze at {squeeze.name}")
        confidence_exp = only_user(confidence, "Exp")
        for user in users[squeeze.output[0]]:
            _require(
                user is not None
                and user.op_type == "Reshape"
                and shapes[user.output[0]] == [1, 1, *shapes[source][2:]],
                f"depth stacking after {squeeze.name}",
            )
            user.input[0] = depth_exp.output[0]
        depth_name, confidence_name = f"{depth.output[0]}/nchw", f"{confidence.output[0]}/nchw"
        added[transpose.name] = [channel(source, 0, depth_name), channel(source, 1, confidence_name)]
        depth_exp.input[0], confidence_exp.input[0] = depth_name, confidence_name
        removed.update({transpose.name, depth.name, confidence.name, squeeze.name})
    _require(len(added) == VIEWS, f"expected {VIEWS} per-view head outputs, found {len(added)}")
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
    model = _unbatch_views(model)
    model = _select_head_channels(model)
    model = _collapse_data_movement(model)
    _validate_interface(model)
    left = sorted({node.op_type for node in model.graph.node} & {"Cast", "Squeeze", "Unsqueeze"})
    _require(not left, f"leftover data movement {left}")
    tensors = [_shape(value) for value in (*model.graph.value_info, *model.graph.output)]
    _require(
        all(len(shape) <= 4 and shape[:1] == [1] for shape in tensors), "every tensor must be batch one and at most 4-D"
    )
    path = output_dir / "depth_anything_3_small.onnx"
    onnx.save(model, path)
    return path
