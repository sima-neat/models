"""Adapt the unmodified DA3-Small ONNX to MLA and check both outputs against the source.

QKV selection becomes fixed splits, camera-token assignment becomes concatenation,
and the final learned projection is split into separate depth/confidence rows.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from model_registry.dependencies import require_onnx_opset_version
from onnx import helper, numpy_helper
from preprocess import HEIGHT, WIDTH

INPUT_SHAPES = {"image": [1, 3, HEIGHT, WIDTH]}
OUTPUT_SHAPES = {"depth": [1, 1, HEIGHT, WIDTH], "depth_conf": [1, 1, HEIGHT, WIDTH]}
INPUTS, OUTPUTS = tuple(INPUT_SHAPES), tuple(OUTPUT_SHAPES)
BLOCKS = 12


def prepare_model(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort

    model = onnx.load(source)
    onnx.checker.check_model(model)
    _validate_interface(model)
    _require(not ({node.op_type for node in model.graph.node} & {"Constant", "Shape"}), "constant-folded source")
    names = [node.name for node in model.graph.node]
    _require(all(names) and len(set(names)) == len(names), "unique node names")
    # 1. Express the fixed camera-token overwrite without ScatterND.
    model = replace_camera_token(model)
    # 2. Keep attention's Q/K/V tensors within MLA's four-dimensional limit.
    model = split_attention_qkv(model)
    # 3. Produce depth/confidence directly, without layout moves or selector convolutions.
    model = split_output_projection(model)
    # 4. Remove identity and consecutive layout operations left by export and the rewrites.
    model = remove_redundant_layout_changes(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "depth_anything_3_small.onnx"
    onnx.save(model, path)
    validate_export(path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    reference = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    prepared = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for sample in sample_inputs():
        inputs = {"image": sample}
        compare_outputs(reference.run(list(OUTPUTS), inputs), prepared.run(list(OUTPUTS), inputs))
    return path


def replace_camera_token(model):
    """Replace the one-view camera-token ScatterND with Slice + Concat.

    The first token is overwritten by a fixed learned token; every remaining
    patch token is retained. Concatenation expresses this without ScatterND.
    """
    shapes, constants, _, _ = _index(model)
    added, removed = {}, set()
    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue
        source, indices, update = node.input
        shape = shapes[source]
        _require(len(shape) == 4 and shape[:2] == [1, 1], "one-view camera-token tensor")
        _require(np.array_equal(constants[indices], np.zeros((1, 1, 1, 3), np.int64)), "camera token at index zero")
        _require(constants[update].shape == (1, 1, 1, shape[3]), "one learned camera token")
        _require(_attributes(node).get("reduction", b"none") == b"none", "camera-token replacement")
        bounds = [
            _constant(model, f"{node.name}/{label}", np.array([value], np.int64))
            for label, value in (("start", 1), ("end", shape[2]), ("axis", 2))
        ]
        tail = f"{node.name}/patch_tokens"
        added[node.name] = [
            helper.make_node("Slice", [source, *bounds], [tail], name=f"{node.name}/Slice"),
            helper.make_node("Concat", [update, tail], list(node.output), name=f"{node.name}/Concat", axis=2),
        ]
        removed.add(node.name)
    _require(len(added) == 1, "one camera-token assignment")
    _replace_nodes(model, added, removed)
    return _finish(model)


def split_attention_qkv(model):
    """Split Q/K/V before reshaping, so attention never creates a five-dimensional tensor.

    Upstream packs [B, N, 3, heads, width], transposes, then gathers Q/K/V.
    Splitting the last channel axis first gives the same three [B, heads, N, width]
    tensors using only four-dimensional reshapes and transposes.
    """
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
            _require(int(index) in (0, 1, 2) and int(index) not in gathers, "distinct Q/K/V indices")
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
    _require(len(added) == BLOCKS, f"expected {BLOCKS} QKV blocks, found {len(added)}")
    _replace_nodes(model, added, removed)
    return _finish(model)


def split_output_projection(model):
    """Replace the two-channel head and its layout/selection tail with two learned projections.

    Upstream: Conv(32 -> 2) -> NHWC -> select depth/confidence -> exp / (exp + 1).
    Prepared: Conv(32 -> 1) -> exp, and Conv(32 -> 1) -> exp + 1, both in NCHW.
    Copying the original weight rows and biases preserves both outputs and avoids
    the extra quantized selector convolutions that AFE otherwise creates.
    """
    shapes, constants, producers, users = _index(model)

    def producer(tensor, kind):
        node = producers.get(tensor)
        _require(node is not None and node.op_type == kind, f"expected {kind} producing {tensor}")
        return node

    # Check the original output paths before replacing any computation.
    depth_shape = producer("depth", "Reshape")
    depth_squeeze = producer(depth_shape.input[0], "Squeeze")
    depth_exp = producer(depth_squeeze.input[0], "Exp")
    depth_slice = producer(depth_exp.input[0], "Slice")
    layout = producer(depth_slice.input[0], "Transpose")
    projection = producer(layout.input[0], "Conv")
    confidence_shape = producer("depth_conf", "Reshape")
    confidence_add = producer(confidence_shape.input[0], "Add")
    confidence_exp = producer(confidence_add.input[0], "Exp")
    confidence_gather = producer(confidence_exp.input[0], "Gather")
    _require(confidence_gather.input[0] == layout.output[0], "both outputs select from the same head")

    _require(_attributes(layout).get("perm") == [0, 2, 3, 1], "NCHW to NHWC head layout")
    _require(
        [constants[name].tolist() for name in depth_slice.input[1:]] == [[0], [-1], [3], [1]],
        "depth selects NHWC channel zero",
    )
    _require(constants[depth_squeeze.input[1]].tolist() == [3], "depth squeezes only the channel axis")
    _require(
        _attributes(confidence_gather).get("axis") == 3 and constants[confidence_gather.input[1]].tolist() == -1,
        "confidence selects the last NHWC channel",
    )
    _require(constants[confidence_add.input[1]].item() == 1, "confidence is exp(logit) + 1")
    _require(
        all(constants[node.input[1]].tolist() == [1, 1, HEIGHT, WIDTH] for node in (depth_shape, confidence_shape)),
        "both outputs retain the full-resolution single-view shape",
    )
    _require(len(projection.input) == 3, "final projection has a bias")
    weight, bias = (constants[name] for name in projection.input[1:])
    attributes = _attributes(projection)
    _require(
        weight.shape == (2, 32, 1, 1)
        and bias.shape == (2,)
        and weight.dtype == bias.dtype == np.float32
        and attributes.get("group", 1) == 1
        and shapes[projection.output[0]] == [1, 2, HEIGHT, WIDTH],
        "full-resolution learned float32 32-to-2 projection",
    )
    tail = [
        projection,
        layout,
        depth_slice,
        depth_exp,
        depth_squeeze,
        depth_shape,
        confidence_gather,
        confidence_exp,
        confidence_add,
        confidence_shape,
    ]
    removed = {node.name for node in tail}
    _require(
        all(user is None or user.name in removed for node in tail for user in users[node.output[0]]),
        "the output tail has no other consumers",
    )

    # Each convolution uses one original learned row. No new weights or calibration are introduced.
    nodes = []
    for index, output in enumerate(OUTPUTS):
        prefix = f"{projection.name}/{output}"
        weights = _constant(model, f"{prefix}/weight", weight[index : index + 1].copy())
        biases = _constant(model, f"{prefix}/bias", bias[index : index + 1].copy())
        logit, value = f"{prefix}/logit", f"{prefix}/exp"
        nodes.append(
            helper.make_node(
                "Conv", [projection.input[0], weights, biases], [logit], name=f"{prefix}/Conv", **attributes
            )
        )
        nodes.append(helper.make_node("Exp", [logit], [output if index == 0 else value], name=f"{prefix}/Exp"))
        if index == 1:
            nodes.append(helper.make_node("Add", [value, confidence_add.input[1]], [output], name=f"{prefix}/Add"))
    _replace_nodes(model, {projection.name: nodes}, removed)
    return _finish(model)


def remove_redundant_layout_changes(model):
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


def validate_export(path):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    _validate_interface(model)
    require_onnx_opset_version(model)
    values = [*model.graph.input, *model.graph.output]
    _require(all(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for value in values), "float32 interface")
    forbidden = {"Constant", "Shape", "Range", "ScatterND", "Einsum", "Sin", "Cos", "Gather"}
    _require(not ({node.op_type for node in model.graph.node} & forbidden), "static graph without dynamic selection")
    shapes = [_shape(value) for value in (*model.graph.value_info, *model.graph.output)]
    _require(
        all(0 < len(shape) <= 4 and shape[0] == 1 and all(d > 0 for d in shape) for shape in shapes),
        "static batch-one tensors of at most four dimensions",
    )


def compare_outputs(expected, actual):
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        _require(list(reference.shape) == list(candidate.shape) == OUTPUT_SHAPES[name], f"shape of {name}")
        _require(np.isfinite(reference).all() and np.isfinite(candidate).all(), f"finite {name}")
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3, err_msg=name)


def sample_inputs():
    """Three normalized noise images and one smooth RGB gradient."""
    from preprocess import MEAN, STD

    samples = [
        np.random.RandomState(seed).uniform(0, 1, (HEIGHT, WIDTH, 3)).astype(np.float32) for seed in range(123, 126)
    ]
    y, x = np.meshgrid(
        np.linspace(0, 1, HEIGHT, dtype=np.float32), np.linspace(0, 1, WIDTH, dtype=np.float32), indexing="ij"
    )
    samples.append(np.stack([x, y, (x + y) / 2], axis=-1))
    return [np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1)[None]) for image in samples]


def _validate_interface(model):
    inputs = {value.name: _shape(value) for value in model.graph.input}
    outputs = {value.name: _shape(value) for value in model.graph.output}
    _require(inputs == INPUT_SHAPES and outputs == OUTPUT_SHAPES, f"interface {inputs} -> {outputs}")


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
