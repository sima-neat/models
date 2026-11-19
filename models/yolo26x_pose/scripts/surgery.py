"""Cut YOLO26x-pose at its raw one-to-one Pose heads, before the in-graph box and keypoint decode and TopK.

Each attention block then becomes an equivalent form the MLA runs without moving data between layouts: Q, K and V
are [1, D, H, T] (D channels of each of H heads at T positions), the scores an einsum over D, and the softmax runs
over the key positions held as channels. The exporter packs each head's channels together; the block needs each
channel's heads together. That fixed reorder is folded into the weights of the convolutions around the block.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

IMAGE_SIZE = 640
CLASS_COUNT = 1
KEYPOINT_CHANNELS = 17 * 3
QUERY_COUNT = 300
STRIDES = (8, 16, 32)
# Two in the C2PSA stage of the backbone and one in the last C3k2 stage of the neck.
ATTENTION_BLOCKS = 3

INPUT = "images"
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
SOURCE_OUTPUT_SHAPES = {"output0": [1, QUERY_COUNT, 6 + KEYPOINT_CHANNELS]}
# Box side distances (cv2), person logits (cv3), then keypoints (cv4_kpts), each from P3 to P5. Every head tensor
# is the output of the last convolution in its one-to-one Pose branch.
BRANCHES = {
    "bbox": ("one2one_cv2.{level}/one2one_cv2.{level}.2", 4),
    "class_logit": ("one2one_cv3.{level}/one2one_cv3.{level}.2", CLASS_COUNT),
    "keypoint": ("one2one_cv4_kpts.{level}", KEYPOINT_CHANNELS),
}
HEADS = {
    f"{name}_{level}": f"/model.23/{branch.format(level=level)}/Conv_output_0"
    for name, (branch, _) in BRANCHES.items()
    for level in range(len(STRIDES))
}
OUTPUT_SHAPES = {
    f"{name}_{level}": [1, channels, IMAGE_SIZE // stride, IMAGE_SIZE // stride]
    for name, (_, channels) in BRANCHES.items()
    for level, stride in enumerate(STRIDES)
}
OUTPUTS = list(OUTPUT_SHAPES)
# Softmax reduces over one axis from opset 13 on; before that it flattens every axis after the one it names.
OPSET = 13
VALIDATION_SEED = 123
# The einsums sum in another order than the exporter's MatMuls, which moves the outputs by float32 rounding.
REWRITE_TOLERANCE = 1e-3


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported YOLO26x-pose graph: {message}")


def _shapes(values) -> dict[str, list[int]]:
    return {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim] for value in values}


def _rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for values in (node.input, node.output):
            for index, value in enumerate(values):
                if value == old:
                    values[index] = new


def _prune(model: onnx.ModelProto) -> None:
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


def _cut_at_heads(model: onnx.ModelProto) -> None:
    """End the graph at the raw heads and publish them under the package's output names."""
    producers = {name: node for node in model.graph.node for name in node.output}
    for name, head in HEADS.items():
        _require(head in producers and producers[head].op_type == "Conv", f"no Conv produces {head}")
        _rename_value(model, head, name)
    del model.graph.output[:]
    model.graph.output.extend(
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in OUTPUT_SHAPES.items()
    )
    _prune(model)
    # convert.py folds the export with onnxsim; the kept graph must hold no static-value computation.
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Constant", "Shape"})
    _require(not unfolded, f"heads still depend on {unfolded} nodes; fold the source with onnxsim first")


def _raise_opset(model: onnx.ModelProto) -> None:
    """Move the graph from the exporter's opset to OPSET, where a Split's sizes are an input instead of an attribute."""
    for node in model.graph.node:
        if node.op_type != "Split":
            continue
        (sizes,) = [attribute for attribute in node.attribute if attribute.name == "split"]
        node.attribute.remove(sizes)
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(sizes.ints, dtype=np.int64), f"{node.name}/sizes")
        )
        node.input.append(f"{node.name}/sizes")
    (opset,) = [entry for entry in model.opset_import if not entry.domain]
    opset.version = OPSET


@dataclass
class GraphIndex:
    """Lookups over one ONNX graph, built once per rewrite."""

    shapes: dict[str, list[int]]
    constants: dict[str, np.ndarray]
    producers: dict[str, onnx.NodeProto]
    readers: defaultdict[str, list[onnx.NodeProto]]


def _index(model: onnx.ModelProto) -> GraphIndex:
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = _shapes((*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output))
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    producers, readers = {}, defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            readers[name].append(node)
    return GraphIndex(shapes=shapes, constants=constants, producers=producers, readers=readers)


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def _producer(graph: GraphIndex, value: str, kind: str) -> onnx.NodeProto:
    node = graph.producers.get(value)
    _require(node is not None and node.op_type == kind, f"expected {kind} producing {value}")
    return node


def _sole_reader(graph: GraphIndex, value: str, kind: str) -> onnx.NodeProto:
    readers = graph.readers[value]
    _require(len(readers) == 1 and readers[0].op_type == kind, f"expected one {kind} reading {value}")
    return readers[0]


def _reorder_channels(
    model: onnx.ModelProto, graph: GraphIndex, conv: onnx.NodeProto, order: np.ndarray, axis: int
) -> None:
    """Give a convolution weights whose output (axis 0) or input (axis 1) channels follow `order`."""
    _require(len(conv.input) == 3, f"{conv.name} needs a weight and a bias")
    # A depthwise convolution maps channel to channel, so reordering its outputs also reorders the inputs it reads.
    reordered = {1: np.take(graph.constants[conv.input[1]], order, axis=axis)}
    if axis == 0:
        reordered[2] = graph.constants[conv.input[2]][order]
    for index, value in reordered.items():
        name = f"{conv.name}/reordered_{'weight' if index == 1 else 'bias'}"
        model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
        conv.input[index] = name


def _keep_attention_in_nchw(model: onnx.ModelProto, expected: int) -> None:
    """Rewrite every attention block to einsums on [1, D, H, T] with the softmax over the key positions as channels."""
    graph = _index(model)
    replacements, removed = {}, set()
    for softmax in [node for node in model.graph.node if node.op_type == "Softmax"]:
        # The exporter's block: the qkv convolution, a Reshape to [1, H, 2 * key + value, T], a Split into Q, K and V,
        # softmax(Q^T K * scale) over the keys, V times those weights, and a Reshape back to an image.
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), f"last-axis softmax required at {softmax.name}")
        scale = _producer(graph, softmax.input[0], "Mul")
        scores, factor = scale.input
        _require(graph.constants.get(factor, np.empty(0)).size == 1, f"scalar score scale required at {scale.name}")
        query_key = _producer(graph, scores, "MatMul")
        query_rows = _producer(graph, query_key.input[0], "Transpose")
        weight_columns = _sole_reader(graph, softmax.output[0], "Transpose")
        weighted_values = _sole_reader(graph, weight_columns.output[0], "MatMul")
        split = _producer(graph, query_rows.input[0], "Split")
        heads_apart = _producer(graph, split.input[0], "Reshape")
        qkv = _producer(graph, heads_apart.input[0], "Conv")
        image = _sole_reader(graph, weighted_values.output[0], "Reshape")
        q, k, v = split.output
        value_image = next((node for node in graph.readers[v] if node.op_type == "Reshape"), None)
        _require(
            _sole_reader(graph, scores, "Mul") is scale
            and _sole_reader(graph, scale.output[0], "Softmax") is softmax
            and _sole_reader(graph, q, "Transpose") is query_rows
            and _sole_reader(graph, query_rows.output[0], "MatMul") is query_key
            and _sole_reader(graph, k, "MatMul") is query_key
            and list(weighted_values.input) == [v, weight_columns.output[0]]
            and value_image is not None
            and len(graph.readers[v]) == 2
            and _sole_reader(graph, qkv.output[0], "Reshape") is heads_apart
            and _sole_reader(graph, heads_apart.output[0], "Split") is split,
            f"unexpected attention wiring at {softmax.name}",
        )
        _require(
            _attributes(query_rows)["perm"] == _attributes(weight_columns)["perm"] == [0, 1, 3, 2]
            and _attributes(split)["axis"] == 2,
            f"unexpected attention axes at {softmax.name}",
        )
        position = _sole_reader(graph, value_image.output[0], "Conv")
        residual = _sole_reader(graph, position.output[0], "Add")
        projection = _sole_reader(graph, residual.output[0], "Conv")
        _require(
            _sole_reader(graph, image.output[0], "Add") is residual, f"unexpected attention output at {softmax.name}"
        )

        batch, heads, head_channels, positions = graph.shapes[heads_apart.output[0]]
        key_dim, second_key_dim, value_dim = (graph.shapes[value][2] for value in (q, k, v))
        channels = heads * value_dim
        image_shape = graph.shapes[image.output[0]]
        _require(
            batch == 1 and key_dim == second_key_dim and head_channels == 2 * key_dim + value_dim,
            f"expected batch-one Q, K and V of equal heads at {softmax.name}",
        )
        _require(
            graph.shapes[qkv.output[0]] == [1, heads * head_channels, *image_shape[2:]]
            and image_shape == graph.shapes[value_image.output[0]]
            and image_shape[:2] == [1, channels]
            and image_shape[2] * image_shape[3] == positions,
            f"unexpected attention shapes at {softmax.name}",
        )
        _require(
            _attributes(position).get("group") == channels and _attributes(projection).get("group", 1) == 1,
            f"expected a depthwise position convolution and a dense output projection at {softmax.name}",
        )

        # qkv channel [head, offset] moves to Q [dimension, head], then K, then V: a [1, D * H, h, w] block whose
        # channel is dimension * H + head reshapes to [1, D, H, T] without moving any value.
        by_head = np.arange(heads * head_channels).reshape(heads, head_channels)
        ranges = ((0, key_dim), (key_dim, 2 * key_dim), (2 * key_dim, head_channels))
        _reorder_channels(
            model, graph, qkv, np.concatenate([by_head[:, start:stop].T.ravel() for start, stop in ranges]), axis=0
        )
        # The attention output and V leave in the same order, so the convolutions that read them follow it.
        by_dimension = np.arange(channels).reshape(heads, value_dim).T.ravel()
        _reorder_channels(model, graph, position, by_dimension, axis=0)
        _reorder_channels(model, graph, projection, by_dimension, axis=1)

        prefix = softmax.name + "/nchw/"
        nodes = []

        def node(kind, inputs, name, outputs=None, **attributes):
            outputs = outputs or [prefix + name]
            nodes.append(onnx.helper.make_node(kind, inputs, outputs, name=prefix + name, **attributes))
            return outputs[0]

        def constant(name, value):
            model.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), prefix + name))
            return prefix + name

        def heads_on_height(value, name, dimension):
            return node("Reshape", [value, constant(name + "_shape", [1, dimension, heads, positions])], name)

        sizes = constant("split_sizes", [heads * key_dim, heads * key_dim, channels])
        # V already is the image the position convolution reads.
        packed = [prefix + "q_packed", prefix + "k_packed", value_image.output[0]]
        node("Split", [qkv.output[0], sizes], "split", outputs=packed, axis=1)
        q4, k4 = heads_on_height(packed[0], "q", key_dim), heads_on_height(packed[1], "k", key_dim)
        v4 = heads_on_height(packed[2], "v", value_dim)
        # Scores [1, keys, H, queries]: the softmax then reduces over the key positions held as channels.
        attention_scores = node("Einsum", [q4, k4], "scores", equation="nchw,nchq->nqhw")
        scaled = node("Mul", [attention_scores, factor], "scaled")
        weights = node("Softmax", [scaled], "weights", axis=1)
        values = node("Einsum", [weights, v4], "values", equation="nchw,nqhc->nqhw")
        node("Reshape", [values, constant("image_shape", image_shape)], "image", outputs=list(image.output))

        replacements[heads_apart.output[0]] = nodes
        removed.update(
            block_node.output[0]
            for block_node in (
                heads_apart,
                split,
                query_rows,
                query_key,
                scale,
                softmax,
                weight_columns,
                weighted_values,
                image,
                value_image,
            )
        )
    _require(len(replacements) == expected, f"expected {expected} attention blocks, found {len(replacements)}")
    nodes = []
    for original in model.graph.node:
        nodes.extend(replacements.get(original.output[0], []))
        if original.output[0] not in removed:
            nodes.append(original)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    _prune(model)


def validate_rewrite(cut: bytes, rewritten: Path) -> None:
    """Require the rewritten graph to reproduce the cut graph's outputs on one seeded image."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    # An explicit thread count keeps ONNX Runtime inside the compile job's CPU allocation.
    options.intra_op_num_threads = 4
    sample = np.random.default_rng(VALIDATION_SEED).random(INPUT_SHAPE, dtype=np.float32)
    expected, actual = (
        ort.InferenceSession(graph, sess_options=options, providers=["CPUExecutionProvider"]).run(
            OUTPUTS, {INPUT: sample}
        )
        for graph in (cut, str(rewritten))
    )
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        error = float(np.max(np.abs(candidate - reference)))
        if not np.isfinite(candidate).all() or error > REWRITE_TOLERANCE:
            raise RuntimeError(f"rewritten YOLO26x-pose output {name} differs from the cut graph by {error:.2e}")


def prepare_model(source: Path, output_dir: Path) -> Path:
    """Cut the converted graph, rewrite its attention, check the result against the cut, and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _require(_shapes(model.graph.input) == {INPUT: INPUT_SHAPE}, f"inputs {_shapes(model.graph.input)}")
    _require(_shapes(model.graph.output) == SOURCE_OUTPUT_SHAPES, f"outputs {_shapes(model.graph.output)}")
    _cut_at_heads(model)
    cut = model.SerializeToString()

    _raise_opset(model)
    _keep_attention_in_nchw(model, ATTENTION_BLOCKS)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    path = output_dir / "yolo26x_pose_heads.onnx"
    onnx.save(model, path)
    validate_rewrite(cut, path)
    return path
