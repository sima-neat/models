"""Rewrite the converted DETR ResNet-101 graph into equivalent forms the MLA compiles efficiently.

Three exact rewrites run on the folded export, each guarded by the number of matches it expects:

- every frozen batch norm (a Mul and an Add by per-channel constants) folds into the convolution before it;
- every attention mask, which is all zeros for an unpadded input, is dropped from its scores;
- every attention block keeps its scores, softmax and output projection in an NCHW layout: Q, K and V become
  [1, D, H, T], the scores an einsum over D, the softmax runs over the key axis as channels, and the output
  projection becomes a 1x1 convolution on the head-packed values.

The graph then ends the box head before its final Sigmoid, so the package returns box logits for Neat's DETR
box decoder, which applies that sigmoid itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT_HEIGHT, INPUT_WIDTH = 640, 640
QUERY_COUNT = 100
CLASS_COUNT = 92
# The ResNet-101 backbone's 104 convolutions each carry a frozen batch norm.
BATCH_NORMS = 104
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
# The six encoder self-attentions and six decoder cross-attentions read a padding mask.
MASKED_ATTENTIONS = ENCODER_LAYERS + DECODER_LAYERS
# onnxsim folds the first decoder self-attention, whose input is all zeros, into a constant.
ATTENTIONS = ENCODER_LAYERS + 2 * DECODER_LAYERS - 1

INPUT = "input"
INPUT_SHAPE = [1, 3, INPUT_HEIGHT, INPUT_WIDTH]
SOURCE_OUTPUT_SHAPES = {"logits": [1, QUERY_COUNT, CLASS_COUNT], "pred_boxes": [1, QUERY_COUNT, 4]}
SOURCE_OUTPUTS = list(SOURCE_OUTPUT_SHAPES)
OUTPUT_SHAPES = {"logits": [1, QUERY_COUNT, CLASS_COUNT], "box_logits": [1, QUERY_COUNT, 4]}
OUTPUTS = list(OUTPUT_SHAPES)
VALIDATION_SEED = 123
# The rewrites change float32 accumulation order: on ten COCO images the outputs stay below 5e-4 in the logits and
# 3e-5 in the box fractions, and every confident detection keeps its class.
LOGITS_TOLERANCE = 3e-3
BOXES_TOLERANCE = 1e-3
# The operations a W8A8 build can run BF16, by region, and how many each region has.
BF16_REGIONS = {
    # 17 blocks of scores, scale, softmax, weighted values, head merge (Concat) and output projection; 50 Q, K and V
    # Linear layers (MatMul, Add) and their head splits (Reshape, Transpose), the first cross-attention's queries being
    # a constant; and 17 position embedding Adds, shared by each self-attention's Q and K and by every
    # cross-attention's keys.
    "attention": 319,
    # 30 LayerNorms and the 29 residual sums they normalize.
    "residual": 59,
    # Twelve feed-forward networks of five operations.
    "ffn": 60,
    # The class Linear (MatMul, Add) and the box MLP up to its logits (three MatMul and Add, two Relu).
    "heads": 10,
}
# Operations that move values without computing: a region ends at the first other operation that reads it.
LAYOUT_OPS = {"Reshape", "Transpose", "Split", "Concat"}


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Unsupported DETR graph: {message}")


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _validate_interface(model: onnx.ModelProto, output_shapes: dict[str, list[int]]) -> None:
    inputs = {value.name: _shape(value) for value in model.graph.input}
    outputs = {value.name: _shape(value) for value in model.graph.output}
    _require(inputs == {INPUT: INPUT_SHAPE}, f"unexpected input interface {inputs}")
    _require(outputs == output_shapes, f"unexpected output interface {outputs}")


def _require_folded(model: onnx.ModelProto) -> None:
    # convert.py folds the export with onnxsim; matching relies on static shapes and constants.
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Constant", "Shape"})
    _require(not unfolded, f"source still has {unfolded} nodes; fold it with onnxsim first")


@dataclass
class GraphIndex:
    """Lookups over one ONNX graph, built once per rewrite."""

    shapes: dict[str, list[int]]
    constants: dict[str, np.ndarray]
    producers: dict[str, onnx.NodeProto]
    # A None reader marks a graph output.
    users: defaultdict[str, list[onnx.NodeProto | None]]


def _index(model: onnx.ModelProto) -> GraphIndex:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    shapes = {value.name: _shape(value) for value in values}
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    shapes.update({name: list(value.shape) for name, value in constants.items()})
    producers, users = {}, defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            users[name].append(node)
    for value in model.graph.output:
        users[value.name].append(None)
    return GraphIndex(shapes=shapes, constants=constants, producers=producers, users=users)


def _sole_reader(graph: GraphIndex, node: onnx.NodeProto, kind: str) -> onnx.NodeProto:
    following = graph.users[node.output[0]]
    _require(
        len(following) == 1 and following[0] is not None and following[0].op_type == kind,
        f"expected private {node.name} -> {kind}",
    )
    return following[0]


def _producer(graph: GraphIndex, value: str, kind: str) -> onnx.NodeProto:
    node = graph.producers.get(value)
    _require(node is not None and node.op_type == kind, f"expected {kind} producing {value}")
    return node


def _constant_operand(graph: GraphIndex, node: onnx.NodeProto) -> tuple[str, str]:
    """Return (variable input, constant input) of a binary node with exactly one constant operand."""
    first, second = node.input
    _require((first in graph.constants) != (second in graph.constants), f"{node.name} needs one constant operand")
    return (second, first) if first in graph.constants else (first, second)


def _replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    _require(old not in {value.name for value in model.graph.output}, f"cannot bypass public output {old}")
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def _prune_and_check(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop nodes and initializers that no longer reach a graph output, then re-infer and check every shape."""
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
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def _fold_frozen_batch_norms(model: onnx.ModelProto, expected: int) -> onnx.ModelProto:
    """Conv(x, W, b) * s + t is Conv(x, W * s, b * s + t) for per-output-channel constants s and t."""
    graph = _index(model)
    folded = 0
    for conv in list(model.graph.node):
        if conv.op_type != "Conv":
            continue
        readers = graph.users[conv.output[0]]
        if len(readers) != 1 or readers[0] is None or readers[0].op_type != "Mul":
            continue
        scale_node = readers[0]
        shift_node = _sole_reader(graph, scale_node, "Add")
        _, scale_name = _constant_operand(graph, scale_node)
        _, shift_name = _constant_operand(graph, shift_node)
        weight = graph.constants[conv.input[1]]
        channels = weight.shape[0]
        scale, shift = graph.constants[scale_name], graph.constants[shift_name]
        _require(scale.shape == shift.shape == (1, channels, 1, 1), f"per-channel batch norm after {conv.name}")
        bias = graph.constants[conv.input[2]] if len(conv.input) > 2 else np.zeros(channels, np.float32)
        prefix = conv.name + "/folded_batch_norm/"
        weight_name, bias_name = prefix + "weight", prefix + "bias"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array((weight * scale.reshape(channels, 1, 1, 1)).astype(np.float32), weight_name),
                numpy_helper.from_array((bias * scale.ravel() + shift.ravel()).astype(np.float32), bias_name),
            ]
        )
        del conv.input[1:]
        conv.input.extend([weight_name, bias_name])
        _replace_uses(model, shift_node.output[0], conv.output[0])
        folded += 1
    _require(folded == expected, f"expected {expected} frozen batch norms, found {folded}")
    return _prune_and_check(model)


def _remove_zero_masks(model: onnx.ModelProto, expected: int) -> onnx.ModelProto:
    """Adding an all-zero mask to the attention scores changes nothing, so the softmax reads the scores directly."""
    graph = _index(model)
    removed = 0
    for node in list(model.graph.node):
        if node.op_type != "Add" or not any(
            reader and reader.op_type == "Softmax" for reader in graph.users[node.output[0]]
        ):
            continue
        scores, mask = _constant_operand(graph, node)
        _require(not graph.constants[mask].any(), f"mask at {node.name} excludes positions")
        _require(graph.shapes.get(scores) == graph.shapes.get(node.output[0]), f"broadcasting mask at {node.name}")
        _replace_uses(model, node.output[0], scores)
        removed += 1
    _require(removed == expected, f"expected {expected} attention masks, found {removed}")
    return _prune_and_check(model)


def _optimize_attention(model: onnx.ModelProto, expected: int) -> onnx.ModelProto:
    """Keep each attention's scores, softmax, head merge and output projection in NCHW."""
    graph = _index(model)
    replacements, removed = {}, set()
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), f"last-axis softmax required at {softmax.name}")
        scale = _producer(graph, softmax.input[0], "Mul")
        _require(_sole_reader(graph, scale, "Softmax") is softmax, "private score scale")
        scores, factor = _constant_operand(graph, scale)
        _require(graph.constants[factor].size == 1, f"scalar score scale at {scale.name}")
        qk = _producer(graph, scores, "MatMul")
        _require(_sole_reader(graph, qk, "Mul") is scale, "private attention scores")
        pv = _sole_reader(graph, softmax, "MatMul")
        _require(pv.input[0] == softmax.output[0], "probabilities must be the left PV operand")
        merge = _sole_reader(graph, pv, "Transpose")
        flatten = _sole_reader(graph, merge, "Reshape")
        projection = _sole_reader(graph, flatten, "MatMul")
        restore = _sole_reader(graph, projection, "Add")
        weight_name = projection.input[1]
        _, bias_name = _constant_operand(graph, restore)
        q, k = qk.input
        v = pv.input[1]
        _require(len(graph.shapes.get(q, [])) == 4 and min(graph.shapes[q]) > 0, "static Q shape required")
        batch, heads, queries, dim = graph.shapes[q]
        keys = graph.shapes.get(k, [0, 0, 0, 0])[3]
        _require(
            batch == 1
            and graph.shapes.get(k) == [1, heads, dim, keys]
            and graph.shapes.get(v) == [1, heads, keys, dim],
            f"expected batch-one equal-head attention at {softmax.name}",
        )
        _require(
            _attributes(merge).get("perm") == [0, 2, 1, 3]
            and graph.shapes.get(flatten.output[0]) == [1, queries, heads * dim],
            f"unexpected head merge at {merge.name}",
        )
        _require(weight_name in graph.constants, f"constant projection weight required at {projection.name}")
        weight, bias = graph.constants[weight_name], graph.constants[bias_name]
        channels = weight.shape[1]
        _require(weight.shape[0] == heads * dim and bias.shape == (channels,), "projection shape mismatch")
        _require(graph.shapes.get(restore.output[0]) == [1, queries, channels], "restored sequence shape")
        prefix = softmax.name + "/nchw/"
        nodes = []

        def node(kind, inputs, name, outputs=None, **attributes):
            outputs = outputs or [prefix + name]
            nodes.append(helper.make_node(kind, inputs, outputs, name=prefix + name, **attributes))
            return outputs[0]

        def constant(name, value):
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), prefix + name))
            return prefix + name

        def channels_first(value, name):
            # [1,H,T,D] (Q, V) or [1,H,D,T] (K) -> [1,D,H,T], folding the exporter's head Transpose into it.
            target = [0, 2, 1, 3] if value == k else [0, 3, 1, 2]
            if value in graph.constants:  # The first cross-attention's queries come from the folded first layer.
                _require(len(graph.users[value]) == 1, f"shared constant {value}")
                return constant(name, np.ascontiguousarray(graph.constants[value].transpose(target)))
            source = graph.producers[value]
            if source.op_type == "Transpose" and len(graph.users[value]) == 1:
                perm = _attributes(source)["perm"]
                return node("Transpose", [source.input[0]], name, perm=[perm[axis] for axis in target])
            return node("Transpose", [value], name, perm=target)

        q4, k4, v4 = channels_first(q, "q"), channels_first(k, "k"), channels_first(v, "v")
        # Scores [1, keys, H, queries]: the softmax then reduces over the key axis held as channels.
        attention_scores = node("Einsum", [q4, k4], "scores", equation="nchw,nchq->nqhw")
        scaled = node("Mul", [attention_scores, factor], "scaled")
        probabilities = node("Softmax", [scaled], "probabilities", axis=1)
        values = node("Einsum", [probabilities, v4], "values", equation="nchw,nqhc->nqhw")
        split = [prefix + f"head{index}" for index in range(heads)]
        node("Split", [values, constant("split_sizes", np.ones(heads, dtype=np.int64))], "split", outputs=split, axis=2)
        packed = node("Concat", split, "packed", axis=1)
        # MatMul computes X @ W + b with W stored [in, out]; a 1x1 Conv needs [out, in, 1, 1].
        conv_weight = constant("weight", np.ascontiguousarray(weight.T.reshape(channels, heads * dim, 1, 1)))
        projected = node("Conv", [packed, conv_weight, bias_name], "projection", kernel_shape=[1, 1])
        sequence = node("Transpose", [projected], "sequence", perm=[0, 3, 2, 1])
        node(
            "Reshape",
            [sequence, constant("output_shape", np.array([1, queries, channels], dtype=np.int64))],
            "output",
            outputs=list(restore.output),
        )
        # Insert at the original output Add, after every Q, K, V and bias producer.
        replacements[restore.output[0]] = nodes
        removed.update(n.output[0] for n in (qk, scale, softmax, pv, merge, flatten, projection, restore))
    _require(len(replacements) == expected, f"expected {expected} attention blocks, found {len(replacements)}")
    nodes = []
    for original in model.graph.node:
        nodes.extend(replacements.get(original.output[0], []))
        if original.output[0] not in removed:
            nodes.append(original)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return _prune_and_check(model)


def _cut_box_sigmoid(model: onnx.ModelProto) -> onnx.ModelProto:
    """End the box head at the logits its final Sigmoid reads, and publish them as box_logits."""
    graph = _index(model)
    sigmoid = _producer(graph, "pred_boxes", "Sigmoid")
    logits = sigmoid.input[0]
    _require(graph.users[logits] == [sigmoid], f"box logits read outside {sigmoid.name}")
    head = graph.producers[logits]
    head.output[list(head.output).index(logits)] = "box_logits"
    model.graph.node.remove(sigmoid)
    next(value for value in model.graph.output if value.name == "pred_boxes").name = "box_logits"
    return _prune_and_check(model)


def _head_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Every operation between the decoder's last LayerNorm and the two outputs."""
    found: dict[str, onnx.NodeProto] = {}
    pending = list(OUTPUTS)
    while pending:
        node = graph.producers.get(pending.pop())
        if node is None or node.name in found or node.op_type == "LayerNormalization":
            continue
        found[node.name] = node
        pending.extend(node.input)
    return list(found.values())


def _projection_operations(graph: GraphIndex, value: str) -> list[onnx.NodeProto]:
    """The Linear layer (MatMul, Add) that projects an attention's Q, K or V, the layout operations that split it into
    heads, and the position embedding it reads."""
    if value in graph.constants:  # The first cross-attention's queries are a folded constant.
        return []
    node, layout = graph.producers[value], []
    while node.op_type in LAYOUT_OPS:
        layout.append(node)
        node = graph.producers[node.input[0]]
    _require(node.op_type == "Add", f"expected the projection bias producing {value}")
    projection = _producer(graph, _constant_operand(graph, node)[0], "MatMul")
    operations = [projection, node, *layout]
    embedding = graph.producers.get(projection.input[0])
    if embedding is not None and embedding.op_type == "Add" and len(set(embedding.input) & set(graph.constants)) == 1:
        operations.append(embedding)
    return operations


def _attention_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Every attention block's Q, K and V projections, scores, softmax, weighted values, head merge and output
    projection.

    AFE runs a layout operation at the global precision unless it is marked, so the head split and merge inside a block
    are marked too. The Split has eight outputs and cannot carry a mark; its slices take the precision of their input.
    """
    operations = []
    for softmax in [node for node in graph.producers.values() if node.op_type == "Softmax"]:
        scale = _producer(graph, softmax.input[0], "Mul")
        scores = _producer(graph, scale.input[0], "Einsum")
        values = _sole_reader(graph, softmax, "Einsum")
        split = _sole_reader(graph, values, "Split")
        packed = graph.users[split.output[0]][0]
        projection = _sole_reader(graph, packed, "Conv")
        operations += [scores, scale, softmax, values, packed, projection]
        for value in (*scores.input, values.input[1]):
            operations += _projection_operations(graph, value)
    return operations


def _residual_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Every LayerNorm and the residual sum it normalizes."""
    operations = []
    for norm in [node for node in graph.producers.values() if node.op_type == "LayerNormalization"]:
        operations.append(norm)
        source = graph.producers[norm.input[0]]
        if source.op_type == "Add":  # The decoder's last LayerNorm reads the final layer's LayerNorm instead.
            operations.append(source)
    return operations


def _ffn_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Every layer's feed-forward network: two Linear layers (MatMul, Add) around a Relu.

    The backbone's Relus read convolutions. The box head's MLP has the same shape as a feed-forward network, but
    its second Linear feeds a Relu rather than a residual sum.
    """
    operations = []
    for relu in [node for node in graph.producers.values() if node.op_type == "Relu"]:
        expand = graph.producers[relu.input[0]]
        variables = [value for value in expand.input if value not in graph.constants]
        linear = graph.producers.get(variables[0]) if expand.op_type == "Add" and len(variables) == 1 else None
        if linear is None or linear.op_type != "MatMul":
            continue
        contract = _sole_reader(graph, relu, "MatMul")
        reduce = _sole_reader(graph, contract, "Add")
        if [reader.op_type for reader in graph.users[reduce.output[0]] if reader is not None] == ["Add"]:
            operations += [linear, expand, relu, contract, reduce]
    return operations


REGION_OPERATIONS = {
    "attention": _attention_operations,
    "residual": _residual_operations,
    "ffn": _ffn_operations,
    "heads": _head_operations,
}


def _first_operations(graph: GraphIndex, value: str) -> list[onnx.NodeProto]:
    """The operations that read value, looking through layout operations."""
    operations = []
    for reader in graph.users[value]:
        if reader is None:
            continue
        if reader.op_type in LAYOUT_OPS:
            for output in reader.output:
                operations += _first_operations(graph, output)
        else:
            operations.append(reader)
    return operations


def precision_marks(model: onnx.ModelProto, regions: list[str]) -> dict[str, str]:
    """Return the precision to annotate after each operation, by name: bfloat16 for the regions' operations and int8
    for the operations that first read their results.

    AFE gives an annotation's precision to the operation producing its input, and each later operation takes the
    precision its inputs share, so an int8 mark on each first reader ends a region.
    """
    unknown = sorted(set(regions) - set(BF16_REGIONS))
    _require(not unknown, f"unknown BF16 regions {unknown}; choose from {list(BF16_REGIONS)}")
    graph = _index(model)
    promoted = {}
    for region in regions:
        operations = {node.name: node for node in REGION_OPERATIONS[region](graph)}
        _require(
            len(operations) == BF16_REGIONS[region],
            f"expected {BF16_REGIONS[region]} {region} operations, found {len(operations)}",
        )
        promoted.update(operations)
    marks = dict.fromkeys(promoted, "bfloat16")
    for node in promoted.values():
        for reader in _first_operations(graph, node.output[0]):
            marks.setdefault(reader.name, "int8")
    return marks


def annotate_bf16_regions(graph: Path, regions: list[str]) -> dict[str, int]:
    """Mark the regions' operations BF16 for AFE, in place; return how many marks of each precision it wrote."""
    model = onnx.load(graph)
    marks = precision_marks(model, regions)
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if node.name in marks:
            _require(len(node.output) == 1, f"one output to annotate at {node.name}")
            value = node.output[0]
            node.output[0] = f"{value}/unannotated"
            nodes.append(
                helper.make_node(
                    "AnnotatePrecision",
                    [node.output[0]],
                    [value],
                    name=f"{node.name}/AnnotatePrecision",
                    domain="ai.sima",
                    precision=marks[node.name],
                )
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid("ai.sima", 1))
    onnx.checker.check_model(model)
    onnx.save(model, graph)
    return {precision: list(marks.values()).count(precision) for precision in ("bfloat16", "int8")}


def validate_rewrite(source: Path, rewritten: Path, inputs: np.ndarray) -> None:
    """Require the rewritten graph to reproduce the converted graph's outputs on NCHW float32 inputs.

    The box logits are compared after the sigmoid the decoder applies, as the converted graph's boxes.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    # An explicit thread count keeps ONNX Runtime inside the compile job's CPU allocation.
    options.intra_op_num_threads = 4
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for path in (source, rewritten)
    ]
    for sample in inputs:
        expected = sessions[0].run(SOURCE_OUTPUTS, {INPUT: sample[None]})
        logits, box_logits = sessions[1].run(OUTPUTS, {INPUT: sample[None]})
        actual = [logits, 1 / (1 + np.exp(-box_logits))]
        for name, reference, candidate in zip(SOURCE_OUTPUTS, expected, actual, strict=True):
            tolerance = LOGITS_TOLERANCE if name == "logits" else BOXES_TOLERANCE
            error = float(np.max(np.abs(candidate - reference)))
            if not np.isfinite(candidate).all() or error > tolerance:
                raise RuntimeError(f"rewritten DETR output {name} differs from the converted graph by {error:.2e}")


def prepare_model(source: Path, output_dir: Path) -> Path:
    """Rewrite the converted graph, check it against the source on a seeded input, and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _validate_interface(model, SOURCE_OUTPUT_SHAPES)
    _require_folded(model)
    model = _fold_frozen_batch_norms(model, BATCH_NORMS)
    model = _remove_zero_masks(model, MASKED_ATTENTIONS)
    model = _optimize_attention(model, ATTENTIONS)
    model = _cut_box_sigmoid(model)
    _validate_interface(model, OUTPUT_SHAPES)
    path = output_dir / "detr_resnet_101_nchw_attention.onnx"
    onnx.save(model, path)
    sample = np.random.default_rng(VALIDATION_SEED).standard_normal([1, *INPUT_SHAPE[1:]], dtype=np.float32)
    validate_rewrite(source, path, sample)
    return path
