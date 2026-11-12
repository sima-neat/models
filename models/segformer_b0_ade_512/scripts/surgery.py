"""Mark regions of the converted SegFormer-B0 graph to run BF16 in a W8A8 build.

The converted graph compiles unchanged; this module only adds precision annotations for a mixed build. Each region
is found by walking the graph from anchor operations, and the walk fails when a region does not have its expected
number of operations, so a changed export cannot silently mark the wrong operations.

The regions hold the operations INT8 handles worst, while every Linear layer and convolution stays INT8: the
attention softmax over 256 keys, the LayerNorms and residual sums (residual-stream channels reach 15-18 times the
median channel's peak in stages 3 and 4), and the GELU.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ENCODER_BLOCKS = 8
# The computing operations a W8A8 build can run BF16, by region, and how many each region has.
BF16_REGIONS = {
    # Per block: the attention scores (MatMul), their constant scale (Div), the softmax and the weighted values.
    "attention_core": 4 * ENCODER_BLOCKS,
    # All thirty LayerNorms (two per block, one closing each stage, one after each patch embedding and one after each
    # of the six key/value spatial reductions) and every block's two residual sums.
    "norms": 30 + 2 * ENCODER_BLOCKS,
    # Per block: the final multiply of the exported GELU, which carries the fused GELU's precision.
    "gelu": ENCODER_BLOCKS,
}
# Operations that move values without computing: a region ends at the first other operation that reads it.
LAYOUT_OPS = {"Reshape", "Transpose", "Split", "Concat"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Unsupported SegFormer graph: {message}")


@dataclass
class GraphIndex:
    constants: dict[str, np.ndarray]
    producers: dict[str, onnx.NodeProto]
    # A None reader marks a graph output.
    users: defaultdict[str, list[onnx.NodeProto | None]]


def _index(model: onnx.ModelProto) -> GraphIndex:
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    producers, users = {}, defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            users[name].append(node)
    for value in model.graph.output:
        users[value.name].append(None)
    return GraphIndex(constants=constants, producers=producers, users=users)


def _producer(graph: GraphIndex, value: str, kind: str) -> onnx.NodeProto:
    node = graph.producers.get(value)
    _require(node is not None and node.op_type == kind, f"expected {kind} producing {value}")
    return node


def _sole_reader(graph: GraphIndex, node: onnx.NodeProto, kind: str) -> onnx.NodeProto:
    following = graph.users[node.output[0]]
    _require(
        len(following) == 1 and following[0] is not None and following[0].op_type == kind,
        f"expected private {node.name} -> {kind}",
    )
    return following[0]


def _variable_inputs(graph: GraphIndex, node: onnx.NodeProto) -> list[str]:
    return [value for value in node.input if value and value not in graph.constants]


def _attention_core_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Each block's attention scores, constant scale, softmax over the keys and weighted values."""
    operations = []
    for softmax in [node for node in graph.producers.values() if node.op_type == "Softmax"]:
        scale = _producer(graph, softmax.input[0], "Div")
        _require(scale.input[1] in graph.constants, f"constant attention scale at {scale.name}")
        scores = _producer(graph, scale.input[0], "MatMul")
        _require(len(_variable_inputs(graph, scores)) == 2, f"two activation operands at {scores.name}")
        values = _sole_reader(graph, softmax, "MatMul")
        operations += [scores, scale, softmax, values]
    return operations


def _norm_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Every LayerNorm, and every residual sum: the Adds of two activations (a bias or GELU Add reads a constant)."""
    return [
        node for node in graph.producers.values()
        if node.op_type == "LayerNormalization" or (node.op_type == "Add" and len(_variable_inputs(graph, node)) == 2)
    ]


def _gelu_operations(graph: GraphIndex) -> list[onnx.NodeProto]:
    """Each block's GELU, x * (1 + erf(x / sqrt(2))) / 2 exported as Div, Erf, Add, Mul, Mul, by its last multiply.

    AFE fuses the five operations into one and gives the fused operation the precision annotated at its output; a
    mark inside the pattern would stop the fusion, so only the final multiply is marked.
    """
    operations = []
    for erf in [node for node in graph.producers.values() if node.op_type == "Erf"]:
        divide = _producer(graph, erf.input[0], "Div")
        _require(divide.input[1] in graph.constants, f"constant GELU scale at {divide.name}")
        shift = _sole_reader(graph, erf, "Add")
        gate = _sole_reader(graph, shift, "Mul")
        _require(divide.input[0] in gate.input, f"GELU gate must read its input at {gate.name}")
        half = _sole_reader(graph, gate, "Mul")
        operations.append(half)
    return operations


REGION_OPERATIONS = {
    "attention_core": _attention_core_operations,
    "norms": _norm_operations,
    "gelu": _gelu_operations,
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
    promoted: dict[str, onnx.NodeProto] = {}
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
