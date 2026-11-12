"""BF16 normalization, residual and attention regions inside the INT8 depth model."""

from collections import Counter, defaultdict

import onnx
from onnx import helper, numpy_helper

BF16_REGIONS = ("normalization_residual", "attention")
LAYOUT = {"Reshape", "Transpose", "Slice", "Gather", "Concat", "Squeeze", "Identity"}


def regions(model: onnx.ModelProto) -> dict[str, set[str]]:
    """Keep LayerScale outside the BF16 regions so fusion preserves INT8 MLP projections."""
    selected = {name: set() for name in BF16_REGIONS}
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        normalization = "/norm" in node.name and (
            node.op_type in {"Div", "Mul"}
            or node.op_type == "Add"
            and any(name in constants and constants[name].shape == (384,) for name in node.input)
        )
        residual = (
            node.name.startswith("/blocks.")
            and node.name.count("/") == 2
            and node.name.rsplit("/", 1)[-1] in {"Add", "Add_1"}
        )
        if normalization or residual or node.name == "/Add_2":
            selected["normalization_residual"].add(node.name)
        # Only QK/PV products belong to BF16 attention. Unfused learned QKV/projection
        # MatMuls have constant weights and must keep the profile's INT8 precision.
        attention = node.op_type in {"Softmax", "Mul"} or (
            node.op_type == "MatMul" and not any(value in constants for value in node.input)
        )
        if "/attn/" in node.name and attention:
            selected["attention"].add(node.name)
    if {name: len(nodes) for name, nodes in selected.items()} != {"normalization_residual": 106, "attention": 48}:
        raise ValueError("unexpected Depth Anything V2 normalization, residual or attention graph")
    return selected


def annotate_bf16_regions(model: onnx.ModelProto, names: list[str]) -> dict[str, int]:
    """Mark BF16 producers and reset their first compute readers outside each region to INT8."""
    if len(set(names)) != len(names) or set(names) - set(BF16_REGIONS):
        raise ValueError(f"expected distinct BF16 regions from {BF16_REGIONS}, got {names}")
    available = regions(model)
    selected = set().union(*(available[name] for name in names))
    users = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            users[value].append(node)

    def readers(value):
        for node in users[value]:
            if node.op_type in LAYOUT and node.name not in selected:
                for output in node.output:
                    yield from readers(output)
            else:
                yield node

    marks = dict.fromkeys(selected, "bfloat16")
    for node in model.graph.node:
        if node.name in selected:
            for value in node.output:
                for reader in readers(value):
                    if reader.name not in selected:
                        marks[reader.name] = "int8"
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if node.name in marks:
            for index, value in enumerate(list(node.output)):
                node.output[index] = value + "/unannotated"
                nodes.append(
                    helper.make_node(
                        "AnnotatePrecision",
                        [node.output[index]],
                        [value],
                        name=node.name + f"/AnnotatePrecision_{index}",
                        domain="ai.sima",
                        precision=marks[node.name],
                    )
                )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid("ai.sima", 1))
    return dict(Counter(marks.values()))
