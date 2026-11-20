"""Collapse SESR's linear blocks into single convolutions, as SESR's own collapse() does for inference.

Each collapsible linear block is a k x k convolution to 256 channels with no bias, then a 1x1 convolution back down.
Both are linear, so they equal one k x k convolution whose weights are the product of the two; a residual block's
skip connection folds in as an identity at the kernel centre. The rewrite is exact up to float rounding and cuts the
network's multiply-adds about sixteen-fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from sesr import INPUT, INPUT_SHAPE, OUTPUT, OUTPUT_SHAPE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Unsupported SESR graph: {message}")


def _attributes(node: onnx.NodeProto) -> dict:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def _is_pointwise(node: onnx.NodeProto, weight: np.ndarray) -> bool:
    attributes = _attributes(node)
    return (
        weight.shape[2:] == (1, 1)
        and attributes.get("group", 1) == 1
        and list(attributes.get("strides", [1, 1])) == [1, 1]
        and not any(attributes.get("pads", [0, 0, 0, 0]))
    )


def collapse(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Return the model with every expand/squeeze convolution pair fused, and how many pairs were fused."""
    graph = model.graph
    weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    removed: set[int] = set()
    renamed: dict[str, str] = {}
    fused = 0
    for expand in list(graph.node):
        if expand.op_type != "Conv" or len(expand.input) != 2 or expand.input[1] not in weights:
            continue
        followers = consumers.get(expand.output[0], [])
        if len(followers) != 1 or followers[0].op_type != "Conv" or len(followers[0].input) != 3:
            continue
        squeeze = followers[0]
        squeeze_weight = weights[squeeze.input[1]]
        if not _is_pointwise(squeeze, squeeze_weight):
            continue
        expand_weight = weights[expand.input[1]]
        kernel = np.einsum("ot,tikl->oikl", squeeze_weight[:, :, 0, 0], expand_weight).astype(np.float32)
        bias = weights[squeeze.input[2]].astype(np.float32)
        output = squeeze.output[0]
        # A residual block adds its input back before the activation: fold it in as an identity kernel.
        after = consumers.get(output, [])
        if len(after) == 1 and after[0].op_type == "Add" and expand.input[0] in after[0].input:
            _require(kernel.shape[0] == kernel.shape[1], "residual block changes the channel count")
            centre = kernel.shape[2] // 2
            kernel[np.arange(kernel.shape[0]), np.arange(kernel.shape[0]), centre, centre] += 1
            removed.add(id(after[0]))
            renamed[after[0].output[0]] = output
        stem = expand.input[1]
        graph.initializer.extend(
            [numpy_helper.from_array(kernel, f"{stem}.collapsed"), numpy_helper.from_array(bias, f"{stem}.bias")]
        )
        expand.input[1] = f"{stem}.collapsed"
        expand.input.append(f"{stem}.bias")
        expand.output[0] = output
        removed.add(id(squeeze))
        fused += 1
    kept = [node for node in graph.node if id(node) not in removed]
    for node in kept:
        for index, name in enumerate(node.input):
            node.input[index] = renamed.get(name, name)
    del graph.node[:]
    graph.node.extend(kept)
    used = {name for node in graph.node for name in node.input}
    unused = [tensor for tensor in graph.initializer if tensor.name not in used]
    for tensor in unused:
        graph.initializer.remove(tensor)
    return model, fused


def prepare_model(source: Path, directory: Path) -> Path:
    """Write the collapsed graph the compiler loads and return its path."""
    model = onnx.load(str(source))
    shapes = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in [*model.graph.input, *model.graph.output]
    }
    _require(shapes == {INPUT: INPUT_SHAPE, OUTPUT: OUTPUT_SHAPE}, f"interface {shapes}")
    model, fused = collapse(model)
    _require(fused > 0, "found no linear blocks to collapse")
    _require(all(node.op_type != "Add" or len(node.input) == 2 for node in model.graph.node), "malformed Add")
    onnx.checker.check_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sesr_collapsed.onnx"
    onnx.save(model, str(path))
    return path
