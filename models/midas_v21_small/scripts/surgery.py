"""Exact compile-time rewrites of the converted MiDaS v2.1 Small ONNX for the Model Compiler.

Each RefineNet fusion block upsamples 2x bilinearly and then projects with a 1x1 convolution. A 1x1 convolution
mixes channels pixel by pixel, and bilinear interpolation mixes pixels channel by channel with weights that sum to 1,
so the two commute, bias included: Conv(Resize(x)) == Resize(Conv(x)). Projecting first runs the convolution on a
quarter of the pixels, and in the three blocks whose projection halves the channels the upsample moves half as much
data. The conversion stays the upstream graph; this module rewrites it only for compilation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx

INPUT = "pixel_values"
INPUT_SHAPE = (1, 3, 256, 256)
OUTPUT = "predicted_depth"
# The four RefineNet blocks, each a bilinear Resize followed by its 1x1 projection.
PROJECTIONS = 4
# The float check's inputs and tolerance, relative to the output's scale. On ten noise seeds the prepared graph
# differed from the converted one by at most 4.3e-7 of the scale; the margin covers other CPUs' rounding.
SEEDS = range(3)
TOLERANCE = 1e-4


def _attributes(node: onnx.NodeProto) -> dict:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def _is_pointwise(conv: onnx.NodeProto, weights: dict[str, onnx.TensorProto]) -> bool:
    attributes = _attributes(conv)
    weight = weights.get(conv.input[1])
    return (
        weight is not None
        and list(weight.dims[2:]) == [1, 1]
        and attributes.get("group", 1) == 1
        and set(attributes.get("strides", [1, 1])) == {1}
        and set(attributes.get("dilations", [1, 1])) == {1}
        and set(attributes.get("pads", [0, 0, 0, 0])) == {0}
    )


def commute_projections(model: onnx.ModelProto) -> int:
    """Move each 1x1 convolution that alone consumes a bilinear Resize in front of it. Returns how many moved."""
    graph = model.graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    moved = 0
    for conv in list(graph.node):
        resize = producers.get(conv.input[0]) if conv.op_type == "Conv" else None
        if (
            resize is None
            or resize.op_type != "Resize"
            or _attributes(resize).get("mode", b"nearest") != b"linear"
            or len(consumers[resize.output[0]]) != 1
            or not _is_pointwise(conv, weights)
        ):
            continue
        # Resize(x) -> Conv  becomes  Conv(x) -> Resize, keeping the block's output name for its consumers.
        projected = f"{conv.output[0]}_before_upsample"
        conv.input[0], conv.output[0], resize.input[0], resize.output[0] = (
            resize.input[0],
            projected,
            projected,
            conv.output[0],
        )
        moved += 1
    _sort_topologically(graph)
    return moved


def _sort_topologically(graph: onnx.GraphProto) -> None:
    """Reorder the nodes so each runs after the producers of its inputs, keeping the original order otherwise."""
    available = {value.name for value in graph.input} | {tensor.name for tensor in graph.initializer} | {""}
    pending, ordered = list(graph.node), []
    while pending:
        ready = next(node for node in pending if all(name in available for name in node.input))
        pending.remove(ready)
        ordered.append(ready)
        available.update(ready.output)
    nodes = [onnx.NodeProto.FromString(node.SerializeToString()) for node in ordered]
    graph.ClearField("node")
    graph.node.extend(nodes)


def prepare_model(source: Path, output: Path) -> Path:
    """Write the graph the recipe compiles: the converted ONNX with every projection moved before its upsample."""
    model = onnx.load(source)
    moved = commute_projections(model)
    if moved != PROJECTIONS:
        raise ValueError(f"unexpected MiDaS graph: moved {moved} projections, expected {PROJECTIONS}")
    # The tensors between the moved nodes changed shape; infer them again.
    model.graph.ClearField("value_info")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    _check_equivalent(source, output)
    return output


def _check_equivalent(source: Path, prepared: Path) -> None:
    """The rewrite is exact, so the prepared graph must reproduce the converted one up to float rounding."""
    import onnxruntime as ort

    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (source, prepared)]
    for seed in SEEDS:
        image = np.random.default_rng(seed).standard_normal(INPUT_SHAPE).astype(np.float32)
        expected, actual = (session.run([OUTPUT], {INPUT: image})[0] for session in sessions)
        difference, scale = float(np.max(np.abs(actual - expected))), float(np.max(np.abs(expected)))
        if difference > TOLERANCE * max(scale, 1.0):
            raise ValueError(
                f"the prepared graph departs from the converted ONNX by {difference:.3g} (scale {scale:.3g})"
            )
