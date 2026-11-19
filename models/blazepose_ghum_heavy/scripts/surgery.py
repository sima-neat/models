"""Keep BlazePose GHUM Heavy's three landmark outputs and replace its channel Pads with 1x1 convolutions.

Model Compiler 3.0.0.1 cannot place a Pad that appends channels on the MLA ("Pad mode must be constant and pad
width must be 0"): with the three Pads it splits the model into 7 MLA and APU segments, with the rewrite into 1.
A 1x1 convolution by an identity matrix over zero rows computes the same tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT = "input_1"
INPUT_SHAPE = [1, 256, 256, 3]
SOURCE_OUTPUT_SHAPES = {
    "Identity": [1, 195],
    "Identity_1": [1, 1],
    "Identity_2": [1, 256, 256, 1],
    "Identity_3": [1, 64, 64, 39],
    "Identity_4": [1, 117],
}
# Image landmarks, pose presence and world landmarks; the segmentation mask and heatmaps are dropped.
OUTPUT_SHAPES = {name: SOURCE_OUTPUT_SHAPES[name] for name in ("Identity", "Identity_1", "Identity_4")}
OUTPUTS = list(OUTPUT_SHAPES)
# Three Pads append zero channels to a MaxPool output.
CHANNEL_PADS = 3
SEEDS = range(123, 129)
# The rewrite only copies channels and appends zeros: ONNX Runtime measured 0.0 difference on every seed.
TOLERANCE = 1e-5


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported BlazePose graph: {message}")


def _shapes(values) -> dict[str, list[int]]:
    return {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim] for value in values}


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


def _appended_channels(pads: np.ndarray) -> int:
    """The zero channels a Pad appends to an NCHW tensor, or 0 when it pads anything else."""
    pads = pads.tolist()
    return pads[5] if len(pads) == 8 and pads[5] > 0 and not any(pads[:5] + pads[6:]) else 0


def _replace_channel_pads(model: onnx.ModelProto) -> None:
    """Replace each channel Pad with a 1x1 convolution whose weight is an identity matrix over zero rows."""
    initializers = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = _shapes([*inferred.graph.input, *inferred.graph.value_info])
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Pad" or node.input[1] not in initializers:
            continue
        appended = _appended_channels(initializers[node.input[1]])
        if not appended:
            continue
        _require(len(node.input) == 2 and not node.attribute, f"{node.name} does not pad with zeros")
        channels = shapes[node.input[0]][1]
        weight = np.zeros((channels + appended, channels, 1, 1), np.float32)
        weight[np.arange(channels), np.arange(channels)] = 1
        weight_name = f"{node.output[0]}/identity_weight"
        model.graph.initializer.append(numpy_helper.from_array(weight, weight_name))
        node.CopyFrom(
            helper.make_node(
                "Conv", [node.input[0], weight_name], list(node.output), name=node.name, kernel_shape=[1, 1]
            )
        )
        replaced += 1
    _require(replaced == CHANNEL_PADS, f"found {replaced} channel Pads, expected {CHANNEL_PADS}")


def validate_rewrite(source: Path, prepared: Path) -> None:
    """Require the prepared graph to reproduce the converted ONNX's kept outputs in ONNX Runtime."""
    import onnxruntime as ort

    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(prepared), providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = np.random.RandomState(seed).uniform(0, 1, INPUT_SHAPE).astype(np.float32)
        expected = reference.run(OUTPUTS, {INPUT: sample})
        actual = candidate.run(OUTPUTS, {INPUT: sample})
        for name, expected_value, actual_value in zip(OUTPUTS, expected, actual, strict=True):
            error = float(np.max(np.abs(actual_value - expected_value)))
            if not np.isfinite(actual_value).all() or error > TOLERANCE:
                raise RuntimeError(f"prepared output {name} differs from the converted graph by {error} (seed {seed})")


def prepare_model(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _require(_shapes(model.graph.input) == {INPUT: INPUT_SHAPE}, f"inputs {_shapes(model.graph.input)}")
    _require(_shapes(model.graph.output) == SOURCE_OUTPUT_SHAPES, f"outputs {_shapes(model.graph.output)}")
    outputs = {value.name: value for value in model.graph.output}
    del model.graph.output[:]
    model.graph.output.extend(outputs[name] for name in OUTPUTS)
    _prune(model)
    _replace_channel_pads(model)
    _prune(model)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    path = output_dir / "blazepose_ghum_heavy_landmarks.onnx"
    onnx.save(model, path)
    validate_rewrite(source, path)
    return path
