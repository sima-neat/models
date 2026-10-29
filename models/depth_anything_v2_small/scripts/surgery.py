"""Fold the Depth Anything V2 Small ONNX export to its static input for the Model Compiler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx

INPUT = "pixel_values"
OUTPUT = "predicted_depth"
INPUT_SHAPE = [1, 3, 518, 686]
OUTPUT_SHAPE = [1, 518, 686]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unexpected Depth Anything V2 graph: {message}")


def _shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    return {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}


def fold(source: Path) -> onnx.ModelProto:
    """Fix the export's dynamic shapes to one 518x686 image and fold every shape computation."""
    from onnxsim import simplify

    model, valid = simplify(onnx.load(source), overwrite_input_shapes={INPUT: INPUT_SHAPE})
    _require(valid, "onnxsim could not validate the folded graph")
    model = onnx.shape_inference.infer_shapes(model)
    shapes = _shapes(model)
    _require(shapes.get(INPUT) == INPUT_SHAPE and shapes.get(OUTPUT) == OUTPUT_SHAPE, "input or output interface")
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Shape", "ScatterND", "Range", "Where"})
    _require(not unfolded, f"shape computations survived folding: {unfolded}")
    return model


def prepare_model(source: Path, output: Path) -> Path:
    """Write the graph the recipe imports."""
    model = fold(source)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return output


def max_difference(source: Path, prepared: Path, image: np.ndarray) -> float:
    """Largest depth difference between the source and the prepared graph on one NCHW image."""
    import onnxruntime as ort

    outputs = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run([OUTPUT], {INPUT: image})[0]
        for path in (source, prepared)
    ]
    return float(np.max(np.abs(outputs[0] - outputs[1])))
