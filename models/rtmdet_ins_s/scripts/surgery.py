"""Prepare the converted RTMDet-Ins S ONNX for the compiler: take RGB input instead of BGR.

MMDetection normalizes BGR images with ImageNet's per-channel mean and std. Reversing the input channels of the
first convolution's weights lets the graph take the same image in RGB order, normalized with the same constants in
RGB order, which is Neat's ImageNet preset. The rewrite is exact: it only reorders weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

IMAGE_SIZE = 640
CLASS_COUNT = 80
KERNEL_PARAMETERS = 169
MASK_FEATURES = 8
STRIDES = (8, 16, 32)
LEVELS = [IMAGE_SIZE // stride for stride in STRIDES]
INPUT = "input"
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
OUTPUT_SHAPES = {
    **{f"cls_score_{i}": [1, CLASS_COUNT, size, size] for i, size in enumerate(LEVELS)},
    **{f"bbox_pred_{i}": [1, 4, size, size] for i, size in enumerate(LEVELS)},
    **{f"kernel_pred_{i}": [1, KERNEL_PARAMETERS, size, size] for i, size in enumerate(LEVELS)},
    "mask_feat": [1, MASK_FEATURES, LEVELS[0], LEVELS[0]],
}
OUTPUTS = list(OUTPUT_SHAPES)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Unsupported RTMDet-Ins graph: {message}")


def _shapes(values) -> dict[str, list[int]]:
    return {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}


def graph_path(output_dir: Path) -> Path:
    return output_dir / "rtmdet_ins_s_rgb.onnx"


def take_rgb_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """Reverse the input channels of the one convolution that reads the image."""
    _require(_shapes(model.graph.input) == {INPUT: INPUT_SHAPE}, "input interface")
    _require(_shapes(model.graph.output) == OUTPUT_SHAPES, "output interface")
    readers = [node for node in model.graph.node if INPUT in node.input]
    _require(len(readers) == 1 and readers[0].op_type == "Conv", "one convolution reads the image")
    conv = readers[0]
    _require(conv.input[0] == INPUT, "the image is the convolution's data input")
    initializers = {value.name: value for value in model.graph.initializer}
    weight_name = conv.input[1]
    _require(weight_name in initializers, "a constant first-convolution weight")
    _require(sum(weight_name in node.input for node in model.graph.node) == 1, "a private first-convolution weight")
    weight = numpy_helper.to_array(initializers[weight_name])
    _require(weight.ndim == 4 and weight.shape[1] == 3, f"a 3-channel first convolution, found {weight.shape}")
    initializers[weight_name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weight[:, ::-1]), weight_name))
    return model


def prepare_model(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    model = take_rgb_input(model)
    model.ir_version = 8
    onnx.checker.check_model(model)
    graph = graph_path(output_dir)
    onnx.save(model, graph)
    return graph


def validate_rgb_input(source: Path, graph: Path, image: np.ndarray) -> None:
    """The prepared graph on an RGB image must match the converted graph on the same image in BGR order."""
    import onnxruntime as ort

    original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    prepared = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    rgb = np.ascontiguousarray(image[None])
    expected = original.run(OUTPUTS, {INPUT: np.ascontiguousarray(rgb[:, ::-1])})
    actual = prepared.run(OUTPUTS, {INPUT: rgb})
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        # Reordering a convolution's input channels changes only float32 accumulation order.
        if not np.allclose(candidate, reference, rtol=1e-4, atol=1e-3):
            raise RuntimeError(f"RGB graph output {name} differs from the converted graph: "
                               f"{float(np.max(np.abs(candidate - reference)))}")
