"""Cut YOLOv8s-obb at its raw OBB heads, before the in-graph DFL, sigmoid and rotated-box decode."""

from __future__ import annotations

from pathlib import Path

import onnx

IMAGE_SIZE = 1024
CLASS_COUNT = 15
DFL_BINS = 16
STRIDES = (8, 16, 32)
ANCHOR_COUNT = sum((IMAGE_SIZE // stride) ** 2 for stride in STRIDES)

INPUT = "images"
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
# Each exported anchor is x, y, w, h, the 15 class scores and the angle.
SOURCE_OUTPUT_SHAPES = {"output0": [1, 4 + CLASS_COUNT + 1, ANCHOR_COUNT]}
# DFL box logits (cv2), class logits (cv3), then angle logits (cv4), each from P3 to P5. Every head tensor is the
# output of the last convolution in its branch, before the DFL softmax and the class and angle sigmoids.
BRANCHES = {
    "bbox": ("cv2.{level}/cv2.{level}.2", 4 * DFL_BINS),
    "class_logit": ("cv3.{level}/cv3.{level}.2", CLASS_COUNT),
    "angle": ("cv4.{level}/cv4.{level}.2", 1),
}
HEADS = {
    f"{name}_{level}": f"/model.22/{branch.format(level=level)}/Conv_output_0"
    for name, (branch, _) in BRANCHES.items()
    for level in range(len(STRIDES))
}
OUTPUT_SHAPES = {
    f"{name}_{level}": [1, channels, IMAGE_SIZE // stride, IMAGE_SIZE // stride]
    for name, (_, channels) in BRANCHES.items()
    for level, stride in enumerate(STRIDES)
}
OUTPUTS = list(OUTPUT_SHAPES)


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported YOLOv8s-obb graph: {message}")


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


def prepare_model(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _require(_shapes(model.graph.input) == {INPUT: INPUT_SHAPE}, f"inputs {_shapes(model.graph.input)}")
    _require(_shapes(model.graph.output) == SOURCE_OUTPUT_SHAPES, f"outputs {_shapes(model.graph.output)}")
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
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    path = output_dir / "yolov8s_obb_heads.onnx"
    onnx.save(model, path)
    return path
