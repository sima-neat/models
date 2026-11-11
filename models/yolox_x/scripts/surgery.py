"""Expose the nine YOLOX raw heads for Neat's interleaved YOLOX decoder.

Cut before upstream score sigmoid, concatenation, flattening and grid decode. Preserve the original
backbone, including Focus: no performance benefit is assumed from rewriting an already-supported stem.
"""

from pathlib import Path

import onnx

INPUT = "images"
IMAGE_SIZE = 640
CLASS_COUNT = 80
STRIDES = (8, 16, 32)
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
SOURCE_OUTPUT_SHAPES = {"predictions": [1, sum((IMAGE_SIZE // s) ** 2 for s in STRIDES), CLASS_COUNT + 5]}
HEADS = {
    f"{role}_{level}": f"head.{branch}.{level}.weight"
    for level in range(len(STRIDES))
    for role, branch in (("bbox", "reg_preds"), ("objectness_logit", "obj_preds"), ("class_logit", "cls_preds"))
}
OUTPUT_SHAPES = {
    f"{role}_{level}": [1, channels, IMAGE_SIZE // stride, IMAGE_SIZE // stride]
    for level, stride in enumerate(STRIDES)
    for role, channels in (("bbox", 4), ("objectness_logit", 1), ("class_logit", CLASS_COUNT))
}
OUTPUTS = list(OUTPUT_SHAPES)


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported YOLOX graph: {message}")


def _shapes(values):
    return {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in values}


def _prune(model):
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
    from onnxsim import simplify

    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    _require(_shapes(model.graph.input) == {INPUT: INPUT_SHAPE}, f"inputs {_shapes(model.graph.input)}")
    _require(_shapes(model.graph.output) == SOURCE_OUTPUT_SHAPES, f"outputs {_shapes(model.graph.output)}")
    # Prediction weights are stable upstream names; derive their Conv outputs instead of guessing node indices.
    for output, weight in HEADS.items():
        candidates = [node for node in model.graph.node if node.op_type == "Conv" and node.input[1] == weight]
        _require(len(candidates) == 1, f"expected one prediction Conv using {weight}")
        old = candidates[0].output[0]
        for node in model.graph.node:
            for names in (node.input, node.output):
                for index, name in enumerate(names):
                    if name == old:
                        names[index] = output
    del model.graph.output[:]
    model.graph.output.extend(
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in OUTPUT_SHAPES.items()
    )
    _prune(model)
    del model.graph.value_info[:]
    # Fold static Slice/Resize controls without changing the backbone's data-dependent computation.
    model, valid = simplify(model, perform_optimization=False, overwrite_input_shapes={INPUT: INPUT_SHAPE})
    _require(valid, "onnxsim could not validate constant folding")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    _require(_shapes(model.graph.output) == OUTPUT_SHAPES, "unexpected raw-head shapes")
    output = output_dir / "yolox_x_heads.onnx"
    onnx.save(model, output)
    return output
