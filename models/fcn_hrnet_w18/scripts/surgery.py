"""Cut the converted FCN-HRNet graph back to the decode head's logits, before the resize to the 512x512 input.

convert.py exports MMSegmentation's inference forward, which ends by resizing the decode head's 128x128 logits
bilinearly to the 512x512 input. The package returns the 128x128 logits instead: an application resizes them once to
each image's own size, as accuracy.py does, and the 512x512 copy would only make the MPK's output 16 times larger.
Removing that final Resize leaves every other operation as exported, so the cut graph's logits are the decode head's
exactly.
"""

from pathlib import Path

import onnx
from onnx import helper

INPUT = "image"
OUTPUT = "logits"
IMAGE_SIZE = 512
CLASS_COUNT = 150
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
SOURCE_LOGITS_SHAPE = [1, CLASS_COUNT, IMAGE_SIZE, IMAGE_SIZE]
LOGITS_SHAPE = [1, CLASS_COUNT, IMAGE_SIZE // 4, IMAGE_SIZE // 4]


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported FCN-HRNet graph: {message}")


def _shapes(values):
    return {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in values}


def prepare_model(source: Path, output_dir: Path) -> Path:
    """Write the converted graph without its final Resize to output_dir and return the file's path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(source), strict_mode=True)
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {value.name for value in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializers]
    _require(_shapes(inputs) == {INPUT: INPUT_SHAPE}, f"inputs {_shapes(inputs)}")
    _require(_shapes(graph.output) == {OUTPUT: SOURCE_LOGITS_SHAPE}, f"outputs {_shapes(graph.output)}")
    finals = [index for index, node in enumerate(graph.node) if OUTPUT in node.output]
    _require(len(finals) == 1 and graph.node[finals[0]].op_type == "Resize", "the logits do not come from one Resize")
    resize = graph.node[finals[0]]
    modes = {attribute.name: attribute.s.decode() for attribute in resize.attribute if attribute.s}
    _require(
        modes.get("mode") == "linear" and modes.get("coordinate_transformation_mode") == "half_pixel",
        f"the final Resize is {modes}",
    )
    head = resize.input[0]
    _require(_shapes(graph.value_info).get(head) == LOGITS_SHAPE, f"the final Resize reads {head}")
    readers = [node.op_type for node in graph.node if head in node.input]
    _require(readers == ["Resize"], f"the decode head's logits also feed {readers}")
    del graph.node[finals[0]]
    for node in graph.node:
        node.output[:] = [OUTPUT if name == head else name for name in node.output]
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(OUTPUT, onnx.TensorProto.FLOAT, LOGITS_SHAPE))
    used = {name for node in graph.node for name in node.input}
    kept_initializers = [value for value in graph.initializer if value.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    _require(_shapes(model.graph.output) == {OUTPUT: LOGITS_SHAPE}, "unexpected logits shape after the cut")
    output = output_dir / "fcn_hrnet_logits.onnx"
    onnx.save(model, output)
    return output
