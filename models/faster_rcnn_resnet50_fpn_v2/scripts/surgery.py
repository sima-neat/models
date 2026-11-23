#!/usr/bin/env python3
"""Cut torchvision's Faster R-CNN ResNet-50 FPN v2 ONNX export into its two MLA components.

The export runs the whole detector, preprocessing and NMS included, at a fixed 224x224 input. surgery.py keeps
only its two convolutional networks and leaves the steps between them to the CPU:

- backbone: the normalized, padded 1344x1344 image -> the FPN maps P2-P5 and the RPN head's objectness and
  box deltas on P2-P6, as one output (calibration.py documents its layout). ResNet-50, the FPN and the RPN head,
  folded to this one input shape. With several outputs the Model Compiler gives some their own output sections
  and packs the rest into a shared one, and Neat refuses such an ELF; one output always has one section.
- box_head: torchvision's per-region box head (four 3x3 convs, one fully connected layer and the class and box
  predictors), rewritten to run all pooled regions as one image. The regions' 7x7 maps sit in a grid of 25 rows
  at an 8-pixel pitch with zero gaps, 25x40 for torchvision's 1,000 proposals; each 3x3 conv's output is masked
  back to zero in the gaps, so no region sees its neighbours, and the fully connected layers become 7x7 stride-8
  and 1x1 convs. Its outputs hold one region per grid cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from calibration import (
    ANCHORS,
    BACKBONE_CHANNELS,
    CHANNELS,
    CLASSES,
    DELTA_GAIN,
    GRID_ROWS,
    INPUT_SIZE,
    LEVEL_ROWS,
    LEVEL_SIZES,
    OBJECTNESS_GAIN,
    PITCH,
    POOL,
    POST_NMS_TOP_N,
    RPN_CHANNELS,
    grid_cols,
    region_rows,
    tile_regions,
    unpack_backbone,
)
from onnx import TensorProto, helper, numpy_helper
from onnx.utils import Extractor

INPUT = "image"
FEATURES = tuple(f"p{level}" for level in range(2, 6))
OBJECTNESS = tuple(f"objectness_p{level}" for level in range(2, 7))
BOX_DELTAS = tuple(f"box_deltas_p{level}" for level in range(2, 7))
LEVEL_OUTPUTS = FEATURES + OBJECTNESS + BOX_DELTAS
BACKBONE_OUTPUT = "backbone"
ROIS = "rois"
CLASS_LOGITS = "class_logits"
BOX_REGRESSION = "box_regression"
BOX_HEAD_OUTPUTS = (CLASS_LOGITS, BOX_REGRESSION)


def _stride_size(stride: int) -> int:
    return INPUT_SIZE // stride


LEVEL_SHAPES = {
    **{name: (1, CHANNELS, _stride_size(4 << i), _stride_size(4 << i)) for i, name in enumerate(FEATURES)},
    **{name: (1, ANCHORS, _stride_size(4 << i), _stride_size(4 << i)) for i, name in enumerate(OBJECTNESS)},
    **{name: (1, 4 * ANCHORS, _stride_size(4 << i), _stride_size(4 << i)) for i, name in enumerate(BOX_DELTAS)},
}
BACKBONE_SHAPE = (1, BACKBONE_CHANNELS, 1, LEVEL_ROWS[-1])


def rois_shape(regions: int) -> tuple[int, int, int, int]:
    """The box head's NCHW input for a number of proposals."""
    return 1, CHANNELS, GRID_ROWS * PITCH, grid_cols(regions) * PITCH


def box_head_shapes(regions: int) -> dict[str, tuple[int, int, int, int]]:
    """The box head's NCHW outputs for a number of proposals: one region per grid cell."""
    cols = grid_cols(regions)
    return {CLASS_LOGITS: (1, CLASSES, GRID_ROWS, cols), BOX_REGRESSION: (1, 4 * CLASSES, GRID_ROWS, cols)}


# Tensors of the export at the cut: the normalized, padded image entering ResNet-50, the FPN outputs, the RPN
# head's per-level outputs and the pooled regions entering the box head.
SOURCE_IMAGE = "/transform/Concat_8_output_0"
SOURCE_FEATURES = tuple(f"/backbone/fpn/layer_blocks.{i}/layer_blocks.{i}.0/Conv_output_0" for i in range(4))
_LEVEL_SUFFIXES = ("", "_1", "_2", "_3", "_4")
SOURCE_OBJECTNESS = tuple(f"/rpn/head/cls_logits{suffix}/Conv_output_0" for suffix in _LEVEL_SUFFIXES)
SOURCE_BOX_DELTAS = tuple(f"/rpn/head/bbox_pred{suffix}/Conv_output_0" for suffix in _LEVEL_SUFFIXES)
SOURCE_ROIS = "/roi_heads/box_roi_pool/ScatterElements_3_output_0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unexpected Faster R-CNN graph: {message}")


def _rename(model: onnx.ModelProto, names: dict[str, str]) -> None:
    for node in model.graph.node:
        node.input[:] = [names.get(name, name) for name in node.input]
        node.output[:] = [names.get(name, name) for name in node.output]
    for value in (*model.graph.input, *model.graph.output):
        value.name = names.get(value.name, value.name)


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)


def backbone_graph(source: onnx.ModelProto) -> onnx.ModelProto:
    """ResNet-50, the FPN and the RPN head from the padded image to the declared outputs, at 1344x1344."""
    from onnxsim import simplify

    sources = SOURCE_FEATURES + SOURCE_OBJECTNESS + SOURCE_BOX_DELTAS
    model = Extractor(source).extract_model([SOURCE_IMAGE], list(sources))
    # Drop the 224x224 export's shapes; onnxsim infers them again from the 1344x1344 input.
    del model.graph.value_info[:]
    for value in (*model.graph.input, *model.graph.output):
        for axis, dim in enumerate(value.type.tensor_type.shape.dim):
            dim.dim_param = f"{value.name}_{axis}"
    model, valid = simplify(model, overwrite_input_shapes={SOURCE_IMAGE: [1, 3, INPUT_SIZE, INPUT_SIZE]})
    _require(valid, "onnxsim could not validate the folded backbone")
    _rename(model, {SOURCE_IMAGE: INPUT, **dict(zip(sources, LEVEL_OUTPUTS, strict=True))})
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {value.name: _shape(value) for value in model.graph.output}
    _require(shapes == LEVEL_SHAPES, f"backbone outputs {shapes}")
    operators = {node.op_type for node in model.graph.node}
    _require(operators <= {"Conv", "Relu", "Add", "MaxPool", "Resize"}, f"backbone operators {sorted(operators)}")
    return _join_outputs(model)


def _join_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Join the 14 level outputs into one (1, 271, 1, R): each level's cells as rows, features then RPN head."""
    graph = model.graph
    nodes = []

    def constant(name: str, value: np.ndarray) -> str:
        graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    def rows(tensor: str, channels: int, size: int) -> str:
        shape = constant(f"{tensor}_rows_shape", np.array([1, channels, 1, size * size], np.int64))
        nodes.append(helper.make_node("Reshape", [tensor, shape], [f"{tensor}_rows"], f"{tensor}_rows"))
        return f"{tensor}_rows"

    # P6 has no feature map; its rows hold zeros in the feature channels.
    padding = constant("p6_features", np.zeros((1, CHANNELS, 1, LEVEL_SIZES[4] ** 2), np.float32))
    features = [rows(name, CHANNELS, size) for name, size in zip(FEATURES, LEVEL_SIZES, strict=False)] + [padding]
    nodes.append(helper.make_node("Concat", features, ["features"], "features", axis=3))
    objectness_gain = constant("objectness_gain", np.array(OBJECTNESS_GAIN, np.float32))
    delta_gain = constant("delta_gain", np.array(DELTA_GAIN, np.float32))
    rpn = []
    for objectness, deltas, size in zip(OBJECTNESS, BOX_DELTAS, LEVEL_SIZES, strict=True):
        nodes += [
            helper.make_node("Mul", [objectness, objectness_gain], [f"{objectness}_scaled"], f"{objectness}_scaled"),
            helper.make_node("Mul", [deltas, delta_gain], [f"{deltas}_scaled"], f"{deltas}_scaled"),
            helper.make_node(
                "Concat",
                [f"{objectness}_scaled", f"{deltas}_scaled"],
                [f"{objectness}_rpn"],
                f"{objectness}_rpn",
                axis=1,
            ),
        ]
        rpn.append(rows(f"{objectness}_rpn", RPN_CHANNELS, size))
    nodes += [
        helper.make_node("Concat", rpn, ["rpn"], "rpn", axis=3),
        helper.make_node("Concat", ["features", "rpn"], [BACKBONE_OUTPUT], BACKBONE_OUTPUT, axis=1),
    ]
    graph.node.extend(nodes)
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(BACKBONE_OUTPUT, TensorProto.FLOAT, list(BACKBONE_SHAPE)))
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def gap_mask(regions: int = POST_NMS_TOP_N) -> np.ndarray:
    """1 on each region's 7x7 cells of the grid, 0 in the one-pixel gaps between them."""
    row = np.arange(GRID_ROWS * PITCH) % PITCH < POOL
    column = np.arange(grid_cols(regions) * PITCH) % PITCH < POOL
    return (row[:, None] & column[None, :]).astype(np.float32)[None, None]


def box_head_graph(source: onnx.ModelProto, regions: int = POST_NMS_TOP_N) -> onnx.ModelProto:
    """The box head and predictor on all regions at once, as one image of regions, 200x320 for 1,000."""
    initializers = {item.name: numpy_helper.to_array(item) for item in source.graph.initializer}
    nodes = [
        node
        for node in source.graph.node
        if node.name.startswith(("/roi_heads/box_head/", "/roi_heads/box_predictor/"))
    ]
    convs = [node for node in nodes if node.op_type == "Conv"]
    gemms = [node for node in nodes if node.op_type == "Gemm"]
    _require(len(convs) == 4 and len(gemms) == 3, f"box head of {len(convs)} convs and {len(gemms)} Gemms")
    _require(convs[0].input[0] == SOURCE_ROIS, "box head input")
    for conv in convs:
        attributes = {item.name: helper.get_attribute_value(item) for item in conv.attribute}
        _require(attributes.get("pads") == [1, 1, 1, 1] and attributes.get("strides") == [1, 1], f"{conv.name}")
    for gemm in gemms:
        attributes = {item.name: helper.get_attribute_value(item) for item in gemm.attribute}
        _require(attributes.get("transB") == 1, f"{gemm.name} layout")
    fc, classes, boxes = gemms

    weights = [numpy_helper.from_array(gap_mask(regions), "gap_mask")]
    graph_nodes = []
    tensor = ROIS
    for index, conv in enumerate(convs):
        weights += [
            numpy_helper.from_array(initializers[conv.input[1]], f"conv{index}_weight"),
            numpy_helper.from_array(initializers[conv.input[2]], f"conv{index}_bias"),
        ]
        graph_nodes += [
            helper.make_node(
                "Conv",
                [tensor, f"conv{index}_weight", f"conv{index}_bias"],
                [f"conv{index}"],
                f"conv{index}",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", [f"conv{index}"], [f"relu{index}"], f"relu{index}"),
        ]
        tensor = f"relu{index}"
        if index < len(convs) - 1:
            # The next conv's padding must see zeros between regions, as each region's own padding does.
            graph_nodes.append(helper.make_node("Mul", [tensor, "gap_mask"], [f"masked{index}"], f"masked{index}"))
            tensor = f"masked{index}"
    fc_weight = initializers[fc.input[1]]
    _require(fc_weight.shape == (1024, CHANNELS * POOL * POOL), f"fully connected weight {fc_weight.shape}")
    weights += [
        numpy_helper.from_array(fc_weight.reshape(1024, CHANNELS, POOL, POOL), "fc_weight"),
        numpy_helper.from_array(initializers[fc.input[2]], "fc_bias"),
        numpy_helper.from_array(initializers[classes.input[1]][:, :, None, None], "class_weight"),
        numpy_helper.from_array(initializers[classes.input[2]], "class_bias"),
        numpy_helper.from_array(initializers[boxes.input[1]][:, :, None, None], "box_weight"),
        numpy_helper.from_array(initializers[boxes.input[2]], "box_bias"),
    ]
    graph_nodes += [
        helper.make_node(
            "Conv", [tensor, "fc_weight", "fc_bias"], ["fc"], "fc", kernel_shape=[POOL, POOL], strides=[PITCH, PITCH]
        ),
        helper.make_node("Relu", ["fc"], ["fc_relu"], "fc_relu"),
        helper.make_node(
            "Conv", ["fc_relu", "class_weight", "class_bias"], [CLASS_LOGITS], "class_score", kernel_shape=[1, 1]
        ),
        helper.make_node(
            "Conv", ["fc_relu", "box_weight", "box_bias"], [BOX_REGRESSION], "box_prediction", kernel_shape=[1, 1]
        ),
    ]
    graph = helper.make_graph(
        graph_nodes,
        "faster_rcnn_box_head",
        [helper.make_tensor_value_info(ROIS, TensorProto.FLOAT, list(rois_shape(regions)))],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))
            for name, shape in box_head_shapes(regions).items()
        ],
        weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def graph_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "backbone.onnx", directory / "box_head.onnx"


def prepare_model(source: Path, directory: Path, regions: int = POST_NMS_TOP_N) -> tuple[Path, Path]:
    """Write both component graphs the recipe imports, the box head for a number of proposals."""
    model = onnx.load(source)
    names = {value.name for value in model.graph.input}
    _require(names == {"images"}, f"inputs {sorted(names)}")
    backbone_path, box_head_path = graph_paths(directory)
    directory.mkdir(parents=True, exist_ok=True)
    onnx.save(backbone_graph(model), backbone_path)
    onnx.save(box_head_graph(model, regions), box_head_path)
    return backbone_path, box_head_path


def max_split_difference(
    source: Path, backbone: Path, box_head: Path, regions: int = POST_NMS_TOP_N
) -> dict[str, float]:
    """Largest output difference of each component against the matching part of the source, on random inputs.

    The backbone is compared with the source's own subgraph fed the same 1344x1344 image, and the box head with
    the source's per-region head fed the same regions, as many as the box head takes.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    model = onnx.load(source)
    reference_backbone = Extractor(model).extract_model(
        [SOURCE_IMAGE], list(SOURCE_FEATURES + SOURCE_OBJECTNESS + SOURCE_BOX_DELTAS)
    )
    reference_box_head = Extractor(model).extract_model(
        [SOURCE_ROIS],
        ["/roi_heads/box_predictor/cls_score/Gemm_output_0", "/roi_heads/box_predictor/bbox_pred/Gemm_output_0"],
    )
    providers = ["CPUExecutionProvider"]

    def run(graph: onnx.ModelProto | Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        serialized = graph.SerializeToString() if isinstance(graph, onnx.ModelProto) else str(graph)
        return ort.InferenceSession(serialized, providers=providers).run(None, feeds)

    image = rng.standard_normal((1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    reference = run(reference_backbone, {SOURCE_IMAGE: image})
    features, rpn = unpack_backbone(run(backbone, {INPUT: image})[0])
    expected = [output[0].transpose(1, 2, 0) for output in reference[:4]]
    expected += [output[0].transpose(1, 2, 0).reshape(-1) for output in reference[4:9]]
    expected += [output[0].transpose(1, 2, 0).reshape(-1, 4) for output in reference[9:]]
    actual = features + [logits for logits, _ in rpn] + [deltas for _, deltas in rpn]
    pooled = np.maximum(rng.standard_normal((regions, POOL, POOL, CHANNELS)).astype(np.float32), 0)
    expected_head = run(reference_box_head, {SOURCE_ROIS: np.ascontiguousarray(pooled.transpose(0, 3, 1, 2))})
    tiles = np.ascontiguousarray(tile_regions(pooled, regions).transpose(2, 0, 1))[None]
    actual_head = [region_rows(output) for output in run(box_head, {ROIS: tiles})]
    return {
        "backbone": max(float(np.abs(a - b).max()) for a, b in zip(expected, actual, strict=True)),
        "box_head": max(float(np.abs(a - b).max()) for a, b in zip(expected_head, actual_head, strict=True)),
    }
