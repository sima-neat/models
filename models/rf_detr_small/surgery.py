"""Split RF-DETR around TopK and optimize each partition's attention layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from attention import optimize_decoder_attention, optimize_window_attention

INPUT = "input"
FEATURES = "backbone_features"
SCORES = "proposal_scores"
BOXES = "proposal_boxes"
PROPOSALS = "proposals"
OUTPUTS = ["dets", "labels"]

BOUNDARY_SHAPES = {
    FEATURES: [1, 256, 32, 32],
    SCORES: [1, 1024],
    BOXES: [1, 1024, 4],
    PROPOSALS: [1, 300, 4],
}

_FEATURES = "/backbone/backbone.0/projector/stages.0/stages.0.1/Transpose_1_output_0"
_SCORES = "/transformer/ReduceMax_output_0"
_BOXES = "/transformer/Concat_5_output_0"
_PROPOSALS = "/transformer/GatherElements_1_output_0"


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _validate_source(model: onnx.ModelProto) -> None:
    inputs = {value.name: _shape(value) for value in model.graph.input}
    outputs = {value.name: _shape(value) for value in model.graph.output}
    if inputs != {INPUT: [1, 3, 512, 512]}:
        raise ValueError(f"unexpected RF-DETR input interface: {inputs}")
    expected_outputs = {"dets": [1, 300, 4], "labels": [1, 300, 91]}
    if outputs != expected_outputs:
        raise ValueError(f"unexpected RF-DETR output interface: {outputs}")


def _node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    matches = [node for node in model.graph.node if node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one ONNX node named {name!r}, found {len(matches)}")
    return matches[0]


def _rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for values in (node.input, node.output):
            for index, value in enumerate(values):
                if value == old:
                    values[index] = new
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name == old:
            value.name = new


def _rewrite_proposals(model: onnx.ModelProto) -> None:
    normalized = "/transformer/enc_output_norm.0/LayerNormalization_output_0"
    anchors = "/transformer/Cast_4_output_0"
    gathered = _node(model, "/transformer/GatherElements_1")
    box_mlp = _node(model, "/transformer/enc_out_bbox_embed.0/layers.0/MatMul")
    first_slice = _node(model, "/transformer/Slice_2")
    second_slice = _node(model, "/transformer/Slice_3")
    decoder_input = _node(model, "/transformer/Concat_6")

    box_mlp.input[0] = normalized
    first_slice.input[0] = anchors
    second_slice.input[0] = anchors
    gathered.input[0] = _BOXES
    decoder_input.input[0] = _PROPOSALS
    model.graph.node.remove(gathered)
    model.graph.node.insert(list(model.graph.node).index(decoder_input), gathered)


def _set_value_info(model: onnx.ModelProto, name: str, shape: list[int]) -> None:
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    for value in values:
        if value.name == name:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            del value.type.tensor_type.shape.dim[:]
            for size in shape:
                value.type.tensor_type.shape.dim.add(dim_value=size)
            return
    model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))


def prepare_model(source: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    from onnxsim import simplify

    output_dir.mkdir(parents=True, exist_ok=True)
    full = onnx.load(source)
    onnx.checker.check_model(full)
    _validate_source(full)
    _rewrite_proposals(full)
    for old, new in ((_FEATURES, FEATURES), (_SCORES, SCORES), (_BOXES, BOXES), (_PROPOSALS, PROPOSALS)):
        _rename_value(full, old, new)
    for name, shape in BOUNDARY_SHAPES.items():
        _set_value_info(full, name, shape)
    full = onnx.shape_inference.infer_shapes(full)
    _validate_source(full)
    full.ir_version = 8
    onnx.checker.check_model(full)

    full_path = output_dir / "rf_detr_small_full.onnx"
    backbone_path = output_dir / "rf_detr_small_backbone.onnx"
    transformer_path = output_dir / "rf_detr_small_transformer.onnx"
    onnx.save(full, full_path)
    onnx.utils.extract_model(str(full_path), str(backbone_path), [INPUT], [FEATURES, SCORES, BOXES])
    onnx.utils.extract_model(str(full_path), str(transformer_path), [FEATURES, PROPOSALS], OUTPUTS)

    transformer = onnx.load(transformer_path)
    transformer, valid = simplify(
        transformer,
        overwrite_input_shapes={
            FEATURES: BOUNDARY_SHAPES[FEATURES],
            PROPOSALS: BOUNDARY_SHAPES[PROPOSALS],
        },
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("RF-DETR transformer simplification failed validation")
    transformer = optimize_decoder_attention(transformer)
    transformer.ir_version = 8
    onnx.checker.check_model(transformer)
    onnx.save(transformer, transformer_path)

    backbone = optimize_window_attention(onnx.load(backbone_path))
    backbone.ir_version = 8
    onnx.checker.check_model(backbone)
    onnx.save(backbone, backbone_path)
    return full_path, backbone_path, transformer_path


def validate_float_split(
    source: Path,
    backbone: Path,
    transformer: Path,
    image: np.ndarray,
) -> None:
    import onnxruntime as ort
    from calibration import stable_topk_gather

    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    backbone_session = ort.InferenceSession(str(backbone), providers=["CPUExecutionProvider"])
    transformer_session = ort.InferenceSession(str(transformer), providers=["CPUExecutionProvider"])
    sample = image[None]
    features, scores, boxes = backbone_session.run(None, {INPUT: sample})
    proposals = stable_topk_gather(scores, boxes)
    expected = reference.run(None, {INPUT: sample})
    actual = transformer_session.run(None, {FEATURES: features, PROPOSALS: proposals})
    for name, expected_value, actual_value in zip(OUTPUTS, expected, actual, strict=True):
        if actual_value.shape != expected_value.shape or not (
            np.isfinite(expected_value).all() and np.isfinite(actual_value).all()
        ):
            raise RuntimeError(f"RF-DETR split output {name} has an invalid shape or non-finite values")
        # Gemm -> NCHW Conv preserves the projection but changes float32
        # accumulation order. This tolerance covers the validated ONNX rewrite.
        if not np.allclose(actual_value, expected_value, rtol=2e-4, atol=2e-4):
            maximum_error = float(np.max(np.abs(actual_value - expected_value)))
            raise RuntimeError(f"RF-DETR split output {name} does not match the original graph: {maximum_error}")
