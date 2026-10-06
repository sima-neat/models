#!/usr/bin/env python3
"""Extract the six raw SSD-MobileNet V1 box and class predictor heads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference
from onnx.utils import Extractor

SOURCE_SHA256 = "b8fba5e404077d4048d27fcd1667e85e27e192eb9bf51e696c46a3acd7d21058"
MODEL_SIZE = 300
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, MODEL_SIZE, MODEL_SIZE)
BOUNDARY_INPUT = "Preprocessor/sub:0"
LEVELS = (
    (19, 19, 12, 273),
    (10, 10, 24, 546),
    (5, 5, 24, 546),
    (3, 3, 24, 546),
    (2, 2, 24, 546),
    (1, 1, 24, 546),
)
OUTPUTS = tuple(
    (f"ssd_{role}_{level}", (1, channels, height, width))
    for role, channel_index in (("loc", 2), ("confidence", 3))
    for level, (height, width, loc_channels, confidence_channels) in enumerate(LEVELS)
    for channels in ((loc_channels, confidence_channels)[channel_index - 2],)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_predictor_output(model: onnx.ModelProto, level: int, role: str) -> str:
    prefix = f"BoxPredictor_{level}/{role}/BiasAdd"
    candidates: list[tuple[str, bool]] = []
    for node in model.graph.node:
        if node.name.startswith(prefix):
            candidates.extend((output, node.op_type == "Transpose") for output in node.output)

    convolution_outputs = [name for name, is_nhwc in candidates if not is_nhwc]
    if len(convolution_outputs) == 1:
        return convolution_outputs[0]

    nhwc_outputs = [name for name, is_nhwc in candidates if is_nhwc]
    if len(nhwc_outputs) == 1:
        producer = next(node for node in model.graph.node if nhwc_outputs[0] in node.output)
        permutation = next((attribute.ints for attribute in producer.attribute if attribute.name == "perm"), ())
        if producer.op_type == "Transpose" and list(permutation) == [0, 2, 3, 1]:
            return producer.input[0]
    raise ValueError(f"expected one raw predictor output for {prefix}, found {candidates}")


def _rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            if value.name == old:
                value.name = new


def validate_prepared_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    inputs = [
        (value.name, tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)) for value in model.graph.input
    ]
    outputs = [
        (value.name, tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)) for value in model.graph.output
    ]
    if inputs != [(INPUT_NAME, INPUT_SHAPE)]:
        raise ValueError(f"prepared SSD input mismatch: {inputs}")
    if outputs != list(OUTPUTS):
        raise ValueError(f"prepared SSD output mismatch: {outputs}; expected {list(OUTPUTS)}")
    if any(node.op_type == "Identity" for node in model.graph.node):
        raise ValueError("prepared SSD graph must not add output Identity nodes")


def prepare_model(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(f"SSD source SHA-256 mismatch: {source}")

    model = shape_inference.infer_shapes(onnx.load(source), strict_mode=False, data_prop=True)
    value_info = next((value for value in model.graph.value_info if value.name == BOUNDARY_INPUT), None)
    if value_info is None:
        model.graph.value_info.append(helper.make_tensor_value_info(BOUNDARY_INPUT, TensorProto.FLOAT, INPUT_SHAPE))
    else:
        boundary_shape = tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)
        if boundary_shape not in {INPUT_SHAPE, (0, 0, 0, 0)}:
            raise ValueError(f"SSD normalized boundary shape mismatch: {boundary_shape}")
        value_info.CopyFrom(helper.make_tensor_value_info(BOUNDARY_INPUT, TensorProto.FLOAT, INPUT_SHAPE))
    source_outputs = [
        _find_predictor_output(model, level, role)
        for role in ("BoxEncodingPredictor", "ClassPredictor")
        for level in range(len(LEVELS))
    ]
    extracted = Extractor(model).extract_model([BOUNDARY_INPUT], source_outputs)
    _rename_value(extracted, BOUNDARY_INPUT, INPUT_NAME)
    for source_name, (stable_name, _) in zip(source_outputs, OUTPUTS, strict=True):
        # Renaming the producer output is deliberately zero-cost. An Identity per
        # output forces twelve otherwise unnecessary A65 publication stages.
        _rename_value(extracted, source_name, stable_name)

    extracted.graph.name = "ssd_mobilenet_v1_heads"
    extracted.ir_version = 8
    extracted.producer_name = "sima-neat-ssd-head-surgery"
    extracted.producer_version = "1"
    del extracted.metadata_props[:]
    helper.set_model_props(
        extracted,
        {
            "sima.output_order": "loc[0..5],confidence[0..5]",
            "sima.preprocess": "RGB float32 NCHW [-1,1], stretch 300x300",
            "sima.recipe": "ssd-mobile-300-v1",
            "source.sha256": SOURCE_SHA256,
        },
    )
    extracted = shape_inference.infer_shapes(extracted, strict_mode=True, data_prop=True)
    validate_prepared_model(extracted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(extracted, destination)
    return destination


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_model(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
