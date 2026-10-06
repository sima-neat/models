#!/usr/bin/env python3
"""Give the pinned SSD-MobileNet V2 raw-head graph a stable public interface."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto, helper

SOURCE_SHA256 = "f0d9458cd8ae5c0cde8eadf22b75c6451059f4d9dd46df266e996dcbec417a8f"
MODEL_SIZE = 300
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, MODEL_SIZE, MODEL_SIZE)
SOURCE_INPUT = "FeatureExtractor/MobilenetV2/MobilenetV2/input:0"
EXPECTED_CONVOLUTION_COUNT = 76
EXPECTED_OPERATOR_COUNTS = Counter({"Conv": 76, "Clip": 47, "Add": 31, "Mul": 21})
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
SOURCE_OUTPUTS = tuple(
    (
        f"BoxPredictor_{level}/{source_role}/BiasAdd:0",
        (1, channels, height, width),
    )
    for level, (height, width, loc_channels, confidence_channels) in enumerate(LEVELS)
    for source_role, channels in (
        ("BoxEncodingPredictor", loc_channels),
        ("ClassPredictor", confidence_channels),
    )
)
SOURCE_OUTPUT_ORDER = tuple(
    f"BoxPredictor_{level}/{source_role}/BiasAdd:0"
    for source_role in ("BoxEncodingPredictor", "ClassPredictor")
    for level in range(len(LEVELS))
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_signature(value: onnx.ValueInfoProto) -> tuple[str, int, tuple[int, ...]]:
    tensor = value.type.tensor_type
    return value.name, tensor.elem_type, tuple(dimension.dim_value for dimension in tensor.shape.dim)


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


def validate_source_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    expected_input = [(SOURCE_INPUT, TensorProto.FLOAT, INPUT_SHAPE)]
    actual_inputs = [_value_signature(value) for value in model.graph.input]
    if actual_inputs != expected_input:
        raise ValueError(f"SSD-MobileNet V2 source input mismatch: {actual_inputs}")
    expected_outputs = [(name, TensorProto.FLOAT, shape) for name, shape in SOURCE_OUTPUTS]
    actual_outputs = [_value_signature(value) for value in model.graph.output]
    if actual_outputs != expected_outputs:
        raise ValueError(f"SSD-MobileNet V2 source outputs mismatch: {actual_outputs}")
    default_opsets = [entry.version for entry in model.opset_import if entry.domain in {"", "ai.onnx"}]
    if default_opsets != [13]:
        raise ValueError(f"SSD-MobileNet V2 source must use ONNX opset 13, found {default_opsets}")
    operator_counts = Counter(node.op_type for node in model.graph.node)
    if operator_counts != EXPECTED_OPERATOR_COUNTS:
        raise ValueError(f"SSD-MobileNet V2 source operator mismatch: {operator_counts}")


def validate_prepared_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    inputs = [_value_signature(value) for value in model.graph.input]
    outputs = [_value_signature(value) for value in model.graph.output]
    expected_input = [(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    expected_outputs = [(name, TensorProto.FLOAT, shape) for name, shape in OUTPUTS]
    if inputs != expected_input:
        raise ValueError(f"prepared SSD input mismatch: {inputs}")
    if outputs != expected_outputs:
        raise ValueError(f"prepared SSD outputs mismatch: {outputs}; expected {expected_outputs}")
    if any(node.op_type == "Identity" for node in model.graph.node):
        raise ValueError("prepared SSD graph must not add output Identity nodes")
    operator_counts = Counter(node.op_type for node in model.graph.node)
    if operator_counts != EXPECTED_OPERATOR_COUNTS:
        raise ValueError(f"prepared SSD graph changed source operators: {operator_counts}")


def prepare_model(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(f"SSD source SHA-256 mismatch: {source}")

    prepared = onnx.load(source)
    validate_source_model(prepared)
    _rename_value(prepared, SOURCE_INPUT, INPUT_NAME)

    stable_name_by_source = dict(zip(SOURCE_OUTPUT_ORDER, (name for name, _ in OUTPUTS), strict=True))
    for source_name, stable_name in stable_name_by_source.items():
        # Renaming the producer output is deliberately zero-cost. An Identity per
        # output would force twelve unnecessary public-output stages.
        _rename_value(prepared, source_name, stable_name)
    output_by_name = {value.name: value for value in prepared.graph.output}
    ordered_outputs = [output_by_name[name] for name, _ in OUTPUTS]
    del prepared.graph.output[:]
    prepared.graph.output.extend(ordered_outputs)

    prepared.graph.name = "ssd_mobilenet_v2_heads"
    prepared.producer_name = "sima-neat-ssd-head-surgery"
    prepared.producer_version = "1"
    del prepared.metadata_props[:]
    helper.set_model_props(
        prepared,
        {
            "sima.output_order": "loc[0..5],confidence[0..5]",
            "sima.preprocess": "RGB float32 NCHW [-1,1], stretch 300x300",
            "sima.recipe": "ssd-mobile-300-v2",
            "source.sha256": SOURCE_SHA256,
        },
    )
    validate_prepared_model(prepared)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(prepared, destination)
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
