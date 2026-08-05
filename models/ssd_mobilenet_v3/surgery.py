#!/usr/bin/env python3
"""Validate and publish the pinned SSD-MobileNet V3 raw-head graph."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SOURCE_SHA256 = "3b8a49dae52f4053e9197755aaa11fc227af8e234c7df19c7fe42ac847eed1c5"
UPSTREAM_ARCHIVE_SHA256 = "8dade970d957ae5cb8b47e9fcefbe124e40b38c8641e484d2ab15b81d9b5126f"
MODEL_SIZE = 320
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, MODEL_SIZE, MODEL_SIZE)
CONVOLUTION_COUNT = 98
NEGLIGIBLE_WEIGHT_FLOOR = np.float32(1e-4)
EXPECTED_STABILIZED_CHANNELS = 865
LEVELS = (
    (20, 20, 12, 273),
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


def _signature(values: list[onnx.ValueInfoProto]) -> list[tuple[str, int, tuple[int, ...]]]:
    return [
        (
            value.name,
            value.type.tensor_type.elem_type,
            tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim),
        )
        for value in values
    ]


def validate_prepared_model(model: onnx.ModelProto) -> None:
    """Reject any source whose public contract or zero-copy graph has drifted."""

    onnx.checker.check_model(model, full_check=True)
    expected_inputs = [(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    expected_outputs = [(name, TensorProto.FLOAT, shape) for name, shape in OUTPUTS]
    inputs = _signature(list(model.graph.input))
    outputs = _signature(list(model.graph.output))
    if inputs != expected_inputs:
        raise ValueError(f"prepared SSD input mismatch: {inputs}")
    if outputs != expected_outputs:
        raise ValueError(f"prepared SSD output mismatch: {outputs}; expected {expected_outputs}")

    forbidden = sorted(
        {node.op_type for node in model.graph.node if node.op_type in {"Identity", "BatchNormalization"}}
    )
    if forbidden:
        raise ValueError(f"prepared SSD graph contains forbidden runtime stages: {forbidden}")
    convolution_count = sum(node.op_type == "Conv" for node in model.graph.node)
    if convolution_count != CONVOLUTION_COUNT:
        raise ValueError(f"prepared SSD convolution count mismatch: {convolution_count}")
    default_opsets = [item.version for item in model.opset_import if not item.domain]
    if default_opsets != [12]:
        raise ValueError(f"prepared SSD ONNX opset mismatch: {default_opsets}")
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get("source.archive.sha256") != UPSTREAM_ARCHIVE_SHA256:
        raise ValueError("prepared SSD graph is not tied to the pinned TensorFlow source archive")
    if metadata.get("sima.output_order") != "loc[0..5],confidence[0..5]":
        raise ValueError("prepared SSD output-order provenance is missing")


def _stabilize_negligible_weight_channels(
    model: onnx.ModelProto,
    floor: np.float32 = NEGLIGIBLE_WEIGHT_FLOOR,
) -> int:
    """Keep trained nonzero channels representable without changing topology."""

    changed = 0
    for index, initializer in enumerate(model.graph.initializer):
        array = numpy_helper.to_array(initializer)
        if array.dtype != np.float32 or array.ndim < 2 or "weight" not in initializer.name.lower():
            continue
        flattened = array.reshape(array.shape[0], -1)
        maxima = np.max(np.abs(flattened), axis=1)
        channels = np.flatnonzero((maxima > 0.0) & (maxima < floor))
        if not channels.size:
            continue
        updated = array.copy()
        updated_flattened = updated.reshape(updated.shape[0], -1)
        for channel in channels:
            position = int(np.argmax(np.abs(updated_flattened[channel])))
            updated_flattened[channel, position] = np.copysign(
                floor,
                updated_flattened[channel, position],
            )
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(updated, initializer.name))
        changed += int(channels.size)
    return changed


def validate_int8_prepared_model(model: onnx.ModelProto) -> None:
    """Prove the INT8-only stabilization contract and public graph contract."""

    validate_prepared_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get("sima.int8.stabilized_weight_channels") != str(EXPECTED_STABILIZED_CHANNELS):
        raise ValueError("prepared SSD INT8 stabilization count is missing or incorrect")
    if metadata.get("sima.int8.negligible_weight_floor") != format(float(NEGLIGIBLE_WEIGHT_FLOOR), ".9g"):
        raise ValueError("prepared SSD INT8 stabilization floor is missing or incorrect")
    remaining = 0
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.dtype != np.float32 or array.ndim < 2 or "weight" not in initializer.name.lower():
            continue
        maxima = np.max(np.abs(array.reshape(array.shape[0], -1)), axis=1)
        remaining += int(np.count_nonzero((maxima > 0.0) & (maxima < NEGLIGIBLE_WEIGHT_FLOOR)))
    if remaining:
        raise ValueError(f"prepared SSD still contains {remaining} negligible nonzero weight channels")


def prepare_model(source: Path, destination: Path) -> Path:
    """Perform deterministic, zero-cost surgery by validating and copying bytes."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(f"SSD source SHA-256 mismatch: {source}")
    validate_prepared_model(onnx.load(source, load_external_data=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(destination) != SOURCE_SHA256:
        raise RuntimeError("prepared SSD copy is not byte-for-byte reproducible")
    return destination


def prepare_int8_model(source: Path, destination: Path) -> Path:
    """Prepare the pinned graph for W8A8 PTQ without adding runtime stages."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(f"SSD source SHA-256 mismatch: {source}")
    model = onnx.load(source, load_external_data=False)
    validate_prepared_model(model)
    stabilized = _stabilize_negligible_weight_channels(model)
    if stabilized != EXPECTED_STABILIZED_CHANNELS:
        raise ValueError(
            f"SSD INT8 stabilization count mismatch: {stabilized}; expected {EXPECTED_STABILIZED_CHANNELS}"
        )
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update(
        {
            "sima.int8.negligible_weight_floor": format(float(NEGLIGIBLE_WEIGHT_FLOOR), ".9g"),
            "sima.int8.stabilized_weight_channels": str(stabilized),
        }
    )
    helper.set_model_props(model, metadata)
    validate_int8_prepared_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    validate_int8_prepared_model(onnx.load(destination, load_external_data=False))
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
