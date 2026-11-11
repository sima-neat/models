"""Cut the complete TorchVision source at normalized input and twelve raw heads.

Only ONNX/compiler-declared dependencies are required. The original source owns
preprocessing, decode and NMS; the native pipeline implements those removed steps.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import onnx

SOURCE_INPUT = "image"
WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, 320, 320)
FEATURE_SIZES = (20, 10, 5, 3, 2, 1)
OUTPUTS = tuple(
    (f"ssd_{role}_{level}", (1, 6 * columns, size, size))
    for role, columns in (("loc", 4), ("confidence", 91))
    for level, size in enumerate(FEATURE_SIZES)
)

HEADS = tuple(
    f"/detector/head/{role}_head/module_list.{level}/module_list.{level}.1/Conv_output_0"
    for role in ("regression", "classification")
    for level in range(6)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path: Path) -> dict:
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    operators = Counter(node.op_type for node in graph.graph.node)
    if (
        operators["Conv"] != 98
        or operators["BatchNormalization"]
        or operators["QuantizeLinear"]
        or operators["DequantizeLinear"]
    ):
        raise ValueError(f"unexpected FP32 operator contract: {operators}")
    if [(item.domain, item.version) for item in graph.opset_import] != [("", 17)]:
        raise ValueError("unexpected ONNX opset")

    def contract(values):
        return [
            (
                item.name,
                tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim),
                item.type.tensor_type.elem_type,
            )
            for item in values
        ]

    if contract(graph.graph.input) != [(INPUT_NAME, INPUT_SHAPE, onnx.TensorProto.FLOAT)]:
        raise ValueError("unexpected public input contract")
    if contract(graph.graph.output) != [(name, shape, onnx.TensorProto.FLOAT) for name, shape in OUTPUTS]:
        raise ValueError("unexpected ordered public output contract")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("source must be one self-contained ONNX file")
    return {
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "operators": dict(sorted(operators.items())),
    }


def audit_source(path: Path) -> dict:
    """Require the full upstream graph; conversion receipts separately bind its bytes."""
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    operators = Counter(node.op_type for node in graph.graph.node)

    props = {item.key: item.value for item in graph.metadata_props}
    if props.get("sima.torchvision.weights_sha256") != WEIGHTS_SHA256:
        raise ValueError("unexpected source weights provenance")

    def nested_nodes(graph):
        for node in graph.node:
            yield node
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from nested_nodes(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        yield from nested_nodes(subgraph)

    if sum(node.op_type == "NonMaxSuppression" for node in nested_nodes(graph.graph)) != 1:
        raise ValueError("full source must retain the original NMS")
    if operators["Conv"] != 98 or operators["BatchNormalization"] != 70 or operators["Softmax"] != 1:
        raise ValueError(f"unexpected full-source operator contract: {operators}")
    if operators["QuantizeLinear"] or operators["DequantizeLinear"]:
        raise ValueError("expected an FP32 source, not QAT")
    if [(item.domain, item.version) for item in graph.opset_import] != [("", 17)]:
        raise ValueError("unexpected source ONNX opset")
    inputs = graph.graph.input
    if (
        len(inputs) != 1
        or inputs[0].name != SOURCE_INPUT
        or (
            tuple(dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim) != INPUT_SHAPE
            or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        )
    ):
        raise ValueError("unexpected full-source input contract")
    outputs = graph.graph.output
    if [value.name for value in outputs] != ["boxes", "scores", "labels"]:
        raise ValueError("unexpected full-source output order")
    for value, rank, dtype in zip(
        outputs, (2, 1, 1), (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT, onnx.TensorProto.INT64), strict=True
    ):
        tensor = value.type.tensor_type
        if tensor.elem_type != dtype or len(tensor.shape.dim) != rank:
            raise ValueError("unexpected full-source output type/shape")
        if not tensor.shape.dim[0].dim_param:
            raise ValueError("source detection count must remain dynamic")
    if outputs[0].type.tensor_type.shape.dim[1].dim_value != 4:
        raise ValueError("source boxes must contain four coordinates")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("source must be self-contained")
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size, "operators": dict(sorted(operators.items()))}


def cut_raw_heads(graph: onnx.ModelProto) -> onnx.ModelProto:
    """Extract exactly the declared normalized-input/raw-head boundary."""
    convolutions = [node for node in graph.graph.node if node.op_type == "Conv"]
    # Scope the input cut to the pinned backbone, never to an arbitrary first Conv.
    first = [
        node for node in convolutions if node.name == "/detector/backbone/features.0/features.0.0/features.0.0.0/Conv"
    ]
    if len(first) != 1:
        raise ValueError("missing unique TorchVision backbone input cut")
    entry = first[0].input[0]
    cuts = list(HEADS)
    for name in cuts:
        matches = [node for node in convolutions if name in node.output]
        if len(matches) != 1:
            raise ValueError(f"missing unique raw-head Conv: {name}")
    infos = {entry: INPUT_SHAPE, **{name: shape for name, (_, shape) in zip(cuts, OUTPUTS, strict=True)}}
    # Replace inferred shapes at the explicit boundary with its fixed contract.
    kept = [value for value in graph.graph.value_info if value.name not in infos]
    del graph.graph.value_info[:]
    graph.graph.value_info.extend(kept)
    graph.graph.value_info.extend(
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in infos.items()
    )
    raw = onnx.utils.Extractor(graph).extract_model([entry], cuts)
    rename = {entry: INPUT_NAME, **{old: name for old, (name, _) in zip(cuts, OUTPUTS, strict=True)}}
    for node in raw.graph.node:
        for values in (node.input, node.output):
            for index, value in enumerate(values):
                if value in rename:
                    values[index] = rename[value]
    for values in (raw.graph.input, raw.graph.output, raw.graph.value_info):
        for value in values:
            if value.name in rename:
                value.name = rename[value.name]
    return raw


def prepare_model(source: Path, output: Path) -> Path:
    from onnxsim import simplify

    audit_source(source)
    raw = cut_raw_heads(onnx.load(source))
    raw, valid = simplify(raw)
    if not valid:
        raise ValueError("onnxsim could not validate SSD BatchNorm/constant folding")
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(raw, output)
    audit(output)
    return output
