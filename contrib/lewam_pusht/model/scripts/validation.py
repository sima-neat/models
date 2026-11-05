"""Validate the candidate four-graph deployment interface, not numerical parity.

This adapter deliberately does not rerun an unproven generic normalization rewrite.
Front's existing direct-compiler ELF is not an ONNX-to-MPK qualification. A fresh
standard compiler build must succeed and pass the independent precision gate.
"""

from __future__ import annotations

from pathlib import Path

from model_registry.fetch import sha256_file

COMPONENTS = ("front", "body", "head", "policy")
INPUT_NAMES = {"front": "pixels", "body": "features", "head": "features", "policy": "policy_input"}
SHAPES = {
    "front": ((1, 3, 224, 224), (1, 64, 51, 51)),
    "body": ((1, 64, 51, 51), (1, 512, 7, 7)),
    "head": ((1, 512, 7, 7), (1, 192, 1, 1)),
    "policy": ((1, 192, 1, 34), (1, 192, 1, 10)),
}


def digest(path: Path) -> str:
    return sha256_file(path)[0]


def validate_graph(path: Path, component: str) -> dict:
    import onnx

    graph = onnx.load(path, load_external_data=False)
    if any(x.data_location == onnx.TensorProto.EXTERNAL for x in graph.graph.initializer):
        raise ValueError("Self-contained ONNX files required; external tensor paths are not admitted")
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError(f"{component}: one explicit input and output required")
    values = [graph.graph.input[0], graph.graph.output[0]]
    for value, shape in zip(values, SHAPES[component], strict=True):
        tensor = value.type.tensor_type
        actual = tuple(x.dim_value for x in tensor.shape.dim)
        if tensor.elem_type != onnx.TensorProto.FLOAT or actual != shape:
            raise ValueError(f"{component}/{value.name}: expected FP32 NCHW source {shape}, got {actual}")
    if component == "policy" and [x.name for x in values] != ["policy_input", "velocity_padded"]:
        raise ValueError("The original full19/full10 policy ABI is required")
    if values[0].name != INPUT_NAMES[component]:
        raise ValueError(f"{component}: expected input name {INPUT_NAMES[component]}")
    return {
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
        "input": values[0].name,
        "output": values[1].name,
        "input_shape": list(SHAPES[component][0]),
        "output_shape": list(SHAPES[component][1]),
        "operators": sorted({node.op_type for node in graph.graph.node}),
    }


def validate_sources(directory: Path) -> dict:
    return {name: validate_graph(directory / f"{name}.onnx", name) for name in COMPONENTS}
