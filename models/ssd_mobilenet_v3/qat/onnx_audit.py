"""Independent structural validation for FP32 and SiMa-QAT SSDLite ONNX."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import onnx
from contract import CONVOLUTION_COUNT, INPUT_NAME, INPUT_SHAPE, OUTPUTS
from onnx import TensorProto, numpy_helper


def sha256_file(path: Path) -> str:
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


def audit_onnx(path: Path, *, expect_qat: bool) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    expected_inputs = [(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    expected_outputs = [(name, TensorProto.FLOAT, shape) for name, shape in OUTPUTS]
    inputs = _signature(list(model.graph.input))
    outputs = _signature(list(model.graph.output))
    if inputs != expected_inputs:
        raise ValueError(f"SSDLite input contract mismatch: {inputs}")
    if outputs != expected_outputs:
        raise ValueError(f"SSDLite output contract mismatch: {outputs}; expected {expected_outputs}")
    default_opsets = [item.version for item in model.opset_import if not item.domain]
    if default_opsets != [17]:
        raise ValueError(f"SSDLite ONNX must use opset 17, found {default_opsets}")

    convolutions = [node for node in model.graph.node if node.op_type == "Conv"]
    if len(convolutions) != CONVOLUTION_COUNT:
        raise ValueError(f"expected {CONVOLUTION_COUNT} SSDLite convolutions, found {len(convolutions)}")
    forbidden = sorted({node.op_type for node in model.graph.node if node.op_type == "BatchNormalization"})
    if forbidden:
        raise ValueError(f"exported SSDLite graph contains unfused operations: {forbidden}")

    quantizers = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dequantizers = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    if not expect_qat:
        if quantizers or dequantizers:
            raise ValueError("FP32 SSDLite export unexpectedly contains QDQ operators")
        return {
            "convolution_count": len(convolutions),
            "format": "fp32",
            "onnx_sha256": sha256_file(path),
            "opset": default_opsets[0],
            "output_count": len(outputs),
        }
    if not quantizers or not dequantizers:
        raise ValueError("QAT SSDLite export contains no QDQ operators")

    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}

    def constant_value(name: str) -> Any | None:
        visited = set()
        while name not in initializers:
            if name in visited:
                return None
            visited.add(name)
            producer = producers.get(name)
            if producer is not None and producer.op_type == "Constant":
                values = [attribute.t for attribute in producer.attribute if attribute.name == "value"]
                return numpy_helper.to_array(values[0]) if len(values) == 1 else None
            if producer is None or producer.op_type != "Identity" or len(producer.input) != 1:
                return None
            name = producer.input[0]
        return initializers[name]

    per_channel_weights = 0
    zero_weight_channels = 0
    minimum_weight_scale = float("inf")
    for convolution in convolutions:
        producer = producers.get(convolution.input[1])
        if producer is None or producer.op_type != "DequantizeLinear":
            raise ValueError(f"convolution {convolution.name!r} has no quantized weight input")
        if len(producer.input) < 3:
            raise ValueError(f"convolution {convolution.name!r} weight quantizer is not constant")
        scale = constant_value(producer.input[1])
        zero_point = constant_value(producer.input[2])
        if scale is None or zero_point is None:
            raise ValueError(f"convolution {convolution.name!r} weight quantizer is not constant")
        axes = [attribute.i for attribute in producer.attribute if attribute.name == "axis"]
        if scale.ndim != 1 or scale.size <= 1 or zero_point.shape != scale.shape or axes != [0]:
            raise ValueError(f"convolution {convolution.name!r} weights are not symmetric per-output-channel QDQ")
        if zero_point.any():
            raise ValueError(f"convolution {convolution.name!r} uses asymmetric weight quantization")
        quantized_weight = constant_value(producer.input[0])
        if quantized_weight is None or quantized_weight.shape[0] != scale.size:
            raise ValueError(f"convolution {convolution.name!r} has no constant quantized weight tensor")
        flattened_weight = quantized_weight.reshape(quantized_weight.shape[0], -1)
        flattened_zero = zero_point.reshape(-1, 1)
        zero_weight_channels += int(((flattened_weight == flattened_zero).all(axis=1)).sum())
        minimum_weight_scale = min(minimum_weight_scale, float(scale.min()))
        per_channel_weights += 1

    activation_qdq = 0
    for dequantizer in dequantizers:
        if len(dequantizer.input) < 2:
            continue
        scale = constant_value(dequantizer.input[1])
        if scale is None:
            continue
        axes = [attribute.i for attribute in dequantizer.attribute if attribute.name == "axis"]
        if scale.size == 1 and not axes:
            activation_qdq += 1
    if activation_qdq == 0:
        raise ValueError("QAT SSDLite export has no per-tensor activation QDQ boundaries")

    return {
        "activation_qdq_count": activation_qdq,
        "convolution_count": len(convolutions),
        "dequantize_linear_count": len(dequantizers),
        "format": "int8_qat_qdq",
        "onnx_sha256": sha256_file(path),
        "opset": default_opsets[0],
        "output_count": len(outputs),
        "per_channel_weight_count": per_channel_weights,
        "minimum_weight_scale": minimum_weight_scale,
        "quantize_linear_count": len(quantizers),
        "zero_weight_channel_count": zero_weight_channels,
    }
