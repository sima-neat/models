"""Structural qualification for the full-INT8 YOLO26-P2 QAT source."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from contract import DEFAULT_CONTRACT, INPUT_NAME, DeploymentContract
from onnx import helper, numpy_helper, shape_inference

EXPECTED_ATTENTION_MATMULS = 8
EXPECTED_ATTENTION_SOFTMAXES = 4
ATTENTION_EINSUM_EQUATION = "abij,abjk->abik"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)


def _constant_array(
    name: str,
    initializers: dict[str, Any],
    producers: dict[str, Any],
    seen: set[str] | None = None,
) -> np.ndarray | None:
    seen = set() if seen is None else seen
    if name in seen:
        return None
    seen.add(name)
    tensor = initializers.get(name)
    if tensor is not None:
        return numpy_helper.to_array(tensor)
    producer = producers.get(name)
    if producer is None:
        return None
    if producer.op_type == "Identity":
        return _constant_array(producer.input[0], initializers, producers, seen)
    if producer.op_type != "Constant":
        return None
    for attribute in producer.attribute:
        if attribute.name == "value":
            return numpy_helper.to_array(attribute.t)
    return None


def canonicalize_attention_for_modelsdk(
    path: Path,
    *,
    expected_matmuls: int = EXPECTED_ATTENTION_MATMULS,
) -> dict[str, Any]:
    """Express rank-4 attention MatMul through ModelSDK's supported Einsum path.

    ModelSDK 2.1 imports ONNX MatMul as ``nn.batch_matmul``, for which its MLA
    checker has no placement entry. Its documented batch-matmul representation
    is a two-input Einsum. This exact operator substitution preserves all QDQ
    edges and tensor shapes; only the ONNX spelling changes.
    """
    model = onnx.load(path)
    inferred = shape_inference.infer_shapes(model)
    shapes = {
        value.name: _shape(value)
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    if len(matmuls) != expected_matmuls:
        raise RuntimeError(f"expected {expected_matmuls} attention MatMul operators, found {len(matmuls)}")

    rewritten = []
    converted = 0
    for node in model.graph.node:
        if node.op_type != "MatMul":
            rewritten.append(node)
            continue
        input_shapes = [shapes.get(name) for name in node.input]
        output_shapes = [shapes.get(name) for name in node.output]
        if (
            len(node.input) != 2
            or len(node.output) != 1
            or any(shape is None or len(shape) != 4 for shape in (*input_shapes, *output_shapes))
        ):
            raise RuntimeError(
                f"attention MatMul {node.name} is not a statically shaped rank-4 operator: "
                f"{input_shapes} -> {output_shapes}"
            )
        lhs = input_shapes[0]
        rhs = input_shapes[1]
        result = output_shapes[0]
        if lhs is None or rhs is None or result is None:
            raise RuntimeError(f"attention MatMul {node.name} has unresolved shapes")
        if (
            any(dimension <= 0 for shape in (lhs, rhs, result) for dimension in shape)
            or lhs[:2] != rhs[:2]
            or lhs[:2] != result[:2]
            or lhs[2] != result[2]
            or rhs[3] != result[3]
            or lhs[3] != rhs[2]
        ):
            raise RuntimeError(f"attention MatMul {node.name} has incompatible shapes: {lhs}, {rhs} -> {result}")
        rewritten.append(
            helper.make_node(
                "Einsum",
                list(node.input),
                list(node.output),
                name=node.name.replace("MatMul", "Einsum"),
                equation=ATTENTION_EINSUM_EQUATION,
            )
        )
        converted += 1

    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    temporary = path.with_suffix(".tmp.onnx")
    onnx.save(model, temporary)
    temporary.replace(path)
    return {
        "converted_matmuls": converted,
        "onnx_operator": "Einsum",
        "equation": ATTENTION_EINSUM_EQUATION,
        "reason": "ModelSDK 2.1 documented batch-matmul import path",
    }


def audit_onnx(path: Path, contract: DeploymentContract = DEFAULT_CONTRACT) -> dict[str, Any]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or model.graph.input[0].name != INPUT_NAME:
        raise RuntimeError("YOLO26 QAT graph must have exactly the public images input")
    if _shape(model.graph.input[0]) != contract.input_shape:
        raise RuntimeError(f"input shape mismatch: {_shape(model.graph.input[0])}")
    actual_outputs = [(value.name, _shape(value)) for value in model.graph.output]
    if actual_outputs != list(contract.outputs):
        raise RuntimeError(f"raw-head output contract mismatch: {actual_outputs}")

    operators = Counter(node.op_type for node in model.graph.node)
    forbidden = {name: operators[name] for name in ("BatchNormalization", "MatMul") if operators[name]}
    if forbidden:
        raise RuntimeError(f"non-INT8-friendly operators remain: {forbidden}")
    if operators["QuantizeLinear"] == 0 or operators["DequantizeLinear"] == 0:
        raise RuntimeError("QAT graph contains no QDQ operators")

    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    expected_attention = {
        "Einsum": EXPECTED_ATTENTION_MATMULS,
        "Softmax": EXPECTED_ATTENTION_SOFTMAXES,
    }
    for operator, expected_count in expected_attention.items():
        nodes = [node for node in model.graph.node if node.op_type == operator]
        if len(nodes) != expected_count:
            raise RuntimeError(f"expected {expected_count} attention {operator} operators, found {len(nodes)}")
        required_inputs = 2 if operator == "Einsum" else 1
        for node in nodes:
            if operator == "Einsum":
                attributes = {
                    attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
                }
                if attributes.get("equation") != ATTENTION_EINSUM_EQUATION.encode():
                    raise RuntimeError(f"attention Einsum {node.name} has an unexpected equation")
            elif operator == "Softmax":
                attributes = {
                    attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
                }
                if attributes.get("axis") != -1:
                    raise RuntimeError(f"attention Softmax {node.name} must normalize its final sequence axis")
            if len(node.input) < required_inputs or any(
                producers.get(name) is None or producers[name].op_type != "DequantizeLinear"
                for name in node.input[:required_inputs]
            ):
                raise RuntimeError(f"attention {operator} {node.name} does not have INT8 QDQ inputs")
            if any(
                not any(consumer.op_type == "QuantizeLinear" for consumer in consumers.get(output, []))
                for output in node.output
            ):
                raise RuntimeError(f"attention {operator} {node.name} does not have an INT8 QDQ output")

    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    activation_quantizers = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    for node in activation_quantizers:
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        if "axis" in attributes:
            raise RuntimeError(f"activation quantizer {node.name} is not per-tensor")
        scale = _constant_array(node.input[1], initializers, producers)
        zero_point = _constant_array(node.input[2], initializers, producers)
        if scale is None or zero_point is None:
            raise RuntimeError(f"activation quantizer {node.name} has dynamic parameters")
        if scale.size != 1 or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise RuntimeError(f"activation quantizer {node.name} has an invalid scalar scale")
        if zero_point.size != 1 or zero_point.dtype != np.int8:
            raise RuntimeError(f"activation quantizer {node.name} is not signed per-tensor INT8")

    convolution_count = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        convolution_count += 1
        activation_dequantize = producers.get(node.input[0])
        if activation_dequantize is None or activation_dequantize.op_type != "DequantizeLinear":
            raise RuntimeError(f"convolution {node.name} activation is not fed by DequantizeLinear")
        weight_dequantize = producers.get(node.input[1])
        if weight_dequantize is None or weight_dequantize.op_type != "DequantizeLinear":
            raise RuntimeError(f"convolution {node.name} weight is not fed by DequantizeLinear")
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in weight_dequantize.attribute
        }
        if attributes.get("axis") != 0:
            raise RuntimeError(f"convolution {node.name} weight quantization is not axis-0 per-channel")
        weight_array = _constant_array(weight_dequantize.input[0], initializers, producers)
        scale_array = _constant_array(weight_dequantize.input[1], initializers, producers)
        zero_array = _constant_array(weight_dequantize.input[2], initializers, producers)
        if weight_array is None or scale_array is None or zero_array is None:
            raise RuntimeError(f"convolution {node.name} has non-constant weight quantization")
        if weight_array.dtype != np.int8 or zero_array.dtype != np.int8:
            raise RuntimeError(f"convolution {node.name} weights are not signed INT8")
        if scale_array.ndim != 1 or scale_array.shape[0] != weight_array.shape[0]:
            raise RuntimeError(f"convolution {node.name} has invalid per-channel scales")
        if np.any(zero_array != 0):
            raise RuntimeError(f"convolution {node.name} weights are not symmetric")
    if convolution_count == 0:
        raise RuntimeError("QAT graph contains no convolutions")

    return {
        "sha256": sha256_file(path),
        "opset": model.opset_import[0].version,
        "operator_counts": dict(sorted(operators.items())),
        "convolution_count": convolution_count,
        "activation_quantizer_count": len(activation_quantizers),
        "weight_quantization": "signed_int8_symmetric_axis0_per_channel",
        "activation_quantization": "signed_int8_per_tensor_qdq",
        "attention": {
            "matmul_count": operators["Einsum"],
            "onnx_operator": "Einsum",
            "equation": ATTENTION_EINSUM_EQUATION,
            "softmax_count": operators["Softmax"],
            "quantization": "signed_int8_per_tensor_qdq_inputs_and_outputs",
        },
        "outputs": [{"name": name, "shape": list(shape)} for name, shape in actual_outputs],
    }
