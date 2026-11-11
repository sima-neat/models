#!/usr/bin/env python3
"""Export the complete TorchVision SSDLite320 forward pass from pinned COCO weights.

Preprocessing, heads, box decoding and NMS stay in this source graph. The separate
surgery owns the raw-head cut and BatchNorm folding required by Model Compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from model_registry.source import SourceSpec

# Intentionally separate from surgery.py: conversion validates the complete upstream
# source, whereas surgery changes only compilation inputs and must not invalidate it.
SOURCE_INPUT = "image"
INPUT_SHAPE = (1, 3, 320, 320)
HEADS = tuple(
    f"/detector/head/{role}_head/module_list.{level}/module_list.{level}.1/Conv_output_0"
    for role in ("regression", "classification")
    for level in range(6)
)

WEIGHTS_URL = "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
WEIGHTS_SIZE = 14069355
WEIGHTS_SOURCE = {"uri": WEIGHTS_URL, "sha256": WEIGHTS_SHA256, "size_bytes": WEIGHTS_SIZE}
EXPECTED_VERSIONS = {
    "torch": "2.8.0+cpu",
    "torchvision": "0.23.0+cpu",
    "onnx": "1.22.0",
    "onnxruntime": "1.22.1",
    "numpy": "2.5.1",
}
MAX_ABSOLUTE_ERROR = 5e-4
# Absolute FP32 tolerance for raw heads and final boxes/scores. Labels and output
# counts must match exactly. Real COCO final-output checks gate the converted stage
# in tests/build/test_conversion.py; no optional acceptance mode is supported.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> dict:
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


def verified_weights(path: Path) -> Path:
    if sha256(path) != WEIGHTS_SHA256 or path.stat().st_size != WEIGHTS_SIZE:
        raise ValueError("official TorchVision COCO weight digest mismatch")
    return path


def load_detector(path: Path):
    import torch
    from torch import nn
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    verified_weights(path)
    # Both parameters must be None: the original COCO factory uses the reduced
    # tail architecture and no separately downloaded ImageNet backbone weights.
    detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
    detector.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if detector.transform.image_mean != [0.5] * 3 or detector.transform.image_std != [0.5] * 3:
        raise ValueError("TorchVision factory input normalization changed")
    if detector.anchor_generator.num_anchors_per_location() != [6] * 6:
        raise ValueError("TorchVision anchor contract changed")
    if sum(isinstance(module, nn.Conv2d) for module in detector.modules()) != 98:
        raise ValueError("TorchVision convolution contract changed")
    return detector


def flatten_heads(heads: list[np.ndarray], columns: int) -> np.ndarray:
    flattened = []
    for head in heads:
        batch, _channels, height, width = head.shape
        flattened.append(
            head.reshape(batch, 6, columns, height, width).transpose(0, 3, 4, 1, 2).reshape(batch, -1, columns)
        )
    return np.concatenate(flattened, axis=1)


def parity(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite output or changed output shape")
    error = actual.astype(np.float64) - reference.astype(np.float64)
    maximum = float(np.max(np.abs(error)))
    rms_reference = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)))))
    nrmse = float(np.sqrt(np.mean(np.square(error)))) / max(rms_reference, 1e-12)
    denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    cosine_denominator = float(
        np.linalg.norm(reference.astype(np.float64).ravel()) * np.linalg.norm(actual.astype(np.float64).ravel())
    )
    cosine = (
        (
            float(
                np.dot(
                    reference.astype(np.float64).ravel(),
                    actual.astype(np.float64).ravel(),
                )
            )
            / cosine_denominator
        )
        if cosine_denominator
        else float(np.array_equal(reference, actual))
    )
    return {
        "max_abs": maximum,
        "normalized_rmse": nrmse,
        "max_relative_abs_floor_1e-6": float(np.max(np.abs(error) / denominator)),
        "cosine_similarity": cosine,
    }


def compare_heads(expected: dict, actual: list[np.ndarray]) -> dict:
    """Compare the kept tensors, without treating near-tied NMS order as a head regression."""
    if len(actual) != 12:
        raise ValueError("expected twelve raw-head probes")
    results = {}
    for key, columns, section in (("bbox_regression", 4, slice(0, 6)), ("cls_logits", 91, slice(6, 12))):
        result = parity(expected[key], flatten_heads(actual[section], columns))
        if result["max_abs"] > MAX_ABSOLUTE_ERROR:
            raise ValueError(f"source head parity exceeds the existing absolute limit: {key}: {result}")
        results[key] = result
    return results


def expected_versions(machine: str) -> dict[str, str]:
    if machine not in {"x86_64", "aarch64"}:
        raise ValueError(f"unsupported conversion architecture: {machine}")
    return {**EXPECTED_VERSIONS, "torchvision": "0.23.0" if machine == "aarch64" else "0.23.0+cpu"}


def resolve_weights(path: Path | None) -> Path:
    """Use the canonical verified source cache; an explicit local file is verified too."""
    if path is None:
        model = load_and_validate_model(Path(__file__).resolve().parents[1])
        source = model["conversion"]["default"]["source"]
        expected = {"type": "file", "sha256": WEIGHTS_SHA256, "size_bytes": WEIGHTS_SIZE}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("model.yaml conversion source differs from the pinned TorchVision weights")
        if source.get("uri") != WEIGHTS_URL:
            raise ValueError("model.yaml must use the pinned official TorchVision weights URL")
        spec = SourceSpec.from_mapping(source)
        path, _cache_hit = Fetcher().fetch(spec)
    return verified_weights(path)


def export(path: Path, weights: Path) -> dict:
    import onnxruntime as ort
    import torch

    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != expected_versions(platform.machine()):
        raise ValueError(f"export environment differs from pinned source recipe: {versions}")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    detector = load_detector(weights)

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = detector

        def forward(self, image):
            result = self.detector([image[0]])[0]
            return result["boxes"], result["scores"], result["labels"]

    source = Source().eval()
    example = torch.linspace(0, 1, int(np.prod(INPUT_SHAPE))).reshape(INPUT_SHAPE)
    with torch.inference_mode():
        torch.onnx.export(
            source,
            example,
            str(path),
            opset_version=17,
            dynamo=False,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.PRESERVE,
            input_names=[SOURCE_INPUT],
            output_names=["boxes", "scores", "labels"],
        )
    graph = onnx.load(path)
    # The number surviving score filtering/NMS varies with image content.
    for value in graph.graph.output:
        value.type.tensor_type.shape.dim[0].dim_param = "detections"
    onnx.helper.set_model_props(
        graph,
        {
            "sima.architecture": "torchvision.ssdlite320_mobilenet_v3_large",
            "sima.precision.source": "fp32_pretrained_not_qat",
            "sima.input.normalization": "rgb_float32_0_to_1; normalization inside graph",
            "sima.torchvision.weights_url": WEIGHTS_URL,
            "sima.torchvision.weights_sha256": WEIGHTS_SHA256,
            "sima.export.versions": json.dumps(versions, sort_keys=True),
        },
    )
    onnx.save(graph, path)
    artifact = audit(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # Verify the exported operations, not optional ORT Conv/BN fusions whose
    # FP32 rounding can reorder near-tied NMS results. Native accuracy is separate.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Probe the kept internal tensors in memory; never publish a cut conversion.
    probe = onnx.load(path)
    produced = {name for node in probe.graph.node for name in node.output}
    if not set(HEADS) <= produced:
        raise ValueError("full source is missing raw-head probes")
    probe.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in HEADS)
    session = ort.InferenceSession(probe.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    trials = []
    for name, image in (
        ("black_320", torch.zeros(INPUT_SHAPE)),
        ("white_320", torch.ones(INPUT_SHAPE)),
        ("ramp_320", example),
        ("seed0_320", torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(0))),
    ):
        with torch.inference_mode():
            transformed, _ = detector.transform([image[0]], None)
            heads = detector.head(list(detector.backbone(transformed.tensors).values()))
        actual = session.run(None, {SOURCE_INPUT: image.numpy()})
        trial = {
            "input": name,
            "raw_heads": compare_heads({key: value.numpy() for key, value in heads.items()}, actual[3:]),
        }
        trials.append(trial)
    return {
        "schema_version": 1,
        "artifact": artifact,
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "versions": versions,
        "export_script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "parity": trials,
        "source_contract": "complete upstream FP32 forward; no cut, BN fold or QAT",
        "parity_absolute_tolerance": MAX_ABSOLUTE_ERROR,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", choices=["default"])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = resolve_weights(args.weights)
    path = args.output_dir / "model.onnx"
    report = export(path, verified_weights(weights))
    (args.output_dir / "export-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(path), **report["artifact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
