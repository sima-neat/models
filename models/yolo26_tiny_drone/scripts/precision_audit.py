"""Read the pinned Compiler 3.0 public save format to admit W8A8 computation.

This inspects the quantized model passed to compile, not every backend
instruction or exact folded QAT parameters. Public FP32 boundaries, INT32
accumulators/biases and stale pre-quantization attributes are not compute ports.
No serialized model is loaded back or modified.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import yaml
from afe.apis.defines import scalar_type_from_dtype
from model_registry.fetch import sha256_file


def _port_scalars(value: Any) -> list[int]:
    """Only authoritative execution TensorTypes; never recursively scan attrs."""
    if isinstance(value, list) and value:
        return [scalar for item in value for scalar in _port_scalars(item)]
    if isinstance(value, dict) and value.get("__decode_key__") == "TensorType":
        scalar = value.get("scalar", {})
        if scalar.get("__decode_key__") == "ScalarType" and isinstance(scalar.get("value"), int):
            return [scalar["value"]]
    raise RuntimeError("missing or unsupported quantized execution TensorType")


def _saved_layers(saved: Path) -> dict[str, Any]:
    document = json.loads(saved.with_suffix(saved.suffix + ".json").read_text(encoding="utf-8"))
    try:
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("unsupported model save format") from exc
    if not layers or len({layer.get("name") for layer in layers}) != len(layers):
        raise RuntimeError("missing or ambiguous model nodes")
    return {layer["name"]: layer for layer in layers}


def _archive_nodes(archive: zipfile.ZipFile) -> dict[str, Any]:
    descriptions = [name for name in archive.namelist() if name.endswith(".sima.yaml")]
    if len(descriptions) != 1:
        raise RuntimeError("expected one bound archive graph")
    graph = yaml.safe_load(archive.read(descriptions[0]))
    if not isinstance(graph, list) or len(graph) != 4 or graph[1] != "Format version 1":
        raise RuntimeError("unsupported archive graph format")
    result: dict[str, Any] = {}

    def collect(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            name = node["name"]
            if name in result:
                raise RuntimeError(f"ambiguous archive node: {name}")
            result[name] = node
            collect(node.get("ir", {}).get("nodes", []))

    collect(graph[3]["nodes"])
    return result


def _port_shapes(value: Any) -> Any:
    """Preserve argument/tuple nesting while comparing logical port geometry."""
    if isinstance(value, list) and value:
        return [_port_shapes(item) for item in value]
    if isinstance(value, dict) and value.get("__decode_key__") == "TensorType":
        shape = value.get("shape", {})
        dims = shape.get("vals")
        if shape.get("__decode_key__") == "tuple" and isinstance(dims, list) and dims:
            if all(type(dim) is int and dim > 0 for dim in dims):
                return tuple(dims)
    raise RuntimeError("missing or unsupported attention execution shape")


def _attention_coverage(normalized: Path, layers: dict[str, Any], nodes: dict[str, Any]) -> int:
    """Bind attention to the paired, normalized FP32 graph from this public save.

    Compiler 3.0 loses original Einsum labels in the full detector. Its paired
    normalized graph retains the actual BatchMatmul identities, topology and
    geometry. This checks those operations, not an invented mapping back to
    original ONNX names; source semantics still require numerical qualification.
    """
    before = _saved_layers(normalized)
    with zipfile.ZipFile(normalized) as archive:
        imported = _archive_nodes(archive)
    attention = {
        name: node
        for name, node in imported.items()
        if name.startswith("MLA_0/") and node.get("ir", {}).get("op") == "batch_matmul"
    }
    for name, original in attention.items():
        if name not in before or name not in layers or name not in nodes:
            raise RuntimeError(f"missing paired normalized attention operation: {name}")
        current = nodes[name]
        if (
            not str(layers[name].get("class_name", "")).startswith("MLA_0/")
            or layers[name].get("class_name") != before[name].get("class_name")
            or current.get("status") != "sima_quantized"
        ):
            raise RuntimeError(f"normalized attention has no matching MLA compute owner: {name}")
        quant = current.get("ir", {}).get("quant") or {}
        if current.get("ir", {}).get("op") != "batch_matmul" or quant.get("op") != "batch_matmul":
            raise RuntimeError(f"normalized attention operation changed: {name}")
        attrs = original["ir"].get("attrs", {})
        quant_attrs = quant.get("attrs", {})
        for field in ("input_shapes", "transpose_b"):
            if field not in attrs or field not in quant_attrs or attrs[field] != quant_attrs[field]:
                raise RuntimeError(f"normalized attention {field} changed: {name}")
        if (
            not original.get("inputs")
            or original["inputs"] != current.get("inputs")
            or not before[name].get("inbound_nodes")
            or before[name]["inbound_nodes"] != layers[name].get("inbound_nodes")
        ):
            raise RuntimeError(f"normalized attention inputs changed: {name}")
        for field in ("_input_types", "_output_types"):
            if _port_shapes(before[name]["config"].get(field)) != _port_shapes(layers[name]["config"].get(field)):
                raise RuntimeError(f"normalized attention {field} geometry changed: {name}")
    return len(attention)


def audit_saved_quantization(saved: Path, source: Path) -> dict[str, Any]:
    """Check source coverage, every MLA activation port and actual weight bytes.

    Conv/Softmax use original source provenance, allowing fusion. Attention uses
    the paired normalized graph because the importer drops original Einsum
    labels. Optimized counts and generated names are never hardcoded.
    """
    expected: dict[str, str] = {}
    for node in onnx.load(source).graph.node:
        if node.op_type in {"Conv", "Einsum", "Softmax"}:
            if not node.name or node.name in expected:
                raise RuntimeError("precision audit requires unique named source compute operations")
            expected[node.name] = node.op_type
    if not expected:
        raise RuntimeError("precision audit has no source compute operations")
    layers = _saved_layers(saved)
    mla = [layer for layer in layers.values() if str(layer.get("class_name", "")).startswith("MLA_0/")]
    if not mla or len({layer.get("name") for layer in mla}) != len(mla):
        raise RuntimeError("missing or ambiguous MLA compute nodes")
    int8 = scalar_type_from_dtype("int8").value
    coverage: dict[str, list[str]] = {name: [] for name, kind in expected.items() if kind != "Einsum"}
    normalized = saved.with_name(saved.stem + ".fp32" + saved.suffix)
    weights_checked = 0
    with zipfile.ZipFile(saved) as archive:
        members = archive.namelist()
        payloads = [name for name in members if name.endswith(".sima.npz")]
        if len(payloads) != 1:
            raise RuntimeError("expected one bound quantized weight payload")
        archive_nodes = _archive_nodes(archive)
        attention_count = _attention_coverage(normalized, layers, archive_nodes)
        if "Einsum" in expected.values() and not attention_count:
            raise RuntimeError("source attention has no paired normalized BatchMatmul operations")
        with np.load(io.BytesIO(archive.read(payloads[0])), allow_pickle=False) as weights:
            for layer in mla:
                name, config = layer["name"], layer.get("config", {})
                archived = archive_nodes.get(name, {})
                operation = archived.get("ir", {}).get("op")
                structural = operation in {"tuple", "tuple_get_item"}
                status = config.get("_status")
                # Compiler-owned tuple packing/projection has no arithmetic
                # quantization step. Identify it through this same archive,
                # not generated names, and still check every INT8 port below.
                if status != "SIMA_QUANTIZED" and not (
                    structural and status == "RELAY" and archived.get("status") == "relay"
                ):
                    raise RuntimeError(
                        f"MLA operation is not quantized: {name}, class={layer.get('class_name')!r}, "
                        f"status={status!r}, operation={operation!r}"
                    )
                for field in ("_input_types", "_output_types"):
                    try:
                        scalars = _port_scalars(config.get(field))
                    except RuntimeError as exc:
                        raise RuntimeError(f"{name} {field}: {exc}") from exc
                    if any(scalar != int8 for scalar in scalars):
                        raise RuntimeError(f"MLA operation does not have signed INT8 {field}: {name}")
                sources = config.get("Source names", [])
                if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
                    raise RuntimeError(f"unsupported source provenance: {name}")
                claimed = set(sources) & expected.keys()
                if structural and claimed:
                    raise RuntimeError(f"structural MLA operation cannot cover source computation: {name} {claimed}")
                for original in claimed & coverage.keys():
                    coverage[original].append(name)
                has_conv = any(expected.get(original) == "Conv" for original in sources)
                if has_conv or "weight_quant_data" in config:
                    # JSON and archive serialization have independent np_arr
                    # namespaces. Bind the node, then use YAML -> same-ZIP NPZ.
                    archived = archive_nodes.get(name, {})
                    quant = archived.get("ir", {}).get("quant") or {}
                    if archived.get("status") != "sima_quantized":
                        raise RuntimeError(f"missing quantized archive weight owner: {name}")
                    if set(archived.get("source_node_names") or []) != set(sources):
                        raise RuntimeError(f"quantized archive source provenance mismatch: {name}")
                    key = quant.get("weight_quant_data")
                    if not isinstance(key, str) or key not in weights:
                        raise RuntimeError(f"missing quantized weight payload: {name} {key!r}")
                    value = weights[key]
                    if value.dtype != np.int8 or not value.size or quant.get("per_channel") is not True:
                        raise RuntimeError(f"MLA weights must be nonempty per-channel signed INT8: {name} {key}")
                    weights_checked += 1
    invalid = {name: owners for name, owners in coverage.items() if len(owners) != 1}
    if invalid:
        raise RuntimeError(f"missing or ambiguous quantized source coverage: {invalid}")
    return {
        "source_sha256": sha256_file(source)[0],
        "quantized_model_sha256": sha256_file(saved)[0],
        "typed_graph_sha256": sha256_file(saved.with_suffix(saved.suffix + ".json"))[0],
        "source_operations": dict(Counter(expected.values())),
        "direct_source_coverage": dict(Counter(expected[name] for name in coverage)),
        "normalized_attention_operations": attention_count,
        "attention_coverage_authority": "paired_normalized_fp32_graph",
        "normalized_model_sha256": sha256_file(normalized)[0],
        "normalized_typed_graph_sha256": sha256_file(normalized.with_suffix(normalized.suffix + ".json"))[0],
        "mla_nodes_with_int8_ports": len(mla),
        "per_channel_int8_weights": weights_checked,
    }


def audit_quantized_model(model: Any, source: Path, directory: Path) -> dict[str, Any]:
    """Save the exact in-memory candidate through the public API before compile."""
    directory.mkdir()
    model.save(model_name="quantization", output_directory=str(directory), include_unquantized_net=True)
    return audit_saved_quantization(directory / "quantization.sima", source)
