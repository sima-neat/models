#!/usr/bin/env python3
"""Materialize OSTrack's exact static QAT grids as ONNX Q/DQ boundaries.

Dynamic Q/K, dynamic logit requantization, and AV scale transport remain typed
ModelSDK workspace programs.  This source encodes every static grid and folds
the selected per-output W8 codes into ordinary constant initializers so ONNX
layout conversion still sees constant Conv/MatMul weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_name(node_name: str, available: set[str]) -> str:
    parts = [part for part in node_name.split("/") if part]
    if parts and parts[0] == "m":
        parts = parts[1:]
    if parts and parts[-1] in ("MatMul", "Conv", "Gemm"):
        parts = parts[:-1]
    if len(parts) >= 2 and parts[-1].startswith(parts[-2] + "."):
        parts = parts[:-2] + [parts[-1]]
    candidate = ".".join(parts)
    if candidate not in available and candidate.endswith("_1") and candidate[:-2] in available:
        candidate = candidate[:-2]
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite QDQ output or receipt")

    catalogue = json.loads(args.catalogue.read_text())
    learned = {row["module"]: row for row in catalogue["learned_affines"]}
    roles = {row["role"]: row for row in catalogue["activation_boundaries"]}
    model = onnx.load(args.source, load_external_data=True)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}

    weight_rows = []
    learned_nodes = []
    for node in model.graph.node:
        is_learned = (
            node.op_type in ("Conv", "MatMul", "Gemm") and len(node.input) > 1 and node.input[1] in initializers
        )
        if not is_learned:
            continue
        name = module_name(node.name, set(learned))
        if name not in learned:
            raise RuntimeError(f"cannot map learned ONNX node {node.name!r} to QAT module {name!r}")
        row = learned[name]
        weight = numpy_helper.to_array(initializers[node.input[1]]).astype(np.float32)
        # PyTorch Linear stores [output,input], whereas the exported ONNX
        # MatMul initializer is [input,output].
        axis = 0 if node.op_type == "Conv" else 1
        scales = np.asarray(row["weight"]["scales"], dtype=np.float32)
        shape = [1] * weight.ndim
        shape[axis] = scales.size
        view = scales.reshape(shape)
        codes = np.clip(np.rint(weight / view), -127, 127).astype(np.int8)
        dequantized = codes.astype(np.float32) * view
        replacement = numpy_helper.from_array(dequantized, node.input[1])
        initializers[node.input[1]].CopyFrom(replacement)
        learned_nodes.append((node, name, row))
        weight_rows.append(
            {
                "node": node.name,
                "module": name,
                "axis": axis,
                "code_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                "codes": int(codes.size),
            }
        )
    if len(learned_nodes) != 65:
        raise RuntimeError(f"expected 65 learned callsites, found {len(learned_nodes)}")

    boundary_scales: dict[str, tuple[float, int, str]] = {}
    for node in model.graph.node:
        name = node.name
        if name.endswith("__Bias") and "LayerNormalization" in name:
            owner_path = name.rsplit("__", 1)[0]
            if owner_path.endswith("/LayerNormalization"):
                owner_path = owner_path[: -len("/LayerNormalization")]
            owner = owner_path.removeprefix("/m/").replace("/", ".")
            role = owner + ".output"
            boundary_scales[node.output[0]] = (roles[role]["scale"], 0, role)
        if "/mlp/act/Div" in name and node.op_type == "Div":
            owner = name.rsplit("/Div", 1)[0].removeprefix("/m/").replace("/", ".")
            role = owner + ".input"
            boundary_scales[node.input[0]] = (roles[role]["scale"], 0, role)
        if "/mlp/act/Mul_1" in name and node.op_type == "Mul":
            owner = name.rsplit("/Mul_1", 1)[0].removeprefix("/m/").replace("/", ".")
            role = owner + ".output"
            boundary_scales[node.output[0]] = (roles[role]["scale"], 0, role)
        match = re.fullmatch(r"/m/backbone/blocks\.(\d+)/Add(_1)?", name)
        if node.op_type == "Add" and match:
            role = f"block{match.group(1)}.residual{'2' if match.group(2) else '1'}"
            boundary_scales[node.output[0]] = (roles[role]["scale"], 0, role)
        if node.op_type == "Softmax" and "/attn/Softmax" in name:
            boundary_scales[node.output[0]] = (1 / 127, 0, name + ":probability_q7")
        if node.op_type == "MatMul" and "/attn/MatMul_1" in name:
            block = int(re.search(r"blocks\.(\d+)", name).group(1))
            v_role = f"block{block}.V"
            av_role = f"block{block}.AV_output"
            boundary_scales[node.input[1]] = (roles[v_role]["scale"], 0, v_role)
            boundary_scales[node.output[0]] = (roles[av_role]["scale"], 0, av_role)

    expected_static_boundaries = 25 + 24 + 24 + 12 + 12 + 12
    if len(boundary_scales) != expected_static_boundaries:
        raise RuntimeError(f"expected {expected_static_boundaries} static boundaries, found {len(boundary_scales)}")

    dq_by_tensor: dict[str, str] = {}
    qdq_after: dict[str, list] = {}
    qdq_rows = []
    for index, (tensor, (scale, zero_point, role)) in enumerate(sorted(boundary_scales.items())):
        tag = f"qat_static_boundary_{index}"
        scale_name, zero_name = tag + "_scale", tag + "_zero_point"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(scale, np.float32), scale_name),
                numpy_helper.from_array(np.asarray(zero_point, np.int8), zero_name),
            ]
        )
        quantized, dequantized = tag + "_quantized", tag + "_dequantized"
        qdq_after[tensor] = [
            helper.make_node(
                "QuantizeLinear", [tensor, scale_name, zero_name], [quantized], name=tag + "_QuantizeLinear"
            ),
            helper.make_node(
                "DequantizeLinear", [quantized, scale_name, zero_name], [dequantized], name=tag + "_DequantizeLinear"
            ),
        ]
        dq_by_tensor[tensor] = dequantized
        qdq_rows.append({"tensor": tensor, "role": role, "scale": scale, "zero_point": zero_point})

    learned_by_identity = {id(node): (name, row) for node, name, row in learned_nodes}
    ordered = []
    learned_qdq_rows = []
    for index, node in enumerate(model.graph.node):
        for input_index, input_name in enumerate(node.input):
            if input_name in dq_by_tensor:
                node.input[input_index] = dq_by_tensor[input_name]
        learned_row = learned_by_identity.get(id(node))
        if learned_row is not None:
            name, row = learned_row
            tag = f"qat_learned_input_{index}"
            scale = float(row["activation"]["scale"])
            scale_name, zero_name = tag + "_scale", tag + "_zero_point"
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(np.asarray(scale, np.float32), scale_name),
                    numpy_helper.from_array(np.asarray(0, np.int8), zero_name),
                ]
            )
            quantized, dequantized = tag + "_quantized", tag + "_dequantized"
            ordered.extend(
                [
                    helper.make_node(
                        "QuantizeLinear",
                        [node.input[0], scale_name, zero_name],
                        [quantized],
                        name=tag + "_QuantizeLinear",
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [quantized, scale_name, zero_name],
                        [dequantized],
                        name=tag + "_DequantizeLinear",
                    ),
                ]
            )
            node.input[0] = dequantized
            learned_qdq_rows.append({"node": node.name, "module": name, "scale": scale})
        ordered.append(node)
        for output in node.output:
            ordered.extend(qdq_after.get(output, ()))

    model.graph.ClearField("node")
    model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    onnx.save(model, temporary)
    os.replace(temporary, args.output)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": sha256_file(args.source),
        "catalogue_sha256": sha256_file(args.catalogue),
        "output_sha256": sha256_file(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "learned_input_qdq_count": len(learned_qdq_rows),
        "static_boundary_qdq_count": len(qdq_rows),
        "folded_w8_callsite_count": len(weight_rows),
        "dynamic_contract": {
            "qk": 12,
            "logit_requantization": 12,
            "av_scale_transport": 12,
            "encoded_in_onnx": False,
            "required_workspace_program": "mla_qk_pow2/p64_hwc16_v2 plus target AV contract",
        },
        "weights": weight_rows,
        "learned_inputs": learned_qdq_rows,
        "static_boundaries": qdq_rows,
        "evidence_level": "ONNX QDQ preview source; dynamic programs and AFE authority remain pending",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_sha256": receipt["output_sha256"],
                "learned_input_qdq_count": len(learned_qdq_rows),
                "static_boundary_qdq_count": len(qdq_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
