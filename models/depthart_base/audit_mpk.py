#!/usr/bin/env python3
"""Audit the finished DepthART MPK for a single, floating-point-free MLA graph."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path

EXPECTED_RECURRENCE_OPERATORS = {
    "depthart_dynamic_product.BlockP64HiddenProductOperator": 918,
    "depthart_p64_to_static.DepthARTP64ToStaticOperator": 918,
    "depthart_static_int8_to_p64.StaticInt8ToP64Operator": 2754,
    "residual_dynamic_add.ResidualDynamicAddOperator": 918,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with tarfile.open(args.mpk, "r:gz") as archive:
        names = archive.getnames()
        precision_member = next(
            (
                name
                for name in names
                if Path(name).name == "modelgraph_precision_map.json"
            ),
            None,
        )
        mpk_json_member = next(
            (name for name in names if name.endswith("_mpk.json")), None
        )
        if precision_member is None:
            raise RuntimeError("missing modelgraph_precision_map.json")
        if mpk_json_member is None:
            raise RuntimeError("missing _mpk.json")
        precision_stream = archive.extractfile(precision_member)
        mpk_stream = archive.extractfile(mpk_json_member)
        if precision_stream is None or mpk_stream is None:
            raise RuntimeError("MPK metadata member is not a regular file")
        precision = json.load(precision_stream)
        mpk = json.load(mpk_stream)
        elfs = [name for name in names if name.endswith(".elf")]
        elf_hashes = {}
        for name in elfs:
            stream = archive.extractfile(name)
            if stream is None:
                raise RuntimeError(f"ELF member is not a regular file: {name}")
            elf_hashes[name] = hashlib.sha256(stream.read()).hexdigest()

    shared_objects = [name for name in names if name.endswith(".so")]
    vertices = [
        vertex
        for stage in precision["stages"]
        for vertex in stage["vertices"]
    ]
    dtype_counts = collections.Counter(
        vertex.get("output_shape", {}).get("dtype", "MISSING")
        for vertex in vertices
    )
    native_dtype_counts = collections.Counter(
        vertex.get("output_shape", {}).get("native_dtype", "MISSING")
        for vertex in vertices
    )
    operation_counts = collections.Counter(
        vertex["operation"] for vertex in vertices
    )

    def output_dtype(vertex: dict) -> str:
        return str(vertex.get("output_shape", {}).get("dtype", "")).upper()

    def is_constant(vertex: dict) -> bool:
        return vertex["operation"].endswith(".ConstantOperator")

    bf16 = [vertex for vertex in vertices if "BF16" in output_dtype(vertex)]
    fp16 = [
        vertex
        for vertex in vertices
        if "FP16" in output_dtype(vertex) or "FLOAT16" in output_dtype(vertex)
    ]
    fp32 = [
        vertex
        for vertex in vertices
        if "FP32" in output_dtype(vertex) or "FLOAT32" in output_dtype(vertex)
    ]
    floating = {vertex["vertex_id"]: vertex for vertex in (*bf16, *fp16, *fp32)}
    plugins = mpk.get("plugins", [])
    public_inputs = mpk.get("input_nodes") or []
    public_outputs = [
        node
        for plugin in plugins
        for node in (plugin.get("output_nodes") or [])
    ]
    recurrence_operator_counts = {
        operation: operation_counts[operation]
        for operation in EXPECTED_RECURRENCE_OPERATORS
    }

    acceptance = {
        "exactly_one_stage": len(precision["stages"]) == 1,
        "exactly_one_elf": len(elfs) == 1,
        "exactly_one_mla_plugin": (
            len(plugins) == 1 and plugins[0].get("processor") == "MLA"
        ),
        "public_input_int8": bool(public_inputs)
        and all(str(node.get("dtype", "")).lower() == "int8" for node in public_inputs),
        "public_output_int8": bool(public_outputs)
        and all(str(node.get("dtype", "")).lower() == "int8" for node in public_outputs),
        "zero_bf16": not bf16,
        "zero_fp16": not fp16,
        "zero_fp32": not fp32,
        "zero_floating_tensors": not floating,
        "zero_shared_objects": not shared_objects,
        "complete_depthart_recurrence_program": (
            recurrence_operator_counts == EXPECTED_RECURRENCE_OPERATORS
        ),
    }
    acceptance["pass"] = all(acceptance.values())

    report = {
        "schema": "depthart-mpk-precision-audit/v2",
        "mpk": str(args.mpk.resolve()),
        "mpk_sha256": sha256(args.mpk),
        "members": len(names),
        "precision_map_member": precision_member,
        "stage_count": len(precision["stages"]),
        "vertex_count": len(vertices),
        "edge_count": precision.get("edge_count"),
        "output_dtype_counts": dict(sorted(dtype_counts.items())),
        "native_dtype_counts": dict(sorted(native_dtype_counts.items())),
        "bf16_tensor_count": len(bf16),
        "bf16_nonconstant_operator_count": sum(not is_constant(v) for v in bf16),
        "bf16_constant_count": sum(is_constant(v) for v in bf16),
        "fp16_tensor_count": len(fp16),
        "fp16_nonconstant_operator_count": sum(not is_constant(v) for v in fp16),
        "fp32_tensor_count": len(fp32),
        "fp32_nonconstant_operator_count": sum(not is_constant(v) for v in fp32),
        "floating_tensor_count": len(floating),
        "floating_vertices_sample": [
            {
                "id": vertex["vertex_id"],
                "name": vertex["vertex_name"],
                "operation": vertex["operation"],
                "dtype": output_dtype(vertex),
            }
            for vertex in list(floating.values())[:20]
        ],
        "operation_counts": dict(sorted(operation_counts.items())),
        "expected_recurrence_operator_counts": EXPECTED_RECURRENCE_OPERATORS,
        "recurrence_operator_counts": recurrence_operator_counts,
        "elf_count": len(elfs),
        "elfs": [
            {"member": name, "sha256": elf_hashes[name]} for name in elfs
        ],
        "shared_object_count": len(shared_objects),
        "shared_objects": shared_objects,
        "plugins": [
            {"name": plugin.get("name"), "processor": plugin.get("processor")}
            for plugin in plugins
        ],
        "public_inputs": public_inputs,
        "plugin_io": [
            {
                "name": plugin.get("name"),
                "processor": plugin.get("processor"),
                "input_types": plugin.get("config_params", {}).get("input_types"),
                "output_types": plugin.get("config_params", {}).get("output_types"),
                "outputs": plugin.get("output_nodes"),
            }
            for plugin in plugins
        ],
        "acceptance": acceptance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
