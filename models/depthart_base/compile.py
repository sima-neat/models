#!/usr/bin/env python3
"""Compile the qualified DepthART Base strict-INT8 snapshot for Modalix."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec

MODEL_DIR = Path(__file__).resolve().parent
PROFILE = MODEL_DIR / "depthart_dynamic_p64.json"
EXPECTED_SOURCE_SHA256 = "34c508fc9d200a8bffb281bbfe484623bec2ec45b294f4541894365516790d9c"
EXPECTED_PROFILE_SHA256 = "7627678137983ed5cf8c351e3ea3a81460c512e629133822519a62f972da52cc"
EXPECTED_PROFILE_MANIFEST = "f0d0fdceb161d08a73e04b3cee776347c1781430508f47f62dd6127a69224d46"
EXPECTED_SIMA_SHA256 = "b3f1c1a2001891beb92814126c2fac1489a04fda63600e8cfe8173a72692664c"
EXPECTED_BINDINGS = 918
INPUT_SHAPE_NCHW = (1, 3, 448, 576)
INPUT_SHAPE_NHWC = (1, 448, 576, 3)
OUTPUT_SHAPE_NHWC = (1, 448, 576, 1)
LOG = logging.getLogger("depthart_base.compile")


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _canonical_sha256(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["compile_variants"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one DepthART variant named {variant_id!r}")
    variant = matches[0]
    parameters = variant["parameters"]
    required = {
        "precision": "int8",
        "quantization_method": "qat",
        "public_input_dtype": "int8",
        "public_output_dtype": "int8",
        "recurrence": "dynamic_q8_p64",
        "physical_state_channels": 128,
        "expected_bindings": EXPECTED_BINDINGS,
        "require_single_mla": True,
        "require_zero_float_compute": True,
    }
    mismatches = {
        key: (parameters.get(key), value)
        for key, value in required.items()
        if parameters.get(key) != value
    }
    if variant["target"] != "modalix" or mismatches:
        raise ValueError(
            f"unsupported DepthART compile contract: target={variant['target']!r}, "
            f"mismatches={mismatches}"
        )
    return variant


def audit_source(path: Path) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto

    digest, size = sha256_file(path)
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"DepthART ONNX changed: expected {EXPECTED_SOURCE_SHA256}, found {digest}"
        )
    model = onnx.load(str(path), load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    outputs = list(model.graph.output)

    def signature(value: Any) -> tuple[str, int, tuple[int, ...]]:
        tensor = value.type.tensor_type
        shape = tuple(int(dimension.dim_value) for dimension in tensor.shape.dim)
        return value.name, int(tensor.elem_type), shape

    if [signature(value) for value in inputs] != [
        ("image", TensorProto.FLOAT, INPUT_SHAPE_NCHW)
    ]:
        raise RuntimeError("DepthART source input contract changed")
    if [signature(value) for value in outputs] != [
        ("depth", TensorProto.FLOAT, (1, 1, 448, 576))
    ]:
        raise RuntimeError("DepthART source output contract changed")
    return {
        "sha256": digest,
        "size_bytes": size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def audit_profile(path: Path, source_sha256: str) -> dict[str, Any]:
    digest, size = sha256_file(path)
    if digest != EXPECTED_PROFILE_SHA256:
        raise RuntimeError(
            f"DepthART compile profile changed: expected {EXPECTED_PROFILE_SHA256}, found {digest}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(document)
    claimed = str(unsigned.pop("manifest_sha256", ""))
    if claimed != EXPECTED_PROFILE_MANIFEST or _canonical_sha256(unsigned) != claimed:
        raise RuntimeError("DepthART compile-profile manifest is not canonical/hash-valid")
    if document.get("onnx_sha256") != source_sha256:
        raise RuntimeError("DepthART compile profile is not bound to the selected ONNX")
    bindings = document.get("bindings", [])
    if len(bindings) != EXPECTED_BINDINGS:
        raise RuntimeError(
            f"DepthART recurrence binding count changed: {len(bindings)}"
        )
    sources = [
        str(row[role]["source_node"])
        for row in bindings
        for role in ("product", "state_add")
    ]
    if len(sources) != 2 * EXPECTED_BINDINGS or len(set(sources)) != len(sources):
        raise RuntimeError("DepthART recurrence bindings are empty or ambiguous")
    return {
        "sha256": digest,
        "size_bytes": size,
        "manifest_sha256": claimed,
        "bindings": len(bindings),
        "carrier_abi": document.get("carrier_abi"),
    }


def _safe_snapshot(archive: Path, destination: Path) -> tuple[Path, dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe quantized snapshot member: {member.name}")
            if not member.isfile():
                raise RuntimeError(f"snapshot contains a non-file member: {member.name}")
        sima_members = [member for member in members if member.name.endswith(".sima")]
        receipt_members = [
            member for member in members if PurePosixPath(member.name).name == "provenance.json"
        ]
        if len(sima_members) != 1 or len(receipt_members) != 1:
            raise RuntimeError("snapshot requires exactly one .sima and provenance.json")
        sima_source = stream.extractfile(sima_members[0])
        receipt_source = stream.extractfile(receipt_members[0])
        if sima_source is None or receipt_source is None:
            raise RuntimeError("snapshot members must be regular files")
        sima = destination / PurePosixPath(sima_members[0].name).name
        with sima.open("wb") as output:
            shutil.copyfileobj(sima_source, output)
        receipt = json.load(receipt_source)
    digest, _ = sha256_file(sima)
    expected_receipt = {
        "schema": "depthart-quantized-snapshot/v1",
        "source_onnx_sha256": EXPECTED_SOURCE_SHA256,
        "compile_profile_sha256": EXPECTED_PROFILE_SHA256,
        "compile_profile_manifest_sha256": EXPECTED_PROFILE_MANIFEST,
        "sima_sha256": EXPECTED_SIMA_SHA256,
        "recurrence_bindings": EXPECTED_BINDINGS,
    }
    mismatches = {
        key: (receipt.get(key), value)
        for key, value in expected_receipt.items()
        if receipt.get(key) != value
    }
    if digest != EXPECTED_SIMA_SHA256 or mismatches:
        raise RuntimeError(
            f"quantized snapshot provenance mismatch: sima={digest}, fields={mismatches}"
        )
    return sima, receipt


def _internal_model_name(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".sima.yaml")]
    if len(members) != 1:
        raise RuntimeError(f"expected one .sima.yaml member, found {members}")
    return Path(members[0]).name.removesuffix(".yaml")


def _stage_for_model_load(path: Path, directory: Path) -> tuple[str, Path]:
    internal = _internal_model_name(path)
    staged = directory / internal
    if staged.resolve() != path.resolve():
        shutil.copy2(path, staged)
    return internal, staged


def _tessellation(model: Any) -> dict[str, Any]:
    from afe.apis.defines import (
        TensorDRAMLayout,
        TensorTessellateParameters,
    )
    from afe.ir.node import node_is_tuple

    mla_nodes = [
        node
        for name, node in model._net.nodes.items()
        if str(name).startswith("MLA") and hasattr(getattr(node, "ir", None), "nodes")
    ]
    if len(mla_nodes) != 1:
        raise RuntimeError(f"expected one MLA subnet, found {len(mla_nodes)}")
    mla = mla_nodes[0]
    if len(mla.input_names) != 1:
        raise RuntimeError(f"expected one MLA input, found {mla.input_names}")
    input_name = mla.input_names[0]
    input_shape = tuple(mla.ir.nodes[input_name].get_type().output.value.shape)
    if len(input_shape) != 4:
        raise RuntimeError(f"expected rank-4 MLA input, found {input_shape}")
    parameters = {
        input_name: TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC,
        )
    }
    output = mla.ir.nodes[mla.ir.output_node_name]
    output_names = output.input_node_names if node_is_tuple(output) else [output.name]
    if len(output_names) != 1:
        raise RuntimeError(f"expected one MLA output, found {output_names}")
    parameters[f"{output_names[0]}_output"] = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        # Logical C1 is physically padded to C16 by MLA publication.
        dram_layout=TensorDRAMLayout.HWC16,
    )
    return parameters


def _audit_strict_model(model: Any) -> dict[str, Any]:
    import copy

    from afe.backends.mla.afe_to_n2a_compiler.n2a_compiler_operations import (
        check_single_mla_stage_contract,
        get_backend_stage_distribution,
        translate_sub_awesome_net_to_modelgraph,
    )
    from afe.ir.tensor_type import ScalarType
    from afe.quantization.context import activate_quantization_compile_context

    top = model._net
    public_input = top.nodes[top.input_node_names[0]].get_type().output.value
    public_output = top.nodes[top.output_node_name].get_type().output.value
    if public_input.scalar != ScalarType.int8 or tuple(public_input.shape) != INPUT_SHAPE_NHWC:
        raise RuntimeError(f"strict public input changed: {public_input}")
    if public_output.scalar != ScalarType.int8 or tuple(public_output.shape) != OUTPUT_SHAPE_NHWC:
        raise RuntimeError(f"strict public output changed: {public_output}")
    with activate_quantization_compile_context(top):
        backend = translate_sub_awesome_net_to_modelgraph(copy.deepcopy(top))
    check_single_mla_stage_contract(backend, context="depthart_base")
    distribution = get_backend_stage_distribution(backend)
    return {
        "public_input": {"dtype": "int8", "shape": list(INPUT_SHAPE_NHWC)},
        "public_output": {"dtype": "int8", "shape": list(OUTPUT_SHAPE_NHWC)},
        "stage_distribution": {str(key): int(value) for key, value in distribution.items()},
    }


def compile_variant(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    parameters = variant["parameters"]
    source_report = audit_source(source)
    profile_report = audit_profile(PROFILE, source_report["sha256"])
    snapshot_spec = SourceSpec.from_mapping(parameters["quantized_snapshot"])
    snapshot_path, _cache_hit = Fetcher().fetch(snapshot_spec)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    # The fully unrolled scan is deep but acyclic. The product compiler build
    # owns deep-graph serialization; this public recursion limit only prevents
    # Python-side graph traversal from imposing the default depth of 1000.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
    with tempfile.TemporaryDirectory(prefix="depthart-quantized-") as temporary:
        staging = Path(temporary)
        sima, snapshot_receipt = _safe_snapshot(snapshot_path, staging)
        internal_name, _staged = _stage_for_model_load(sima, staging)
        from afe.apis.error_handling_variables import enable_verbose_error_messages
        from afe.apis.model import Model

        enable_verbose_error_messages()
        model = Model.load(model_name=internal_name, network_directory=str(staging))
        if model is None:
            raise RuntimeError("Model Compiler returned no model for the snapshot")
        strict_report = _audit_strict_model(model)
        model.compile(
            output_path=str(output_dir),
            batch_size=1,
            log_level=logging.INFO,
            tessellate_parameters=_tessellation(model),
            require_single_mla=True,
        )

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK, found {archives}")
    archive = archives[0]
    audit_path = output_dir / "depthart-base-compile-audit.json"
    report = {
        "schema": "depthart-registry-compile/v1",
        "model_id": "depthart_base",
        "variant_id": variant_id,
        "source": source_report,
        "compile_profile": profile_report,
        "quantized_snapshot": {
            "archive_sha256": snapshot_spec.sha256,
            "receipt": snapshot_receipt,
        },
        "strict_model": strict_report,
        "mpk": {"sha256": sha256_file(archive)[0], "size_bytes": archive.stat().st_size},
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    LOG.info("Compiled MPK: %s", archive)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_variant(source, args.output_dir.expanduser().resolve(), args.variant)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
