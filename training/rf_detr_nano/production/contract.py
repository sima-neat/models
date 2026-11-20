"""Verified local handoff inputs; no source hosting or registry download logic."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from numerics import Int8Grid

COMPONENTS = ("backbone", "decoder")
SHAPES = {
    "backbone": {"inputs": [(1, 384, 384, 3)], "outputs": [(1, 1, 576, 256), (1, 1, 1, 576), (1, 576, 4)]},
    "decoder": {"inputs": [(1, 300, 4), (1, 1, 576, 256)], "outputs": [(1, 300, 4), (1, 300, 91), (1, 300, 4)]},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 identity")
    return value


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != require_sha(expected_sha256):
        raise ValueError("handoff contract checksum mismatch")
    value = json.loads(path.read_text())
    required = {
        "schema",
        "checkpoint_sha256",
        "reference_sha256",
        "input_grid",
        "output_grids",
        "components",
        "topk",
        "preprocessing",
        "native_seals",
    }
    if set(value) != required or value["schema"] != "rfdetr.nano_qat_handoff/v1":
        raise ValueError("unknown or incomplete QAT handoff contract")
    require_sha(value["checkpoint_sha256"])
    require_sha(value["reference_sha256"])
    if set(value["components"]) != set(COMPONENTS):
        raise ValueError("handoff must bind both backbone and decoder")
    if set(value["native_seals"]) != set(COMPONENTS):
        raise ValueError("handoff must bind both native numerical seals")
    for component in COMPONENTS:
        require_sha(value["components"][component]["sha256"])
        for field in ("native_graph_sha256", "source_contract_sha256", "numerical_plan_sha256"):
            require_sha(value["native_seals"][component][field])
        for role in ("inputs", "outputs"):
            ports = value["components"][component][role]
            if any(any(type(size) is not int or size < 1 for size in port["shape"]) for port in ports):
                raise ValueError("port dimensions must be positive integers")
            if any(not port["name"] for port in ports) or len({port["name"] for port in ports}) != len(ports):
                raise ValueError("each public port requires a unique name")
            if [tuple(port["shape"]) for port in ports] != SHAPES[component][role]:
                raise ValueError(f"unexpected {component} {role} shapes/order")
            if any(port["dtype"] != "int8" for port in ports):
                raise ValueError("this handoff admits only its captured INT8 public tensor ports")
    if value["topk"] != {"k": 300, "ties": "lower_original_index_first", "exhaustion": "MaskMinRepeatZero"}:
        raise ValueError("unrecognized proposal-selection contract")
    if value["preprocessing"] != {
        "stages": ["Rescale", "Resize", "Normalize", "Quantize"],
        "resize": "BILINEAR_ANTIALIASED",
        "rescale_float32_bits": 998277249,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }:
        raise ValueError("unrecognized ordered RF-DETR source preprocessing")
    input_grid(value)
    output_grids(value)
    return value


def input_grid(contract: dict[str, Any]) -> Int8Grid:
    return Int8Grid(**contract["input_grid"])


def output_grids(contract: dict[str, Any]) -> tuple[Int8Grid, ...]:
    values = contract["output_grids"]
    if [value["name"] for value in values] != ["box_deltas", "labels", "box_references"]:
        raise ValueError("output grids must name deltas, logits and references in that order")
    return tuple(Int8Grid(value["step_bits"], value["zero_point"]) for value in values)


def verify_packages(contract: dict[str, Any], packages: dict[str, Path]) -> None:
    if set(packages) != set(COMPONENTS):
        raise ValueError("exactly --mpk backbone=... --mpk decoder=... are required")
    for component, path in packages.items():
        if sha256_file(path) != contract["components"][component]["sha256"]:
            raise ValueError(f"{component} package does not match this checkpoint's handoff")


def parse_components(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in COMPONENTS or name in result or not path:
            raise ValueError("components must be unique backbone=path and decoder=path pairs")
        result[name] = Path(path)
    if set(result) != set(COMPONENTS):
        raise ValueError("both backbone and decoder are required")
    return result
