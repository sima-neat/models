"""Safely select a qualified precision source from the immutable QAT bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from onnx_audit import audit_onnx, sha256_file

ARCHITECTURE = "torchvision.ssdlite320_mobilenet_v3_large"
MANIFEST_NAME = "manifest.json"
MAX_MEMBER_SIZE = 256 * 1024 * 1024


def _read_members(bundle: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 1
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"unsafe SSDLite source bundle member: {member.name}")
            if member.size > MAX_MEMBER_SIZE:
                raise ValueError(f"oversized SSDLite source bundle member: {member.name}")
            if member.name in members:
                raise ValueError(f"duplicate SSDLite source bundle member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read SSDLite source bundle member: {member.name}")
            members[member.name] = stream.read()
    return members


def read_manifest(bundle: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    members = _read_members(bundle)
    if MANIFEST_NAME not in members:
        raise ValueError("SSDLite source bundle has no manifest.json")
    try:
        manifest = json.loads(members[MANIFEST_NAME])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SSDLite source bundle manifest is invalid") from exc
    if manifest.get("schema_version") != 1 or manifest.get("architecture") != ARCHITECTURE:
        raise ValueError("SSDLite source bundle identity is invalid")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"bf16", "int8"}:
        raise ValueError("SSDLite source bundle must declare exactly BF16 and INT8 sources")
    if not all(isinstance(value, dict) for value in sources.values()):
        raise ValueError("SSDLite source bundle source declarations are invalid")
    declared = {value.get("file") for value in sources.values()}
    expected = {MANIFEST_NAME, *declared}
    if None in declared or len(declared) != 2 or set(members) != expected:
        raise ValueError(f"SSDLite source bundle members do not match its manifest: {sorted(members)}")
    return manifest, members


def validate_accuracy(manifest: dict[str, Any], *, allow_unqualified: bool) -> None:
    accuracy = manifest.get("accuracy")
    required = {"fp32_ap", "qat_ap", "ap_drop_points"}
    if not isinstance(accuracy, dict) or set(accuracy) != required:
        if allow_unqualified:
            return
        raise ValueError("SSDLite source bundle has no complete COCO QAT qualification")
    fp32_ap = float(accuracy["fp32_ap"])
    qat_ap = float(accuracy["qat_ap"])
    ap_drop_points = float(accuracy["ap_drop_points"])
    if not all(math.isfinite(value) for value in (fp32_ap, qat_ap, ap_drop_points)):
        raise ValueError("SSDLite source bundle contains non-finite COCO accuracy")
    if not 0.0 <= fp32_ap <= 1.0 or not 0.0 <= qat_ap <= 1.0:
        raise ValueError("SSDLite source bundle COCO AP must be in [0, 1]")
    calculated_drop = (fp32_ap - qat_ap) * 100.0
    if not math.isclose(ap_drop_points, calculated_drop, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("SSDLite source bundle COCO AP-drop evidence is inconsistent")
    if ap_drop_points > 1.5:
        raise ValueError(f"SSDLite source bundle exceeds the 1.5 AP-point QAT gate: {accuracy}")


def prepare_source(
    bundle: Path,
    destination: Path,
    precision: str,
    *,
    allow_unqualified: bool = False,
) -> Path:
    if precision not in {"int8", "bf16"}:
        raise ValueError(f"unsupported SSDLite bundle precision: {precision}")
    manifest, members = read_manifest(bundle)
    validate_accuracy(manifest, allow_unqualified=allow_unqualified)
    source = manifest["sources"][precision]
    name = source["file"]
    payload = members[name]
    expected_sha256 = source.get("audit", {}).get("onnx_sha256")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"SSDLite {precision} source digest mismatch: expected {expected_sha256}, found {observed_sha256}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        report = audit_onnx(temporary, expect_qat=precision == "int8")
        if report["onnx_sha256"] != expected_sha256 or sha256_file(temporary) != expected_sha256:
            raise RuntimeError(f"SSDLite {precision} source changed during preparation")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
