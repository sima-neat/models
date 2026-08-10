#!/usr/bin/env python3
"""Build and validate the qualified ModelSDK quantized-model snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_ROOT = PurePosixPath("yolo26_tiny_drone_quantized_v1")
MANIFEST_NAME = ARCHIVE_ROOT / "manifest.json"
MODEL_NAME = "yolo26n_p2_tiny_drone_int8_qat_b1"
MODEL_FILES = (
    f"{MODEL_NAME}.sima",
    f"{MODEL_NAME}.sima.json",
    f"{MODEL_NAME}.fp32.sima",
    f"{MODEL_NAME}.fp32.sima.json",
)
EXPECTED_MODEL_SDK_VERSION = "2.1.3"
EXPECTED_ONNX_SHA256 = "2627762fc29dbd377a3f471a3ab0407569dd39c13cd324d46ca426629ec80f66"
EXPECTED_CALIBRATION_SHA256 = "8418b35d20ced1a9135eb8862e5554544f261023963ec6290e587057d2b11b2b"
EXPECTED_CALIBRATION_TENSOR_SHA256 = "d8830d19b33fe336c5c2cec4822c3b278be4169d125ca249be476771be5b544e"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _safe_members(stream: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    total_size = 0
    for member in stream.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or not member.isfile()
            or member.name in members
        ):
            raise ValueError(f"unsafe quantized snapshot member: {member.name}")
        total_size += member.size
        if total_size > MAX_ARCHIVE_BYTES:
            raise ValueError("quantized snapshot exceeds the uncompressed size limit")
        members[member.name] = member
    return members


def _read_member(stream: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = stream.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read quantized snapshot member: {member.name}")
    return source.read()


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("quantized snapshot manifest is not an object")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model_name") != MODEL_NAME
        or manifest.get("model_sdk_version") != EXPECTED_MODEL_SDK_VERSION
        or manifest.get("onnx_sha256") != EXPECTED_ONNX_SHA256
        or manifest.get("calibration_sha256") != EXPECTED_CALIBRATION_SHA256
        or manifest.get("calibration_tensor_sha256") != EXPECTED_CALIBRATION_TENSOR_SHA256
        or not isinstance(files, dict)
        or set(files) != set(MODEL_FILES)
    ):
        raise ValueError("quantized snapshot manifest does not match the qualified model")
    for name, digest in files.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid digest for quantized snapshot member: {name}")
    return manifest


def materialize_snapshot(archive: Path, output_dir: Path) -> dict[str, Any]:
    """Validate an immutable snapshot and materialize only its declared files."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite snapshot directory: {output_dir}")
    with tarfile.open(archive, "r:gz") as stream:
        members = _safe_members(stream)
        manifest_member = members.get(str(MANIFEST_NAME))
        if manifest_member is None:
            raise ValueError(f"quantized snapshot is missing {MANIFEST_NAME}")
        try:
            manifest = _validate_manifest(json.loads(_read_member(stream, manifest_member)))
        except json.JSONDecodeError as exc:
            raise ValueError("quantized snapshot manifest is not valid JSON") from exc

        expected_names = {str(MANIFEST_NAME)} | {str(ARCHIVE_ROOT / name) for name in MODEL_FILES}
        if set(members) != expected_names:
            raise ValueError("quantized snapshot has missing or unexpected members")
        payloads: dict[str, bytes] = {}
        for name in MODEL_FILES:
            payload = _read_member(stream, members[str(ARCHIVE_ROOT / name)])
            if _sha256_bytes(payload) != manifest["files"][name]:
                raise ValueError(f"quantized snapshot member digest mismatch: {name}")
            payloads[name] = payload

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for name, payload in payloads.items():
            (output_dir / name).write_bytes(payload)
    except Exception:
        for name in payloads:
            (output_dir / name).unlink(missing_ok=True)
        output_dir.rmdir()
        raise
    return manifest


def build_archive(compile_report: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    """Create a deterministic snapshot from an already-qualified ModelSDK model."""
    report = json.loads(compile_report.read_text(encoding="utf-8"))
    calibration = report.get("calibration", {})
    quantized = report.get("quantized_model", {})
    attention = quantized.get("attention", {})
    if (
        report.get("schema_version") != 1
        or report.get("architecture") != "yolo26n-p2-attention-one-class"
        or report.get("qualification_eligible") is not True
        or report.get("source_qualification_eligible") is not True
        or report.get("source", {}).get("onnx_sha256") != EXPECTED_ONNX_SHA256
        or calibration.get("count") != 128
        or calibration.get("tensor_sha256") != EXPECTED_CALIBRATION_TENSOR_SHA256
        or quantized.get("compute_convolution_count") != 129
        or quantized.get("compute_scalar_type") != "int8"
        or quantized.get("weight_quantization") != "per_channel_int8"
        or attention.get("BatchMatmul", {}).get("count") != 8
        or attention.get("BatchMatmul", {}).get("input_scalar_type") != "int8"
        or attention.get("BatchMatmul", {}).get("output_scalar_type") != "int8"
        or attention.get("Softmax", {}).get("count") != 4
        or attention.get("Softmax", {}).get("input_scalar_type") != "int8"
        or attention.get("Softmax", {}).get("output_scalar_type") != "int8"
        or report.get("mpk", {}).get("compiler_version") != EXPECTED_MODEL_SDK_VERSION
    ):
        raise ValueError("compile report is not the qualified YOLO26 INT8 QAT model")

    payloads: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for name in MODEL_FILES:
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"quantized model file does not exist: {path}")
        payload = path.read_bytes()
        payloads[name] = payload
        digests[name] = _sha256_bytes(payload)
    if digests[f"{MODEL_NAME}.sima.json"] != quantized.get("quantized_model_json_sha256"):
        raise ValueError("quantized model JSON does not match its qualification report")

    manifest = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "model_sdk_version": EXPECTED_MODEL_SDK_VERSION,
        "onnx_sha256": EXPECTED_ONNX_SHA256,
        "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
        "calibration_tensor_sha256": EXPECTED_CALIBRATION_TENSOR_SHA256,
        "files": digests,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite quantized snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as stream:
                    stream.addfile(_tar_info(str(MANIFEST_NAME), len(manifest_bytes)), io.BytesIO(manifest_bytes))
                    for name in MODEL_FILES:
                        payload = payloads[name]
                        stream.addfile(_tar_info(str(ARCHIVE_ROOT / name), len(payload)), io.BytesIO(payload))
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(
        args.compile_report.expanduser().resolve(),
        args.source_dir.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
