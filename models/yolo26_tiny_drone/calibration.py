#!/usr/bin/env python3
"""Build and validate the immutable YOLO26 tiny-drone calibration snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

ARCHIVE_ROOT = PurePosixPath("yolo26_tiny_drone_calibration_v1")
MANIFEST_NAME = ARCHIVE_ROOT / "manifest.json"
EXPECTED_IMAGES = 128
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
            raise ValueError(f"unsafe calibration member: {member.name}")
        total_size += member.size
        if total_size > MAX_ARCHIVE_BYTES:
            raise ValueError("calibration archive exceeds the uncompressed size limit")
        members[member.name] = member
    return members


def _read_member(stream: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = stream.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read calibration member: {member.name}")
    return source.read()


def validate_calibration(
    archive: Path,
    contract: Any,
    *,
    expected_tensor_sha256: str,
    preprocess_rgb_image: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify all source bytes and reconstruct the exact NHWC FP32 tensor."""
    with tarfile.open(archive, "r:gz") as stream:
        members = _safe_members(stream)
        manifest_member = members.get(str(MANIFEST_NAME))
        if manifest_member is None:
            raise ValueError(f"calibration archive is missing {MANIFEST_NAME}")
        try:
            manifest = json.loads(_read_member(stream, manifest_member))
        except json.JSONDecodeError as exc:
            raise ValueError("calibration manifest is not valid JSON") from exc
        images = manifest.get("images")
        expected_preprocessing = f"rgb_letterbox_{contract.model_size}_fill_114_divide_255_nhwc_float32"
        if (
            manifest.get("schema_version") != 1
            or manifest.get("count") != EXPECTED_IMAGES
            or manifest.get("preprocessing") != expected_preprocessing
            or manifest.get("tensor_sha256") != expected_tensor_sha256
            or not isinstance(images, list)
            or len(images) != EXPECTED_IMAGES
        ):
            raise ValueError("calibration manifest does not match the deployment contract")

        expected_names = {str(MANIFEST_NAME)}
        samples: list[np.ndarray] = []
        observed_images: list[dict[str, str]] = []
        for index, declaration in enumerate(images):
            if not isinstance(declaration, dict):
                raise ValueError(f"calibration image {index} is not an object")
            relative = PurePosixPath(str(declaration.get("file", "")))
            digest = str(declaration.get("sha256", ""))
            sequence = str(declaration.get("sequence", ""))
            expected_prefix = ARCHIVE_ROOT / "images" / sequence
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parent != expected_prefix
                or not sequence
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"invalid calibration image declaration at index {index}")
            name = str(relative)
            if name in expected_names or name not in members:
                raise ValueError(f"missing or duplicate calibration image: {name}")
            expected_names.add(name)
            payload = _read_member(stream, members[name])
            if _sha256_bytes(payload) != digest:
                raise ValueError(f"calibration image digest mismatch: {name}")
            with Image.open(io.BytesIO(payload)) as source:
                samples.append(preprocess_rgb_image(source.convert("RGB"), contract))
            observed_images.append({"file": name, "sequence": sequence, "sha256": digest})

        if set(members) != expected_names:
            unexpected = sorted(set(members) - expected_names)
            raise ValueError(f"calibration archive has unexpected members: {unexpected}")

    calibration = np.stack(samples)
    expected_shape = (EXPECTED_IMAGES, contract.model_size, contract.model_size, 3)
    if calibration.shape != expected_shape or calibration.dtype != np.float32:
        raise ValueError(f"invalid calibration tensor: shape={calibration.shape}, dtype={calibration.dtype}")
    tensor_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(calibration)).cast("B")).hexdigest()
    if tensor_sha256 != expected_tensor_sha256:
        raise ValueError(
            f"calibration tensor digest mismatch: expected {expected_tensor_sha256}, found {tensor_sha256}"
        )
    evidence = {
        "archive": archive.name,
        "count": EXPECTED_IMAGES,
        "images": observed_images,
        "preprocessing": expected_preprocessing,
        "tensor_sha256": tensor_sha256,
    }
    return calibration, evidence


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


def build_archive(compile_report: Path, output: Path) -> dict[str, Any]:
    """Create a deterministic archive from a qualified compilation report."""
    report = json.loads(compile_report.read_text(encoding="utf-8"))
    calibration = report.get("calibration", {})
    declarations = calibration.get("images")
    if (
        report.get("schema_version") != 1
        or report.get("architecture") != "yolo26n-p2-attention-one-class"
        or not report.get("qualification_eligible")
        or calibration.get("count") != EXPECTED_IMAGES
        or not isinstance(declarations, list)
        or len(declarations) != EXPECTED_IMAGES
    ):
        raise ValueError("compile report is not a qualified 128-image YOLO26 build")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, Path, bytes, dict[str, str]]] = []
    names: set[str] = set()
    for index, declaration in enumerate(declarations):
        source = Path(str(declaration.get("path", ""))).resolve()
        sequence = str(declaration.get("sequence", ""))
        digest = str(declaration.get("sha256", ""))
        relative = ARCHIVE_ROOT / "images" / sequence / source.name
        name = str(relative)
        if not source.is_file() or not sequence or name in names:
            raise ValueError(f"invalid calibration source at index {index}: {source}")
        payload = source.read_bytes()
        if _sha256_bytes(payload) != digest:
            raise ValueError(f"calibration source digest mismatch: {source}")
        names.add(name)
        images.append(
            (
                name,
                source,
                payload,
                {"file": name, "sequence": sequence, "sha256": digest},
            )
        )

    manifest = {
        "schema_version": 1,
        "count": EXPECTED_IMAGES,
        "preprocessing": calibration["preprocessing"],
        "tensor_sha256": calibration["tensor_sha256"],
        "images": [declaration for _name, _source, _payload, declaration in images],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    try:
        with output.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as stream:
                    stream.addfile(_tar_info(str(MANIFEST_NAME), len(manifest_bytes)), io.BytesIO(manifest_bytes))
                    for name, _source, payload, _declaration in images:
                        stream.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(args.compile_report.expanduser().resolve(), args.output.expanduser().resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
