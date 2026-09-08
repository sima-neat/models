# Vendored from sima-neat/vulcan scripts/onnx-conversion/artifacts.py
# Revision: 7bdc42cb662b5c5c0e73092b430bd2b40dc7e1ec
# Upstream SHA-256: 53a5c14a8a802cd200fcfb9d5e2f00a5c16e336a45e84042e5d61bd334d1a249
# Local adaptation: defer botocore import to the publishing function.
"""Immutable conversion bundles, shared by smoke and model conversion callers.

Publication uses conditional object creation; manifest.json is the completion
record. Consumers pin its SHA-256 and validate every file before execution.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PART_SIZE = 1024**3



def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def safe_path(name):
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or any(v in ("", ".", "..") for v in name.split("/"))
        or "\\" in name
    ):
        raise ValueError(f"Unsafe artifact path: {name}")
    return p


def package(directory, provenance):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            safe_path(name)
            files[name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    if "model.onnx" not in files or "validation.json" not in files:
        raise ValueError("Missing ONNX or validation report")
    if (
        json.loads((directory / "validation.json").read_text()).get("status")
        != "passed"
    ):
        raise ValueError("Only validated exports can be packaged")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "files": files,
        "provenance": provenance,
    }
    # Store uncompressed: large tensor payloads should not require a second RAM copy.
    bundle = directory.parent / "onnx-bundle.tar"
    with tarfile.open(bundle, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, metadata in files.items():
            # Filesystem timestamps, ownership and modes differ after Spot retries.
            # Construct headers from content metadata only, with a stable file mode.
            entry = tarfile.TarInfo(name)
            entry.size = metadata["size_bytes"]
            entry.mode = 0o644
            with (directory / name).open("rb") as source:
                archive.addfile(entry, source)
    manifest["bundle"] = {
        "name": bundle.name,
        "sha256": digest(bundle),
        "size_bytes": bundle.stat().st_size,
    }
    return bundle, manifest


def verify_extract(bundle, manifest, destination):
    if manifest.get("status") != "completed" or manifest.get("schema_version") != 1:
        raise ValueError("Artifact is not completed")
    bundle = Path(bundle)
    if (
        digest(bundle) != manifest["bundle"]["sha256"]
        or bundle.stat().st_size != manifest["bundle"]["size_bytes"]
    ):
        raise ValueError("Bundle checksum/size mismatch")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(bundle) as archive:
        for member in archive:
            safe_path(member.name)
            if (
                not member.isfile()
                or member.name not in manifest["files"]
                or member.name in seen
            ):
                raise ValueError("Unexpected, duplicate or non-file archive member")
            expected = manifest["files"][member.name]
            if member.size != expected["size_bytes"]:
                raise ValueError("File size mismatch")
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with archive.extractfile(member) as source, path.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(chunk)
                    output.write(chunk)
            if h.hexdigest() != expected["sha256"]:
                raise ValueError("File checksum mismatch")
            seen.add(member.name)
    if seen != set(manifest["files"]):
        raise ValueError("Missing bundle files")
    return destination


def put_immutable(s3, bucket, key, body, sha256, size, kms_key):
    """Same bytes replay successfully; a differing replay fails, never overwrites."""
    from botocore.exceptions import ClientError

    expected = base64.b64encode(bytes.fromhex(sha256)).decode()
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentLength=size,
            ChecksumSHA256=expected,
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=kms_key,
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
            raise
    head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if (
        head.get("ChecksumSHA256") != expected
        or head["ContentLength"] != size
        or head.get("ServerSideEncryption") != "aws:kms"
        or not head.get("VersionId")
        or head.get("VersionId") == "null"
        or head.get("SSEKMSKeyId") != kms_key
    ):
        raise ValueError(
            "Immutable object conflict or invalid encryption/version/checksum"
        )
    return {
        "key": key,
        "version_id": head["VersionId"],
        "sha256": sha256,
        "size_bytes": size,
    }


def publish(s3, bucket, prefix, bundle, manifest, kms_key, journal):
    safe_path(prefix.rstrip("/"))
    if not prefix.startswith("models-onnx/") or len(prefix.rstrip("/").split("/")) < 4:
        raise ValueError("Expected versioned models-onnx prefix")
    prefix = prefix.rstrip("/") + "/"
    bundle = Path(bundle)
    result = dict(manifest)
    result["objects"] = []
    # Each part remains below S3's single PUT limit and has a native full SHA-256.
    # Avoid multipart composite checksums or buffering a model in memory.
    with bundle.open("rb") as source:
        part = 0
        while True:
            with tempfile.TemporaryFile() as body:
                size = 0
                checksum = hashlib.sha256()
                while size < PART_SIZE:
                    chunk = source.read(min(1024 * 1024, PART_SIZE - size))
                    if not chunk:
                        break
                    body.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
                if not size:
                    break
                body.seek(0)
                name = (
                    bundle.name
                    if bundle.stat().st_size <= PART_SIZE
                    else f"{bundle.name}.part{part:05d}"
                )
                obj = put_immutable(
                    s3, bucket, prefix + name, body, checksum.hexdigest(), size, kms_key
                )
                result["objects"].append(obj)
                journal.append(obj)
                part += 1
    # Commit last. The digest pins the exact payload versions as well as model files.
    raw = canonical(result)
    obj = put_immutable(
        s3,
        bucket,
        prefix + "manifest.json",
        io.BytesIO(raw),
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        kms_key,
    )
    journal.append(obj)
    return result, obj


def assemble_download(directory, manifest):
    """Verify downloaded parts in manifest order and reconstruct a large bundle."""
    directory = Path(directory)
    name = manifest["bundle"]["name"]
    safe_path(name)
    objects = manifest["objects"]
    if not objects or len({obj["key"] for obj in objects}) != len(objects):
        raise ValueError("Invalid payload object list")
    paths = []
    for obj in objects:
        safe_path(obj["key"])
        path = directory / PurePosixPath(obj["key"]).name
        if path.stat().st_size != obj["size_bytes"] or digest(path) != obj["sha256"]:
            raise ValueError("Downloaded part checksum/size mismatch")
        paths.append(path)
    destination = directory / name
    if len(paths) == 1 and paths[0] == destination:
        return destination
    if destination in paths:
        raise ValueError("Bundle name conflicts with part name")
    with destination.open("xb") as output:
        for path in paths:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    return destination
