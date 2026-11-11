from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from .errors import SourcePolicyError

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
HF_URI_RE = re.compile(
    r"^hf://(?P<repo_type>models|datasets|spaces)/"
    r"(?P<owner>[^/@]+)/(?P<repo>[^/@]+)@(?P<revision>[a-f0-9]{40})/"
    r"(?P<path>.+)$"
)
HF_RESOLVE_RE = re.compile(r"^/(?:(?:datasets|spaces)/)?[^/]+/[^/]+/resolve/[a-f0-9]{40}/.+$")
VULCAN_ONNX_RE = re.compile(r"^/models-onnx/[a-z][a-z0-9_]*/[a-f0-9]{64}/[0-9]+-[0-9]+/model\.onnx$")
GITHUB_RELEASE_RE = re.compile(
    r"^/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/releases/download/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$"
)
PYPI_WHEEL_RE = re.compile(r"^/packages/[a-f0-9]{2}/[a-f0-9]{2}/[a-f0-9]{60}/[A-Za-z0-9._+-]+\.whl$")


def _require_hugging_face_commit(path: str) -> None:
    if not HF_RESOLVE_RE.fullmatch(path):
        raise SourcePolicyError("Hugging Face HTTPS sources must use a full commit in a resolve URL")


def _require_vulcan_onnx(path: str) -> None:
    if not VULCAN_ONNX_RE.fullmatch(path):
        raise SourcePolicyError("Vulcan sources must identify a versioned ONNX file")


def _require_github_release_file(path: str) -> None:
    if not GITHUB_RELEASE_RE.fullmatch(path):
        raise SourcePolicyError("GitHub sources must be a release file addressed by its tag")


def _require_pypi_wheel(path: str) -> None:
    if not PYPI_WHEEL_RE.fullmatch(path):
        raise SourcePolicyError("PyPI sources must be a wheel addressed by its hashed package path")


# Every approved HTTPS host and the check its URL path must pass. Approving a host is one entry here.
HTTPS_HOST_CHECKS: dict[str, Callable[[str], None]] = {
    "artifacts.stg.neat.sima.ai": _require_vulcan_onnx,
    "files.pythonhosted.org": _require_pypi_wheel,
    "github.com": _require_github_release_file,
    "huggingface.co": _require_hugging_face_commit,
}


def source_identity(source: dict[str, Any]) -> str:
    """Return the canonical content identity of a single file or file set."""
    if source["type"] in {"file", "conversion"}:
        return str(source["sha256"])
    members = [
        {
            "path": member["path"],
            "sha256": member["sha256"],
            "size_bytes": member["size_bytes"],
            "media_type": member.get("media_type", "application/octet-stream"),
        }
        for member in sorted(source["files"], key=lambda item: item["path"])
    ]
    # A resource's file set has no entrypoint; a compile source's identity keeps the exact document it had.
    document = {"files": members}
    if "entrypoint" in source:
        document["entrypoint"] = source["entrypoint"]
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def redact_url(value: str) -> str:
    """Remove credentials and query/fragment data before displaying a URL."""
    parts = urlsplit(value)
    if not parts.scheme:
        return "<source>"
    hostname = parts.hostname or ""
    if parts.port:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


@dataclass(frozen=True)
class SourceSpec:
    uri: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SourceSpec:
        uri = str(value["uri"])
        digest = str(value["sha256"])
        size = int(value["size_bytes"])
        media_type = str(value.get("media_type", "application/octet-stream"))
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")
        if size < 1:
            raise ValueError("size_bytes must be positive")
        spec = cls(uri=uri, sha256=digest, size_bytes=size, media_type=media_type)
        spec.resolve_url()
        return spec

    @property
    def filename(self) -> str:
        match = HF_URI_RE.fullmatch(self.uri)
        raw_path = match.group("path") if match else urlsplit(self.uri).path
        filename = PurePosixPath(raw_path).name
        if not filename or filename in {".", ".."}:
            raise SourcePolicyError("source URI must end with a filename")
        return filename

    def resolve_url(self, allowed_https_hosts: set[str] | None = None) -> str:
        match = HF_URI_RE.fullmatch(self.uri)
        if match:
            repo_type = match.group("repo_type")
            prefix = "" if repo_type == "models" else f"{repo_type}/"
            owner = quote(match.group("owner"), safe="")
            repo = quote(match.group("repo"), safe="")
            revision = match.group("revision")
            raw_path = match.group("path")
            if any(part in {"", ".", ".."} for part in raw_path.split("/")):
                raise SourcePolicyError("Hugging Face source path must not contain empty, dot, or parent segments")
            path = quote(raw_path, safe="/")
            return f"https://huggingface.co/{prefix}{owner}/{repo}/resolve/{revision}/{path}"

        parts = urlsplit(self.uri)
        if parts.scheme != "https":
            raise SourcePolicyError("source URI scheme must be hf or https")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise SourcePolicyError("source URI must not contain credentials, query parameters, or fragments")
        if parts.port not in {None, 443}:
            raise SourcePolicyError("HTTPS source URI must use the default TLS port")
        if any(part in {".", ".."} for part in parts.path.split("/")):
            raise SourcePolicyError("HTTPS source path must not contain dot or parent segments")
        hosts = allowed_https_hosts or set(HTTPS_HOST_CHECKS)
        hostname = (parts.hostname or "").lower()
        if hostname not in hosts:
            raise SourcePolicyError(f"HTTPS source host is not approved: {hostname or '<missing>'}")
        require_path = HTTPS_HOST_CHECKS.get(hostname)
        if require_path:
            require_path(parts.path)
        return self.uri
