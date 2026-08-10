from __future__ import annotations

import re
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
GITHUB_RELEASE_RE = re.compile(r"^/[^/]+/[^/]+/releases/download/[^/]+/[^/]+$")
GITHUB_RELEASE_API_RE = re.compile(r"^/repos/[^/]+/[^/]+/releases/assets/[0-9]+$")
DEFAULT_HTTPS_HOSTS = {"api.github.com", "github.com", "huggingface.co"}


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
        hosts = allowed_https_hosts or DEFAULT_HTTPS_HOSTS
        hostname = (parts.hostname or "").lower()
        if hostname not in hosts:
            raise SourcePolicyError(f"HTTPS source host is not approved: {hostname or '<missing>'}")
        if hostname == "huggingface.co" and not HF_RESOLVE_RE.fullmatch(parts.path):
            raise SourcePolicyError("Hugging Face HTTPS sources must use a full commit in a resolve URL")
        if hostname == "github.com" and not GITHUB_RELEASE_RE.fullmatch(parts.path):
            raise SourcePolicyError("GitHub sources must be release assets")
        if hostname == "api.github.com" and not GITHUB_RELEASE_API_RE.fullmatch(parts.path):
            raise SourcePolicyError("GitHub API sources must identify a release asset")
        return self.uri
