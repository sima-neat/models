from __future__ import annotations

import hashlib
import os
import ssl
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from .errors import FetchError
from .source import SourceSpec, redact_url

OpenUrl = Callable[..., BinaryIO]


def open_verified_url(request: Request, timeout: int) -> BinaryIO:
    context = ssl.create_default_context(cafile=certifi.where())
    return urlopen(request, timeout=timeout, context=context)


def default_cache_dir() -> Path:
    configured = os.environ.get("MODEL_REGISTRY_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "model-registry").resolve()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class Fetcher:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        opener: OpenUrl | None = None,
        allowed_https_hosts: set[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir()
        self.opener = opener or open_verified_url
        self.allowed_https_hosts = allowed_https_hosts

    def cache_path(self, source: SourceSpec) -> Path:
        return self.cache_dir / "sha256" / source.sha256[:2] / source.sha256 / source.filename

    def _verify(self, path: Path, source: SourceSpec) -> None:
        digest, size = sha256_file(path)
        if size != source.size_bytes:
            raise FetchError(f"size mismatch for {source.filename}: expected {source.size_bytes}, got {size}")
        if digest != source.sha256:
            raise FetchError(f"SHA-256 mismatch for {source.filename}: expected {source.sha256}, got {digest}")

    def fetch(self, source: SourceSpec) -> tuple[Path, bool]:
        destination = self.cache_path(source)
        if destination.is_file():
            try:
                self._verify(destination, source)
                return destination, True
            except FetchError:
                destination.unlink()

        destination.parent.mkdir(parents=True, exist_ok=True)
        url = source.resolve_url(self.allowed_https_hosts)
        request = Request(url, headers={"User-Agent": "sima-neat-model-registry/0.1"})
        hostname = urlsplit_hostname(url)
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        github_token = os.environ.get("GITHUB_TOKEN", "").strip()
        if hf_token and hostname == "huggingface.co":
            # urllib does not copy unredirected headers to Hugging Face's signed
            # CDN redirect, so the bearer token remains scoped to huggingface.co.
            request.add_unredirected_header("Authorization", f"Bearer {hf_token}")
        if github_token and hostname in {"api.github.com", "github.com"}:
            # Private release downloads redirect to a signed object URL. Keep
            # repository credentials scoped to GitHub; redirect authorization
            # is carried by the short-lived signature.
            request.add_unredirected_header("Authorization", f"Bearer {github_token}")
            request.add_header("Accept", "application/octet-stream")

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{source.filename}.", suffix=".partial", dir=destination.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                with self.opener(request, timeout=60) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            self._verify(temporary, source)
            os.replace(temporary, destination)
            temporary = None
            return destination, False
        except (HTTPError, URLError, OSError, TimeoutError, FetchError) as exc:
            raise FetchError(f"failed to fetch {redact_url(url)}: {exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def urlsplit_hostname(url: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower()
