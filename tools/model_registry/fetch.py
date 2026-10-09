from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from http.client import HTTPException
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from .errors import FetchError
from .source import SourceSpec, redact_url, source_identity

OpenUrl = Callable[..., BinaryIO]
Sleeper = Callable[[float], None]


class _SizeMismatch(FetchError):
    """A transfer ended before the declared payload size was received."""


class _ChecksumMismatch(FetchError):
    """A complete payload did not match its immutable digest."""


class _TransferError(FetchError):
    """A transport operation failed before the payload reached local storage."""


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
        max_attempts: int = 3,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir()
        self.opener = opener or open_verified_url
        self.allowed_https_hosts = allowed_https_hosts
        self.max_attempts = max_attempts
        self.sleeper = sleeper

    def cache_path(self, source: SourceSpec) -> Path:
        return self.cache_dir / "sha256" / source.sha256[:2] / source.sha256 / source.filename

    def _verify(self, path: Path, source: SourceSpec) -> None:
        digest, size = sha256_file(path)
        if size != source.size_bytes:
            raise _SizeMismatch(f"size mismatch for {source.filename}: expected {source.size_bytes}, got {size}")
        if digest != source.sha256:
            raise _ChecksumMismatch(f"SHA-256 mismatch for {source.filename}: expected {source.sha256}, got {digest}")

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, _SizeMismatch):
            return True
        if isinstance(exc, HTTPError):
            return exc.code in {408, 429} or 500 <= exc.code < 600
        return isinstance(exc, (URLError, _TransferError))

    @staticmethod
    def _request(url: str, hostname: str, hf_token: str, github_token: str) -> Request:
        request = Request(url, headers={"User-Agent": "sima-neat-model-registry/0.1"})
        if hf_token and hostname == "huggingface.co":
            # urllib does not copy unredirected headers to Hugging Face's signed
            # CDN redirect, so the bearer token remains scoped to huggingface.co.
            request.add_unredirected_header("Authorization", f"Bearer {hf_token}")
        if github_token and hostname in {"api.github.com", "github.com"}:
            # Private release downloads redirect to a signed object URL. Keep
            # repository credentials scoped to GitHub; the redirect itself is
            # authorized by its short-lived signature.
            request.add_unredirected_header("Authorization", f"Bearer {github_token}")
            request.add_header("Accept", "application/octet-stream")
        return request

    def _download_once(self, request: Request, source: SourceSpec, destination: Path) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{source.filename}.", suffix=".partial", dir=destination.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                try:
                    response = self.opener(request, timeout=60)
                except HTTPError:
                    raise
                except (OSError, TimeoutError, HTTPException) as exc:
                    raise _TransferError(str(exc)) from exc
                with response:
                    while True:
                        try:
                            chunk = response.read(1024 * 1024)
                        except (OSError, TimeoutError, HTTPException) as exc:
                            raise _TransferError(str(exc)) from exc
                        if not chunk:
                            break
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            self._verify(temporary, source)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

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
        hostname = urlsplit_hostname(url)
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        github_token = os.environ.get("GITHUB_TOKEN", "").strip()

        failures: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                request = self._request(url, hostname, hf_token, github_token)
                self._download_once(request, source, destination)
                return destination, False
            except (HTTPError, URLError, OSError, TimeoutError, HTTPException, FetchError) as exc:
                failures.append(f"attempt {attempt}: {exc}")
                if attempt == self.max_attempts or not self._retryable(exc):
                    detail = "; ".join(failures)
                    raise FetchError(f"failed to fetch {redact_url(url)} after {attempt} attempt(s): {detail}") from exc
                self.sleeper(2 ** (attempt - 1))

        raise AssertionError("unreachable")

    def fetch_source(self, source: dict[str, object]) -> tuple[Path, bool, str, int]:
        """Fetch one source contract and return its compiler entrypoint."""
        identity = source_identity(source)
        if source["type"] == "file":
            spec = SourceSpec.from_mapping(source)
            path, cache_hit = self.fetch(spec)
            return path, cache_hit, identity, spec.size_bytes

        members = list(source["files"])
        resolved: list[tuple[dict[str, object], Path]] = []
        cache_hit = True
        total_size = 0
        for member in members:
            spec = SourceSpec.from_mapping(member)
            path, member_hit = self.fetch(spec)
            resolved.append((member, path))
            cache_hit = cache_hit and member_hit
            total_size += spec.size_bytes

        source_root = self.cache_dir / "source-sets" / identity[:2] / identity
        manifest_path = source_root / "source-set.json"
        if source_root.is_dir():
            for member, _ in resolved:
                self._verify(source_root / str(member["path"]), SourceSpec.from_mapping(member))
            if not manifest_path.is_file():
                raise FetchError(f"cached source set {identity} is missing source-set.json")
            return source_root / str(source["entrypoint"]), True, identity, total_size

        source_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=source_root.parent))
        try:
            for member, cached in resolved:
                destination = temporary / str(member["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(cached, destination)
                except OSError:
                    shutil.copyfile(cached, destination)
            manifest = {
                "schema_version": 1,
                "identity": identity,
                "entrypoint": source["entrypoint"],
                "files": sorted(members, key=lambda item: str(item["path"])),
            }
            (temporary / "source-set.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            try:
                os.replace(temporary, source_root)
                temporary = None
            except OSError:
                if not source_root.is_dir():
                    raise
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        return source_root / str(source["entrypoint"]), cache_hit, identity, total_size


def urlsplit_hostname(url: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower()


def _verify_dataset(path: Path, dataset: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FetchError(f"dataset does not exist: {resolved}")
    digest, size = sha256_file(resolved)
    if size != int(dataset["size_bytes"]):
        raise FetchError(f"dataset size mismatch: expected {dataset['size_bytes']}, found {size}")
    if digest != dataset["sha256"]:
        raise FetchError(f"dataset SHA-256 mismatch: expected {dataset['sha256']}, found {digest}")
    return resolved


def resolve_dataset(dataset: dict[str, Any], cache_dir: Path | None = None) -> Path:
    """Resolve a declared dataset archive through dataset-registry and verify its size and digest."""
    executable = shutil.which("dataset-registry")
    installed_script = Path(sys.executable).resolve().with_name("dataset-registry")
    if executable is not None:
        command = [executable]
    elif installed_script.is_file():
        command = [str(installed_script)]
    else:
        command = [sys.executable, "-m", "dataset_registry"]
    command += ["resolve", str(dataset["id"]), "--variant", str(dataset["variant"])]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir.expanduser().resolve())]
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        path = Path(json.loads(completed.stdout)["path"])
    except FileNotFoundError as exc:
        raise FetchError("dataset-registry is required to resolve the configured dataset") from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, TypeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise FetchError(f"dataset-registry resolution failed: {detail.strip()}") from exc
    return _verify_dataset(path, dataset)
