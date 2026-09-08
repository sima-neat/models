from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import tarfile
import tempfile
import time
from collections.abc import Callable
from http.client import HTTPException
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from ._vulcan_onnx_artifacts import assemble_download, verify_extract
from .errors import FetchError
from .source import SHA256_RE, SourceSpec, redact_url, source_identity

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

    def _fetch_onnx_bundle(self, source: dict[str, object]) -> tuple[Path, bool, str, int]:
        """Materialize a pinned Vulcan completion manifest using the ordinary file cache."""
        from urllib.parse import urlsplit

        spec = SourceSpec.from_mapping(source)
        location = urlsplit(spec.uri)
        if location.hostname != "artifacts.stg.neat.sima.ai" or not location.path.endswith("/manifest.json"):
            raise FetchError("ONNX bundles require a versioned Vulcan completion manifest")
        manifest_path, manifest_hit = self.fetch(spec)
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest["schema_version"] != 1 or manifest["status"] != "completed":
                raise ValueError("ONNX bundle manifest must be completed schema version 1")
            files = manifest["files"]
            objects = manifest["objects"]
            bundle = manifest["bundle"]
            if not isinstance(files, dict) or not {"model.onnx", "validation.json"} <= files.keys():
                raise ValueError("ONNX bundle must contain model.onnx and validation.json")
            if not isinstance(objects, list) or not objects:
                raise ValueError("ONNX bundle has no payload objects")
            names: set[str] = set()
            for name, metadata in files.items():
                _bundle_path(name)
                if name.casefold() in names:
                    raise ValueError("Case-colliding ONNX bundle members")
                names.add(name.casefold())
                _bundle_metadata(metadata)
            _bundle_path(bundle["name"])
            if "/" in bundle["name"]:
                raise ValueError("Bundle archive name must be a filename")
            _bundle_metadata(bundle)
            prefix = location.path.removeprefix("/").rsplit("/", 1)[0] + "/"
            object_names = set()
            for obj in objects:
                _bundle_path(obj["key"])
                if not obj["key"].startswith(prefix) or "/" in obj["key"][len(prefix):]:
                    raise ValueError("Payload objects must be adjacent to the completion manifest")
                if obj["key"] in object_names:
                    raise ValueError("Duplicate payload object")
                object_names.add(obj["key"])
                if not isinstance(obj["version_id"], str) or obj["version_id"] in {"", "null"}:
                    raise ValueError("Payload objects must pin S3 versions")
                _bundle_metadata(obj)
            if sum(obj["size_bytes"] for obj in objects) != bundle["size_bytes"]:
                raise ValueError("Payload sizes do not match bundle size")
            destination = self.cache_dir / "onnx-bundles" / spec.sha256
            if destination.is_symlink():
                raise ValueError("ONNX bundle cache must not be a symlink")
            if destination.is_dir():
                _verify_bundle_directory(destination, files)
                return destination / "model.onnx", manifest_hit, spec.sha256, sum(
                    item["size_bytes"] for item in files.values()
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{spec.sha256}.", dir=destination.parent) as temporary:
                temporary_root = Path(temporary)
                for obj in objects:
                    uri = f"https://{location.hostname}/{obj['key']}"
                    part, _ = self.fetch(SourceSpec.from_mapping({**obj, "uri": uri}))
                    local = temporary_root / obj["key"].rsplit("/", 1)[-1]
                    try:
                        os.link(part, local)
                    except OSError:
                        shutil.copyfile(part, local)
                archive_path = assemble_download(temporary_root, manifest)
                extracted = verify_extract(archive_path, manifest, temporary_root / "extracted")
                _verify_bundle_directory(extracted, files)
                os.replace(extracted, destination)
            return destination / "model.onnx", False, spec.sha256, sum(
                item["size_bytes"] for item in files.values()
            )
        except (KeyError, TypeError, ValueError, OSError, tarfile.TarError) as exc:
            raise FetchError(f"Invalid ONNX bundle: {exc}") from exc

    def fetch_source(self, source: dict[str, object]) -> tuple[Path, bool, str, int]:
        """Fetch one source contract and return its compiler entrypoint."""
        if source["type"] == "onnx_bundle":
            return self._fetch_onnx_bundle(source)
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

def _bundle_path(name: str) -> None:
    if not isinstance(name, str) or not name or "\\" in name or any(
        part in {"", ".", ".."} for part in name.split("/")
    ) or any(ord(char) < 32 for char in name):
        raise ValueError(f"Unsafe ONNX bundle path: {name!r}")


def _bundle_metadata(metadata: dict[str, object]) -> None:
    if not isinstance(metadata, dict) or not SHA256_RE.fullmatch(str(metadata["sha256"])):
        raise ValueError("Invalid ONNX bundle checksum")
    if type(metadata["size_bytes"]) is not int or metadata["size_bytes"] < 0:
        raise ValueError("Invalid ONNX bundle size")


def _verify_bundle_directory(directory: Path, files: dict[str, dict[str, object]]) -> None:
    present = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("ONNX bundle cache contains a non-regular file")
        if path.is_file():
            present.add(path.relative_to(directory).as_posix())
    if present != set(files):
        raise ValueError("ONNX bundle cache has missing or unexpected files")
    for name, metadata in files.items():
        digest, size = sha256_file(directory / name)
        if digest != metadata["sha256"] or size != metadata["size_bytes"]:
            raise ValueError(f"ONNX bundle member checksum/size mismatch: {name}")
    if not (directory / "model.onnx").stat().st_size:
        raise ValueError("Empty ONNX model")
    validation = json.loads((directory / "validation.json").read_text())
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("ONNX bundle validation did not pass")
