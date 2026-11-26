from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from .errors import FetchError, ValidationError
from .registry_api.transport import ReadRequest, read_registry_json
from .source import SourceSpec, redact_url, source_identity

OpenUrl = Callable[..., BinaryIO]
Sleeper = Callable[[float], None]
DownloadRequest = Callable[[str], bytes]

# The stage job's list of a staged build profile's files, which the DevKit downloads by compile run.
BUILD_FILES = "build-files.json"
# The receipt naming the ordinary deployment a speculative profile is benchmarked against, at the staged root.
BENCHMARK_BASELINE = "benchmark-baseline.json"
# Attempts per registry download, each with a link the registry signs for it.
REGISTRY_DOWNLOAD_ATTEMPTS = 5
# HTTP statuses a registry download retries: a signed link that expired during an earlier attempt (403), throttling
# (429), and a gateway that dropped the transfer (502 to 504).
RETRIED_DOWNLOAD_STATUSES = frozenset({403, 429, 502, 503, 504})


class _SizeMismatch(FetchError):
    """A transfer ended before the declared payload size was received."""


class _OversizedPayload(FetchError):
    """A response carried more bytes than the declared payload size."""


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
    def _request(url: str, hostname: str, hf_token: str) -> Request:
        request = Request(url, headers={"User-Agent": "sima-neat-model-registry/0.1"})
        if hf_token and hostname == "huggingface.co":
            # urllib does not copy unredirected headers to Hugging Face's signed
            # CDN redirect, so the bearer token remains scoped to huggingface.co.
            request.add_unredirected_header("Authorization", f"Bearer {hf_token}")
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
                    received = 0
                    while True:
                        try:
                            chunk = response.read(1024 * 1024)
                        except (OSError, TimeoutError, HTTPException) as exc:
                            raise _TransferError(str(exc)) from exc
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > source.size_bytes:
                            raise _OversizedPayload(
                                f"size mismatch for {source.filename}: expected {source.size_bytes}, got more"
                            )
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

        failures: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                request = self._request(url, hostname, hf_token)
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
    """Resolve a declared dataset archive through dataset-registry and verify its size and digest.

    dataset-registry trusts its cache by registry path, so an archive republished at the same path
    leaves a stale cached copy. A cached archive that differs from the declaration is removed and
    resolved once more, which downloads the current archive.
    """
    path = _resolve_with_dataset_registry(dataset, cache_dir)
    try:
        return _verify_dataset(path, dataset)
    except FetchError as exc:
        print(f"{exc}; removing {path} and resolving again", file=sys.stderr)
        path.unlink(missing_ok=True)
    return _verify_dataset(_resolve_with_dataset_registry(dataset, cache_dir), dataset)


def _resolve_with_dataset_registry(dataset: dict[str, Any], cache_dir: Path | None) -> Path:
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
    return path


# The index at the root of the DevKit pool's shared dataset archive share, and the prefix of each archive key it
# lists. A key without the prefix is the archive's path below the index.
SHARED_ARCHIVE_INDEX = "_release.json"
SHARED_ARCHIVE_KEY_PREFIX = "datasets/artifactory/"


def unpack_shared_dataset(dataset: dict[str, Any], shared_archives: Path, destination: Path) -> bool:
    """Unpack a pinned dataset archive from the DevKit pool's shared dataset archive share into ``destination``.

    The share's index lists each archive with its SHA-256 and size, so the pin, never the dataset's name, finds
    the archive, which is read in place and never copied. Returns True once the unpack is complete and the archive
    matches the pin. Returns False when the share cannot give the archive, and a failed unpack removes
    ``destination`` with everything in it. A machine without the share returns False quietly. Any other miss says
    why in one line on stderr.
    """
    # Follow a link at the share's path once, so the index and the archive come from one folder even when the link
    # moves meanwhile.
    root = shared_archives.resolve()
    index = root / SHARED_ARCHIVE_INDEX
    if not index.is_file():
        return False
    label = f"{dataset['id']} {dataset['variant']}"
    try:
        archives = json.loads(index.read_text(encoding="utf-8"))["archives"]
        key = next((item["key"] for item in archives
                    if item["sha256"] == dataset["sha256"] and item["size"] == dataset["size_bytes"]), None)
        if key is None:
            print(f"The shared dataset archive share does not hold {label}. Resolving it through the Dataset "
                  "Registry.", file=sys.stderr)
            return False
        if not key.startswith(SHARED_ARCHIVE_KEY_PREFIX):
            raise ValueError(f"archive key {key!r} does not start with {SHARED_ARCHIVE_KEY_PREFIX}")
        destination.mkdir(parents=True, exist_ok=True)
        _unpack_verified(root / key.removeprefix(SHARED_ARCHIVE_KEY_PREFIX), destination, dataset)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, FetchError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        print(f"The shared dataset archive share cannot give {label} ({exc}). Resolving it through the Dataset "
              "Registry.", file=sys.stderr)
        return False


def _unpack_verified(archive: Path, destination: Path, dataset: dict[str, Any]) -> None:
    """Unpack ``archive`` into ``destination`` with tar, reading it once, and check it against the dataset's pin.

    Each chunk passes through SHA-256 on its way into tar, so the unpacked files count only once this returns.
    Raises FetchError when the size, the digest, or the unpack is wrong, in that order.
    """
    digest, size, writing = hashlib.sha256(), 0, True
    with archive.open("rb") as stream, tempfile.TemporaryFile() as tar_errors:
        tar = subprocess.Popen(["tar", "-xzf", "-", "-C", str(destination)], stdin=subprocess.PIPE, stderr=tar_errors)
        try:
            # Reading goes on after tar stops, so a wrong archive is reported by its digest, not by tar.
            while size <= dataset["size_bytes"] and (chunk := stream.read(1024 * 1024)):
                digest.update(chunk)
                size += len(chunk)
                if writing:
                    try:
                        tar.stdin.write(chunk)
                    except BrokenPipeError:
                        writing = False
        finally:
            with contextlib.suppress(BrokenPipeError):
                tar.stdin.close()
            tar.wait()
        tar_errors.seek(0)
        complaint = tar_errors.read().decode(errors="replace").strip().splitlines()
    if size != dataset["size_bytes"]:
        found = "more" if size > dataset["size_bytes"] else size
        raise FetchError(f"dataset size mismatch: expected {dataset['size_bytes']}, found {found}")
    if digest.hexdigest() != dataset["sha256"]:
        raise FetchError(f"dataset SHA-256 mismatch: expected {dataset['sha256']}, found {digest.hexdigest()}")
    if tar.returncode != 0:
        raise FetchError(f"tar exited with {tar.returncode}: {complaint[-1] if complaint else 'no message'}")


@dataclass(frozen=True)
class BuildFile:
    """One registry file of a staged build: its path under the profile folder and the compile or package run
    artifact that holds it."""

    path: str
    run_id: str
    artifact: str
    sha256: str
    size_bytes: int


def download_build_file(read: ReadRequest, api_url: str, profile_dir: Path, item: BuildFile,
                        download: DownloadRequest | None = None) -> Path:
    """Download one build file to its path under ``profile_dir`` and return that path. The file is kept only when
    its bytes have the item's SHA-256 and size.

    Each attempt reads the artifact from the registry again, so it downloads through a freshly signed link and only
    while the registry still records the item's SHA-256 and size. A failed or cut-off transfer is retried. A
    registry record that differs, a longer download, and a complete download with another SHA-256 are not.
    """
    destination = _build_file_path(profile_dir, item.path)
    artifact_url = f"{api_url}/v1/runs/{item.run_id}/artifacts/{item.artifact}"
    # Every build of an orchestrator run downloads at once, so a throttled or dropped transfer is expected.
    for attempt in range(1, REGISTRY_DOWNLOAD_ATTEMPTS + 1):
        url = _registered_download_url(read, artifact_url, item)
        try:
            _download_registry_file_once(url, destination, item, download)
            return destination
        except (_TransferError, _SizeMismatch) as exc:
            refused = isinstance(exc.__cause__, HTTPError) and exc.__cause__.code not in RETRIED_DOWNLOAD_STATUSES
            if refused or attempt == REGISTRY_DOWNLOAD_ATTEMPTS:
                raise FetchError(f"could not download {item.path} after {attempt} attempt(s): {exc}") from exc
        time.sleep(2 ** (attempt - 1) + random.uniform(0, 1))
    raise AssertionError("unreachable")


def _registered_download_url(read: ReadRequest, artifact_url: str, item: BuildFile) -> str:
    """The download link the registry signs now for the artifact, which must still have the item's SHA-256 and
    size."""
    response = read(artifact_url)
    registered = response.get("artifact") if isinstance(response, dict) else None
    if (not isinstance(registered, dict) or registered.get("sha256") != item.sha256
            or registered.get("size_bytes") != item.size_bytes
            or not isinstance(registered.get("download_url"), str)):
        raise ValidationError(f"{artifact_url} does not record the SHA-256 and size staged for {item.path}")
    return registered["download_url"]


def _download_registry_file_once(url: str, destination: Path, item: BuildFile,
                                 download: DownloadRequest | None) -> None:
    """Stream one download into a temporary file beside ``destination``, and replace ``destination`` with it only
    when it is complete and has the item's SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            for chunk in _stream_download(url, download):
                size += len(chunk)
                if size > item.size_bytes:
                    raise ValidationError(f"{destination.name}: download exceeds its declared size")
                output.write(chunk)
                digest.update(chunk)
        if size < item.size_bytes:
            raise _SizeMismatch(f"the download ended after {size} of {item.size_bytes} bytes")
        if digest.hexdigest() != item.sha256:
            raise ValidationError(f"{destination.name}: download failed SHA-256 verification")
        temporary.replace(destination)
    except (OSError, HTTPException) as exc:
        raise _TransferError(str(exc)) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stream_download(url: str, download: DownloadRequest | None) -> Iterator[bytes]:
    """Yield the body served at ``url`` in chunks, or the bytes an injected ``download`` returns for it."""
    if download is not None:
        yield download(url)
        return
    request = Request(url, headers={"User-Agent": "sima-neat-model-registry/0.1"})
    with urlopen(request, timeout=120) as response:
        yield from iter(lambda: response.read(1024 * 1024), b"")


@dataclass(frozen=True)
class StagedBuildFiles:
    """What the DevKit downloads for a staged build profile: the registry the stage job read, every file the
    benchmark reads, and the package whose runtime files the profile restores when the compile declared any."""

    api_url: str
    files: tuple[BuildFile, ...]
    runtime_package: BuildFile | None = None

    def write(self, profile_dir: Path) -> None:
        (profile_dir / BUILD_FILES).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")

    @classmethod
    def read(cls, profile_dir: Path) -> StagedBuildFiles:
        path = profile_dir / BUILD_FILES
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            package = document.get("runtime_package")
            return cls(
                api_url=document["api_url"].rstrip("/"),
                files=tuple(BuildFile(**item) for item in document["files"]),
                runtime_package=BuildFile(**package) if package else None,
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValidationError(f"could not read the staged build files {path}: {exc}") from exc


def download_build(compiled_dir: str | Path, model_id: str, build_profile_id: str, *,
                   read: ReadRequest | None = None, download: DownloadRequest | None = None) -> dict[str, int]:
    """Download every file of a staged build profile, and of its staged benchmark baseline, by compile run.

    The registry must still record each file with the SHA-256 and size the stage job staged, and the downloaded
    bytes must match them. Runtime files are restored from their package archive, which is then removed.
    Returns the number of files downloaded per ``<model>/<profile>``.
    """
    root = Path(compiled_dir).expanduser().resolve()
    profiles = [(model_id, build_profile_id)]
    baseline = root / BENCHMARK_BASELINE
    if baseline.is_file():
        identity = json.loads(baseline.read_text(encoding="utf-8"))
        profiles.append((identity["model_id"], identity["build_profile_id"]))
    return {
        f"{model}/{profile}": _download_staged_profile(root / model / profile, read or read_registry_json,
                                                       download)
        for model, profile in profiles
    }


def _download_staged_profile(profile_dir: Path, read: ReadRequest, download: DownloadRequest | None) -> int:
    staged = StagedBuildFiles.read(profile_dir)

    def fetch(item: BuildFile) -> Path:
        return download_build_file(read, staged.api_url, profile_dir, item, download)

    # The largest files start first, so the four transfers end close together.
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, sorted(staged.files, key=lambda item: item.size_bytes, reverse=True)))
    if staged.runtime_package is None:
        return len(staged.files)

    from .runtime_files import restore_runtime_files

    archive = fetch(staged.runtime_package)
    outputs = json.loads((profile_dir / "compile-outputs.json").read_text(encoding="utf-8"))
    restore_runtime_files(archive, outputs["runtime_files"], profile_dir)
    archive.unlink()
    return len(staged.files) + 1


def _build_file_path(profile_dir: Path, relative: str) -> Path:
    """Resolve a staged file path inside its profile folder, through no symbolic link."""
    posix = PurePosixPath(relative)
    if not posix.parts or posix.is_absolute() or ".." in posix.parts or posix.as_posix() != relative:
        raise ValidationError(f"unsafe staged build file path: {relative!r}")
    path = profile_dir
    for part in posix.parts:
        path = path / part
        if path.is_symlink():
            raise ValidationError(f"unsafe staged build file path: {relative!r}")
    return path
