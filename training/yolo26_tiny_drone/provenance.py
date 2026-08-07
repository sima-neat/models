"""Immutable upstream pins and environment provenance checks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ULTRALYTICS_REVISION = "1367566337fb8056223a1aeb469360747f1b1bcd"
ULTRALYTICS_PACKAGE_SHA256 = "d3237e8acc6d57a64c709040a58233c53eae691a1ff6efebf1df56deccc759e5"
SWML_QAT_REVISION = "dee091582acc7d776a4d4a582f8e25655f2ffd90"
SWML_QAT_PACKAGE_SHA256 = "759d9507f7c42a3372d651b8ec4764e6fc18020bd64617fcfa203e95283e0c37"
YOLO26N_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def snapshot_file(source: Path, destination: Path, expected_sha256: str) -> Path:
    """Copy verified bytes to a private read-only path for deterministic consumption."""
    shutil.copyfile(source, destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"source changed while creating its private snapshot: expected {expected_sha256}, got {actual}"
        )
    destination.chmod(0o400)
    return destination


def _git_revision(package_root: Path) -> str | None:
    for candidate in (package_root, *package_root.parents):
        if not (candidate / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def _distribution_revision() -> str | None:
    try:
        direct_url = importlib.metadata.distribution("ultralytics").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    document = json.loads(direct_url)
    vcs = document.get("vcs_info") or {}
    revision = vcs.get("commit_id")
    return revision if isinstance(revision, str) else None


def runtime_package_sha256(package_root: Path) -> str:
    """Hash an installed runtime package independently of VCS metadata."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".md"
    )
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def ultralytics_package_sha256(package_root: Path) -> str:
    return runtime_package_sha256(package_root)


def require_swml_qat_runtime() -> str:
    """Require the exact internal QAT implementation pinned by this recipe."""
    specification = importlib.util.find_spec("sima_qat")
    locations = None if specification is None else specification.submodule_search_locations
    if not locations:
        raise RuntimeError("cannot locate the installed sima_qat runtime package")
    package_root = Path(next(iter(locations))).resolve()
    package_digest = runtime_package_sha256(package_root)
    if package_digest != SWML_QAT_PACKAGE_SHA256:
        raise RuntimeError(
            f"swml-qat package content mismatch: expected {SWML_QAT_PACKAGE_SHA256}, got {package_digest}"
        )
    return package_digest


@dataclass(frozen=True)
class VerifiedRuntimePackage:
    package_root: Path
    package_sha256: str
    temporary: tempfile.TemporaryDirectory


def activate_verified_swml_qat_runtime() -> VerifiedRuntimePackage:
    """Import sima_qat from a private snapshot of the bytes that passed verification."""
    if "sima_qat" in sys.modules:
        raise RuntimeError("sima_qat was imported before its runtime snapshot was activated")
    specification = importlib.util.find_spec("sima_qat")
    locations = None if specification is None else specification.submodule_search_locations
    if not locations:
        raise RuntimeError("cannot locate the installed sima_qat runtime package")
    source_root = Path(next(iter(locations))).resolve()
    source_digest = require_swml_qat_runtime()
    temporary = tempfile.TemporaryDirectory(prefix="yolo26-swml-qat-")
    snapshot_root = Path(temporary.name) / "sima_qat"
    try:
        shutil.copytree(
            source_root,
            snapshot_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.md"),
        )
        snapshot_digest = runtime_package_sha256(snapshot_root)
        if snapshot_digest != source_digest:
            raise RuntimeError(
                f"swml-qat changed while creating its private snapshot: expected {source_digest}, got {snapshot_digest}"
            )
        for path in snapshot_root.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
        sys.path.insert(0, temporary.name)
        importlib.invalidate_caches()
        module = importlib.import_module("sima_qat")
        loaded_root = Path(module.__file__).resolve().parent
        if loaded_root != snapshot_root:
            raise RuntimeError(f"sima_qat loaded outside its verified snapshot: {loaded_root}")
    except BaseException:
        sys.path = [entry for entry in sys.path if entry != temporary.name]
        sys.modules.pop("sima_qat", None)
        temporary.cleanup()
        raise
    return VerifiedRuntimePackage(snapshot_root, snapshot_digest, temporary)


def require_ultralytics_revision(package_root: Path) -> str:
    # Prefer the installed distribution's PEP 610 provenance. A project-local
    # virtual environment can live below an unrelated Git worktree, and using
    # that enclosing repository first would incorrectly report its revision as
    # the Ultralytics revision.
    actual = _distribution_revision() or _git_revision(package_root)
    if actual is None:
        raise RuntimeError("cannot prove the installed Ultralytics source revision")
    if actual != ULTRALYTICS_REVISION:
        raise RuntimeError(f"Ultralytics revision mismatch: expected {ULTRALYTICS_REVISION}, got {actual}")
    package_digest = ultralytics_package_sha256(package_root)
    if package_digest != ULTRALYTICS_PACKAGE_SHA256:
        raise RuntimeError(
            f"Ultralytics package content mismatch: expected {ULTRALYTICS_PACKAGE_SHA256}, got {package_digest}"
        )
    return actual
