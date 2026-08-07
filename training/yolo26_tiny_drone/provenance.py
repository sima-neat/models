"""Immutable upstream pins and environment provenance checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

ULTRALYTICS_REVISION = "1367566337fb8056223a1aeb469360747f1b1bcd"
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
    return actual
