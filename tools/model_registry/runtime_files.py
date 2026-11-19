"""Validate, read and restore supporting files shipped beside compiled model components."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ValidationError
from .fetch import sha256_file


def runtime_file_path(root: Path, name: str) -> Path:
    """Resolve a package supporting file without allowing it to replace build metadata."""
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != name
        or "\\" in name
        or relative.name.endswith("_mpk.tar.gz")
        or relative.name
        in {
            "compile-outputs.json",
            "compile-result.json",
            "registry-publication.json",
            "package-result.json",
            "deployment-manifest.json",
        }
    ):
        raise ValidationError(f"unsafe runtime file path: {name!r}")
    path = root.resolve()
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValidationError(f"unsafe runtime file path: {name!r}")
    return path


def _runtime_file_entries(files: Any) -> list[dict[str, str]]:
    if not isinstance(files, list) or not files:
        raise ValidationError("runtime_files must be a non-empty list")
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item["path"], str):
            raise ValidationError("runtime_files entries must contain path and sha256")
    return files


def validate_runtime_files(files: Any, root: Path) -> list[dict[str, str]]:
    """Verify the compile's supporting files before packaging or using them on a board."""
    files = _runtime_file_entries(files)
    names: set[str] = set()
    for item in files:
        path = runtime_file_path(root, item["path"])
        if item["path"] in names:
            raise ValidationError(f"duplicate runtime file: {item['path']}")
        names.add(item["path"])
        if not path.is_file() or sha256_file(path)[0] != item["sha256"]:
            raise ValidationError(f"runtime file is missing or its SHA-256 does not match: {item['path']}")
    return files


def read_runtime_files(profile_dir: Path) -> list[dict[str, str]]:
    manifest = profile_dir / "compile-outputs.json"
    if not manifest.is_file():
        return []
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"could not read compile outputs {manifest}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValidationError(f"compile outputs must be an object: {manifest}")
    return validate_runtime_files(document["runtime_files"], profile_dir) if "runtime_files" in document else []


def copy_runtime_files(
    profile_dir: Path, output_dir: Path, *, copy_function: Callable = shutil.copy2
) -> Path | None:
    """Copy verified supporting files into a shared resource root without replacing other resources.

    An identical existing file is reusable. Check every destination before copying, then verify the result;
    benchmark preparation and retained-component staging therefore use the same checksum and path rules.
    """
    files = read_runtime_files(profile_dir)
    if not files:
        return None
    destinations = [(item, runtime_file_path(output_dir, item["path"])) for item in files]
    for item, target in destinations:
        if target.exists() and (not target.is_file() or sha256_file(target)[0] != item["sha256"]):
            raise ValidationError(f"runtime file destination already differs: {target}")
    for item, target in destinations:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_function(str(runtime_file_path(profile_dir, item["path"])), str(target))
    validate_runtime_files(files, output_dir)
    return output_dir.resolve()


def restore_runtime_files(archive: Path, files: Any, profile_dir: Path) -> None:
    """Restore only declared supporting files from an already checksum-verified package ZIP."""
    files = _runtime_file_entries(files)
    with tempfile.TemporaryDirectory(prefix="runtime-files-", dir=profile_dir) as temporary:
        staged = Path(temporary)
        try:
            with zipfile.ZipFile(archive) as package:
                names = package.namelist()
                for item in files:
                    destination = runtime_file_path(staged, item["path"])
                    if names.count(item["path"]) != 1:
                        raise ValidationError(f"package must contain exactly one runtime file: {item['path']}")
                    member = package.getinfo(item["path"])
                    if member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValidationError(f"runtime file is not a regular file: {item['path']}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
            validate_runtime_files(files, staged)
            for item in files:
                destination = runtime_file_path(profile_dir, item["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(staged / item["path"], destination)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ValidationError(f"could not restore package runtime files: {exc}") from exc
