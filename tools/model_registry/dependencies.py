from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import ValidationError

SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
MODEL_COMPILER_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
MODEL_COMPILER_CHANNEL = re.compile(r"^[a-z][a-z0-9-]*$")
MODEL_COMPILER_REVISION = re.compile(r"^[0-9a-f]{12}$")


def _load_manifest(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    manifest_path = root / "deps" / "manifest.json"
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not read dependency manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"dependency manifest is not valid JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValidationError("dependency manifest must be a JSON object")
    return manifest


def resolve_model_compiler(repository: str | Path) -> dict[str, str]:
    manifest = _load_manifest(repository)
    dependency = manifest.get("model-compiler")
    if not isinstance(dependency, dict):
        raise ValidationError("dependency manifest must define model-compiler as an object")
    version = dependency.get("version")
    if not isinstance(version, str) or MODEL_COMPILER_VERSION.fullmatch(version) is None:
        raise ValidationError(
            "model-compiler.version must use MAJOR.MINOR.PATCH or "
            "MAJOR.MINOR.PATCH.SERVICE format"
        )
    channel = dependency.get("channel")
    revision = dependency.get("revision")
    if channel is None and revision is None:
        return {"identity": version, "label": f"model-compiler-{version}", "version": version}
    if not isinstance(channel, str) or MODEL_COMPILER_CHANNEL.fullmatch(channel) is None:
        raise ValidationError("model-compiler.channel must use lowercase letters, digits, and hyphens")
    if not isinstance(revision, str) or MODEL_COMPILER_REVISION.fullmatch(revision) is None:
        raise ValidationError("model-compiler.revision must be a 12-character lowercase Git hash")
    return {
        "identity": f"{version}+{channel}.{revision}",
        "label": f"model-compiler-{channel}-{revision}",
        "version": version,
    }


def resolve_benchmark_runtime(repository: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(repository)
    devkit = manifest.get("devkit")
    core = manifest.get("neat-core")
    if not isinstance(devkit, dict):
        raise ValidationError("dependency manifest must define devkit as an object")
    if not isinstance(core, dict):
        raise ValidationError("dependency manifest must define neat-core as an object")

    platform_version = manifest.get("platform-version")
    core_version = core.get("version")
    for field, value in (
        ("platform-version", platform_version),
        ("neat-core.version", core_version),
    ):
        if not isinstance(value, str) or SEMANTIC_VERSION.fullmatch(value) is None:
            raise ValidationError(f"{field} must use MAJOR.MINOR.PATCH format")

    board = devkit.get("board")
    os_type = devkit.get("os_type")
    build = devkit.get("build")
    install_target = core.get("install_target")
    if not isinstance(build, str) or re.fullmatch(r"B[0-9]+", build) is None:
        raise ValidationError("devkit.build must use B followed by the build number")
    if not isinstance(board, str) or not board:
        raise ValidationError("devkit.board must be a non-empty string")
    if not isinstance(os_type, str) or not os_type:
        raise ValidationError("devkit.os_type must be a non-empty string")
    if not isinstance(install_target, str) or re.fullmatch(
        r"core@[A-Za-z0-9][A-Za-z0-9._/-]*(?::[a-f0-9]{12,40})?", install_target
    ) is None:
        raise ValidationError("neat-core.install_target must name a Core release or branch package")

    return {
        "devkit": {
            "board": board,
            "os_type": os_type,
            "build": build,
            "platform_version": platform_version,
        },
        "neat_core": {"version": core_version, "install_target": install_target},
    }
