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
    if not isinstance(platform_version, str) or SEMANTIC_VERSION.fullmatch(platform_version) is None:
        raise ValidationError("platform-version must use MAJOR.MINOR.PATCH format")

    board = devkit.get("board")
    os_type = devkit.get("os_type")
    install_target = core.get("install_target")
    if not isinstance(board, str) or not board:
        raise ValidationError("devkit.board must be a non-empty string")
    if not isinstance(os_type, str) or not os_type:
        raise ValidationError("devkit.os_type must be a non-empty string")
    if not isinstance(install_target, str) or re.fullmatch(
        r"core@[A-Za-z0-9][A-Za-z0-9._/-]*(?::[a-f0-9]{12,40})?", install_target
    ) is None:
        raise ValidationError("neat-core.install_target must name a Core release or branch package")

    runtime = {
        "devkit": {
            "board": board,
            "os_type": os_type,
            "platform_version": platform_version,
        },
        "neat_core": {"install_target": install_target},
    }
    if "llima" in manifest:
        llima = manifest["llima"]
        if not isinstance(llima, dict) or not re.fullmatch(
            r"llima@[A-Za-z0-9][A-Za-z0-9._/-]*:[a-f0-9]{12,40}", str(llima.get("install_target", ""))
        ):
            raise ValidationError("llima.install_target must pin a runtime build revision")
        runtime["llima"] = llima
    return runtime


def resolve_catalog_inputs(repository: str | Path) -> dict[str, str] | None:
    """The platform and Neat Core release this checkout's results were measured with, as the registry names
    them. None when the manifest pins a Core branch, whose results belong to no release."""
    runtime = resolve_benchmark_runtime(repository)
    release = re.fullmatch(r"core@v(\d+\.\d+\.\d+)", runtime["neat_core"]["install_target"])
    revision = _load_manifest(repository)["neat-core"].get("revision")
    if release is None:
        if revision is not None:
            raise ValidationError("neat-core.revision applies only to a Core release pin")
        return None
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValidationError("neat-core.revision must be the full commit of the pinned Core release")
    return {
        "platform_version": runtime["devkit"]["platform_version"],
        "neat_core_version": release.group(1),
        "neat_core_revision": revision,
    }
