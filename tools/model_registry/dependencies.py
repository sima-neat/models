from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import ValidationError

MODEL_COMPILER_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def resolve_model_compiler(repository: str | Path) -> dict[str, str]:
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
    dependency = manifest.get("model-compiler")
    if not isinstance(dependency, dict):
        raise ValidationError("dependency manifest must define model-compiler as an object")
    version = dependency.get("version")
    if not isinstance(version, str) or MODEL_COMPILER_VERSION.fullmatch(version) is None:
        raise ValidationError("model-compiler.version must use MAJOR.MINOR.PATCH format")
    return {"label": f"model-compiler-{version}", "version": version}
