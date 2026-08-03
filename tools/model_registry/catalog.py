from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ValidationError
from .model import load_and_validate_model

SHARED_BUILD_PATHS = (
    ".github/workflows/Vulcan-Model-Compile.yml",
    "deps/",
    "pyproject.toml",
    "schemas/",
    "tools/model_registry/",
    "tools/runtime/",
)


def discover_catalog(repository: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(repository).expanduser().resolve()
    model_root = root / "models"
    discovered: dict[str, dict[str, Any]] = {}
    if not model_root.is_dir():
        raise ValidationError(f"model catalog not found: {model_root}")

    for definition in sorted(model_root.glob("*/model.yaml")):
        model = load_and_validate_model(definition.parent, root / "schemas" / "model.schema.json")
        model_id = str(model["id"])
        if model_id in discovered:
            raise ValidationError(f"duplicate model id {model_id!r}: {definition}")
        discovered[model_id] = model
    return discovered


def git_changed_paths(repository: str | Path, base: str, head: str) -> list[str]:
    root = Path(repository).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRD", base, head, "--"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise ValidationError(f"could not determine changed paths for {base}..{head}: {detail}") from exc
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def _is_shared_build_path(path: str) -> bool:
    return any(path == prefix or (prefix.endswith("/") and path.startswith(prefix)) for prefix in SHARED_BUILD_PATHS)


def select_model_ids(catalog: dict[str, dict[str, Any]], changed_paths: list[str]) -> list[str]:
    normalized_paths: set[str] = set()
    for raw_path in changed_paths:
        if not raw_path.strip():
            continue
        path = PurePosixPath(raw_path).as_posix()
        normalized_paths.add(path[2:] if path.startswith("./") else path)
    normalized = sorted(normalized_paths)
    if any(_is_shared_build_path(path) for path in normalized):
        return sorted(catalog)

    selected: set[str] = set()
    for path in normalized:
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[0] == "models" and parts[1] in catalog:
            selected.add(parts[1])
    return sorted(selected)


def build_matrix(catalog: dict[str, dict[str, Any]], model_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    unknown = sorted(set(model_ids) - set(catalog))
    if unknown:
        raise ValidationError(f"unknown model id(s): {', '.join(unknown)}")

    include: list[dict[str, str]] = []
    for model_id in sorted(set(model_ids)):
        model = catalog[model_id]
        for variant in sorted(model["compile_variants"], key=lambda item: item["id"]):
            include.append(
                {
                    "model_dir": f"models/{model_id}",
                    "model_id": model_id,
                    "recipe": variant["recipe"],
                    "target": variant["target"],
                    "variant_id": variant["id"],
                }
            )
    return {"include": include}
