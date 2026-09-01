from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError

TAXONOMY_RELATIVE_PATH = Path("taxonomies/model-taxonomy-v3.yaml")
TAXONOMY_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def taxonomy_relative_path(version: int) -> Path:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValidationError(f"unsupported taxonomy version {version!r}")
    return Path(f"taxonomies/model-taxonomy-v{version}.yaml")


def find_taxonomy(
    model_dir: str | Path,
    taxonomy_path: str | Path | None = None,
    taxonomy_version: int | None = None,
) -> Path:
    if taxonomy_path is not None:
        candidate = Path(taxonomy_path).expanduser().resolve()
        if not candidate.is_file():
            raise ValidationError(f"taxonomy not found: {candidate}")
        return candidate

    relative_path = TAXONOMY_RELATIVE_PATH if taxonomy_version is None else taxonomy_relative_path(taxonomy_version)
    configured_root = os.environ.get("MODEL_REGISTRY_ROOT", "").strip()
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser().resolve() / relative_path)

    directory = Path(model_dir).expanduser().resolve()
    candidates.extend(parent / relative_path for parent in (directory, *directory.parents))
    candidates.extend(parent / relative_path for parent in (Path.cwd(), *Path.cwd().parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValidationError(
        f"{directory}: could not locate {relative_path}; run inside the repository or set MODEL_REGISTRY_ROOT"
    )


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"taxonomy {field} must be an object")
    return value


def _ids(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"taxonomy {field} must be a non-empty list")
    if any(not isinstance(item, str) or TAXONOMY_ID.fullmatch(item) is None for item in value):
        raise ValidationError(f"taxonomy {field} must contain normalized IDs")
    if len(value) != len(set(value)):
        raise ValidationError(f"taxonomy {field} must not contain duplicate IDs")
    return value


def _validate_entries(entries: dict[str, Any], field: str, allowed_fields: set[str]) -> None:
    if not entries:
        raise ValidationError(f"taxonomy {field} must not be empty")
    for entry_id, raw_entry in entries.items():
        if not isinstance(entry_id, str) or TAXONOMY_ID.fullmatch(entry_id) is None:
            raise ValidationError(f"taxonomy {field} contains invalid ID {entry_id!r}")
        entry = _mapping(raw_entry, f"{field}.{entry_id}")
        unknown = sorted(set(entry) - allowed_fields)
        if unknown:
            raise ValidationError(f"taxonomy {field}.{entry_id} contains unknown fields: {', '.join(unknown)}")
        if not isinstance(entry.get("display_name"), str) or not entry["display_name"].strip():
            raise ValidationError(f"taxonomy {field}.{entry_id}.display_name must be non-empty")


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    taxonomy_path = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{taxonomy_path}: could not load taxonomy: {exc}") from exc
    taxonomy = _mapping(value, "<root>")
    expected = {"version", "model_types", "modalities", "architecture_paradigms", "architectures", "tasks"}
    unknown = sorted(set(taxonomy) - expected)
    missing = sorted(expected - set(taxonomy))
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValidationError(f"{taxonomy_path}: taxonomy {'; '.join(details)}")
    if isinstance(taxonomy["version"], bool) or not isinstance(taxonomy["version"], int) or taxonomy["version"] < 1:
        raise ValidationError(f"{taxonomy_path}: unsupported taxonomy version {taxonomy['version']!r}")

    model_types = _mapping(taxonomy["model_types"], "model_types")
    modalities = _mapping(taxonomy["modalities"], "modalities")
    paradigms = _mapping(taxonomy["architecture_paradigms"], "architecture_paradigms")
    architectures = _mapping(taxonomy["architectures"], "architectures")
    tasks = _mapping(taxonomy["tasks"], "tasks")
    _validate_entries(model_types, "model_types", {"display_name"})
    _validate_entries(modalities, "modalities", {"display_name"})
    _validate_entries(paradigms, "architecture_paradigms", {"display_name"})
    _validate_entries(architectures, "architectures", {"display_name", "paradigm", "allowed_model_types"})
    _validate_entries(tasks, "tasks", {"display_name", "allowed_model_types"})

    for architecture_id, architecture in architectures.items():
        paradigm = architecture.get("paradigm")
        if paradigm not in paradigms:
            raise ValidationError(
                f"taxonomy architectures.{architecture_id}.paradigm references unknown ID {paradigm!r}"
            )
        allowed = _ids(architecture.get("allowed_model_types"), f"architectures.{architecture_id}.allowed_model_types")
        unknown_types = sorted(set(allowed) - set(model_types))
        if unknown_types:
            raise ValidationError(
                f"taxonomy architectures.{architecture_id}.allowed_model_types references unknown IDs: "
                f"{', '.join(unknown_types)}"
            )
    for task_id, task in tasks.items():
        allowed = _ids(task.get("allowed_model_types"), f"tasks.{task_id}.allowed_model_types")
        unknown_types = sorted(set(allowed) - set(model_types))
        if unknown_types:
            raise ValidationError(
                f"taxonomy tasks.{task_id}.allowed_model_types references unknown IDs: {', '.join(unknown_types)}"
            )
    return taxonomy


def load_taxonomy_for_model(repository: str | Path, model: dict[str, Any]) -> dict[str, Any]:
    version = model.get("taxonomy_version")
    path = Path(repository).expanduser().resolve() / taxonomy_relative_path(version)
    taxonomy = load_taxonomy(path)
    if taxonomy["version"] != version:
        raise ValidationError(f"{path}: taxonomy document version does not match {version!r}")
    return taxonomy


def canonical_taxonomy_document(taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Return the public, deterministically ordered taxonomy document."""
    return json.loads(json.dumps(taxonomy, sort_keys=True))


def taxonomy_sha256(taxonomy: dict[str, Any]) -> str:
    payload = json.dumps(taxonomy, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_model_categorization(model: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    schema_version = model.get("schema_version")
    if schema_version != 2:
        raise ValidationError("model schema_version 2 is required; taxonomy categorization is mandatory")

    taxonomy_version = model["taxonomy_version"]
    if taxonomy_version != taxonomy["version"]:
        raise ValidationError(
            f"taxonomy_version {taxonomy_version!r} does not match available taxonomy version {taxonomy['version']}"
        )
    model_type = model["model_type"]
    if model_type not in taxonomy["model_types"]:
        raise ValidationError(f"model_type references unknown taxonomy ID {model_type!r}")

    task_ids = model["tasks"]
    for task_id in task_ids:
        task = taxonomy["tasks"].get(task_id)
        if task is None:
            raise ValidationError(f"tasks references unknown taxonomy ID {task_id!r}")
        if model_type not in task["allowed_model_types"]:
            raise ValidationError(f"tasks entry {task_id!r} is not compatible with model_type {model_type!r}")

    inputs = model["modalities"]["inputs"]
    outputs = model["modalities"]["outputs"]
    for field, values in (("modalities.inputs", inputs), ("modalities.outputs", outputs)):
        for modality_id in values:
            if modality_id not in taxonomy["modalities"]:
                raise ValidationError(f"{field} references unknown taxonomy ID {modality_id!r}")

    architecture = model["architecture"]
    architecture_id = architecture["id"]
    architecture_entry = taxonomy["architectures"].get(architecture_id)
    if architecture_entry is None:
        raise ValidationError(f"architecture.id references unknown taxonomy ID {architecture_id!r}")
    if model_type not in architecture_entry["allowed_model_types"]:
        raise ValidationError(f"architecture {architecture_id!r} is not compatible with model_type {model_type!r}")

    component_roles: set[str] = set()
    for index, component in enumerate(architecture.get("components", [])):
        role = component["role"]
        if role in component_roles:
            raise ValidationError(f"architecture.components[{index}].role duplicates {role!r}")
        component_roles.add(role)
        component_id = component["architecture_id"]
        if component_id not in taxonomy["architectures"]:
            raise ValidationError(
                f"architecture.components[{index}].architecture_id references unknown taxonomy ID {component_id!r}"
            )

    return {
        "taxonomy_version": taxonomy_version,
        "model_type": model_type,
        "tasks": list(task_ids),
        "modalities": {"inputs": list(inputs), "outputs": list(outputs)},
        "architecture": {
            "id": architecture_id,
            "paradigm": architecture_entry["paradigm"],
            **({"variant": architecture["variant"]} if architecture.get("variant") else {}),
            **({"components": list(architecture["components"])} if architecture.get("components") else {}),
        },
        "tags": list(model.get("tags", [])),
    }
