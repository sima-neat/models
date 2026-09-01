from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from .accuracy_config import load_accuracy_config, validate_accuracy_model_contract
from .errors import SourcePolicyError, ValidationError
from .source import SourceSpec
from .taxonomy import find_taxonomy, load_taxonomy, validate_model_categorization

MODEL_FILE = "model.yaml"
SCHEMA_RELATIVE_PATH = Path("schemas/model.schema.json")


def variant_requires_calibration(variant: dict[str, Any]) -> bool:
    return "calibration_samples" in variant.get("parameters", {})


def model_file(model_dir: str | Path) -> Path:
    directory = Path(model_dir).expanduser().resolve()
    path = directory / MODEL_FILE
    if not path.is_file():
        raise ValidationError(f"{directory}: missing {MODEL_FILE}")
    return path


def load_model(model_dir: str | Path) -> dict[str, Any]:
    path = model_file(model_dir)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: could not load YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a YAML object")
    return value


def find_schema(model_dir: str | Path, schema_path: str | Path | None = None) -> Path:
    if schema_path is not None:
        candidate = Path(schema_path).expanduser().resolve()
        if not candidate.is_file():
            raise ValidationError(f"schema not found: {candidate}")
        return candidate

    configured_root = os.environ.get("MODEL_REGISTRY_ROOT", "").strip()
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser().resolve() / SCHEMA_RELATIVE_PATH)

    directory = Path(model_dir).expanduser().resolve()
    candidates.extend(parent / SCHEMA_RELATIVE_PATH for parent in (directory, *directory.parents))
    candidates.extend(parent / SCHEMA_RELATIVE_PATH for parent in (Path.cwd(), *Path.cwd().parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValidationError(
        f"{directory}: could not locate {SCHEMA_RELATIVE_PATH}; run inside the repository or set MODEL_REGISTRY_ROOT"
    )


def _format_path(parts: list[Any]) -> str:
    rendered = ""
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + str(part)
    return rendered or "<root>"


def _error_path(error: Any) -> str:
    parts = list(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = next((name for name in error.validator_value if name not in error.instance), None)
        if missing is not None:
            parts.append(missing)
    return _format_path(parts)


def validate_model(
    model_dir: str | Path,
    model: dict[str, Any],
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> None:
    directory = Path(model_dir).expanduser().resolve()
    schema_file = find_schema(directory, schema_path)
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{schema_file}: could not load schema: {exc}") from exc

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(model), key=lambda item: tuple(str(part) for part in item.absolute_path))
    if errors:
        details = "; ".join(f"{_error_path(error)}: {error.message}" for error in errors)
        raise ValidationError(f"{directory}: {details}")

    taxonomy = load_taxonomy(find_taxonomy(directory, taxonomy_path))
    try:
        validate_model_categorization(model, taxonomy)
    except ValidationError as exc:
        raise ValidationError(f"{directory}: {exc}") from exc

    model_id = str(model["id"])
    if directory.name != model_id:
        raise ValidationError(f"{directory}: id must match directory name; found {model_id!r}")

    try:
        SourceSpec.from_mapping(model["source"])
        calibration = model["calibration"]
        if calibration["status"] == "ready" and "source" in calibration:
            SourceSpec.from_mapping(calibration["source"])
    except (KeyError, TypeError, ValueError, SourcePolicyError) as exc:
        raise ValidationError(f"{directory}: source policy: {exc}") from exc

    variant_ids = [variant["id"] for variant in model["compile_variants"]]
    duplicates = sorted({item for item in variant_ids if variant_ids.count(item) > 1})
    if duplicates:
        raise ValidationError(f"{directory}: duplicate compile variant ids: {', '.join(duplicates)}")

    referenced_paths = [case["config"] for group in model["tests"].values() for case in group]
    referenced_paths.extend(variant["recipe"] for variant in model["compile_variants"])
    if model["surgery"]["type"] == "python":
        referenced_paths.append(model["surgery"]["entrypoint"].split(":", 1)[0])
    for relative in referenced_paths:
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            raise ValidationError(f"{directory}: referenced file does not exist inside model directory: {relative}")

    for case in model["tests"]["correctness"]:
        candidate = (directory / case["config"]).resolve()
        try:
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationError(f"{candidate}: could not inspect correctness YAML: {exc}") from exc
        if isinstance(raw, dict) and raw.get("engine") == "model_registry":
            config = load_accuracy_config(candidate)
            validate_accuracy_model_contract(model, config, candidate)


def load_and_validate_model(
    model_dir: str | Path,
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    model = load_model(model_dir)
    validate_model(model_dir, model, schema_path, taxonomy_path)
    return model
