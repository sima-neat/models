from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .errors import ValidationError

ACCURACY_SCHEMA_RELATIVE_PATH = Path("schemas/accuracy.schema.json")
PREPROCESSING_PARAMETERS = {
    "decode_image": {"color_space"},
    "resize": {"shortest_edge", "resample"},
    "center_crop": {"height", "width"},
    "rescale": {"factor"},
    "normalize": {"mean", "std"},
    "transpose": {"from", "to"},
}
POSTPROCESSING_PARAMETERS = {
    "softmax": {"axis"},
    "top_k": {"k"},
}


def find_accuracy_schema(config_path: str | Path, schema_path: str | Path | None = None) -> Path:
    if schema_path is not None:
        candidate = Path(schema_path).expanduser().resolve()
        if not candidate.is_file():
            raise ValidationError(f"accuracy schema not found: {candidate}")
        return candidate

    configured_root = os.environ.get("MODEL_REGISTRY_ROOT", "").strip()
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser().resolve() / ACCURACY_SCHEMA_RELATIVE_PATH)

    path = Path(config_path).expanduser().resolve()
    candidates.extend(parent / ACCURACY_SCHEMA_RELATIVE_PATH for parent in (path.parent, *path.parents))
    candidates.extend(parent / ACCURACY_SCHEMA_RELATIVE_PATH for parent in (Path.cwd(), *Path.cwd().parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValidationError(
        f"{path}: could not locate {ACCURACY_SCHEMA_RELATIVE_PATH}; "
        "run inside the repository or set MODEL_REGISTRY_ROOT"
    )


def load_accuracy_config(path: str | Path, schema_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{config_path}: could not load accuracy YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{config_path}: expected a YAML object")
    if value.get("engine") != "model_registry":
        raise ValidationError(f"{config_path}: engine must be 'model_registry'")

    schema_file = find_accuracy_schema(config_path, schema_path)
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{schema_file}: could not load accuracy schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors
        )
        raise ValidationError(f"{config_path}: {details}")
    return value


def select_accuracy_config(
    model_dir: str | Path,
    model: dict[str, Any],
    test_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    directory = Path(model_dir).expanduser().resolve()
    cases = model["tests"]["correctness"]
    if test_id is None:
        selected = find_accuracy_config(directory, model)
        if selected is None:
            raise ValidationError(
                f"{directory}: expected exactly one executable correctness test; pass --test-id explicitly"
            )
        return selected
    else:
        matches = [case for case in cases if case["id"] == test_id]
        if len(matches) != 1:
            raise ValidationError(f"{directory}: expected exactly one correctness test named {test_id!r}")
        selected = matches[0]
    path = (directory / selected["config"]).resolve()
    return path, load_accuracy_config(path)


def find_accuracy_config(
    model_dir: str | Path,
    model: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    directory = Path(model_dir).expanduser().resolve()
    executable = []
    for case in model["tests"]["correctness"]:
        path = directory / case["config"]
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationError(f"{path}: could not inspect accuracy YAML: {exc}") from exc
        if isinstance(raw, dict) and raw.get("engine") == "model_registry":
            executable.append(case)
    if not executable:
        return None
    if len(executable) != 1:
        raise ValidationError(f"{directory}: expected exactly one executable correctness test")
    path = (directory / executable[0]["config"]).resolve()
    return path, load_accuracy_config(path)


def accuracy_applies_to_variant(config: dict[str, Any], variant_id: str) -> bool:
    variants = config.get("variants")
    return variants is None or variant_id in variants


def validate_accuracy_model_contract(
    model: dict[str, Any],
    config: dict[str, Any],
    config_path: str | Path,
) -> None:
    path = Path(config_path).expanduser().resolve()
    configured_variants = set(config.get("variants", ()))
    unknown_variants = configured_variants - {variant["id"] for variant in model["compile_variants"]}
    if unknown_variants:
        raise ValidationError(
            f"{path}: accuracy variants do not match compile variants: {', '.join(sorted(unknown_variants))}"
        )

    inputs = model["interface"]["inputs"]
    if len(inputs) != 1 or inputs[0].get("layout") != "NCHW":
        raise ValidationError(f"{path}: image_classification evaluation requires one NCHW model input")

    for section, supported in (
        ("preprocessing", PREPROCESSING_PARAMETERS),
        ("postprocessing", POSTPROCESSING_PARAMETERS),
    ):
        for index, stage in enumerate(model[section]["pipeline"]):
            operation = stage["op"]
            if operation not in supported:
                raise ValidationError(f"{path}: unsupported {section} operation {operation!r}")
            missing = supported[operation] - set(stage["parameters"])
            if missing:
                raise ValidationError(
                    f"{path}: {section}.pipeline[{index}] {operation!r} missing parameters: "
                    f"{', '.join(sorted(missing))}"
                )

    executors = config["executors"]
    expected = {
        "reference": ("onnxruntime", "source", "NCHW"),
        "candidate": ("afe_sima", "compile_variant", "NHWC"),
    }
    for role, values in expected.items():
        actual = tuple(executors[role][key] for key in ("type", "artifact", "input_layout"))
        if actual != values:
            raise ValidationError(f"{path}: unsupported {role} executor contract: {actual}")

    classes = int(config["dataset"]["splits"]["validation"]["classes"])
    top_k = {int(metric["k"]) for metric in config["metrics"] if metric["type"] == "top_k_accuracy"}
    if 1 not in top_k:
        raise ValidationError(f"{path}: metrics must include top_k_accuracy with k: 1")
    if max(top_k) > classes:
        raise ValidationError(f"{path}: top-k metric exceeds configured class count {classes}")
