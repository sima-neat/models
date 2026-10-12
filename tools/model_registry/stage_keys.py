"""Stage keys: what a compile, a DevKit performance run, or a DevKit accuracy run depends on.

A key hashes everything a stage's result depends on. Each published run carries its key as the run
fingerprint, so a registry run with the same key means that stage need not run again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .accuracy_config import accuracy_components, find_devkit_accuracy_config
from .accuracy_result import ACCURACY_RESULT_SCHEMA
from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError
from .model import (
    ACCURACY_REQUIREMENTS,
    ACCURACY_SCRIPT,
    CONVERSION_FILES,
    PERFORMANCE_SCRIPT,
    SCRIPTS_DIR,
    compile_invocations_for,
    load_and_validate_model,
)

MODEL_COMPILE_INPUT_PATHS = {
    "yolo26_tiny_drone": (
        "training/yolo26_tiny_drone/compile.py",
        "training/yolo26_tiny_drone/contract.py",
        "training/yolo26_tiny_drone/dataset.py",
        "training/yolo26_tiny_drone/onnx_audit.py",
        "training/yolo26_tiny_drone/paths.py",
    ),
}

# Shared code on the compile path; a change to it recompiles every model.
COMPILE_TOOLING = ("scripts/run_model_compilation.sh",)

_IGNORED_BUILD_PARTS = {".git", "__pycache__"}
_IGNORED_BUILD_SUFFIXES = {".pyc", ".pyo"}
# Conversion produces the pinned source, and the board stages consume the MPK; none changes compilation.
NON_COMPILE_SCRIPTS = {PERFORMANCE_SCRIPT, ACCURACY_SCRIPT, *CONVERSION_FILES}

PERFORMANCE_RESULT_SCHEMA = "schemas/performance-result.schema.json"
PERFORMANCE_RUN_TYPE = "model_benchmark"
ACCURACY_RUN_TYPE = "model_accuracy"
# Vulcan stores a run's fingerprint version from its environment and resolves version 1 by default.
PERFORMANCE_FINGERPRINT_VERSION = 2
ACCURACY_FINGERPRINT_VERSION = 1
# The workflow sets up the board run and benchmark_publication shapes the published run.
BOARD_STAGE_TOOLING = (
    ".github/workflows/vulcan-model-benchmarking.yml",
    "tools/model_registry/benchmark_publication.py",
)
PERFORMANCE_TOOLING = (
    *BOARD_STAGE_TOOLING,
    "scripts/run_model_performance.sh",
    "tools/model_registry/core_benchmark.py",
    "tools/model_registry/benchmark_manifest.py",
    PERFORMANCE_RESULT_SCHEMA,
)
ACCURACY_TOOLING = (
    *BOARD_STAGE_TOOLING,
    "scripts/install_dataset_registry.sh",
    "scripts/cache_accuracy_dataset.sh",
    "scripts/run_model_accuracy.sh",
    "tools/model_registry/accuracy_result.py",
    ACCURACY_RESULT_SCHEMA,
)


def _hash_files(digest: Any, root: Path, paths: set[Path] | list[Path]) -> None:
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def _key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_digests(root: Path, required: list[str], optional: list[str] = ()) -> dict[str, str | None]:
    digests: dict[str, str | None] = {}
    for relative in [*required, *optional]:
        path = root / relative
        if not path.is_file():
            if relative in required:
                raise ValidationError(f"stage input not found: {relative}")
            digests[relative] = None
            continue
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _compile_input_files(root: Path, model_id: str) -> list[Path]:
    """Code the compile runs: the shared runner, model scripts, and any declared training code."""
    candidates: list[Path] = []
    for prefix in (*COMPILE_TOOLING, *MODEL_COMPILE_INPUT_PATHS.get(model_id, ())):
        path = root / prefix.rstrip("/")
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise ValidationError(f"model compile input not found: {path}")
    model_dir = root / "models" / model_id
    candidates.extend(
        item
        for item in (model_dir / SCRIPTS_DIR).rglob("*")
        if item.is_file()
        and item.suffix in {".py", ".sh"}
        and item.relative_to(model_dir).as_posix() not in NON_COMPILE_SCRIPTS
    )
    symlinks = sorted(item for item in candidates if item.is_symlink())
    if symlinks:
        relative = ", ".join(item.relative_to(root).as_posix() for item in symlinks)
        raise ValidationError(f"build inputs must not be symbolic links: {relative}")
    return sorted(
        {
            item
            for item in candidates
            if not (_IGNORED_BUILD_PARTS & set(item.relative_to(root).parts))
            and item.suffix not in _IGNORED_BUILD_SUFFIXES
        },
        key=lambda item: item.relative_to(root).as_posix(),
    )


def _calibration_projection(model: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any] | None:
    calibration_id = binding.get("calibration")
    if calibration_id is None:
        return None
    try:
        return next(item for item in model["calibrations"] if item["id"] == calibration_id)
    except StopIteration as exc:  # Kept fail-closed even when called without full catalog validation.
        raise ValidationError(f"unknown calibration {calibration_id!r}") from exc


def _compile_consumed_fidelity(model: dict[str, Any], profile_id: str) -> list[dict[str, Any]]:
    """Fidelity entries the compile recipe evaluates for a build profile."""
    # Selection already depends on scope and profiles; hash only what the recipe reads.
    return [
        {key: value for key, value in entry.items() if key not in {"scope", "build_profiles"}}
        for entry in model["fidelity"]
        if profile_id in entry["build_profiles"]
    ]


def _component_compile_projection(component: dict[str, Any]) -> dict[str, Any]:
    return {
        key: component[key]
        for key in ("id", "source", "interface", "surgery", "recipe", "depends_on")
        if key in component
    }


def _binding_compile_projection(
    model: dict[str, Any], component: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    return {
        **{
            key: binding[key]
            for key in ("id", "parameters", "shape_profile")
            if key in binding
        },
        **(
            {
                "calibration": _calibration_projection(model, binding),
                "preprocessing": component.get("preprocessing", {}),
            }
            if binding.get("calibration") is not None
            else {}
        ),
    }


def compile_key(
    repository: str | Path,
    model_id: str,
    profile_id: str,
    compile_invocation_id: str,
    compiler_version: str,
) -> str:
    """Identity of one compile invocation: what it compiles, how, and with which compiler."""
    root = Path(repository).expanduser().resolve()
    compiler = compiler_version.strip()
    if not compiler:
        raise ValidationError("compiler version must not be empty")
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    try:
        profile = next(item for item in model["build_profiles"] if item["id"] == profile_id)
        invocation = next(
            item for item in compile_invocations_for(profile) if item["id"] == compile_invocation_id
        )
    except StopIteration as exc:
        raise ValidationError(
            f"unknown compile invocation: {model_id}/{profile_id}/{compile_invocation_id}"
        ) from exc
    components = {item["id"]: item for item in model["components"]}
    bindings = {item["id"]: item for item in profile["components"]}
    identity = {
        "model_id": model_id,
        "compile_invocation": {
            "id": compile_invocation_id,
            "components": [
                {
                    "component": _component_compile_projection(components[component_id]),
                    "binding": _binding_compile_projection(model, components[component_id], bindings[component_id]),
                }
                for component_id in invocation["components"]
            ],
        },
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
            "compatibility": profile.get("compatibility", {}),
        },
        "fidelity": _compile_consumed_fidelity(model, profile_id),
        "compiler_version": compiler,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    _hash_files(digest, root, _compile_input_files(root, model_id))
    return digest.hexdigest()


def performance_key(
    repository: str | Path,
    model: dict[str, Any],
    build_profile_id: str,
    component_id: str,
    *,
    compile_run_id: str,
    artifact_sha256: str,
) -> str:
    """Identity of one component's DevKit performance measurement."""
    root = Path(repository).expanduser().resolve()
    model_id = str(model["id"])
    entries = sorted(
        (
            entry
            for entry in model["benchmark"]
            if entry["scope"] == {"type": "component", "component": component_id}
            and build_profile_id in entry["build_profiles"]
        ),
        key=lambda item: item["id"],
    )
    if not entries:
        raise ValidationError(f"no benchmark entry applies to {model_id}/{build_profile_id}/{component_id}")
    return _key({
        "stage": "performance",
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "component_id": component_id,
        "compile_run_id": compile_run_id,
        "artifact_sha256": artifact_sha256,
        "benchmark": entries,
        "files": _file_digests(root, [f"models/{model_id}/{PERFORMANCE_SCRIPT}", *PERFORMANCE_TOOLING]),
        "runtime": resolve_benchmark_runtime(root),
    })


def accuracy_key(
    repository: str | Path,
    model: dict[str, Any],
    build_profile_id: str,
    builds: dict[str, dict[str, Any]],
) -> str:
    """Identity of a build profile's DevKit accuracy measurement over its scoped compiled components.

    ``builds`` maps each component ID to its compile ``run_id`` and ``artifact_sha256``.
    """
    root = Path(repository).expanduser().resolve()
    model_id = str(model["id"])
    entry = find_devkit_accuracy_config(model, build_profile_id)
    if entry is None:
        raise ValidationError(f"no DevKit accuracy entry applies to {model_id}/{build_profile_id}")
    scoped = accuracy_components(model, entry)
    missing = [component_id for component_id in scoped if component_id not in builds]
    if missing:
        raise ValidationError(f"accuracy {entry['id']!r} needs compiled component {missing[0]!r}")
    model_dir = f"models/{model_id}"
    return _key({
        "stage": "accuracy",
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "accuracy": entry,
        "builds": {
            component_id: {
                "compile_run_id": builds[component_id]["run_id"],
                "artifact_sha256": builds[component_id]["artifact_sha256"],
            }
            for component_id in scoped
        },
        "files": _file_digests(
            root, [f"{model_dir}/{ACCURACY_SCRIPT}", *ACCURACY_TOOLING], [f"{model_dir}/{ACCURACY_REQUIREMENTS}"]
        ),
        "runtime": resolve_benchmark_runtime(root),
    })
