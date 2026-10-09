from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .accuracy_config import find_devkit_accuracy_config
from .artifact_metadata import validate_artifact_metadata
from .errors import ValidationError
from .model import (
    ACCURACY_SCRIPT,
    CONVERSION_FILES,
    SCRIPTS_DIR,
    compile_invocations_for,
    effective_runner_class,
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
PERFORMANCE_SCRIPT = f"{SCRIPTS_DIR}/performance.py"
# Conversion produces the pinned source, and the board stages consume the MPK; none changes compilation.
NON_COMPILE_SCRIPTS = {PERFORMANCE_SCRIPT, ACCURACY_SCRIPT, *CONVERSION_FILES}


def _hash_files(digest: Any, root: Path, paths: set[Path] | list[Path]) -> None:
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def discover_catalog(repository: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(repository).expanduser().resolve()
    model_root = root / "models"
    discovered: dict[str, dict[str, Any]] = {}
    if not model_root.is_dir():
        raise ValidationError(f"model catalog not found: {model_root}")

    for definition in sorted(model_root.glob("*/model.yaml")):
        model = load_and_validate_model(definition.parent, root / "schemas" / "model.schema.json")
        validate_artifact_metadata(model)
        model_id = str(model["id"])
        if model_id in discovered:
            raise ValidationError(f"duplicate model id {model_id!r}: {definition}")
        discovered[model_id] = model
    return discovered


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise ValidationError(f"git {' '.join(arguments)} failed: {detail}") from exc
    return result.stdout.strip()


def resolve_base_branch(
    repository: str | Path,
    *,
    current_branch: str,
    head: str,
    preferred_branch: str | None = None,
    default_branch: str = "develop",
) -> dict[str, str]:
    """Resolve a deterministic comparison branch for an initial branch build.

    Git does not record the branch from which another branch was created. An
    explicit PR/dispatch base therefore wins. Otherwise, use the closest remote
    branch by graph distance and prefer the configured integration branch for
    ties.
    """
    root = Path(repository).expanduser().resolve()
    current = current_branch.strip()
    if not current:
        raise ValidationError("current branch must not be empty")

    refs = [
        line.strip()
        for line in _git(root, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").splitlines()
        if line.strip() and line.strip() != "origin/HEAD"
    ]
    branches = {ref.removeprefix("origin/"): ref for ref in refs if ref.startswith("origin/")}

    preferred = preferred_branch.strip() if preferred_branch else ""
    if preferred:
        ref = branches.get(preferred)
        if ref is None:
            raise ValidationError(f"base branch {preferred!r} was not fetched from origin")
        merge_base = _git(root, "merge-base", head, ref)
        return {"base_branch": preferred, "base_sha": _git(root, "rev-parse", ref), "merge_base": merge_base}

    branches.pop(current, None)
    candidates: list[tuple[int, int, int, str, str, str]] = []
    for branch, ref in branches.items():
        try:
            merge_base = _git(root, "merge-base", head, ref)
            ahead = int(_git(root, "rev-list", "--count", f"{merge_base}..{head}"))
            behind = int(_git(root, "rev-list", "--count", f"{merge_base}..{ref}"))
        except (ValidationError, ValueError):
            continue
        default_penalty = 0 if branch == default_branch else 1
        candidates.append((ahead + behind, behind, default_penalty, branch, ref, merge_base))

    if not candidates:
        raise ValidationError("could not resolve a related remote base branch")
    _, _, _, branch, ref, merge_base = min(candidates)
    return {"base_branch": branch, "base_sha": _git(root, "rev-parse", ref), "merge_base": merge_base}


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


def compile_fingerprint(
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


def build_matrix(
    catalog: dict[str, dict[str, Any]],
    model_ids: list[str],
    profile_ids: list[str] | None = None,
    component_ids: list[str] | None = None,
    *,
    repository: str | Path | None = None,
    compiler_version: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    unknown = sorted(set(model_ids) - set(catalog))
    if unknown:
        raise ValidationError(f"unknown model id(s): {', '.join(unknown)}")

    requested_profiles = set(profile_ids or [])
    available_profiles = {
        profile["id"] for model_id in set(model_ids) for profile in catalog[model_id]["build_profiles"]
    }
    unknown_profiles = sorted(requested_profiles - available_profiles)
    if unknown_profiles:
        raise ValidationError(f"unknown build profile id(s): {', '.join(unknown_profiles)}")

    requested_components = set(component_ids or [])
    available_components = {
        component["id"] for model_id in set(model_ids) for component in catalog[model_id]["components"]
    }
    unknown_components = sorted(requested_components - available_components)
    if unknown_components:
        raise ValidationError(f"unknown component id(s): {', '.join(unknown_components)}")

    if (repository is None) != (compiler_version is None):
        raise ValidationError("repository and compiler version must be supplied together")
    include: list[dict[str, Any]] = []
    for model_id in sorted(set(model_ids)):
        model = catalog[model_id]
        for profile in sorted(model["build_profiles"], key=lambda item: item["id"]):
            if requested_profiles and profile["id"] not in requested_profiles:
                continue
            bindings = {item["id"]: item for item in profile["components"]}
            accuracy = find_devkit_accuracy_config(model, profile["id"])
            for invocation in sorted(compile_invocations_for(profile), key=lambda item: item["id"]):
                invocation_component_ids = list(invocation["components"])
                if requested_components and not requested_components.intersection(invocation_component_ids):
                    continue
                first_component_id = invocation_component_ids[0]
                entry: dict[str, Any] = {
                    "model_dir": f"models/{model_id}",
                    "model_id": model_id,
                    "compile_invocation_id": invocation["id"],
                    "component_id": first_component_id,
                    "component_ids": invocation_component_ids,
                    "build_profile_id": profile["id"],
                    "requested_runner_class": effective_runner_class(profile, bindings[first_component_id]),
                    "target": profile["target"],
                    "benchmark_component_ids": [
                        component_id
                        for component_id in invocation_component_ids
                        if any(
                            case["scope"] == {"type": "component", "component": component_id}
                            and profile["id"] in case["build_profiles"]
                            for case in model["benchmark"]
                        )
                    ],
                    # Model validation keeps an accuracy entry's components in one invocation.
                    "accuracy_id": (
                        accuracy["id"]
                        if accuracy is not None and accuracy["result_component"] in invocation_component_ids
                        else None
                    ),
                }
                if repository is not None and compiler_version is not None:
                    entry["build_fingerprint"] = compile_fingerprint(
                        repository, model_id, profile["id"], invocation["id"], compiler_version
                    )
                include.append(entry)
    return {"include": include}
