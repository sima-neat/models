from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ValidationError
from .model import compile_invocations_for, effective_runner_class, load_and_validate_model

GLOBAL_COMPILE_INPUT_PATHS = (
    "deps/",
)

MODEL_COMPILE_INPUT_PATHS = {
    "yolo26_tiny_drone": ("training/yolo26_tiny_drone/",),
}

_IGNORED_BUILD_PARTS = {".git", "__pycache__"}
_IGNORED_BUILD_SUFFIXES = {".pyc", ".pyo"}
_IGNORED_MODEL_BUILD_NAMES = {"README.md"}


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


def _component_build_input_files(
    root: Path,
    model: dict[str, Any],
    component: dict[str, Any],
    profile_id: str,
) -> list[Path]:
    model_id = str(model["id"])
    candidates: list[Path] = []
    global_prefixes = GLOBAL_COMPILE_INPUT_PATHS
    model_prefixes = MODEL_COMPILE_INPUT_PATHS.get(model_id, ())
    for prefix in (*global_prefixes, *model_prefixes):
        path = root / prefix.rstrip("/")
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
        elif prefix in model_prefixes:
            raise ValidationError(f"model compile input not found: {path}")
    model_dir = root / "models" / model_id
    candidates.extend(
        item
        for item in model_dir.rglob("*")
        if item.is_file()
        and item.name not in _IGNORED_MODEL_BUILD_NAMES
        and "tests" not in item.relative_to(model_dir).parts
    )
    candidates.extend(
        model_dir / case["config"]
        for case in model["tests"]["performance"]
        if case["scope"] == {"type": "component", "component": component["id"]}
        and profile_id in case["build_profiles"]
    )
    missing = sorted(path for path in candidates if not path.is_file())
    if missing:
        raise ValidationError(f"model compile input not found: {missing[0]}")
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


def component_build_fingerprint(
    repository: str | Path,
    model_id: str,
    component_id: str,
    profile_id: str,
    compiler_version: str,
) -> str:
    root = Path(repository).expanduser().resolve()
    compiler = compiler_version.strip()
    if not compiler:
        raise ValidationError("compiler version must not be empty")
    digest = hashlib.sha256()
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    try:
        component = next(item for item in model["components"] if item["id"] == component_id)
        profile = next(item for item in model["build_profiles"] if item["id"] == profile_id)
        binding = next(item for item in profile["components"] if item["id"] == component_id)
    except StopIteration as exc:
        raise ValidationError(f"unknown component/profile pair: {model_id}/{profile_id}/{component_id}") from exc
    identity = {
        "schema": 2,
        "model_id": model_id,
        "component": component,
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
            "compatibility": profile.get("compatibility", {}),
            "component": binding,
        },
        "compiler_version": compiler,
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    for path in _component_build_input_files(root, model, component, profile_id):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def compile_invocation_fingerprint(
    repository: str | Path,
    model_id: str,
    compile_invocation_id: str,
    profile_id: str,
    compiler_version: str,
) -> str:
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
    component_ids = list(invocation["components"])
    identity = {
        "schema": 1,
        "model_id": model_id,
        "compile_invocation": {
            "id": compile_invocation_id,
            "python_packages": invocation.get("python_packages", []),
            "components": [
                {
                    "component": components[component_id],
                    "binding": bindings[component_id],
                }
                for component_id in component_ids
            ],
        },
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
            "compatibility": profile.get("compatibility", {}),
            "runner_class": effective_runner_class(profile, bindings[component_ids[0]]),
        },
        "compiler_version": compiler,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    input_files = {
        path
        for component_id in component_ids
        for path in _component_build_input_files(root, model, components[component_id], profile_id)
    }
    for path in sorted(input_files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _is_global_compile_input(path: str) -> bool:
    return any(_matches_path_prefix(path, prefix) for prefix in GLOBAL_COMPILE_INPUT_PATHS)


def _matches_path_prefix(path: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def select_model_ids(catalog: dict[str, dict[str, Any]], changed_paths: list[str]) -> list[str]:
    normalized_paths: set[str] = set()
    for raw_path in changed_paths:
        if not raw_path.strip():
            continue
        path = PurePosixPath(raw_path).as_posix()
        normalized_paths.add(path[2:] if path.startswith("./") else path)
    normalized = sorted(normalized_paths)
    if any(_is_global_compile_input(path) for path in normalized):
        return sorted(catalog)

    selected: set[str] = set()
    for path in normalized:
        for model_id, prefixes in MODEL_COMPILE_INPUT_PATHS.items():
            if model_id in catalog and any(_matches_path_prefix(path, prefix) for prefix in prefixes):
                selected.add(model_id)
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[0] == "models" and parts[1] in catalog:
            selected.add(parts[1])
    return sorted(selected)


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

    include: list[dict[str, Any]] = []
    for model_id in sorted(set(model_ids)):
        model = catalog[model_id]
        if repository is not None or compiler_version is not None:
            if repository is None or compiler_version is None:
                raise ValidationError("repository and compiler version must be supplied together")
        components = {item["id"]: item for item in model["components"]}
        for profile in sorted(model["build_profiles"], key=lambda item: item["id"]):
            if requested_profiles and profile["id"] not in requested_profiles:
                continue
            bindings = {item["id"]: item for item in profile["components"]}
            has_declared_invocations = "compile_invocations" in profile
            for invocation in sorted(compile_invocations_for(profile), key=lambda item: item["id"]):
                invocation_component_ids = list(invocation["components"])
                if requested_components and not requested_components.intersection(invocation_component_ids):
                    continue
                fingerprint = ""
                if repository is not None and compiler_version is not None:
                    if has_declared_invocations:
                        fingerprint = compile_invocation_fingerprint(
                            repository,
                            model_id,
                            invocation["id"],
                            profile["id"],
                            compiler_version,
                        )
                    else:
                        fingerprint = component_build_fingerprint(
                            repository,
                            model_id,
                            invocation_component_ids[0],
                            profile["id"],
                            compiler_version,
                        )
                first_component_id = invocation_component_ids[0]
                benchmark_component_ids = [
                    component_id
                    for component_id in invocation_component_ids
                    if any(
                        case["scope"] == {"type": "component", "component": component_id}
                        and profile["id"] in case["build_profiles"]
                        for case in model["tests"]["performance"]
                    )
                ]
                entry: dict[str, Any] = {
                    "model_dir": f"models/{model_id}",
                    "model_id": model_id,
                    "compile_invocation_id": invocation["id"],
                    "component_id": first_component_id,
                    "component_ids": invocation_component_ids,
                    "explicit_invocation": has_declared_invocations,
                    "build_profile_id": profile["id"],
                    "recipe": components[first_component_id]["recipe"],
                    "python_packages": invocation.get("python_packages", []),
                    "requested_runner_class": effective_runner_class(
                        profile, bindings[first_component_id]
                    ),
                    "target": profile["target"],
                    "benchmark_component_ids": benchmark_component_ids,
                    **({"build_fingerprint": fingerprint} if fingerprint else {}),
                }
                include.append(entry)
    return {"include": include}
