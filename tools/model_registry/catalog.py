from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ValidationError
from .model import load_and_validate_model

GLOBAL_COMPILE_INPUT_PATHS = (
    "deps/",
    # Model recipes import these modules while preparing immutable source and
    # calibration inputs. Changes can therefore affect every compiled MPK.
    "tools/model_registry/__init__.py",
    "tools/model_registry/errors.py",
    "tools/model_registry/fetch.py",
    "tools/model_registry/model.py",
    "tools/model_registry/source.py",
)

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


def _build_input_files(root: Path, model_id: str) -> list[Path]:
    candidates: list[Path] = []
    for prefix in GLOBAL_COMPILE_INPUT_PATHS:
        path = root / prefix.rstrip("/")
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
    model_dir = root / "models" / model_id
    candidates.extend(
        item
        for item in model_dir.rglob("*")
        if item.is_file()
        and item.name not in _IGNORED_MODEL_BUILD_NAMES
        and "tests" not in item.relative_to(model_dir).parts
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


def model_build_fingerprint(repository: str | Path, model_id: str, compiler_version: str) -> str:
    root = Path(repository).expanduser().resolve()
    compiler = compiler_version.strip()
    if not compiler:
        raise ValidationError("compiler version must not be empty")
    digest = hashlib.sha256()
    identity = {"schema": 1, "model_id": model_id, "compiler_version": compiler}
    digest.update(json.dumps(identity, sort_keys=True).encode())
    for path in _build_input_files(root, model_id):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _is_global_compile_input(path: str) -> bool:
    return any(
        path == prefix or (prefix.endswith("/") and path.startswith(prefix))
        for prefix in GLOBAL_COMPILE_INPUT_PATHS
    )


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
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[0] == "models" and parts[1] in catalog:
            selected.add(parts[1])
    return sorted(selected)


def build_matrix(
    catalog: dict[str, dict[str, Any]],
    model_ids: list[str],
    variant_ids: list[str] | None = None,
    *,
    repository: str | Path | None = None,
    compiler_version: str | None = None,
    previous_models: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    unknown = sorted(set(model_ids) - set(catalog))
    if unknown:
        raise ValidationError(f"unknown model id(s): {', '.join(unknown)}")

    requested_variants = set(variant_ids or [])
    available_variants = {
        variant["id"] for model_id in set(model_ids) for variant in catalog[model_id]["compile_variants"]
    }
    unknown_variants = sorted(requested_variants - available_variants)
    if unknown_variants:
        raise ValidationError(f"unknown variant id(s): {', '.join(unknown_variants)}")

    unsupported_only: list[str] = []
    for variant_id in sorted(requested_variants):
        matches = [
            (model_id, variant)
            for model_id in sorted(set(model_ids))
            for variant in catalog[model_id]["compile_variants"]
            if variant["id"] == variant_id
        ]
        if matches and all(variant.get("status", "supported") == "unsupported" for _, variant in matches):
            evidence = "; ".join(f"{model_id}: {variant['reason']}" for model_id, variant in matches)
            unsupported_only.append(f"{variant_id} ({evidence})")
    if unsupported_only:
        raise ValidationError(f"requested variant id(s) are unsupported: {', '.join(unsupported_only)}")

    include: list[dict[str, str]] = []
    for model_id in sorted(set(model_ids)):
        model = catalog[model_id]
        fingerprint = ""
        if repository is not None or compiler_version is not None or previous_models is not None:
            if repository is None or compiler_version is None:
                raise ValidationError("repository and compiler version are required for reusable matrix selection")
            fingerprint = model_build_fingerprint(repository, model_id, compiler_version)
            previous = (previous_models or {}).get(model_id)
            if isinstance(previous, dict) and previous.get("build_fingerprint") == fingerprint:
                continue
        for variant in sorted(model["compile_variants"], key=lambda item: item["id"]):
            if requested_variants and variant["id"] not in requested_variants:
                continue
            if variant.get("status", "supported") == "unsupported":
                continue
            include.append(
                {
                    "model_dir": f"models/{model_id}",
                    "model_id": model_id,
                    "recipe": variant["recipe"],
                    "target": variant["target"],
                    "variant_id": variant["id"],
                    **({"build_fingerprint": fingerprint} if fingerprint else {}),
                }
            )
    return {"include": include}
