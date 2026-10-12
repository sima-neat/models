"""The stage plan: which builds compile, and which DevKit stages a reused build still has to run.

A stage runs when no completed registry run carries its current key; see stage_keys.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .compile_matrix import build_matrix
from .errors import ModelRegistryError, ValidationError
from .model import load_models
from .stage_keys import (
    ACCURACY_FINGERPRINT_VERSION,
    ACCURACY_RUN_TYPE,
    PERFORMANCE_FINGERPRINT_VERSION,
    PERFORMANCE_RUN_TYPE,
    accuracy_key,
    performance_key,
)

REPOSITORY_API_PATH = "/v1/repositories/sima-neat/models"
# Besides the default branch, branches are cut only from release and integration branches.
BASE_BRANCH_PREFIXES = ("release-", "integration/")

ReadRequest = Callable[[str], dict[str, Any]]


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
    base branch by graph distance and prefer the configured integration branch
    for ties. Feature branches are never candidates: a sibling that shares a
    commit would otherwise look closer than the branch both were cut from.
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
        if branch != default_branch and not branch.startswith(BASE_BRANCH_PREFIXES):
            continue
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


def applicable_stages(entry: dict[str, Any]) -> list[str]:
    """Board stages a compile invocation declares, from its build-matrix entry."""
    return [
        *(["performance"] if entry["benchmark_component_ids"] else []),
        *(["accuracy"] if entry["accuracy_id"] else []),
    ]


def _is_not_found(error: ModelRegistryError, code: str) -> bool:
    return f"HTTP 404 {code}" in str(error)


def read_current_catalog(
    read: ReadRequest, read_api_url: str, branch: str, base_branch: str | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the branch's models record, and the document of the branch models or else the base's."""
    base_url = read_api_url.rstrip("/")
    for source in [branch, *([base_branch] if base_branch and base_branch != branch else [])]:
        try:
            payload = read(
                f"{base_url}{REPOSITORY_API_PATH}/catalogs/current?"
                f"git_ref={quote(f'refs/heads/{source}', safe='')}"
            )
        except ModelRegistryError as exc:
            if _is_not_found(exc, "catalog_not_found"):
                continue
            raise
        record = payload.get("catalog")
        if not isinstance(record, dict):
            continue
        document = record.get("document") if isinstance(record.get("document"), dict) else record
        return (record if source == branch else None), document
    return None, None


def stage_run_exists(read: ReadRequest, read_api_url: str, run_type: str, key: str, fingerprint_version: int) -> bool:
    """Whether a completed, unsuperseded registry run of this type carries the key."""
    try:
        read(
            f"{read_api_url.rstrip('/')}{REPOSITORY_API_PATH}/resolve?"
            f"run_type={run_type}&fingerprint={key}&fingerprint_version={fingerprint_version}"
        )
    except ModelRegistryError as exc:
        if _is_not_found(exc, "fingerprint_not_found"):
            return False
        raise
    return True


def catalog_builds(document: dict[str, Any] | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index a models document's builds by (model, build profile, component)."""
    builds: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model in (document or {}).get("models", []):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        for build in model.get("builds", []):
            if not isinstance(build, dict):
                continue
            key = (model["id"], build.get("build_profile_id"), build.get("component_id"))
            if all(isinstance(value, str) for value in key):
                builds[key] = build
    return builds


def plan_stages(
    repository: str | Path,
    *,
    compiler_version: str,
    read: ReadRequest | None,
    read_api_url: str | None,
    branch: str,
    base_branch: str | None = None,
    model_ids: list[str] | None = None,
    profile_ids: list[str] | None = None,
    component_ids: list[str] | None = None,
    force_rebuild_all: bool = False,
) -> dict[str, Any]:
    """Select compile invocations whose compile key changed, and the board stages reused builds lack.

    Without a reader, nothing is known to be published, so every selected invocation compiles.
    """
    root = Path(repository).expanduser().resolve()
    models = load_models(root)
    head = build_matrix(
        models,
        model_ids or sorted(models),
        profile_ids,
        component_ids,
        repository=root,
        compiler_version=compiler_version,
    )["include"]
    document = None
    if read is not None:
        if not read_api_url:
            raise ValidationError("a registry read URL is required to plan against published runs")
        _, document = read_current_catalog(read, read_api_url, branch, base_branch)
    builds = catalog_builds(document)

    compile_entries: list[dict[str, Any]] = []
    measure_entries: list[dict[str, Any]] = []
    for entry in head:
        prior = {
            component_id: builds.get((entry["model_id"], entry["build_profile_id"], component_id))
            for component_id in entry["component_ids"]
        }
        if force_rebuild_all or any(
            build is None or build.get("compile_fingerprint") != entry["build_fingerprint"]
            for build in prior.values()
        ):
            compile_entries.append({**entry, "stages": applicable_stages(entry)})
            continue
        model = models[entry["model_id"]]
        profile_id = entry["build_profile_id"]
        performance = [
            component_id
            for component_id in entry["benchmark_component_ids"]
            if not stage_run_exists(
                read,
                read_api_url,
                PERFORMANCE_RUN_TYPE,
                performance_key(
                    root,
                    model,
                    profile_id,
                    component_id,
                    compile_run_id=prior[component_id]["run_id"],
                    artifact_sha256=prior[component_id]["artifact_sha256"],
                ),
                PERFORMANCE_FINGERPRINT_VERSION,
            )
        ]
        stages = ["performance"] if performance else []
        if entry["accuracy_id"] and not stage_run_exists(
            read,
            read_api_url,
            ACCURACY_RUN_TYPE,
            accuracy_key(root, model, profile_id, prior),
            ACCURACY_FINGERPRINT_VERSION,
        ):
            stages.append("accuracy")
        if stages:
            measure_entries.append({**entry, "benchmark_component_ids": performance, "stages": stages})
    return {
        "compile": {"include": compile_entries},
        "measure": {"include": measure_entries},
        "catalog_found": document is not None,
    }
