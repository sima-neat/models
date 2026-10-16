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
DEFAULT_BRANCH = "develop"
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


def is_long_lived(branch: str) -> bool:
    """Whether the branch keeps its own catalog: main, the default branch, and the branches others are cut from."""
    return branch in {DEFAULT_BRANCH, "main"} or branch.startswith(BASE_BRANCH_PREFIXES)


def resolve_base_branches(
    repository: str | Path,
    *,
    current_branch: str,
    head: str,
    preferred_branch: str | None = None,
) -> list[str]:
    """Return the branches whose catalogs the current branch may reuse builds from, closest first.

    Git does not record the branch from which another branch was created. An
    explicit PR/dispatch base therefore wins. Otherwise, every base branch that
    shares history with the head is a candidate, ordered by graph distance with
    the default branch first on ties. Feature branches are never candidates: a
    sibling that shares a commit would otherwise look closer than the branch both
    were cut from. The list may be empty, for example on a push to the default branch.
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
        if preferred not in branches:
            raise ValidationError(f"base branch {preferred!r} was not fetched from origin")
        return [] if preferred == current else [preferred]

    candidates: list[tuple[int, int, int, str]] = []
    for branch, ref in branches.items():
        if branch == current or not (branch == DEFAULT_BRANCH or branch.startswith(BASE_BRANCH_PREFIXES)):
            continue
        try:
            merge_base = _git(root, "merge-base", head, ref)
            ahead = int(_git(root, "rev-list", "--count", f"{merge_base}..{head}"))
            behind = int(_git(root, "rev-list", "--count", f"{merge_base}..{ref}"))
        except (ValidationError, ValueError):
            continue
        candidates.append((ahead + behind, behind, 0 if branch == DEFAULT_BRANCH else 1, branch))
    return [branch for *_, branch in sorted(candidates)]


def applicable_stages(entry: dict[str, Any]) -> list[str]:
    """Board stages a compile invocation declares, from its build-matrix entry."""
    return [
        *(["performance"] if entry["benchmark_component_ids"] else []),
        *(["accuracy"] if entry["accuracy_id"] else []),
    ]


def _is_not_found(error: ModelRegistryError, code: str) -> bool:
    return f"HTTP 404 {code}" in str(error)


def _read_catalog(read: ReadRequest, read_api_url: str, branch: str) -> dict[str, Any] | None:
    try:
        payload = read(
            f"{read_api_url.rstrip('/')}{REPOSITORY_API_PATH}/catalogs/current?"
            f"git_ref={quote(f'refs/heads/{branch}', safe='')}"
        )
    except ModelRegistryError as exc:
        if _is_not_found(exc, "catalog_not_found"):
            return None
        raise
    record = payload.get("catalog")
    return record if isinstance(record, dict) else None


def _document(record: dict[str, Any]) -> dict[str, Any]:
    return record["document"] if isinstance(record.get("document"), dict) else record


def read_current_catalog(
    read: ReadRequest, read_api_url: str, branch: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the branch's own catalog record and its models document."""
    record = _read_catalog(read, read_api_url, branch)
    return record, (_document(record) if record is not None else None)


def read_catalog_sources(
    read: ReadRequest, read_api_url: str, branch: str, base_branches: list[str] | None = None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the branch's own catalog record, and the documents its builds may come from, in priority order.

    A long-lived branch keeps to its own catalog and inherits only on its first push, from the closest
    base branch that has a catalog. A feature branch also reuses matching builds from every base branch.
    """
    record = _read_catalog(read, read_api_url, branch)
    documents = [_document(record)] if record is not None else []
    if record is not None and is_long_lived(branch):
        return record, documents
    for base in base_branches or []:
        base_record = _read_catalog(read, read_api_url, base) if base != branch else None
        if base_record is None:
            continue
        documents.append(_document(base_record))
        if is_long_lived(branch):
            break
    return record, documents


def _profiles(document: dict[str, Any]) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    """Index a models document's builds and packages by (model, build profile)."""
    profiles: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for model in document.get("models", []):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        for kind in ("builds", "packages"):
            for item in model.get(kind, []):
                if isinstance(item, dict) and isinstance(item.get("build_profile_id"), str):
                    profile = profiles.setdefault(
                        (model["id"], item["build_profile_id"]), {"builds": [], "packages": []}
                    )
                    profile[kind].append(item)
    return profiles


def _carries_keys(builds: list[dict[str, Any]], keys: dict[str, str]) -> bool:
    """Whether the builds include every component at its head compile key."""
    published = {build.get("component_id"): build.get("compile_fingerprint") for build in builds}
    return all(published.get(component_id) == key for component_id, key in keys.items())


def merge_catalog_sources(documents: list[dict[str, Any]], matrix: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine source catalogs into the one prior catalog a revision is planned and published against.

    Each model build profile takes its builds and packages together from the first source whose builds
    carry every compile key the head declares for that profile, else from the first source that has the
    profile at all, so its prior builds stay visible while the replacement compiles.
    """
    if not documents:
        return None
    head_keys: dict[tuple[str, str], dict[str, str]] = {}
    for entry in matrix:
        profile_keys = head_keys.setdefault((entry["model_id"], entry["build_profile_id"]), {})
        for component_id in entry["component_ids"]:
            profile_keys[component_id] = entry["build_fingerprint"]

    sources = [_profiles(document) for document in documents]
    models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (model_id, profile_id), keys in sorted(head_keys.items()):
        present = [source[(model_id, profile_id)] for source in sources if (model_id, profile_id) in source]
        matching = [profile for profile in present if _carries_keys(profile["builds"], keys)]
        chosen = matching or present
        if chosen:
            model = models.setdefault(model_id, {"builds": [], "packages": []})
            model["builds"].extend(chosen[0]["builds"])
            model["packages"].extend(chosen[0]["packages"])
    return {"models": [{"id": model_id, **entries} for model_id, entries in sorted(models.items())]}


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
    base_branches: list[str] | None = None,
    model_ids: list[str] | None = None,
    profile_ids: list[str] | None = None,
    component_ids: list[str] | None = None,
    force_rebuild_all: bool = False,
    include_on_demand: bool = False,
) -> dict[str, Any]:
    """Select compile invocations whose compile key changed, and the board stages reused builds lack.

    On-demand models require explicit model/profile selection and include_on_demand.
    Without a reader, every selected automatic invocation compiles.
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
    if include_on_demand and any(entry.get("build_on_demand", False) for entry in head) and not (
        model_ids and profile_ids
    ):
        raise ValidationError("on-demand builds require explicit --models and --profiles")
    if include_on_demand and component_ids:
        for model_id, profile_id in {(item["model_id"], item["build_profile_id"])
                                     for item in head if item.get("build_on_demand", False)}:
            profile = next(item for item in models[model_id]["build_profiles"] if item["id"] == profile_id)
            selected = {component for item in head if (item["model_id"], item["build_profile_id"])
                        == (model_id, profile_id) for component in item["component_ids"]}
            if selected != {item["id"] for item in profile["components"]}:
                raise ValidationError("on-demand candidates require a complete profile; omit --components")
    document = None
    if read is not None:
        if not read_api_url:
            raise ValidationError("a registry read URL is required to plan against published runs")
        _, sources = read_catalog_sources(read, read_api_url, branch, base_branches)
        document = merge_catalog_sources(sources, head)
    builds = catalog_builds(document)

    compile_entries: list[dict[str, Any]] = []
    measure_entries: list[dict[str, Any]] = []
    for entry in head:
        if entry.get("build_on_demand", False) and not include_on_demand:
            continue
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
    candidate_profiles = {(item["model_id"], item["build_profile_id"]) for item in compile_entries
                          if item.get("build_on_demand", False)}
    compiled_ids = {(item["model_id"], item["build_profile_id"], item["compile_invocation_id"])
                    for item in compile_entries}
    # A candidate cannot attach unchanged components from an older source commit.
    for item in head:
        if ((item["model_id"], item["build_profile_id"]) in candidate_profiles
                and (item["model_id"], item["build_profile_id"], item["compile_invocation_id"]) not in compiled_ids):
            compile_entries.append({**item, "stages": applicable_stages(item)})
    measure_entries = [item for item in measure_entries if not item.get("build_on_demand", False)]
    return {
        "compile": {"include": compile_entries},
        "measure": {"include": measure_entries},
        "catalog_found": document is not None,
    }
