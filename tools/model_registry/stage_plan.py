"""The stage plan: which builds compile, and which benchmarks a reused build still has to run.

A stage runs when no completed registry run carries its current fingerprint; see fingerprints.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .compile_matrix import build_matrix
from .errors import ModelRegistryError, ValidationError
from .fingerprints import BENCHMARK_RUN_TYPES, benchmark_fingerprint
from .model import load_models
from .registry_api.runs import REPOSITORY_PATH
from .registry_api.transport import ReadRequest

DEFAULT_BRANCH = "develop"
# Customers read this branch's catalog, so it ships what the default branch built and measured.
PRODUCTION_BRANCH = "main"
# Besides the default branch, branches are cut only from release and integration branches.
BASE_BRANCH_PREFIXES = ("release-", "integration/")



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


def _is_not_found(error: ModelRegistryError, code: str) -> bool:
    return f"HTTP 404 {code}" in str(error)


def _read_catalog(read: ReadRequest, read_api_url: str, branch: str) -> dict[str, Any] | None:
    try:
        payload = read(
            f"{read_api_url.rstrip('/')}{REPOSITORY_PATH}/catalogs/current?"
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


def _profiles(document: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index outputs, approved deployments and benchmark fingerprints by (model, build profile)."""
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
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
        for profile_id, selection in model.get("approved_deployments", {}).items():
            profile = profiles.setdefault((model["id"], profile_id), {"builds": [], "packages": []})
            profile["approved_deployments"] = selection
        for profile_id, fingerprints in (model.get("benchmark_fingerprints") or {}).items():
            if (model["id"], profile_id) in profiles:
                profiles[(model["id"], profile_id)]["benchmark_fingerprints"] = fingerprints
    return profiles


def _pending_fingerprints(document: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    """Each build profile the catalog is compiling, by (model, build profile): its compile fingerprint per component."""
    return {
        (item["model_id"], item["build_profile_id"]): {
            component_id: item["compile_fingerprint"] for component_id in item["component_ids"]
        }
        for item in document.get("pending_builds", [])
    }


def _head_fingerprints(matrix: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    """Each build profile the head declares, by (model, build profile): its compile fingerprint per component."""
    fingerprints: dict[tuple[str, str], dict[str, str]] = {}
    for entry in matrix:
        profile_fingerprints = fingerprints.setdefault((entry["model_id"], entry["build_profile_id"]), {})
        for component_id in entry["component_ids"]:
            profile_fingerprints[component_id] = entry["compile_fingerprint"]
    return fingerprints


def _carries_fingerprints(profile: dict[str, Any], fingerprints: dict[str, str]) -> bool:
    """Whether the selected deployment or builds carry every component's head compile fingerprint."""
    published = {build.get("component_id"): build.get("compile_fingerprint") for build in profile["builds"]}
    selection = profile.get("approved_deployments")
    if selection:
        published = {component_id: invocation["compile_fingerprint"]
                     for invocation in selection["requirements"]["invocations"]
                     for component_id in invocation["component_ids"]}
    return all(published.get(component_id) == fingerprint for component_id, fingerprint in fingerprints.items())


def merge_catalog_sources(documents: list[dict[str, Any]], matrix: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine source catalogs into the one prior catalog a revision is planned and published against.

    Each model build profile takes its outputs, approved deployments and benchmark fingerprints from the first
    source whose builds carry every compile fingerprint the head declares for that profile, else from the first
    source that is compiling those fingerprints, else from the first source that has the profile at all, so its
    prior builds and their benchmarks stay visible while the replacement compiles.
    """
    if not documents:
        return None
    sources = [(_profiles(document), _pending_fingerprints(document)) for document in documents]
    models: dict[str, dict[str, Any]] = {}
    for (model_id, profile_id), fingerprints in sorted(_head_fingerprints(matrix).items()):
        present = [
            (profiles[(model_id, profile_id)], pending)
            for profiles, pending in sources
            if (model_id, profile_id) in profiles
        ]
        matching = [profile for profile, _ in present if _carries_fingerprints(profile, fingerprints)]
        compiling = [profile for profile, pending in present if pending.get((model_id, profile_id)) == fingerprints]
        chosen = matching or compiling or [profile for profile, _ in present]
        if chosen:
            model = models.setdefault(model_id, {"builds": [], "packages": []})
            model["builds"].extend(chosen[0]["builds"])
            model["packages"].extend(chosen[0]["packages"])
            if "approved_deployments" in chosen[0]:
                model.setdefault("approved_deployments", {})[profile_id] = chosen[0]["approved_deployments"]
            if "benchmark_fingerprints" in chosen[0]:
                model.setdefault("benchmark_fingerprints", {})[profile_id] = chosen[0]["benchmark_fingerprints"]
    return {"models": [{"id": model_id, **entries} for model_id, entries in sorted(models.items())]}


def read_default_branch_builds(
    read: ReadRequest, read_api_url: str, branch: str, matrix: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the catalog source the production branch takes finished builds from: none for any other branch.

    The source holds each automatic build profile the default branch has built at the head compile
    fingerprints. Placed ahead of the production branch's own catalog, it makes a merge list the runs the
    default branch measured, so their results apply and nothing compiles again. An on-demand profile is
    left out, because the registry rejects a snapshot that replaces the builds of a promoted profile. A
    build stays in the source while the default branch compiles its replacement, so a push to the default
    branch between this branch's plan and its publication does not take away a build the plan relied on.
    """
    if branch != PRODUCTION_BRANCH:
        return []
    record = _read_catalog(read, read_api_url, DEFAULT_BRANCH)
    if record is None:
        return []
    document = _document(record)
    profiles = _profiles(document)
    on_demand = {entry["model_id"] for entry in matrix if entry.get("build_on_demand", False)}
    models: dict[str, dict[str, Any]] = {}
    for (model_id, profile_id), fingerprints in sorted(_head_fingerprints(matrix).items()):
        profile = profiles.get((model_id, profile_id))
        if model_id in on_demand or profile is None:
            continue
        if not _carries_fingerprints({"builds": profile["builds"]}, fingerprints):
            continue
        model = models.setdefault(model_id, {"builds": [], "packages": []})
        model["builds"].extend(profile["builds"])
        model["packages"].extend(profile["packages"])
        if "benchmark_fingerprints" in profile:
            model.setdefault("benchmark_fingerprints", {})[profile_id] = profile["benchmark_fingerprints"]
    return [{"models": [{"id": model_id, **entries} for model_id, entries in sorted(models.items())]}]


def stage_run_exists(read: ReadRequest, read_api_url: str, run_type: str, fingerprint: str) -> bool:
    """Whether a completed, unsuperseded registry run of this type carries the fingerprint."""
    try:
        read(
            f"{read_api_url.rstrip('/')}{REPOSITORY_PATH}/resolve?"
            f"run_type={run_type}&fingerprint={fingerprint}"
        )
    except ModelRegistryError as exc:
        if _is_not_found(exc, "fingerprint_not_found"):
            return False
        raise
    return True


def plan_promoted_benchmarks(
    *, branch: str, stages: list[str], read: ReadRequest, read_api_url: str,
) -> dict[str, Any]:
    """Plan missing approved benchmarks, excluding text generation, without compiling or changing the catalog."""
    if branch not in {"develop", "main"}:
        raise ValidationError("Promoted benchmark batches require the develop or main branch")
    if not stages or set(stages) - set(BENCHMARK_RUN_TYPES):
        raise ValidationError("Benchmark stages must be performance and/or accuracy")
    _, document = read_current_catalog(read, read_api_url, branch)
    if document is None:
        raise ValidationError(f"No current catalog for {branch}")
    include, skipped = [], []
    for model in document["models"]:
        if not model.get("build_on_demand") or {"text_generation", "visual_question_answering"}.intersection(
            model["categories"]["tasks"]
        ):
            continue
        for profile in sorted(model.get("approved_deployments", {})):
            fingerprints = model.get("benchmark_fingerprints", {}).get(profile, {})
            requested = [stage for stage in dict.fromkeys(stages) if stage in fingerprints]
            missing = [stage for stage in requested if not stage_run_exists(
                read, read_api_url, BENCHMARK_RUN_TYPES[stage], fingerprints[stage],
            )]
            if missing:
                include.append({"model_id": model["id"], "build_profile_id": profile,
                                "compile_invocation_id": profile, "stages": missing})
            else:
                skipped.append({"model_id": model["id"], "build_profile_id": profile,
                                "reason": "already measured" if requested else "no requested benchmark references"})
    return {"source_branch": branch, "source_sha": document["catalog_revision"], "include": include, "skipped": skipped}


def _missing_results(
    read: ReadRequest, read_api_url: str, model: dict[str, Any], profile_id: str
) -> list[str]:
    """The benchmark stages of a catalog model's build profile whose fingerprint no completed run carries."""
    fingerprints = (model.get("benchmark_fingerprints") or {}).get(profile_id, {})
    return [
        stage for stage in BENCHMARK_RUN_TYPES
        if stage in fingerprints
        and not stage_run_exists(read, read_api_url, BENCHMARK_RUN_TYPES[stage], fingerprints[stage])
    ]


def _promoted_on_production(
    model: dict[str, Any], production_selections: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Split the profiles the production branch promotes: same deployment as this catalog model's, or another."""
    selections = model.get("approved_deployments", {})
    same = [
        profile_id for profile_id, selection in sorted(production_selections.items())
        if selections.get(profile_id, {}).get("deployment_id") == selection.get("deployment_id")
    ]
    return same, sorted(set(production_selections) - set(same))


def find_main_merge_blockers(
    repository: str | Path, *, head: str, read: ReadRequest, read_api_url: str
) -> dict[str, list[str]]:
    """Say why merging the default branch's `head` into the production branch would change what customers see.

    `repository` is checked out at the merge result. A blocker means the production branch would compile,
    would show a model without the results it shows today, or would keep offering a promoted build of a
    build profile the merge result no longer declares. A warning names a promoted build the merge leaves
    without results until it is promoted on the production branch too.
    """
    root = Path(repository).expanduser().resolve()
    blockers: list[str] = []
    warnings: list[str] = []
    if _git(root, "rev-parse", "HEAD^{tree}") != _git(root, "rev-parse", f"{head}^{{tree}}"):
        blockers.append(
            f"the merge result differs from {DEFAULT_BRANCH}: merge {PRODUCTION_BRANCH} into {DEFAULT_BRANCH} first"
        )
    _, document = read_current_catalog(read, read_api_url, DEFAULT_BRANCH)
    if document is None:
        return {"blockers": [*blockers, f"{DEFAULT_BRANCH} has no catalog"], "warnings": warnings}
    if document.get("catalog_revision") != head.lower():
        blockers.append(
            f"the {DEFAULT_BRANCH} catalog is at {document.get('catalog_revision')}, not at {head}: "
            f"wait for the {DEFAULT_BRANCH} run of this commit"
        )
    blockers.extend(
        f"{item['model_id']}/{item['build_profile_id']} is still compiling on {DEFAULT_BRANCH}"
        for item in document.get("pending_builds", [])
    )
    _, production = read_current_catalog(read, read_api_url, PRODUCTION_BRANCH)
    promoted = {
        model["id"]: model.get("approved_deployments", {}) for model in (production or {}).get("models", [])
    }
    for model in document.get("models", []):
        if model.get("build_on_demand", False):
            shown, replaced = _promoted_on_production(model, promoted.get(model["id"], {}))
            warnings.extend(
                f"{model['id']}/{profile_id} is promoted to another build on {DEFAULT_BRANCH}: promote it on "
                f"{PRODUCTION_BRANCH} after the merge, until then it has no results there"
                for profile_id in replaced
            )
        else:
            shown = sorted(model.get("benchmark_fingerprints") or {})
        blockers.extend(
            f"{model['id']}/{profile_id} has no {stage} result on {DEFAULT_BRANCH}"
            for profile_id in shown
            for stage in _missing_results(read, read_api_url, model, profile_id)
        )
    declared = {
        model_id: {profile["id"] for profile in model["build_profiles"]}
        for model_id, model in load_models(root).items()
    }
    # Only a withdrawal takes a promoted build out of a catalog, so the registry keeps offering this one.
    blockers.extend(
        f"{model['id']}/{profile_id} is promoted on {branch}, and model.yaml no longer declares it: "
        f"withdraw the promotion on {branch}"
        for branch, catalog in ((DEFAULT_BRANCH, document), (PRODUCTION_BRANCH, production or {}))
        for model in catalog.get("models", [])
        for profile_id in sorted(model.get("approved_deployments", {}))
        if profile_id not in declared.get(model["id"], set())
    )
    return {"blockers": blockers, "warnings": warnings}


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
    conversion_client: Any = None,
) -> dict[str, Any]:
    """Select compile invocations whose compile fingerprint changed, and the benchmarks reused builds lack.

    A feature branch leaves to its closest base branch every build profile that branch is compiling at the head
    fingerprints and every benchmark whose fingerprint its catalog records, so a merged base branch's work is not
    repeated.

    On-demand models require explicit model/profile selection and include_on_demand.
    Without a reader, every selected automatic invocation compiles.
    """
    root = Path(repository).expanduser().resolve()
    models = load_models(root)
    from .conversion import plan_conversions

    selected_ids = model_ids or sorted(models)
    unknown = sorted(set(selected_ids) - set(models))
    if unknown:
        raise ValidationError(f"unknown model id(s): {', '.join(unknown)}")
    conversion_sources = plan_conversions(
        root, {model_id: models[model_id] for model_id in sorted(set(selected_ids))},
        read=read, read_api_url=read_api_url, client=conversion_client, branch=branch,
    )
    head = build_matrix(
        models,
        selected_ids,
        profile_ids,
        component_ids,
        repository=root,
        compiler_version=compiler_version,
        conversion_sources=conversion_sources,
    )["include"]
    if include_on_demand and any(entry.get("build_on_demand", False) for entry in head) and not (
        model_ids and profile_ids
    ):
        raise ValidationError("on-demand builds require explicit --models and --profiles")
    document = None
    base: dict[str, Any] = {}
    if read is not None:
        if not read_api_url:
            raise ValidationError("a registry read URL is required to plan against published runs")
        record, sources = read_catalog_sources(read, read_api_url, branch, base_branches)
        base_sources = sources[1:] if record is not None else sources
        default_branch_builds = read_default_branch_builds(read, read_api_url, branch, head)
        revision = record.get("catalog_revision") if record is not None else None
        if default_branch_builds and revision and revision == _git(root, "rev-parse", "HEAD"):
            # Like publish_catalog, a second run of a revision keeps the builds its first snapshot listed.
            default_branch_builds = []
        document = merge_catalog_sources([*default_branch_builds, *sources], head)
        if base_sources and not is_long_lived(branch) and not force_rebuild_all:
            base = base_sources[0]
    builds = catalog_builds(document)
    base_pending = _pending_fingerprints(base)
    base_benchmarks = {
        (model["id"], profile_id): fingerprints
        for model in base.get("models", [])
        for profile_id, fingerprints in (model.get("benchmark_fingerprints") or {}).items()
    }

    compile_entries: list[dict[str, Any]] = []
    benchmark_entries: list[dict[str, Any]] = []
    for entry in head:
        if entry.get("build_on_demand", False) and not include_on_demand:
            continue
        key = (entry["model_id"], entry["build_profile_id"])
        fingerprints = {component_id: entry["compile_fingerprint"] for component_id in entry["component_ids"]}
        if base_pending.get(key) == fingerprints:
            continue
        prior = {
            component_id: builds.get((entry["model_id"], entry["build_profile_id"], component_id))
            for component_id in entry["component_ids"]
        }
        if force_rebuild_all or any(
            build is None or build.get("compile_fingerprint") != entry["compile_fingerprint"]
            for build in prior.values()
        ):
            compile_entries.append(entry)
            continue
        model = models[entry["model_id"]]
        base_benchmark_fingerprints = base_benchmarks.get(key, {})
        stages = []
        for stage in entry["stages"]:
            fingerprint = benchmark_fingerprint(root, model, entry["build_profile_id"], prior, stage)
            if base_benchmark_fingerprints.get(stage) != fingerprint and not stage_run_exists(
                read, read_api_url, BENCHMARK_RUN_TYPES[stage], fingerprint
            ):
                stages.append(stage)
        if stages:
            benchmark_entries.append({**entry, "stages": stages})
    benchmark_entries = [item for item in benchmark_entries if not item.get("build_on_demand", False)]
    return {
        "compile": {"include": compile_entries},
        "benchmark": {"include": benchmark_entries},
        "catalog_found": document is not None,
        "conversion_sources": conversion_sources,
    }
