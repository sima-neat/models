"""Stage keys and the stage plan: which builds compile, and what a reused build still has to measure.

A key identifies everything a stage's result depends on. Each published stage run carries its key as
the run fingerprint, so a completed registry run with the same key means that stage need not run again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .accuracy_config import accuracy_components, find_devkit_accuracy_config
from .accuracy_result import ACCURACY_RESULT_SCHEMA
from .catalog import PERFORMANCE_SCRIPT, build_matrix, discover_catalog
from .dependencies import resolve_benchmark_runtime
from .errors import ModelRegistryError, ValidationError
from .model import ACCURACY_REQUIREMENTS, ACCURACY_SCRIPT

PERFORMANCE_RESULT_SCHEMA = "schemas/performance-result.schema.json"
PERFORMANCE_RUN_TYPE = "model_benchmark"
ACCURACY_RUN_TYPE = "model_accuracy"
# Vulcan stores a run's fingerprint version from its environment and resolves version 1 by default.
PERFORMANCE_FINGERPRINT_VERSION = 2
ACCURACY_FINGERPRINT_VERSION = 1
PERFORMANCE_TOOLING = (
    "scripts/run_model_performance.sh",
    "tools/model_registry/core_benchmark.py",
    "tools/model_registry/benchmark_manifest.py",
    PERFORMANCE_RESULT_SCHEMA,
)
ACCURACY_TOOLING = (
    "scripts/run_model_accuracy.sh",
    "tools/model_registry/accuracy_result.py",
    ACCURACY_RESULT_SCHEMA,
)
REPOSITORY_API_PATH = "/v1/repositories/sima-neat/models"

ReadRequest = Callable[[str], dict[str, Any]]


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


def applicable_stages(entry: dict[str, Any]) -> list[str]:
    """Board stages a compile invocation declares, from its build-matrix entry."""
    return [
        *(["performance"] if entry["benchmark_component_ids"] else []),
        *(["accuracy"] if entry["accuracy_id"] else []),
    ]


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


def _is_not_found(error: ModelRegistryError, code: str) -> bool:
    return f"HTTP 404 {code}" in str(error)


def read_current_catalog(
    read: ReadRequest, read_api_url: str, branch: str, base_branch: str | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the branch's catalog record, and the document of the branch catalog or else the base's."""
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
    """Index a catalog document's builds by (model, build profile, component)."""
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
    catalog = discover_catalog(root)
    head = build_matrix(
        catalog,
        model_ids or sorted(catalog),
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
        model = catalog[entry["model_id"]]
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
