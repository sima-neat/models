"""Fingerprints: what a compile, a performance benchmark, or an accuracy benchmark depends on.

A fingerprint hashes everything a stage's result depends on. Each published run carries its fingerprint as
the run fingerprint, so a registry run with the same fingerprint means that stage need not run again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmark_declarations import ACCURACY_REQUIREMENTS, SCRIPTS
from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError
from .model import CONVERSION_FILES, SCRIPTS_DIR, build_profile_for, compile_invocations_for, load_and_validate_model
from .package import package_resources
from .source import source_identity

# Shared code used by ONNX builds.
COMPILE_TOOLING = (
    "scripts/run_model_compilation.sh",
    "tools/model_registry/model.py",
    "tools/model_registry/mla_stage_simulation.py",
)

_IGNORED_BUILD_PARTS = {".git", "__pycache__"}
_IGNORED_BUILD_SUFFIXES = {".pyc", ".pyo"}
# Conversion produces the pinned source, and the benchmarks consume the MPK; none changes compilation.
NON_COMPILE_SCRIPTS = {*SCRIPTS.values(), *CONVERSION_FILES}

# Each benchmark stage's registry run type.
BENCHMARK_RUN_TYPES = {"performance": "model_performance", "accuracy": "model_accuracy"}
# The workflow runs the benchmark, model_report writes its report, and report_publication checks and publishes it.
BOARD_STAGE_TOOLING = (
    ".github/workflows/vulcan-model-benchmarking.yml",
    "tools/model_report/__init__.py",
    "tools/model_report/metrics.json",
    "schemas/model-report.schema.json",
    "tools/model_registry/report_publication.py",
)
PERFORMANCE_TOOLING = (*BOARD_STAGE_TOOLING, "scripts/run_model_performance.sh")
ACCURACY_TOOLING = (
    *BOARD_STAGE_TOOLING,
    "scripts/install_dataset_registry.sh",
    "scripts/cache_accuracy_dataset.sh",
    "scripts/run_model_accuracy.sh",
)


def _hash_files(digest: Any, root: Path, paths: set[Path] | list[Path]) -> None:
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def _fingerprint(identity: dict[str, Any]) -> str:
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
    """Code the compile runs: the shared runner and the model's scripts."""
    candidates: list[Path] = []
    for relative in COMPILE_TOOLING:
        path = root / relative
        if not path.is_file():
            raise ValidationError(f"model compile input not found: {path}")
        candidates.append(path)
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


def _component_compile_projection(component: dict[str, Any]) -> dict[str, Any]:
    source = component["source"]
    if source["type"] == "conversion":
        source = {"type": "conversion", "sha256": source["sha256"]}
    return {"id": component["id"], "source": source}


def _binding_compile_projection(
    model: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    return {
        **{
            key: binding[key]
            for key in ("id", "parameters")
            if key in binding
        },
        **(
            {"calibration": _calibration_projection(model, binding)}
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
    *,
    conversion_sources: dict[str, Any] | None = None,
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
    from .conversion import apply_selection

    apply_selection(model, conversion_sources, root=root)
    components = {item["id"]: item for item in model["components"]}
    bindings = {item["id"]: item for item in profile["components"]}
    if any(components[item]["interface"]["format"] == "huggingface" for item in invocation["components"]):
        # LLiMa recipes orchestrate a pinned compiler; graph options belong in parameters.
        return _fingerprint({
            "compiler_version": compiler,
            "target": profile["target"],
            "components": [
                {
                    "source_sha256": source_identity(components[component_id]["source"]),
                    "parameters": bindings[component_id]["parameters"],
                }
                for component_id in invocation["components"]
            ],
        })
    identity = {
        "model_id": model_id,
        "compile_invocation": {
            "id": compile_invocation_id,
            "components": [
                {
                    "component": _component_compile_projection(components[component_id]),
                    "binding": _binding_compile_projection(model, bindings[component_id]),
                }
                for component_id in invocation["components"]
            ],
        },
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
        },
        # scripts/compile.py measures numerical fidelity as it compiles.
        "fidelity": model.get("numerical_fidelity"),
        # Packages are assembled only when their profile compiles, so a changed resource must recompile it.
        # Models without resources keep their fingerprints.
        **({"resources": [
            {"id": item["id"], "source_sha256": source_identity(item["source"]), "install_path": item["install_path"]}
            for item in package_resources(model)
        ]} if model.get("resources") else {}),
        "compiler_version": compiler,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    _hash_files(digest, root, _compile_input_files(root, model_id))
    return digest.hexdigest()


def benchmark_fingerprint(
    repository: str | Path,
    model: dict[str, Any],
    build_profile_id: str,
    builds: dict[str, dict[str, Any]],
    stage: str,
) -> str:
    """Identity of a build profile's whole-model benchmark on a DevKit.

    ``builds`` maps each component ID to its compile ``run_id`` and ``artifact_sha256``; the fingerprint covers
    the build of every component, the stage's declaration, the model's script, and the shared tooling. An
    accuracy declaration holds each dataset it evaluates with the metrics its datasets/<id>.yaml requires for
    the model's tasks, so changing another task's entry or another dataset's file keeps this fingerprint. The
    declaration's ``build_profiles`` are left out, so benchmarking another profile keeps it too.
    """
    root = Path(repository).expanduser().resolve()
    model_id = str(model["id"])
    if stage not in model:
        raise ValidationError(f"{model_id} declares no {stage}")
    components = [component["id"] for component in model["components"]]
    missing = [component_id for component_id in components if component_id not in builds]
    if missing:
        raise ValidationError(f"{stage} of {model_id}/{build_profile_id} needs compiled component {missing[0]!r}")
    model_dir = f"models/{model_id}"
    required = [f"{model_dir}/{SCRIPTS[stage]}", *(PERFORMANCE_TOOLING if stage == "performance" else ACCURACY_TOOLING)]
    optional = [f"{model_dir}/{ACCURACY_REQUIREMENTS}"] if stage == "accuracy" else []
    baseline = build_profile_for(model, build_profile_id).get("benchmark_baseline") if stage == "performance" else None
    return _fingerprint({
        "stage": stage,
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "builds": {
            component_id: {
                "compile_run_id": builds[component_id]["run_id"],
                "artifact_sha256": builds[component_id]["artifact_sha256"],
            }
            for component_id in components
        },
        stage: {key: value for key, value in model[stage].items() if key != "build_profiles"},
        **({"benchmark_baseline": baseline} if baseline else {}),
        "files": _file_digests(root, required, optional),
        "runtime": resolve_benchmark_runtime(root),
    })
