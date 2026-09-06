from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .dependencies import resolve_benchmark_runtime, resolve_model_compiler
from .errors import ValidationError
from .model import compile_invocations_for, effective_runner_class, load_and_validate_model
from .taxonomy import TAXONOMY_RELATIVE_PATH, load_taxonomy, taxonomy_sha256

GLOBAL_COMPILE_INPUT_PATHS = (
    "deps/",
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

CALIBRATION_COMPILE_INPUT_PATHS = (
    "schemas/accuracy.schema.json",
    "tools/model_registry/accuracy.py",
    "tools/model_registry/accuracy_config.py",
    "tools/model_registry/model.py",
)

_IGNORED_BUILD_PARTS = {".git", "__pycache__"}
_IGNORED_BUILD_SUFFIXES = {".pyc", ".pyo"}
_IGNORED_MODEL_BUILD_NAMES = {"README.md"}
COMPILE_FINGERPRINT_SCHEMA = 3
VALIDATION_FINGERPRINT_SCHEMA = 1
BENCHMARK_FINGERPRINT_SCHEMA = 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _integer_constant(path: Path, name: str) -> int:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValidationError(f"could not read {name} from {path}: {exc}") from exc
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, int)
        ):
            return statement.value.value
    raise ValidationError(f"{path}: missing integer constant {name}")


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


@contextmanager
def _revision_tree(repository: Path, revision: str):
    """Materialize a committed catalog without executing code from that revision."""
    with tempfile.TemporaryDirectory(prefix="model-registry-revision-") as temporary:
        target = Path(temporary)
        archive = target / "repository.tar"
        try:
            subprocess.run(
                ["git", "archive", "--format=tar", f"--output={archive}", revision],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            with tarfile.open(archive, "r") as stream:
                stream.extractall(target / "tree", filter="data")
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            raise ValidationError(f"could not materialize catalog revision {revision}: {detail}") from exc
        yield target / "tree"


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
    model_prefixes = MODEL_COMPILE_INPUT_PATHS.get(model_id, ())
    for prefix in model_prefixes:
        path = root / prefix.rstrip("/")
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
        elif prefix in model_prefixes:
            raise ValidationError(f"model compile input not found: {path}")
    model_dir = root / "models" / model_id
    # Model-local executable helpers are compile inputs. Declarative metadata,
    # test configuration, documentation, and model.yaml are represented by the
    # semantic projection or by their own stage fingerprint instead.
    candidates.extend(
        item
        for item in model_dir.rglob("*")
        if item.is_file()
        and item.suffix in {".py", ".sh"}
        and "tests" not in item.relative_to(model_dir).parts
    )
    profile = next(item for item in model["build_profiles"] if item["id"] == profile_id)
    binding = next(item for item in profile["components"] if item["id"] == component["id"])
    if binding.get("calibration") is not None:
        candidates.extend(root / relative for relative in CALIBRATION_COMPILE_INPUT_PATHS)
        for case in model["tests"]["correctness"]:
            if profile_id not in case["build_profiles"]:
                continue
            scope = case["scope"]
            if scope != {"type": "component", "component": component["id"]}:
                continue
            config_path = root / "models" / model_id / case["config"]
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                raise ValidationError(f"could not read calibration config {config_path}: {exc}") from exc
            if isinstance(config, dict) and config.get("engine") == "model_registry":
                candidates.append(config_path)
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


def _calibration_projection(model: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any] | None:
    calibration_id = binding.get("calibration")
    if calibration_id is None:
        return None
    try:
        return next(item for item in model["calibrations"] if item["id"] == calibration_id)
    except StopIteration as exc:  # Kept fail-closed even when called without full catalog validation.
        raise ValidationError(f"unknown calibration {calibration_id!r}") from exc


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
        "schema": COMPILE_FINGERPRINT_SCHEMA,
        "model_id": model_id,
        "component": _component_compile_projection(component),
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
            "compatibility": profile.get("compatibility", {}),
            "component": _binding_compile_projection(model, component, binding),
        },
        "compiler_version": compiler,
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    _hash_files(digest, root, _component_build_input_files(root, model, component, profile_id))
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
        "schema": COMPILE_FINGERPRINT_SCHEMA,
        "model_id": model_id,
        "compile_invocation": {
            "id": compile_invocation_id,
            "python_packages": invocation.get("python_packages", []),
            "components": [
                {
                    "component": _component_compile_projection(components[component_id]),
                    "binding": _binding_compile_projection(
                        model, components[component_id], bindings[component_id]
                    ),
                }
                for component_id in component_ids
            ],
        },
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
            "compatibility": profile.get("compatibility", {}),
        },
        "compiler_version": compiler,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    input_files = {
        path
        for component_id in component_ids
        for path in _component_build_input_files(root, model, components[component_id], profile_id)
    }
    for case in model["tests"]["correctness"]:
        if profile_id not in case["build_profiles"] or case["scope"].get("type") != "package":
            continue
        config_path = root / "models" / model_id / case["config"]
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValidationError(f"could not read compile-consumed correctness config {config_path}: {exc}") from exc
        if isinstance(config, dict) and config.get("engine") == "model_recipe":
            input_files.add(config_path)
    _hash_files(digest, root, input_files)
    return digest.hexdigest()


def _test_case_applies(model: dict[str, Any], case: dict[str, Any], profile_id: str, component_id: str) -> bool:
    if profile_id not in case["build_profiles"]:
        return False
    scope = case["scope"]
    if scope == {"type": "component", "component": component_id}:
        return True
    if scope.get("type") != "package":
        return False
    return any(
        package["id"] == scope.get("package")
        and profile_id in package["build_profiles"]
        and component_id in package["components"]
        for package in model["packages"]
    )


def validation_fingerprint(
    repository: str | Path,
    model_id: str,
    component_id: str,
    profile_id: str,
    compiler_version: str,
    *,
    compile_fingerprint: str | None = None,
    artifact_sha256: str | None = None,
) -> str:
    root = Path(repository).expanduser().resolve()
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    compiled = compile_fingerprint or component_build_fingerprint(
        root, model_id, component_id, profile_id, compiler_version
    )
    cases = [
        case
        for case in model["tests"]["correctness"]
        if _test_case_applies(model, case, profile_id, component_id)
    ]
    documents = []
    uses_registry_evaluator = False
    for case in sorted(cases, key=lambda item: item["id"]):
        config_path = root / "models" / model_id / case["config"]
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValidationError(f"could not read correctness config {config_path}: {exc}") from exc
        uses_registry_evaluator = uses_registry_evaluator or (
            isinstance(config, dict) and config.get("engine") == "model_registry"
        )
        documents.append({"case": case, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
    evaluator_files = {}
    if uses_registry_evaluator:
        for relative in (
            "schemas/accuracy.schema.json",
            "tools/model_registry/accuracy.py",
            "tools/model_registry/accuracy_config.py",
        ):
            path = root / relative
            evaluator_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _canonical_sha256(
        {
            "schema": VALIDATION_FINGERPRINT_SCHEMA,
            "model_id": model_id,
            "build_profile_id": profile_id,
            "component_id": component_id,
            "compile_fingerprint": compiled,
            "compiled_artifact_sha256": artifact_sha256,
            "correctness": documents,
            "evaluator_files": evaluator_files,
        }
    )


def benchmark_fingerprint(
    repository: str | Path,
    model_id: str,
    component_id: str,
    profile_id: str,
    compiler_version: str,
    *,
    compile_fingerprint: str | None = None,
    artifact_sha256: str | None = None,
) -> str:
    root = Path(repository).expanduser().resolve()
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    compiled = compile_fingerprint or component_build_fingerprint(
        root, model_id, component_id, profile_id, compiler_version
    )
    cases = [
        case
        for case in model["tests"]["performance"]
        if _test_case_applies(model, case, profile_id, component_id)
    ]
    documents = []
    for case in sorted(cases, key=lambda item: item["id"]):
        config_path = root / "models" / model_id / case["config"]
        documents.append({"case": case, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
    return _canonical_sha256(
        {
            "schema": BENCHMARK_FINGERPRINT_SCHEMA,
            "model_id": model_id,
            "build_profile_id": profile_id,
            "component_id": component_id,
            "compile_fingerprint": compiled,
            "compiled_artifact_sha256": artifact_sha256,
            "performance": documents,
            "runtime": resolve_benchmark_runtime(root),
        }
    )


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
                validation_fingerprints: dict[str, str] = {}
                benchmark_fingerprints: dict[str, str] = {}
                if repository is not None and compiler_version is not None:
                    validation_fingerprints = {
                        component_id: validation_fingerprint(
                            repository, model_id, component_id, profile["id"], compiler_version,
                            compile_fingerprint=fingerprint,
                        )
                        for component_id in invocation_component_ids
                        if any(
                            _test_case_applies(model, case, profile["id"], component_id)
                            for case in model["tests"]["correctness"]
                        )
                    }
                    benchmark_fingerprints = {
                        component_id: benchmark_fingerprint(
                            repository, model_id, component_id, profile["id"], compiler_version,
                            compile_fingerprint=fingerprint,
                        )
                        for component_id in invocation_component_ids
                        if any(
                            _test_case_applies(model, case, profile["id"], component_id)
                            for case in model["tests"]["performance"]
                        )
                    }
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
                    **(
                        {"validation_fingerprints": validation_fingerprints}
                        if repository is not None and compiler_version is not None
                        else {}
                    ),
                    **(
                        {"benchmark_fingerprints": benchmark_fingerprints}
                        if repository is not None and compiler_version is not None
                        else {}
                    ),
                }
                include.append(entry)
    return {"include": include}


def semantic_change_plan(
    repository: str | Path,
    base: str,
    head: str,
    *,
    compiler_version: str,
    force_rebuild_all: bool = False,
) -> dict[str, Any]:
    """Compare stage inputs at two revisions and return only the work that changed.

    Head is read from the checked-out tree so uncommitted unit-test fixtures can use
    the same code path. Base is data-only materialized with ``git archive``.
    """
    root = Path(repository).expanduser().resolve()
    head_catalog = discover_catalog(root)
    head_entries = build_matrix(
        head_catalog,
        sorted(head_catalog),
        repository=root,
        compiler_version=compiler_version,
    )["include"]
    changed_paths = git_changed_paths(root, base, head)
    fallback_reason = None
    base_entries: list[dict[str, Any]] = []
    base_catalog: dict[str, dict[str, Any]] = {}
    base_taxonomy_sha = None
    base_compile_schema = None
    base_validation_schema = None
    base_benchmark_schema = None
    try:
        with _revision_tree(root, base) as base_root:
            base_catalog = discover_catalog(base_root)
            base_taxonomy_sha = taxonomy_sha256(load_taxonomy(base_root / TAXONOMY_RELATIVE_PATH))
            base_compile_schema = _integer_constant(
                base_root / "tools" / "model_registry" / "catalog.py", "COMPILE_FINGERPRINT_SCHEMA"
            )
            base_validation_schema = _integer_constant(
                base_root / "tools" / "model_registry" / "catalog.py", "VALIDATION_FINGERPRINT_SCHEMA"
            )
            base_benchmark_schema = _integer_constant(
                base_root / "tools" / "model_registry" / "catalog.py", "BENCHMARK_FINGERPRINT_SCHEMA"
            )
            base_compiler_version = resolve_model_compiler(base_root)["version"]
            base_entries = build_matrix(
                base_catalog,
                sorted(base_catalog),
                repository=base_root,
                compiler_version=base_compiler_version,
            )["include"]
    except ValidationError as exc:
        # Contract migrations are deliberately fail-safe: the first comparison
        # against an older catalog recompiles once, then subsequent revisions are
        # selected semantically.
        fallback_reason = str(exc)

    def invocation_key(entry: dict[str, Any]) -> tuple[str, str, str]:
        return entry["model_id"], entry["build_profile_id"], entry["compile_invocation_id"]

    base_by_key = {invocation_key(entry): entry for entry in base_entries}
    head_by_key = {invocation_key(entry): entry for entry in head_entries}
    compile_include: list[dict[str, Any]] = []
    validation_include: list[dict[str, Any]] = []
    benchmark_include: list[dict[str, Any]] = []
    for entry in head_entries:
        prior = base_by_key.get(invocation_key(entry))
        compile_changed = (
            force_rebuild_all
            or fallback_reason is not None
            or base_compile_schema != COMPILE_FINGERPRINT_SCHEMA
            or prior is None
            or prior.get("build_fingerprint") != entry["build_fingerprint"]
        )
        if compile_changed:
            compile_include.append(entry)
        current_validation = entry["validation_fingerprints"]
        previous_validation = prior.get("validation_fingerprints", {}) if prior is not None else {}
        for component_id, fingerprint in sorted(current_validation.items()):
            if (
                compile_changed
                or base_validation_schema != VALIDATION_FINGERPRINT_SCHEMA
                or previous_validation.get(component_id) != fingerprint
            ):
                validation_include.append({
                    "model_id": entry["model_id"],
                    "model_dir": entry["model_dir"],
                    "build_profile_id": entry["build_profile_id"],
                    "compile_invocation_id": entry["compile_invocation_id"],
                    "component_id": component_id,
                    "compile_fingerprint": entry["build_fingerprint"],
                    "validation_fingerprint": fingerprint,
                    "compile_changed": compile_changed,
                })
        current_benchmarks = entry["benchmark_fingerprints"]
        previous_benchmarks = prior.get("benchmark_fingerprints", {}) if prior is not None else {}
        if current_benchmarks and (
            compile_changed
            or base_benchmark_schema != BENCHMARK_FINGERPRINT_SCHEMA
            or any(
                previous_benchmarks.get(component_id) != fingerprint
                for component_id, fingerprint in current_benchmarks.items()
            )
        ):
            benchmark_include.append({
                "model_id": entry["model_id"],
                "model_dir": entry["model_dir"],
                "build_profile_id": entry["build_profile_id"],
                "compile_invocation_id": entry["compile_invocation_id"],
                "component_ids": sorted(current_benchmarks),
                "compile_fingerprint": entry["build_fingerprint"],
                "benchmark_fingerprints": current_benchmarks,
                "compile_changed": compile_changed,
            })

    deleted_invocations = [
        {"model_id": key[0], "build_profile_id": key[1], "compile_invocation_id": key[2]}
        for key in sorted(set(base_by_key) - set(head_by_key))
    ]
    head_taxonomy_sha = taxonomy_sha256(load_taxonomy(root / TAXONOMY_RELATIVE_PATH))
    taxonomy_changed = base_taxonomy_sha != head_taxonomy_sha

    def publication_metadata(catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
        fields = ("display_name", "description", "license", "maturity", "categories")
        return {
            model_id: {field: model.get(field) for field in fields}
            for model_id, model in sorted(catalog.items())
        }

    metadata_changed = fallback_reason is not None or (
        publication_metadata(base_catalog) != publication_metadata(head_catalog)
    )
    return {
        "include": compile_include,
        "validation_include": validation_include,
        "benchmark_include": benchmark_include,
        "metadata_changed": metadata_changed,
        "taxonomy_changed": taxonomy_changed,
        "deleted_invocations": deleted_invocations,
        "comparison": {"base": base, "head": head, "fallback_reason": fallback_reason},
        "changed_paths": changed_paths,
    }
