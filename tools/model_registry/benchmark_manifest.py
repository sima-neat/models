from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .accuracy_config import accuracy_components
from .artifact import validate_compile_outputs
from .compile_matrix import build_matrix
from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError
from .fetch import sha256_file
from .model import SCRIPTS_DIR, benchmark_entry, load_and_validate_model

IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object: {path}")
    return value


def validate_benchmark_components(
    repository: Path,
    compiled_dir: Path,
    model_id: str,
    build_profile_id: str,
    compile_invocation_id: str,
    component_ids: list[str],
) -> None:
    model = load_and_validate_model(repository / "models" / model_id, repository / "schemas" / "model.schema.json")
    matrix = build_matrix({model_id: model}, [model_id], [build_profile_id])
    invocation = next(
        (item for item in matrix["include"] if item["compile_invocation_id"] == compile_invocation_id), None
    )
    if invocation is None:
        raise ValidationError(f"unknown benchmark compile invocation: {compile_invocation_id}")
    output_root = compiled_dir / model_id / build_profile_id
    staged_path = output_root / "registry-staged.json"
    if staged_path.is_file():
        staged = _load_json(staged_path, "registry staging manifest")
        expected_fields = {"model_id", "build_profile_id", "compile_invocation_id", "outputs"}
        if set(staged) != expected_fields or any(
            staged.get(field) != expected for field, expected in {
                "model_id": model_id,
                "build_profile_id": build_profile_id,
                "compile_invocation_id": compile_invocation_id,
            }.items()
        ):
            raise ValidationError("registry staging manifest identity is invalid")
        outputs = staged.get("outputs")
        staged_components = {
            item.get("component_id") for item in outputs if isinstance(item, dict)
        } if isinstance(outputs, list) else set()
        if not isinstance(outputs, list) or staged_components != set(invocation["component_ids"]):
            raise ValidationError("registry staging manifest must cover the complete invocation")
        for item in outputs:
            if not isinstance(item, dict) or set(item) != {"component_id", "mpk_path", "mpk_sha256", "run_id"}:
                raise ValidationError("registry staging output fields are invalid")
            relative = PurePosixPath(str(item["mpk_path"]))
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                raise ValidationError("registry staging manifest contains an unsafe MPK path")
            archive = output_root.joinpath(*relative.parts)
            digest, _ = sha256_file(archive)
            if digest != item["mpk_sha256"]:
                raise ValidationError(f"registry-staged MPK SHA-256 does not match: {item['component_id']}")
    else:
        validate_compile_outputs(
            output_root / "compile-outputs.json",
            output_root,
            model_id=model_id,
            build_profile_id=build_profile_id,
            compile_invocation_id=compile_invocation_id,
            component_ids=invocation["component_ids"],
        )
    # A reused build measures only the components whose performance is missing; an accuracy-only run measures none.
    declared = invocation["benchmark_component_ids"]
    if len(set(component_ids)) != len(component_ids) or component_ids != [c for c in declared if c in component_ids]:
        raise ValidationError("dispatched benchmark components must be declared benchmark entries, in declared order")


def make_benchmark_manifest(
    *,
    repository: str | Path,
    compiled_dir: str | Path,
    output_path: str | Path,
    branch: str,
    git_sha: str,
    model_id: str,
    build_profile_id: str,
    component_id: str,
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    compiled = Path(compiled_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if IDENTIFIER.fullmatch(model_id) is None:
        raise ValidationError(f"invalid benchmark model id: {model_id}")
    if IDENTIFIER.fullmatch(build_profile_id) is None:
        raise ValidationError(f"invalid benchmark build profile id: {build_profile_id}")
    if IDENTIFIER.fullmatch(component_id) is None:
        raise ValidationError(f"invalid benchmark component id: {component_id}")
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")

    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    entry = benchmark_entry(model, component_id, build_profile_id)
    if entry is None:
        raise ValidationError(f"no benchmark entry applies to {model_id}/{build_profile_id}/{component_id}")
    runtime = resolve_benchmark_runtime(root)
    script = root / "models" / model_id / SCRIPTS_DIR / "performance.py"
    if not script.is_file():
        raise ValidationError(f"model performance script is missing: {script}")
    compiled_builds = [
        _compiled_build(compiled, model_id, build_profile_id, item) for item in accuracy_components(model, entry)
    ]
    builds = [build for build, _ in compiled_builds]
    # The run belongs to the component's build; a package's last component's build hosts the package's run.
    host, compiler_version = compiled_builds[-1]
    manifest = {
        "model": {
            "id": model_id,
            "build_profile_id": build_profile_id,
            **{key: host[key] for key in ("component_id", "path", "sha256", "uri")},
        },
        "compatibility": {
            "platform_version": runtime["devkit"]["platform_version"],
        },
        "script": str(script),
        "runtime": runtime,
        "provenance": {
            "compile_run_id": host["compile_run_id"],
            "models_commit": git_sha.lower(),
            "models_branch": branch,
            "compiler_version": compiler_version,
            "execution_provider": "github-actions-devkit-runner",
        },
    }
    if entry["scope"]["type"] == "package":
        # A package's result names the package; its components are the builds it ran, in pipeline order.
        manifest["model"].update(
            component_id=None, package_id=entry["scope"]["package"], path=None, sha256=None, uri=None
        )
        manifest["components"] = builds
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(output), "compile_run_id": host["compile_run_id"]}


def _compiled_build(
    compiled: Path, model_id: str, build_profile_id: str, component_id: str
) -> tuple[dict[str, Any], str | None]:
    """A component's compiled MPK with the compile run that registered it, and the compiler that built it."""
    component_dir = compiled / model_id / build_profile_id / component_id
    compile_result = _load_json(component_dir / "compile-result.json", "compile result")
    publication = _load_json(component_dir / "registry-publication.json", "registry publication")
    runs = publication.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValidationError("registry publication must contain exactly one run")
    published = runs[0]
    if (
        published.get("model_id") != model_id
        or published.get("build_profile_id") != build_profile_id
        or published.get("component_id") != component_id
    ):
        raise ValidationError("registry publication build unit does not match the requested benchmark")

    archive = compile_result.get("archive")
    sha256 = compile_result.get("sha256")
    compile_run_id = published.get("run_id")
    if not isinstance(archive, str) or not archive:
        raise ValidationError("compile result archive must be a non-empty string")
    if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValidationError("compile result sha256 must be a lowercase SHA-256 digest")
    if not isinstance(compile_run_id, str) or not compile_run_id:
        raise ValidationError("registry publication run_id must be a non-empty string")
    actual_sha256, _ = sha256_file(component_dir / archive)
    if actual_sha256 != sha256:
        raise ValidationError("compiled artifact digest does not match the benchmark input")
    build = {
        "component_id": component_id,
        "path": str((component_dir / archive).resolve()),
        "sha256": sha256,
        "uri": f"vulcan://runs/{compile_run_id}/artifacts/mpk",
        "compile_run_id": compile_run_id,
    }
    return build, compile_result.get("provenance", {}).get("compiler_version")
