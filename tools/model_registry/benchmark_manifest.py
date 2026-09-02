from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError
from .model import load_and_validate_model
from .performance_config import benchmark_settings, load_performance_config

IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object: {path}")
    return value


def _performance_configuration(
    root: Path, model_id: str, component_id: str, build_profile_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = root / "models" / model_id
    model = load_and_validate_model(model_dir, root / "schemas" / "model.schema.json")
    cases = [
        case
        for case in model["tests"]["performance"]
        if case["scope"] == {"type": "component", "component": component_id}
        and build_profile_id in case["build_profiles"]
    ]
    if len(cases) != 1:
        raise ValidationError(
            f"expected exactly one performance test for {model_id}/{build_profile_id}/{component_id}"
        )
    path = model_dir / cases[0]["config"]
    document = load_performance_config(path, root / "schemas" / "performance-config-v1.schema.json")
    return benchmark_settings(document)


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

    runtime = resolve_benchmark_runtime(root)
    benchmark, model_options = _performance_configuration(root, model_id, component_id, build_profile_id)
    manifest = {
        "schema_version": 1,
        "model": {
            "id": model_id,
            "build_profile_id": build_profile_id,
            "component_id": component_id,
            "path": str((component_dir / archive).resolve()),
            "sha256": sha256,
            "uri": f"vulcan://runs/{compile_run_id}/artifacts/mpk",
        },
        "compatibility": {
            "platform_version": runtime["devkit"]["platform_version"],
            "neat_version": runtime["neat_core"]["version"],
        },
        "benchmark": benchmark,
        "provenance": {
            "compile_run_id": compile_run_id,
            "models_commit": git_sha.lower(),
            "models_branch": branch,
            "compiler_version": compile_result.get("provenance", {}).get("compiler_version"),
            "execution_provider": "github-actions-devkit-runner",
        },
    }
    if model_options:
        manifest["model_options"] = model_options
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(output), "compile_run_id": compile_run_id, "model_path": manifest["model"]["path"]}
