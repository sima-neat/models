from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError

IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
DEFAULT_BENCHMARK = {
    "samples": 1000,
    "trials": 3,
    "include_plugin_latency": False,
}
BOUNDED_RUNNER_FIELDS = {
    "mode",
    "samples",
    "trials",
    "warmup_samples",
    "include_plugin_latency",
    "original_width",
    "original_height",
}


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object: {path}")
    return value


def _performance_configuration(root: Path, model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "models" / model_id / "tests" / "performance.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"could not read benchmark configuration {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValidationError(f"benchmark configuration must be a mapping: {path}")
    model_options = document.get("model_options", {})
    if not isinstance(model_options, dict):
        raise ValidationError(f"model_options must be a mapping: {path}")
    configured = document.get("benchmark")
    if configured is None:
        return dict(DEFAULT_BENCHMARK), model_options
    if not isinstance(configured, dict):
        raise ValidationError(f"benchmark must be a mapping: {path}")
    unknown = sorted(set(configured) - BOUNDED_RUNNER_FIELDS)
    if unknown:
        raise ValidationError(f"benchmark contains unknown field(s): {', '.join(unknown)}")
    if configured.get("mode") != "bounded_runner":
        raise ValidationError("benchmark.mode must be bounded_runner")
    samples = configured.get("samples")
    trials = configured.get("trials", DEFAULT_BENCHMARK["trials"])
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (samples, trials)):
        raise ValidationError("benchmark samples and trials must be positive integers")
    if trials > 100:
        raise ValidationError("benchmark.trials must be <= 100")
    result = {
        "samples": samples,
        "trials": trials,
        "include_plugin_latency": configured.get("include_plugin_latency", False),
    }
    if not isinstance(result["include_plugin_latency"], bool):
        raise ValidationError("benchmark.include_plugin_latency must be a boolean")
    width, height = configured.get("original_width"), configured.get("original_height")
    if (width is None) != (height is None):
        raise ValidationError("benchmark.original_width and original_height must be provided together")
    if width is not None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (width, height)):
            raise ValidationError("benchmark original_width and original_height must be positive integers")
        result.update({"original_width": width, "original_height": height})
    warmup_samples = configured.get("warmup_samples")
    if not isinstance(warmup_samples, int) or isinstance(warmup_samples, bool) or warmup_samples <= 0:
        raise ValidationError("benchmark.warmup_samples must be a positive integer")
    result.update({"mode": "bounded_runner", "warmup_samples": warmup_samples})

    return result, model_options


def make_benchmark_manifest(
    *,
    repository: str | Path,
    compiled_dir: str | Path,
    output_path: str | Path,
    branch: str,
    git_sha: str,
    model_id: str,
    variant_id: str,
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    compiled = Path(compiled_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if IDENTIFIER.fullmatch(model_id) is None:
        raise ValidationError(f"invalid benchmark model id: {model_id}")
    if IDENTIFIER.fullmatch(variant_id) is None:
        raise ValidationError(f"invalid benchmark variant id: {variant_id}")
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")

    variant_dir = compiled / model_id / variant_id
    compile_result = _load_json(variant_dir / "compile-result.json", "compile result")
    publication = _load_json(variant_dir / "registry-publication.json", "registry publication")
    runs = publication.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValidationError("registry publication must contain exactly one run")
    published = runs[0]
    if published.get("model_id") != model_id or published.get("variant_id") != variant_id:
        raise ValidationError("registry publication model/variant does not match the requested benchmark")

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
    benchmark, model_options = _performance_configuration(root, model_id)
    manifest = {
        "schema_version": 1,
        "model": {
            "id": model_id,
            "variant_id": variant_id,
            "path": str((variant_dir / archive).resolve()),
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
