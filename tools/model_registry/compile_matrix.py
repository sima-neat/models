"""The compile matrix: one entry per compile invocation the models declare."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark_declarations import BENCHMARK_STAGES, benchmarked_profiles
from .errors import ValidationError
from .fingerprints import compile_fingerprint
from .model import compile_invocations_for, effective_runner_class, effective_runner_market


def build_matrix(
    models: dict[str, dict[str, Any]],
    model_ids: list[str],
    profile_ids: list[str] | None = None,
    component_ids: list[str] | None = None,
    *,
    repository: str | Path | None = None,
    compiler_version: str | None = None,
    conversion_sources: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from .conversion import selections

    conversion_sources = selections() if conversion_sources is None else conversion_sources
    unknown = sorted(set(model_ids) - set(models))
    if unknown:
        raise ValidationError(f"unknown model id(s): {', '.join(unknown)}")

    requested_profiles = set(profile_ids or [])
    available_profiles = {
        profile["id"] for model_id in set(model_ids) for profile in models[model_id]["build_profiles"]
    }
    unknown_profiles = sorted(requested_profiles - available_profiles)
    if unknown_profiles:
        raise ValidationError(f"unknown build profile id(s): {', '.join(unknown_profiles)}")

    requested_components = set(component_ids or [])
    available_components = {
        component["id"] for model_id in set(model_ids) for component in models[model_id]["components"]
    }
    unknown_components = sorted(requested_components - available_components)
    if unknown_components:
        raise ValidationError(f"unknown component id(s): {', '.join(unknown_components)}")

    if (repository is None) != (compiler_version is None):
        raise ValidationError("repository and compiler version must be supplied together")
    include: list[dict[str, Any]] = []
    for model_id in sorted(set(model_ids)):
        model = models[model_id]
        for profile in sorted(model["build_profiles"], key=lambda item: item["id"]):
            if requested_profiles and profile["id"] not in requested_profiles:
                continue
            bindings = {item["id"]: item for item in profile["components"]}
            for invocation in sorted(compile_invocations_for(profile), key=lambda item: item["id"]):
                invocation_component_ids = list(invocation["components"])
                if requested_components and not requested_components.intersection(invocation_component_ids):
                    continue
                markets = {effective_runner_market(profile, bindings[c]) for c in invocation_component_ids}
                if len(markets) != 1 or not markets <= {"spot", "on-demand"}:
                    raise ValidationError(
                        f"{model_id}/{profile['id']} has conflicting or unsupported runner_market requests"
                    )
                first_component_id = invocation_component_ids[0]
                entry: dict[str, Any] = {
                    "model_dir": f"models/{model_id}",
                    "model_id": model_id,
                    "compile_invocation_id": invocation["id"],
                    "component_id": first_component_id,
                    "component_ids": invocation_component_ids,
                    "build_profile_id": profile["id"],
                    "requested_runner_class": effective_runner_class(profile, bindings[first_component_id]),
                    "requested_runner_market": effective_runner_market(profile, bindings[first_component_id]),
                    "target": profile["target"],
                    # The benchmarks measure the whole model once this profile's package is published.
                    "stages": [
                        stage for stage in BENCHMARK_STAGES if profile["id"] in benchmarked_profiles(model, stage)
                    ],
                }
                if model["components"][0]["interface"]["format"] == "huggingface":
                    entry["runner_script"] = "scripts/run_llima_compilation.sh"
                if model.get("build-on-demand", False):
                    entry["build_on_demand"] = True
                if repository is not None and compiler_version is not None:
                    entry["compile_fingerprint"] = compile_fingerprint(
                        repository, model_id, profile["id"], invocation["id"], compiler_version,
                        conversion_sources=conversion_sources
                    )
                include.append(entry)
    return {"include": include}
