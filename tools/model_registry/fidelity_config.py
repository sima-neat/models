from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ValidationError


def validate_fidelity_result(
    config: dict[str, Any],
    report: dict[str, Any],
    path: str | Path,
) -> None:
    report_path = Path(path)
    expected = {
        "component_id": config["result_component"],
        "status": config["status"],
        "dataset": {
            key: config["dataset"][key]
            for key in ("id", "variant", "sha256")
        },
        "selection": config["selection"],
        "samples": config["selection"]["samples"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValidationError(f"{report_path}: numerical fidelity {field} does not match its configuration")

    metrics = {metric["name"]: metric for metric in report["metrics"]}
    if set(metrics) != set(config["outputs"]):
        raise ValidationError(f"{report_path}: numerical fidelity metrics do not match configured outputs")


def find_fidelity_config(
    model_dir: str | Path,
    model: dict[str, Any],
    *,
    build_profile_id: str,
    component_id: str,
) -> dict[str, Any] | None:
    directory = Path(model_dir).expanduser().resolve()
    packages = {package["id"]: package for package in model["packages"]}
    matches = []
    for entry in model["fidelity"]:
        if build_profile_id not in entry["build_profiles"]:
            continue
        scope = entry["scope"]
        applies = scope == {"type": "component", "component": component_id}
        if scope["type"] == "package":
            applies = component_id in packages[scope["package"]]["components"]
        if applies:
            matches.append(entry)
    if not matches:
        return None
    if len(matches) != 1:
        raise ValidationError(
            f"{directory}: expected exactly one fidelity entry for {build_profile_id}/{component_id}"
        )
    return matches[0]
