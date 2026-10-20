from __future__ import annotations

from typing import Any

DEVKIT_ACCURACY_ENGINE = "devkit_script"


def find_devkit_accuracy_config(model: dict[str, Any], build_profile_id: str) -> dict[str, Any] | None:
    """Return the devkit_script accuracy entry for a build profile; model validation allows at most one."""
    return next(
        (
            entry
            for entry in model["accuracy"]
            if entry.get("engine") == DEVKIT_ACCURACY_ENGINE and build_profile_id in entry["build_profiles"]
        ),
        None,
    )


def accuracy_components(model: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Components whose compiled MPKs an accuracy or benchmark entry runs, in package order."""
    scope = entry["scope"]
    if scope["type"] == "component":
        return [scope["component"]]
    return next(package["components"] for package in model["packages"] if package["id"] == scope["package"])
