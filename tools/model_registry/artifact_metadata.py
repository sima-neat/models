"""Validate publication metadata independently of compilation inputs."""

from .errors import ValidationError


def validate_artifact_metadata(model: dict) -> None:
    for profile in model["build_profiles"]:
        for component in profile["components"]:
            suffix = component.get("artifact_suffix")
            if suffix is None:
                continue

            filename = f"{model['id']}_{suffix}_mpk.tar.gz"
            if len(filename) > 255:
                raise ValidationError(
                    f"artifact_suffix for {model['id']}/"
                    f"{profile['id']}/{component['id']} "
                    "exceeds the 255-character filename limit"
                )
