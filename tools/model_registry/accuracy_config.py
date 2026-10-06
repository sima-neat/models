from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ValidationError

PREPROCESSING_PARAMETERS = {
    "decode_image": {"color_space"},
    "resize": {"shortest_edge", "resample"},
    "center_crop": {"height", "width"},
    "rescale": {"factor"},
    "normalize": {"mean", "std"},
    "transpose": {"from", "to"},
}
POSTPROCESSING_PARAMETERS = {
    "softmax": {"axis"},
    "top_k": {"k"},
}


def select_accuracy_config(
    model_dir: str | Path,
    model: dict[str, Any],
    test_id: str | None = None,
    *,
    build_profile_id: str | None = None,
    component_id: str | None = None,
) -> dict[str, Any]:
    """Return the model_registry validation entry selected by ID or by build coordinates."""
    directory = Path(model_dir).expanduser().resolve()
    if test_id is not None:
        matches = [entry for entry in model["validation"] if entry["id"] == test_id]
        if len(matches) != 1 or matches[0].get("engine") != "model_registry":
            raise ValidationError(f"{directory}: expected exactly one model_registry validation entry {test_id!r}")
        return matches[0]
    selected = find_accuracy_config(directory, model, build_profile_id=build_profile_id, component_id=component_id)
    if selected is None and (build_profile_id is not None or component_id is not None):
        selected = find_accuracy_config(directory, model)
    if selected is None:
        raise ValidationError(
            f"{directory}: expected exactly one model_registry validation entry; pass --test-id explicitly"
        )
    return selected


def find_accuracy_config(
    model_dir: str | Path,
    model: dict[str, Any],
    *,
    build_profile_id: str | None = None,
    component_id: str | None = None,
) -> dict[str, Any] | None:
    directory = Path(model_dir).expanduser().resolve()
    executable = [
        entry
        for entry in model["validation"]
        if entry.get("engine") == "model_registry"
        and (build_profile_id is None or build_profile_id in entry["build_profiles"])
        and (component_id is None or entry["scope"] == {"type": "component", "component": component_id})
    ]
    if not executable:
        return None
    if len(executable) != 1:
        scope = f" for {build_profile_id or '*'}/{component_id or '*'}"
        raise ValidationError(f"{directory}: expected exactly one model_registry validation entry{scope}")
    return executable[0]


def validate_accuracy_model_contract(
    model: dict[str, Any],
    config: dict[str, Any],
    model_dir: str | Path,
) -> None:
    label = f"{Path(model_dir).expanduser().resolve()}: validation {config['id']!r}"
    scope = config["scope"]
    if scope["type"] != "component":
        raise ValidationError(f"{label}: image_classification evaluation requires component scope")
    component = next(item for item in model["components"] if item["id"] == scope["component"])
    entrypoint = component["interface"]["entrypoints"][0]
    inputs = entrypoint["inputs"]
    if len(inputs) != 1 or inputs[0].get("layout") != "NCHW":
        raise ValidationError(f"{label}: image_classification evaluation requires one NCHW model input")

    for section, supported in (
        ("preprocessing", PREPROCESSING_PARAMETERS),
        ("postprocessing", POSTPROCESSING_PARAMETERS),
    ):
        for index, stage in enumerate(component[section]["pipeline"]):
            operation = stage["op"]
            if operation not in supported:
                raise ValidationError(f"{label}: unsupported {section} operation {operation!r}")
            missing = supported[operation] - set(stage["parameters"])
            if missing:
                raise ValidationError(
                    f"{label}: {section}.pipeline[{index}] {operation!r} missing parameters: "
                    f"{', '.join(sorted(missing))}"
                )

    executors = config["executors"]
    expected = {
        "reference": ("onnxruntime", "source", "NCHW"),
        "candidate": ("afe_sima", "compiled_component", "NHWC"),
    }
    for role, values in expected.items():
        actual = tuple(executors[role][key] for key in ("type", "artifact", "input_layout"))
        if actual != values:
            raise ValidationError(f"{label}: unsupported {role} executor contract: {actual}")

    classes = int(config["dataset"]["splits"]["validation"]["classes"])
    top_k = {int(metric["k"]) for metric in config["metrics"] if metric["type"] == "top_k_accuracy"}
    if 1 not in top_k:
        raise ValidationError(f"{label}: metrics must include top_k_accuracy with k: 1")
    if max(top_k) > classes:
        raise ValidationError(f"{label}: top-k metric exceeds configured class count {classes}")
