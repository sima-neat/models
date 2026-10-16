from __future__ import annotations

from datetime import datetime, timezone

import pytest
from model_registry.errors import ValidationError
from model_registry.runner_approval import authorize_runner_matrix

NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)
REVISION = "a" * 64


def _matrix(runner_class: str = "default") -> dict:
    return {
        "include": [
            {
                "model_dir": "models/efficientdet_d3",
                "model_id": "efficientdet_d3",
                "component_id": "model",
                "recipe": "compile.py",
                "requested_runner_class": runner_class,
                "target": "modalix",
                "build_profile_id": "w8a8_1536",
            }
        ]
    }


def _grouped_matrix(runner_class: str = "default") -> dict:
    matrix = _matrix(runner_class)
    entry = matrix["include"][0]
    entry["compile_invocation_id"] = "combined"
    entry["component_ids"] = ["model", "postprocessor"]
    return matrix


def _approval(maximum: str = "expensive", **overrides: str) -> dict:
    value = {
        "repository": "sima-neat/models",
        "model_id": "efficientdet_d3",
        "component_id": "model",
        "build_profile_id": "w8a8_1536",
        "environment": "staging",
        "maximum_runner_class": maximum,
        "approved_by": "platform-team",
        "approved_at": "2026-09-01T00:00:00Z",
        "reason": "Peak RSS exceeds the default runner capacity",
        "evidence": "benchmark-run-12345",
        "expires_at": "2027-03-01T00:00:00Z",
    }
    value.update(overrides)
    return value


def _spec(*approvals: dict) -> dict:
    return {
        "model_runner_approvals_revision": REVISION,
        "model_runner_approvals": list(approvals),
    }


def _authorize(matrix: dict, spec: dict) -> dict:
    return authorize_runner_matrix(
        matrix,
        spec,
        repository="sima-neat/models",
        environment="staging",
        now=NOW,
    )


def test_default_runner_needs_no_approval() -> None:
    entry = _authorize(_matrix(), _spec())["include"][0]
    assert entry["requested_runner_class"] == "default"
    assert entry["runner_class"] == "default"
    assert entry["runner_approval_id"] == ""
    assert entry["runner_approval_spec_revision"] == REVISION


def test_matching_approval_authorizes_only_the_requested_class() -> None:
    entry = _authorize(_matrix("expensive"), _spec(_approval("most-expensive")))["include"][0]
    assert entry["runner_class"] == "expensive"
    assert entry["runner_approval_id"] == (
        "sima-neat/models:efficientdet_d3:w8a8_1536:model:staging"
    )


def test_grouped_default_runner_preserves_component_list_without_approval() -> None:
    entry = _authorize(_grouped_matrix(), _spec())["include"][0]

    assert entry["component_ids"] == ["model", "postprocessor"]
    assert entry["runner_approval_id"] == ""
    assert entry["runner_approval_ids"] == []


def test_grouped_elevated_runner_requires_and_records_every_component_approval() -> None:
    entry = _authorize(
        _grouped_matrix("expensive"),
        _spec(_approval(), _approval(component_id="postprocessor")),
    )["include"][0]

    assert entry["runner_approval_id"] == (
        "sima-neat/models:efficientdet_d3:w8a8_1536:model:staging"
    )
    assert entry["runner_approval_ids"] == [
        "sima-neat/models:efficientdet_d3:w8a8_1536:model:staging",
        "sima-neat/models:efficientdet_d3:w8a8_1536:postprocessor:staging",
    ]


def test_grouped_elevated_runner_fails_when_one_component_lacks_approval() -> None:
    with pytest.raises(ValidationError, match="postprocessor.*has no unique Vulcan approval"):
        _authorize(_grouped_matrix("expensive"), _spec(_approval()))


def test_legacy_variant_approval_is_limited_to_conventional_model_component() -> None:
    approval = _approval()
    approval["variant_id"] = approval.pop("build_profile_id")
    del approval["component_id"]

    entry = _authorize(_matrix("expensive"), _spec(approval))["include"][0]
    assert entry["runner_class"] == "expensive"

    matrix = _matrix("expensive")
    matrix["include"][0]["component_id"] = "encoder"
    with pytest.raises(ValidationError, match="has no unique Vulcan approval"):
        _authorize(matrix, _spec(approval))


def test_elevated_request_without_matching_approval_fails_closed() -> None:
    with pytest.raises(ValidationError, match="has no unique Vulcan approval"):
        _authorize(_matrix("expensive"), _spec())


def test_request_above_approved_maximum_fails_closed() -> None:
    with pytest.raises(ValidationError, match="above approved maximum"):
        _authorize(_matrix("most-expensive"), _spec(_approval("expensive")))


def test_expired_approval_fails_closed() -> None:
    with pytest.raises(ValidationError, match="expired"):
        _authorize(_matrix("expensive"), _spec(_approval(expires_at="2026-09-01T12:00:00Z")))


def test_approval_is_scoped_to_repository_variant_and_environment() -> None:
    with pytest.raises(ValidationError, match="has no unique Vulcan approval"):
        _authorize(_matrix("expensive"), _spec(_approval(environment="production")))


def test_duplicate_approval_identity_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate Vulcan model runner approval"):
        _authorize(_matrix("expensive"), _spec(_approval(), _approval()))


def test_invalid_spec_revision_is_rejected() -> None:
    spec = _spec()
    spec["model_runner_approvals_revision"] = "mutable"
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        _authorize(_matrix(), spec)


def _shared_spec(*approvals: dict) -> dict:
    records = [
        {key: value for key, value in approval.items() if key not in {"environment", "approved_at", "expires_at"}}
        for approval in approvals
    ]
    return {**_spec(*records), "model_runner_approvals_schema_version": 2}


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_shared_approval_authorizes_identical_capacity_and_provenance(environment: str) -> None:
    result = authorize_runner_matrix(
        _matrix("expensive"), _shared_spec(_approval()),
        repository="sima-neat/models", environment=environment,
        now=datetime(2040, 1, 1, tzinfo=timezone.utc),
    )["include"][0]
    assert result["runner_class"] == "expensive"
    assert result["runner_approval_id"] == "sima-neat/models:efficientdet_d3:modalix_int8_1536:model"
    assert result["runner_approval_spec_revision"] == REVISION


@pytest.mark.parametrize("version", [0, 3, "2", None, True])
def test_unknown_approval_schema_fails_closed(version: object) -> None:
    spec = {**_shared_spec(_approval()), "model_runner_approvals_schema_version": version}
    with pytest.raises(ValidationError, match="schema version"):
        _authorize(_matrix("expensive"), spec)


@pytest.mark.parametrize("field,value", [
    ("environment", "staging"), ("expires_at", "2027-01-01T00:00:00Z"), ("approved_at", "2026-09-01T00:00:00Z"),
])
def test_shared_approval_rejects_legacy_controls(field: str, value: str) -> None:
    spec = _shared_spec(_approval())
    spec["model_runner_approvals"][0][field] = value
    with pytest.raises(ValidationError, match="cannot contain"):
        _authorize(_matrix("expensive"), spec)


def test_shared_approval_preserves_ceiling_and_component_scope() -> None:
    with pytest.raises(ValidationError, match="above approved maximum"):
        _authorize(_matrix("most-expensive"), _shared_spec(_approval()))
    with pytest.raises(ValidationError, match="no unique"):
        _authorize(_grouped_matrix("expensive"), _shared_spec(_approval()))
    with pytest.raises(ValidationError, match="duplicate"):
        _authorize(_matrix("expensive"), _shared_spec(_approval(), _approval()))
    with pytest.raises(ValidationError, match="no unique"):
        _authorize(_matrix("expensive"), _shared_spec(_approval(build_profile_id="another_profile")))


def test_shared_grouped_approval_records_all_components() -> None:
    result = _authorize(
        _grouped_matrix("expensive"), _shared_spec(_approval(), _approval(component_id="postprocessor")),
    )["include"][0]
    assert result["runner_approval_ids"] == [
        "sima-neat/models:efficientdet_d3:modalix_int8_1536:model",
        "sima-neat/models:efficientdet_d3:modalix_int8_1536:postprocessor",
    ]
