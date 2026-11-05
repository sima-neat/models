from __future__ import annotations

import pytest
from model_registry.errors import ValidationError
from model_registry.runner_approval import authorize_runner_matrix

REVISION = "a" * 64


def _matrix(runner_class: str = "default") -> dict:
    return {
        "include": [
            {
                "model_dir": "models/efficientdet_d3",
                "model_id": "efficientdet_d3",
                "compile_invocation_id": "w8a8_1536",
                "component_id": "model",
                "component_ids": ["model"],
                "requested_runner_class": runner_class,
                "target": "modalix",
                "build_profile_id": "w8a8_1536",
            }
        ]
    }


def _approval(maximum: str = "expensive", **overrides: str) -> dict:
    value = {
        "repository": "sima-neat/models",
        "model_id": "efficientdet_d3",
        "maximum_runner_class": maximum,
        "approved_by": "platform-team",
        "reason": "Peak RSS exceeds the default runner capacity",
        "evidence": "benchmark-run-12345",
    }
    value.update(overrides)
    return value


def _spec(*approvals: dict) -> dict:
    return {
        "model_runner_approvals_schema_version": 2,
        "model_runner_approvals_revision": REVISION,
        "model_runner_approvals": list(approvals),
    }


def _authorize(matrix: dict, spec: dict) -> dict:
    return authorize_runner_matrix(matrix, spec, repository="sima-neat/models")


def test_default_runner_needs_no_approval() -> None:
    entry = _authorize(_matrix(), _spec())["include"][0]
    assert entry["runner_class"] == "default"
    assert entry["runner_approval_id"] == ""
    assert entry["runner_approval_spec_revision"] == REVISION


def test_model_approval_authorizes_every_profile_and_component() -> None:
    matrix = _matrix("expensive")
    matrix["include"][0].update(build_profile_id="w16a16", component_ids=["backbone", "decoder"])
    entry = _authorize(matrix, _spec(_approval("most-expensive")))["include"][0]
    assert entry["runner_class"] == "expensive"
    assert entry["runner_approval_id"] == "sima-neat/models:efficientdet_d3"
    assert entry["component_ids"] == ["backbone", "decoder"]


def test_approval_can_precede_model_registration() -> None:
    matrix = _matrix("expensive")
    matrix["include"][0]["model_id"] = "future_model"
    entry = _authorize(matrix, _spec(_approval(model_id="future_model")))["include"][0]
    assert entry["runner_approval_id"] == "sima-neat/models:future_model"


def test_elevated_request_without_matching_approval_fails_closed() -> None:
    with pytest.raises(ValidationError, match="has no Vulcan approval"):
        _authorize(_matrix("expensive"), _spec())
    with pytest.raises(ValidationError, match="has no Vulcan approval"):
        _authorize(_matrix("expensive"), _spec(_approval(model_id="another_model")))
    with pytest.raises(ValidationError, match="has no Vulcan approval"):
        _authorize(_matrix("expensive"), _spec(_approval(repository="another/models")))


def test_request_above_approved_maximum_fails_closed() -> None:
    with pytest.raises(ValidationError, match="above approved maximum"):
        _authorize(_matrix("most-expensive"), _spec(_approval("expensive")))


def test_duplicate_approval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate Vulcan model runner approval"):
        _authorize(_matrix("expensive"), _spec(_approval(), _approval()))


def test_invalid_spec_revision_is_rejected() -> None:
    spec = _spec()
    spec["model_runner_approvals_revision"] = "mutable"
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        _authorize(_matrix(), spec)


@pytest.mark.parametrize("version", [None, 1, 4, "2", True])
def test_unknown_approval_schema_fails_closed(version: object) -> None:
    spec = {**_spec(_approval()), "model_runner_approvals_schema_version": version}
    with pytest.raises(ValidationError, match="schema version"):
        _authorize(_matrix("expensive"), spec)


@pytest.mark.parametrize("field", ["build_profile_id", "component_id", "environment", "expires_at"])
def test_approval_rejects_unknown_fields(field: str) -> None:
    with pytest.raises(ValidationError, match="cannot contain"):
        _authorize(_matrix("expensive"), _spec(_approval(**{field: "value"})))


@pytest.mark.parametrize("runner_class", ["default", "expensive"])
def test_v3_approves_on_demand_independently_of_size(runner_class):
    matrix = _matrix(runner_class)
    matrix["include"][0]["requested_runner_market"] = "on-demand"
    spec = _spec(_approval(runner_class, allow_on_demand=True))
    spec["model_runner_approvals_schema_version"] = 3
    entry = _authorize(matrix, spec)["include"][0]
    assert entry["runner_market"] == entry["requested_runner_market"] == "on-demand"
    assert entry["runner_class"] == runner_class
    assert entry["runner_approval_id"] == "sima-neat/models:efficientdet_d3"
    matrix["include"][0]["requested_runner_market"] = "spot"
    assert _authorize(matrix, spec)["include"][0]["runner_market"] == "spot"


@pytest.mark.parametrize("version,allow", [(2, None), (3, False)])
def test_size_approval_does_not_approve_market(version, allow):
    matrix = _matrix()
    matrix["include"][0]["requested_runner_market"] = "on-demand"
    grant = _approval()
    if allow is not None:
        grant["allow_on_demand"] = allow
    spec = _spec(grant)
    spec["model_runner_approvals_schema_version"] = version
    with pytest.raises(ValidationError, match="no Vulcan on-demand approval"):
        _authorize(matrix, spec)


@pytest.mark.parametrize("value", [None, "true", 1, [], {}])
def test_v3_rejects_malformed_market_grants(value):
    spec = _spec(_approval(allow_on_demand=value))
    spec["model_runner_approvals_schema_version"] = 3
    with pytest.raises(ValidationError, match="boolean"):
        _authorize(_matrix(), spec)


@pytest.mark.parametrize("market", ["invalid", "", None, [], {}])
def test_unknown_market_fails_closed(market):
    matrix = _matrix()
    matrix["include"][0]["requested_runner_market"] = market
    with pytest.raises(ValidationError, match="unsupported runner market"):
        _authorize(matrix, _spec())


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("maximum", ["moderate", "expensive", "most-expensive"])
@pytest.mark.parametrize("requested", ["moderate", "expensive", "most-expensive"])
def test_extended_runner_classes_respect_maximum(version, maximum, requested):
    order = ["default", "moderate", "expensive", "most-expensive"]
    spec = _spec(_approval(maximum, **({"allow_on_demand": False} if version == 3 else {})))
    spec["model_runner_approvals_schema_version"] = version
    if order.index(requested) > order.index(maximum):
        with pytest.raises(ValidationError, match="above approved maximum"):
            _authorize(_matrix(requested), spec)
    else:
        assert _authorize(_matrix(requested), spec)["include"][0]["runner_class"] == requested


@pytest.mark.parametrize("requested", ["moderate"])
def test_new_classes_require_grant_and_independent_market_permission(requested):
    matrix = _matrix(requested)
    with pytest.raises(ValidationError, match="has no Vulcan approval"):
        _authorize(matrix, _spec())
    matrix["include"][0]["requested_runner_market"] = "on-demand"
    spec = _spec(_approval(requested, allow_on_demand=False))
    spec["model_runner_approvals_schema_version"] = 3
    with pytest.raises(ValidationError, match="has no Vulcan on-demand approval"):
        _authorize(matrix, spec)
    spec["model_runner_approvals"][0]["allow_on_demand"] = True
    assert _authorize(matrix, spec)["include"][0]["runner_market"] == "on-demand"
