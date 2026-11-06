from __future__ import annotations

import copy
import re
from typing import Any

from .errors import ValidationError

RUNNER_CLASS_RANK = {
    "default": 0,
    "moderate": 1,
    "expensive": 2,
    "most-expensive": 3,
}
ELEVATED_RUNNER_CLASSES = frozenset(RUNNER_CLASS_RANK) - {"default"}
# Vulcan publishes shared approvals: one grant per model, identical in every environment.
APPROVALS_SCHEMA_VERSIONS = (2, 3)
_SPEC_REVISION = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_APPROVAL_FIELDS = {"repository", "model_id", "maximum_runner_class", "approved_by", "reason", "evidence"}


def _non_empty_string(record: dict[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"model_runner_approvals[{index}].{field} must be a non-empty string")
    return value.strip()


def _validated_approvals(spec: dict[str, Any]) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
    """Return the spec revision and each (repository, model) grant."""
    schema_version = spec.get("model_runner_approvals_schema_version")
    if type(schema_version) is not int or schema_version not in APPROVALS_SCHEMA_VERSIONS:
        raise ValidationError("unsupported Vulcan model runner approvals schema version")
    revision = spec.get("model_runner_approvals_revision")
    if not isinstance(revision, str) or _SPEC_REVISION.fullmatch(revision) is None:
        raise ValidationError("Vulcan spec model_runner_approvals_revision must be a lowercase SHA-256")

    approvals = spec.get("model_runner_approvals")
    if not isinstance(approvals, list):
        raise ValidationError("Vulcan spec model_runner_approvals must be an array")

    grants: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(approvals):
        if not isinstance(record, dict):
            raise ValidationError(f"model_runner_approvals[{index}] must be an object")
        unknown = sorted(set(record) - (_APPROVAL_FIELDS | ({"allow_on_demand"} if schema_version == 3 else set())))
        if unknown:
            raise ValidationError(f"model_runner_approvals[{index}] cannot contain {', '.join(unknown)}")
        repository = _non_empty_string(record, "repository", index)
        model_id = _non_empty_string(record, "model_id", index)
        maximum = _non_empty_string(record, "maximum_runner_class", index)
        _non_empty_string(record, "approved_by", index)
        _non_empty_string(record, "reason", index)
        _non_empty_string(record, "evidence", index)

        if _REPOSITORY.fullmatch(repository) is None:
            raise ValidationError(f"model_runner_approvals[{index}].repository is invalid")
        if _IDENTIFIER.fullmatch(model_id) is None:
            raise ValidationError(f"model_runner_approvals[{index}].model_id is invalid")
        if maximum not in (RUNNER_CLASS_RANK if schema_version == 3 else ELEVATED_RUNNER_CLASSES):
            raise ValidationError(
                f"model_runner_approvals[{index}].maximum_runner_class is unsupported for this policy version"
            )
        if (repository, model_id) in grants:
            raise ValidationError(f"duplicate Vulcan model runner approval for {repository}/{model_id}")
        if schema_version == 3 and type(record.get("allow_on_demand")) is not bool:
            raise ValidationError("v3 approval allow_on_demand must be a boolean")
        grants[(repository, model_id)] = {**record, "maximum_runner_class": maximum}
    return revision, grants


def authorize_runner_matrix(
    matrix: dict[str, Any],
    vulcan_spec: dict[str, Any],
    *,
    repository: str,
) -> dict[str, list[dict[str, Any]]]:
    """Authorize a compile matrix against the Terraform-published Vulcan spec."""

    if _REPOSITORY.fullmatch(repository) is None:
        raise ValidationError(f"invalid repository: {repository}")
    revision, grants = _validated_approvals(vulcan_spec)

    include = matrix.get("include")
    if not isinstance(include, list):
        raise ValidationError("compile matrix include must be an array")

    authorized: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(include):
        if not isinstance(raw_entry, dict):
            raise ValidationError(f"compile matrix include[{index}] must be an object")
        entry = copy.deepcopy(raw_entry)
        model_id = entry["model_id"]
        requested = entry["requested_runner_class"]
        if requested not in RUNNER_CLASS_RANK:
            raise ValidationError(
                f"{model_id}/{entry['build_profile_id']}/{entry['compile_invocation_id']} "
                f"requests unknown runner class {requested!r}"
            )

        approval_id = ""
        if requested in ELEVATED_RUNNER_CLASSES:
            grant = grants.get((repository, model_id))
            maximum = grant["maximum_runner_class"] if grant else None
            if maximum is None:
                raise ValidationError(f"{model_id} requests {requested} but has no Vulcan approval for {repository}")
            if RUNNER_CLASS_RANK[requested] > RUNNER_CLASS_RANK[maximum]:
                raise ValidationError(f"{model_id} requests {requested}, above approved maximum {maximum}")
            approval_id = f"{repository}:{model_id}"

        market = entry.get("requested_runner_market", "spot")
        if market not in ("spot", "on-demand"):
            raise ValidationError(f"{model_id} requests unsupported runner market {market!r}")
        if market == "on-demand":
            grant = grants.get((repository, model_id), {})
            if vulcan_spec["model_runner_approvals_schema_version"] != 3 or not grant.get("allow_on_demand", False):
                raise ValidationError(
                    f"{model_id} requests on-demand but has no Vulcan on-demand approval. "
                    "Request an exact model ID grant in sima-neat/vulcan config/model-runner-approvals.json "
                    "with owner, reason, compile durations/interruption evidence, and cost tradeoff."
                )
            approval_id = f"{repository}:{model_id}"
        entry["requested_runner_market"] = market
        entry["runner_market"] = market
        entry["runner_class"] = requested
        entry["runner_approval_id"] = approval_id
        entry["runner_approval_spec_revision"] = revision
        authorized.append(entry)
    return {"include": authorized}


RUNNER_PROVENANCE_FIELDS = (
    "requested_runner_class", "runner_class", "requested_runner_market", "runner_market",
    "runner_name", "runner_instance_id", "runner_instance_type", "runner_actual_market",
    "runner_approval_id", "runner_approval_spec_revision",
)


def runner_provenance(provenance: dict[str, Any]) -> dict[str, str]:
    """Carry optional runner audit fields without inventing values for older builds."""
    result = {}
    for field in RUNNER_PROVENANCE_FIELDS:
        if field in provenance:
            if not isinstance(provenance[field], str):
                raise ValidationError(f"runner provenance {field} must be a string")
            result[field] = provenance[field]
    return result
