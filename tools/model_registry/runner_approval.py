from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any

from .errors import ValidationError

RUNNER_CLASS_RANK = {
    "default": 0,
    "expensive": 1,
    "most-expensive": 2,
}
ELEVATED_RUNNER_CLASSES = frozenset({"expensive", "most-expensive"})
_SPEC_REVISION = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty RFC 3339 timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _non_empty_string(record: dict[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"model_runner_approvals[{index}].{field} must be a non-empty string")
    return value.strip()


def _validated_approvals(spec: dict[str, Any], now: datetime) -> tuple[str, list[dict[str, Any]]]:
    revision = spec.get("model_runner_approvals_revision")
    if not isinstance(revision, str) or _SPEC_REVISION.fullmatch(revision) is None:
        raise ValidationError("Vulcan spec model_runner_approvals_revision must be a lowercase SHA-256")

    approvals = spec.get("model_runner_approvals")
    if not isinstance(approvals, list):
        raise ValidationError("Vulcan spec model_runner_approvals must be an array")

    validated: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, str, str]] = set()
    for index, raw in enumerate(approvals):
        if not isinstance(raw, dict):
            raise ValidationError(f"model_runner_approvals[{index}] must be an object")
        record = dict(raw)
        repository = _non_empty_string(record, "repository", index)
        model_id = _non_empty_string(record, "model_id", index)
        component_id = _non_empty_string(record, "component_id", index)
        build_profile_id = _non_empty_string(record, "build_profile_id", index)
        environment = _non_empty_string(record, "environment", index)
        maximum = _non_empty_string(record, "maximum_runner_class", index)
        _non_empty_string(record, "approved_by", index)
        _non_empty_string(record, "reason", index)
        _non_empty_string(record, "evidence", index)

        if _REPOSITORY.fullmatch(repository) is None:
            raise ValidationError(f"model_runner_approvals[{index}].repository is invalid")
        if _IDENTIFIER.fullmatch(model_id) is None:
            raise ValidationError(f"model_runner_approvals[{index}].model_id is invalid")
        if _IDENTIFIER.fullmatch(component_id) is None:
            raise ValidationError(f"model_runner_approvals[{index}].component_id is invalid")
        if _IDENTIFIER.fullmatch(build_profile_id) is None:
            raise ValidationError(f"model_runner_approvals[{index}].build_profile_id is invalid")
        if maximum not in ELEVATED_RUNNER_CLASSES:
            raise ValidationError(
                f"model_runner_approvals[{index}].maximum_runner_class must be expensive or most-expensive"
            )

        approved_at = _timestamp(record.get("approved_at"), f"model_runner_approvals[{index}].approved_at")
        if approved_at > now:
            raise ValidationError(f"model_runner_approvals[{index}].approved_at is in the future")
        expires_at_raw = record.get("expires_at")
        if expires_at_raw is not None:
            expires_at = _timestamp(expires_at_raw, f"model_runner_approvals[{index}].expires_at")
            if expires_at <= approved_at:
                raise ValidationError(f"model_runner_approvals[{index}].expires_at must follow approved_at")
            record["_expires_at"] = expires_at

        identity = (repository, model_id, component_id, build_profile_id, environment)
        if identity in identities:
            raise ValidationError(
                "duplicate Vulcan model runner approval for "
                f"{repository}/{model_id}/{build_profile_id}/{component_id} in {environment}"
            )
        identities.add(identity)
        validated.append(record)
    return revision, validated


def authorize_runner_matrix(
    matrix: dict[str, Any],
    vulcan_spec: dict[str, Any],
    *,
    repository: str,
    environment: str,
    now: datetime | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Authorize a compile matrix against the Terraform-published Vulcan spec."""

    if _REPOSITORY.fullmatch(repository) is None:
        raise ValidationError(f"invalid repository: {repository}")
    if not environment.strip():
        raise ValidationError("environment must not be empty")
    current = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    revision, approvals = _validated_approvals(vulcan_spec, current)

    include = matrix.get("include")
    if not isinstance(include, list):
        raise ValidationError("compile matrix include must be an array")

    authorized: list[dict[str, str]] = []
    for index, raw_entry in enumerate(include):
        if not isinstance(raw_entry, dict):
            raise ValidationError(f"compile matrix include[{index}] must be an object")
        entry = copy.deepcopy(raw_entry)
        model_id = str(entry.get("model_id") or "")
        component_id = str(entry.get("component_id") or "")
        build_profile_id = str(entry.get("build_profile_id") or "")
        requested = str(entry.get("requested_runner_class") or "default")
        if requested not in RUNNER_CLASS_RANK:
            raise ValidationError(
                f"{model_id}/{build_profile_id}/{component_id} requests unknown runner class {requested!r}"
            )

        approval_id = ""
        if requested in ELEVATED_RUNNER_CLASSES:
            matches = [
                record
                for record in approvals
                if record["repository"] == repository
                and record["model_id"] == model_id
                and record["component_id"] == component_id
                and record["build_profile_id"] == build_profile_id
                and record["environment"] == environment
            ]
            if len(matches) != 1:
                raise ValidationError(
                    f"{model_id}/{build_profile_id}/{component_id} requests {requested} but has no unique "
                    f"Vulcan approval for {repository} in {environment}"
                )
            approval = matches[0]
            expires_at = approval.get("_expires_at")
            if expires_at is not None and expires_at <= current:
                raise ValidationError(
                    f"Vulcan runner approval for {model_id}/{build_profile_id}/{component_id} expired at "
                    f"{approval['expires_at']}"
                )
            maximum = str(approval["maximum_runner_class"])
            if RUNNER_CLASS_RANK[requested] > RUNNER_CLASS_RANK[maximum]:
                raise ValidationError(
                    f"{model_id}/{build_profile_id}/{component_id} requests {requested}, "
                    f"above approved maximum {maximum}"
                )
            approval_id = f"{repository}:{model_id}:{build_profile_id}:{component_id}:{environment}"

        entry["requested_runner_class"] = requested
        entry["runner_class"] = requested
        entry["runner_approval_id"] = approval_id
        entry["runner_approval_spec_revision"] = revision
        authorized.append({str(key): str(value) for key, value in entry.items()})
    return {"include": authorized}
