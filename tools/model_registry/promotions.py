"""Pinned, retry-safe promotion operations; no compilation or catalog writes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from .errors import ValidationError
from .registry import ReadRequest, SigV4RegistryClient, _public_registry_json, canonical_sha256

ROOT = "/v1/repositories/sima-neat/models"
TARGET = "refs/heads/develop"


def configuration() -> dict:
    environment = os.environ.get("VULCAN_ENV", "")
    if environment not in {"staging", "production"}:
        raise ValidationError("VULCAN_ENV must be staging or production")
    config = {"vulcan_environment": environment}
    for key, variable in (("api_url", "VULCAN_ARTIFACT_REGISTRY_API_URL"),
                          ("read_api_url", "VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL")):
        url = os.environ.get(variable, "").rstrip("/")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path):
            raise ValidationError(f"{variable} must be an HTTPS API origin")
        config[key] = url
    return config


def check_configuration(record: dict, config: dict) -> None:
    if any(record.get(key) != value for key, value in config.items()):
        raise ValidationError("Saved promotion environment/URL differs; use its original configuration")


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Cannot read promotion record {path}: {exc}") from exc


def identifier(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValidationError("A valid candidate/promotion UUID is required") from exc


def inspect(client, promotion_id: str) -> dict:
    result = client.request("GET", f"{ROOT}/promotions/{identifier(promotion_id)}", None, "inspect")
    if result.get("target_git_ref") != TARGET:
        raise ValidationError("Only Develop promotion is supported")
    return result


def prepare(client, *, deployment_id: str, reason: str, workflow: dict, identity: str,
            read_api_url: str, read_request: ReadRequest | None = None,
            expected_model_id: str | None = None, expected_profile_id: str | None = None) -> dict:
    if not reason.strip() or len(reason) > 2000:
        raise ValidationError("Promotion reason must contain 1–2000 characters")
    candidate = client.request("GET", f"{ROOT}/deployments/{identifier(deployment_id)}", None, "inspect")
    if candidate.get("state") != "finalized" or not candidate.get("manifest_digest"):
        raise ValidationError("Promotion requires a finalized candidate with an immutable digest")
    if expected_model_id is not None and (
        candidate.get("model_id"), candidate.get("profile_id")
    ) != (expected_model_id, expected_profile_id):
        raise ValidationError("Vulcan candidate does not match the requested model/profile")
    # Catalogs are public-reader routes, not publisher IAM routes. Never send AWS credentials here.
    target = (read_request or _public_registry_json)(
        f"{read_api_url.rstrip('/')}{ROOT}/catalogs/current?git_ref=refs%2Fheads%2Fdevelop"
    )
    body = {
        "deployment_id": identifier(deployment_id), "manifest_digest": candidate["manifest_digest"],
        "target_git_ref": TARGET, "expected_target_snapshot": target["catalog"]["canonical_sha256"],
        "reason": reason, "workflow": workflow,
    }
    return {"body": body, "key": "promotion-request-" + canonical_sha256([identity, body])}


def check_request(record: dict, body: dict) -> None:
    for field in ("deployment_id", "manifest_digest", "target_git_ref",
                  "expected_target_snapshot", "reason", "workflow"):
        if record.get(field) != body.get(field):
            raise ValidationError(f"Stored promotion differs from the saved request: {field}")


def request(client, plan: dict) -> dict:
    body = plan["body"]
    if body.get("target_git_ref") != TARGET:
        raise ValidationError("Only Develop promotion is supported")
    receipt = client.request("POST", f"{ROOT}/promotions", body, plan["key"])
    # Idempotent POST receipts are historical; inspect authoritative current state.
    record = inspect(client, receipt["promotion_id"])
    check_request(record, body)
    if record["state"] not in {"pending", "promoted"}:
        raise ValidationError(
            f"Promotion {record['promotion_id']} is {record['state']}; start a new request and review"
        )
    if record["state"] == "pending" and not record.get("pending_protection"):
        raise ValidationError("Vulcan did not confirm pending protection")
    return record


def decide(client, record: dict, *, operation: str, reason: str, identity: str) -> dict:
    if operation not in {"approve", "reject", "cancel"}:
        raise ValidationError("Unknown promotion decision")
    if not reason.strip() or len(reason) > 2000:
        raise ValidationError("Decision reason must contain 1–2000 characters")
    promotion_id = identifier(record["promotion_id"])
    current = inspect(client, promotion_id)
    check_request(current, record)
    for field in ("selection", "previous_selection", "source"):
        if current.get(field) != record.get(field):
            raise ValidationError(f"Promotion review content changed: {field}")
    expected = {"approve": "promoted", "reject": "rejected", "cancel": "cancelled"}[operation]
    if current["state"] not in {"pending", expected}:
        raise ValidationError(f"Promotion is {current['state']}; cannot {operation}")
    body = {"reason": reason}
    key = "promotion-decision-" + canonical_sha256([identity, promotion_id, operation, body])
    # Inspect after an uncertain response; a still-pending decision reuses the exact key/body.
    if current["state"] == expected:
        if operation == "approve" and not current.get("result_snapshot"):
            raise ValidationError("Promoted request has no result snapshot; inspect backend state")
        return current
    receipt = client.request("POST", f"{ROOT}/promotions/{promotion_id}/{operation}", body, key)
    if receipt.get("state") != expected or (operation == "approve" and not receipt.get("result_snapshot")):
        raise ValidationError("Vulcan did not confirm the requested decision")
    return receipt


def summary(record: dict) -> str:
    # Fenced JSON prevents candidate/user strings from forging review headings or links.
    payload = json.dumps(record, indent=2, sort_keys=True).replace("`", "\\u0060").replace("<", "\\u003c")
    return (f"## Develop promotion (Vulcan {record.get('vulcan_environment', 'unspecified')})\n\n"
            "Review the exact candidate, source, previous/proposed selections, reason, evidence, and expiry below. "
            "GitHub approval alone does not publish a selection.\n\n```json\n" + payload + "\n```\n\n"
            "If a GitHub job is rejected or cancelled, the backend may remain pending until explicit cancellation "
            "or its seven-day expiry. A changed target requires a new request and review.\n")


def add_parser(subcommands) -> None:
    parser = subcommands.add_parser("promotion", help="manage pinned Develop promotion requests (CI only)")
    parser.add_argument("action", choices=("prepare", "request", "inspect", "approve", "reject", "cancel"))
    parser.add_argument("--record", type=Path, required=True, help="saved plan/request; inspect and prepare write it")
    parser.add_argument("--output", type=Path, help="write the API receipt separately from the immutable input")
    parser.add_argument("--deployment-id")
    parser.add_argument("--promotion-id")
    parser.add_argument("--reason", default="")
    parser.add_argument("--identity", default="", help="stable operation identity, unchanged across retries")
    parser.add_argument("--run-url")
    parser.add_argument("--issue-url")
    parser.add_argument("--aws-region", default="us-west-2")
    parser.add_argument("--summary", type=Path)


def run(args: argparse.Namespace) -> int:
    config = configuration()
    client = SigV4RegistryClient(config["api_url"], args.aws_region)
    if args.action not in {"prepare", "inspect"}:
        check_configuration(read(args.record), config)
    if args.action == "inspect":
        result = inspect(client, args.promotion_id)
    elif args.action == "prepare":
        if not args.identity or not args.run_url:
            raise ValidationError("prepare requires --identity and --run-url")
        if args.record.exists():
            raise ValidationError("Request plan already exists; reuse it rather than refreshing its target")
        workflow = {"run_id": args.identity, "run_url": args.run_url}
        if args.issue_url:
            workflow["issue_url"] = args.issue_url
        result = prepare(client, deployment_id=args.deployment_id, reason=args.reason,
                         workflow=workflow, identity=args.identity, read_api_url=config["read_api_url"])
    elif args.action == "request":
        result = request(client, read(args.record))
    else:
        if not args.identity:
            raise ValidationError("decisions require a stable --identity")
        result = decide(client, read(args.record), operation=args.action, reason=args.reason, identity=args.identity)
    result = {**result, **config}
    if args.action in {"prepare", "inspect"}:
        save(args.record, result)
    elif args.output:
        save(args.output, result)
    if args.summary:
        with args.summary.open("a") as stream:
            stream.write(summary(result))
    print(json.dumps(result, sort_keys=True))
    return 0
