"""Pinned promotion operations and benchmark reference publication; no compilation or benchmarking."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from .errors import ValidationError
from .registry import CATALOG_CONFLICT_RETRY_SECONDS, _refresh_approved_benchmarks, read_current_catalog
from .registry_api.runs import REPOSITORY_PATH
from .registry_api.transport import (
    ReadRequest,
    RegistryRequestError,
    SigV4RegistryClient,
    canonical_sha256,
    is_catalog_conflict,
    read_registry_json,
)

TARGET = "refs/heads/develop"


def configuration(target_branch: str | None = None) -> dict:
    environment = os.environ.get("VULCAN_ENV", "")
    if environment not in {"staging", "production"}:
        raise ValidationError("VULCAN_ENV must be staging or production")
    target_branch = target_branch or os.environ.get("PROMOTION_TARGET_BRANCH", "develop")
    if target_branch not in {"develop", "main"}:
        raise ValidationError("Promotion target must be develop or main")
    if target_branch == "main" and environment != "production":
        raise ValidationError("Main promotion is available only in production")
    config = {"vulcan_environment": environment, "target_branch": target_branch}
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
    # Legacy plans had only Develop as a target. Never reinterpret one as Main.
    saved = dict(record)
    if "target_branch" not in saved:
        ref = (record.get("target_git_ref") or record.get("body", {}).get("target_git_ref")
               or record.get("record", {}).get("target_git_ref") or TARGET)
        saved["target_branch"] = ref.removeprefix("refs/heads/")
    if any(saved.get(key) != value for key, value in config.items()):
        raise ValidationError("Saved promotion environment/URL/target differs; use its original configuration")


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


def inspect(client, promotion_id: str, *, target_branch: str | None = None) -> dict:
    target = "refs/heads/" + configuration(target_branch)["target_branch"]
    result = client.request("GET", f"{REPOSITORY_PATH}/promotions/{identifier(promotion_id)}", None, "inspect")
    if result.get("target_git_ref") != target:
        raise ValidationError("Promotion target differs from the selected target")
    return result


def prepare(client, *, deployment_id: str, reason: str, workflow: dict, identity: str,
            read_api_url: str, read_request: ReadRequest | None = None,
            expected_model_id: str | None = None, expected_profile_id: str | None = None,
            target_branch: str | None = None) -> dict:
    target_ref = "refs/heads/" + configuration(target_branch)["target_branch"]
    if not reason.strip() or len(reason) > 2000:
        raise ValidationError("Promotion reason must contain 1–2000 characters")
    candidate = client.request("GET", f"{REPOSITORY_PATH}/deployments/{identifier(deployment_id)}", None, "inspect")
    if candidate.get("state") != "finalized" or not candidate.get("manifest_digest"):
        raise ValidationError("Promotion requires a finalized candidate with an immutable digest")
    if expected_model_id is not None and (
        candidate.get("model_id"), candidate.get("profile_id")
    ) != (expected_model_id, expected_profile_id):
        raise ValidationError("Vulcan candidate does not match the requested model/profile")
    # Catalogs are public-reader routes, not publisher IAM routes. Never send AWS credentials here.
    target_name = target_ref.removeprefix("refs/heads/")
    target = (read_request or read_registry_json)(
        f"{read_api_url.rstrip('/')}{REPOSITORY_PATH}/catalogs/current?git_ref=refs%2Fheads%2F{target_name}"
    )
    body = {
        "deployment_id": identifier(deployment_id), "manifest_digest": candidate["manifest_digest"],
        "target_git_ref": target_ref, "expected_target_snapshot": target["catalog"]["canonical_sha256"],
        "reason": reason, "workflow": workflow,
    }
    return {"body": body, "key": "promotion-request-" + canonical_sha256([identity, body])}


def check_request(record: dict, body: dict) -> None:
    for field in ("deployment_id", "manifest_digest", "target_git_ref",
                  "expected_target_snapshot", "reason", "workflow"):
        if record.get(field) != body.get(field):
            raise ValidationError(f"Stored promotion differs from the saved request: {field}")


def request(client, plan: dict, *, target_branch: str | None = None) -> dict:
    body = plan["body"]
    target = "refs/heads/" + configuration(target_branch)["target_branch"]
    if body.get("target_git_ref") != target:
        raise ValidationError("Saved promotion target differs from the selected target")
    receipt = client.request("POST", f"{REPOSITORY_PATH}/promotions", body, plan["key"])
    # Idempotent POST receipts are historical; inspect authoritative current state.
    record = inspect(client, receipt["promotion_id"], target_branch=target_branch)
    check_request(record, body)
    if record["state"] not in {"pending", "promoted"}:
        raise ValidationError(
            f"Promotion {record['promotion_id']} is {record['state']}; start a new request and review"
        )
    if record["state"] == "pending" and not record.get("pending_protection"):
        raise ValidationError("Vulcan did not confirm pending protection")
    return record


def decide(client, record: dict, *, operation: str, reason: str, identity: str,
           target_branch: str | None = None) -> dict:
    if operation not in {"approve", "reject", "cancel"}:
        raise ValidationError("Unknown promotion decision")
    if not reason.strip() or len(reason) > 2000:
        raise ValidationError("Decision reason must contain 1–2000 characters")
    promotion_id = identifier(record["promotion_id"])
    current = inspect(client, promotion_id, target_branch=target_branch)
    check_request(current, record)
    for field in ("selection", "previous_selection", "source", "source_promotion_id", "source_approval"):
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
    receipt = client.request("POST", f"{REPOSITORY_PATH}/promotions/{promotion_id}/{operation}", body, key)
    if receipt.get("state") != expected or (operation == "approve" and not receipt.get("result_snapshot")):
        raise ValidationError("Vulcan did not confirm the requested decision")
    return receipt


def withdraw(client, record: dict, *, reason: str, identity: str) -> dict:
    """Take a promoted build out of its target catalog. Vulcan keeps the request `promoted`."""
    promotion_id = identifier(record["promotion_id"])
    body = {"reason": reason}
    # A retry sends the same key and body, so Vulcan replays the first receipt.
    key = "promotion-withdrawal-" + canonical_sha256([identity, promotion_id, body])
    receipt = client.request("POST", f"{REPOSITORY_PATH}/promotions/{promotion_id}/withdraw", body, key)
    if not receipt.get("result_snapshot"):
        raise ValidationError("Vulcan did not confirm the withdrawal")
    return receipt


def benchmark_catalog(record: dict, *, read_api_url: str, read_request: ReadRequest | None = None) -> dict | None:
    """Return the target source revision only while this deployment is still selected."""
    if record.get("state") != "promoted" or not record.get("result_snapshot"):
        return None
    branch = record["target_git_ref"].removeprefix("refs/heads/")
    catalog, document = read_current_catalog(read_request or read_registry_json, read_api_url, branch)
    if not catalog or not document:
        raise ValidationError("Promotion target catalog is missing")
    model = next((item for item in document["models"] if item["id"] == record["source"]["model_id"]), {})
    selected = model.get("approved_deployments", {}).get(record["source"]["profile_id"], {})
    if selected.get("deployment_id") != record["deployment_id"]:
        return None  # A later promotion superseded this request; never restore its references.
    return catalog


def refresh_benchmarks(
    client, record: dict, *, repository: Path, revision: str, read_api_url: str,
    read_request: ReadRequest | None = None,
) -> dict:
    """Publish current benchmark identities after approval, using the target revision's source files."""
    read = read_request or read_registry_json
    deadline = time.monotonic() + CATALOG_CONFLICT_RETRY_SECONDS
    while True:
        catalog = benchmark_catalog(record, read_api_url=read_api_url, read_request=read)
        if catalog is None:
            return {"updated": False}
        if catalog["catalog_revision"] != revision:
            raise ValidationError(
                "Promotion target revision changed; rerun the report job to refresh benchmark references"
            )
        document = copy.deepcopy(catalog["document"])
        document["schema_version"] = 2
        document.setdefault("pending_builds", [])
        _refresh_approved_benchmarks(
            repository, document, catalog["document"], read_api_url, read,
            model_id=record["source"]["model_id"], profile_id=record["source"]["profile_id"],
        )
        if document == catalog["document"]:
            return {"updated": False}
        body = {"initialize": False, "previous_snapshot": catalog["canonical_sha256"], "catalog": document}
        try:
            result = client.request(
                "POST", f"{REPOSITORY_PATH}/catalogs/{revision}/snapshots", body,
                "promotion-benchmarks-" + canonical_sha256([record["promotion_id"], body]),
            )
        except RegistryRequestError as exc:
            # Another promotion replaced the catalog between the read and this write.
            if not is_catalog_conflict(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(random.uniform(0.2, 2))
            continue
        return {"updated": True, "registry": result}


def summary(record: dict) -> str:
    # Fenced JSON prevents candidate/user strings from forging review headings or links.
    payload = json.dumps(record, indent=2, sort_keys=True).replace("`", "\\u0060").replace("<", "\\u003c")
    is_main = record.get("target_branch") == "main" or record.get("target_git_ref") == "refs/heads/main"
    label = "Main" if is_main else "Develop"
    return (f"## {label} promotion (Vulcan {record.get('vulcan_environment', 'unspecified')})\n\n"
            "Review the exact candidate, source, previous/proposed selections, reason, evidence, and expiry below. "
            "GitHub approval alone does not publish a selection.\n\n```json\n" + payload + "\n```\n\n"
            "If a GitHub job is rejected or cancelled, the backend may remain pending until explicit cancellation "
            "or its seven-day expiry. A changed target requires a new request and review.\n")


def add_parser(subcommands) -> None:
    parser = subcommands.add_parser(
        "promotion", help="manage pinned Develop or production Main promotion requests (CI only)")
    parser.add_argument("action", choices=("prepare", "request", "inspect", "approve", "reject", "cancel"))
    parser.add_argument("--record", type=Path, required=True, help="saved plan/request; inspect and prepare write it")
    parser.add_argument("--output", type=Path, help="write the API receipt separately from the immutable input")
    parser.add_argument("--target-branch", choices=("develop", "main"),
                        help="target catalog; defaults to PROMOTION_TARGET_BRANCH or develop; Main is production-only")
    parser.add_argument("--deployment-id")
    parser.add_argument("--promotion-id")
    parser.add_argument("--reason", default="")
    parser.add_argument("--identity", default="", help="stable operation identity, unchanged across retries")
    parser.add_argument("--run-url")
    parser.add_argument("--issue-url")
    parser.add_argument("--aws-region", default="us-west-2")
    parser.add_argument("--summary", type=Path)


def run(args: argparse.Namespace) -> int:
    config = configuration(getattr(args, "target_branch", None))
    client = SigV4RegistryClient(config["api_url"], args.aws_region)
    if args.action not in {"prepare", "inspect"}:
        check_configuration(read(args.record), config)
    if args.action == "inspect":
        result = inspect(client, args.promotion_id, target_branch=config["target_branch"])
    elif args.action == "prepare":
        if not args.identity or not args.run_url:
            raise ValidationError("prepare requires --identity and --run-url")
        if args.record.exists():
            raise ValidationError("Request plan already exists; reuse it rather than refreshing its target")
        workflow = {"run_id": args.identity, "run_url": args.run_url}
        if args.issue_url:
            workflow["issue_url"] = args.issue_url
        result = prepare(client, deployment_id=args.deployment_id, reason=args.reason,
                         workflow=workflow, identity=args.identity, read_api_url=config["read_api_url"],
                         target_branch=config["target_branch"])
    elif args.action == "request":
        result = request(client, read(args.record), target_branch=config["target_branch"])
    else:
        if not args.identity:
            raise ValidationError("decisions require a stable --identity")
        result = decide(client, read(args.record), operation=args.action, reason=args.reason, identity=args.identity,
                        target_branch=config["target_branch"])
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
