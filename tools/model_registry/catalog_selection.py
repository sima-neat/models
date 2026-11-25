"""Publish a branch's complete catalog as the selection for its platform and Neat Core release.

A selection points one platform and exact Core commit at one catalog. The customer view reads the selections and
each platform's default, so publishing one changes what customers see without a site deploy. The registry accepts a
selection only for a catalog that records catalog_inputs and has a completed catalog_qualification run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .dependencies import catalog_inputs
from .errors import ValidationError
from .registry import S3ArtifactStore, SigV4RegistryClient, canonical_sha256
from .registry_http import read_registry_json_authenticated
from .report_publication import Request, put_run_file
from .stage_plan import REPOSITORY_API_PATH, ReadRequest, read_current_catalog

JsonObject = dict[str, Any]


def publish_selection(
    *,
    repository: str | Path,
    api_url: str,
    read_api_url: str,
    aws_region: str,
    bucket: str,
    kms_key_id: str | None,
    environment: str,
    branch: str,
    git_sha: str,
    github_run_url: str | None = None,
    request: Request | None = None,
    read_request: ReadRequest | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    """Publish the branch's catalog at git_sha as its platform and Core selection, and the platform's default."""
    if not api_url.strip() or not read_api_url.strip():
        raise ValidationError("publishing a selection needs the registry's write and read API URLs")
    inputs = catalog_inputs(Path(repository).expanduser().resolve())
    if inputs is None:
        raise ValidationError("a selection needs a Core release, and deps/manifest.json pins a Core branch package")
    read = read_request or read_registry_json_authenticated
    record, document = read_current_catalog(read, read_api_url, branch)
    revision = record.get("catalog_revision") if record else None
    if revision != git_sha.lower():
        raise ValidationError(f"the {branch} catalog is at {revision}, not at {git_sha}")
    if document.get("catalog_inputs") != inputs:
        raise ValidationError(f"the catalog records {document.get('catalog_inputs')}, deps/manifest.json pins {inputs}")
    if document.get("pending_builds"):
        raise ValidationError(f"the catalog still has {len(document['pending_builds'])} pending builds")

    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)
    # One key per catalog and workflow run: a retried attempt replays its writes, and a new run sends its own.
    key = f"models-selection-{record['id']}-{canonical_sha256(github_run_url or '')[:16]}"
    evidence_run_id = _publish_qualification(
        client, store, record=record, inputs=inputs, environment=environment, branch=branch,
        github_run_url=github_run_url, bucket=bucket, key=key,
    )

    filters = read(
        f"{read_api_url.rstrip('/')}{REPOSITORY_API_PATH}/models/filters?"
        f"platform_version={quote(inputs['platform_version'], safe='')}"
    )
    selections = filters.get("selections", [])
    current = next((item for item in selections if item["neat_core_revision"] == inputs["neat_core_revision"]), None)
    default = next((item for item in selections if item["is_default"]), None)
    # A rerun finds its own selection already published and only completes the default.
    if current is not None and current["catalog_id"] == record["id"]:
        selection_id = current["selection_id"]
    else:
        published = client.request("POST", f"{REPOSITORY_API_PATH}/catalog-selections", {
            "catalog_id": record["id"],
            "evidence_run_id": evidence_run_id,
            "expected_selection_id": current["selection_id"] if current else None,
        }, f"{key}-selection")
        selection_id = published["selection"]["id"]
    # Replacing the default selection moves the default with it.
    if not (current is not None and current["is_default"]):
        client.request("POST", f"{REPOSITORY_API_PATH}/catalog-selections/default", {
            "selection_id": selection_id,
            "expected_default_id": default["selection_id"] if default else None,
        }, f"{key}-default")
    return {"selection_id": selection_id, "catalog_id": record["id"], "evidence_run_id": evidence_run_id, **inputs}


def _publish_qualification(
    client: SigV4RegistryClient,
    store: S3ArtifactStore,
    *,
    record: JsonObject,
    inputs: dict[str, str],
    environment: str,
    branch: str,
    github_run_url: str | None,
    bucket: str,
    key: str,
) -> str:
    """Publish the catalog_qualification run that attests the catalog is complete and names the pinned inputs."""
    report = {
        "catalog_id": record["id"],
        "catalog_revision": record["catalog_revision"],
        "canonical_sha256": record["canonical_sha256"],
        "catalog_inputs": inputs,
        "checks": ["no pending builds", "catalog_inputs match deps/manifest.json"],
    }
    fingerprint = canonical_sha256(report)
    body: JsonObject = {
        "run_type": "catalog_qualification",
        "commit_sha": record["catalog_revision"],
        "git_ref": f"refs/heads/{branch}",
        "run_fingerprint": fingerprint,
        "environment": {
            "fingerprint": canonical_sha256(inputs),
            "properties": {"catalog_inputs": inputs, "vulcan_environment": environment},
        },
        "metadata": {"catalog_id": record["id"]},
    }
    if github_run_url:
        body.update({"ci_provider": "github-actions", "ci_run_url": github_run_url})
    run_id = client.request("POST", "/v1/runs", body, f"{key}-qualification")["run_id"]

    client.request("PUT", f"/v1/runs/{run_id}/inputs/catalog", {
        "input_type": "model-catalog",
        "source_reference": f"vulcan://catalogs/{record['id']}",
        "sha256": record["canonical_sha256"],
        "media_type": "application/json",
        "metadata": {"catalog_id": record["id"]},
    }, f"{key}-input")
    client.request("PUT", f"/v1/runs/{run_id}/parameters",
                   {"parameters": report, "canonical_sha256": fingerprint}, f"{key}-parameters")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "qualification.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = put_run_file(
            client, store, path, run_id=run_id, name="report", artifact_type="test-report",
            media_type="application/json", bucket=bucket,
            s3_key=f"models/{quote(branch, safe='')}/{record['catalog_revision'][:12]}/catalog-qualification/"
                   f"{fingerprint}/qualification.json",
            metadata={"catalog_id": record["id"]}, stable_key=key,
        )
    client.request("PUT", f"/v1/runs/{run_id}/test-results/catalog/complete", {
        "status": "passed", "summary": report, "report_artifact_id": artifact["artifact_id"],
    }, f"{key}-test-result")
    client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{key}-finalize")
    return run_id
