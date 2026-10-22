"""Publish one complete model/profile as a retained Vulcan candidate."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .errors import ValidationError
from .model import load_models
from .package import FIDELITY_FILE, _link_or_copy, assemble_package
from .registry import (
    JsonObject,
    ReadRequest,
    RegistryPublicationError,
    Request,
    S3ArtifactStore,
    SigV4RegistryClient,
    _head_matrix,
    _preflight_compilations,
    _public_registry_json,
    canonical_sha256,
    publish_compilations,
    publish_packages,
)
from .stage_plan import read_current_catalog

DEPLOYMENTS = "/v1/repositories/sima-neat/models/deployments"


def _save(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish_branch_deployment(
    candidate_file: str | Path, *, api_url: str, read_api_url: str, aws_region: str,
    request: Request | None = None, read_request: ReadRequest | None = None,
) -> JsonObject:
    """Select a finalized candidate in its source feature catalog without approving it."""
    path = Path(candidate_file)
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"could not read candidate receipt {path}: {exc}") from exc
    if (not isinstance(candidate, dict) or candidate.get("state") != "finalized"
            or not candidate.get("deployment_id") or not candidate.get("manifest_digest")):
        raise ValidationError("branch publication requires a finalized candidate receipt")
    ref = candidate.get("git_ref", "")
    if not isinstance(ref, str) or not ref.startswith("refs/heads/") or ref in {
        "refs/heads/",
        "refs/heads/develop", "refs/heads/main", "refs/heads/master",
    }:
        raise ValidationError("unapproved deployment selection requires a source feature branch")
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    for attempt in range(8):
        record, _ = read_current_catalog(
            read_request or _public_registry_json, read_api_url, ref.removeprefix("refs/heads/"),
        )
        if record is None:
            raise ValidationError("publish the feature-branch model requirements before selecting its deployment")
        body = {
            "git_ref": ref, "deployment_id": candidate["deployment_id"],
            "manifest_digest": candidate["manifest_digest"], "expected_snapshot": record["canonical_sha256"],
        }
        try:
            result = client.request("POST", DEPLOYMENTS.rsplit("/", 1)[0] + "/branch-deployments", body,
                                    "branch-deployment-" + canonical_sha256(body))
        except RegistryPublicationError as exc:
            if "catalog_snapshot_conflict" not in str(exc) or attempt == 7:
                raise
            continue
        _save(path.parent / "branch-publication.json", result)
        return result
    raise AssertionError("unreachable")


def publish_candidate(
    *,
    repository: str | Path,
    dist_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    build_profile_id: str,
    api_url: str,
    aws_region: str,
    bucket: str,
    kms_key_id: str | None,
    environment: str,
    branch: str,
    git_sha: str,
    github_run_id: str,
    github_run_attempt: str,
    github_run_url: str,
    request: Request | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    """Join all invocations, bind their runs, and finalize without catalog promotion.

    ``dist_dir`` is a compile output tree or a directory of downloaded invocation
    artifacts. Retries of a GitHub run reuse its candidate and publication keys.
    """
    if environment != "staging":
        raise ValidationError("candidate publication is currently enabled only in staging")
    root, dist, output = (Path(path).resolve() for path in (repository, dist_dir, output_dir))
    if not dist.is_dir():
        raise ValidationError(f"compiler output directory not found: {dist}")
    if output.is_relative_to(dist) or dist.is_relative_to(output):
        raise ValidationError("candidate output must be outside the compiler input tree")
    models = load_models(root)
    model = models.get(model_id)
    if model is None or not model.get("build-on-demand", False):
        raise ValidationError("publish-candidate requires an on-demand model")
    invocations = [item for item in _head_matrix(root, models)
                   if item["model_id"] == model_id and item["build_profile_id"] == build_profile_id]
    expected = {component for item in invocations for component in item["component_ids"]}
    if not expected:
        raise ValidationError("candidate build profile does not exist")
    sources = [dist] if list(dist.glob("*/*/compile-outputs.json")) else sorted(
        path for path in dist.iterdir() if path.is_dir()
    )
    prepared = []
    for source in sources:
        prepared.extend(_preflight_compilations(root, source, models, environment=environment, git_sha=git_sha))
    actual = [item["component_id"] for item in prepared]
    if (len(actual) != len(set(actual)) or set(actual) != expected
            or any((item["model_id"], item["build_profile_id"]) != (model_id, build_profile_id)
                   for item in prepared)):
        raise ValidationError("candidate requires exactly every component of one model/profile")
    roles = {}
    for item in prepared:
        artifacts = ["llima" if item["files"] else "mpk", *(file["artifact"] for file in item["files"])]
        roles[f"compile-{item['component_id']}"] = {
            "run_type": "model_compile", "artifacts": sorted(artifacts), "inputs": ["source-component"],
        }
    packages = [item for item in model["packages"] if build_profile_id in item["build_profiles"]]
    by_component = {item["component_id"]: item for item in prepared}
    for package in packages:
        files = by_component[package["components"][0]]["files"]
        roles[f"package-{package['id']}"] = {
            "run_type": "model_package",
            "artifacts": sorted([
                "package", *(file["artifact"] for file in files),
                *(["numerical-fidelity"] if "numerical_fidelity" in model else []),
            ]),
            "inputs": sorted(f"component-{component}" for component in package["components"]),
        }
    if len(roles) > 32 or sum(len(role["artifacts"]) + len(role["inputs"]) for role in roles.values()) > 512:
        raise ValidationError("candidate exceeds Vulcan's 32-role / 512-required-name limits")
    body = {
        "git_ref": f"refs/heads/{branch}", "commit_sha": git_sha,
        "model_id": model_id, "profile_id": build_profile_id,
        "contract": {"version": 1, "roles": roles},
    }
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    identity = "candidate-" + canonical_sha256([github_run_id, environment, body])
    receipt_path = output / "candidate.json"
    candidate = client.request("POST", DEPLOYMENTS, body, f"{identity}-create")
    _save(receipt_path, candidate)
    deployment_path = f"{DEPLOYMENTS}/{candidate['deployment_id']}"
    candidate = client.request("GET", deployment_path, None, f"{identity}-inspect")
    _save(receipt_path, candidate)
    if candidate["contract_digest"] != canonical_sha256(body["contract"]):
        raise ValidationError("candidate contract digest does not match the publication contract")
    if candidate["state"] == "finalized":
        if candidate.get("availability") != "registered" or not candidate.get("protection"):
            raise ValidationError("finalized candidate is unavailable or no longer protected; start a new build run")
        return candidate
    if datetime.fromisoformat(candidate["assembly_expires_at"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise ValidationError("candidate assembly expired; start a new workflow run to publish a new candidate")
    common = dict(
        repository=root, api_url=api_url, aws_region=aws_region, bucket=bucket,
        kms_key_id=kms_key_id, environment=environment, branch=branch, git_sha=git_sha,
        github_run_id=github_run_id, github_run_attempt=github_run_attempt, github_run_url=github_run_url,
        request=request, artifact_store=artifact_store or S3ArtifactStore(
            bucket, kms_key_id=kms_key_id, receipt_dir=output / "uploads"),
        candidate=candidate,
    )
    runs = []
    for source in sources:
        publication = publish_compilations(dist_dir=source, **common)
        runs.extend(publication["runs"])
        _save(output / "publication.json", {"runs": runs})
        for run in publication["runs"]:
            path = source / model_id / build_profile_id / run["component_id"] / "registry-publication.json"
            _save(path, {"count": 1, "runs": [run]})
    # Join the component directories, and the profile's numerical fidelity the package run publishes.
    compiled = output / "compiled"
    for item in prepared:
        destination = compiled / model_id / build_profile_id / item["component_id"]
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(item["result_path"].parent, destination, copy_function=_link_or_copy)
    for source in sources:
        fidelity = source / model_id / build_profile_id / FIDELITY_FILE
        if fidelity.is_file():
            shutil.copyfile(fidelity, compiled / model_id / build_profile_id / FIDELITY_FILE)
    package_root = output / "packages"
    for package in packages:
        assemble_package(
            repository=root, compiled_dir=compiled, output_dir=package_root,
            model_id=model_id, build_profile_id=build_profile_id, package_id=package["id"],
        )
    if packages:
        publication = publish_packages(package_dir=package_root, **common)
        _save(output / "publication.json", {"runs": runs, "packages": publication["packages"]})
        for package in publication["packages"]:
            _save(package_root / model_id / build_profile_id / "packages" / package["package_id"]
                  / "registry-publication.json", {"packages": [package]})
    client.request("POST", f"{deployment_path}/finalize", {"contract_digest": candidate["contract_digest"]},
                   f"{identity}-finalize")
    candidate = client.request("GET", deployment_path, None, f"{identity}-inspect")
    _save(receipt_path, candidate)
    if (candidate.get("state") != "finalized" or not candidate.get("manifest_digest")
            or candidate.get("availability") != "registered" or not candidate.get("protection")):
        raise ValidationError("Vulcan did not confirm a finalized, available, protected candidate")
    return candidate
