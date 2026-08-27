from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .catalog import discover_catalog
from .errors import ModelRegistryError, ValidationError
from .fetch import sha256_file
from .taxonomy import (
    TAXONOMY_RELATIVE_PATH,
    canonical_taxonomy_document,
    load_taxonomy,
    taxonomy_sha256,
    validate_model_categorization,
)

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class RegistryPublicationError(ModelRegistryError):
    """A model compilation could not be published to the artifact registry."""


class SigV4RegistryClient:
    def __init__(self, api_url: str, region: str, *, request: Request | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.region = region
        self._request = request or self._curl_request

    def request(self, method: str, path: str, body: JsonObject | None, idempotency_key: str) -> JsonObject:
        try:
            return self._request(method, path, body, idempotency_key)
        except RegistryPublicationError as exc:
            raise RegistryPublicationError(f"{method} {path}: {exc}") from exc

    def _curl_request(self, method: str, path: str, body: JsonObject | None, idempotency_key: str) -> JsonObject:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        session_token = os.environ.get("AWS_SESSION_TOKEN", "")
        if not access_key or not secret_key or not session_token:
            raise RegistryPublicationError("AWS session credentials are required for registry publication")

        command = [
            "curl",
            "--silent",
            "--show-error",
            "--aws-sigv4",
            f"aws:amz:{self.region}:execute-api",
            "--user",
            f"{access_key}:{secret_key}",
            "--header",
            f"x-amz-security-token: {session_token}",
            "--header",
            "content-type: application/json",
            "--header",
            f"idempotency-key: {idempotency_key}",
            "--request",
            method,
            "--write-out",
            "\n%{http_code}",
            f"{self.api_url}{path}",
        ]
        payload = None
        if body is not None:
            command.extend(["--data-binary", "@-"])
            payload = json.dumps(body, sort_keys=True, separators=(",", ":"))

        for attempt in range(1, 6):
            completed = subprocess.run(command, input=payload, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                if attempt == 5:
                    raise RegistryPublicationError(
                        f"artifact registry request failed before receiving HTTP: {completed.stderr.strip()}"
                    )
                time.sleep(2 ** (attempt - 1))
                continue
            response_body, separator, status_text = completed.stdout.rpartition("\n")
            if not separator or not status_text.isdigit():
                raise RegistryPublicationError("artifact registry returned an invalid HTTP response")
            status = int(status_text)
            try:
                decoded = json.loads(response_body) if response_body else {}
            except json.JSONDecodeError as exc:
                raise RegistryPublicationError(f"artifact registry returned non-JSON HTTP {status}") from exc
            if 200 <= status < 300:
                if not isinstance(decoded, dict):
                    raise RegistryPublicationError("artifact registry response must be a JSON object")
                return decoded
            if status in {429, 502, 503, 504} and attempt < 5:
                time.sleep(2 ** (attempt - 1))
                continue
            error = decoded.get("error", {}) if isinstance(decoded, dict) else {}
            code = error.get("code", "unknown_error")
            message = error.get("message", "request rejected")
            raise RegistryPublicationError(f"artifact registry HTTP {status} {code}: {message}")
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class S3Object:
    key: str
    version_id: str
    sha256: str
    size_bytes: int


class S3ArtifactStore:
    def __init__(self, bucket: str, *, kms_key_id: str | None = None) -> None:
        self.bucket = bucket
        self.kms_key_id = (kms_key_id or "").strip()

    def ensure_native_sha256(
        self,
        path: Path,
        key: str,
        sha256: str,
        *,
        media_type: str = "application/gzip",
    ) -> S3Object:
        expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        head = self._head(key)
        if (
            head.get("ChecksumSHA256") == expected_checksum
            and head.get("ContentLength") == path.stat().st_size
            and isinstance(head.get("VersionId"), str)
            and head["VersionId"]
        ):
            return S3Object(key, str(head["VersionId"]), sha256, path.stat().st_size)

        if path.stat().st_size > 5 * 1024**3:
            raise RegistryPublicationError(f"registry artifact exceeds the 5 GiB single-upload limit: {path}")
        command = [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            self.bucket,
            "--key",
            key,
            "--body",
            str(path),
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            expected_checksum,
            "--server-side-encryption",
            "aws:kms",
            "--content-type",
            media_type,
            "--output",
            "json",
            "--no-cli-pager",
        ]
        if self.kms_key_id:
            command.extend(["--ssekms-key-id", self.kms_key_id])
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RegistryPublicationError(f"Could not upload s3://{self.bucket}/{key}: {completed.stderr.strip()}")
        try:
            uploaded = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RegistryPublicationError(f"Could not parse S3 upload response for {key}") from exc
        version_id = uploaded.get("VersionId") if isinstance(uploaded, dict) else None
        if not isinstance(version_id, str) or not version_id:
            raise RegistryPublicationError(f"S3 versioning did not return a version for {key}")
        verified = self._head(key, version_id)
        if verified.get("ChecksumSHA256") != expected_checksum:
            raise RegistryPublicationError(f"S3 native SHA-256 verification failed for {key}")
        if verified.get("ContentLength") != path.stat().st_size:
            raise RegistryPublicationError(f"S3 object size does not match the packaged artifact for {key}")
        return S3Object(key, version_id, sha256, path.stat().st_size)

    def _head(self, key: str, version_id: str | None = None) -> JsonObject:
        command = [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            self.bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            "--output",
            "json",
        ]
        if version_id:
            command.extend(["--version-id", version_id])
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            return {}
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RegistryPublicationError(f"S3 returned invalid metadata for {key}") from exc
        return value if isinstance(value, dict) else {}


def _publish_taxonomy(client: SigV4RegistryClient, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    taxonomy = load_taxonomy(root / TAXONOMY_RELATIVE_PATH)
    taxonomy_document = canonical_taxonomy_document(taxonomy)
    taxonomy_digest = taxonomy_sha256(taxonomy_document)
    taxonomy_version = taxonomy_document["version"]
    response_document = client.request(
        "PUT",
        f"/v1/repositories/sima-neat/models/taxonomies/{taxonomy_version}",
        {
            "schema_version": 1,
            "canonical_sha256": taxonomy_digest,
            "taxonomy": taxonomy_document,
        },
        f"models-taxonomy-{taxonomy_version}-{taxonomy_digest}",
    )
    return taxonomy, {
        "version": taxonomy_version,
        "sha256": taxonomy_digest,
        "registry": response_document,
    }


def publish_taxonomy(
    *,
    repository: str | Path,
    api_url: str,
    aws_region: str,
    request: Request | None = None,
) -> JsonObject:
    root = Path(repository).expanduser().resolve()
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    _, result = _publish_taxonomy(client, root)
    return result


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"compile result {field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_text(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValidationError(f"compile result {field} must be a lowercase SHA-256 digest")
    return digest


def publish_compilations(
    *,
    repository: str | Path,
    dist_dir: str | Path,
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
    root = Path(repository).expanduser().resolve()
    dist = Path(dist_dir).expanduser().resolve()
    if not dist.is_dir():
        raise ValidationError(f"packaged model directory not found: {dist}")
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip() or not github_run_id.strip() or not github_run_attempt.strip():
        raise ValidationError("branch, GitHub run ID, and run attempt must not be empty")

    catalog = discover_catalog(root)
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)
    taxonomy, taxonomy_result = _publish_taxonomy(client, root)
    branch_key = quote(branch, safe="")
    commit_tag = git_sha[:12]
    published: list[JsonObject] = []

    for result_path in sorted(dist.glob("*/*/compile-result.json")):
        model_id, variant_id = result_path.relative_to(dist).parts[:2]
        if model_id not in catalog:
            raise ValidationError(f"compiled output references unknown model: {model_id}")
        variants = {item["id"]: item for item in catalog[model_id]["compile_variants"]}
        if variant_id not in variants:
            raise ValidationError(f"compiled output references unknown variant: {model_id}/{variant_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = result.get("provenance")
        if not isinstance(provenance, dict):
            raise ValidationError(f"{result_path}: provenance must be an object")
        expected = {"model_id": model_id, "variant_id": variant_id, "git_sha": git_sha, "environment": environment}
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise ValidationError(f"{result_path}: provenance {field} must be {value!r}")

        archive_name = _require_text(result.get("archive"), "archive")
        archive = result_path.parent / archive_name
        if not archive.is_file() or archive.parent != result_path.parent:
            raise ValidationError(f"{result_path}: archive is missing from the packaged output")
        artifact_sha = _require_sha256(result.get("sha256"), "sha256")
        actual_sha, actual_size = sha256_file(archive)
        if artifact_sha != actual_sha:
            raise ValidationError(f"{result_path}: sha256 does not match the packaged archive")
        if int(result.get("size_bytes", -1)) != actual_size:
            raise ValidationError(f"{result_path}: size_bytes does not match the packaged archive")

        compiler_version = _require_text(provenance.get("compiler_version"), "compiler_version")
        compiler_label = _require_text(provenance.get("compiler_label"), "compiler_label")
        target = _require_text(provenance.get("target"), "target")
        source_sha = _require_sha256(provenance.get("source_sha256"), "source_sha256")
        build_fingerprint = _require_sha256(provenance.get("build_fingerprint"), "build_fingerprint")
        model = catalog[model_id]
        categorization = validate_model_categorization(model, taxonomy)
        variant = variants[variant_id]
        parameters = {
            "schema_version": 1,
            "model_id": model_id,
            "variant_id": variant_id,
            "target": target,
            "compiler_version": compiler_version,
            "compiler_label": compiler_label,
            "source_sha256": source_sha,
            "build_fingerprint": build_fingerprint,
            "recipe": variant["recipe"],
            "compile_parameters": variant.get("parameters", {}),
        }
        environment_document = {
            "schema_version": 1,
            "vulcan_environment": environment,
            "toolchain_name": "sima-model-compiler",
            "toolchain_version": compiler_version,
            "compiler_label": compiler_label,
            "target_platform": target,
        }
        run_fingerprint = canonical_sha256({**parameters, "artifact_sha256": artifact_sha})
        environment_fingerprint = canonical_sha256(environment_document)
        # A new attempt of the same workflow run resumes the same registry
        # transaction instead of creating a duplicate record per attempt.
        key_digest = hashlib.sha256(
            f"{github_run_id}:{environment}:{branch}:{git_sha}:{model_id}:{variant_id}:{run_fingerprint}".encode()
        ).hexdigest()
        stable_key = f"models-{key_digest}"
        create = client.request(
            "POST",
            "/v1/runs",
            {
                "schema_version": 1,
                "run_type": "model_compile",
                "commit_sha": git_sha,
                "git_ref": f"refs/heads/{branch}",
                "ci_provider": "github-actions",
                "ci_run_url": github_run_url,
                "run_fingerprint": run_fingerprint,
                "environment": {
                    "fingerprint": environment_fingerprint,
                    "fingerprint_version": 1,
                    "toolchain_name": "sima-model-compiler",
                    "toolchain_version": compiler_version,
                    "target_platform": target,
                    "properties": {"compiler_label": compiler_label, "vulcan_environment": environment},
                },
                "metadata": {
                    "model_id": model_id,
                    "variant_id": variant_id,
                    "artifact_kind": "mpk",
                    "github_run_id": github_run_id,
                },
                **({"model_categories": categorization} if categorization is not None else {}),
            },
            f"{stable_key}-create",
        )
        run_id = _require_text(create.get("run_id"), "run_id")
        source_metadata = {"model_id": model_id}
        model_card = model.get("upstream", {}).get("model_card")
        if isinstance(model_card, str) and model_card:
            source_metadata["model_card"] = model_card
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/inputs/source-model",
            {
                "input_type": "onnx",
                "source_reference": model["source"]["uri"],
                "sha256": source_sha,
                "size_bytes": model["source"]["size_bytes"],
                "media_type": model["source"].get("media_type", "application/onnx"),
                "metadata": source_metadata,
            },
            f"{stable_key}-input",
        )
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/parameters",
            {"parameters": parameters, "canonical_sha256": canonical_sha256(parameters), "canonicalization_version": 1},
            f"{stable_key}-parameters",
        )

        relative_archive = archive.relative_to(dist).as_posix()
        object_key = f"models/{branch_key}/{commit_tag}/{relative_archive}"
        stored = store.ensure_native_sha256(archive, object_key, artifact_sha)
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/artifacts/mpk",
            {
                "artifact_type": "model-pack",
                "s3_bucket": bucket,
                "s3_key": stored.key,
                "s3_version_id": stored.version_id,
                "sha256": stored.sha256,
                "size_bytes": stored.size_bytes,
                "media_type": "application/gzip",
                "retention_class": "branch",
                "metadata": {"model_id": model_id, "variant_id": variant_id, "target": target},
            },
            f"{stable_key}-artifact",
        )
        # This is compiler-output validation, not a DevKit correctness or performance test.
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/test-results/compile/mpk-validation",
            {
                "status": "passed",
                "version": "1",
                "summary": {"archive_validated": True, "checksum_verified": True},
            },
            f"{stable_key}-validation",
        )
        client.request(
            "POST",
            f"/v1/runs/{run_id}/finalize",
            {"status": "completed"},
            f"{stable_key}-finalize",
        )
        published.append({"model_id": model_id, "variant_id": variant_id, "run_id": run_id, "s3_key": stored.key})

    return {
        "count": len(published),
        "runs": published,
        "taxonomy": {"version": taxonomy_result["version"], "sha256": taxonomy_result["sha256"]},
    }
