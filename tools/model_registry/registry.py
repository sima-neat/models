from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from jsonschema import Draft202012Validator

from .accuracy_config import find_devkit_accuracy_config
from .artifact import validate_compile_outputs, verify_mpk
from .compile_matrix import build_matrix
from .dependencies import resolve_model_compiler
from .documentation import load_model_documentation
from .errors import ModelRegistryError, ValidationError
from .fetch import sha256_file
from .fidelity_config import find_fidelity_config, validate_fidelity_result
from .model import compile_invocations_for, load_and_validate_model, load_models, profile_component_for
from .source import source_identity
from .stage_keys import accuracy_key, compile_key, performance_key
from .stage_plan import read_current_catalog
from .taxonomy import (
    TAXONOMY_RELATIVE_PATH,
    canonical_taxonomy_document,
    load_taxonomy,
    taxonomy_sha256,
    validate_model_categorization,
)

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]
ReadRequest = Callable[[str], JsonObject]
DownloadRequest = Callable[[str], bytes]


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


def _publication_receipts(compiled_dir: Path | None) -> dict[tuple[str, str, str], JsonObject]:
    receipts: dict[tuple[str, str, str], JsonObject] = {}
    if compiled_dir is None or not compiled_dir.exists():
        return receipts
    for path in sorted(compiled_dir.glob("*/*/*/registry-publication.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            runs = document["runs"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ValidationError(f"could not read registry publication receipt {path}: {exc}") from exc
        if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
            raise ValidationError(f"{path}: registry publication receipt must contain exactly one run")
        run = runs[0]
        key = tuple(str(run.get(field, "")) for field in ("model_id", "build_profile_id", "component_id"))
        if not all(key) or key in receipts:
            raise ValidationError(f"{path}: registry publication receipt has an invalid or duplicate build unit")
        receipts[key] = run
    return receipts


def _package_publication_receipts(package_dir: Path | None) -> dict[tuple[str, str, str], JsonObject]:
    """Read durable package receipts written after a package run is finalized."""
    receipts: dict[tuple[str, str, str], JsonObject] = {}
    if package_dir is None or not package_dir.exists():
        return receipts
    for path in sorted(package_dir.glob("*/*/packages/*/registry-publication.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            packages = document["packages"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ValidationError(f"could not read package publication receipt {path}: {exc}") from exc
        if not isinstance(packages, list) or len(packages) != 1 or not isinstance(packages[0], dict):
            raise ValidationError(f"{path}: package publication receipt must contain exactly one package")
        package = packages[0]
        key = tuple(str(package.get(field, "")) for field in ("model_id", "build_profile_id", "package_id"))
        if not all(key) or key in receipts:
            raise ValidationError(f"{path}: package publication receipt has an invalid or duplicate package")
        receipts[key] = package
    return receipts


def _selected_invocation_keys(selected_matrix: Path | None) -> set[tuple[str, str, str]]:
    if selected_matrix is None:
        return set()
    try:
        document = json.loads(selected_matrix.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read selected build matrix {selected_matrix}: {exc}") from exc
    include = document.get("include") if isinstance(document, dict) else None
    if not isinstance(include, list):
        raise ValidationError(f"{selected_matrix}: selected build matrix must contain an include list")
    selected: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(include):
        if not isinstance(entry, dict):
            raise ValidationError(f"{selected_matrix}: selected build matrix entry {index} must be an object")
        key = tuple(str(entry.get(field, "")) for field in (
            "model_id", "build_profile_id", "compile_invocation_id",
        ))
        if not all(key):
            raise ValidationError(
                f"{selected_matrix}: selected build matrix entry {index} must identify a compile invocation"
            )
        if key in selected:
            raise ValidationError(f"{selected_matrix}: selected build matrix contains duplicate invocation {key}")
        selected.add(key)
    return selected


def build_catalog_snapshot(
    *,
    repository: str | Path,
    branch: str,
    git_sha: str,
    compiled_dir: str | Path | None = None,
    package_dir: str | Path | None = None,
    selected_matrix: str | Path | None = None,
    previous: JsonObject | None = None,
    incremental: bool = False,
    documentation_receipts: dict[str, JsonObject] | None = None,
) -> JsonObject:
    root = Path(repository).expanduser().resolve()
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip():
        raise ValidationError("branch must not be empty")
    models = load_models(root)
    taxonomy = canonical_taxonomy_document(load_taxonomy(root / TAXONOMY_RELATIVE_PATH))
    taxonomy_digest = taxonomy_sha256(taxonomy)
    compiler_version = resolve_model_compiler(root)["identity"]
    matrix = build_matrix(
        models,
        sorted(models),
        repository=root,
        compiler_version=compiler_version,
    )["include"]
    receipts = _publication_receipts(Path(compiled_dir).expanduser().resolve() if compiled_dir else None)
    package_receipts = _package_publication_receipts(
        Path(package_dir).expanduser().resolve() if package_dir else None
    )
    selected_invocations = _selected_invocation_keys(
        Path(selected_matrix).expanduser().resolve() if selected_matrix else None
    )
    known_invocations = {
        (entry["model_id"], entry["build_profile_id"], entry["compile_invocation_id"])
        for entry in matrix
    }
    unknown_invocations = sorted(selected_invocations - known_invocations)
    if unknown_invocations:
        model_id, profile_id, invocation_id = unknown_invocations[0]
        raise ValidationError(
            "selected build matrix references unknown compile invocation "
            f"{model_id}/{profile_id}/{invocation_id}"
        )
    selected_profiles = {(model_id, profile_id) for model_id, profile_id, _ in selected_invocations}
    previous_builds: dict[tuple[str, str, str], JsonObject] = {}
    previous_packages: dict[tuple[str, str, str], JsonObject] = {}
    for model in (previous or {}).get("models", []):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        for build in model.get("builds", []):
            if not isinstance(build, dict):
                continue
            key = (model["id"], build.get("build_profile_id"), build.get("component_id"))
            if all(isinstance(value, str) for value in key):
                previous_builds[key] = build
        for package in model.get("packages", []):
            if not isinstance(package, dict):
                continue
            key = (model["id"], package.get("build_profile_id"), package.get("package_id"))
            if all(isinstance(value, str) for value in key):
                previous_packages[key] = package
    expected_keys: set[tuple[str, str, str]] = set()
    builds_by_model: dict[str, list[JsonObject]] = {model_id: [] for model_id in models}
    packages_by_model: dict[str, list[JsonObject]] = {model_id: [] for model_id in models}
    pending_builds: list[JsonObject] = []
    for invocation in matrix:
        keys = [
            (invocation["model_id"], invocation["build_profile_id"], component_id)
            for component_id in invocation["component_ids"]
        ]
        expected_keys.update(keys)
        invocation_receipts = [receipts.get(key) for key in keys]
        if any(receipt is not None for receipt in invocation_receipts) and not all(
            receipt is not None for receipt in invocation_receipts
        ):
            raise ValidationError(
                "publication receipts must include every component of "
                f"{invocation['model_id']}/{invocation['build_profile_id']}/"
                f"{invocation['compile_invocation_id']}"
            )
        requires_pending = not any(invocation_receipts) and any(
            previous_builds.get(key, {}).get("compile_fingerprint") != invocation["build_fingerprint"]
            for key in keys
        )
        if requires_pending:
            if not incremental:
                model_id, profile_id, component_id = keys[0]
                raise ValidationError(
                    f"no immutable compile run matches {model_id}/{profile_id}/{component_id} "
                    f"fingerprint {invocation['build_fingerprint']}"
                )
            pending_builds.append(
                {
                    "model_id": invocation["model_id"],
                    "build_profile_id": invocation["build_profile_id"],
                    "compile_invocation_id": invocation["compile_invocation_id"],
                    "component_ids": invocation["component_ids"],
                    "compile_fingerprint": invocation["build_fingerprint"],
                }
            )
            # Preserve prior, verified builds while the replacement invocation
            # is pending. A model with no prior build remains visible through
            # its pending invocation alone.
            for key in keys:
                prior = previous_builds.get(key)
                if prior is not None:
                    builds_by_model[key[0]].append(prior)
            continue
        model = models[invocation["model_id"]]
        invocation_builds: list[JsonObject] = []
        for component_id in invocation["component_ids"]:
            key = (invocation["model_id"], invocation["build_profile_id"], component_id)
            receipt = receipts.get(key)
            prior = previous_builds.get(key)
            if receipt is not None and receipt.get("compile_fingerprint") != invocation["build_fingerprint"]:
                raise ValidationError(
                    f"publication receipt fingerprint does not match {key[0]}/{key[1]}/{key[2]}"
                )
            run_id = receipt.get("run_id") if receipt is not None else None
            artifact_sha256 = receipt.get("artifact_sha256") if receipt is not None else None
            if (
                run_id is None
                and prior is not None
                and prior.get("compile_fingerprint") == invocation["build_fingerprint"]
            ):
                run_id = prior.get("run_id")
                artifact_sha256 = prior.get("artifact_sha256")
            if not isinstance(run_id, str) or not run_id:
                raise ValidationError(
                    f"no immutable compile run matches {key[0]}/{key[1]}/{key[2]} "
                    f"fingerprint {invocation['build_fingerprint']}"
                )
            if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
                raise ValidationError(
                    f"no immutable MPK checksum matches {key[0]}/{key[1]}/{key[2]}"
                )
            _, binding = profile_component_for(model, key[1], key[2])
            if "artifact_suffix" in binding and len(f"{key[0]}_{binding['artifact_suffix']}_mpk.tar.gz") > 255:
                raise ValidationError(f"artifact_suffix for {key[0]}/{key[1]}/{key[2]} exceeds the filename limit")
            invocation_builds.append(
                {
                    "build_profile_id": key[1],
                    "compile_invocation_id": invocation["compile_invocation_id"],
                    "component_id": key[2],
                    **({"artifact_suffix": binding["artifact_suffix"]} if "artifact_suffix" in binding else {}),
                    "compile_fingerprint": invocation["build_fingerprint"],
                    "artifact_sha256": artifact_sha256,
                    "validation_fingerprint": None,
                    "benchmark_fingerprint": (
                        performance_key(
                            root, model, key[1], key[2], compile_run_id=run_id, artifact_sha256=artifact_sha256
                        )
                        if key[2] in invocation["benchmark_component_ids"]
                        else None
                    ),
                    "run_id": run_id,
                }
            )
        if invocation["accuracy_id"]:
            # The accuracy key lives on the build that reports the result; the field name predates it.
            by_component = {build["component_id"]: build for build in invocation_builds}
            result_component = find_devkit_accuracy_config(model, invocation["build_profile_id"])["result_component"]
            by_component[result_component]["validation_fingerprint"] = accuracy_key(
                root, model, invocation["build_profile_id"], by_component
            )
        builds_by_model[invocation["model_id"]].extend(invocation_builds)
    unexpected = sorted(set(receipts) - expected_keys)
    if unexpected:
        model_id, profile_id, component_id = unexpected[0]
        raise ValidationError(
            f"publication receipt references removed build unit: {model_id}/{profile_id}/{component_id}"
        )

    for model_id, model in sorted(models.items()):
        for declaration in model["packages"]:
            for profile_id in declaration["build_profiles"]:
                key = (model_id, profile_id, declaration["id"])
                receipt = package_receipts.get(key)
                # A selected profile may be rebuilt even when its compile
                # fingerprint is unchanged (for example, after changing only
                # its runner class). Do not seed the new catalog revision with
                # the prior package: packages are immutable within a revision,
                # and the replacement package must link the new component runs.
                prior = None if (model_id, profile_id) in selected_profiles else previous_packages.get(key)
                package = receipt or prior
                if package is None:
                    # This model/profile was not rebuilt in the current invocation.
                    # Component publication can still be catalogued; a full-package
                    # action is intentionally absent until a package is finalized.
                    continue
                if receipt is not None and receipt.get("package_fingerprint") != receipt.get("manifest_sha256"):
                    raise ValidationError(
                        f"package receipt fingerprint is invalid for {model_id}/{profile_id}/{declaration['id']}"
                    )
                required = {component_id for component_id in declaration["components"]}
                component_runs = package.get("component_run_ids")
                if not isinstance(component_runs, dict) or set(component_runs) != required:
                    raise ValidationError(
                        f"package receipt does not link every component for {model_id}/{profile_id}/{declaration['id']}"
                    )
                packages_by_model[model_id].append({
                    "package_id": declaration["id"],
                    "build_profile_id": profile_id,
                    "run_id": package["run_id"],
                    "artifact_sha256": package["artifact_sha256"],
                    "size_bytes": package["size_bytes"],
                    "package_fingerprint": package["package_fingerprint"],
                    "component_run_ids": component_runs,
                })

    model_entries = []
    for model_id, model in sorted(models.items()):
        entry = {
            "id": model_id,
            "display_name": model["display_name"],
            "description": model["description"],
            "license": model["license"],
            "maturity": model["maturity"],
            "categories": validate_model_categorization(model, taxonomy),
            "builds": sorted(
                builds_by_model[model_id],
                key=lambda item: (item["build_profile_id"], item["component_id"]),
            ),
            "packages": sorted(
                packages_by_model[model_id], key=lambda item: (item["build_profile_id"], item["package_id"])
            ),
        }
        if documentation_receipts is not None:
            documentation = documentation_receipts.get(model_id)
            if not isinstance(documentation, dict):
                raise ValidationError(f"model documentation receipt is missing: {model_id}")
            entry["documentation"] = documentation
        model_entries.append(entry)
    document: JsonObject = {
        "schema_version": 2 if incremental else 1,
        "repository": "sima-neat/models",
        "git_ref": f"refs/heads/{branch}",
        "catalog_revision": git_sha.lower(),
        "taxonomy_sha256": taxonomy_digest,
        "taxonomy": taxonomy,
        "models": model_entries,
    }
    if incremental:
        document["pending_builds"] = pending_builds
    return document


def publish_model_documentation(
    *,
    repository: str | Path,
    api_url: str,
    aws_region: str,
    bucket: str,
    kms_key_id: str | None,
    environment: str,
    branch: str,
    git_sha: str,
    github_run_url: str | None = None,
    request: Request | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> dict[str, JsonObject]:
    root = Path(repository).expanduser().resolve()
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip() or not environment.strip():
        raise ValidationError("branch and environment must not be empty")
    models = load_models(root)
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)
    branch_key = quote(branch, safe="")
    branch_identity = hashlib.sha256(branch.encode()).hexdigest()[:16]
    receipts: dict[str, JsonObject] = {}
    for model_id, model in sorted(models.items()):
        documentation = load_model_documentation(root / "models" / model_id, model["display_name"], repository=root)
        digest = documentation["readme_sha256"]
        stored = store.ensure_native_sha256(
            documentation["path"],
            f"models/{branch_key}/{git_sha.lower()}/documentation/{model_id}/README.md",
            digest,
            media_type="text/markdown; charset=utf-8",
        )
        parameters = {
            "schema_version": 2,
            "model_id": model_id,
            "source_path": documentation["source_path"],
            "source_commit": git_sha.lower(),
            "readme_sha256": digest,
            "overview_markdown": documentation["overview_markdown"],
        }
        hero = documentation.get("hero")
        stored_hero = None
        if hero is not None:
            stored_hero = store.ensure_native_sha256(
                hero["path"],
                f"models/{branch_key}/{git_sha.lower()}/documentation/{model_id}/{hero['markdown_path']}",
                hero["sha256"],
                media_type=hero["media_type"],
            )
            parameters["hero"] = {
                "source_path": hero["source_path"],
                "markdown_path": hero["markdown_path"],
                "alt_text": hero["alt_text"],
                "sha256": stored_hero.sha256,
                "size_bytes": stored_hero.size_bytes,
                "media_type": hero["media_type"],
            }
        run_fingerprint = canonical_sha256(parameters)
        environment_document = {
            "schema_version": 1,
            "vulcan_environment": environment,
            "target_platform": "documentation",
            "publisher": "model-registry",
        }
        create_body = {
            "schema_version": 1,
            "run_type": "model_documentation",
            "commit_sha": git_sha.lower(),
            "git_ref": f"refs/heads/{branch}",
            "ci_provider": "github-actions",
            "ci_run_url": github_run_url,
            "run_fingerprint": run_fingerprint,
            "environment": {
                "fingerprint": canonical_sha256(environment_document),
                "fingerprint_version": 1,
                "toolchain_name": "sima-model-registry",
                "toolchain_version": "1",
                "target_platform": "documentation",
                "properties": {"vulcan_environment": environment},
            },
            "metadata": {
                "model_id": model_id,
                "artifact_kind": "model-readme",
                "source_path": documentation["source_path"],
            },
        }
        if hero is not None:
            create_body["metadata"]["hero_source_path"] = hero["source_path"]
        invocation_identity = canonical_sha256(create_body)[:16]
        stable_key = (
            f"model-documentation-{branch_identity}-{git_sha.lower()}-{model_id}-"
            f"{run_fingerprint[:16]}-{invocation_identity}"
        )
        create = client.request("POST", "/v1/runs", create_body, f"{stable_key}-create")
        run_id = _require_text(create.get("run_id"), "run_id")
        client.request("PUT", f"/v1/runs/{run_id}/parameters", {
            "parameters": parameters,
            "canonical_sha256": canonical_sha256(parameters),
            "canonicalization_version": 1,
        }, f"{stable_key}-parameters")
        client.request("PUT", f"/v1/runs/{run_id}/artifacts/readme", {
            "artifact_type": "model-readme",
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": "text/markdown; charset=utf-8",
            "retention_class": "branch",
            "metadata": {
                "model_id": model_id,
                "source_path": documentation["source_path"],
                "source_commit": git_sha.lower(),
            },
        }, f"{stable_key}-artifact")
        if hero is not None and stored_hero is not None:
            client.request("PUT", f"/v1/runs/{run_id}/artifacts/{hero['artifact_name']}", {
                "artifact_type": "model-readme-hero",
                "s3_bucket": bucket,
                "s3_key": stored_hero.key,
                "s3_version_id": stored_hero.version_id,
                "sha256": stored_hero.sha256,
                "size_bytes": stored_hero.size_bytes,
                "media_type": hero["media_type"],
                "retention_class": "branch",
                "metadata": {
                    "model_id": model_id,
                    "source_path": hero["source_path"],
                    "source_commit": git_sha.lower(),
                },
            }, f"{stable_key}-hero-artifact")
        client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable_key}-finalize")
        receipts[model_id] = {
            "format": "markdown",
            "overview_markdown": documentation["overview_markdown"],
            "run_id": run_id,
            "artifact_name": "readme",
            "artifact_sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": "text/markdown; charset=utf-8",
            "source_path": documentation["source_path"],
            "source_commit": git_sha.lower(),
        }
        if hero is not None and stored_hero is not None:
            receipts[model_id]["hero"] = {
                "artifact_name": hero["artifact_name"],
                "markdown_path": hero["markdown_path"],
                "alt_text": hero["alt_text"],
                "artifact_sha256": stored_hero.sha256,
                "size_bytes": stored_hero.size_bytes,
                "media_type": hero["media_type"],
                "source_path": hero["source_path"],
            }
    return receipts


def publish_catalog(
    *,
    repository: str | Path,
    api_url: str,
    read_api_url: str,
    aws_region: str,
    branch: str,
    git_sha: str,
    compiled_dir: str | Path | None = None,
    package_dir: str | Path | None = None,
    selected_matrix: str | Path | None = None,
    base_branch: str | None = None,
    bucket: str | None = None,
    kms_key_id: str | None = None,
    environment: str | None = None,
    github_run_url: str | None = None,
    request: Request | None = None,
    read_request: ReadRequest | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    read = read_request or _public_registry_json
    documentation_receipts = None
    if bucket is not None:
        if not environment:
            raise ValidationError("environment is required when publishing model documentation")
        # The registry rejects model metadata changes within a revision, and each
        # workflow run creates new documentation run IDs, so a re-dispatch on a
        # cataloged commit keeps the receipts already in that revision's catalog.
        documentation_receipts = _revision_documentation_receipts(read, read_api_url, branch, git_sha)
    if bucket is not None and documentation_receipts is None:
        documentation_receipts = publish_model_documentation(
            repository=repository,
            api_url=api_url,
            aws_region=aws_region,
            bucket=bucket,
            kms_key_id=kms_key_id,
            environment=environment,
            branch=branch,
            git_sha=git_sha,
            github_run_url=github_run_url,
            request=request,
            artifact_store=artifact_store,
        )
    # Read the catalog last so another publisher has the least time to replace it
    # before the compare-and-swap POST below.
    target_catalog, previous = read_current_catalog(read, read_api_url, branch, base_branch)
    document = build_catalog_snapshot(
        repository=repository,
        branch=branch,
        git_sha=git_sha,
        compiled_dir=compiled_dir,
        package_dir=package_dir,
        selected_matrix=selected_matrix,
        previous=previous,
        incremental=True,
        documentation_receipts=documentation_receipts,
    )
    target_revision = target_catalog.get("catalog_revision") if isinstance(target_catalog, dict) else None
    previous_snapshot = target_catalog.get("canonical_sha256") if isinstance(target_catalog, dict) else None
    snapshot = {
        "initialize": target_revision != git_sha.lower(),
        "previous_snapshot": previous_snapshot,
        "catalog": document,
    }
    snapshot_identity = canonical_sha256(snapshot)[:16]
    result = client.request(
        "POST",
        f"/v1/repositories/sima-neat/models/catalogs/{git_sha.lower()}/snapshots",
        snapshot,
        # Bind the key to the complete compare-and-swap request. Documentation
        # receipt run IDs and incrementally published packages can differ across
        # workflow dispatches even for the same branch and Git revision.
        f"models-catalog-{branch}-{git_sha.lower()}-{(previous_snapshot or 'initial')[:16]}-{snapshot_identity}",
    )
    return {"catalog": document, "registry": result}


def _revision_documentation_receipts(
    read: ReadRequest, read_api_url: str, branch: str, git_sha: str
) -> dict[str, JsonObject] | None:
    record, document = read_current_catalog(read, read_api_url, branch)
    if record is None or document is None or record.get("catalog_revision") != git_sha.lower():
        return None
    receipts = {model.get("id"): model.get("documentation") for model in document.get("models", [])}
    if not receipts or not all(isinstance(receipt, dict) for receipt in receipts.values()):
        return None
    return receipts


def _public_registry_json(url: str) -> JsonObject:
    try:
        with urlopen(UrlRequest(url, headers={"User-Agent": "sima-neat-model-registry/0.1"}), timeout=30) as response:
            value = json.load(response)
    except HTTPError as exc:
        code = "unknown_error"
        message = str(exc)
        try:
            payload = json.loads(exc.read())
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                code = str(error.get("code", code))
                message = str(error.get("message", message))
        except (OSError, json.JSONDecodeError):
            pass
        raise RegistryPublicationError(
            f"public artifact registry HTTP {exc.code} {code}: {message}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryPublicationError(f"could not read public artifact registry endpoint {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryPublicationError(f"artifact registry endpoint did not return an object: {url}")
    return value


def _download_registry_bytes(url: str) -> bytes:
    try:
        with urlopen(UrlRequest(url, headers={"User-Agent": "sima-neat-model-registry/0.1"}), timeout=120) as response:
            return response.read()
    except OSError as exc:
        raise RegistryPublicationError(f"could not download registry artifact: {exc}") from exc


def stage_catalog_build(
    *,
    repository: str | Path,
    output_dir: str | Path,
    api_url: str,
    branch: str,
    catalog_revision: str,
    model_id: str,
    build_profile_id: str,
    compile_invocation_id: str,
    read_request: ReadRequest | None = None,
    download_request: DownloadRequest | None = None,
) -> JsonObject:
    """Restore a complete invocation from durable registry MPKs for later stages."""
    root = Path(repository).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    read = read_request or _public_registry_json
    download = download_request or _download_registry_bytes
    base_url = api_url.rstrip("/")
    catalog_payload = read(
        f"{base_url}/v1/repositories/sima-neat/models/catalogs/current?"
        f"git_ref={quote(f'refs/heads/{branch}', safe='')}"
    )
    catalog_record = catalog_payload.get("catalog")
    document = catalog_record.get("document") if isinstance(catalog_record, dict) else None
    if not isinstance(document, dict) or document.get("catalog_revision") != catalog_revision:
        raise ValidationError("current catalog does not match the requested catalog revision")
    catalog_model = next(
        (item for item in document.get("models", []) if isinstance(item, dict) and item.get("id") == model_id),
        None,
    )
    if catalog_model is None:
        raise ValidationError(f"model is absent from the current catalog: {model_id}")
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    invocation = next(
        (
            item for item in build_matrix({model_id: model}, [model_id], [build_profile_id])["include"]
            if item["compile_invocation_id"] == compile_invocation_id
        ),
        None,
    )
    if invocation is None:
        raise ValidationError(f"compile invocation is absent from the current model: {compile_invocation_id}")
    builds = {
        item["component_id"]: item
        for item in catalog_model.get("builds", [])
        if isinstance(item, dict)
        and item.get("build_profile_id") == build_profile_id
        and item.get("compile_invocation_id") == compile_invocation_id
    }
    if set(builds) != set(invocation["component_ids"]):
        raise ValidationError("current catalog does not link every component in the compile invocation")

    staged_outputs = []
    for component_id in invocation["component_ids"]:
        build = builds[component_id]
        run_id = build.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValidationError(f"current catalog build has no retained compile run: {component_id}")
        artifact_payload = read(f"{base_url}/v1/runs/{run_id}/artifacts/mpk")
        artifact = artifact_payload.get("artifact")
        if not isinstance(artifact, dict):
            raise ValidationError(f"artifact registry returned no MPK metadata for {run_id}")
        digest = artifact.get("sha256")
        download_url = artifact.get("download_url")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(download_url, str):
            raise ValidationError(f"artifact registry returned incomplete MPK metadata for {run_id}")
        if digest != build.get("artifact_sha256"):
            raise ValidationError(f"registry MPK checksum does not match the current catalog for {run_id}")
        payload = download(download_url)
        if hashlib.sha256(payload).hexdigest() != digest or len(payload) != artifact.get("size_bytes"):
            raise ValidationError(f"downloaded MPK failed size or SHA-256 verification for {run_id}")
        component_dir = output / model_id / build_profile_id / component_id
        component_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"{model_id}_{build_profile_id}_{component_id}_mpk.tar.gz"
        archive = component_dir / archive_name
        archive.write_bytes(payload)
        verify_mpk(component_dir)
        run_payload = read(f"{base_url}/v1/runs/{run_id}")
        run = run_payload.get("run", {}) if isinstance(run_payload, dict) else {}
        compiler_version = run.get("toolchain_version") if isinstance(run, dict) else None
        compile_result = {
            "archive": archive_name,
            "sha256": digest,
            "size_bytes": len(payload),
            "status": "success",
            "provenance": {"compiler_version": compiler_version},
        }
        (component_dir / "compile-result.json").write_text(
            json.dumps(compile_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (component_dir / "registry-publication.json").write_text(
            json.dumps({"runs": [{
                "model_id": model_id,
                "build_profile_id": build_profile_id,
                "component_id": component_id,
                "run_id": run_id,
            }]}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_outputs.append(
            {
                "component_id": component_id,
                "mpk_path": f"{component_id}/{archive_name}",
                "mpk_sha256": digest,
                "run_id": run_id,
            }
        )
    profile_dir = output / model_id / build_profile_id
    # The same manifest a compile writes, so later stages read restored and fresh builds alike.
    (profile_dir / "compile-outputs.json").write_text(
        json.dumps({
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "compile_invocation_id": compile_invocation_id,
            "outputs": [
                {key: item[key] for key in ("component_id", "mpk_path", "mpk_sha256")} for item in staged_outputs
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (profile_dir / "registry-staged.json").write_text(
        json.dumps({
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "compile_invocation_id": compile_invocation_id,
            "outputs": staged_outputs,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"count": len(staged_outputs), "outputs": staged_outputs}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"compile result {field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_text(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValidationError(f"compile result {field} must be a lowercase SHA-256 digest")
    return digest


def _component_source_media_type(source: JsonObject) -> str:
    declared = source.get("media_type")
    if isinstance(declared, str) and declared:
        return declared
    return "application/onnx" if source.get("type") == "file" else "application/vnd.sima.source-file-set+json"


def _fidelity_result(
    path: Path,
    schema_path: Path,
    *,
    model_id: str,
    build_profile_id: str,
    component_id: str,
    config: JsonObject | None = None,
) -> JsonObject:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: could not read numerical fidelity result: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ValidationError(f"{path}: numerical fidelity {location}: {first.message}")
    expected = {
        "schema_version": 1,
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "component_id": component_id,
    }
    if not isinstance(report, dict):
        raise ValidationError(f"{path}: numerical fidelity result must be an object")
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValidationError(f"{path}: numerical fidelity {field} must be {value!r}")
    metrics = report.get("metrics")
    names: set[str] = set()
    for index, metric in enumerate(metrics):
        name = metric["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValidationError(f"{path}: numerical fidelity metric names must be unique non-empty strings")
        names.add(name)
        cosine = metric["cosine_similarity"]
        if (
            not isinstance(cosine, (int, float))
            or isinstance(cosine, bool)
            or not math.isfinite(cosine)
            or not -1 <= cosine <= 1
        ):
            raise ValidationError(f"{path}: numerical fidelity cosine_similarity must be in -1..1")
        for field in ("normalized_rmse", "mean_absolute_error", "max_absolute_error"):
            value = metric[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValidationError(f"{path}: numerical fidelity {field} must be non-negative")
    if config is not None:
        validate_fidelity_result(config, report, path)
    return report


def _fidelity_measurements(report: JsonObject, fingerprint: str) -> list[JsonObject]:
    fields = (
        ("cosine_similarity", "model.fidelity.cosine_similarity", "ratio", "higher"),
        ("normalized_rmse", "model.fidelity.normalized_rmse", "ratio", "lower"),
        ("mean_absolute_error", "model.fidelity.mean_absolute_error", "value", "lower"),
        ("max_absolute_error", "model.fidelity.max_absolute_error", "value", "lower"),
    )
    return [
        {
            "measurement_key": f"{fingerprint}-{index}-{field}",
            "name": name,
            "unit": unit,
            "value": metric[field],
            "direction": direction,
            "sample_count": report["samples"],
            "aggregation": "max" if field == "max_absolute_error" else "mean",
            "dimensions": {
                "model_id": report["model_id"],
                "build_profile_id": report["build_profile_id"],
                "component_id": report["component_id"],
                "scope": report["scope"],
                "output": metric["name"],
            },
        }
        for index, metric in enumerate(report["metrics"])
        for field, name, unit, direction in fields
    ]


def _compile_timing(path: Path) -> JsonObject | None:
    """Read optional recipe timing; registry run timestamps measure publication."""
    if not path.exists():
        return None
    try:
        timing = json.loads(path.read_text(encoding="utf-8"))
        if timing["scope"] != "compile_recipe":
            raise ValueError("unsupported timing scope")
        start = datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(timing["finished_at"].replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("invalid timing interval")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValidationError(f"{path}: invalid compile timing") from exc
    return {"scope": "compile_recipe", "started_at": start.isoformat(), "finished_at": end.isoformat()}


def _preflight_compilations(
    root: Path,
    dist: Path,
    models: dict[str, JsonObject],
    *,
    environment: str,
    git_sha: str,
) -> list[JsonObject]:
    compiler_dependency = resolve_model_compiler(root)
    expected_results: set[Path] = set()
    invocation_by_result: dict[Path, str] = {}
    output_by_result: dict[Path, JsonObject] = {}
    manifests = sorted(dist.glob("*/*/compile-outputs.json"))
    for manifest_path in manifests:
        model_id, build_profile_id = manifest_path.relative_to(dist).parts[:2]
        if model_id not in models:
            raise ValidationError(f"compile outputs reference unknown model: {model_id}")
        model = models[model_id]
        profiles = {item["id"]: item for item in model["build_profiles"]}
        if build_profile_id not in profiles:
            raise ValidationError(f"compile outputs reference unknown build profile: {model_id}/{build_profile_id}")
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation_id = raw_manifest["compile_invocation_id"]
            invocation = next(
                item for item in compile_invocations_for(profiles[build_profile_id]) if item["id"] == invocation_id
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, StopIteration) as exc:
            raise ValidationError(f"{manifest_path}: unknown or invalid compile invocation") from exc
        validated = validate_compile_outputs(
            manifest_path,
            manifest_path.parent,
            model_id=model_id,
            build_profile_id=build_profile_id,
            compile_invocation_id=invocation_id,
            component_ids=list(invocation["components"]),
        )
        for output in validated["outputs"]:
            result_path = manifest_path.parent / output["component_id"] / "compile-result.json"
            expected_results.add(result_path)
            invocation_by_result[result_path] = invocation_id
            output_by_result[result_path] = output

    actual_results = set(dist.glob("*/*/*/compile-result.json"))
    if actual_results != expected_results:
        raise ValidationError("compiled output tree does not match its compile output manifests")

    prepared = []
    for result_path in sorted(actual_results):
        model_id, build_profile_id, component_id = result_path.relative_to(dist).parts[:3]
        model = models[model_id]
        profiles = {item["id"]: item for item in model["build_profiles"]}
        components = {item["id"]: item for item in model["components"]}
        profile = profiles[build_profile_id]
        bindings = {item["id"]: item for item in profile["components"]}
        if component_id not in components or component_id not in bindings:
            raise ValidationError(
                f"compiled output references unavailable component: {model_id}/{build_profile_id}/{component_id}"
            )
        component = components[component_id]
        binding = bindings[component_id]
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"{result_path}: could not read compile result: {exc}") from exc
        provenance = result.get("provenance")
        if not isinstance(provenance, dict):
            raise ValidationError(f"{result_path}: provenance must be an object")
        expected = {
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "component_id": component_id,
            "git_sha": git_sha,
            "environment": environment,
        }
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise ValidationError(f"{result_path}: provenance {field} must be {value!r}")

        archive_name = _require_text(result.get("archive"), "archive")
        archive = result_path.parent / archive_name
        if not archive.is_file() or archive.parent != result_path.parent:
            raise ValidationError(f"{result_path}: archive is missing from the packaged output")
        artifact_sha = _require_sha256(result.get("sha256"), "sha256")
        declared_output = output_by_result[result_path]
        if (
            archive.resolve() != Path(declared_output["mpk_path"]).resolve()
            or artifact_sha != declared_output["mpk_sha256"]
        ):
            raise ValidationError(f"{result_path}: archive does not match the compile output manifest")
        actual_sha, actual_size = sha256_file(archive)
        if artifact_sha != actual_sha or int(result.get("size_bytes", -1)) != actual_size:
            raise ValidationError(f"{result_path}: archive checksum or size does not match")

        compiler_version = _require_text(provenance.get("compiler_version"), "compiler_version")
        compiler_label = _require_text(provenance.get("compiler_label"), "compiler_label")
        if compiler_version != compiler_dependency["version"]:
            raise ValidationError(f"{result_path}: compiler_version does not match the dependency manifest")
        if compiler_label != compiler_dependency["label"]:
            raise ValidationError(f"{result_path}: compiler_label does not match the dependency manifest")
        compiler_identity = compiler_dependency["identity"]
        target = _require_text(provenance.get("target"), "target")
        source_sha = _require_sha256(provenance.get("source_sha256"), "source_sha256")
        if source_sha != source_identity(component["source"]):
            raise ValidationError(f"{result_path}: source_sha256 does not match the component source contract")
        build_fingerprint = _require_sha256(provenance.get("build_fingerprint"), "build_fingerprint")
        expected_build_fingerprint = compile_key(
            root, model_id, build_profile_id, invocation_by_result[result_path], compiler_identity
        )
        if build_fingerprint != expected_build_fingerprint:
            raise ValidationError(f"{result_path}: build_fingerprint does not match semantic compile inputs")
        fidelity_path = result_path.parent / "fidelity.json"
        configured_fidelity = find_fidelity_config(
            root / "models" / model_id,
            model,
            build_profile_id=build_profile_id,
            component_id=component_id,
        )
        if configured_fidelity is not None:
            owns_fidelity = configured_fidelity["result_component"] == component_id
            if owns_fidelity != fidelity_path.is_file():
                expectation = "requires" if owns_fidelity else "must not contain"
                raise ValidationError(
                    f"{result_path.parent}: configured numerical fidelity {expectation} fidelity.json"
                )
        fidelity_evidence = None
        if fidelity_path.is_file():
            fidelity = _fidelity_result(
                fidelity_path,
                root / "schemas" / "numerical-fidelity.schema.json",
                model_id=model_id,
                build_profile_id=build_profile_id,
                component_id=component_id,
                config=configured_fidelity,
            )
            fidelity_evidence = {
                "path": fidelity_path,
                "document": fidelity,
                "sha256": sha256_file(fidelity_path)[0],
            }
        prepared.append(
            {
                "result_path": result_path,
                "compile_timing": _compile_timing(result_path.parent.parent / "compile-timing.json"),
                "model_id": model_id,
                "build_profile_id": build_profile_id,
                "compile_invocation_id": invocation_by_result[result_path],
                "component_id": component_id,
                "model": model,
                "profile": profile,
                "component": component,
                "binding": binding,
                "archive": archive,
                "artifact_sha": artifact_sha,
                "compiler_version": compiler_version,
                "compiler_label": compiler_label,
                "target": target,
                "source_sha": source_sha,
                "build_fingerprint": build_fingerprint,
                "fidelity": fidelity_evidence,
            }
        )
    return prepared


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

    models = load_models(root)
    compilations = _preflight_compilations(
        root,
        dist,
        models,
        environment=environment,
        git_sha=git_sha,
    )
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)
    branch_key = quote(branch, safe="")
    commit_tag = git_sha[:12]
    published: list[JsonObject] = []

    for compilation in compilations:
        model_id = compilation["model_id"]
        build_profile_id = compilation["build_profile_id"]
        compile_invocation_id = compilation["compile_invocation_id"]
        component_id = compilation["component_id"]
        model = compilation["model"]
        component = compilation["component"]
        binding = compilation["binding"]
        archive = compilation["archive"]
        artifact_sha = compilation["artifact_sha"]
        compiler_version = compilation["compiler_version"]
        compiler_label = compilation["compiler_label"]
        target = compilation["target"]
        source_sha = compilation["source_sha"]
        build_fingerprint = compilation["build_fingerprint"]
        parameters = {
            "schema_version": 2,
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "compile_invocation_id": compile_invocation_id,
            "component_id": component_id,
            "target": target,
            "compiler_version": compiler_version,
            "compiler_label": compiler_label,
            "source_sha256": source_sha,
            "build_fingerprint": build_fingerprint,
            "recipe": component["recipe"],
            "compile_parameters": binding.get("parameters", {}),
            "shape_profile": binding.get("shape_profile"),
            "calibration": binding.get("calibration"),
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
            f"{github_run_id}:{environment}:{branch}:{git_sha}:{model_id}:"
            f"{build_profile_id}:{component_id}:{run_fingerprint}".encode()
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
                    "build_profile_id": build_profile_id,
                    "component_id": component_id,
                    "artifact_kind": "mpk",
                    "github_run_id": github_run_id,
                    **({"compile_timing": compilation["compile_timing"]} if compilation["compile_timing"] else {}),
                    "license": model["license"],
                },
            },
            f"{stable_key}-create",
        )
        run_id = _require_text(create.get("run_id"), "run_id")
        source = component["source"]
        source_metadata = {"model_id": model_id, "component_id": component_id, "source_type": source["type"]}
        model_card = model.get("upstream", {}).get("model_card")
        if isinstance(model_card, str) and model_card:
            source_metadata["model_card"] = model_card
        source_reference = source.get("uri")
        if not isinstance(source_reference, str):
            source_reference = f"{model['upstream']['repository']}#{source['entrypoint']}"
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/inputs/source-component",
            {
                "input_type": "onnx" if source["type"] == "file" else "model-source-file-set",
                "source_reference": source_reference,
                "sha256": source_sha,
                "size_bytes": source.get(
                    "size_bytes", sum(int(member["size_bytes"]) for member in source.get("files", ()))
                ),
                "media_type": _component_source_media_type(source),
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
                "artifact_sha256": artifact_sha,
                "s3_version_id": stored.version_id,
                "sha256": stored.sha256,
                "size_bytes": stored.size_bytes,
                "media_type": "application/gzip",
                "retention_class": "branch",
                "metadata": {
                    "model_id": model_id,
                    "build_profile_id": build_profile_id,
                    "component_id": component_id,
                    "target": target,
                },
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
        fidelity_evidence = compilation["fidelity"]
        fidelity_publication = None
        if fidelity_evidence is not None:
            fidelity_path = fidelity_evidence["path"]
            fidelity = fidelity_evidence["document"]
            fidelity_sha = fidelity_evidence["sha256"]
            fidelity_fingerprint = canonical_sha256(fidelity)
            fidelity_key = (
                f"models/{branch_key}/{commit_tag}/{model_id}/{build_profile_id}/{component_id}/fidelity/"
                f"{fidelity_fingerprint}/numerical-fidelity.json"
            )
            fidelity_stored = store.ensure_native_sha256(
                fidelity_path,
                fidelity_key,
                fidelity_sha,
                media_type="application/json",
            )
            fidelity_artifact = client.request(
                "PUT",
                f"/v1/runs/{run_id}/artifacts/numerical-fidelity",
                {
                    "artifact_type": "numerical-fidelity",
                    "s3_bucket": bucket,
                    "s3_key": fidelity_stored.key,
                    "s3_version_id": fidelity_stored.version_id,
                    "sha256": fidelity_stored.sha256,
                    "size_bytes": fidelity_stored.size_bytes,
                    "media_type": "application/json",
                    "retention_class": "branch",
                    "metadata": {
                        "model_id": model_id,
                        "build_profile_id": build_profile_id,
                        "component_id": component_id,
                        "scope": fidelity["scope"],
                        "samples": fidelity["samples"],
                    },
                },
                f"{stable_key}-fidelity-artifact",
            )
            fidelity_test: JsonObject = {
                "status": "passed",
                "version": str(fidelity["schema_version"]),
                "summary": fidelity,
            }
            if fidelity_artifact.get("artifact_id"):
                fidelity_test["report_artifact_id"] = fidelity_artifact["artifact_id"]
            client.request(
                "PUT",
                f"/v1/runs/{run_id}/test-results/correctness/numerical-fidelity",
                fidelity_test,
                f"{stable_key}-fidelity-test-result",
            )
            client.request(
                "POST",
                f"/v1/runs/{run_id}/measurements/batch",
                {"measurements": _fidelity_measurements(fidelity, fidelity_fingerprint)},
                f"{stable_key}-fidelity-measurements",
            )
            fidelity_publication = {
                "s3_key": fidelity_stored.key,
                "result_fingerprint": fidelity_fingerprint,
                "samples": fidelity["samples"],
            }
        client.request(
            "POST",
            f"/v1/runs/{run_id}/finalize",
            {"status": "completed"},
            f"{stable_key}-finalize",
        )
        published.append(
            {
                "model_id": model_id,
                "build_profile_id": build_profile_id,
                "component_id": component_id,
                "run_id": run_id,
                "compile_fingerprint": build_fingerprint,
                "artifact_sha256": artifact_sha,
                "s3_key": stored.key,
                "fidelity": fidelity_publication,
            }
        )

    return {
        "count": len(published),
        "runs": published,
    }


def publish_packages(
    *,
    repository: str | Path,
    package_dir: str | Path,
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
    """Publish already-assembled full packages; GitHub artifacts remain transport only."""
    root = Path(repository).expanduser().resolve()
    packages_root = Path(package_dir).expanduser().resolve()
    if not packages_root.is_dir():
        raise ValidationError(f"assembled package directory not found: {packages_root}")
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip() or not github_run_id.strip() or not github_run_attempt.strip():
        raise ValidationError("branch, GitHub run ID, and run attempt must not be empty")

    models = load_models(root)
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)
    branch_key, commit_tag = quote(branch, safe=""), git_sha[:12]
    published: list[JsonObject] = []

    for result_path in sorted(packages_root.glob("*/*/packages/*/package-result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"could not read package result {result_path}: {exc}") from exc
        if not isinstance(result, dict):
            raise ValidationError(f"{result_path}: package result must be an object")
        manifest = result.get("manifest")
        if not isinstance(manifest, dict):
            raise ValidationError(f"{result_path}: package result manifest must be an object")
        model_id = _require_text(result.get("model_id"), "package model_id")
        profile_id = _require_text(result.get("build_profile_id"), "package build_profile_id")
        package_id = _require_text(result.get("package_id"), "package package_id")
        if model_id not in models:
            raise ValidationError(f"{result_path}: package references unknown model {model_id}")
        model = models[model_id]
        declaration = next((item for item in model["packages"] if item["id"] == package_id), None)
        if declaration is None or profile_id not in declaration["build_profiles"]:
            raise ValidationError(f"{result_path}: package is not declared for {model_id}/{profile_id}/{package_id}")
        archive_name = _require_text(result.get("archive"), "package archive")
        archive = result_path.parent / archive_name
        if archive.parent != result_path.parent or archive.suffix != ".zip" or not archive.is_file():
            raise ValidationError(f"{result_path}: expected one local ZIP package archive")
        artifact_sha, artifact_size = sha256_file(archive)
        if result.get("sha256") != artifact_sha or result.get("size_bytes") != artifact_size:
            raise ValidationError(f"{result_path}: package archive does not match package-result.json")
        if (
            manifest.get("model_id") != model_id
            or manifest.get("build_profile_id") != profile_id
            or manifest.get("package_id") != package_id
        ):
            raise ValidationError(f"{result_path}: package identity does not match its manifest")
        if manifest.get("license", {}).get("redistribution") in {"prohibited", "unknown"}:
            raise ValidationError(f"{result_path}: package redistribution policy forbids publication")
        components = manifest.get("components")
        component_ids = (
            {item.get("id") for item in components if isinstance(item, dict)}
            if isinstance(components, list)
            else set()
        )
        if component_ids != set(declaration["components"]):
            raise ValidationError(f"{result_path}: package manifest does not include exactly the declared components")
        component_runs: dict[str, str] = {}
        for component in components:
            if not isinstance(component, dict):
                raise ValidationError(f"{result_path}: package component must be an object")
            component_id = _require_text(component.get("id"), "package component id")
            run_id = component.get("compile_run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValidationError(f"{result_path}: package component {component_id} has no finalized compile run")
            component_runs[component_id] = run_id

        manifest_sha = canonical_sha256(manifest)
        parameters = {
            "schema_version": 1,
            "model_id": model_id,
            "build_profile_id": profile_id,
            "package_id": package_id,
            "package_manifest_sha256": manifest_sha,
            "component_run_ids": component_runs,
        }
        run_fingerprint = canonical_sha256({**parameters, "artifact_sha256": artifact_sha})
        environment_document = {
            "schema_version": 1, "vulcan_environment": environment,
            "target_platform": manifest.get("target"), "packager": "model-registry",
        }
        stable_key = "model-package-" + hashlib.sha256(
            f"{github_run_id}:{environment}:{branch}:{git_sha}:{model_id}:{profile_id}:{package_id}:{run_fingerprint}".encode()
        ).hexdigest()
        create = client.request("POST", "/v1/runs", {
            "schema_version": 1, "run_type": "model_package", "commit_sha": git_sha,
            "git_ref": f"refs/heads/{branch}", "ci_provider": "github-actions", "ci_run_url": github_run_url,
            "run_fingerprint": run_fingerprint,
            "environment": {
                "fingerprint": canonical_sha256(environment_document), "fingerprint_version": 1,
                "toolchain_name": "sima-model-registry", "toolchain_version": "1",
                "target_platform": manifest.get("target"),
                "properties": {"vulcan_environment": environment},
            },
            "metadata": {
                "model_id": model_id, "build_profile_id": profile_id, "package_id": package_id,
                "artifact_kind": "full-package", "github_run_id": github_run_id,
                "license": manifest["license"], "component_run_ids": component_runs,
            },
        }, f"{stable_key}-create")
        run_id = _require_text(create.get("run_id"), "run_id")
        for component_id, component_run_id in sorted(component_runs.items()):
            client.request("PUT", f"/v1/runs/{run_id}/inputs/component-{component_id}", {
                "input_type": "model-pack", "source_reference": f"registry-run:{component_run_id}",
                "sha256": next(item["sha256"] for item in components if item["id"] == component_id),
                "size_bytes": next(item["size_bytes"] for item in components if item["id"] == component_id),
                "media_type": "application/gzip",
                "metadata": {"component_id": component_id, "compile_run_id": component_run_id},
            }, f"{stable_key}-component-{component_id}")
        client.request("PUT", f"/v1/runs/{run_id}/parameters", {
            "parameters": parameters, "canonical_sha256": canonical_sha256(parameters), "canonicalization_version": 1,
        }, f"{stable_key}-parameters")
        stored = store.ensure_native_sha256(
            archive,
            f"models/{branch_key}/{commit_tag}/{archive.relative_to(packages_root).as_posix()}",
            artifact_sha,
            media_type="application/zip",
        )
        client.request("PUT", f"/v1/runs/{run_id}/artifacts/package", {
            "artifact_type": "model-package", "s3_bucket": bucket, "s3_key": stored.key,
            "s3_version_id": stored.version_id, "sha256": stored.sha256, "size_bytes": stored.size_bytes,
            "media_type": "application/zip", "retention_class": "branch",
            "metadata": {"model_id": model_id, "build_profile_id": profile_id, "package_id": package_id,
                         "manifest_sha256": manifest_sha, "component_run_ids": component_runs},
        }, f"{stable_key}-artifact")
        client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable_key}-finalize")
        published.append({
            "model_id": model_id, "build_profile_id": profile_id, "package_id": package_id, "run_id": run_id,
            "artifact_sha256": artifact_sha, "size_bytes": artifact_size, "manifest_sha256": manifest_sha,
            "package_fingerprint": manifest_sha, "component_run_ids": component_runs, "s3_key": stored.key,
        })
    if not published:
        raise ValidationError("no assembled package results were found")
    return {"count": len(published), "packages": published}
