from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import random
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from jsonschema import Draft202012Validator

from .artifact import (
    compile_output_fields,
    deployment_files,
    llima_compiled_parameters,
    validate_compile_outputs,
    verify_artifact,
    verify_llima,
)
from .benchmark_declarations import benchmarked_profiles
from .compile_matrix import build_matrix
from .dependencies import resolve_model_compiler
from .documentation import load_model_documentation
from .errors import ModelRegistryError, ValidationError
from .fetch import sha256_file
from .fingerprints import benchmark_fingerprint, compile_fingerprint
from .mla_stage_simulation import SIMULATION_FILE
from .model import build_profile_for, compile_invocations_for, load_and_validate_model, load_models, mpk_filename
from .package import FIDELITY_FILE
from .registry_http import RegistryPublicationError, SigV4RegistryClient
from .registry_http import read_registry_json_authenticated as _public_registry_json
from .runner_approval import runner_provenance
from .runtime_files import restore_runtime_files
from .source import source_identity
from .stage_plan import merge_catalog_sources, read_catalog_sources, read_current_catalog
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
# Catalog fields the registry owns for each ref: a snapshot may omit them but not carry another ref's.
SERVER_OWNED_SELECTIONS = ("approved_deployments", "branch_deployments")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()




@dataclass(frozen=True)
class S3Object:
    key: str
    version_id: str
    sha256: str
    size_bytes: int


class S3ArtifactStore:
    def __init__(
        self, bucket: str, *, kms_key_id: str | None = None, receipt_dir: Path | None = None, s3_client=None
    ) -> None:
        self.bucket = bucket
        self.kms_key_id = (kms_key_id or "").strip()
        self.receipt_dir = receipt_dir
        self._s3 = s3_client

    def _record(self, stored: S3Object) -> S3Object:
        if self.receipt_dir is not None:
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            path = self.receipt_dir / f"{canonical_sha256([self.bucket, stored.key])}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "s3_bucket": self.bucket, "s3_key": stored.key, "s3_version_id": stored.version_id,
                "sha256": stored.sha256, "size_bytes": stored.size_bytes,
            }, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
        return stored

    def ensure_native_sha256(
        self,
        path: Path,
        key: str,
        sha256: str,
        *,
        media_type: str = "application/gzip",
        previous_object: tuple[str, str | None] | None = None,
    ) -> S3Object:
        expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        saved = None
        if self.receipt_dir is not None:
            receipt = self.receipt_dir / f"{canonical_sha256([self.bucket, key])}.json"
            if receipt.is_file():
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if (saved.get("sha256"), saved.get("size_bytes")) != (sha256, path.stat().st_size):
                    raise RegistryPublicationError(f"saved upload differs from {key}; start a new candidate")
        head = self._head(key, saved["s3_version_id"]) if saved else self._head(key)
        size = path.stat().st_size
        if _matches_s3_object(head, expected_checksum, size):
            return self._record(S3Object(key, head["VersionId"], sha256, size))
        if saved:
            raise RegistryPublicationError(
                f"saved S3 version is unavailable or changed for {key}; start a new candidate"
            )
        if previous_object:
            previous_key, previous_version = previous_object
            prior = self._head(previous_key, previous_version)
            if _matches_s3_object(prior, expected_checksum, size):
                return self._record(S3Object(previous_key, prior["VersionId"], sha256, size))

        if size > 5 * 1024**3:
            raise RegistryPublicationError(f"registry artifact exceeds the 5 GiB single-upload limit: {path}")
        if self._s3 is None:
            version_id = self._put_cli(path, key, expected_checksum, media_type)
        else:
            version_id = self._put_sdk(path, key, expected_checksum, media_type)
        if not isinstance(version_id, str) or not version_id:
            raise RegistryPublicationError(f"S3 versioning did not return a version for {key}")
        verified = self._head(key, version_id)
        if verified.get("ChecksumSHA256") != expected_checksum:
            raise RegistryPublicationError(f"S3 native SHA-256 verification failed for {key}")
        if verified.get("ContentLength") != size:
            raise RegistryPublicationError(f"S3 object size does not match the packaged artifact for {key}")
        return self._record(S3Object(key, version_id, sha256, size))

    def _put_cli(self, path: Path, key: str, expected_checksum: str, media_type: str) -> str | None:
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
        return uploaded.get("VersionId") if isinstance(uploaded, dict) else None

    def _put_sdk(self, path: Path, key: str, checksum: str, media_type: str) -> str | None:
        from botocore.exceptions import BotoCoreError, ClientError

        arguments = {
            "Bucket": self.bucket,
            "Key": key,
            "ChecksumAlgorithm": "SHA256",
            "ChecksumSHA256": checksum,
            "ContentType": media_type,
            "ServerSideEncryption": "aws:kms",
            "IfNoneMatch": "*",
        }
        if self.kms_key_id:
            arguments["SSEKMSKeyId"] = self.kms_key_id
        for attempt in range(3):
            try:
                with path.open("rb") as body:
                    return self._s3.put_object(Body=body, **arguments).get("VersionId")
            except ClientError as exc:
                code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code in {409, 412}:
                    current = self._head(key)
                    if _matches_s3_object(current, checksum, path.stat().st_size):
                        return current["VersionId"]
                    if code == 409 and attempt < 2:
                        continue
                raise RegistryPublicationError(f"Could not publish immutable s3://{self.bucket}/{key}: {exc}") from exc
            except BotoCoreError as exc:
                raise RegistryPublicationError(f"Could not upload s3://{self.bucket}/{key}: {exc}") from exc
        raise AssertionError("unreachable")

    def _head(self, key: str, version_id: str | None = None) -> JsonObject:
        if self._s3 is not None:
            from botocore.exceptions import BotoCoreError, ClientError

            arguments = {"Bucket": self.bucket, "Key": key, "ChecksumMode": "ENABLED"}
            if version_id:
                arguments["VersionId"] = version_id
            try:
                return self._s3.head_object(**arguments)
            except ClientError as exc:
                code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                # Without ListBucket, the publisher sees 403 on a missing key. A conditional
                # PUT can create the object, but cannot replace an existing inaccessible object.
                if not version_id and code in {403, 404}:
                    return {}
                raise RegistryPublicationError(f"Could not verify s3://{self.bucket}/{key}: {exc}") from exc
            except BotoCoreError as exc:
                raise RegistryPublicationError(f"Could not verify s3://{self.bucket}/{key}: {exc}") from exc
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


def _matches_s3_object(head: JsonObject, checksum: str, size: int) -> bool:
    return (
        head.get("ChecksumSHA256") == checksum
        and head.get("ContentLength") == size
        and isinstance(head.get("VersionId"), str)
        and bool(head["VersionId"])
    )


def _publish_deployment_files(
    client: SigV4RegistryClient,
    store: S3ArtifactStore,
    *,
    run_id: str,
    stable_key: str,
    bucket: str,
    deployment: Path,
    files: list[JsonObject],
    metadata: JsonObject,
    retention_class: str,
    upload_prefix: str | None = None,
) -> None:
    """Register file references on this run, reusing immutable S3 payloads across runs."""
    deployment_files({"files": files})

    def publish(items: list[JsonObject]) -> None:
        item = items[0]
        stored = store.ensure_native_sha256(
            deployment / item["path"],
            f"{upload_prefix or 'models'}/files/{item['sha256']}/{Path(item['path']).name}",
            item["sha256"],
            media_type="application/octet-stream",
        )
        for item in items:
            client.request("PUT", f"/v1/runs/{run_id}/artifacts/{item['artifact']}", {
                "artifact_type": "model-file", "s3_bucket": bucket, "s3_key": stored.key,
                "s3_version_id": stored.version_id, "sha256": stored.sha256, "size_bytes": stored.size_bytes,
                "media_type": "application/octet-stream", "retention_class": retention_class,
                "metadata": {**metadata, "path": item["path"], "storage_backend": "s3"},
            }, "file-" + canonical_sha256([stable_key, item["artifact"]]))

    payloads: dict[tuple[str, str], list[JsonObject]] = {}
    for item in files:
        payloads.setdefault((item["sha256"], Path(item["path"]).name), []).append(item)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(publish, sorted(payloads.values(), key=lambda items: items[0]["size_bytes"], reverse=True)))


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


def _catalog_models(models: dict[str, Any]) -> dict[str, Any]:
    """Only refresh conversion-backed catalog entries selected by this workflow."""
    from .conversion import selections

    selected = selections()
    return {key: model for key, model in models.items()
            if model["components"][0]["source"]["type"] != "conversion" or key in selected}


def _head_matrix(root: Path, models: dict[str, Any]) -> list[JsonObject]:
    """Every compile invocation the checkout declares, with its compile fingerprint."""
    models = _catalog_models(models)
    compiler_version = resolve_model_compiler(root)["identity"]
    return build_matrix(
        models, sorted(models), repository=root, compiler_version=compiler_version,
    )["include"]


def _catalog_outputs(model: JsonObject) -> tuple[list[JsonObject], list[JsonObject]]:
    """Resolve selected deployments without changing server-owned approval records."""
    selections = {**model.get("approved_deployments", {}), **model.get("branch_deployments", {})}
    builds = [item for item in model.get("builds", []) if item.get("build_profile_id") not in selections]
    packages = [item for item in model.get("packages", []) if item.get("build_profile_id") not in selections]
    for profile_id, selection in selections.items():
        requirements = selection["requirements"]
        components = [item for item in selection["artifacts"] if item.get("component_id")]
        for component in components:
            invocation = next(item for item in requirements["invocations"]
                              if component["component_id"] in item["component_ids"])
            builds.append({
                **component, "build_profile_id": profile_id, "compile_invocation_id": profile_id,
                "compile_fingerprint": invocation["compile_fingerprint"],
            })
        for package in (item for item in selection["artifacts"] if item.get("package_id")):
            requirement = next(item for item in requirements["packages"] if item["id"] == package["package_id"])
            packages.append({
                **package, "build_profile_id": profile_id,
                "component_run_ids": {item["component_id"]: item["run_id"] for item in components
                                      if item["component_id"] in requirement["component_ids"]},
            })
    return builds, packages


def _refresh_approved_benchmarks(
    root: Path, document: JsonObject, source: JsonObject, read_api_url: str, read: ReadRequest,
    *, model_id: str | None = None, profile_id: str | None = None,
) -> None:
    """Point at the approved producing runs, without changing approval-owned outputs or fingerprint formats."""
    if document.get("git_ref") not in {"refs/heads/develop", "refs/heads/main"}:
        return
    previous = {model["id"]: model for model in source.get("models", [])}
    digests = {}
    for entry in document.get("models", []):
        if model_id is not None and entry["id"] != model_id:
            continue
        selections = previous.get(entry["id"], {}).get("approved_deployments", {})
        if not selections:
            continue
        model = load_and_validate_model(root / "models" / entry["id"])
        fingerprints = dict(entry.get("benchmark_fingerprints", {}))
        for selected_profile, selection in selections.items():
            if profile_id is not None and selected_profile != profile_id:
                continue
            # Raw builds may still describe an older automatic build of this profile.
            fingerprints.pop(selected_profile, None)
            stages = [stage for stage in ("performance", "accuracy")
                      if selected_profile in benchmarked_profiles(model, stage)]
            if not stages:
                continue
            builds = {build["component_id"]: build for build in _catalog_outputs({
                "approved_deployments": {selected_profile: selection},
            })[0]}
            if set(builds) != {component["id"] for component in model["components"]}:
                continue
            for component in model["components"]:
                if component["interface"]["format"] == "huggingface":
                    continue
                build = builds[component["id"]]
                run_id = build["run_id"]
                if run_id not in digests:
                    artifact = read(f"{read_api_url.rstrip('/')}/v1/runs/{quote(run_id, safe='')}/artifacts/mpk")
                    digests[run_id] = _require_sha256(artifact.get("artifact", {}).get("sha256"), "artifact sha256")
                build["artifact_sha256"] = digests[run_id]
            fingerprints[selected_profile] = {
                stage: benchmark_fingerprint(root, model, selected_profile, builds, stage) for stage in stages
            }
        if fingerprints:
            entry["benchmark_fingerprints"] = fingerprints
        else:
            entry.pop("benchmark_fingerprints", None)


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
    all_models = load_models(root)
    models = _catalog_models(all_models)
    deferred = set(all_models) - set(models)
    taxonomy = canonical_taxonomy_document(load_taxonomy(root / TAXONOMY_RELATIVE_PATH))
    taxonomy_digest = taxonomy_sha256(taxonomy)
    matrix = _head_matrix(root, models)
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
    previous_benchmark_fingerprints: dict[tuple[str, str], JsonObject] = {}
    for model in (previous or {}).get("models", []):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        for profile_id, fingerprints in (model.get("benchmark_fingerprints") or {}).items():
            previous_benchmark_fingerprints[(model["id"], profile_id)] = fingerprints
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
    benchmark_fingerprints_by_model: dict[str, dict[str, JsonObject]] = {model_id: {} for model_id in models}
    pending_builds: list[JsonObject] = []

    def carry_forward(keys: list[tuple[str, str, str]]) -> None:
        """Keep a profile's prior builds and their benchmark fingerprints visible."""
        for key in keys:
            prior = previous_builds.get(key)
            if prior is not None:
                builds_by_model[key[0]].append(prior)
        fingerprints = previous_benchmark_fingerprints.get(keys[0][:2])
        if fingerprints is not None and any(key in previous_builds for key in keys):
            benchmark_fingerprints_by_model[keys[0][0]][keys[0][1]] = fingerprints

    for invocation in matrix:
        keys = [
            (invocation["model_id"], invocation["build_profile_id"], component_id)
            for component_id in invocation["component_ids"]
        ]
        expected_keys.update(keys)
        if invocation.get("build_on_demand", False):
            # A manual build publishes a candidate. Only approved catalog entries
            # carry forward here, retaining the inputs and provenance they used.
            carry_forward(keys)
            continue
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
            previous_builds.get(key, {}).get("compile_fingerprint") != invocation["compile_fingerprint"]
            for key in keys
        )
        if requires_pending:
            if not incremental:
                model_id, profile_id, component_id = keys[0]
                raise ValidationError(
                    f"no immutable compile run matches {model_id}/{profile_id}/{component_id} "
                    f"fingerprint {invocation['compile_fingerprint']}"
                )
            pending_builds.append(
                {
                    "model_id": invocation["model_id"],
                    "build_profile_id": invocation["build_profile_id"],
                    "compile_invocation_id": invocation["compile_invocation_id"],
                    "component_ids": invocation["component_ids"],
                    "compile_fingerprint": invocation["compile_fingerprint"],
                }
            )
            # Preserve prior, verified builds while the replacement invocation
            # is pending. A model with no prior build remains visible through
            # its pending invocation alone.
            carry_forward(keys)
            continue
        model = models[invocation["model_id"]]
        invocation_builds: list[JsonObject] = []
        for component_id in invocation["component_ids"]:
            key = (invocation["model_id"], invocation["build_profile_id"], component_id)
            receipt = receipts.get(key)
            prior = previous_builds.get(key)
            if receipt is not None and receipt.get("compile_fingerprint") != invocation["compile_fingerprint"]:
                raise ValidationError(
                    f"publication receipt fingerprint does not match {key[0]}/{key[1]}/{key[2]}"
                )
            run_id = receipt.get("run_id") if receipt is not None else None
            artifact_sha256 = receipt.get("artifact_sha256") if receipt is not None else None
            if (
                run_id is None
                and prior is not None
                and prior.get("compile_fingerprint") == invocation["compile_fingerprint"]
            ):
                run_id = prior.get("run_id")
                artifact_sha256 = prior.get("artifact_sha256")
            if not isinstance(run_id, str) or not run_id:
                raise ValidationError(
                    f"no immutable compile run matches {key[0]}/{key[1]}/{key[2]} "
                    f"fingerprint {invocation['compile_fingerprint']}"
                )
            if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
                raise ValidationError(
                    f"no immutable MPK checksum matches {key[0]}/{key[1]}/{key[2]}"
                )
            invocation_builds.append(
                {
                    "build_profile_id": key[1],
                    "compile_invocation_id": invocation["compile_invocation_id"],
                    "component_id": key[2],
                    "compile_fingerprint": invocation["compile_fingerprint"],
                    "artifact_sha256": artifact_sha256,
                    "run_id": run_id,
                }
            )
        by_component = {build["component_id"]: build for build in invocation_builds}
        # Each benchmark fingerprint names the profile's whole-model run over every component build.
        fingerprints = {
            stage: benchmark_fingerprint(root, model, invocation["build_profile_id"], by_component, stage)
            for stage in invocation["stages"]
        }
        if fingerprints:
            benchmark_fingerprints_by_model[invocation["model_id"]][invocation["build_profile_id"]] = fingerprints
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
                if model.get("build-on-demand", False):
                    prior = previous_packages.get(key)
                    if prior is not None:
                        packages_by_model[model_id].append(prior)
                    continue
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
        if "parameter_count_billion" in model:
            entry["parameter_count_billion"] = model["parameter_count_billion"]
        if benchmark_fingerprints_by_model[model_id]:
            entry["benchmark_fingerprints"] = dict(sorted(benchmark_fingerprints_by_model[model_id].items()))
        if model.get("build-on-demand", False):
            entry["build_on_demand"] = True
            components = {item["id"]: item for item in model["components"]}
            requirements = {}
            for profile in sorted(model["build_profiles"], key=lambda item: item["id"]):
                packages = []
                for package in sorted(model["packages"], key=lambda item: item["id"]):
                    if profile["id"] not in package["build_profiles"]:
                        continue
                    native = components[package["components"][0]]["interface"]["format"] == "huggingface"
                    packages.append({
                        "id": package["id"], "component_ids": sorted(package["components"]),
                        "layout": "directory" if native else "archive", "format": "llima" if native else "zip",
                    })
                requirements[profile["id"]] = {
                    "version": 1, "compiler_version": resolve_model_compiler(root)["version"],
                    "invocations": [
                        {"id": item["compile_invocation_id"], "component_ids": sorted(item["component_ids"]),
                         "compile_fingerprint": item["compile_fingerprint"]}
                        for item in matrix if item["model_id"] == model_id
                        and item["build_profile_id"] == profile["id"]
                    ],
                    "packages": packages,
                }
            entry["build_requirements"] = requirements
        if documentation_receipts is not None:
            documentation = documentation_receipts.get(model_id)
            if not isinstance(documentation, dict):
                raise ValidationError(f"model documentation receipt is missing: {model_id}")
            entry["documentation"] = documentation
        model_entries.append(entry)
    # Unselected conversions retain their complete published model entry, including
    # package links and on-demand requirements. New ones wait for a selected plan.
    for model in (previous or {}).get("models", []):
        if model["id"] not in deferred:
            continue
        entry = dict(model)
        # Categories follow this checkout's taxonomy, which may have renamed IDs
        # since the carried entry was published.
        entry["categories"] = validate_model_categorization(all_models[model["id"]], taxonomy)
        if documentation_receipts is not None:
            documentation = documentation_receipts.get(model["id"])
            if not isinstance(documentation, dict):
                raise ValidationError(f"model documentation receipt is missing: {model['id']}")
            # Documentation must belong to the new catalog revision even when
            # the model's conversion/build artifacts are carried forward.
            entry["documentation"] = documentation
        model_entries.append(entry)
    model_entries.sort(key=lambda model: model["id"])
    pending_builds.extend(item for item in (previous or {}).get("pending_builds", []) if item["model_id"] in deferred)
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


def _previous_documentation_object(
    read: ReadRequest, read_api_url: str, run_id: str, bucket: str, content: JsonObject, branch_key: str,
) -> tuple[str, str] | None:
    artifact_name = content["artifact_name"]
    try:
        result = read(f"{read_api_url.rstrip('/')}/v1/runs/{quote(run_id, safe='')}/artifacts/{artifact_name}")
    except ModelRegistryError as exc:
        if any(f"HTTP {status} " in str(exc) for status in (401, 403, 404, 410)):
            return None
        raise
    artifact = result.get("artifact")
    if not isinstance(artifact, dict):
        return None
    key, version = artifact.get("s3_key"), artifact.get("s3_version_id")
    if (
        artifact.get("s3_bucket") == bucket
        and isinstance(key, str) and key.startswith(f"models/{branch_key}/")
        and isinstance(version, str) and version
        and artifact.get("sha256") == content["artifact_sha256"]
        and artifact.get("size_bytes") == content["size_bytes"]
    ):
        return key, version
    return None


def _store_documentation_file(
    store: S3ArtifactStore, path: Path, content: JsonObject, prior: JsonObject, branch_key: str,
    model_id: str, publication_commit: str | None, resolve_previous: Callable[[JsonObject], tuple[str, str] | None],
) -> S3Object:
    digest = content["artifact_sha256"]
    key = f"models/{branch_key}/documentation/{model_id}/{digest}/{path.name}"
    options = {"media_type": content["media_type"]}
    if (publication_commit and prior.get("artifact_sha256") == digest
            and prior.get("size_bytes") == content["size_bytes"]):
        previous_object = resolve_previous(content)
        if previous_object is None:
            # Older publishers keyed objects by the original publication commit.
            previous_object = (f"models/{branch_key}/{publication_commit}/documentation/{model_id}/{path.name}", None)
        options["previous_object"] = previous_object
    return store.ensure_native_sha256(path, key, digest, **options)


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
    read_api_url: str | None = None,
    read_request: ReadRequest | None = None,
    previous_receipts: dict[str, JsonObject] | None = None,
) -> dict[str, JsonObject]:
    root = Path(repository).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_sha):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip() or not environment.strip():
        raise ValidationError("branch and environment must not be empty")
    models = load_models(root)
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    read = read_request or _public_registry_json
    store = artifact_store
    previous_receipts = previous_receipts or {}
    git_sha = git_sha.lower()
    branch_key = quote(branch, safe="")
    branch_identity = hashlib.sha256(branch.encode()).hexdigest()[:16]
    receipts: dict[str, JsonObject] = {}
    for model_id, model in sorted(models.items()):
        prior = previous_receipts.get(model_id, {})
        if prior.get("source_commit") == git_sha:
            # Incremental catalog updates must preserve the current revision's receipt verbatim.
            receipts[model_id] = prior
            continue
        documentation = load_model_documentation(root / "models" / model_id, model["display_name"], repository=root,
            is_genai=any(c["interface"]["format"] == "huggingface" for c in model["components"]))
        hero = documentation["hero"]
        hero_content = {
            "artifact_name": hero["artifact_name"],
            "markdown_path": hero["markdown_path"],
            "alt_text": hero["alt_text"],
            "artifact_sha256": hero["sha256"],
            "size_bytes": hero["path"].stat().st_size,
            "media_type": hero["media_type"],
            "source_path": hero["source_path"],
        }
        content = {
            "format": "markdown",
            "overview_markdown": documentation["overview_markdown"],
            "artifact_name": "readme",
            "artifact_sha256": documentation["readme_sha256"],
            "size_bytes": documentation["path"].stat().st_size,
            "media_type": "text/markdown; charset=utf-8",
            "source_path": documentation["source_path"],
            "hero": hero_content,
        }
        original_commit = prior.get("publication_commit", prior.get("source_commit"))
        if not isinstance(original_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", original_commit):
            original_commit = None
        if original_commit and isinstance(prior.get("run_id"), str) and all(
            prior.get(field) == value for field, value in content.items()
        ):
            receipts[model_id] = {
                **content,
                "run_id": prior["run_id"],
                "source_commit": git_sha,
                "publication_commit": original_commit,
            }
            continue

        if store is None:
            import boto3

            store = S3ArtifactStore(bucket, kms_key_id=kms_key_id, s3_client=boto3.client("s3", region_name=aws_region))
        def previous_object(file_content: JsonObject) -> tuple[str, str] | None:
            if not read_api_url or not isinstance(prior.get("run_id"), str):
                return None
            return _previous_documentation_object(
                read, read_api_url, prior["run_id"], bucket, file_content, branch_key,
            )

        stored = _store_documentation_file(
            store, documentation["path"], content, prior, branch_key, model_id, original_commit,
            previous_object,
        )
        prior_hero = prior.get("hero")
        stored_hero = _store_documentation_file(
            store, hero["path"], hero_content, prior_hero if isinstance(prior_hero, dict) else {},
            branch_key, model_id, original_commit, previous_object,
        )
        parameters = {
            "schema_version": 2,
            "model_id": model_id,
            "source_path": content["source_path"],
            "source_commit": git_sha,
            "readme_sha256": stored.sha256,
            "overview_markdown": content["overview_markdown"],
            "hero": {
                "source_path": hero_content["source_path"],
                "markdown_path": hero_content["markdown_path"],
                "alt_text": hero_content["alt_text"],
                "sha256": stored_hero.sha256,
                "size_bytes": stored_hero.size_bytes,
                "media_type": hero_content["media_type"],
            },
        }
        run_fingerprint = canonical_sha256(parameters)
        environment_document = {
            "vulcan_environment": environment,
            "target_platform": "documentation",
            "publisher": "model-registry",
        }
        create_body = {
            "run_type": "model_documentation",
            "commit_sha": git_sha,
            "git_ref": f"refs/heads/{branch}",
            "ci_provider": "github-actions",
            "ci_run_url": github_run_url,
            "run_fingerprint": run_fingerprint,
            "environment": {
                "fingerprint": canonical_sha256(environment_document),
                "toolchain_name": "sima-model-registry",
                "toolchain_version": "1",
                "target_platform": "documentation",
                "properties": {"vulcan_environment": environment},
            },
            "metadata": {
                "model_id": model_id,
                "artifact_kind": "model-readme",
                "source_path": documentation["source_path"],
                "hero_source_path": hero["source_path"],
                **({"build_on_demand": True} if model.get("build-on-demand", False) else {}),
            },
        }
        invocation_identity = canonical_sha256(create_body)[:16]
        stable_key = (
            f"model-documentation-{branch_identity}-{git_sha}-{model_id}-"
            f"{run_fingerprint[:16]}-{invocation_identity}"
        )
        create = client.request("POST", "/v1/runs", create_body, f"{stable_key}-create")
        run_id = _require_text(create.get("run_id"), "run_id")
        client.request("PUT", f"/v1/runs/{run_id}/parameters", {
            "parameters": parameters,
            "canonical_sha256": canonical_sha256(parameters),
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
                "source_commit": git_sha,
            },
        }, f"{stable_key}-artifact")
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
                "source_commit": git_sha,
            },
        }, f"{stable_key}-hero-artifact")
        client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable_key}-finalize")
        receipts[model_id] = {**content, "run_id": run_id, "source_commit": git_sha}
    return receipts


def _remaining_pending_builds(prior: JsonObject, document: JsonObject) -> list[JsonObject]:
    """The prior catalog's pending builds this snapshot does not publish, in the prior order.

    A progress snapshot may only clear the pending entries whose builds it carries. The
    registry rejects every other change to the list: a different order, or entries a base
    branch started pending after this revision was initialized, which the inherited
    carry above would otherwise add or reorder. Entries the snapshot planned but the
    revision never marked pending cannot be added either; they are logged and left out,
    and their builds need a new Git revision.
    """
    builds = {
        (model["id"], build["build_profile_id"], build["component_id"]): build.get("compile_fingerprint")
        for model in document.get("models", [])
        for build in model.get("builds", [])
    }
    remaining = []
    for pending in prior.get("pending_builds", []):
        published = [
            builds.get((pending["model_id"], pending["build_profile_id"], component)) == pending["compile_fingerprint"]
            for component in pending["component_ids"]
        ]
        if not all(published):
            remaining.append(pending)
    def key(item):
        return item["model_id"], item["build_profile_id"], item["compile_fingerprint"]

    known = {key(item) for item in prior["pending_builds"]}
    for item in document.get("pending_builds", []):
        if key(item) not in known:
            logging.warning(
                "%s/%s is not pending in this catalog revision; publish its build from a new Git revision",
                item["model_id"], item["build_profile_id"],
            )
    return remaining


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
    force_rebuild_all: bool = False,
    base_branches: list[str] | None = None,
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
        record, document = read_current_catalog(read, read_api_url, branch)
        catalog_models = document.get("models", []) if document else []
        previous_receipts = {
            model["id"]: model["documentation"]
            for model in catalog_models
            if isinstance(model.get("documentation"), dict)
        }
        if (record and record.get("catalog_revision") == git_sha.lower()
                and previous_receipts and len(previous_receipts) == len(catalog_models)):
            documentation_receipts = previous_receipts
        else:
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
                read_api_url=read_api_url,
                read_request=read,
                previous_receipts=previous_receipts,
            )
    # Read the catalog last so another publisher has the least time to replace it
    # before the compare-and-swap POST below.
    root = Path(repository).expanduser().resolve()
    target_catalog, sources = read_catalog_sources(read, read_api_url, branch, base_branches)
    target_revision = target_catalog.get("catalog_revision") if isinstance(target_catalog, dict) else None
    continuing_revision = target_revision == git_sha.lower()
    if continuing_revision and documentation_receipts is not None and sources:
        # A concurrent publisher may have initialized this revision while we were preparing it.
        documentation_receipts.update({
            model["id"]: model["documentation"]
            for model in sources[0].get("models", [])
            if isinstance(model.get("documentation"), dict)
        })
    if continuing_revision and force_rebuild_all:
        selected = _selected_invocation_keys(
            Path(selected_matrix).expanduser().resolve() if selected_matrix else None
        )
        selected_profiles = {(model_id, profile_id) for model_id, profile_id, _ in selected}
        published_profiles = {
            (model["id"], package["build_profile_id"])
            for model in target_catalog["document"].get("models", [])
            for package in model.get("packages", [])
            if not model.get("build_on_demand", False)
        }
        conflicts = sorted(selected_profiles & published_profiles)
        if conflicts:
            profiles = ", ".join(f"{model}/{profile}" for model, profile in conflicts)
            raise ValidationError(
                f"cannot force rebuild published packages in current catalog revision {git_sha}: {profiles}. "
                "Publish the forced rebuild from a new Git revision; packages are immutable within a revision. "
                "For an initialization retry, omit --force-rebuild-all."
            )
    # Base branches seed a new revision only. During progress publication they
    # could restore a deliberately withheld package, or replace a published one
    # when a base branch finishes rebuilding before this branch does.
    models = load_models(root)
    # Apply the same revision boundary when carrying deferred conversion entries.
    sources = sources[:1] if continuing_revision else sources
    previous = merge_catalog_sources(sources, _head_matrix(root, models))
    deferred = set(models) - set(_catalog_models(models))
    if deferred:
        previous = previous or {"models": []}
        for source in sources:
            for model in source.get("models", []):
                if model["id"] in deferred:
                    # The registry fills in this ref's own approved and branch deployments and rejects a
                    # snapshot that carries another ref's, such as a base branch's promoted builds.
                    previous["models"].append(
                        {key: value for key, value in model.items() if key not in SERVER_OWNED_SELECTIONS}
                    )
                    previous.setdefault("pending_builds", []).extend(
                        item for item in source.get("pending_builds", []) if item["model_id"] == model["id"]
                    )
                    deferred.remove(model["id"])
    document = build_catalog_snapshot(
        repository=repository,
        branch=branch,
        git_sha=git_sha,
        compiled_dir=compiled_dir,
        package_dir=package_dir,
        selected_matrix=None if continuing_revision else selected_matrix,
        previous=previous,
        incremental=True,
        documentation_receipts=documentation_receipts,
    )
    _refresh_approved_benchmarks(
        root, document, target_catalog.get("document", {}) if target_catalog else {}, read_api_url, read,
    )
    if continuing_revision and isinstance(target_catalog, dict) and "pending_builds" in target_catalog["document"]:
        document["pending_builds"] = _remaining_pending_builds(target_catalog["document"], document)
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


def _download_registry_file(url: str, destination: Path, artifact: JsonObject,
                            download: DownloadRequest | None = None) -> None:
    # An orchestrator run stages every build's benchmark at once, so retry a dropped or throttled download.
    for attempt in range(1, 6):
        try:
            return _download_registry_file_once(url, destination, artifact, download)
        except RegistryPublicationError as exc:
            cause = exc.__cause__
            if isinstance(cause, HTTPError) and cause.code not in {429, 502, 503, 504} or attempt == 5:
                raise
        time.sleep(2 ** (attempt - 1) + random.uniform(0, 1))


def _download_registry_file_once(url: str, destination: Path, artifact: JsonObject,
                                 download: DownloadRequest | None = None) -> None:
    """Stream into a temporary file and install only after size and checksum verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            if download is not None:
                payload = download(url)
                output.write(payload)
                digest.update(payload)
                size = len(payload)
            else:
                request = UrlRequest(url, headers={"User-Agent": "sima-neat-model-registry/0.1"})
                with urlopen(request, timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > artifact["size_bytes"]:
                            raise ValidationError("downloaded artifact exceeds its declared size")
                        output.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != artifact["sha256"] or size != artifact["size_bytes"]:
            raise ValidationError("downloaded artifact failed size or SHA-256 verification")
        temporary.replace(destination)
    except OSError as exc:
        raise RegistryPublicationError(f"could not download registry artifact: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    expected_compile_fingerprint: str | None = None,
    deployment_branch: str | None = None,
) -> JsonObject:
    """Restore a complete invocation from durable registry MPKs for later stages."""
    root = Path(repository).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    read = read_request or _public_registry_json
    base_url = api_url.rstrip("/")
    catalog_branch = deployment_branch or branch
    catalog_payload = read(
        f"{base_url}/v1/repositories/sima-neat/models/catalogs/current?"
        f"git_ref={quote(f'refs/heads/{catalog_branch}', safe='')}"
    )
    catalog_record = catalog_payload.get("catalog")
    document = catalog_record.get("document") if isinstance(catalog_record, dict) else None
    if not isinstance(document, dict) or (
        catalog_branch == branch and document.get("catalog_revision") != catalog_revision
    ):
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
    if catalog_branch != branch and expected_compile_fingerprint is None:
        expected_compile_fingerprint = compile_fingerprint(
            root, model_id, build_profile_id, compile_invocation_id, resolve_model_compiler(root)["identity"]
        )
    catalog_builds, catalog_packages = _catalog_outputs(catalog_model)
    builds = {
        item["component_id"]: item
        for item in catalog_builds
        if isinstance(item, dict)
        and item.get("build_profile_id") == build_profile_id
        and item.get("compile_invocation_id") == compile_invocation_id
    }
    if set(builds) != set(invocation["component_ids"]):
        raise ValidationError("current catalog does not link every component in the compile invocation")
    if expected_compile_fingerprint and any(
        build.get("compile_fingerprint") != expected_compile_fingerprint for build in builds.values()
    ):
        raise ValidationError("benchmark needs a registered build matching its current model/compiler")

    staged_outputs = []
    for component_id in invocation["component_ids"]:
        build = builds[component_id]
        run_id = build.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValidationError(f"current catalog build has no retained compile run: {component_id}")
        component = next(item for item in model["components"] if item["id"] == component_id)
        native = component["interface"]["format"] == "huggingface"
        kind = "llima" if native else "mpk"
        artifact_payload = read(f"{base_url}/v1/runs/{run_id}/artifacts/{kind}")
        artifact = artifact_payload.get("artifact")
        if not isinstance(artifact, dict):
            raise ValidationError(f"artifact registry returned no MPK metadata for {run_id}")
        digest = artifact.get("sha256")
        download_url = artifact.get("download_url")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(download_url, str):
            raise ValidationError(f"artifact registry returned incomplete MPK metadata for {run_id}")
        if digest != build.get("artifact_sha256", digest):
            raise ValidationError(f"registry MPK checksum does not match the current catalog for {run_id}")
        component_dir = output / model_id / build_profile_id / component_id
        component_dir.mkdir(parents=True, exist_ok=True)
        archive_name = "deployment-manifest.json" if native else mpk_filename(model, build_profile_id, component_id)
        archive = component_dir / archive_name
        _download_registry_file(download_url, archive, artifact, download_request)
        if native:
            try:
                manifest = json.loads(archive.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValidationError(f"invalid deployment manifest for {run_id}: {exc}") from exc
            if not isinstance(manifest, dict) or manifest.get("format") != "llima":
                raise ValidationError(f"invalid LLiMa deployment manifest for {run_id}")
            files = deployment_files(manifest)
            deployment = component_dir / "deployment"
            deployment.mkdir(exist_ok=True)

            def fetch_file(item: JsonObject) -> None:
                response = read(f"{base_url}/v1/runs/{run_id}/artifacts/{item['artifact']}")
                registered = response.get("artifact", {})
                if (registered.get("sha256") != item["sha256"]
                        or registered.get("size_bytes") != item["size_bytes"]
                        or not isinstance(registered.get("download_url"), str)):
                    raise ValidationError(f"inconsistent deployment artifact: {item['path']}")
                path = deployment / item["path"]
                if any(parent.is_symlink() for parent in (path, *path.parents)):
                    raise ValidationError(f"unsafe deployment output path: {path}")
                if (path.is_file() and path.stat().st_size == item["size_bytes"]
                        and sha256_file(path)[0] == item["sha256"]):
                    return
                _download_registry_file(registered["download_url"], path, registered, download_request)

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(fetch_file, sorted(files, key=lambda item: item["size_bytes"], reverse=True)))
            declared = {item["path"] for item in files}
            for path in deployment.rglob("*"):
                if path.is_symlink():
                    raise ValidationError(f"unsafe deployment output path: {path}")
                if path.is_file() and path.relative_to(deployment).as_posix() not in declared:
                    path.unlink()
        verified = verify_artifact(component_dir)
        run_payload = read(f"{base_url}/v1/runs/{run_id}")
        run = run_payload.get("run", {}) if isinstance(run_payload, dict) else {}
        compiler_version = run.get("toolchain_version") if isinstance(run, dict) else None
        compile_result = {**verified, "provenance": {"compiler_version": compiler_version}}
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
                "artifact_path" if native else "mpk_path": f"{component_id}/{archive_name}",
                "artifact_sha256" if native else "mpk_sha256": digest,
                "run_id": run_id,
            }
        )
    profile_dir = output / model_id / build_profile_id
    runtime_files = []
    package = next((
        item for item in catalog_packages
        if isinstance(item, dict) and (item.get("build_profile_id"), item.get("package_id")) == (
            build_profile_id, "default"
        )
    ), None)
    if package is not None:
        response = read(f"{base_url}/v1/runs/{package['run_id']}/artifacts/package")
        artifact = response.get("artifact")
        if not isinstance(artifact, dict):
            raise ValidationError("artifact registry returned no package metadata")
        metadata = artifact.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValidationError("artifact registry returned invalid package metadata")
        if "runtime_files" in metadata:
            component_runs = {item["component_id"]: item["run_id"] for item in staged_outputs}
            if (artifact.get("sha256") != package.get("artifact_sha256", artifact.get("sha256"))
                    or artifact.get("size_bytes") != package.get("size_bytes", artifact.get("size_bytes"))
                    or not isinstance(artifact.get("download_url"), str)
                    or metadata.get("component_run_ids") != component_runs
                    or package.get("component_run_ids") != component_runs):
                raise ValidationError("runtime-file package does not match the current catalog build")
            with tempfile.TemporaryDirectory(prefix="restore-package-", dir=profile_dir) as temporary:
                archive = Path(temporary) / "package.zip"
                _download_registry_file(artifact["download_url"], archive, artifact, download_request)
                runtime_files = metadata["runtime_files"]
                restore_runtime_files(archive, runtime_files, profile_dir)
    # The same manifest a compile writes, so later stages read restored and fresh builds alike.
    (profile_dir / "compile-outputs.json").write_text(
        json.dumps({
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "compile_invocation_id": compile_invocation_id,
            "outputs": [
                {key: item[key] for key in ("component_id", *compile_output_fields(item))} for item in staged_outputs
            ],
            **({"runtime_files": runtime_files} if runtime_files else {}),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # A benchmark runs under the profile's package run, as after a fresh build's publish-package.
    if package is not None:
        receipt = profile_dir / "packages" / "default" / "registry-publication.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps({"packages": [{"model_id": model_id, **package}]}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {"count": len(staged_outputs), "outputs": staged_outputs}



def stage_benchmark_baseline(
    *, repository: str | Path, output_dir: str | Path, model_id: str, build_profile_id: str, **registry_options
) -> JsonObject:
    """Restore the declared ordinary deployment and retain its compile identities beside the benchmark input."""
    root = Path(repository).resolve()
    output = Path(output_dir).resolve()
    receipt = output / "benchmark-baseline.json"
    receipt.unlink(missing_ok=True)
    model = load_and_validate_model(root / "models" / model_id)
    baseline = build_profile_for(model, build_profile_id).get("benchmark_baseline")
    if not baseline:
        return {"count": 0}
    reference = load_and_validate_model(root / "models" / baseline["model_id"])
    profile = baseline["build_profile"]
    invocation = build_matrix({reference["id"]: reference}, [reference["id"]], [profile],
        repository=root, compiler_version=resolve_model_compiler(root)["identity"])["include"][0]
    if reference["components"][0]["interface"]["format"] != "huggingface":
        raise ValidationError("a speculative benchmark baseline must be a native LLiMa deployment")
    result = stage_catalog_build(repository=root, output_dir=output, model_id=reference["id"],
        build_profile_id=profile, compile_invocation_id=invocation["compile_invocation_id"],
        expected_compile_fingerprint=invocation["compile_fingerprint"], **registry_options)
    component = result["outputs"][0]
    identity = {"model_id": reference["id"], "display_name": reference["display_name"], "build_profile_id": profile,
                "deployment": f"{reference['id']}/{profile}/{component['component_id']}/deployment",
                "compile_run_id": component["run_id"], "artifact_sha256": component["artifact_sha256"]}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(identity, indent=2) + "\n")
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
    config: JsonObject,
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
    # The schema fixes the format; these fields must name the build and match model.yaml's numerical_fidelity.
    expected = {
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "status": config["status"],
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": config["selection"],
        "samples": config["selection"]["samples"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValidationError(f"{path}: numerical fidelity {field} must be {value!r}")
    names = [metric["name"] for metric in report["metrics"]]
    if sorted(names) != sorted(config["outputs"]):
        raise ValidationError(f"{path}: numerical fidelity metrics must cover exactly {config['outputs']}")
    for metric in report["metrics"]:
        values = [metric[field] for field in ("cosine_similarity", "normalized_rmse", "mean_absolute_error",
                                               "max_absolute_error")]
        if not all(math.isfinite(value) for value in values):
            raise ValidationError(f"{path}: numerical fidelity {metric['name']} must be finite")
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
                "output": metric["name"],
            },
        }
        for index, metric in enumerate(report["metrics"])
        for field, name, unit, direction in fields
    ]


def _publish_fidelity(
    client: SigV4RegistryClient,
    store: S3ArtifactStore,
    path: Path,
    fidelity: JsonObject,
    *,
    run_id: str,
    stable_key: str,
    bucket: str,
    retention_class: str,
    s3_key: str,
) -> None:
    """Attach a profile's checked numerical fidelity to its package run, the whole-model build it compares."""
    fingerprint = canonical_sha256(fidelity)
    stored = store.ensure_native_sha256(path, s3_key, sha256_file(path)[0], media_type="application/json")
    artifact = client.request(
        "PUT",
        f"/v1/runs/{run_id}/artifacts/numerical-fidelity",
        {
            "artifact_type": "numerical-fidelity",
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": "application/json",
            "retention_class": retention_class,
            "metadata": {key: fidelity[key] for key in ("model_id", "build_profile_id", "samples")},
        },
        f"{stable_key}-fidelity-artifact",
    )
    test: JsonObject = {"status": "passed", "summary": fidelity}
    if artifact.get("artifact_id"):
        test["report_artifact_id"] = artifact["artifact_id"]
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/test-results/correctness/numerical-fidelity",
        test,
        f"{stable_key}-fidelity-test-result",
    )
    client.request(
        "POST",
        f"/v1/runs/{run_id}/measurements/batch",
        {"measurements": _fidelity_measurements(fidelity, fingerprint)},
        f"{stable_key}-fidelity-measurements",
    )


def _mla_stage_simulation_result(path: Path, schema_path: Path, *, model_id: str, build_profile_id: str) -> JsonObject:
    try:
        simulation = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: could not read MLA stage simulation: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(simulation),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ValidationError(f"{path}: MLA stage simulation {location}: {first.message}")
    for field, value in {"model_id": model_id, "build_profile_id": build_profile_id}.items():
        if simulation[field] != value:
            raise ValidationError(f"{path}: MLA stage simulation {field} must be {value!r}")
    return simulation


def _publish_mla_stage_simulation(
    client: SigV4RegistryClient,
    store: S3ArtifactStore,
    path: Path,
    simulation: JsonObject,
    *,
    run_id: str,
    stable_key: str,
    bucket: str,
    retention_class: str,
    s3_key: str,
) -> None:
    """Attach a profile's MLA stage simulation to its package run, where the model browser reads it."""
    stored = store.ensure_native_sha256(path, s3_key, sha256_file(path)[0], media_type="application/json")
    artifact = client.request(
        "PUT",
        f"/v1/runs/{run_id}/artifacts/mla-stage-simulation",
        {
            "artifact_type": "mla-stage-simulation",
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": "application/json",
            "retention_class": retention_class,
            "metadata": {key: simulation[key] for key in ("model_id", "build_profile_id")},
        },
        f"{stable_key}-mla-stage-simulation-artifact",
    )
    # A component mla-msim could not simulate leaves nothing to compare, which is not a failure of the model.
    simulated = all(component["status"] == "ok" for component in simulation["components"].values())
    test: JsonObject = {"status": "passed" if simulated else "skipped", "summary": simulation}
    if artifact.get("artifact_id"):
        test["report_artifact_id"] = artifact["artifact_id"]
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/test-results/performance/mla-stage-simulation",
        test,
        f"{stable_key}-mla-stage-simulation-test-result",
    )


def _compile_timing(path: Path) -> JsonObject:
    """Read the recipe timing the compile runner writes; registry run timestamps measure publication."""
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
        from .conversion import apply_selection

        apply_selection(model, root=root)
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
        from .conversion import apply_selection

        apply_selection(model, root=root)
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

        native = component["interface"]["format"] == "huggingface"
        archive_name = _require_text(result.get("manifest" if native else "archive"), "compiled artifact")
        if not native:
            expected_name = mpk_filename(model, build_profile_id, component_id)
            if archive_name != expected_name:
                raise ValidationError(
                    f"{result_path}: compiled MPK must be named {expected_name}, found {archive_name}"
                )
        archive = result_path.parent / archive_name
        if not archive.is_file() or archive.parent != result_path.parent:
            raise ValidationError(f"{result_path}: archive is missing from the packaged output")
        if result.get("format", "mpk") != ("llima" if native else "mpk"):
            raise ValidationError(f"{result_path}: artifact format does not match its source interface")
        artifact_sha = _require_sha256(result.get("sha256"), "sha256")
        declared_output = output_by_result[result_path]
        if (
            archive.resolve() != Path(declared_output[compile_output_fields(declared_output)[0]]).resolve()
            or artifact_sha != declared_output[compile_output_fields(declared_output)[1]]
        ):
            raise ValidationError(f"{result_path}: archive does not match the compile output manifest")
        actual_sha, actual_size = sha256_file(archive)
        files = deployment_files(json.loads(archive.read_text(encoding="utf-8"))) if native else []
        if native:
            actual_size = sum(item["size_bytes"] for item in files)
        if artifact_sha != actual_sha or int(result.get("size_bytes", -1)) != actual_size:
            raise ValidationError(f"{result_path}: archive checksum or size does not match")

        compiled_parameters = {}
        if native:
            compiled_parameters = llima_compiled_parameters(archive.parent / "deployment")
            for name in ("max_num_tokens", "language_group_size", "input_height", "input_width", "speculative_method"):
                if name in binding["parameters"] and compiled_parameters.get(name) != binding["parameters"][name]:
                    raise ValidationError(f"{result_path}: compiled {name} does not match model.yaml")

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
        recorded_fingerprint = _require_sha256(provenance.get("compile_fingerprint"), "compile_fingerprint")
        expected_fingerprint = compile_fingerprint(
            root, model_id, build_profile_id, invocation_by_result[result_path], compiler_identity
        )
        if recorded_fingerprint != expected_fingerprint:
            raise ValidationError(f"{result_path}: compile_fingerprint does not match semantic compile inputs")
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
                "compiled_parameters": compiled_parameters,
                "archive": archive,
                "files": files,
                "artifact_sha": artifact_sha,
                "compiler_version": compiler_version,
                "compiler_label": compiler_label,
                "llima_version": provenance.get("llima_version"),
                "runner_provenance": runner_provenance(provenance),
                "target": target,
                "source_sha": source_sha,
                "compile_fingerprint": recorded_fingerprint,
            }
        )
    return prepared


def _conversion_input(source, source_metadata):
    from .conversion import input_record

    record = input_record(source)
    return {**record, "metadata": {**source_metadata, **record["metadata"]}}


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
    candidate: JsonObject | None = None,
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
    if any(item["model"].get("build-on-demand", False) for item in compilations) and candidate is None:
        raise ValidationError("on-demand publication requires publish-candidate")
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
        files = compilation["files"]
        native = bool(files)
        on_demand = model.get("build-on-demand", False)
        retention_class = "on-demand" if on_demand else "branch"
        binding = compilation["binding"]
        archive = compilation["archive"]
        artifact_sha = compilation["artifact_sha"]
        compiler_version = compilation["compiler_version"]
        compiler_label = compilation["compiler_label"]
        target = compilation["target"]
        source_sha = compilation["source_sha"]
        parameters = {
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "compile_invocation_id": compile_invocation_id,
            "component_id": component_id,
            "target": target,
            "compiler_version": compiler_version,
            "compiler_label": compiler_label,
            "source_sha256": source_sha,
            # Vulcan promotion reads the compile fingerprint under this parameter name.
            "build_fingerprint": compilation["compile_fingerprint"],
            "recipe": component["recipe"],
            "compile_parameters": binding.get("parameters", {}),
            **({"compiled_parameters": compilation["compiled_parameters"]} if native else {}),
            "calibration": binding.get("calibration"),
            **({"llima_version": compilation["llima_version"]} if compilation["llima_version"] else {}),
        }
        environment_document = {
            "vulcan_environment": environment,
            "toolchain_name": "sima-model-compiler",
            "toolchain_version": compiler_version,
            "compiler_label": compiler_label,
            "target_platform": target,
        }
        run_fingerprint = canonical_sha256({**parameters, "artifact_sha256": artifact_sha})
        # Runner allocation is audit metadata, not a compile/run identity input.
        parameters.update(compilation["runner_provenance"])
        environment_fingerprint = canonical_sha256(environment_document)
        # A new attempt of the same workflow run resumes the same registry
        # transaction instead of creating a duplicate record per attempt.
        key_digest = hashlib.sha256(
            (f"{github_run_id}:{environment}:{branch}:{git_sha}:{model_id}:"
             f"{build_profile_id}:{component_id}:{run_fingerprint}"
             + (f":{candidate['deployment_id']}" if candidate else "")).encode()
        ).hexdigest()
        stable_key = f"models-{key_digest}"
        create = client.request(
            "POST",
            "/v1/runs",
            {
                "run_type": "model_compile",
                "commit_sha": git_sha,
                "git_ref": f"refs/heads/{branch}",
                "ci_provider": "github-actions",
                "ci_run_url": github_run_url,
                "run_fingerprint": run_fingerprint,
                "environment": {
                    "fingerprint": environment_fingerprint,
                    "toolchain_name": "sima-model-compiler",
                    "toolchain_version": compiler_version,
                    "target_platform": target,
                    "properties": {"compiler_label": compiler_label, "vulcan_environment": environment},
                },
                "metadata": {
                    "model_id": model_id,
                    "build_profile_id": build_profile_id,
                    "component_id": component_id,
                    "artifact_kind": "llima" if native else "mpk",
                    **({"build_on_demand": True} if on_demand else {}),
                    "github_run_id": github_run_id,
                    "compile_timing": compilation["compile_timing"],
                    "license": model["license"],
                },
            },
            f"{stable_key}-create",
        )
        run_id = _require_text(create.get("run_id"), "run_id")
        if candidate is not None:
            deployment_path = f"/v1/repositories/sima-neat/models/deployments/{candidate['deployment_id']}"
            client.request("PUT", f"{deployment_path}/runs/{run_id}",
                           {"role": f"compile-{component_id}"}, f"{stable_key}-bind")
        source = component["source"]
        source_metadata = {"model_id": model_id, "component_id": component_id, "source_type": source["type"]}
        if native:
            source_metadata.update({
                "revision": model["upstream"]["revision"], "entrypoint": source["entrypoint"],
                "files": source["files"],
            })
        model_card = model.get("upstream", {}).get("model_card")
        if isinstance(model_card, str) and model_card:
            source_metadata["model_card"] = model_card
        source_reference = source.get("uri")
        if source["type"] == "conversion":
            source_reference = f"run:{source['run_id']}/artifacts/onnx"
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
                **(_conversion_input(source, source_metadata) if source["type"] == "conversion" else {}),
            },
            f"{stable_key}-input",
        )
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/parameters",
            {"parameters": parameters, "canonical_sha256": canonical_sha256(parameters)},
            f"{stable_key}-parameters",
        )

        artifact_metadata = {
            "model_id": model_id, "build_profile_id": build_profile_id,
            "component_id": component_id, "target": target,
            **({"build_on_demand": True} if on_demand else {}),
        }
        if native:
            _publish_deployment_files(
                client, store, run_id=run_id, stable_key=stable_key, bucket=bucket,
                deployment=archive.parent / "deployment", files=files,
                metadata=artifact_metadata, retention_class=retention_class,
                upload_prefix=candidate["upload_prefix"].rstrip("/") if candidate else None,
            )
            artifact_metadata.update({
                "layout": "directory", "storage_backend": "s3", "file_count": len(files),
                "deployment_size_bytes": sum(item["size_bytes"] for item in files),
            })
        relative_archive = archive.relative_to(dist).as_posix()
        object_key = f"models/{branch_key}/{commit_tag}/{relative_archive}"
        if candidate is not None:
            object_key = f"{candidate['upload_prefix'].rstrip('/')}/{run_id}/{archive.name}"
        stored = store.ensure_native_sha256(
            archive, object_key, artifact_sha,
            media_type="application/json" if native else "application/gzip",
        )
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/artifacts/{'llima' if native else 'mpk'}",
            {
                "artifact_type": "model-pack",
                "s3_bucket": bucket,
                "s3_key": stored.key,
                "artifact_sha256": artifact_sha,
                "s3_version_id": stored.version_id,
                "sha256": stored.sha256,
                "size_bytes": stored.size_bytes,
                "media_type": "application/json" if native else "application/gzip",
                "retention_class": retention_class,
                "metadata": artifact_metadata,
            },
            f"{stable_key}-artifact",
        )
        # This is compiler-output validation, not a DevKit correctness or performance test.
        validation = "deployment-validation" if native else "mpk-validation"
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/test-results/compile/{validation}",
            {
                "status": "passed",
                "summary": {"deployment_validated" if native else "archive_validated": True,
                            "checksum_verified": True},
            },
            f"{stable_key}-validation",
        )
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
                "compile_fingerprint": compilation["compile_fingerprint"],
                "artifact_sha256": artifact_sha,
                "s3_key": stored.key,
            }
        )

    return {
        "count": len(published),
        "runs": published,
    }


def _package_media_type(archive_name: str) -> str | None:
    """Return the media type of a ZIP package, or None for any other archive."""
    return "application/zip" if archive_name.endswith(".zip") else None


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
    candidate: JsonObject | None = None,
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
        native = result.get("layout") == "directory"
        expected_native = any(item["interface"]["format"] == "huggingface" for item in model["components"]
                              if item["id"] in declaration["components"])
        if native != expected_native:
            raise ValidationError(f"{result_path}: package layout does not match its component interfaces")
        on_demand = model.get("build-on-demand", False)
        if on_demand and candidate is None:
            raise ValidationError("on-demand publication requires publish-candidate")
        retention_class = "on-demand" if on_demand else "branch"
        archive_name = _require_text(result.get("manifest_path" if native else "archive"), "package artifact")
        archive = result_path.parent / archive_name
        media_type = "application/json" if native else _package_media_type(archive_name)
        expected = result_path.parent / "deployment/manifest.json" if native else result_path.parent / archive.name
        if (archive != expected or archive.is_symlink() or not archive.is_file()
                or media_type is None):
            raise ValidationError(f"{result_path}: package artifact is missing or unsafe")
        files = deployment_files(manifest) if native else []
        if native and (json.loads(archive.read_text(encoding="utf-8")) != manifest
                       or verify_llima(archive.parent, allow_manifest=True) != files):
            raise ValidationError(f"{result_path}: deployment files do not match the package manifest")
        artifact_sha, artifact_size = sha256_file(archive)
        package_size = sum(item["size_bytes"] for item in files) if native else artifact_size
        if result.get("sha256") != artifact_sha or result.get("size_bytes") != package_size:
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
        # Check the fidelity before any registry write, so an invalid report leaves no unfinished run.
        fidelity_path = result_path.parent / FIDELITY_FILE
        if fidelity_path.is_file() != ("numerical_fidelity" in model):
            expectation = "requires" if "numerical_fidelity" in model else "must not contain"
            raise ValidationError(f"{result_path.parent}: the package {expectation} {FIDELITY_FILE}")
        fidelity = _fidelity_result(
            fidelity_path,
            root / "schemas" / "numerical-fidelity.schema.json",
            model_id=model_id,
            build_profile_id=profile_id,
            config=model["numerical_fidelity"],
        ) if fidelity_path.is_file() else None
        simulation_path = result_path.parent / SIMULATION_FILE
        simulation = _mla_stage_simulation_result(
            simulation_path,
            root / "schemas" / "mla-stage-simulation.schema.json",
            model_id=model_id,
            build_profile_id=profile_id,
        ) if simulation_path.is_file() else None

        manifest_sha = canonical_sha256(manifest)
        parameters = {
            "model_id": model_id,
            "build_profile_id": profile_id,
            "package_id": package_id,
            "package_manifest_sha256": manifest_sha,
            "component_run_ids": component_runs,
        }
        run_fingerprint = canonical_sha256({**parameters, "artifact_sha256": artifact_sha})
        # Packages publish without an embedded manifest; retain its provenance
        # in the durable package run as well.
        parameters["provenance"] = manifest.get("provenance", {})
        environment_document = {
            "vulcan_environment": environment,
            "target_platform": manifest.get("target"), "packager": "model-registry",
        }
        stable_key = "model-package-" + hashlib.sha256(
            (f"{github_run_id}:{environment}:{branch}:{git_sha}:{model_id}:{profile_id}:{package_id}:{run_fingerprint}"
             + (f":{candidate['deployment_id']}" if candidate else "")).encode()
        ).hexdigest()
        create = client.request("POST", "/v1/runs", {
            "run_type": "model_package", "commit_sha": git_sha,
            "git_ref": f"refs/heads/{branch}", "ci_provider": "github-actions", "ci_run_url": github_run_url,
            "run_fingerprint": run_fingerprint,
            "environment": {
                "fingerprint": canonical_sha256(environment_document),
                "toolchain_name": "sima-model-registry", "toolchain_version": "1",
                "target_platform": manifest.get("target"),
                "properties": {"vulcan_environment": environment},
            },
            "metadata": {
                "model_id": model_id, "build_profile_id": profile_id, "package_id": package_id,
                "artifact_kind": "full-package", "github_run_id": github_run_id,
                "license": manifest["license"], "component_run_ids": component_runs,
                **({"build_on_demand": True} if on_demand else {}),
            },
        }, f"{stable_key}-create")
        run_id = _require_text(create.get("run_id"), "run_id")
        if candidate is not None:
            deployment_path = f"/v1/repositories/sima-neat/models/deployments/{candidate['deployment_id']}"
            client.request("PUT", f"{deployment_path}/runs/{run_id}",
                           {"role": f"package-{package_id}"}, f"{stable_key}-bind")
        for component_id, component_run_id in sorted(component_runs.items()):
            client.request("PUT", f"/v1/runs/{run_id}/inputs/component-{component_id}", {
                "input_type": "model-pack", "source_reference": f"registry-run:{component_run_id}",
                "sha256": next(item["sha256"] for item in components if item["id"] == component_id),
                "size_bytes": next(item["size_bytes"] for item in components if item["id"] == component_id),
                "media_type": "application/json" if native else "application/gzip",
                "metadata": {"component_id": component_id, "compile_run_id": component_run_id},
            }, f"{stable_key}-component-{component_id}")
        client.request("PUT", f"/v1/runs/{run_id}/parameters", {
            "parameters": parameters, "canonical_sha256": canonical_sha256(parameters),
        }, f"{stable_key}-parameters")
        artifact_metadata = {
            "model_id": model_id, "build_profile_id": profile_id, "package_id": package_id,
            "manifest_sha256": manifest_sha, "component_run_ids": component_runs,
            **({"build_on_demand": True} if on_demand else {}),
        }
        if not native:
            component_paths = {item["path"] for item in components}
            resource_paths = {path for resource in manifest["resources"] for path in resource["paths"]}
            runtime_files = [
                {"path": item["path"], "sha256": item["sha256"]}
                for item in manifest["files"]
                if item["path"] not in component_paths and item["path"] not in resource_paths
            ]
            if runtime_files:
                artifact_metadata["runtime_files"] = runtime_files
        if native:
            _publish_deployment_files(
                client, store, run_id=run_id, stable_key=stable_key, bucket=bucket,
                deployment=archive.parent, files=files, metadata=artifact_metadata,
                retention_class=retention_class,
                upload_prefix=candidate["upload_prefix"].rstrip("/") if candidate else None,
            )
            artifact_metadata.update({
                "layout": "directory", "storage_backend": "s3", "file_count": len(files),
                "deployment_size_bytes": package_size,
            })
        stored = store.ensure_native_sha256(
            archive,
            (f"{candidate['upload_prefix'].rstrip('/')}/{run_id}/{archive.name}" if candidate else
             f"models/{branch_key}/{commit_tag}/{archive.relative_to(packages_root).as_posix()}"),
            artifact_sha,
            media_type=media_type,
        )
        client.request("PUT", f"/v1/runs/{run_id}/artifacts/package", {
            "artifact_type": "model-package", "s3_bucket": bucket, "s3_key": stored.key,
            "s3_version_id": stored.version_id, "sha256": stored.sha256, "size_bytes": stored.size_bytes,
            "media_type": media_type, "retention_class": retention_class,
            "metadata": artifact_metadata,
        }, f"{stable_key}-artifact")
        if fidelity is not None:
            _publish_fidelity(
                client, store, fidelity_path, fidelity, run_id=run_id, stable_key=stable_key, bucket=bucket,
                retention_class=retention_class,
                s3_key=(f"{candidate['upload_prefix'].rstrip('/')}/{run_id}/numerical-fidelity.json" if candidate else
                        f"models/{branch_key}/{commit_tag}/{model_id}/{profile_id}/numerical-fidelity.json"),
            )
        if simulation is not None:
            _publish_mla_stage_simulation(
                client, store, simulation_path, simulation, run_id=run_id, stable_key=stable_key, bucket=bucket,
                retention_class=retention_class,
                s3_key=(f"{candidate['upload_prefix'].rstrip('/')}/{run_id}/{SIMULATION_FILE}" if candidate else
                        f"models/{branch_key}/{commit_tag}/{model_id}/{profile_id}/{SIMULATION_FILE}"),
            )
        client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable_key}-finalize")
        published.append({
            "model_id": model_id, "build_profile_id": profile_id, "package_id": package_id, "run_id": run_id,
            "artifact_sha256": artifact_sha, "size_bytes": package_size, "manifest_sha256": manifest_sha,
            "package_fingerprint": manifest_sha, "component_run_ids": component_runs, "s3_key": stored.key,
            **({key: artifact_metadata[key] for key in
                ("layout", "storage_backend", "file_count", "deployment_size_bytes")} if native else {}),
        })
    if not published:
        raise ValidationError("no assembled package results were found")
    return {"count": len(published), "packages": published}
