"""The registry's S3 bucket: each artifact is uploaded once with a native SHA-256 and recorded by exact version."""
from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .transport import JsonObject, RegistryRequestError, canonical_sha256


@dataclass(frozen=True)
class S3Object:
    """One exact version of an object in the registry's bucket."""

    key: str
    version_id: str
    sha256: str
    size_bytes: int


class S3ArtifactStore:
    """Uploads to the registry's bucket, through boto3 when given a client and the aws CLI otherwise."""

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
                    raise RegistryRequestError(f"saved upload differs from {key}; start a new candidate")
        head = self._head(key, saved["s3_version_id"]) if saved else self._head(key)
        size = path.stat().st_size
        if _matches_s3_object(head, expected_checksum, size):
            return self._record(S3Object(key, head["VersionId"], sha256, size))
        if saved:
            raise RegistryRequestError(
                f"saved S3 version is unavailable or changed for {key}; start a new candidate"
            )
        if previous_object:
            previous_key, previous_version = previous_object
            prior = self._head(previous_key, previous_version)
            if _matches_s3_object(prior, expected_checksum, size):
                return self._record(S3Object(previous_key, prior["VersionId"], sha256, size))

        if size > 5 * 1024**3:
            raise RegistryRequestError(f"registry artifact exceeds the 5 GiB single-upload limit: {path}")
        if self._s3 is None:
            version_id = self._put_cli(path, key, expected_checksum, media_type)
        else:
            version_id = self._put_sdk(path, key, expected_checksum, media_type)
        if not isinstance(version_id, str) or not version_id:
            raise RegistryRequestError(f"S3 versioning did not return a version for {key}")
        verified = self._head(key, version_id)
        if verified.get("ChecksumSHA256") != expected_checksum:
            raise RegistryRequestError(f"S3 native SHA-256 verification failed for {key}")
        if verified.get("ContentLength") != size:
            raise RegistryRequestError(f"S3 object size does not match the packaged artifact for {key}")
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
            raise RegistryRequestError(f"Could not upload s3://{self.bucket}/{key}: {completed.stderr.strip()}")
        try:
            uploaded = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RegistryRequestError(f"Could not parse S3 upload response for {key}") from exc
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
                raise RegistryRequestError(f"Could not publish immutable s3://{self.bucket}/{key}: {exc}") from exc
            except BotoCoreError as exc:
                raise RegistryRequestError(f"Could not upload s3://{self.bucket}/{key}: {exc}") from exc
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
                raise RegistryRequestError(f"Could not verify s3://{self.bucket}/{key}: {exc}") from exc
            except BotoCoreError as exc:
                raise RegistryRequestError(f"Could not verify s3://{self.bucket}/{key}: {exc}") from exc
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
            raise RegistryRequestError(f"S3 returned invalid metadata for {key}") from exc
        return value if isinstance(value, dict) else {}


def _matches_s3_object(head: JsonObject, checksum: str, size: int) -> bool:
    return (
        head.get("ChecksumSHA256") == checksum
        and head.get("ContentLength") == size
        and isinstance(head.get("VersionId"), str)
        and bool(head["VersionId"])
    )
