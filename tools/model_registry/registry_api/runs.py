"""Registry runs: create one, attach its inputs, parameters, artifacts, results, and measurements, then finalize it.

Each request of a run carries the idempotency key ``<stable_key>-<step>``. A publisher derives ``stable_key`` from
what it publishes and the workflow run, so a new attempt of the same workflow run resumes the same registry run.
"""
from __future__ import annotations

from dataclasses import dataclass

from .storage import S3Object
from .transport import JsonObject, RegistryRequestError, SigV4RegistryClient

# Repository-scoped resources (catalogs, resolve, deployments, promotions) live under this path.
REPOSITORY_PATH = "/v1/repositories/sima-neat/models"


def artifact_location(bucket: str, stored: S3Object) -> JsonObject:
    """The fields of an artifact record that name its exact S3 object version."""
    return {
        "s3_bucket": bucket,
        "s3_key": stored.key,
        "s3_version_id": stored.version_id,
        "sha256": stored.sha256,
        "size_bytes": stored.size_bytes,
    }


@dataclass(frozen=True)
class OpenRun:
    """A registry run between its creation and its finalize call."""

    client: SigV4RegistryClient
    run_id: str
    stable_key: str

    def put(self, resource: str, body: JsonObject, step: str) -> JsonObject:
        """Set ``resource`` under the run, such as ``parameters`` or ``inputs/<name>``."""
        return self.client.request("PUT", f"/v1/runs/{self.run_id}/{resource}", body, f"{self.stable_key}-{step}")

    def put_artifact(self, name: str, bucket: str, stored: S3Object, record: JsonObject, step: str) -> JsonObject:
        """Register the uploaded ``stored`` object as artifact ``name``; ``record`` holds its type and metadata."""
        return self.put(f"artifacts/{name}", {**record, **artifact_location(bucket, stored)}, step)

    def add_measurements(self, measurements: list[JsonObject], step: str) -> JsonObject:
        return self.client.request(
            "POST", f"/v1/runs/{self.run_id}/measurements/batch", {"measurements": measurements},
            f"{self.stable_key}-{step}",
        )

    def finalize(self) -> JsonObject:
        return self.client.request(
            "POST", f"/v1/runs/{self.run_id}/finalize", {"status": "completed"}, f"{self.stable_key}-finalize"
        )


def create_run(client: SigV4RegistryClient, body: JsonObject, stable_key: str, step: str = "create") -> OpenRun:
    """Create the run ``body`` describes, or resume the one an earlier attempt created under the same key."""
    created = client.request("POST", "/v1/runs", body, f"{stable_key}-{step}")
    run_id = created.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RegistryRequestError(f"POST /v1/runs: the registry returned no run_id for {stable_key}")
    return OpenRun(client, run_id.strip(), stable_key)
