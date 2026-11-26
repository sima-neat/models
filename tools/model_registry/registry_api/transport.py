"""Requests to the Vulcan artifact registry: signed writes, public or signed reads, and the canonical digest.

The registry serves two endpoints. Its IAM API takes every write, signed with AWS SigV4 by curl, and the reads that
need credentials. Its public endpoint answers unsigned reads. Every request and response body is a JSON object.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from ..errors import ModelRegistryError

JsonObject = dict[str, Any]
# One signed request: method, path under the API, JSON body or None, and its idempotency key.
Request = Callable[[str, str, JsonObject | None, str], JsonObject]
# One read: the full URL, answered with the JSON object it returns.
ReadRequest = Callable[[str], JsonObject]

ATTEMPTS = 5
# Statuses that mean "try again later": rate limiting and a gateway that could not reach the registry.
RETRIED_STATUSES = {429, 502, 503, 504}
USER_AGENT = "sima-neat-model-registry/0.1"


class RegistryRequestError(ModelRegistryError):
    """A registry request failed: refused, unreachable, or answered with something other than a JSON object."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def is_catalog_conflict(error: RegistryRequestError) -> bool:
    """Whether another publisher replaced the catalog between this publisher's read and its write."""
    return "catalog_snapshot_conflict" in str(error)


def canonical_sha256(value: Any) -> str:
    """The SHA-256 of ``value`` as canonical JSON, the form the registry hashes parameters and documents in."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _curl_config_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class SigV4RegistryClient:
    """Signed requests to the registry's IAM API, with the AWS session credentials from the environment."""

    def __init__(
        self, api_url: str, region: str, *, request: Request | None = None, credential_env_prefix: str = "AWS"
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.region = region
        self._request = request or self._curl_request
        self.credential_env_prefix = credential_env_prefix

    def request(self, method: str, path: str, body: JsonObject | None, idempotency_key: str) -> JsonObject:
        try:
            return self._request(method, path, body, idempotency_key)
        except RegistryRequestError as exc:
            raise RegistryRequestError(f"{method} {path}: {exc}", exc.status) from exc

    def _curl_request(self, method: str, path: str, body: JsonObject | None, idempotency_key: str) -> JsonObject:
        access_key = os.environ.get(f"{self.credential_env_prefix}_ACCESS_KEY_ID", "")
        secret_key = os.environ.get(f"{self.credential_env_prefix}_SECRET_ACCESS_KEY", "")
        session_token = os.environ.get(f"{self.credential_env_prefix}_SESSION_TOKEN", "")
        if not access_key or not secret_key or not session_token:
            raise RegistryRequestError("AWS session credentials are required for registry publication")

        # The credentials reach curl through its config on stdin, never its arguments, which any process on the
        # runner can read.
        credentials = (
            f"user = {_curl_config_value(f'{access_key}:{secret_key}')}\n"
            f"header = {_curl_config_value(f'x-amz-security-token: {session_token}')}\n"
        )
        command = [
            "curl",
            "--config",
            "-",
            "--silent",
            "--show-error",
            "--aws-sigv4",
            f"aws:amz:{self.region}:execute-api",
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
        if body is None:
            return self._send(command, credentials)
        # stdin carries the credentials, so the body goes through a file only this process can read.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as payload:
            payload.write(json.dumps(body, sort_keys=True, separators=(",", ":")))
            payload.flush()
            return self._send([*command, "--data-binary", f"@{payload.name}"], credentials)

    @staticmethod
    def _send(command: list[str], credentials: str) -> JsonObject:
        for attempt in range(1, ATTEMPTS + 1):
            completed = subprocess.run(command, input=credentials, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                if attempt == ATTEMPTS:
                    raise RegistryRequestError(
                        f"artifact registry request failed before receiving HTTP: {completed.stderr.strip()}"
                    )
                time.sleep(2 ** (attempt - 1))
                continue
            response_body, separator, status_text = completed.stdout.rpartition("\n")
            if not separator or not status_text.isdigit():
                raise RegistryRequestError("artifact registry returned an invalid HTTP response")
            status = int(status_text)
            try:
                decoded = json.loads(response_body) if response_body else {}
            except json.JSONDecodeError as exc:
                raise RegistryRequestError(f"artifact registry returned non-JSON HTTP {status}", status) from exc
            if 200 <= status < 300:
                if not isinstance(decoded, dict):
                    raise RegistryRequestError("artifact registry response must be a JSON object", status)
                return decoded
            if status in RETRIED_STATUSES and attempt < ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
                continue
            error = decoded.get("error", {}) if isinstance(decoded, dict) else {}
            code = error.get("code", "unknown_error")
            message = error.get("message", "request rejected")
            raise RegistryRequestError(f"artifact registry HTTP {status} {code}: {message}", status)
        raise AssertionError("unreachable")


def iam_read_api_url() -> str:
    """The IAM API whose reads read_registry_json signs, or "" when it signs none."""
    return os.environ.get("MODEL_REGISTRY_IAM_READ_API_URL", "").rstrip("/")


def read_registry_json(url: str) -> JsonObject:
    """Read the JSON object at ``url``: signed when it is under the configured IAM API, otherwise unsigned."""
    # Sign only the explicitly configured API, never artifact download URLs.
    iam_api = iam_read_api_url()
    if iam_api and url.startswith(iam_api + "/v1/"):
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
            raise RegistryRequestError("AWS_REGION is required for authenticated registry reads")
        prefix = "MODEL_REGISTRY_READ_AWS" if os.environ.get("MODEL_REGISTRY_READ_AWS_ACCESS_KEY_ID") else "AWS"
        return SigV4RegistryClient(iam_api, region, credential_env_prefix=prefix).request(
            "GET", url[len(iam_api):], None, ""
        )
    # An orchestrator run starts every build's benchmark at once, so the public endpoint rate-limits a burst of
    # reads; retry those with jittered backoff so the requests spread out.
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return _read_unsigned(url)
        except RegistryRequestError as exc:
            if exc.status not in RETRIED_STATUSES or attempt == ATTEMPTS:
                raise
        time.sleep(2 ** (attempt - 1) + random.uniform(0, 1))
    raise AssertionError("unreachable")


def _read_unsigned(url: str) -> JsonObject:
    try:
        with urlopen(UrlRequest(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
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
        raise RegistryRequestError(f"public artifact registry HTTP {exc.code} {code}: {message}", exc.code) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryRequestError(f"could not read public artifact registry endpoint {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryRequestError(f"artifact registry endpoint did not return an object: {url}")
    return value
