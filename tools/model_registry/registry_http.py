"""Registry HTTP transport shared by author-safe reads and CI publication."""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

from .errors import ModelRegistryError

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]


class RegistryPublicationError(ModelRegistryError):
    """A model compilation could not be published to the artifact registry."""


class SigV4RegistryClient:
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
        except RegistryPublicationError as exc:
            raise RegistryPublicationError(f"{method} {path}: {exc}") from exc

    def _curl_request(self, method: str, path: str, body: JsonObject | None, idempotency_key: str) -> JsonObject:
        access_key = os.environ.get(f"{self.credential_env_prefix}_ACCESS_KEY_ID", "")
        secret_key = os.environ.get(f"{self.credential_env_prefix}_SECRET_ACCESS_KEY", "")
        session_token = os.environ.get(f"{self.credential_env_prefix}_SESSION_TOKEN", "")
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



def read_registry_json_authenticated(url: str) -> JsonObject:
    from .errors import FetchError
    from .fetch import read_registry_json

    # Sign only the explicitly configured API, never artifact download URLs.
    iam_api = os.environ.get("MODEL_REGISTRY_IAM_READ_API_URL", "").rstrip("/")
    if iam_api and url.startswith(iam_api + "/v1/"):
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
            raise RegistryPublicationError("AWS_REGION is required for authenticated registry reads")
        prefix = "MODEL_REGISTRY_READ_AWS" if os.environ.get("MODEL_REGISTRY_READ_AWS_ACCESS_KEY_ID") else "AWS"
        return SigV4RegistryClient(iam_api, region, credential_env_prefix=prefix).request(
            "GET", url[len(iam_api):], None, ""
        )
    try:
        return read_registry_json(url)
    except FetchError as exc:
        raise RegistryPublicationError(str(exc)) from exc


