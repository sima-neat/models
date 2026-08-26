from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request

from .errors import ModelRegistryError, ValidationError
from .fetch import open_verified_url

ENVIRONMENT_BASE_URLS = {
    "staging": "https://artifacts.stg.neat.sima.ai",
    "production": "https://artifacts.neat.sima.ai",
}
OpenUrl = Callable[..., BinaryIO]


def _url(base_url: str, *parts: str) -> str:
    encoded = [quote(part.strip("/"), safe="/._-+~") for part in parts if part.strip("/")]
    return "/".join([base_url.rstrip("/"), *encoded])


def artifact_url(environment: str, branch: str, tag: str, path: str) -> str:
    if environment not in ENVIRONMENT_BASE_URLS:
        raise ValidationError(f"unsupported artifact environment: {environment}")
    return _url(ENVIRONMENT_BASE_URLS[environment], "models", quote(branch, safe=""), tag, path)


def _read(opener: OpenUrl, url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "sima-neat-model-registry/0.1"})
    with opener(request, timeout=60) as response:
        return response.read()


def validate_latest_index(value: Any, *, branch: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") not in {1, 2}:
        raise ValidationError("latest.json must be a supported schema-version 1 or 2 object")
    if value.get("repository") != "sima-neat/models":
        raise ValidationError("latest.json repository must be sima-neat/models")
    if branch is not None and value.get("branch") != branch:
        raise ValidationError("latest.json branch does not match the requested branch")
    if value["schema_version"] == 2:
        taxonomy = value.get("taxonomy")
        if not isinstance(taxonomy, dict) or taxonomy.get("version") != 1 or taxonomy.get("path") != "taxonomy.json":
            raise ValidationError("latest.json schema version 2 requires taxonomy version 1 at taxonomy.json")
        digest = taxonomy.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValidationError("latest.json taxonomy sha256 must be a lowercase SHA-256 digest")
    models = value.get("models")
    if not isinstance(models, dict):
        raise ValidationError("latest.json models must be an object")
    for model_id, entry in models.items():
        if not isinstance(model_id, str) or not model_id or not isinstance(entry, dict):
            raise ValidationError("latest.json contains an invalid model entry")
        tag = entry.get("latest_tag")
        artifact_branch = entry.get("artifact_branch")
        variants = entry.get("variants")
        if not isinstance(tag, str) or not 7 <= len(tag) <= 40 or any(c not in "0123456789abcdef" for c in tag.lower()):
            raise ValidationError(f"latest.json model {model_id!r} has an invalid latest_tag")
        if not isinstance(artifact_branch, str) or not artifact_branch.strip():
            raise ValidationError(f"latest.json model {model_id!r} has an invalid artifact_branch")
        if not isinstance(variants, dict) or not variants:
            raise ValidationError(f"latest.json model {model_id!r} has no variants")
        fingerprint = entry.get("build_fingerprint")
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint.lower())
        ):
            raise ValidationError(f"latest.json model {model_id!r} has an invalid build_fingerprint")
    return value


def fetch_previous_index(
    environment: str,
    branch: str,
    output_file: str | Path,
    *,
    opener: OpenUrl = open_verified_url,
) -> dict[str, Any] | None:
    if environment not in ENVIRONMENT_BASE_URLS:
        raise ValidationError(f"unsupported artifact environment: {environment}")
    branch_name = branch.strip()
    if not branch_name:
        raise ValidationError("branch name must not be empty")
    output = Path(output_file).expanduser().resolve()
    if output.exists():
        raise ValidationError(f"previous index output already exists: {output}")

    url = _url(ENVIRONMENT_BASE_URLS[environment], "models", quote(branch_name, safe=""), "latest.json")
    try:
        value = json.loads(_read(opener, url))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code == 403:
            # A private S3 origin reports a missing object as AccessDenied.
            # Confirm that the public models prefix itself is reachable before
            # interpreting this as a clean first build; otherwise retain the
            # hard failure so an OAC or distribution outage cannot erase
            # carry-forward entries from latest.json.
            marker_url = _url(ENVIRONMENT_BASE_URLS[environment], "models") + "/"
            try:
                _read(opener, marker_url)
            except (HTTPError, OSError, URLError, TimeoutError) as marker_error:
                raise ModelRegistryError(
                    f"could not read previous latest.json: HTTP {exc.code}; models prefix is also unavailable"
                ) from marker_error
            return None
        raise ModelRegistryError(f"could not read previous latest.json: HTTP {exc.code}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, URLError) as exc:
        raise ModelRegistryError(f"could not read previous latest.json: {exc}") from exc

    index = validate_latest_index(value, branch=branch_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index
