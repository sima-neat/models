from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

import pytest
from model_registry.errors import ModelRegistryError, ValidationError
from model_registry.published import ENVIRONMENT_BASE_URLS, _url, artifact_url, fetch_previous_index


class FakeOpener:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses

    def __call__(self, request, timeout: int):
        del timeout
        url = request.full_url
        if url not in self.responses:
            raise HTTPError(url, 404, "not found", {}, None)
        return io.BytesIO(self.responses[url])


def _index(branch: str) -> dict:
    return {
        "schema_version": 1,
        "repository": "sima-neat/models",
        "branch": branch,
        "catalog_tag": "b" * 12,
        "models": {
            "resnet_50": {
                "latest_tag": "a" * 12,
                "artifact_branch": "main",
                "variants": {"modalix_bf16": {"sha256": "c" * 64}},
            }
        },
    }


def test_fetch_previous_index_validates_and_writes_json(tmp_path: Path) -> None:
    branch = "feature/models"
    url = _url(ENVIRONMENT_BASE_URLS["staging"], "models", quote(branch, safe=""), "latest.json")
    output = tmp_path / "previous.json"
    found = fetch_previous_index(
        "staging", branch, output, opener=FakeOpener({url: json.dumps(_index(branch)).encode()})
    )

    assert found == _index(branch)
    assert json.loads(output.read_text(encoding="utf-8")) == _index(branch)


def test_fetch_previous_index_rejects_wrong_branch(tmp_path: Path) -> None:
    branch = "develop"
    url = _url(ENVIRONMENT_BASE_URLS["staging"], "models", branch, "latest.json")
    with pytest.raises(ValidationError, match="branch"):
        fetch_previous_index("staging", branch, tmp_path / "previous.json",
                             opener=FakeOpener({url: json.dumps(_index("main")).encode()}))


def test_missing_previous_index_is_a_clean_first_build(tmp_path: Path) -> None:
    assert fetch_previous_index("production", "new-branch", tmp_path / "previous.json", opener=FakeOpener({})) is None


def test_s3_access_denied_for_missing_index_is_clean_when_models_prefix_is_available(tmp_path: Path) -> None:
    branch = "new-branch"
    base = ENVIRONMENT_BASE_URLS["staging"]
    index_url = _url(base, "models", branch, "latest.json")
    marker_url = _url(base, "models") + "/"

    class S3PrivateOriginOpener(FakeOpener):
        def __call__(self, request, timeout: int):
            if request.full_url == index_url:
                raise HTTPError(index_url, 403, "access denied", {}, None)
            return super().__call__(request, timeout)

    assert fetch_previous_index(
        "staging", branch, tmp_path / "previous.json",
        opener=S3PrivateOriginOpener({marker_url: b""}),
    ) is None


def test_access_denied_remains_fatal_when_models_prefix_is_unavailable(tmp_path: Path) -> None:
    branch = "new-branch"
    base = ENVIRONMENT_BASE_URLS["staging"]
    index_url = _url(base, "models", branch, "latest.json")
    marker_url = _url(base, "models") + "/"

    class BrokenOriginOpener(FakeOpener):
        def __call__(self, request, timeout: int):
            url = request.full_url
            code = 403 if url in {index_url, marker_url} else 404
            raise HTTPError(url, code, "access denied", {}, None)

    with pytest.raises(ModelRegistryError, match="models prefix is also unavailable"):
        fetch_previous_index(
            "staging", branch, tmp_path / "previous.json", opener=BrokenOriginOpener({})
        )


def test_artifact_url_encodes_branch_once_for_storage_key_and_again_for_http() -> None:
    assert artifact_url("staging", "feature/models", "a" * 12, "resnet_50/model.tar.gz") == (
        "https://artifacts.stg.neat.sima.ai/models/feature%252Fmodels/aaaaaaaaaaaa/resnet_50/model.tar.gz"
    )
