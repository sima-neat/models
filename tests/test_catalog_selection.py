from __future__ import annotations

from pathlib import Path

import pytest
from model_registry.catalog_selection import publish_selection
from model_registry.dependencies import catalog_inputs
from model_registry.errors import ValidationError
from model_registry.registry import S3Object

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
CATALOG = "11111111-1111-4111-8111-111111111111"
QUALIFICATION = "22222222-2222-4222-8222-222222222222"
NEW_SELECTION = "33333333-3333-4333-8333-333333333333"
OLD_SELECTION = "44444444-4444-4444-8444-444444444444"
OTHER_SELECTION = "55555555-5555-4555-8555-555555555555"
INPUTS = catalog_inputs(ROOT)
PATH = "/v1/repositories/sima-neat/models"


class Store:
    def ensure_native_sha256(self, path: Path, key: str, sha256: str, *, media_type: str) -> S3Object:
        return S3Object(key=key, version_id="v1", sha256=sha256, size_bytes=path.stat().st_size)


def selection(selection_id: str, *, catalog_id: str = "66666666-6666-4666-8666-666666666666",
              revision: str = INPUTS["neat_core_revision"], default: bool = False) -> dict:
    return {"selection_id": selection_id, "catalog_id": catalog_id, "platform_version": INPUTS["platform_version"],
            "neat_core_version": INPUTS["neat_core_version"], "neat_core_revision": revision, "retired": False,
            "is_default": default}


def publish(
    selections: list[dict], *, document: dict | None = None, run_url: str = "https://ci.example.invalid/runs/1",
    keys: list[str] | None = None,
) -> list[tuple[str, str, dict]]:
    """Publish main's catalog against a registry whose platform already has these selections."""
    calls: list[tuple[str, str, dict]] = []
    record = {"id": CATALOG, "catalog_revision": SHA, "canonical_sha256": "c" * 64,
              "document": {"models": [], "pending_builds": [], "catalog_inputs": INPUTS} if document is None
              else document}

    def read(url: str) -> dict:
        return {"selections": selections} if "/models/filters?" in url else {"catalog": record}

    def request(method: str, path: str, body: dict, key: str) -> dict:
        calls.append((method, path, body))
        if keys is not None:
            keys.append(key)
        if path == "/v1/runs":
            return {"run_id": QUALIFICATION}
        if path.endswith("/artifacts/report"):
            return {"artifact_id": "report-artifact"}
        if path.endswith("/catalog-selections"):
            return {"selection": {"id": NEW_SELECTION}}
        return {}

    publish_selection(
        repository=ROOT, api_url="https://registry.example.invalid", read_api_url="https://read.example.invalid",
        aws_region="us-west-2", bucket="bucket", kms_key_id=None, environment="production", branch="main",
        git_sha=SHA, github_run_url=run_url, request=request, read_request=read, artifact_store=Store(),
    )
    return calls


def writes(calls: list[tuple[str, str, dict]]) -> dict[str, dict]:
    return {path: body for method, path, body in calls if path.startswith(f"{PATH}/catalog-selections")}


def test_qualification_names_the_catalog_and_its_inputs_and_passes() -> None:
    calls = publish([])
    create = calls[0][2]
    test_result = next(body for _, path, body in calls if "/test-results/" in path)

    assert create["run_type"] == "catalog_qualification"
    assert create["metadata"] == {"catalog_id": CATALOG}
    assert create["environment"]["properties"]["catalog_inputs"] == INPUTS
    assert test_result["status"] == "passed"
    assert calls[-3][1] == f"/v1/runs/{QUALIFICATION}/finalize"


def test_the_first_selection_of_a_platform_becomes_its_default() -> None:
    assert writes(publish([])) == {
        f"{PATH}/catalog-selections": {
            "catalog_id": CATALOG, "evidence_run_id": QUALIFICATION, "expected_selection_id": None,
        },
        f"{PATH}/catalog-selections/default": {"selection_id": NEW_SELECTION, "expected_default_id": None},
    }


def test_replacing_the_default_selection_moves_the_default_with_it() -> None:
    assert writes(publish([selection(OLD_SELECTION, default=True)])) == {
        f"{PATH}/catalog-selections": {
            "catalog_id": CATALOG, "evidence_run_id": QUALIFICATION, "expected_selection_id": OLD_SELECTION,
        },
    }


def test_a_new_core_release_adds_a_selection_and_takes_the_default() -> None:
    other = selection(OTHER_SELECTION, revision="b" * 40, default=True)

    assert writes(publish([other])) == {
        f"{PATH}/catalog-selections": {
            "catalog_id": CATALOG, "evidence_run_id": QUALIFICATION, "expected_selection_id": None,
        },
        f"{PATH}/catalog-selections/default": {"selection_id": NEW_SELECTION, "expected_default_id": OTHER_SELECTION},
    }


def test_a_rerun_keeps_the_selection_it_already_published() -> None:
    assert writes(publish([selection(NEW_SELECTION, catalog_id=CATALOG, default=True)])) == {}


def test_a_new_workflow_run_sends_its_own_writes_and_a_retry_replays_them() -> None:
    first, retry, later = [], [], []
    publish([], keys=first)
    publish([], keys=retry)
    publish([], run_url="https://ci.example.invalid/runs/2", keys=later)

    assert first == retry
    assert not set(first) & set(later)


@pytest.mark.parametrize("document,error", [
    ({"models": [], "pending_builds": [{"model_id": "resnet_50"}], "catalog_inputs": INPUTS}, "1 pending builds"),
    ({"models": [], "pending_builds": []}, "the catalog records None"),
])
def test_an_incomplete_catalog_publishes_no_selection(document: dict, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        publish([], document=document)
