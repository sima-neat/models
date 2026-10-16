from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest
from model_registry.candidates import publish_branch_deployment
from model_registry.cli import author_main, main
from model_registry.errors import ValidationError
from model_registry.registry import RegistryPublicationError, canonical_sha256


@pytest.fixture
def candidate(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({
        "deployment_id": "11111111-1111-4111-8111-111111111111", "manifest_digest": "a" * 64,
        "git_ref": "refs/heads/feature/llima-models", "state": "finalized",
    }))
    return path


def test_branch_publication_uses_frozen_identity_and_retries_only_snapshot_conflicts(candidate):
    writes, reads = [], []

    def read(url):
        assert parse_qs(urlparse(url).query)["git_ref"] == ["refs/heads/feature/llima-models"]
        reads.append(url)
        return {"catalog": {"canonical_sha256": str(len(reads)) * 64, "document": {"models": []}}}

    def write(method, path, body, key):
        assert method == "POST" and path == "/v1/repositories/sima-neat/models/branch-deployments"
        assert key == "branch-deployment-" + canonical_sha256(body)
        writes.append(body)
        if len(writes) == 1:
            raise RegistryPublicationError("HTTP 409 catalog_snapshot_conflict")
        return {"git_ref": body["git_ref"], "result_snapshot": "c" * 64,
                "selection": {"deployment_id": body["deployment_id"], "manifest_digest": body["manifest_digest"]}}

    result = publish_branch_deployment(candidate, api_url="https://writer.invalid", read_api_url="https://reader.invalid",
                                       aws_region="us-west-2", request=write, read_request=read)
    assert writes == [{
        "git_ref": "refs/heads/feature/llima-models",
        "deployment_id": json.loads(candidate.read_text())["deployment_id"],
        "manifest_digest": "a" * 64, "expected_snapshot": snapshot * 64,
    } for snapshot in ("1", "2")]
    assert json.loads((candidate.parent / "branch-publication.json").read_text()) == result


@pytest.mark.parametrize("error", ["HTTP 409 stale_branch_selection", "HTTP 422 candidate_incompatible",
                                  "HTTP 422 candidate_unavailable", "HTTP 403 forbidden"])
def test_branch_publication_does_not_retry_rejected_candidates(candidate, error):
    writes = []

    def write(*args):
        writes.append(args)
        raise RegistryPublicationError(error)

    with pytest.raises(RegistryPublicationError, match=error):
        publish_branch_deployment(candidate, api_url="https://writer.invalid", read_api_url="https://reader.invalid",
                                  aws_region="us-west-2", request=write,
                                  read_request=lambda _: {"catalog": {"canonical_sha256": "b" * 64}})
    assert len(writes) == 1
    assert not (candidate.parent / "branch-publication.json").exists()


@pytest.mark.parametrize("fields", [
    {"state": "assembling"}, {"manifest_digest": None}, {"deployment_id": None},
    *({"git_ref": f"refs/heads/{branch}"} for branch in ("develop", "main", "master")),
])
def test_branch_publication_rejects_unfinalized_or_protected_branch_receipts(candidate, fields):
    candidate.write_text(json.dumps({**json.loads(candidate.read_text()), **fields}))
    with pytest.raises(ValidationError):
        publish_branch_deployment(candidate, api_url="https://writer.invalid", read_api_url="https://reader.invalid",
                                  aws_region="us-west-2",
                                  read_request=lambda _: pytest.fail("unexpected registry read"))


def test_branch_publication_requires_catalog_requirements_first(candidate):
    with pytest.raises(ValidationError, match="publish the feature-branch model requirements"):
        publish_branch_deployment(candidate, api_url="https://writer.invalid", read_api_url="https://reader.invalid",
                                  aws_region="us-west-2", read_request=lambda _: {"catalog": None})


def test_branch_publication_cli_stays_ci_only_and_forwards_arguments(candidate, monkeypatch, capsys):
    result = {"result_snapshot": "c" * 64}
    calls = []

    def publish(path, **kwargs):
        calls.append((path, kwargs))
        return result

    monkeypatch.setattr("model_registry.candidates.publish_branch_deployment", publish)
    args = ["publish-branch-deployment", "--candidate", str(candidate), "--api-url", "https://writer.invalid",
            "--read-api-url", "https://reader.invalid", "--aws-region", "us-west-2"]
    assert main(args) == 0
    assert calls == [(candidate, {"api_url": "https://writer.invalid", "read_api_url": "https://reader.invalid",
                                 "aws_region": "us-west-2"})]
    assert json.loads(capsys.readouterr().out) == result
    with pytest.raises(SystemExit, match="2"):
        author_main(args)
