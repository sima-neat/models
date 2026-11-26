from __future__ import annotations

import io
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from model_registry.registry_api.transport import RegistryRequestError, read_registry_json

REGISTRY_URL = "https://registry.example/v1/catalog"


def registry_error(code: int) -> HTTPError:
    body = json.dumps({"error": {"code": "rate_limited" if code == 429 else "not_found", "message": "fixture"}})
    return HTTPError(REGISTRY_URL, code, "fixture", {}, io.BytesIO(body.encode()))


def fake_registry(monkeypatch: pytest.MonkeyPatch, responses: list[HTTPError | dict]) -> tuple[list[str], list[float]]:
    """Answer each public registry read with the next response, and record the reads and the backoff sleeps."""
    reads, sleeps = [], []

    def urlopen(request: Request, timeout: int) -> io.BytesIO:
        reads.append(request.full_url)
        response = responses.pop(0)
        if isinstance(response, HTTPError):
            raise response
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.delenv("MODEL_REGISTRY_IAM_READ_API_URL", raising=False)
    monkeypatch.setattr("model_registry.registry_api.transport.urlopen", urlopen)
    monkeypatch.setattr("model_registry.registry_api.transport.time.sleep", sleeps.append)
    monkeypatch.setattr("model_registry.registry_api.transport.random.uniform", lambda low, high: 0.5)
    return reads, sleeps


def test_rate_limited_registry_read_is_retried_after_a_jittered_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, sleeps = fake_registry(monkeypatch, [registry_error(429), {"catalog": "fixture"}])

    assert read_registry_json(REGISTRY_URL) == {"catalog": "fixture"}
    assert reads == [REGISTRY_URL, REGISTRY_URL]
    assert sleeps == [1.5]


def test_registry_read_that_cannot_succeed_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, sleeps = fake_registry(monkeypatch, [registry_error(404)])

    with pytest.raises(RegistryRequestError, match="public artifact registry HTTP 404 not_found: fixture"):
        read_registry_json(REGISTRY_URL)
    assert reads == [REGISTRY_URL]
    assert sleeps == []


def test_registry_read_fails_after_five_rate_limited_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, sleeps = fake_registry(monkeypatch, [registry_error(429) for _ in range(5)])

    with pytest.raises(RegistryRequestError, match="public artifact registry HTTP 429 rate_limited: fixture"):
        read_registry_json(REGISTRY_URL)
    assert len(reads) == 5
    # The backoff doubles between attempts, and the fifth failure raises without sleeping.
    assert sleeps == [1.5, 2.5, 4.5, 8.5]
