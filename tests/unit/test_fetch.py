from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request

import pytest
from model_registry.errors import FetchError
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec

PAYLOAD = b"model registry fixture\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class Response(io.BytesIO):
    pass


class InterruptedResponse(Response):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.calls = 0

    def read(self, size: int = -1) -> bytes:
        self.calls += 1
        if self.calls > 1:
            raise OSError("connection interrupted")
        return super().read(4)


def source(digest: str = DIGEST, size: int = len(PAYLOAD)) -> SourceSpec:
    return SourceSpec(uri="https://models.example/model.onnx", sha256=digest, size_bytes=size)


def test_download_is_verified_and_reused_from_cache(tmp_path: Path) -> None:
    calls = []

    def opener(request: Request, timeout: int) -> Response:
        calls.append((request, timeout))
        return Response(PAYLOAD)

    fetcher = Fetcher(tmp_path, opener=opener, allowed_https_hosts={"models.example"})
    path, cache_hit = fetcher.fetch(source())
    assert not cache_hit
    assert path.read_bytes() == PAYLOAD
    assert len(calls) == 1
    assert not list(path.parent.glob("*.partial"))

    same_path, cache_hit = fetcher.fetch(source())
    assert cache_hit
    assert same_path == path
    assert len(calls) == 1


def test_corrupt_cache_entry_is_replaced(tmp_path: Path) -> None:
    fetcher = Fetcher(
        tmp_path,
        opener=lambda request, timeout: Response(PAYLOAD),
        allowed_https_hosts={"models.example"},
    )
    destination = fetcher.cache_path(source())
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt")
    path, cache_hit = fetcher.fetch(source())
    assert not cache_hit
    assert path.read_bytes() == PAYLOAD


def test_checksum_mismatch_removes_partial_file(tmp_path: Path) -> None:
    calls = 0

    def opener(request: Request, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        return Response(PAYLOAD)

    fetcher = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=lambda seconds: None,
    )
    bad_source = source(digest="0" * 64)
    with pytest.raises(FetchError, match="SHA-256 mismatch"):
        fetcher.fetch(bad_source)
    destination = fetcher.cache_path(bad_source)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.partial"))
    assert calls == 1


def test_size_mismatch_fails(tmp_path: Path) -> None:
    calls = 0

    def opener(request: Request, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        return Response(PAYLOAD)

    fetcher = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=lambda seconds: None,
    )
    with pytest.raises(FetchError, match="after 3 attempt.*size mismatch"):
        fetcher.fetch(source(size=len(PAYLOAD) + 1))
    assert calls == 3


def test_interruption_removes_partial_file(tmp_path: Path) -> None:
    calls = 0

    def opener(request: Request, timeout: int) -> InterruptedResponse:
        nonlocal calls
        calls += 1
        return InterruptedResponse(PAYLOAD)

    fetcher = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=lambda seconds: None,
    )
    with pytest.raises(FetchError, match="after 3 attempt.*connection interrupted"):
        fetcher.fetch(source())
    destination = fetcher.cache_path(source())
    assert not destination.exists()
    assert not list(destination.parent.glob("*.partial"))
    assert calls == 3


def test_truncated_download_is_retried_and_then_cached(tmp_path: Path) -> None:
    responses = [Response(PAYLOAD[:4]), Response(PAYLOAD[:8]), Response(PAYLOAD)]
    delays = []
    requests = []

    def opener(request: Request, timeout: int) -> Response:
        requests.append(request)
        return responses.pop(0)

    fetcher = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=delays.append,
    )
    path, cache_hit = fetcher.fetch(source())

    assert not cache_hit
    assert path.read_bytes() == PAYLOAD
    assert delays == [1, 2]
    assert len({id(request) for request in requests}) == 3
    assert not responses
    assert not list(path.parent.glob("*.partial"))


def test_local_cache_io_failure_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def opener(request: Request, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        return Response(PAYLOAD)

    def fail_replace(source_path: Path, destination_path: Path) -> None:
        raise OSError("local cache is full")

    monkeypatch.setattr("model_registry.fetch.os.replace", fail_replace)
    fetcher = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=lambda seconds: None,
    )
    with pytest.raises(FetchError, match="after 1 attempt.*local cache is full"):
        fetcher.fetch(source())

    assert calls == 1
    assert not list(fetcher.cache_path(source()).parent.glob("*.partial"))


def test_hugging_face_token_is_sent_but_not_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def opener(request: Request, timeout: int) -> Response:
        observed["authorization"] = request.get_header("Authorization")
        redirected = HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://cdn.example/model.onnx"
        )
        observed["redirect_authorization"] = redirected.get_header("Authorization")
        return Response(PAYLOAD)

    monkeypatch.setenv("HF_TOKEN", "top-secret-token")
    hf_source = SourceSpec(
        uri="hf://models/owner/repository@0123456789abcdef0123456789abcdef01234567/model.onnx",
        sha256=DIGEST,
        size_bytes=len(PAYLOAD),
    )
    Fetcher(tmp_path, opener=opener).fetch(hf_source)
    assert observed["authorization"] == "Bearer top-secret-token"
    assert observed["redirect_authorization"] is None


def test_github_token_is_scoped_to_release_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def opener(request: Request, timeout: int) -> Response:
        observed["authorization"] = request.get_header("Authorization")
        observed["accept"] = request.get_header("Accept")
        redirected = HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://objects.githubusercontent.com/signed-data.npz"
        )
        observed["redirect_authorization"] = redirected.get_header("Authorization")
        return Response(PAYLOAD)

    monkeypatch.setenv("GITHUB_TOKEN", "private-release-token")
    release_source = SourceSpec(
        uri="https://api.github.com/repos/sima-neat/models/releases/assets/123456",
        sha256=DIGEST,
        size_bytes=len(PAYLOAD),
    )
    Fetcher(tmp_path, opener=opener).fetch(release_source)
    assert observed["authorization"] == "Bearer private-release-token"
    assert observed["accept"] == "application/octet-stream"
    assert observed["redirect_authorization"] is None


def test_github_retry_rebuilds_authenticated_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    responses = [Response(PAYLOAD[:4]), Response(PAYLOAD)]

    def opener(request: Request, timeout: int) -> Response:
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setenv("GITHUB_TOKEN", "private-release-token")
    release_source = SourceSpec(
        uri="https://api.github.com/repos/sima-neat/models/releases/assets/123456",
        sha256=DIGEST,
        size_bytes=len(PAYLOAD),
    )
    Fetcher(tmp_path, opener=opener, sleeper=lambda seconds: None).fetch(release_source)

    assert len(requests) == 2
    assert requests[0] is not requests[1]
    assert [request.get_header("Authorization") for request in requests] == [
        "Bearer private-release-token",
        "Bearer private-release-token",
    ]
