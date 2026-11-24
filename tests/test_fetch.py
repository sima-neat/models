from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request

import pytest
from model_registry.errors import FetchError, ValidationError
from model_registry.fetch import (
    Fetcher,
    _build_file_path,
    open_verified_url,
    resolve_dataset,
    unpack_shared_dataset,
)
from model_registry.source import SourceSpec, source_identity

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


class OversizedResponse(Response):
    """Hold eight 1 MiB chunks and count the reads the fetcher makes."""

    def __init__(self) -> None:
        super().__init__(bytes(8 * 1024 * 1024))
        self.reads = 0

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        return super().read(size)


class ReleaseFileHandler(BaseHTTPRequestHandler):
    """Redirect /release to /asset and serve PAYLOAD there, as GitHub does for a release file."""

    def do_GET(self) -> None:
        self.server.requests.append((self.path, self.headers.get("Authorization")))
        if self.path == "/release":
            self.send_response(302)
            self.send_header("Location", "/asset")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def release_server() -> Iterator[HTTPServer]:
    server = HTTPServer(("127.0.0.1", 0), ReleaseFileHandler)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


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


def test_file_set_is_verified_and_materialized_with_its_declared_layout(tmp_path: Path) -> None:
    model = b"onnx"
    weights = b"external weights"
    payloads = {
        "https://huggingface.co/example/models/resolve/0123456789abcdef0123456789abcdef01234567/model-1": model,
        "https://huggingface.co/example/models/resolve/0123456789abcdef0123456789abcdef01234567/model-2": weights,
    }

    def opener(request: Request, timeout: int) -> Response:
        return Response(payloads[request.full_url])

    source_set = {
        "type": "file_set",
        "entrypoint": "graph/model.onnx",
        "files": [
            {
                "path": "graph/model.onnx",
                "uri": "https://huggingface.co/example/models/resolve/0123456789abcdef0123456789abcdef01234567/model-1",
                "sha256": hashlib.sha256(model).hexdigest(),
                "size_bytes": len(model),
                "media_type": "application/onnx",
            },
            {
                "path": "graph/model.data",
                "uri": "https://huggingface.co/example/models/resolve/0123456789abcdef0123456789abcdef01234567/model-2",
                "sha256": hashlib.sha256(weights).hexdigest(),
                "size_bytes": len(weights),
                "media_type": "application/octet-stream",
            },
        ],
    }
    fetcher = Fetcher(tmp_path, opener=opener)

    entrypoint, cache_hit, identity, size = fetcher.fetch_source(source_set)
    assert not cache_hit
    assert identity == source_identity(source_set)
    assert size == len(model) + len(weights)
    assert entrypoint.read_bytes() == model
    assert entrypoint.with_name("model.data").read_bytes() == weights
    assert json.loads(entrypoint.parents[1].joinpath("source-set.json").read_text())["entrypoint"] == (
        "graph/model.onnx"
    )

    same_entrypoint, cache_hit, _, _ = fetcher.fetch_source(source_set)
    assert cache_hit
    assert same_entrypoint == entrypoint


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


def test_oversized_download_stops_at_the_declared_size(tmp_path: Path) -> None:
    responses = []

    def opener(request: Request, timeout: int) -> OversizedResponse:
        responses.append(OversizedResponse())
        return responses[-1]

    fetcher = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=lambda seconds: None,
    )
    with pytest.raises(FetchError, match="after 1 attempt.*size mismatch"):
        fetcher.fetch(source())
    assert [response.reads for response in responses] == [1]
    assert not list(fetcher.cache_path(source()).parent.glob("*.partial"))


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


def test_opener_oserror_is_retried_as_transport_failure(tmp_path: Path) -> None:
    calls = 0
    delays = []

    def opener(request: Request, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("socket disconnected")
        return Response(PAYLOAD)

    path, cache_hit = Fetcher(
        tmp_path,
        opener=opener,
        allowed_https_hosts={"models.example"},
        sleeper=delays.append,
    ).fetch(source())

    assert not cache_hit
    assert path.read_bytes() == PAYLOAD
    assert calls == 2
    assert delays == [1]


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


def test_github_release_file_is_fetched_through_its_redirect_without_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_server: HTTPServer
) -> None:
    declared = []

    def opener(request: Request, timeout: int):
        declared.append((request.full_url, request.get_header("Authorization")))
        local_url = f"http://127.0.0.1:{release_server.server_port}/release"
        return open_verified_url(Request(local_url, headers=dict(request.header_items())), timeout)

    monkeypatch.setenv("HF_TOKEN", "top-secret-token")
    uri = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-obb.pt"
    release_file = SourceSpec(uri=uri, sha256=DIGEST, size_bytes=len(PAYLOAD))
    path, cache_hit = Fetcher(tmp_path, opener=opener).fetch(release_file)

    assert not cache_hit
    assert path.read_bytes() == PAYLOAD
    assert declared == [(uri, None)]
    assert release_server.requests == [("/release", None), ("/asset", None)]


def test_retry_rebuilds_authenticated_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    responses = [Response(PAYLOAD[:4]), Response(PAYLOAD)]

    def opener(request: Request, timeout: int) -> Response:
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setenv("HF_TOKEN", "top-secret-token")
    release_source = SourceSpec(
        uri="hf://models/owner/repository@0123456789abcdef0123456789abcdef01234567/model.onnx",
        sha256=DIGEST,
        size_bytes=len(PAYLOAD),
    )
    Fetcher(tmp_path, opener=opener, sleeper=lambda seconds: None).fetch(release_source)

    assert len(requests) == 2
    assert requests[0] is not requests[1]
    assert [request.get_header("Authorization") for request in requests] == [
        "Bearer top-secret-token",
        "Bearer top-secret-token",
    ]


def fake_dataset_registry(monkeypatch, cached: Path, download: bytes) -> list[Path]:
    """Serve the cached archive like dataset-registry, downloading it when it is missing."""
    calls = []

    def resolve(dataset, cache_dir):
        calls.append(cached)
        if not cached.exists():
            cached.write_bytes(download)
        return cached

    monkeypatch.setattr("model_registry.fetch._resolve_with_dataset_registry", resolve)
    return calls


DATASET = {"id": "fixture", "variant": "val", "sha256": DIGEST, "size_bytes": len(PAYLOAD)}


def test_dataset_matching_its_declaration_is_reused(tmp_path: Path, monkeypatch) -> None:
    cached = tmp_path / "archive.tgz"
    cached.write_bytes(PAYLOAD)
    calls = fake_dataset_registry(monkeypatch, cached, b"unexpected download")

    assert resolve_dataset(DATASET) == cached.resolve()
    assert len(calls) == 1


def test_stale_cached_dataset_is_replaced(tmp_path: Path, monkeypatch) -> None:
    cached = tmp_path / "archive.tgz"
    cached.write_bytes(b"previously published archive")
    calls = fake_dataset_registry(monkeypatch, cached, PAYLOAD)

    assert resolve_dataset(DATASET) == cached.resolve()
    assert cached.read_bytes() == PAYLOAD
    assert len(calls) == 2


def test_dataset_still_differing_after_refetch_fails(tmp_path: Path, monkeypatch) -> None:
    cached = tmp_path / "archive.tgz"
    cached.write_bytes(b"previously published archive")
    calls = fake_dataset_registry(monkeypatch, cached, b"archive differing from the declaration")

    with pytest.raises(FetchError, match="dataset size mismatch"):
        resolve_dataset(DATASET)
    assert len(calls) == 2


def dataset_archive(content: bytes) -> bytes:
    """A gzip tar archive holding val/a.jpg; contents of one length give archives of one size."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            info = tarfile.TarInfo("val/a.jpg")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


ARCHIVE = dataset_archive(b"one")
SHARED_PIN = {"id": "fixture", "variant": "val", "sha256": hashlib.sha256(ARCHIVE).hexdigest(),
              "size_bytes": len(ARCHIVE)}
# The share's index entry for SHARED_PIN's archive, which the share holds at fixture/val.tgz.
SHARED_ARCHIVE = {"key": "datasets/artifactory/fixture/val.tgz", "sha256": SHARED_PIN["sha256"],
                  "size": len(ARCHIVE)}
JUNK = b"not a gzip archive"
JUNK_PIN = {**SHARED_PIN, "sha256": hashlib.sha256(JUNK).hexdigest(), "size_bytes": len(JUNK)}


def shared_archives(tmp_path: Path, index: list[dict] | str, files: dict[str, bytes]) -> Path:
    """A shared dataset archive share, reached through a link, whose index lists ``index`` and holds ``files``."""
    folder = tmp_path / "share" / "releases" / "1"
    folder.mkdir(parents=True)
    for relative, payload in files.items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_bytes(payload)
    text = index if isinstance(index, str) else json.dumps({"archives": index})
    (folder / "_release.json").write_text(text, encoding="utf-8")
    (tmp_path / "share" / "current").symlink_to("releases/1")
    return tmp_path / "share" / "current"


def test_the_share_unpacks_the_archive_its_index_lists_under_the_pin_in_place(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    other = {**SHARED_ARCHIVE, "key": "datasets/artifactory/fixture/other.tgz", "sha256": "0" * 64}
    share = shared_archives(tmp_path, [other, SHARED_ARCHIVE], {"fixture/val.tgz": ARCHIVE,
                                                               "fixture/other.tgz": b"other"})
    archive = (tmp_path / "share/releases/1/fixture/val.tgz").resolve()
    opened = []
    open_path = Path.open

    def counting_open(self: Path, *args, **kwargs):
        opened.append(self.resolve())
        return open_path(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    before = set(tmp_path.rglob("*"))

    assert unpack_shared_dataset(SHARED_PIN, share, tmp_path / "dataset")

    # The archive is read once, in place, and only its members are written.
    assert opened.count(archive) == 1
    assert set(tmp_path.rglob("*")) - before == {tmp_path / "dataset", tmp_path / "dataset/val",
                                                 tmp_path / "dataset/val/a.jpg"}
    assert (tmp_path / "dataset/val/a.jpg").read_bytes() == b"one"
    assert capsys.readouterr().err == ""


def test_a_machine_without_the_share_falls_back_quietly(tmp_path: Path, capsys) -> None:
    assert not unpack_shared_dataset(SHARED_PIN, tmp_path / "absent" / "current", tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "entry",
    [{**SHARED_ARCHIVE, "sha256": "0" * 64}, {**SHARED_ARCHIVE, "size": len(ARCHIVE) + 1}],
    ids=["other_sha256", "other_size"],
)
def test_the_share_gives_only_an_archive_whose_sha256_and_size_match_the_pin(
    tmp_path: Path, capsys, entry: dict
) -> None:
    share = shared_archives(tmp_path, [entry], {"fixture/val.tgz": ARCHIVE})

    assert not unpack_shared_dataset(SHARED_PIN, share, tmp_path / "dataset")
    assert capsys.readouterr().err == (
        "The shared dataset archive share does not hold fixture val. Resolving it through the Dataset Registry.\n"
    )
    assert not (tmp_path / "dataset").exists()


def test_an_archive_of_one_length_has_one_size() -> None:
    # The digest test below depends on it.
    assert len(dataset_archive(b"two")) == len(ARCHIVE)


@pytest.mark.parametrize(
    ("index", "files", "pin", "reason"),
    [
        ("{not json", {}, SHARED_PIN, "Expecting property name"),
        (json.dumps({"releases": []}), {}, SHARED_PIN, "'archives'"),
        ([{"sha256": SHARED_PIN["sha256"], "size": len(ARCHIVE)}], {}, SHARED_PIN, "'key'"),
        ([{**SHARED_ARCHIVE, "key": "fixture/val.tgz"}], {"fixture/val.tgz": ARCHIVE}, SHARED_PIN,
         "does not start with"),
        ([SHARED_ARCHIVE], {}, SHARED_PIN, "No such file"),
        ([SHARED_ARCHIVE], {"fixture/val.tgz": dataset_archive(b"two")}, SHARED_PIN, "dataset SHA-256 mismatch"),
        ([SHARED_ARCHIVE], {"fixture/val.tgz": ARCHIVE + b"x"}, SHARED_PIN, "dataset size mismatch"),
        ([{**SHARED_ARCHIVE, "sha256": JUNK_PIN["sha256"], "size": len(JUNK)}], {"fixture/val.tgz": JUNK},
         JUNK_PIN, "tar exited with"),
    ],
    ids=["malformed_index", "no_archives", "entry_without_key", "key_without_prefix", "listed_file_missing",
         "other_archive_of_the_same_size", "wrong_size", "not_an_archive"],
)
def test_a_broken_share_falls_back_in_one_line_and_leaves_no_folder(
    tmp_path: Path, capsys, index: list[dict] | str, files: dict[str, bytes], pin: dict, reason: str
) -> None:
    share = shared_archives(tmp_path, index, files)

    assert not unpack_shared_dataset(pin, share, tmp_path / "dataset")
    message = capsys.readouterr().err
    assert message.startswith("The shared dataset archive share cannot give fixture val (")
    assert reason in message
    assert message.endswith("). Resolving it through the Dataset Registry.\n")
    assert message.count("\n") == 1
    # Whatever tar unpacked before the check failed is gone.
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize(
    ("relative", "unsafe"),
    [
        ("model/deployment/devkit/vlm_config.json", False),
        ("../model.tar.gz", True),
        ("model/../../model.tar.gz", True),
        ("/tmp/model.tar.gz", True),
        ("model//model.tar.gz", True),
        ("./model/model.tar.gz", True),
        ("model/deployment/", True),
        ("", True),
        ("linked/model.tar.gz", True),
        ("model/linked.tar.gz", True),
    ],
    ids=["valid", "parent", "inner_parent", "absolute", "double_slash", "dot", "trailing_slash", "empty",
         "linked_parent", "linked_file"],
)
def test_a_staged_build_file_stays_inside_its_profile_folder(tmp_path: Path, relative: str, unsafe: bool) -> None:
    profile_dir = tmp_path / "profile"
    (profile_dir / "model").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (profile_dir / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    (profile_dir / "model" / "linked.tar.gz").symlink_to(tmp_path / "elsewhere" / "model.tar.gz")

    if unsafe:
        with pytest.raises(ValidationError, match=re.escape(f"unsafe staged build file path: {relative!r}")):
            _build_file_path(profile_dir, relative)
    else:
        assert _build_file_path(profile_dir, relative) == profile_dir / relative
