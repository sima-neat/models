from __future__ import annotations

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import yaml
from model_registry import conversion
from model_registry.errors import ValidationError
from model_registry.fingerprints import compile_fingerprint
from model_registry.model import load_and_validate_model
from model_registry.registry import RegistryPublicationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    for name in (
        "schemas", "taxonomies", "datasets", "scripts",
        "tools/model_registry", "tools/model_report", ".github/workflows",
    ):
        shutil.copytree(ROOT / name, root / name)
    for name in ("conftest.py", "pyproject.toml"):
        shutil.copyfile(ROOT / name, root / name)
    shutil.copytree(ROOT / "models/yolo26x", root / "models/yolo26x")
    path = root / "models/yolo26x/model.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["compilation"]["source"] = "conversion"
    path.write_text(yaml.safe_dump(doc))
    return root


def output(directory, external=False, external_location="weights.bin"):
    directory.mkdir()
    tensor = onnx.numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="weights")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "weights"], ["y"])],
        "model",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])],
        [tensor],
    )
    model = onnx.helper.make_model(graph)
    onnx.save_model(
        model,
        directory / "model.onnx",
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=external_location,
        size_threshold=0,
    )
    return directory


class Registry:
    def __init__(self):
        self.calls = []
        self.inputs = []
        self.status = "running"
        self.latest = "conversion-1"
        self.artifacts = {
            run: {
                "id": run + "-artifact",
                "artifact_type": "model-onnx",
                "object_status": "available",
                "s3_bucket": "bucket",
                "s3_key": run + "/onnx.tar.gz",
                "s3_version_id": "version",
                "sha256": char * 64,
                "size_bytes": 123,
                "media_type": "application/gzip",
                "metadata": {"validation": "passed"},
            }
            for run, char in [("conversion-1", "a"), ("conversion-2", "b")]
        }

    def request(self, method, path, body, key):
        self.calls.append((method, path, body, key))
        if path == "/v1/runs":
            return {"run_id": "plan"}
        if path.endswith("/inputs/conversion"):
            self.inputs = [{"name": "conversion", **body}]
        if path.endswith("/finalize"):
            self.status = body["status"]
        return {}

    def read(self, url):
        if url.endswith("/provenance"):
            return {"run": {"status": self.status}, "inputs": self.inputs}
        if "/resolve?" in url:
            if self.latest is None:
                raise RegistryPublicationError("artifact registry HTTP 404 fingerprint_not_found")
            return {"run": {"id": self.latest}}
        return {"artifact": self.artifacts[url.split("/")[-3]]}


def plan(root, registry):
    model = load_and_validate_model(root / "models/yolo26x")
    return conversion.plan_conversions(
        root,
        {"yolo26x": model},
        read=registry.read,
        read_api_url="https://registry.test",
        client=registry,
        branch="feature",
    )


def test_schema_is_offline_and_requires_conversion(repository):
    model = load_and_validate_model(repository / "models/yolo26x")
    assert model["components"][0]["source"] == {"type": "conversion"}
    path = repository / "models/yolo26x/model.yaml"
    doc = yaml.safe_load(path.read_text())
    del doc["conversion"]
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValidationError, match="requires an ONNX"):
        load_and_validate_model(path.parent)


def test_retry_retains_selection_and_new_plan_gets_latest(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    registry = Registry()
    first = plan(repository, registry)
    registry.latest = "conversion-2"
    assert plan(repository, registry) == first
    newer = Registry()
    newer.latest = "conversion-2"
    assert plan(repository, newer)["yolo26x"]["sha256"] == "b" * 64
    assert "download_url" not in first["yolo26x"]


def test_missing_and_unavailable_fail_without_fallback(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    registry = Registry()
    registry.latest = None
    with pytest.raises(ValidationError, match="run Vulcan Model Convert"):
        plan(repository, registry)
    registry.latest = "conversion-1"
    registry.artifacts[registry.latest]["object_status"] = "deleted"
    with pytest.raises(ValidationError, match="unavailable"):
        plan(repository, registry)


def test_compile_fingerprint_depends_on_content_not_location(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    selected = plan(repository, Registry())

    def fingerprint(value):
        return compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources=value)

    first = fingerprint(selected)
    other = copy.deepcopy(selected)
    other["yolo26x"].update(run_id="new-run", s3_bucket="production", read_api_url="https://production.test")
    assert fingerprint(other) == first
    other["yolo26x"]["sha256"] = "b" * 64
    assert fingerprint(other) != first


@pytest.mark.parametrize("filename", ["model.onnx", "weights.bin", "onnx.tar.gz"])
def test_bundle_rejects_output_collisions_without_changing_model_data(tmp_path, filename):
    location = filename if filename != "model.onnx" else "weights.bin"
    directory = output(tmp_path / "output", external=True, external_location=location)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValidationError, match="would overwrite model data"):
        conversion.bundle_onnx(directory, directory / filename)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_bundle_rejects_output_alias_of_tensor_data(tmp_path, link_type):
    directory = output(tmp_path / "output", external=True)
    destination = tmp_path / "bundle.tar.gz"
    tensor = directory / "weights.bin"
    if link_type == "symlink":
        destination.symlink_to(tensor)
    else:
        destination.hardlink_to(tensor)
    original = tensor.read_bytes()
    with pytest.raises(ValidationError, match="would overwrite model data"):
        conversion.bundle_onnx(directory, destination)
    assert tensor.read_bytes() == original


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("private", [False, True])
def test_bundle_is_deterministic_and_fetch_checks_exact_version(tmp_path, monkeypatch, external, private):
    directory = output(tmp_path / "output", external)
    first = conversion.bundle_onnx(directory, tmp_path / "first.tar.gz")
    second = conversion.bundle_onnx(directory, tmp_path / "second.tar.gz")
    assert first.read_bytes() == second.read_bytes()
    sha, size = conversion.sha256_file(first)
    source = {
        "type": "conversion",
        "run_id": "run",
        "artifact_id": "artifact",
        "sha256": sha,
        "size_bytes": size,
        "s3_bucket": "bucket",
        "s3_key": "key",
        "s3_version_id": "v1",
        "media_type": "application/gzip",
        "read_api_url": "https://registry.test",
    }
    artifact = {
        **source,
        "id": "artifact",
        "artifact_type": "model-onnx",
        "object_status": "available",
        "metadata": {"validation": "passed"},
        "download_url": first.as_uri(),
    }
    calls = []
    if private:
        monkeypatch.setenv("MODEL_REGISTRY_IAM_READ_API_URL", source["read_api_url"])
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        def signed(self, method, path, body, key):
            calls.append((method, path))
            return {"artifact": artifact}
        monkeypatch.setattr("model_registry.registry.SigV4RegistryClient.request", signed)
        def unsigned(url):
            pytest.fail("private conversion metadata must use SigV4")
        monkeypatch.setattr("model_registry.fetch.read_registry_json", unsigned)
    else:
        monkeypatch.delenv("MODEL_REGISTRY_IAM_READ_API_URL", raising=False)
        monkeypatch.setattr("model_registry.fetch.read_registry_json", lambda _: {"artifact": artifact})
    fetched, hit, _, _ = conversion.fetch_conversion(source, tmp_path / "cache")
    assert not hit
    onnx.checker.check_model(str(fetched))
    assert (fetched.parent / "weights.bin").exists() == external
    assert conversion.fetch_conversion(source, tmp_path / "cache")[1]
    if private:
        assert calls == [("GET", "/v1/runs/run/artifacts/onnx")] * 2
    artifact["s3_version_id"] = "v2"
    with pytest.raises(ValidationError, match="saved plan"):
        conversion.fetch_conversion(source, tmp_path / "cache")


def test_publication_completes_only_after_validated_output(repository, tmp_path):
    directory = output(tmp_path / "output")
    registry = Registry()
    store = SimpleNamespace(ensure_native_sha256=lambda path, key, sha: SimpleNamespace(key=key, version_id="v1"))
    kwargs = dict(
        client=registry,
        store=store,
        bucket="bucket",
        environment="staging",
        branch="feature",
        git_sha="a" * 40,
        workflow="42",
    )
    with pytest.raises(ValidationError, match="passed"):
        conversion.publish_conversion(repository, "yolo26x", directory, validation="skipped", **kwargs)
    assert not registry.calls
    result = conversion.publish_conversion(repository, "yolo26x", directory, validation="passed", **kwargs)
    assert result["run_id"] == "plan"
    assert registry.calls[-1][1].endswith("/finalize")
    artifact = next(c[2] for c in registry.calls if c[1].endswith("/artifacts/onnx"))
    assert artifact["s3_key"].startswith("models/conversions/yolo26x/")
    assert artifact["metadata"]["validation"] == "passed"


def test_planning_resolves_before_compile_reuse(repository, monkeypatch):
    from model_registry import stage_plan

    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    registry = Registry()
    selected = plan(repository, registry)
    fingerprint = compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources=selected)
    catalog = {
        "models": [
            {
                "id": "yolo26x",
                "builds": [{"component_id": "model", "build_profile_id": "w16a16", "compile_fingerprint": fingerprint}],
            }
        ]
    }
    monkeypatch.setattr(stage_plan, "read_catalog_sources", lambda *args: (None, [catalog]))
    monkeypatch.setattr(stage_plan, "benchmark_fingerprint", lambda *args: "c" * 64)
    monkeypatch.setattr(stage_plan, "stage_run_exists", lambda *args: True)

    def stages(client):
        return stage_plan.plan_stages(
            repository,
            compiler_version="test",
            read=client.read,
            read_api_url="https://registry.test",
            branch="feature",
            profile_ids=["w16a16"],
            conversion_client=client,
        )

    assert stages(registry)["compile"]["include"] == []
    newer = Registry()
    newer.latest = "conversion-2"
    assert len(stages(newer)["compile"]["include"]) == 1
    newer.artifacts["conversion-2"]["sha256"] = "a" * 64
    newer.inputs = []
    newer.status = "running"
    assert stages(newer)["compile"]["include"] == []


def test_targeted_direct_source_plan_needs_no_conversion_credentials(repository):
    from model_registry.stage_plan import plan_stages

    shutil.copytree(ROOT / "models/yolo26l", repository / "models/yolo26l")
    result = plan_stages(repository, compiler_version="test", read=None, read_api_url=None,
                         branch="feature", model_ids=["yolo26l"])
    assert result["conversion_sources"] == {}
    assert {entry["model_id"] for entry in result["compile"]["include"]} == {"yolo26l"}


def test_targeted_conversion_plan_creates_only_selected_plan(repository, monkeypatch):
    from model_registry.stage_plan import plan_stages

    shutil.copytree(ROOT / "models/yolo26l", repository / "models/yolo26l")
    path = repository / "models/yolo26l/model.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["compilation"]["source"] = "conversion"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setattr("model_registry.stage_plan.read_catalog_sources", lambda *args: (None, []))
    registry = Registry()
    result = plan_stages(repository, compiler_version="test", read=registry.read,
                         read_api_url="https://registry.test", branch="feature",
                         model_ids=["yolo26x"], conversion_client=registry)
    assert set(result["conversion_sources"]) == {"yolo26x"}
    created = [call[2]["metadata"]["model_id"] for call in registry.calls
               if call[0] == "POST" and call[1] == "/v1/runs"]
    assert created == ["yolo26x"]


def test_unknown_model_rejected_before_conversion_registry_requests(repository):
    from model_registry.stage_plan import plan_stages

    registry = Registry()
    with pytest.raises(ValidationError, match="unknown model id.*missing"):
        plan_stages(repository, compiler_version="test", read=registry.read,
                    read_api_url="https://registry.test", branch="feature",
                    model_ids=["yolo26x", "missing"], conversion_client=registry)
    assert registry.calls == []


def test_conversion_fingerprint_tracks_recipe_and_dependency_pins(repository):
    model = load_and_validate_model(repository / "models/yolo26x")
    original = conversion.conversion_fingerprint(repository, model)
    requirements = repository / "models/yolo26x/scripts/requirements-convert.txt"
    requirements.write_text(requirements.read_text() + "\n# new exporter version\n")
    assert conversion.conversion_fingerprint(repository, model) != original
    with pytest.raises(ValidationError, match="plan-stages"):
        compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources={})


@pytest.mark.parametrize("name", ["image.png", "array.npy", "tensor.bin", "expected.json", "extensionless"])
def test_conversion_fixture_changes_invalidate_saved_selection(repository, name):
    model = load_and_validate_model(repository / "models/yolo26x")
    baseline = conversion.conversion_fingerprint(repository, model)
    fixture = repository / "models/yolo26x/tests/fixtures/nested" / name
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(b"\x00\x01fixture")
    added = conversion.conversion_fingerprint(repository, model)
    assert added != baseline
    fixture.write_bytes(b"\x00\x02fixture")
    assert conversion.conversion_fingerprint(repository, model) != added
    with pytest.raises(ValidationError, match="checkout's recipe"):
        conversion.apply_selection(model, {"yolo26x": {"conversion_fingerprint": added}}, root=repository)
    fixture.unlink()
    assert conversion.conversion_fingerprint(repository, model) == baseline


def test_conversion_fingerprint_ignores_generated_test_caches(repository):
    model = load_and_validate_model(repository / "models/yolo26x")
    original = conversion.conversion_fingerprint(repository, model)
    for relative in ("__pycache__/test_conversion.pyc", ".pytest_cache/v/cache/nodeids"):
        path = repository / "models/yolo26x/tests" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated cache")
    assert conversion.conversion_fingerprint(repository, model) == original


@pytest.mark.parametrize("relative", conversion.CONVERSION_TOOLING)
def test_shared_conversion_dependency_invalidates_saved_selection(repository, relative):
    model = load_and_validate_model(repository / "models/yolo26x")
    original = conversion.conversion_fingerprint(repository, model)
    path = repository / relative
    path.write_bytes(path.read_bytes() + b"\n")
    assert conversion.conversion_fingerprint(repository, model) != original
    with pytest.raises(ValidationError, match="checkout's recipe"):
        conversion.apply_selection(model, {"yolo26x": {"conversion_fingerprint": original}}, root=repository)


def test_saved_selection_rejects_a_different_recipe(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    selected = plan(repository, Registry())
    script = repository / "models/yolo26x/scripts/convert.py"
    script.write_text(script.read_text() + "\n# changed recipe\n")
    with pytest.raises(ValidationError, match="checkout's recipe"):
        compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources=selected)


def test_conversion_workflow_uses_shared_environment_and_passed_tests():
    workflow = yaml.safe_load((ROOT / ".github/workflows/vulcan-model-convert.yml").read_text())
    job = workflow["jobs"]["convert"]
    assert job["environment"] == "${{ inputs.environment }}"
    assert job["env"]["VULCAN_ENVIRONMENT"] == "${{ inputs.environment }}"
    assert job["runs-on"][3] == "${{ inputs.environment == 'staging' && 'vulcan-stg' || 'vulcan-production' }}"
    # The shared config resolver emits stg/prod, not canonical GitHub environment names.
    assert "needs.config.outputs.vulcan_env" not in str(job)
    tests = next(step for step in job["steps"] if step.get("name", "").startswith("Test the converted"))
    assert '[[ "$outcome" == passed ]]' in tests["run"]
    publish = job["steps"][-1]["run"]
    assert "publish-conversion" in publish
    assert "models-onnx" not in publish and "sima-neat-artifacts-staging" not in publish


def test_on_demand_conversion_resolves_catalog_identity_without_compiling(repository, monkeypatch):
    from model_registry.stage_plan import plan_stages
    path = repository / "models/yolo26x/model.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["build-on-demand"] = True
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setattr("model_registry.stage_plan.read_catalog_sources", lambda *args: (None, []))
    registry = Registry()
    result = plan_stages(repository, compiler_version="test", read=registry.read,
                         read_api_url="https://registry.test", branch="feature", conversion_client=registry)
    assert not result["compile"]["include"]
    assert result["conversion_sources"]["yolo26x"]["run_id"] == "conversion-1"


@pytest.mark.parametrize("reserved", ["manifest.json", ".bundle.tar.gz"])
def test_bundle_rejects_reserved_directory_prefix(tmp_path, reserved):
    directory = tmp_path / "output"
    directory.mkdir()
    (directory / reserved).mkdir()
    other = output(tmp_path / "export", external=True)
    model = onnx.load(other / "model.onnx", load_external_data=False)
    for field in model.graph.initializer[0].external_data:
        if field.key == "location":
            field.value = reserved + "/weights.bin"
    onnx.save(model, directory / "model.onnx")
    shutil.copyfile(other / "weights.bin", directory / reserved / "weights.bin")
    with pytest.raises(ValidationError, match="reserved ONNX bundle path"):
        conversion.bundle_onnx(directory, tmp_path / "bundle.tar.gz")
    assert not (tmp_path / "bundle.tar.gz").exists()


@pytest.mark.parametrize("selected_conversion", [False, True])
@pytest.mark.parametrize("has_previous", [False, True])
@pytest.mark.parametrize("deferred_pending", [False, True])
def test_targeted_plan_publishes_catalog_with_unselected_conversion(
    repository, monkeypatch, selected_conversion, has_previous, deferred_pending,
):
    from model_registry import registry as publication
    from model_registry import stage_plan

    shutil.copytree(ROOT / "models/yolo26l", repository / "models/yolo26l")
    if selected_conversion:
        path = repository / "models/yolo26l/model.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["compilation"]["source"] = "conversion"
        path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setattr(stage_plan, "read_catalog_sources", lambda *args: (None, []))
    monkeypatch.setattr(publication, "resolve_model_compiler", lambda root: {"identity": "test", "version": "test"})
    client = Registry()
    result = stage_plan.plan_stages(repository, compiler_version="test", read=client.read,
                                   read_api_url="https://registry.test", branch="feature",
                                   model_ids=["yolo26l"], conversion_client=client)
    import json
    monkeypatch.setenv("MODEL_REGISTRY_CONVERSIONS", json.dumps(result["conversion_sources"]))
    matrix = repository / "selected.json"
    matrix.write_text(json.dumps(result["compile"]))
    prior = {"id": "yolo26x", "builds": [], "packages": [],
             "build_on_demand": not deferred_pending, "build_requirements": {"unchanged": "requirements"},
             "documentation": {"source_commit": "b" * 40}}
    documentation = {model_id: {"source_commit": "a" * 40, "run_id": model_id + "-docs"}
                     for model_id in ("yolo26x", "yolo26l")}
    monkeypatch.setattr(publication, "_revision_documentation_receipts", lambda *args: None)
    monkeypatch.setattr(publication, "publish_model_documentation", lambda **kwargs: documentation)
    pending = [{"model_id": "yolo26x", "build_profile_id": "w16a16", "compile_invocation_id": "w16a16",
                "component_ids": ["model"], "compile_fingerprint": "c" * 64}] if deferred_pending else []
    sources = [{"models": [prior], "pending_builds": pending}] if has_previous else []
    monkeypatch.setattr(publication, "read_catalog_sources", lambda *args: (None, sources))
    for _ in range(2):  # Both initialization and subsequent catalog publication.
        published = publication.publish_catalog(
            repository=repository, api_url="https://registry.test", read_api_url="https://registry.test",
            aws_region="us-west-2", branch="feature", git_sha="a" * 40,
            bucket="test-artifacts", environment="staging",
            selected_matrix=matrix, request=lambda *args: {"stored": True}, read_request=client.read,
        )["catalog"]
        entries = {model["id"]: model for model in published["models"]}
        assert "yolo26l" in entries
        if has_previous:
            assert entries["yolo26x"] == {**prior, "documentation": documentation["yolo26x"]}
        else:
            assert "yolo26x" not in entries
        expected_pending_models = {"yolo26l", "yolo26x"} if has_previous and deferred_pending else {"yolo26l"}
        assert {build["model_id"] for build in published["pending_builds"]} == expected_pending_models
        if has_previous and deferred_pending:
            assert [build for build in published["pending_builds"] if build["model_id"] == "yolo26x"] == pending
        sources[:] = [published]


def test_bundle_many_external_files_under_low_descriptor_limit(tmp_path):
    import subprocess
    import sys
    import tarfile

    directory = tmp_path / "export"
    directory.mkdir()
    tensors = [onnx.numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name=f"weights_{i}")
               for i in range(100)]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "weights_0"], ["y"])], "many-files",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])], tensors,
    )
    onnx.save_model(onnx.helper.make_model(graph), directory / "model.onnx",
                    save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0)
    archive = tmp_path / "bundle.tar.gz"
    # Isolate the low descriptor limit from pytest and its other tests.
    result = subprocess.run([sys.executable, "-c", """
import resource
import sys
from pathlib import Path
import onnx
from model_registry.conversion import bundle_onnx
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
limit = 64 if hard == resource.RLIM_INFINITY else min(64, hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
bundle_onnx(Path(sys.argv[1]), Path(sys.argv[2]))
""", str(directory), str(archive)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with tarfile.open(archive) as bundle:
        assert len(bundle.getmembers()) == 102
        for tensor in tensors:
            assert bundle.extractfile(tensor.name).read() == (directory / tensor.name).read_bytes()


def test_bundle_closes_streams_when_archive_write_fails(tmp_path, monkeypatch):
    import tarfile

    directory = output(tmp_path / "export", external=True)
    archive = tmp_path / "bundle.tar.gz"
    opened = []
    original_open = Path.open
    original_addfile = tarfile.TarFile.addfile

    def track_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.parent == directory or path == archive:
            opened.append(stream)
        return stream

    def fail_tensor_write(self, info, stream=None):
        if info.name == "weights.bin":
            raise OSError("simulated disk failure")
        return original_addfile(self, info, stream)

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(tarfile.TarFile, "addfile", fail_tensor_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        conversion.bundle_onnx(directory, archive)
    assert opened and all(stream.closed for stream in opened)


def test_offline_browser_preview_supports_conversion_models(repository, monkeypatch):
    import runpy

    shutil.copytree(ROOT / "deps", repository / "deps")
    preview = runpy.run_path(str(ROOT / "artifact-browser/preview.py"))
    namespace = preview["_repository"].__wrapped__.__globals__
    monkeypatch.setitem(namespace, "ROOT", repository)
    monkeypatch.setenv("MODEL_REGISTRY_CONVERSIONS", '{"untouched": true}')
    result = preview["_repository"]()
    assert result["revisions"][-1]["document"]["models"][0]["id"] == "yolo26x"
    assert result["revisions"][-1]["document"]["models"][0]["builds"]
    assert result["models"]["yolo26x"]["components"][0]["source"]["sha256"]
    import os
    assert os.environ["MODEL_REGISTRY_CONVERSIONS"] == '{"untouched": true}'
