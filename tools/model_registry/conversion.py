"""Conversion publication and immutable compile selections using ordinary registry runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from .errors import ValidationError
from .fetch import sha256_file
from .model import load_and_validate_model
from .source import source_identity

REPOSITORY_PATH = "/v1/repositories/sima-neat/models"
ARTIFACT = "onnx"
FIELDS = ("s3_bucket", "s3_key", "s3_version_id", "sha256", "size_bytes", "media_type")

# Shared execution and validation dependencies. Keep this explicit so unrelated
# registry/UI changes do not invalidate conversions; update it with new imports.
CONVERSION_TOOLING = (
    ".github/workflows/vulcan-model-convert.yml",
    "scripts/run_model_conversion.sh",
    "scripts/run_model_tests.sh",
    "conftest.py",
    "pyproject.toml",
    "tools/model_registry/conversion.py",
    "tools/model_registry/fetch.py",
    "tools/model_registry/registry_http.py",
    "tools/model_registry/source.py",
    "tools/model_registry/model.py",
    "tools/model_registry/model_tests.py",
    "tools/model_registry/benchmark_declarations.py",
    "tools/model_registry/taxonomy.py",
    "tools/model_registry/errors.py",
    "tools/model_report/__init__.py",
    "tools/model_report/metrics.json",
    "schemas/model.schema.json",
    "schemas/model-tests.schema.json",
    "schemas/dataset-requirements.schema.json",
    "taxonomies/model-taxonomy.yaml",
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def conversion_fingerprint(root: Path, model: dict) -> str:
    directory = root / "models" / model["id"]
    paths = [
        directory / "scripts/convert.py",
        directory / "scripts/requirements-convert.txt",
        *(root / path for path in CONVERSION_TOOLING),
    ]
    # Include local helpers, exporter patches, and validation, but not compile/benchmark recipes.
    paths += [
        p
        for p in (directory / "scripts").rglob("*")
        if p.is_file() and p.name not in {"compile.py", "accuracy.py", "performance.py", "requirements-compile.txt"}
    ]
    # Validation can consume images, arrays, or arbitrary binary fixtures.
    paths += [p for p in (directory / "tests").rglob("*") if p.is_file()]
    files = {}
    for path in sorted(set(paths)):
        if {"__pycache__", ".pytest_cache"}.intersection(path.relative_to(root).parts):
            continue
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"conversion input must be a regular file: {path}")
        files[path.relative_to(root).as_posix()] = sha256_file(path)[0]
    return digest(
        {
            "version": 1,
            "model_id": model["id"],
            "source": source_identity(model["conversion"]["default"]["source"]),
            "files": files,
        }
    )


def selections() -> dict:
    try:
        value = json.loads(os.environ.get("MODEL_REGISTRY_CONVERSIONS", "{}"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except ValueError as exc:
        raise ValidationError("MODEL_REGISTRY_CONVERSIONS must contain the saved plan JSON") from exc


def apply_selection(model: dict, selected: dict | None = None, *, root: Path | None = None) -> None:
    if model["components"][0]["source"]["type"] != "conversion":
        return
    selected = selections() if selected is None else selected
    item = selected.get(model["id"])
    if not isinstance(item, dict):
        raise ValidationError(f"{model['id']}: resolve conversion with plan-stages before compilation")
    if root is not None and item.get("conversion_fingerprint") != conversion_fingerprint(root, model):
        raise ValidationError(f"{model['id']}: saved conversion does not match this checkout's recipe")
    for component in model["components"]:
        component["source"] = {"type": "conversion", **item}


def _artifact(read, base: str, run_id: str) -> dict:
    item = read(f"{base.rstrip('/')}/v1/runs/{quote(run_id, safe='')}/artifacts/{ARTIFACT}")["artifact"]
    if item.get("object_status") != "available" or item.get("artifact_type") != "model-onnx":
        raise ValidationError("Selected conversion output is unavailable; run conversion again")
    if not item.get("s3_version_id") or item.get("metadata", {}).get("validation") != "passed":
        raise ValidationError("Selected conversion output lacks an exact validated object version")
    return item


def input_record(item: dict) -> dict:
    return {
        **{key: item[key] for key in FIELDS},
        "input_type": "onnx",
        "source_reference": f"run:{item['run_id']}/artifacts/{ARTIFACT}",
        "metadata": {
            "conversion_run_id": item["run_id"],
            "artifact_id": item["artifact_id"],
            "conversion_fingerprint": item["conversion_fingerprint"],
        },
    }


def _resolve(read, base, run_type, fingerprint):
    from .stage_plan import _is_not_found

    try:
        return read(f"{base.rstrip('/')}{REPOSITORY_PATH}/resolve?run_type={run_type}&fingerprint={fingerprint}")["run"]
    except Exception as exc:
        if _is_not_found(exc, "fingerprint_not_found"):
            return None
        raise


def plan_conversions(root, models, *, read, read_api_url, client=None, branch="") -> dict:
    result = {}
    for model_id, model in models.items():
        if model["components"][0]["source"]["type"] != "conversion":
            continue
        if read is None or not read_api_url or client is None:
            raise ValidationError(f"{model_id}: conversion planning requires registry reader and publisher credentials")
        fingerprint = conversion_fingerprint(root, model)
        workflow = os.environ.get("GITHUB_RUN_ID", "")
        commit = os.environ.get("GITHUB_SHA", "")
        if not workflow or not commit:
            raise ValidationError("conversion planning requires GITHUB_RUN_ID and GITHUB_SHA for retry identity")
        plan_fingerprint = digest([workflow, commit, model_id, fingerprint])
        key = f"conversion-plan-{plan_fingerprint}"
        plan = client.request(
            "POST",
            "/v1/runs",
            {
                "run_type": "model_compile_plan",
                "commit_sha": commit,
                "git_ref": f"refs/heads/{branch}",
                "run_fingerprint": plan_fingerprint,
                "environment": {"fingerprint": digest(["conversion-plan", read_api_url])},
                "metadata": {"model_id": model_id, "github_run_id": workflow},
            },
            key + "-create",
        )
        path = f"/v1/runs/{plan['run_id']}"
        provenance = read(f"{read_api_url.rstrip('/')}{path}/provenance")
        prior = next((i for i in provenance["inputs"] if i["name"] == "conversion"), None)
        if prior:
            producer = prior["metadata"]["conversion_run_id"]
        else:
            conversion = _resolve(read, read_api_url, "model_conversion", fingerprint)
            if not conversion:
                raise ValidationError(f"{model_id}: no matching validated conversion; run Vulcan Model Convert first")
            producer = conversion["id"]
        artifact = _artifact(read, read_api_url, producer)
        selection = {key: artifact[key] for key in FIELDS}
        selection.update(
            run_id=producer,
            artifact_id=artifact["id"],
            conversion_fingerprint=fingerprint,
            read_api_url=read_api_url.rstrip("/"),
        )
        if prior and any(prior[field] != selection[field] for field in FIELDS):
            raise ValidationError("Recorded conversion object changed; refusing to reselect latest")
        if provenance["run"]["status"] not in {"running", "completed"}:
            raise ValidationError("Compile plan expired; start a new workflow or use its saved selection")
        if provenance["run"]["status"] != "completed":
            client.request(
                "PUT",
                path + "/parameters",
                {
                    "parameters": {
                        "model_id": model_id,
                        "conversion_fingerprint": fingerprint,
                    }
                },
                key + "-parameters",
            )
            client.request("PUT", path + "/inputs/conversion", input_record(selection), key + "-input")
            client.request("POST", path + "/finalize", {"status": "completed"}, key + "-finalize")
        result[model_id] = selection
    return result


def _safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")) or "\\" in name:
        raise ValidationError(f"unsafe ONNX bundle path: {name}")
    return name


def bundle_onnx(directory: Path, output: Path) -> Path:
    import onnx

    entrypoint = directory / "model.onnx"
    model = onnx.load(str(entrypoint), load_external_data=False)
    names = {"model.onnx"}

    def visit(message):
        if isinstance(message, onnx.TensorProto):
            for field in message.external_data:
                if field.key == "location":
                    name = _safe_path(field.value)
                    if PurePosixPath(name).parts[0] in {"manifest.json", ".bundle.tar.gz"}:
                        raise ValidationError(f"reserved ONNX bundle path: {name}")
                    names.add(name)
        for descriptor, value in message.ListFields():
            if descriptor.type == descriptor.TYPE_MESSAGE:
                for child in value if descriptor.is_repeated else [value]:
                    visit(child)

    visit(model)
    files = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file() or path.resolve() != directory.resolve() / name or path.is_symlink():
            raise ValidationError(f"ONNX data must be regular files within output directory: {name}")
        if path.resolve() == output.resolve() or (output.exists() and path.samefile(output)):
            raise ValidationError(f"ONNX bundle output would overwrite model data: {name}")
        sha, size = sha256_file(path)
        files.append({"path": name, "sha256": sha, "size_bytes": size})
    onnx.checker.check_model(str(entrypoint))
    manifest = json.dumps({"version": 1, "entrypoint": "model.onnx", "files": files}, sort_keys=True).encode()
    import io

    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for item in [{"path": "manifest.json", "size_bytes": len(manifest)}, *files]:
                name = item["path"]
                with io.BytesIO(manifest) if name == "manifest.json" else (directory / name).open("rb") as stream:
                    info = tarfile.TarInfo(name)
                    info.size = item["size_bytes"]
                    info.mode = 0o644
                    archive.addfile(info, stream)
    return output


def _verify_bundle(directory: Path, source: dict | None = None) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if source is not None:
        archive_path = directory / ".bundle.tar.gz"
        if sha256_file(archive_path) != (source["sha256"], source["size_bytes"]):
            raise ValidationError("Cached conversion bundle checksum or size mismatch")
        with tarfile.open(archive_path, "r:gz") as archive:
            if json.load(archive.extractfile("manifest.json")) != manifest:
                raise ValidationError("Cached conversion manifest differs from the verified bundle")
    names = [item["path"] for item in manifest["files"]]
    if (
        manifest["version"] != 1
        or manifest["entrypoint"] != "model.onnx"
        or "model.onnx" not in names
        or len(names) != len(set(names))
    ):
        raise ValidationError("Invalid conversion bundle manifest")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if set(names) | {"manifest.json"} | ({".bundle.tar.gz"} if source else set()) != actual:
        raise ValidationError("Conversion bundle members differ from manifest")
    for item in manifest["files"]:
        path = directory / _safe_path(item["path"])
        if path.is_symlink() or sha256_file(path) != (item["sha256"], item["size_bytes"]):
            raise ValidationError("Conversion bundle file checksum or size mismatch")


def fetch_conversion(source: dict, cache_dir: Path | None = None):
    from .fetch import default_cache_dir
    from .registry_http import read_registry_json_authenticated

    artifact = _artifact(read_registry_json_authenticated, source["read_api_url"], source["run_id"])
    if any(artifact[key] != source[key] for key in FIELDS) or artifact["id"] != source["artifact_id"]:
        raise ValidationError("Conversion reference no longer matches the saved plan")
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "conversions" / source["sha256"]
    if destination.exists():
        _verify_bundle(destination, source)
        return destination / "model.onnx", True, source["sha256"], source["size_bytes"]
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        bundle = Path(tmp) / "onnx.tar.gz"
        with urlopen(artifact["download_url"], timeout=120) as response, bundle.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256_file(bundle) != (source["sha256"], source["size_bytes"]):
            raise ValidationError("Conversion download checksum or size mismatch")
        unpacked = Path(tmp) / "output"
        unpacked.mkdir()
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
                raise ValidationError("ONNX bundle must contain unique regular files only")
            for member in members:
                path = unpacked / _safe_path(member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        _verify_bundle(unpacked)
        bundle.rename(unpacked / ".bundle.tar.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            unpacked.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            _verify_bundle(destination, source)
    return destination / "model.onnx", False, source["sha256"], source["size_bytes"]


def publish_conversion(
    root: Path,
    model_id: str,
    directory: Path,
    *,
    client,
    store,
    bucket: str,
    environment: str,
    branch: str,
    git_sha: str,
    workflow: str,
    validation: str,
) -> dict:
    if validation != "passed":
        raise ValidationError("Conversion publication requires passed conversion tests")
    model = load_and_validate_model(root / "models" / model_id)
    fingerprint = conversion_fingerprint(root, model)
    bundle = bundle_onnx(directory, directory / "onnx.tar.gz")
    sha, size = sha256_file(bundle)
    key = "conversion-" + digest([workflow, environment, model_id, fingerprint, sha])
    run = client.request(
        "POST",
        "/v1/runs",
        {
            "run_type": "model_conversion",
            "commit_sha": git_sha,
            "git_ref": f"refs/heads/{branch}",
            "run_fingerprint": fingerprint,
            "environment": {"fingerprint": digest(["conversion", environment])},
            "metadata": {"model_id": model_id, "github_run_id": workflow},
        },
        key + "-create",
    )
    path = f"/v1/runs/{run['run_id']}"
    upstream = model["conversion"]["default"]["source"]
    client.request(
        "PUT",
        path + "/inputs/upstream",
        {
            "input_type": "model-source",
            "source_reference": model["upstream"]["repository"],
            "sha256": source_identity(upstream),
            "metadata": {"source": upstream},
        },
        key + "-input",
    )
    client.request(
        "PUT",
        path + "/parameters",
        {
            "parameters": {
                "model_id": model_id,
                "conversion_fingerprint": fingerprint,
                "conversion": model["conversion"],
            },
        },
        key + "-parameters",
    )
    stored = store.ensure_native_sha256(bundle, f"models/conversions/{model_id}/{sha}/onnx.tar.gz", sha)
    client.request(
        "PUT",
        path + "/artifacts/onnx",
        {
            "artifact_type": "model-onnx",
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": sha,
            "size_bytes": size,
            "media_type": "application/gzip",
            "retention_class": "branch",
            "metadata": {"validation": "passed", "entrypoint": "model.onnx", "bundle_version": 1},
        },
        key + "-artifact",
    )
    client.request(
        "PUT",
        path + "/test-results/correctness/conversion",
        {
            "status": "passed",
            "summary": {"onnx_checker": "passed", "model_tests": validation},
        },
        key + "-test",
    )
    client.request("POST", path + "/finalize", {"status": "completed"}, key + "-finalize")
    return {"run_id": run["run_id"], "conversion_fingerprint": fingerprint, "sha256": sha, "size_bytes": size}
