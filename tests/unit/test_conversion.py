from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from model_registry import conversion
from model_registry.errors import ModelRegistryError, ValidationError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_ROOT", str(ROOT))
    directory = tmp_path / "example_model"
    directory.mkdir()
    declaration = {
        "schema_version": 1,
        "source": {
            "type": "file",
            "uri": "https://storage.googleapis.com/rfdetr/small_coco/model.pth",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "requirements": "requirements.txt",
        },
    }
    (directory / "convert.yaml").write_text(yaml.safe_dump(declaration))
    (directory / "requirements.txt").write_text("example==1 --hash=sha256:" + "0" * 64 + "\n")
    (directory / "convert.py").write_text("""import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--source')
parser.add_argument('--output-dir')
args = parser.parse_args()
assert sys.prefix != sys.base_prefix
assert 'PYTHONPATH' not in os.environ
assert os.environ['PYTHONNOUSERSITE'] == '1'
output = Path(args.output_dir)
output.mkdir()
(output / 'model.onnx').write_bytes(b'fixture-validated-by-model')
(output / 'validation.json').write_text(json.dumps({'status':'passed'}))
""")
    return directory


def test_plan_changes_with_config_lock_and_helpers(model):
    first = conversion.validate_conversion(model)
    assert first["model_id"] == "example_model"
    assert first["environment"]["requirements_sha256"] == first["files"]["requirements.txt"]
    (model / "helper.py").write_text("export_option = 1\n")
    assert conversion.validate_conversion(model)["fingerprint"] != first["fingerprint"]


@pytest.mark.parametrize(
    "case", ["unknown", "traversal", "duplicate", "missing_entrypoint", "reserved", "bundle", "symlink"]
)
def test_rejects_invalid_declarations(model, case):
    declaration = yaml.safe_load((model / "convert.yaml").read_text())
    if case == "unknown":
        declaration["runner"] = "ultra-expensive"
    elif case == "traversal":
        declaration["environment"]["requirements"] = "../requirements.txt"
    elif case == "bundle":
        declaration["source"]["type"] = "onnx_bundle"
    elif case == "symlink":
        (model / "convert.py").unlink()
        (model / "convert.py").symlink_to(model / "requirements.txt")
    else:
        member = {key: value for key, value in declaration["source"].items() if key != "type"}
        member["path"] = "model.pth"
        declaration["source"] = {"type": "file_set", "entrypoint": "model.pth", "files": [member]}
        if case == "duplicate":
            declaration["source"]["files"].append({**member, "path": "MODEL.pth"})
        elif case == "reserved":
            declaration["source"]["files"][0]["path"] = "source-set.json"
            declaration["source"]["entrypoint"] = "source-set.json"
        else:
            declaration["source"]["entrypoint"] = "missing.pth"
    (model / "convert.yaml").write_text(yaml.safe_dump(declaration))
    with pytest.raises(ValidationError):
        conversion.validate_conversion(model)


def test_convert_isolates_python_and_records_provenance(model, tmp_path, monkeypatch):
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"x")
    monkeypatch.setattr(conversion.Fetcher, "fetch_source", lambda *args: (source, True, "a" * 64, 1))
    monkeypatch.setenv("PYTHONPATH", "/must-not-reach-converter")
    original = subprocess.run
    calls = []

    def run(command, **kwargs):
        calls.append((command, dict(kwargs["env"])))
        if "pip" in command and "install" in command:
            assert "--require-hashes" in command
            assert "--only-binary=:all:" in command
            assert "--isolated" in command
            assert "--cache-dir" in command
            return subprocess.CompletedProcess(command, 0)
        return original(command, **kwargs)

    monkeypatch.setattr(conversion.subprocess, "run", run)
    result = conversion.convert_model(
        model, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache", python=sys.executable
    )
    assert Path(result["output_dir"], "provenance.json").is_file()
    assert result["provenance"]["source_identity"] == "a" * 64
    assert all("PYTHONPATH" not in env for _, env in calls)
    assert all("--system-site-packages" not in command for command, _ in calls)
    assert all(command[1:3] == ["-E", "-s"] for command, _ in calls)
    pip_env = next(env for command, env in calls if "pip" in command and "install" in command)
    assert Path(pip_env["PIP_CACHE_DIR"]).is_relative_to(tmp_path / "cache" / "wheels")
    with pytest.raises(ValidationError, match="must be new"):
        conversion.convert_model(model, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")


def test_failed_dependency_install_retains_diagnostics(model, tmp_path, monkeypatch):
    monkeypatch.setattr(conversion.Fetcher, "fetch_source", lambda *args: (tmp_path / "source", False, "a" * 64, 1))

    def run(command, **kwargs):
        if "-c" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "python": f"{sys.version_info.major}.{sys.version_info.minor}.0",
                        "system": "Linux",
                        "machine": "aarch64",
                    }
                ),
            )
        if "pip" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(conversion.subprocess, "run", run)
    with pytest.raises(ModelRegistryError, match="Conversion failed"):
        conversion.convert_model(model, work_dir=tmp_path / "failure", cache_dir=tmp_path / "cache")
    assert json.loads((tmp_path / "failure" / "failure.json").read_text())["status"] == "failed"
    assert not (tmp_path / "failure" / "export" / "provenance.json").exists()


@pytest.mark.parametrize("failure", ["report", "missing", "symlink", "exit"])
def test_converter_failure_never_produces_approved_provenance(model, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(conversion.Fetcher, "fetch_source", lambda *args: (tmp_path / "source", False, "a" * 64, 1))
    output = tmp_path / "work" / "export"

    def run(command, **kwargs):
        if "-c" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "python": f"{sys.version_info.major}.{sys.version_info.minor}.0",
                        "system": "Linux",
                        "machine": "aarch64",
                    }
                ),
            )
        if any(str(arg).endswith("convert.py") for arg in command):
            output.mkdir()
            if failure == "exit":
                raise subprocess.CalledProcessError(1, command)
            (output / "validation.json").write_text(
                json.dumps({"status": "failed" if failure == "report" else "passed"})
            )
            if failure == "symlink":
                (output / "model.onnx").symlink_to(output / "validation.json")
            elif failure != "missing":
                (output / "model.onnx").write_bytes(b"graph")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(conversion.subprocess, "run", run)
    with pytest.raises(ModelRegistryError, match="Conversion failed"):
        conversion.convert_model(model, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    assert not (output / "provenance.json").exists()
    assert (tmp_path / "work" / "failure.json").is_file()
