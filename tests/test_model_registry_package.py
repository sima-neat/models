from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "packaging" / "model-registry" / "build.sh"
INSTALL_SCRIPT = ROOT / "packaging" / "model-registry" / "install_model_registry_package.sh"
RESOLVER = ROOT / "packaging" / "model-registry" / "resolve_model_compiler_python.sh"


def fake_sima_cli(path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

if sys.argv[1:3] != ["packages", "build"]:
    raise SystemExit(f"unexpected arguments: {{sys.argv[1:]}}")
out = pathlib.Path(sys.argv[3])
name = sys.argv[sys.argv.index("--name") + 1]
version = sys.argv[sys.argv.index("--version") + 1]
(out / "metadata.json").write_text(json.dumps({{"name": name, "version": version}}) + "\\n")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def fake_model_compiler(path: Path) -> Path:
    bin_dir = path / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    (bin_dir / "mla-masm").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "mla-masm").chmod(0o755)
    return python


def resolve_with(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **environment}
    for name in (
        "MODEL_COMPILER_HOME",
        "MODEL_COMPILER_PYTHON",
        "NEAT_SDK_EXTENSIONS_DIR",
        "SIMA_SDK_EXTENSIONS_DIR",
        "VIRTUAL_ENV",
    ):
        if name not in environment:
            env.pop(name, None)
    return subprocess.run(
        ["/bin/bash", "-c", f"source {shlex.quote(str(RESOLVER))}; resolve_model_compiler_python"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_builds_author_wheel_and_vulcan_payload(tmp_path: Path) -> None:
    cli = tmp_path / "sima-cli"
    output = tmp_path / "dist"
    fake_sima_cli(cli)
    env = {
        **os.environ,
        "MODEL_REGISTRY_BUILD_PYTHON": sys.executable,
        "MODEL_REGISTRY_PACKAGE_VERSION": "0.1.0+unit",
        "SIMA_CLI_BIN": str(cli),
    }

    subprocess.run([str(BUILD_SCRIPT), str(output)], cwd=ROOT, env=env, check=True)

    wheels = list(output.glob("sima_neat_model_registry-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()
        assert "Name: sima-neat-model-registry" in metadata
        assert "Version: 0.1.0+unit" in metadata
        assert "Requires-Dist: jsonschema<5,>=4.18.4" in metadata
        assert "model_registry/cli.py" in names
        assert "model_registry/conversion.py" in names
        assert "model_registry/registry_http.py" in names
        assert "model_report/metrics.json" in names
        for ci_module in ("report_publication", "registry", "candidates", "promotions"):
            assert f"model_registry/{ci_module}.py" not in names
        assert "author_main" in wheel.read("model_registry/__main__.py").decode()
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "model-registry = model_registry.cli:author_main" in wheel.read(entry_points).decode()
        wheel.extractall(tmp_path / "installed-wheel")

    # Execute the shipped wheel, where CI publication modules do not exist.
    subprocess.run([sys.executable, "-c", r"""
import hashlib, io, json, sys, tarfile, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from model_registry import conversion, fetch
assert 'model_registry.registry' not in sys.modules
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    content = b'author-package-fetch-fixture'
    manifest = {'version': 1, 'entrypoint': 'model.onnx', 'files': [
        {'path': 'model.onnx', 'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}]}
    bundle = root / 'bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for name, data in [('manifest.json', json.dumps(manifest).encode()), ('model.onnx', content)]:
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    sha, size = fetch.sha256_file(bundle)
    source = {'run_id': 'run', 'artifact_id': 'artifact', 'read_api_url': 'https://public.example.test',
              's3_bucket': 'bucket', 's3_key': 'key', 's3_version_id': 'v1', 'sha256': sha,
              'size_bytes': size, 'media_type': 'application/gzip'}
    artifact = {**source, 'id': 'artifact', 'artifact_type': 'model-onnx', 'object_status': 'available',
                'metadata': {'validation': 'passed'}, 'download_url': bundle.as_uri()}
    fetch.read_registry_json = lambda url: {'artifact': artifact}
    path, hit, _, _ = conversion.fetch_conversion(source, root / 'cache')
    assert not hit and path.read_bytes() == content
    assert conversion.fetch_conversion(source, root / 'cache')[1]
""", str(tmp_path / "installed-wheel")], cwd=tmp_path, check=True)

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {"name": "gh:sima-neat/models/model-registry", "version": "0.1.0+unit"}
    assert (output / "install_model_registry_package.sh").stat().st_mode & 0o111
    assert (output / "resolve_model_compiler_python.sh").stat().st_mode & 0o111


@pytest.mark.parametrize(
    "unsafe_output",
    (
        str(ROOT),
        f"{ROOT}/",
        str(ROOT / ".." / ROOT.name),
        str(ROOT.parent),
    ),
)
def test_package_build_rejects_repository_root_and_ancestors(unsafe_output: str) -> None:
    result = subprocess.run(
        [str(BUILD_SCRIPT), unsafe_output],
        cwd=ROOT,
        env={**os.environ, "MODEL_REGISTRY_BUILD_PYTHON": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Refusing unsafe model-registry output directory" in result.stderr
    assert (ROOT / "pyproject.toml").is_file()


def test_installer_uses_the_model_compiler_environment_without_changing_dependencies() -> None:
    script = INSTALL_SCRIPT.read_text(encoding="utf-8")
    resolver = RESOLVER.read_text(encoding="utf-8")

    assert 'source "${RESOLVER}"' in script
    assert '"/sdk-extensions/model-compiler/bin/python"' in resolver
    assert '"/sdk-add-on/model-compiler/bin/python"' in resolver
    assert '"${HOME}/sima-sdk-extensions/model-compiler/bin/python"' in resolver
    assert '"${HOME}/sdk-extensions/model-compiler/bin/python"' in resolver
    assert "NEAT_SDK_EXTENSIONS_DIR" in resolver
    assert "MODEL_COMPILER_HOME" in resolver
    assert "import afe" in resolver
    assert '"${candidate_bin}/mla-masm"' in resolver
    assert "-m pip install --no-deps --force-reinstall" in script
    assert "from jsonschema import Draft202012Validator, FormatChecker" in script
    assert 'version("jsonschema")' not in script
    assert '"${MODEL_COMPILER_PYTHON_BIN}" -m model_registry --help' in script
    assert "python" not in script.split("-m pip install", 1)[1].splitlines()[0].replace(
        '"${MODEL_COMPILER_PYTHON_BIN}"', ""
    )
    assert "-m venv" not in script
    assert ".local/bin/model-registry" not in script
    assert "publish-report" in script
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(RESOLVER)], check=True)
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)


def test_resolver_supports_explicit_model_compiler_python(tmp_path: Path) -> None:
    python = fake_model_compiler(tmp_path / "compiler")
    result = resolve_with({"MODEL_COMPILER_PYTHON": str(python)})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)


def test_resolver_supports_standalone_model_compiler_home(tmp_path: Path) -> None:
    compiler = tmp_path / "standalone" / "model-compiler"
    python = fake_model_compiler(compiler)
    result = resolve_with({"MODEL_COMPILER_HOME": str(compiler)})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)


def test_resolver_prefers_an_active_model_compiler_environment(tmp_path: Path) -> None:
    active = tmp_path / "active-model-compiler"
    python = fake_model_compiler(active)
    result = resolve_with({"VIRTUAL_ENV": str(active)})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)


def test_resolver_supports_configured_neat_sdk_extensions(tmp_path: Path) -> None:
    extensions = tmp_path / "custom-sdk-extensions"
    python = fake_model_compiler(extensions / "model-compiler")
    result = resolve_with({"NEAT_SDK_EXTENSIONS_DIR": str(extensions)})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)


def test_resolver_supports_default_neat_sdk_host_layout(tmp_path: Path) -> None:
    python = fake_model_compiler(tmp_path / "sima-sdk-extensions" / "model-compiler")
    result = resolve_with({"HOME": str(tmp_path)})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)


def test_resolver_supports_default_standalone_user_layout(tmp_path: Path) -> None:
    python = fake_model_compiler(tmp_path / "sdk-extensions" / "model-compiler")
    result = resolve_with({"HOME": str(tmp_path)})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)
