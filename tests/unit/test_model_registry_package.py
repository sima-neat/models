from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_model_registry_vulcan_package.sh"
INSTALL_SCRIPT = ROOT / "packaging" / "model-registry" / "install_model_registry_package.sh"


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
        assert "model_registry/cli.py" in names
        for ci_module in ("benchmark_manifest", "benchmark_publication", "package", "registry"):
            assert f"model_registry/{ci_module}.py" not in names
        assert "author_main" in wheel.read("model_registry/__main__.py").decode()
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "model-registry = model_registry.cli:author_main" in wheel.read(entry_points).decode()

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {"name": "gh:sima-neat/models/model-registry", "version": "0.1.0+unit"}
    assert (output / "install_model_registry_package.sh").stat().st_mode & 0o111


def test_installer_uses_an_isolated_environment_and_user_launcher() -> None:
    script = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert '${HOME}/.local/share/sima-neat/model-registry' in script
    assert '"${PYTHON_BIN}" -m venv "${INSTALL_ROOT}/venv"' in script
    assert "-m pip install --upgrade --force-reinstall" in script
    assert '"${BIN_DIR}/model-registry"' in script
    assert '"${INSTALL_ROOT}/venv/bin/python" -m model_registry --help' in script
    assert "publish-benchmark" in script
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)
