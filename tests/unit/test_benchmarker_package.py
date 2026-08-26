from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_benchmarker_vulcan_package.sh"
INSTALL_SCRIPT = ROOT / "packaging" / "benchmarker" / "install_benchmarker_package.sh"
DEPENDENCIES = ROOT / "packaging" / "benchmarker" / "dependencies.json"


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


def test_builds_benchmarker_wheel_and_vulcan_payload(tmp_path: Path) -> None:
    cli = tmp_path / "sima-cli"
    output = tmp_path / "dist"
    fake_sima_cli(cli)
    env = {
        **os.environ,
        "BENCHMARKER_BUILD_PYTHON": sys.executable,
        "BENCHMARKER_PACKAGE_VERSION": "0.1.0+unit",
        "SIMA_CLI_BIN": str(cli),
    }

    subprocess.run([str(BUILD_SCRIPT), str(output)], cwd=ROOT, env=env, check=True)

    wheels = list(output.glob("sima_neat_benchmarker-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()
        assert "Name: sima-neat-benchmarker" in metadata
        assert "Version: 0.1.0+unit" in metadata
        assert "benchmarker/__init__.py" in names
        assert "benchmarker/schemas/performance-test-v1.schema.json" in names
        assert "benchmarker/schemas/performance-result-v1.schema.json" in names
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "benchmarker = benchmarker:main" in wheel.read(entry_points).decode()

    dependencies = json.loads((output / "dependencies.json").read_text(encoding="utf-8"))
    assert dependencies == {
        "schema_version": 1,
        "platform_version": "2.1.3",
        "dependencies": {"neat-core": {"version": "0.4.0", "install_target": "core@v0.4.0"}},
    }
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {"name": "gh:sima-neat/models/benchmarker", "version": "0.1.0+unit"}


def test_package_installer_uses_declared_core_dependency() -> None:
    dependencies = json.loads(DEPENDENCIES.read_text(encoding="utf-8"))
    assert dependencies == {
        "schema_version": 1,
        "platform_version": "2.1.3",
        "dependencies": {"neat-core": {"version": "0.4.0", "install_target": "core@v0.4.0"}},
    }
    script = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert '"${SIMA_CLI_BIN}" neat install "${CORE_TARGET}"' in script
    assert '"${PYNEAT_PYTHON}" -m pip install --no-deps --force-reinstall' in script
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)
