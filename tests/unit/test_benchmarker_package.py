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
DEPENDENCIES = ROOT / "deps" / "manifest.json"


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
board_platform = sys.argv[sys.argv.index("--board-platform") + 1]
metadata = {{"name": name, "version": version, "board_platform": board_platform}}
(out / "metadata.json").write_text(json.dumps(metadata) + "\\n")
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
        "BENCHMARKER_PACKAGE_VERSION": "0.2.0+unit",
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
        assert "Version: 0.2.0+unit" in metadata
        assert "benchmarker/__init__.py" in names
        assert "benchmarker/schemas/performance-test-v1.schema.json" in names
        assert "benchmarker/schemas/performance-result-v1.schema.json" in names
        assert "benchmarker/schemas/performance-result-v2.schema.json" in names
        assert "benchmarker/schemas/performance-result-v3.schema.json" in names
        assert "benchmarker/schemas/performance-result-v4.schema.json" in names
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "benchmarker = benchmarker:main" in wheel.read(entry_points).decode()

    dependencies = json.loads((output / "dependencies.json").read_text(encoding="utf-8"))
    source = json.loads(DEPENDENCIES.read_text(encoding="utf-8"))
    assert dependencies == {
        "schema_version": 1,
        "platform_version": source["platform-version"],
        "dependencies": {"neat-core": source["neat-core"]},
    }
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "name": "gh:sima-neat/models/benchmarker",
        "version": "0.2.0+unit",
        "board_platform": f"modalix@=={source['platform-version']}",
    }


def test_package_installer_uses_declared_core_dependency() -> None:
    script = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert '"${SIMA_CLI_BIN}" neat install -t minimal "${CORE_TARGET}"' in script
    assert '"${PYNEAT_PYTHON}" -m pip install --no-deps --force-reinstall' in script
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)
