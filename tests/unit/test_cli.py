from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from model_registry.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_validate_json_output(capsys) -> None:
    result = main(["validate", str(ROOT / "models" / "resnet_50"), "--json"])
    assert result == 0
    output = capsys.readouterr().out
    assert '"model_id": "resnet_50"' in output
    assert '"valid": true' in output


def test_validate_catalog_checks_every_manifest(capsys) -> None:
    result = main(["validate-catalog", "--repository", str(ROOT), "--json"])
    assert result == 0
    output = capsys.readouterr().out
    assert '"model_count": 5' in output
    assert '"ssd_mobilenet_v3"' in output
    assert '"valid": true' in output


def test_invalid_model_has_nonzero_exit(capsys) -> None:
    result = main(["validate", str(ROOT / "tests" / "fixtures" / "invalid_model")])
    assert result == 2
    assert "source.sha256" in capsys.readouterr().err


def test_matrix_emits_workflow_json(capsys) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--changed-path", "models/resnet_50/README.md"])
    assert result == 0
    output = capsys.readouterr().out
    assert '"model_id":"resnet_50"' in output
    assert '"recipe":"compile.py"' in output


def test_matrix_rejects_unknown_manual_model(capsys) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--models", "does_not_exist"])
    assert result == 2
    assert "unknown model id" in capsys.readouterr().err


def test_matrix_selects_one_ssd_variant(capsys) -> None:
    variant = "modalix_int8_tess_mla"
    result = main(
        [
            "matrix",
            "--repository",
            str(ROOT),
            "--models",
            "ssd_mobilenet_v1",
            "--variants",
            variant,
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert f'"variant_id":"{variant}"' in output
    assert '"model_id":"ssd_mobilenet_v1"' in output


def test_package_supports_python_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "model_registry", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Model Registry lifecycle tooling" in result.stdout
