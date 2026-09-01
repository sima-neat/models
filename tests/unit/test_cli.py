from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest
from model_registry import cli
from model_registry.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_author_parser_exposes_only_supported_model_contributor_commands() -> None:
    parser = cli._parser(author_safe=True)
    subcommands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))

    assert set(subcommands.choices) == cli.AUTHOR_COMMANDS


@pytest.mark.parametrize(
    "command",
    [
        "fetch-previous",
        "make-benchmark-manifest",
        "package",
        "publish",
        "publish-benchmark",
        "publish-taxonomy",
        "resolve-base",
        "resolve-benchmark-runtime",
    ],
)
def test_author_cli_rejects_ci_only_commands(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.author_main([command])

    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_repository_cli_retains_ci_only_commands() -> None:
    parser = cli._parser()
    subcommands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))

    assert {"package", "publish", "publish-benchmark", "publish-taxonomy"} <= set(subcommands.choices)


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


def test_fetch_calibration_resolves_registered_dataset(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "imagenet-mini.tgz"
    dataset.write_bytes(b"dataset")
    monkeypatch.setattr("model_registry.cli.resolve_dataset", lambda *args, **kwargs: dataset)

    result = main(
        [
            "fetch-calibration",
            str(ROOT / "models" / "resnet_50"),
            "--variant",
            "modalix_int8",
            "--json",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"kind": "dataset"' in output
    assert f'"path": "{dataset}"' in output


def test_fetch_calibration_skips_variant_that_does_not_require_it(monkeypatch, capsys) -> None:
    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("BF16 must not resolve a calibration dataset")

    monkeypatch.setattr("model_registry.cli.resolve_dataset", unexpected_resolve)

    result = main(
        [
            "fetch-calibration",
            str(ROOT / "models" / "resnet_50"),
            "--variant",
            "modalix_bf16",
            "--json",
        ]
    )

    assert result == 0
    assert '"kind": "none"' in capsys.readouterr().out


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
