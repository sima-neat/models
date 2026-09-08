from __future__ import annotations

import argparse
import json
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
        "make-benchmark-manifest",
        "publish",
        "publish-benchmark",
        "publish-catalog",
        "resolve-base",
        "resolve-benchmark-runtime",
        "stage-registry-build",
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

    assert {"publish", "publish-benchmark", "publish-catalog"} <= set(subcommands.choices)
    assert {"package", "fetch-previous"}.isdisjoint(subcommands.choices)


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
    assert '"model_count": 6' in output
    assert '"ssd_mobilenet_v3"' in output
    assert '"valid": true' in output


def test_fetch_reports_file_set_root(monkeypatch, tmp_path: Path, capsys) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    checkpoint = source_root / "checkpoint_best_regular.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "model_registry.cli.Fetcher.fetch_source",
        lambda self, source: (checkpoint, True, "a" * 64, len(b"checkpoint")),
    )

    result = main(
        [
            "fetch",
            str(ROOT / "models" / "rf_detr_small"),
            "--component",
            "backbone",
            "--json",
        ]
    )

    assert result == 0
    assert f'"root": "{source_root}"' in capsys.readouterr().out


def test_fetch_calibration_resolves_registered_dataset(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "imagenet-mini.tgz"
    dataset.write_bytes(b"dataset")
    monkeypatch.setattr("model_registry.cli.resolve_dataset", lambda *args, **kwargs: dataset)

    result = main(
        [
            "fetch-calibration",
            str(ROOT / "models" / "resnet_50"),
            "--build-profile",
            "modalix_int8",
            "--component",
            "model",
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
            "--build-profile",
            "modalix_bf16",
            "--component",
            "model",
            "--json",
        ]
    )

    assert result == 0
    assert '"kind": "none"' in capsys.readouterr().out


def test_fetch_accuracy_dataset_resolves_for_scoped_int8_variant(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "imagenet-mini.tgz"
    dataset.write_bytes(b"dataset")
    monkeypatch.setattr("model_registry.cli.resolve_dataset", lambda *args, **kwargs: dataset)

    result = main(
        [
            "fetch-accuracy-dataset",
            str(ROOT / "models" / "resnet_50"),
            "--build-profile",
            "modalix_int8",
            "--component",
            "model",
            "--json",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"kind": "dataset"' in output
    assert f'"path": "{dataset}"' in output


def test_fetch_accuracy_dataset_resolves_model_recipe_correctness(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "imagenet-mini.tgz"
    dataset.write_bytes(b"dataset")
    monkeypatch.setattr("model_registry.cli.resolve_dataset", lambda *args, **kwargs: dataset)

    result = main(
        [
            "fetch-accuracy-dataset",
            str(ROOT / "models" / "rf_detr_small"),
            "--build-profile",
            "modalix_int8_bf16",
            "--component",
            "backbone",
            "--json",
        ]
    )

    assert result == 0
    assert f'"path": "{dataset}"' in capsys.readouterr().out


def test_fetch_accuracy_dataset_skips_bf16_outside_accuracy_scope(monkeypatch, capsys) -> None:
    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("BF16 must not resolve an accuracy dataset")

    monkeypatch.setattr("model_registry.cli.resolve_dataset", unexpected_resolve)

    result = main(
        [
            "fetch-accuracy-dataset",
            str(ROOT / "models" / "resnet_50"),
            "--build-profile",
            "modalix_bf16",
            "--component",
            "model",
            "--json",
        ]
    )

    assert result == 0
    assert '"kind": "none"' in capsys.readouterr().out


def test_fetch_accuracy_dataset_skips_legacy_correctness_declaration(monkeypatch, capsys) -> None:
    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("legacy correctness declarations must not resolve an accuracy dataset")

    monkeypatch.setattr("model_registry.cli.resolve_dataset", unexpected_resolve)

    result = main(
        [
            "fetch-accuracy-dataset",
            str(ROOT / "models" / "ssd_mobilenet_v1"),
            "--build-profile",
            "modalix_bf16_tess_mla",
            "--component",
            "model",
            "--json",
        ]
    )

    assert result == 0
    assert '"kind": "none"' in capsys.readouterr().out


def test_invalid_model_has_nonzero_exit(capsys) -> None:
    result = main(["validate", str(ROOT / "tests" / "fixtures" / "invalid_model")])
    assert result == 2
    assert "components[0].source" in capsys.readouterr().err


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


def test_matrix_selects_one_ssd_build_profile(capsys) -> None:
    profile = "modalix_int8_tess_mla"
    result = main(
        [
            "matrix",
            "--repository",
            str(ROOT),
            "--models",
            "ssd_mobilenet_v1",
            "--profiles",
            profile,
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert f'"build_profile_id":"{profile}"' in output
    assert '"model_id":"ssd_mobilenet_v1"' in output


def test_publish_writes_one_receipt_per_component(monkeypatch, tmp_path: Path) -> None:
    components = ("backbone", "transformer")
    for component_id in components:
        (tmp_path / "rf_detr_small" / "modalix_bf16" / component_id).mkdir(parents=True)
    publication = {
        "count": 2,
        "runs": [
            {
                "model_id": "rf_detr_small",
                "build_profile_id": "modalix_bf16",
                "component_id": component_id,
                "run_id": f"run-{component_id}",
            }
            for component_id in components
        ],
        "taxonomy": {"version": 4, "sha256": "a" * 64},
    }
    monkeypatch.setattr("model_registry.registry.publish_compilations", lambda **_: publication)
    args = argparse.Namespace(
        repository=ROOT,
        dist_dir=tmp_path,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        kms_key_id=None,
        environment="staging",
        branch="develop",
        git_sha="a" * 40,
        github_run_id="1234",
        github_run_attempt="1",
        github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
        json_output=False,
    )

    assert cli._publish(args) == 0

    for component_id in components:
        path = tmp_path / "rf_detr_small" / "modalix_bf16" / component_id / "registry-publication.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        assert receipt["count"] == 1
        assert receipt["runs"][0]["component_id"] == component_id


def test_tooling_supports_python_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "model_registry", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Model Registry lifecycle tooling" in result.stdout


@pytest.mark.parametrize("base_branch", [None, "develop"])
def test_publish_catalog_passes_optional_base_branch(monkeypatch, base_branch) -> None:
    calls = []

    def publish(**kwargs):
        calls.append(kwargs)
        return {"catalog": {"catalog_revision": "a" * 40}}

    monkeypatch.setattr("model_registry.registry.publish_catalog", publish)
    args = [
        "publish-catalog", "--api-url", "https://writer.example.invalid",
        "--read-api-url", "https://reader.example.invalid", "--branch", "feat/names",
        "--git-sha", "a" * 40,
    ]
    if base_branch:
        args.extend(["--base-branch", base_branch])
    assert main(args) == 0
    assert calls[0]["base_branch"] == base_branch
    assert calls[0]["branch"] == "feat/names"
