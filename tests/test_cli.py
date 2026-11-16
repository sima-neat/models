from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from model_registry import cli
from model_registry.cli import main

from tests.test_stage_plan import repository_without_w16a16_accuracy

ROOT = Path(__file__).resolve().parents[1]


def test_author_parser_exposes_only_supported_model_contributor_commands() -> None:
    parser = cli._parser(author_safe=True)
    subcommands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))

    assert set(subcommands.choices) == cli.AUTHOR_COMMANDS


@pytest.mark.parametrize("model,profile,stages,backend,baseline,timeout", [
    ("resnet_50", "w8a8", "performance,accuracy", "core", "false", "45"),
    ("qwen3_5_0_8b", "w4a16", "performance", "llima", "false", "90"),
    ("gemma_4_e2b_it_mtp", "w4a16", "performance", "llima", "true", "4320"),
    ("gemma_4_e2b_it_mtp", "w4a16", "accuracy", "llima", "false", "45"),
])
def test_benchmark_job_scopes_baseline_and_timeout(model, profile, stages, backend, baseline, timeout, capsys):
    assert main(["benchmark-job", str(ROOT / "models" / model), "--build-profile", profile, "--stages", stages]) == 0
    assert capsys.readouterr().out == f"backend={backend}\nneeds_baseline={baseline}\ntimeout_minutes={timeout}\n"


@pytest.mark.parametrize(
    "command",
    [
        "model-component-mpks",
        "check-report",
        "publish",
        "promotion",
        "conversion-entrypoint",
        "publish-conversion",
        "publish-candidate",
        "publish-branch-deployment",
        "publish-report",
        "publish-catalog",
        "resolve-base",
        "resolve-benchmark-runtime",
        "benchmark-job",
        "stage-registry-build",
        "plan-promoted-benchmarks",
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

    assert {"publish", "model-component-mpks", "check-report", "publish-report", "publish-catalog"} <= set(
        subcommands.choices
    )
    assert {"package", "fetch-previous"}.isdisjoint(subcommands.choices)


def test_promoted_benchmark_planning_command_reads_the_selected_catalog(monkeypatch, capsys):
    from model_registry import registry

    def read(url):
        assert url.endswith("git_ref=refs%2Fheads%2Fmain")
        return {"catalog": {"document": {"catalog_revision": "a" * 40, "models": []}}}

    monkeypatch.setattr(registry, "_public_registry_json", read)
    assert main(["plan-promoted-benchmarks", "--api-url", "https://registry.example.invalid", "--branch", "main"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "source_branch": "main", "source_sha": "a" * 40, "include": [], "skipped": [],
    }


@pytest.mark.parametrize("entrypoint", [main, cli.author_main])
def test_verify_artifact_dispatches_to_the_generic_verifier(entrypoint, monkeypatch, tmp_path, capsys) -> None:
    calls = []
    result = {"manifest": "deployment-manifest.json"}

    def verify(output_dir, *, write_manifest, provenance):
        calls.append((output_dir, write_manifest, provenance))
        return result

    monkeypatch.setattr(cli, "verify_artifact", verify)

    assert entrypoint([
        "verify-artifact", str(tmp_path), "--write-manifest", "--json", "--model-id", "qwen3_5_0_8b",
    ]) == 0
    assert calls == [(tmp_path, True, {"model_id": "qwen3_5_0_8b"})]
    assert json.loads(capsys.readouterr().out) == result


def test_verify_artifact_preserves_runner_provenance(monkeypatch, tmp_path):
    recorded = {}

    def verify(output_dir, *, write_manifest, provenance):
        recorded.update(provenance)
        return {"manifest": "manifest.json"}

    monkeypatch.setattr(cli, "verify_artifact", verify)
    expected = {
        "requested_runner_market": "on-demand", "runner_market": "on-demand",
        "runner_actual_market": "on-demand", "runner_name": "ephemeral-test",
        "runner_instance_id": "i-test", "runner_instance_type": "c8gd.4xlarge",
        "runner_approval_id": "sima-neat/models:test", "runner_approval_spec_revision": "a" * 64,
    }
    flags = [value for key, item in expected.items() for value in ("--" + key.replace("_", "-"), item)]
    assert main(["verify-artifact", str(tmp_path), "--write-manifest", *flags]) == 0
    assert recorded == expected


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_publish_candidate_forwards_workflow_arguments_and_writes_summary(
    monkeypatch, tmp_path, capsys, environment
) -> None:
    calls = []
    result = {
        "deployment_id": "candidate-id", "manifest_digest": "b" * 64,
        "git_ref": "refs/heads/feature/llima-models", "commit_sha": "a" * 40,
        "protection": [{"reason": "latest_successful", "expires_at": None}],
    }

    def publish(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr("model_registry.candidates.publish_candidate", publish)
    summary = tmp_path / "summary.md"
    expected = {
        "repository": tmp_path, "dist_dir": tmp_path / "compiled", "output_dir": tmp_path / "candidate",
        "model_id": "qwen3_5_0_8b", "build_profile_id": "w4a16", "api_url": "https://registry.example.invalid",
        "aws_region": "us-west-2", "bucket": "models", "kms_key_id": "key",
        "environment": environment, "branch": "feature/llima-models", "git_sha": "a" * 40,
        "github_run_id": "123", "github_run_attempt": "1", "github_run_url": "https://example.invalid/run/123",
    }
    assert main([
        "publish-candidate", "--repository", str(expected["repository"]),
        "--dist-dir", str(expected["dist_dir"]), "--output-dir", str(expected["output_dir"]),
        "--model", expected["model_id"], "--build-profile", expected["build_profile_id"],
        "--api-url", expected["api_url"], "--aws-region", expected["aws_region"],
        "--bucket", expected["bucket"], "--kms-key-id", expected["kms_key_id"],
        "--environment", expected["environment"], "--branch", expected["branch"], "--git-sha", expected["git_sha"],
        "--github-run-id", expected["github_run_id"], "--github-run-attempt", expected["github_run_attempt"],
        "--github-run-url", expected["github_run_url"], "--summary", str(summary), "--json",
    ]) == 0
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out) == result
    text = summary.read_text()
    assert f"Candidate ready: **qwen3_5_0_8b/w4a16** ({environment})" in text
    assert all(value in text for value in (result["deployment_id"], result["manifest_digest"], result["commit_sha"]))
    assert "latest_successful" in text


def test_validate_json_output(capsys) -> None:
    result = main(["validate", str(ROOT / "models" / "resnet_50"), "--json"])
    assert result == 0
    output = capsys.readouterr().out
    assert '"model_id": "resnet_50"' in output
    assert '"valid": true' in output


def test_validate_models_checks_every_manifest(capsys) -> None:
    result = main(["validate-models", "--repository", str(ROOT), "--json"])
    assert result == 0
    output = capsys.readouterr().out
    assert f'"model_count": {len(list((ROOT / "models").glob("*/model.yaml")))}' in output
    assert '"yolo26n"' in output
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
            str(ROOT / "models" / "resnet_50"),
            "--component",
            "model",
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
            "w8a8",
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
            "w16a16",
            "--component",
            "model",
            "--json",
        ]
    )

    assert result == 0
    assert '"kind": "none"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("model_id", "profile", "component"),
    [
        ("resnet_50", "w8a8", "model"),
        ("resnet_50", "w16a16", "model"),
        ("rf_detr_small", "w8a8_w16a16", "backbone"),
    ],
)
def test_fetch_fidelity_dataset_resolves_declared_fidelity(
    monkeypatch, tmp_path: Path, capsys, model_id: str, profile: str, component: str
) -> None:
    dataset = tmp_path / "dataset.tgz"
    dataset.write_bytes(b"dataset")
    monkeypatch.setattr("model_registry.cli.resolve_dataset", lambda *args, **kwargs: dataset)

    result = main(
        [
            "fetch-fidelity-dataset",
            str(ROOT / "models" / model_id),
            "--build-profile",
            profile,
            "--component",
            component,
            "--json",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"kind": "dataset"' in output
    assert f'"path": "{dataset}"' in output


def test_fetch_fidelity_dataset_skips_model_without_fidelity(monkeypatch, capsys) -> None:
    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("a model without fidelity entries must not resolve a dataset")

    monkeypatch.setattr("model_registry.cli.resolve_dataset", unexpected_resolve)

    result = main(
        [
            "fetch-fidelity-dataset",
            str(ROOT / "models" / "yolo26n"),
            "--build-profile",
            "w16a16",
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
    assert "compilation.source" in capsys.readouterr().err


@pytest.mark.parametrize("command", ("validate", "validate-models"))
def test_validation_rejects_a_resource_outside_the_source_policy(command: str, monkeypatch, capsys) -> None:
    source = {"uri": "https://unapproved.example/tables.json", "sha256": "b" * 64, "size_bytes": 1}
    model = {"id": "resnet_50", "resources": {"runtime": {"source": source, "install_path": "runtime"}}}
    monkeypatch.setattr(cli, "load_and_validate_model", lambda *_args: model)
    monkeypatch.setattr(cli, "load_models", lambda _repository: {"resnet_50": model})
    arguments = [str(ROOT / "models" / "resnet_50")] if command == "validate" else ["--repository", str(ROOT)]
    assert main([command, *arguments]) == 2
    assert "resource runtime: source policy" in capsys.readouterr().err


def test_fetch_resources_prints_nothing_for_a_model_without_resources(capsys, tmp_path: Path) -> None:
    assert main(["fetch-resources", str(ROOT / "models" / "whisper_small"), "--output-dir", str(tmp_path / "out")]) == 0
    assert capsys.readouterr().out == ""


def test_matrix_emits_workflow_json(capsys) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--models", "resnet_50", "--compiler-version", "fixture"])
    assert result == 0
    entries = json.loads(capsys.readouterr().out)["include"]
    assert {entry["model_id"] for entry in entries} == {"resnet_50"}
    assert all(len(entry["compile_fingerprint"]) == 64 for entry in entries)


def test_component_pipelines_writes_each_declared_pipeline(tmp_path: Path) -> None:
    output = tmp_path / "component-pipelines.json"
    assert main(["component-pipelines", "--repository", str(ROOT), "--output", str(output)]) == 0

    pipelines = json.loads(output.read_text(encoding="utf-8"))
    assert "resnet_50" not in pipelines
    assert pipelines["rf_detr_nano"] == [
        {"component": "backbone"},
        {"cpu": "Top 300 proposals"},
        {"component": "decoder"},
    ]


def test_reference_metrics_writes_each_models_published_scores(tmp_path: Path) -> None:
    output = tmp_path / "reference-metrics.json"
    assert main(["reference-metrics", "--repository", str(ROOT), "--output", str(output)]) == 0

    reference = json.loads(output.read_text(encoding="utf-8"))
    assert "whisper_small" not in reference
    assert reference["resnet_50"][0]["dataset"] == {"id": "imagenet-2012", "split": "val"}
    assert reference["resnet_50"][0]["metrics"] == {"top1_accuracy": 0.80382, "top5_accuracy": 0.94598}


def test_dataset_names_writes_a_name_for_every_dataset_file(tmp_path: Path) -> None:
    output = tmp_path / "dataset-names.json"
    assert main(["dataset-names", "--repository", str(ROOT), "--output", str(output)]) == 0

    names = json.loads(output.read_text(encoding="utf-8"))
    assert set(names) == {path.stem for path in (ROOT / "datasets").glob("*.yaml")}
    assert names["coco-2017"] == "COCO 2017"


def test_low_accuracy_variants_writes_declared_build_profiles(tmp_path: Path) -> None:
    output = tmp_path / "low-accuracy-variants.json"
    assert main(["low-accuracy-variants", "--repository", str(ROOT), "--output", str(output)]) == 0

    models = cli.load_models(ROOT)
    for model_id, profiles in json.loads(output.read_text(encoding="utf-8")).items():
        assert profiles
        assert set(profiles) <= {profile["id"] for profile in models[model_id]["build_profiles"]}


def test_resolve_devkit_accuracy_caches_each_accuracy_dataset_of_the_profile(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    dataset = tmp_path / "imagenet-mini.tgz"
    monkeypatch.setattr("model_registry.cli.resolve_dataset", lambda *args, **kwargs: dataset)

    assert main(["resolve-devkit-accuracy", str(ROOT / "models" / "resnet_50"), "--build-profile", "w8a8"]) == 0

    # One JSON line per dataset model.yaml's accuracy names.
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [{
        "name": "imagenet-2012-val",
        "path": str(dataset),
        "sha256": "018bf95eb75b8b7f1d7766303cdecf3c45ecb02ccf4cf161122ae885c4d7a36f",
        "size_bytes": 6666407650,
    }]


def test_resolve_devkit_accuracy_skips_a_profile_without_accuracy(monkeypatch, capsys, tmp_path: Path) -> None:
    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("a profile without accuracy must not resolve a dataset")

    monkeypatch.setattr("model_registry.cli.resolve_dataset", unexpected_resolve)

    model_dir = repository_without_w16a16_accuracy(tmp_path) / "models" / "resnet_50"
    assert main(["resolve-devkit-accuracy", str(model_dir), "--build-profile", "w16a16"]) == 0

    assert capsys.readouterr().out == ""


def test_model_component_mpks_prints_one_component_mpk_per_line(monkeypatch, tmp_path: Path, capsys) -> None:
    calls = []

    def inputs(*args):
        calls.append(args)
        return ["backbone=/c/backbone.tar.gz", "decoder=/c/decoder.tar.gz"]

    monkeypatch.setattr("model_registry.report_publication.model_component_mpks", inputs)

    assert main([
        "model-component-mpks", str(tmp_path), "--repository", str(ROOT), "--model", "rf_detr_nano",
        "--build-profile", "w8a8_w16a16",
    ]) == 0  # fmt: skip
    assert calls == [(tmp_path, ROOT, "rf_detr_nano", "w8a8_w16a16")]
    assert capsys.readouterr().out == "backbone=/c/backbone.tar.gz\ndecoder=/c/decoder.tar.gz\n"


def test_check_report_records_the_devkit_beside_the_checked_report(monkeypatch, tmp_path: Path, capsys) -> None:
    calls = []

    def check(*args):
        calls.append(args)
        return {"metrics": {"throughput_fps": 400.0, "latency_avg_ms": 2.5}}

    monkeypatch.setattr("model_registry.report_publication.check_report", check)
    monkeypatch.setattr("model_registry.report_publication.runner_environment", lambda: {"board_model": "Modalix"})

    assert main([
        "check-report", str(tmp_path), "--repository", str(ROOT), "--stage", "performance", "--model", "resnet_50",
        "--build-profile", "w8a8",
    ]) == 0  # fmt: skip
    assert calls == [(tmp_path / "model-report.json", ROOT, "resnet_50", "w8a8", "performance")]
    assert json.loads((tmp_path / "environment.json").read_text(encoding="utf-8")) == {"board_model": "Modalix"}
    assert capsys.readouterr().out == "Checked the performance report: latency_avg_ms, throughput_fps\n"


def test_check_report_accepts_independent_components_without_top_level_metrics(monkeypatch, tmp_path, capsys):
    fixture = json.loads((ROOT / "artifact-browser/fixtures/model-report-performance.json").read_text())
    report = {key: fixture[key] for key in ("schema", "schema_version", "stage", "runtime")}
    report.update(scope="components", components={
        name: {
            **fixture["components"]["model"],
            "metrics": {**fixture["metrics"], "throughput_fps": rate},
            "workload": name,
            "telemetry": {},
        }
        for name, rate in (("image_encoder", 230.5), ("text_encoder", 643.9))
    })
    path = tmp_path / "model-report.json"
    content = json.dumps(report)
    path.write_text(content)
    monkeypatch.setattr("model_registry.report_publication.runner_environment", lambda: {"board_model": "Modalix"})

    assert main([
        "check-report", str(tmp_path), "--repository", str(ROOT), "--stage", "performance",
        "--model", "clip_vit_b_16", "--build-profile", "w8a16",
    ]) == 0

    assert path.read_text() == content
    assert json.loads((tmp_path / "environment.json").read_text()) == {"board_model": "Modalix"}
    metrics = ", ".join(sorted(fixture["metrics"]))
    assert capsys.readouterr().out.splitlines() == [
        f"Checked the performance report for image_encoder: {metrics}",
        f"Checked the performance report for text_encoder: {metrics}",
    ]


def test_matrix_rejects_unknown_manual_model(capsys) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--models", "does_not_exist"])
    assert result == 2
    assert "unknown model id" in capsys.readouterr().err


def test_matrix_selects_one_build_profile(capsys) -> None:
    profile = "w8a8"
    result = main(
        [
            "matrix",
            "--repository",
            str(ROOT),
            "--models",
            "yolo26n",
            "--profiles",
            profile,
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert f'"build_profile_id":"{profile}"' in output
    assert '"model_id":"yolo26n"' in output


def test_publish_writes_one_receipt_per_component(monkeypatch, tmp_path: Path) -> None:
    components = ("backbone", "decoder")
    for component_id in components:
        (tmp_path / "rf_detr_small" / "w16a16" / component_id).mkdir(parents=True)
    publication = {
        "count": 2,
        "runs": [
            {
                "model_id": "rf_detr_small",
                "build_profile_id": "w16a16",
                "component_id": component_id,
                "run_id": f"run-{component_id}",
            }
            for component_id in components
        ],
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
        path = tmp_path / "rf_detr_small" / "w16a16" / component_id / "registry-publication.json"
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
    assert "Model Browser lifecycle tooling" in result.stdout


def publish_report(monkeypatch, stage: str, *extra: str) -> list[dict]:
    calls = []

    def publish(**kwargs):
        calls.append(kwargs)
        return {
            "benchmark_fingerprint": "k", "model_id": kwargs["model_id"],
            "build_profile_id": kwargs["build_profile_id"],
        }

    monkeypatch.setattr("model_registry.report_publication.publish_report", publish)
    main([
        "publish-report", f"_{stage}", "--stage", stage, "--repository", ".", "--compiled-dir", "_compiled",
        "--model", "resnet_50", "--build-profile", "w8a8", "--api-url", "https://writer.example.invalid",
        "--bucket", "model-artifacts", "--environment", "staging", "--branch", "feat/tests", "--git-sha", "a" * 40,
        *extra,
    ])  # fmt: skip
    return calls


@pytest.mark.parametrize("stage", ["performance", "accuracy"])
@pytest.mark.parametrize("status", ["passed", "failed", "skipped"])
def test_publishing_a_report_sends_its_stage_and_test_status(monkeypatch, capsys, stage: str, status: str) -> None:
    calls = publish_report(monkeypatch, stage, "--test-status", status)

    assert [(call["output_dir"], call["stage"], call["test_status"]) for call in calls] == [
        (Path(f"_{stage}"), stage, status)
    ]
    assert (calls[0]["compiled_dir"], calls[0]["model_id"], calls[0]["build_profile_id"]) == (
        Path("_compiled"), "resnet_50", "w8a8",
    )
    assert capsys.readouterr().out == f"Published {stage} k for resnet_50/w8a8\n"


@pytest.mark.parametrize("extra", [(), ("--test-status", "error")])
def test_publishing_a_report_requires_a_test_outcome(monkeypatch, extra: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as raised:
        publish_report(monkeypatch, "performance", *extra)

    assert raised.value.code == 2



@pytest.mark.parametrize(
    ("argument", "base_branches"),
    [(None, []), ("", []), ("develop", ["develop"]), ("develop,release-1.0", ["develop", "release-1.0"])],
)
@pytest.mark.parametrize("force_rebuild_all", [False, True])
def test_publish_catalog_passes_optional_base_branches(monkeypatch, argument, base_branches, force_rebuild_all) -> None:
    calls = []

    def publish(**kwargs):
        calls.append(kwargs)
        return {"catalog": {"catalog_revision": "a" * 40}}

    monkeypatch.setattr("model_registry.registry.publish_catalog", publish)
    args = [
        "publish-catalog", "--api-url", "https://writer.example.invalid",
        "--read-api-url", "https://reader.example.invalid", "--branch", "feat/names",
        "--git-sha", "a" * 40, "--bucket", "model-artifacts",
        "--environment", "staging",
    ]
    if force_rebuild_all:
        args.append("--force-rebuild-all")
    if argument is not None:
        args.extend(["--base-branches", argument])
    assert main(args) == 0
    assert calls[0]["force_rebuild_all"] == force_rebuild_all
    assert calls[0]["base_branches"] == base_branches
    assert calls[0]["branch"] == "feat/names"
    assert calls[0]["bucket"] == "model-artifacts"
    assert calls[0]["environment"] == "staging"


@pytest.mark.parametrize(("declared", "printed"), [({}, "model.onnx"), ({"outputs": ["a.onnx", "b.pth"]}, "a.onnx")])
def test_conversion_entrypoint_prints_the_first_output(tmp_path, capsys, declared, printed):
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"conversion": declared}))
    assert main(["conversion-entrypoint", str(tmp_path)]) == 0
    assert capsys.readouterr().out == printed + "\n"


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_publish_conversion_cli_forwards_arguments_and_workflow(monkeypatch, tmp_path, capsys, environment):
    calls = []
    client, store = object(), object()
    monkeypatch.setattr(
        "model_registry.registry.SigV4RegistryClient", lambda api, region: calls.append((api, region)) or client,
    )
    monkeypatch.setattr(
        "model_registry.registry.S3ArtifactStore", lambda bucket, **kw: calls.append((bucket, kw)) or store,
    )
    monkeypatch.setattr(
        "model_registry.conversion.publish_conversion",
        lambda *args, **kw: calls.append((args, kw)) or {"run_id": "converted"},
    )
    for key, value in {"GITHUB_REF_NAME": "canary", "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123"}.items():
        monkeypatch.setenv(key, value)
    assert main(["publish-conversion", "yolo26n", str(tmp_path / "onnx"),
                 "--repository", str(tmp_path), "--api-url", "https://registry.test",
                 "--bucket", "artifacts", "--kms-key-id", "key", "--aws-region", "us-west-2",
                 "--environment", environment, "--validation", "passed"]) == 0
    assert calls == [("https://registry.test", "us-west-2"), ("artifacts", {"kms_key_id": "key"}),
                     ((tmp_path, "yolo26n", tmp_path / "onnx"), {
                         "client": client, "store": store, "bucket": "artifacts", "environment": environment,
                         "branch": "canary", "git_sha": "a" * 40, "workflow": "123", "validation": "passed"})]
    assert json.loads(capsys.readouterr().out) == {"run_id": "converted"}


@pytest.mark.parametrize("validation", ["failed", "skipped"])
def test_publish_conversion_cli_rejects_unvalidated_output(validation):
    with pytest.raises(SystemExit) as error:
        main(["publish-conversion", "yolo26n", "output", "--api-url", "https://registry.test",
              "--bucket", "artifacts", "--environment", "production", "--validation", validation])
    assert error.value.code == 2


def test_model_runtime_directory_verifies_assets_and_omits_unneeded_arguments(tmp_path, capsys):
    import hashlib

    profile = tmp_path / "test/w16a16"
    profile.mkdir(parents=True)
    arguments = ["model-runtime-directory", str(tmp_path), "--model", "test", "--build-profile", "w16a16"]
    assert main(arguments) == 0
    assert capsys.readouterr().out == ""
    asset = profile / "tokenizer.json"
    asset.write_text("{}")
    (profile / "compile-outputs.json").write_text(json.dumps({
        "runtime_files": [{"path": asset.name, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}],
    }))
    assert main(arguments) == 0
    assert capsys.readouterr().out.strip() == str(profile)
    asset.write_text("corrupted")
    assert main(arguments) != 0
    assert "SHA-256" in capsys.readouterr().err
