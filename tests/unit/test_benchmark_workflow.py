from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Benchmarking.yml"


def test_benchmark_workflow_has_a_distinct_actions_identity() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["name"] == "Vulcan Model Benchmarking"
    assert list(workflow["on"]) == ["workflow_dispatch"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "environment",
        "source_run_id",
        "source_sha",
        "source_branch",
        "model_id",
        "build_profile_id",
        "compile_invocation_id",
        "component_ids",
    }
    assert inputs["source_run_id"]["required"] == "false"
    assert all(value["required"] == "true" for name, value in inputs.items() if name != "source_run_id")
    assert "operation" not in inputs
    assert "model-benchmark-" in workflow["concurrency"]["group"]


def test_dispatched_benchmark_stages_only_its_compiled_artifact() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    stage = workflow["jobs"]["stage-dispatched-benchmark"]

    assert stage["needs"] == "resolve-dispatched-benchmark-config"
    assert stage["environment"] == "${{ inputs.environment }}"
    assert stage["permissions"]["id-token"] == "write"
    assert stage["outputs"]["source_branch"] == "${{ steps.source.outputs.branch }}"
    assert "inputs.source_branch == github.ref_name" in stage["if"]
    assert "inputs.operation" not in stage["if"]
    verify = next(step for step in stage["steps"] if step.get("name") == "Verify the source run and compiled artifact")
    assert 'actions/runs/$SOURCE_RUN_ID' in verify["run"]
    assert ".head_sha == $sha" in verify["run"]
    assert ".head_branch == $branch" in verify["run"]
    assert '[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]' in verify["run"]
    assert 'git check-ref-format --branch "$SOURCE_BRANCH"' in verify["run"]
    assert 'artifact_name="compiled-model-$MODEL_ID-$BUILD_PROFILE_ID-$COMPILE_INVOCATION_ID"' in verify["run"]
    assert "compiler-output-" not in verify["run"]
    download = next(step for step in stage["steps"] if step.get("uses") == "actions/download-artifact@v7")
    assert download["with"]["name"] == (
        "compiled-model-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    assert download["with"]["run-id"] == "${{ inputs.source_run_id }}"
    trusted_checkout = next(
        step
        for step in stage["steps"]
        if step.get("name") == "Check out trusted registry tooling from the protected workflow revision"
    )
    assert trusted_checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    source_checkout = next(
        step for step in stage["steps"] if step.get("name") == "Check out the source revision as data"
    )
    assert source_checkout["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "path": "_source",
        "persist-credentials": "false",
    }
    install = next(step for step in stage["steps"] if step.get("name") == "Install registry publication tooling")
    assert install["run"] == "python -m pip install ."
    publication = next(
        step for step in stage["steps"] if step.get("name") == "Register an unpublished compilation for benchmarking"
    )
    credentials = next(
        step for step in stage["steps"] if step.get("name") == "Configure artifact-registry publisher credentials"
    )
    assert credentials["id"] == "artifact-publisher-credentials"
    assert credentials["with"]["output-credentials"] == "true"
    assert credentials["with"]["output-env-credentials"] == "false"
    assert publication["env"]["AWS_ACCESS_KEY_ID"] == (
        "${{ steps.artifact-publisher-credentials.outputs.aws-access-key-id }}"
    )
    assert publication["if"] == "inputs.source_run_id != '' && steps.publication.outputs.required == 'true'"
    assert "model-registry publish" in publication["run"]
    restore = next(
        step for step in stage["steps"]
        if step.get("name") == "Restore the current immutable compilation from the registry"
    )
    assert restore["if"] == "inputs.source_run_id == ''"
    assert "model-registry stage-registry-build" in restore["run"]
    assert '--repository _source' in publication["run"]
    assert '--dist-dir _compiled' in publication["run"]
    assert '--github-run-id "$SOURCE_RUN_ID"' in publication["run"]
    assert '--github-run-attempt "$SOURCE_RUN_ATTEMPT"' in publication["run"]
    assert "path.write_text" not in publication["run"]
    upload = next(step for step in stage["steps"] if step.get("uses") == "actions/upload-artifact@v7")
    assert upload["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )


def test_dispatched_benchmark_runs_on_the_resolved_devkit() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    resolver = workflow["jobs"]["resolve-dispatched-benchmark-runner"]
    benchmark = workflow["jobs"]["benchmark-dispatched-model"]
    steps = benchmark["steps"]

    assert resolver["uses"].endswith("/vulcan-resolve-modalix-runner.yml@main")
    assert "inputs.source_branch == github.ref_name" in resolver["if"]
    assert benchmark["needs"] == [
        "stage-dispatched-benchmark",
        "resolve-dispatched-benchmark-runner",
        "resolve-dispatched-benchmark-config",
    ]
    assert benchmark["environment"] == "${{ inputs.environment }}"
    assert benchmark["permissions"]["id-token"] == "write"
    assert benchmark["runs-on"]["group"] == (
        "${{ needs.resolve-dispatched-benchmark-runner.outputs.modalix_runner_group }}"
    )
    assert benchmark["runs-on"]["labels"] == [
        "self-hosted",
        "Linux",
        "ARM64",
        "modalix",
        "vulcan-${{ inputs.environment }}",
    ]
    assert benchmark["env"]["SIMA_CLI_AUTO_ACCEPT_UPDATE"] == "1"
    assert benchmark["env"]["SOURCE_BRANCH"] == (
        "${{ needs.stage-dispatched-benchmark.outputs.source_branch }}"
    )
    trusted_checkout = next(
        step for step in steps if step.get("name") == "Check out trusted tooling from the protected workflow revision"
    )
    assert trusted_checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    source_checkout = next(
        step for step in steps if step.get("name") == "Check out the source revision as data"
    )
    assert source_checkout["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "path": "_source",
        "persist-credentials": "false",
    }
    registry_install = next(
        step for step in steps if step.get("name") == "Install registry tooling in an isolated environment"
    )
    assert 'pip install .' in registry_install["run"]
    assert "pip install _source" not in registry_install["run"]
    download = next(step for step in steps if step.get("name") == "Download the staged compiled model output")
    assert download["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    runtime = next(step for step in steps if step.get("name") == "Resolve and verify benchmark runtime")
    install = next(step for step in steps if step.get("name") == "Install selected Neat Core")
    execute = next(step for step in steps if step.get("name") == "Generate manifest and run benchmark")
    assert "resolve-benchmark-runtime" in runtime["run"]
    assert "/etc/buildinfo" in runtime["run"]
    assert "sed -n 's/^DISTRO_VERSION=//p'" not in runtime["run"]
    assert "^[[:space:]]*DISTRO_VERSION[[:space:]]*=" in runtime["run"]
    assert 'gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)' in runtime["run"]
    assert "Could not read DISTRO_VERSION from /etc/buildinfo" in runtime["run"]
    assert "DevKit platform mismatch: expected $platform_version, found $actual_platform" in runtime["run"]
    assert '["devkit"]["build"]' not in runtime["run"]
    assert "NEAT_VERSION" not in install["run"]
    assert 'sima-cli neat install --env production --type minimal "$target"' in install["run"]
    assert '["neat_core"]["install_target"]' in install["run"]
    assert '"models/$MODEL_ID/scripts/performance.py"' not in execute["run"]
    assert "report = model.benchmark()" in execute["run"]
    assert "PYBENCH" in execute["run"]
    assert "import-core-benchmark" in execute["run"]
    assert "core-benchmark.json" in execute["run"]
    assert "make-benchmark-manifest" in execute["run"]
    assert '--model "$MODEL_ID"' in execute["run"]
    assert '--build-profile "$BUILD_PROFILE_ID"' in execute["run"]
    assert 'for component in "${components[@]}"' in execute["run"]
    assert '--component "$component"' in execute["run"]
    assert "performance-result-v5.json" in execute["run"]
    assert "kerrigan" not in BENCHMARK_WORKFLOW.read_text(encoding="utf-8").lower()


def test_completed_dispatched_benchmark_is_published_from_a_hosted_runner() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    publish = workflow["jobs"]["publish-dispatched-benchmark"]

    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["needs"] == ["benchmark-dispatched-model", "resolve-dispatched-benchmark-config"]
    checkouts = [step for step in publish["steps"] if step.get("uses") == "actions/checkout@v6"]
    assert checkouts[0]["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    assert checkouts[1]["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "path": "_source",
        "persist-credentials": "false",
    }
    download = next(step for step in publish["steps"] if step.get("uses") == "actions/download-artifact@v7")
    assert download["with"]["name"] == (
        "benchmark-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    publication = next(
        step for step in publish["steps"] if step.get("name") == "Publish every completed schema-v5 result and log"
    )
    credentials = next(
        step for step in publish["steps"] if step.get("id") == "benchmark-publisher-credentials"
    )
    assert credentials["with"]["output-credentials"] == "true"
    assert credentials["with"]["output-env-credentials"] == "false"
    assert publication["env"]["AWS_ACCESS_KEY_ID"] == (
        "${{ steps.benchmark-publisher-credentials.outputs.aws-access-key-id }}"
    )
    assert publication["env"]["COMPONENT_IDS"] == "${{ inputs.component_ids }}"
    assert publication["env"]["SOURCE_BRANCH"] == "${{ inputs.source_branch }}"
    assert "validate-benchmark-set" in publication["run"]
    assert "--repository _source" in publication["run"]
    assert "find _benchmarks -type f -name performance-result-v5.json" in publication["run"]
    assert 'for component in "${components[@]}"' in publication["run"]
    assert "actual_count == ${#components[@]}" in publication["run"]
    assert 'for result in "${results[@]}"' in publication["run"]
    assert "model-registry publish-benchmark" in publication["run"]
    assert "--schema schemas/performance-result-v5.schema.json" in publication["run"]
    assert '--branch "$SOURCE_BRANCH"' in publication["run"]
    assert '--git-sha "$SOURCE_SHA"' in publication["run"]



def test_dispatched_benchmark_caches_declared_datasets_on_nvme() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["benchmark-dispatched-model"]["steps"]
    names = [step.get("name") for step in steps]
    runtime = names.index("Resolve and verify benchmark runtime")
    install = names.index("Install trusted dataset tooling")
    credentials = names.index("Configure dataset reader credentials")
    cache = names.index("Cache declared datasets on NVMe")

    assert runtime < install < credentials < cache
    assert names.index("Check out trusted tooling from the protected workflow revision") < install
    assert 'scripts/install_dataset_registry.sh "$REGISTRY_PYTHON"' in steps[install]["run"]
    assert steps[credentials]["with"] == {
        "aws-region": "${{ needs.resolve-dispatched-benchmark-config.outputs.dataset_aws_region }}",
        "role-to-assume": "${{ needs.resolve-dispatched-benchmark-config.outputs.dataset_reader_role_arn }}",
        "role-session-name": "models-dataset-${{ github.run_id }}-${{ github.run_attempt }}",
        "output-credentials": "true",
        "output-env-credentials": "false",
    }
    assert steps[credentials]["id"] == "dataset-credentials"
    assert steps[cache]["env"] == {
        "AWS_ACCESS_KEY_ID": "${{ steps.dataset-credentials.outputs.aws-access-key-id }}",
        "AWS_SECRET_ACCESS_KEY": "${{ steps.dataset-credentials.outputs.aws-secret-access-key }}",
        "AWS_SESSION_TOKEN": "${{ steps.dataset-credentials.outputs.aws-session-token }}",
        "AWS_REGION": "${{ needs.resolve-dispatched-benchmark-config.outputs.dataset_aws_region }}",
        "AWS_DEFAULT_REGION": "${{ needs.resolve-dispatched-benchmark-config.outputs.dataset_aws_region }}",
        "SIMA_DATASET_S3_URI": "s3://${{ needs.resolve-dispatched-benchmark-config.outputs.dataset_bucket }}",
    }
    assert steps[cache]["run"].startswith('scripts/cache_declared_datasets.sh "_source/models/$MODEL_ID"')
    credentialed_runs = "\n".join(
        step.get("run", "") for step in steps[credentials:] if isinstance(step, dict)
    )
    assert "pip install _source" not in credentialed_runs
    assert '"models/$MODEL_ID/scripts/performance.py"' not in credentialed_runs

    install_script = (ROOT / "scripts" / "install_dataset_registry.sh").read_text(encoding="utf-8")
    assert "--download-only" in install_script
    assert "dataset/dataset-registry@main:latest" in install_script
    cache_script = (ROOT / "scripts" / "cache_declared_datasets.sh").read_text(encoding="utf-8")
    assert 'DATASET_ROOT="${DATASET_ROOT:-$NVME_MOUNT/datasets}"' in cache_script
    assert 'NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"' in cache_script
    assert 'mountpoint -q "$NVME_MOUNT"' in cache_script
    assert 'flock --verbose --wait "$LOCK_WAIT_SECONDS"' in cache_script
    assert "fetch-accuracy-dataset" in cache_script
    assert '--cache-dir "$DATASET_ROOT/artifacts"' in cache_script
    assert "coco" not in cache_script.lower()


def test_dispatched_benchmark_runs_only_on_its_source_branch() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    gated = [
        "stage-dispatched-benchmark",
        "resolve-dispatched-benchmark-runner",
        "resolve-dispatched-benchmark-config",
        "benchmark-dispatched-model",
        "publish-dispatched-benchmark",
    ]

    for name in gated:
        condition = workflow["jobs"][name]["if"]
        assert "inputs.source_branch == github.ref_name" in condition, name
        assert "refs/heads/develop" not in condition, name


def test_dispatched_benchmark_measures_and_publishes_devkit_accuracy() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    steps = jobs["benchmark-dispatched-model"]["steps"]
    names = [step.get("name") for step in steps]
    benchmark = names.index("Generate manifest and run benchmark")
    accuracy = names.index("Measure accuracy on DevKit")
    retain = names.index("Retain accuracy evidence")

    assert benchmark < accuracy < retain
    assert "env" not in steps[accuracy]
    assert steps[accuracy]["run"] == (
        'scripts/run_model_accuracy.sh "_source/models/$MODEL_ID" "$BUILD_PROFILE_ID" _benchmark _accuracy'
    )
    artifact = "accuracy-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    assert steps[retain]["if"] == "always()"
    assert steps[retain]["with"]["name"] == artifact
    assert steps[retain]["with"]["if-no-files-found"] == "ignore"

    publish = jobs["publish-dispatched-benchmark"]["steps"]
    download = next(step for step in publish if step.get("with", {}).get("pattern") == artifact)
    assert download["with"]["path"] == "_accuracy"
    publication = next(
        step for step in publish if step.get("name") == "Publish every completed schema-v5 result and log"
    )
    assert '--accuracy-result "$accuracy"' in publication["run"]
    assert "--accuracy-schema schemas/accuracy-result-v2.schema.json" in publication["run"]
    assert '"${accuracy_args[@]}"' in publication["run"]

    cleanup = jobs["cleanup-dispatched-benchmark-artifacts"]["steps"]
    assert any("accuracy-$MODEL_ID-$BUILD_PROFILE_ID-$COMPILE_INVOCATION_ID" in step.get("run", "") for step in cleanup)
