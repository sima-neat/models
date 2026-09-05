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
    assert "inputs.operation" not in stage["if"]
    verify = next(step for step in stage["steps"] if step.get("name") == "Verify the source run and compiled artifact")
    assert 'actions/runs/$SOURCE_RUN_ID' in verify["run"]
    assert ".head_sha == $sha" in verify["run"]
    download = next(step for step in stage["steps"] if step.get("uses") == "actions/download-artifact@v7")
    assert download["with"]["name"] == (
        "compiled-model-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    assert download["with"]["run-id"] == "${{ inputs.source_run_id }}"
    checkout = next(
        step
        for step in stage["steps"]
        if step.get("name") == "Check out the trusted source revision for registry tooling"
    )
    assert checkout["with"]["ref"] == "${{ inputs.source_sha }}"
    assert checkout["with"]["path"] == "_registry_source"
    publication = next(
        step for step in stage["steps"] if step.get("name") == "Register an unpublished compilation for benchmarking"
    )
    assert publication["if"] == "inputs.source_run_id != '' && steps.publication.outputs.required == 'true'"
    assert "model-registry publish" in publication["run"]
    restore = next(
        step for step in stage["steps"]
        if step.get("name") == "Restore the current immutable compilation from the registry"
    )
    assert restore["if"] == "inputs.source_run_id == ''"
    assert "model-registry stage-registry-build" in restore["run"]
    assert '--repository _registry_source' in publication["run"]
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
    assert benchmark["needs"] == ["stage-dispatched-benchmark", "resolve-dispatched-benchmark-runner"]
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
    checkout = next(step for step in steps if step.get("name") == "Check out the trusted revision")
    assert checkout["with"]["ref"] == "${{ inputs.source_sha }}"
    download = next(step for step in steps if step.get("name") == "Download the staged compiled model output")
    assert download["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    runtime = next(step for step in steps if step.get("name") == "Resolve and verify benchmark runtime")
    install = next(step for step in steps if step.get("name") == "Install branch benchmarker and Neat dependency")
    execute = next(step for step in steps if step.get("name") == "Generate manifest and run benchmark")
    assert "resolve-benchmark-runtime" in runtime["run"]
    assert "/etc/buildinfo" in runtime["run"]
    assert "sed -n 's/^DISTRO_VERSION=//p'" not in runtime["run"]
    assert "^[[:space:]]*DISTRO_VERSION[[:space:]]*=" in runtime["run"]
    assert 'gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)' in runtime["run"]
    assert "Could not read DISTRO_VERSION from /etc/buildinfo" in runtime["run"]
    assert "DevKit platform mismatch: expected $platform_version, found $actual_platform" in runtime["run"]
    assert "sima-cli neat install" in install["run"]
    assert 'tooling_tag="${GITHUB_SHA:0:12}"' in install["run"]
    assert 'target="$name@$GITHUB_REF_NAME:$tooling_tag"' in install["run"]
    assert "for attempt in {1..6}" in install["run"]
    assert "retrying in 5 seconds" in install["run"]
    assert "SOURCE_SHA" not in install["run"]
    assert "latest" not in install["run"]
    assert 'getattr(benchmarker, "RESULT_SCHEMA_VERSION", None) == 4' in install["run"]
    assert "make-benchmark-manifest" in execute["run"]
    assert '--model "$MODEL_ID"' in execute["run"]
    assert '--build-profile "$BUILD_PROFILE_ID"' in execute["run"]
    assert 'for component in "${components[@]}"' in execute["run"]
    assert '--component "$component"' in execute["run"]
    assert "performance-result-v4.json" in execute["run"]
    assert "kerrigan" not in BENCHMARK_WORKFLOW.read_text(encoding="utf-8").lower()


def test_completed_dispatched_benchmark_is_published_from_a_hosted_runner() -> None:
    workflow = yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    publish = workflow["jobs"]["publish-dispatched-benchmark"]

    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["needs"] == ["benchmark-dispatched-model", "resolve-dispatched-benchmark-config"]
    checkout = next(step for step in publish["steps"] if step.get("uses") == "actions/checkout@v6")
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    download = next(step for step in publish["steps"] if step.get("uses") == "actions/download-artifact@v7")
    assert download["with"]["name"] == (
        "benchmark-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    publication = next(
        step for step in publish["steps"] if step.get("name") == "Publish every completed schema-v4 result and log"
    )
    assert publication["env"]["COMPONENT_IDS"] == "${{ inputs.component_ids }}"
    assert "validate-benchmark-set" in publication["run"]
    assert "find _benchmarks -type f -name performance-result-v4.json" in publication["run"]
    assert 'for component in "${components[@]}"' in publication["run"]
    assert "actual_count == ${#components[@]}" in publication["run"]
    assert 'for result in "${results[@]}"' in publication["run"]
    assert "model-registry publish-benchmark" in publication["run"]
    assert "--schema schemas/performance-result-v4.schema.json" in publication["run"]
    assert '--git-sha "$SOURCE_SHA"' in publication["run"]
