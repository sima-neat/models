from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Benchmarking.yml"
MEASURE_PERFORMANCE = "${{ contains(format(',{0},', inputs.stages), ',performance,') }}"
MEASURE_ACCURACY = "${{ contains(format(',{0},', inputs.stages), ',accuracy,') }}"
PUBLISH_PERFORMANCE = "${{ !cancelled() && needs.benchmark-dispatched-model.outputs.performance == 'success' }}"
PUBLISH_ACCURACY = "${{ !cancelled() && needs.benchmark-dispatched-model.outputs.accuracy == 'success' }}"
# Model code from the source revision runs only through these steps' scripts.
MODEL_CODE_MARKERS = ("scripts/run_model_performance.sh", "scripts/run_model_accuracy.sh", "pip install _source")


def load_workflow() -> dict:
    return yaml.load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def step(steps: list[dict], name: str) -> dict:
    return next(item for item in steps if item.get("name") == name)


def test_benchmark_workflow_has_a_distinct_actions_identity() -> None:
    workflow = load_workflow()

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
        "stages",
    }
    optional = {"source_run_id", "component_ids", "stages"}
    assert all(inputs[name]["required"] == "false" for name in optional)
    assert all(value["required"] == "true" for name, value in inputs.items() if name not in optional)
    assert inputs["component_ids"]["default"] == ""
    assert inputs["stages"]["default"] == "performance,accuracy"
    assert inputs["environment"]["options"] == ["staging"]
    assert "operation" not in inputs
    assert "model-benchmark-" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"


def test_dispatched_benchmark_stages_only_its_compiled_artifact() -> None:
    workflow = load_workflow()
    stage = workflow["jobs"]["stage-dispatched-benchmark"]

    assert stage["needs"] == "resolve-dispatched-benchmark-config"
    assert stage["environment"] == "${{ inputs.environment }}"
    assert stage["permissions"]["id-token"] == "write"
    assert stage["outputs"]["source_branch"] == "${{ steps.source.outputs.branch }}"
    assert "inputs.source_branch == github.ref_name" in stage["if"]
    assert "inputs.stages != ''" in stage["if"]
    assert "inputs.operation" not in stage["if"]
    verify = step(stage["steps"], "Verify the source run and compiled artifact")
    assert verify["env"]["STAGES"] == "${{ inputs.stages }}"
    assert '[[ "$stage" == performance || "$stage" == accuracy ]]' in verify["run"]
    # Performance needs named components; an accuracy-only dispatch may name none.
    assert '[[ ",$STAGES," != *",performance,"* || -n "$COMPONENT_IDS" ]]' in verify["run"]
    assert 'actions/runs/$SOURCE_RUN_ID' in verify["run"]
    assert ".head_sha == $sha" in verify["run"]
    assert ".head_branch == $branch" in verify["run"]
    assert '[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]' in verify["run"]
    assert 'git check-ref-format --branch "$SOURCE_BRANCH"' in verify["run"]
    assert 'artifact_name="compiled-model-$MODEL_ID-$BUILD_PROFILE_ID-$COMPILE_INVOCATION_ID"' in verify["run"]
    assert "compiler-output-" not in verify["run"]
    download = next(item for item in stage["steps"] if item.get("uses") == "actions/download-artifact@v7")
    assert download["with"]["name"] == (
        "compiled-model-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    assert download["with"]["run-id"] == "${{ inputs.source_run_id }}"
    trusted_checkout = step(stage["steps"], "Check out trusted registry tooling from the protected workflow revision")
    assert trusted_checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    source_checkout = step(stage["steps"], "Check out the source revision as data")
    assert source_checkout["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "path": "_source",
        "persist-credentials": "false",
    }
    install = step(stage["steps"], "Install registry publication tooling")
    assert install["run"] == "python -m pip install ."
    publication = step(stage["steps"], "Register an unpublished compilation for benchmarking")
    credentials = step(stage["steps"], "Configure artifact-registry publisher credentials")
    assert credentials["id"] == "artifact-publisher-credentials"
    assert credentials["with"]["output-credentials"] == "true"
    assert credentials["with"]["output-env-credentials"] == "false"
    assert publication["env"]["AWS_ACCESS_KEY_ID"] == (
        "${{ steps.artifact-publisher-credentials.outputs.aws-access-key-id }}"
    )
    assert publication["if"] == "inputs.source_run_id != '' && steps.publication.outputs.required == 'true'"
    assert "model-registry publish" in publication["run"]
    restore = step(stage["steps"], "Restore the current immutable compilation from the registry")
    assert restore["if"] == "inputs.source_run_id == ''"
    assert "model-registry stage-registry-build" in restore["run"]
    assert '--repository _source' in publication["run"]
    assert '--dist-dir _compiled' in publication["run"]
    assert '--github-run-id "$SOURCE_RUN_ID"' in publication["run"]
    assert '--github-run-attempt "$SOURCE_RUN_ATTEMPT"' in publication["run"]
    upload = next(item for item in stage["steps"] if item.get("uses") == "actions/upload-artifact@v7")
    assert upload["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )


def test_dispatched_benchmark_runs_on_the_resolved_devkit() -> None:
    workflow = load_workflow()
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
    assert benchmark["env"]["SOURCE_BRANCH"] == "${{ needs.stage-dispatched-benchmark.outputs.source_branch }}"
    assert step(steps, "Check out trusted tooling from the protected workflow revision")["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    assert step(steps, "Check out the source revision as data")["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "path": "_source",
        "persist-credentials": "false",
    }
    registry_install = step(steps, "Install registry tooling in an isolated environment")
    assert "pip install ." in registry_install["run"]
    assert "pip install _source" not in registry_install["run"]
    download = step(steps, "Download the staged compiled model output")
    assert download["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    runtime = step(steps, "Resolve and verify benchmark runtime")["run"]
    install = step(steps, "Install selected Neat Core")["run"]
    assert "resolve-benchmark-runtime --repository _source" in runtime
    assert "^[[:space:]]*DISTRO_VERSION[[:space:]]*=" in runtime
    assert "Could not read DISTRO_VERSION from /etc/buildinfo" in runtime
    assert "DevKit platform mismatch: expected $platform_version, found $actual_platform" in runtime
    assert 'sima-cli neat install --env production --type minimal "$target"' in install
    assert '["neat_core"]["install_target"]' in install
    assert "kerrigan" not in BENCHMARK_WORKFLOW.read_text(encoding="utf-8").lower()


def test_devkit_stages_run_only_when_requested() -> None:
    benchmark = load_workflow()["jobs"]["benchmark-dispatched-model"]
    steps = benchmark["steps"]
    artifact_suffix = "${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"

    assert benchmark["env"]["MEASURE_PERFORMANCE"] == MEASURE_PERFORMANCE
    assert benchmark["env"]["MEASURE_ACCURACY"] == MEASURE_ACCURACY
    performance = step(steps, "Measure performance on DevKit")
    assert performance["if"] == (
        "${{ !cancelled() && steps.core.outcome == 'success' && env.MEASURE_PERFORMANCE == 'true' }}"
    )
    assert performance["run"] == (
        'scripts/run_model_performance.sh "_source/models/$MODEL_ID" "$BUILD_PROFILE_ID" _compiled _benchmark '
        '"$COMPONENT_IDS"'
    )
    performance_evidence = step(steps, "Retain benchmark evidence")
    assert performance_evidence["if"] == "always() && env.MEASURE_PERFORMANCE == 'true'"
    assert performance_evidence["with"]["name"] == f"benchmark-{artifact_suffix}"

    accuracy = step(steps, "Measure accuracy on DevKit")
    assert accuracy["if"] == (
        "${{ !cancelled() && steps.core.outcome == 'success' && steps.dataset-cache.outcome == 'success' "
        "&& env.MEASURE_ACCURACY == 'true' }}"
    )
    assert accuracy["run"] == (
        'scripts/run_model_accuracy.sh "_source/models/$MODEL_ID" "$BUILD_PROFILE_ID" _compiled _accuracy'
    )
    accuracy_evidence = step(steps, "Retain accuracy evidence")
    assert accuracy_evidence["if"] == "always() && env.MEASURE_ACCURACY == 'true'"
    assert accuracy_evidence["with"]["name"] == f"accuracy-{artifact_suffix}"
    assert accuracy_evidence["with"]["if-no-files-found"] == "error"
    for name in ("Install trusted dataset tooling", "Configure dataset reader credentials",
                 "Cache the accuracy dataset on NVMe"):
        assert step(steps, name)["if"] == "env.MEASURE_ACCURACY == 'true'", name


def test_each_devkit_stage_runs_and_publishes_on_its_own_outcome() -> None:
    workflow = load_workflow()
    benchmark = workflow["jobs"]["benchmark-dispatched-model"]
    publish = workflow["jobs"]["publish-dispatched-benchmark"]

    assert benchmark["outputs"] == {
        "performance": "${{ steps.performance.outcome }}",
        "accuracy": "${{ steps.accuracy.outcome }}",
    }
    # Shared setup gates both stages; a failed accuracy dataset step does not skip performance.
    assert step(benchmark["steps"], "Install selected Neat Core")["if"] == (
        "${{ !cancelled() && steps.runtime.outcome == 'success' }}"
    )
    assert "needs.benchmark-dispatched-model.result" not in publish["if"]
    assert "needs.benchmark-dispatched-model.outputs.performance == 'success' ||" in publish["if"]
    assert "needs.benchmark-dispatched-model.outputs.accuracy == 'success'" in publish["if"]


def test_credentialed_devkit_steps_never_run_model_code() -> None:
    benchmark = load_workflow()["jobs"]["benchmark-dispatched-model"]
    steps = benchmark["steps"]
    names = [item.get("name") for item in steps]
    credentials = step(steps, "Configure dataset reader credentials")
    cache = step(steps, "Cache the accuracy dataset on NVMe")

    # Credentials stay in step outputs, so only steps that name them receive them.
    assert credentials["id"] == "dataset-credentials"
    assert credentials["with"]["output-credentials"] == "true"
    assert credentials["with"]["output-env-credentials"] == "false"
    assert credentials["with"]["role-to-assume"] == (
        "${{ needs.resolve-dispatched-benchmark-config.outputs.dataset_reader_role_arn }}"
    )
    credentialed = [item["name"] for item in steps if "steps.dataset-credentials." in yaml.safe_dump(item)]
    assert credentialed == ["Cache the accuracy dataset on NVMe"]
    assert "AWS_" not in yaml.safe_dump(benchmark.get("env", {}))
    assert "${{ secrets." not in yaml.safe_dump(benchmark)
    # The trusted checkout's script resolves the dataset; the source revision is read only as model metadata.
    assert cache["run"] == 'scripts/cache_accuracy_dataset.sh "_source/models/$MODEL_ID" "$BUILD_PROFILE_ID"'
    assert all(marker not in cache["run"] for marker in MODEL_CODE_MARKERS)
    assert names.index("Check out trusted tooling from the protected workflow revision") < names.index(
        "Cache the accuracy dataset on NVMe"
    )
    # Steps that run model code receive no credentials.
    for item in steps:
        if any(marker in item.get("run", "") for marker in MODEL_CODE_MARKERS):
            assert "AWS_" not in yaml.safe_dump(item.get("env", {})), item["name"]

    cache_script = (ROOT / "scripts" / "cache_accuracy_dataset.sh").read_text(encoding="utf-8")
    assert "resolve-devkit-accuracy" in cache_script
    for model_code in ("accuracy.py", "performance.py", "compile.py", "pip install"):
        assert model_code not in cache_script


def test_devkit_steps_use_only_tools_the_board_provides() -> None:
    steps = load_workflow()["jobs"]["benchmark-dispatched-model"]["steps"]
    runs = "\n".join(item.get("run", "") for item in steps)
    scripts = sorted({word.strip('"') for word in runs.split() if word.startswith("scripts/") and word.endswith(".sh")})

    assert "scripts/cache_accuracy_dataset.sh" in scripts
    # The DevKit image has no jq; JSON is read with the registry Python.
    for text in [runs, *((ROOT / script).read_text(encoding="utf-8") for script in scripts)]:
        assert "jq " not in text


def test_accuracy_dataset_is_cached_on_nvme_under_a_shared_lock() -> None:
    steps = load_workflow()["jobs"]["benchmark-dispatched-model"]["steps"]
    names = [item.get("name") for item in steps]
    order = [
        names.index(name)
        for name in (
            "Resolve and verify benchmark runtime",
            "Install trusted dataset tooling",
            "Configure dataset reader credentials",
            "Cache the accuracy dataset on NVMe",
            "Measure accuracy on DevKit",
        )
    ]

    assert order == sorted(order)
    install = step(steps, "Install trusted dataset tooling")
    assert 'scripts/install_dataset_registry.sh "$REGISTRY_PYTHON"' in install["run"]
    install_script = (ROOT / "scripts" / "install_dataset_registry.sh").read_text(encoding="utf-8")
    assert "--download-only" in install_script
    assert "dataset/dataset-registry@main:latest" in install_script
    cache_script = (ROOT / "scripts" / "cache_accuracy_dataset.sh").read_text(encoding="utf-8")
    accuracy_script = (ROOT / "scripts" / "run_model_accuracy.sh").read_text(encoding="utf-8")
    assert 'NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"' in cache_script
    assert 'mountpoint -q "$NVME_MOUNT"' in cache_script
    assert "coco" not in cache_script.lower()
    # Both scripts serialize resolution through the board-wide dataset lock, under a timeout.
    assert 'DATASET_LOCK="${DATASET_LOCK:-$NVME_MOUNT/datasets/locks/prepare.lock}"' in cache_script
    assert 'DATASET_LOCK="${DATASET_LOCK:-/media/nvme/datasets/locks/prepare.lock}"' in accuracy_script
    for script in (cache_script, accuracy_script):
        assert 'flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK"' in script
        assert 'timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS"' in script
    assert "/media/nvme/datasets/artifacts" in accuracy_script
    assert "$NVME_MOUNT/datasets/artifacts" in cache_script


def test_dispatched_benchmark_runs_only_on_its_source_branch() -> None:
    workflow = load_workflow()
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


def test_completed_stages_are_published_from_a_hosted_runner() -> None:
    publish = load_workflow()["jobs"]["publish-dispatched-benchmark"]
    steps = publish["steps"]

    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["needs"] == ["benchmark-dispatched-model", "resolve-dispatched-benchmark-config"]
    checkouts = [item for item in steps if item.get("uses") == "actions/checkout@v6"]
    assert checkouts[0]["with"] == {"ref": "${{ github.sha }}", "persist-credentials": "false"}
    assert checkouts[1]["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "path": "_source",
        "persist-credentials": "false",
    }
    assert all(marker not in item.get("run", "") for item in steps for marker in MODEL_CODE_MARKERS)
    credentials = next(item for item in steps if item.get("id") == "benchmark-publisher-credentials")
    assert credentials["with"]["output-credentials"] == "true"
    assert credentials["with"]["output-env-credentials"] == "false"
    credentialed = [item["name"] for item in steps if "steps.benchmark-publisher-credentials." in yaml.safe_dump(item)]
    assert credentialed == ["Publish each component's performance run", "Publish the accuracy run"]


@pytest.mark.parametrize(
    ("name", "condition"),
    [
        ("Download performance evidence", PUBLISH_PERFORMANCE),
        ("Download accuracy evidence", PUBLISH_ACCURACY),
        ("Download the registered compilation the accuracy measured", PUBLISH_ACCURACY),
        ("Publish each component's performance run", PUBLISH_PERFORMANCE),
        ("Publish the accuracy run", PUBLISH_ACCURACY),
    ],
)
def test_publication_steps_follow_the_requested_stages(name, condition) -> None:
    steps = load_workflow()["jobs"]["publish-dispatched-benchmark"]["steps"]

    assert step(steps, name)["if"] == condition


def test_each_component_performance_run_is_published_against_the_source_revision() -> None:
    steps = load_workflow()["jobs"]["publish-dispatched-benchmark"]["steps"]
    download = step(steps, "Download performance evidence")
    publication = step(steps, "Publish each component's performance run")
    run = publication["run"]

    assert download["with"] == {
        "name": "benchmark-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}",
        "path": "_benchmarks",
    }
    assert publication["env"]["AWS_ACCESS_KEY_ID"] == (
        "${{ steps.benchmark-publisher-credentials.outputs.aws-access-key-id }}"
    )
    assert publication["env"]["COMPONENT_IDS"] == "${{ inputs.component_ids }}"
    assert publication["env"]["SOURCE_BRANCH"] == "${{ inputs.source_branch }}"
    assert run.index("model-registry validate-benchmark-set") < run.index('for component in "${components[@]}"')
    assert run.index('for component in "${components[@]}"') < run.index("model-registry publish-benchmark")
    assert run.count("--repository _source") == 2
    assert '--result "_benchmarks/$component/performance-result.json"' in run
    assert '--log "_benchmarks/$component/benchmark.log"' in run
    assert "--schema schemas/performance-result.schema.json" in run
    assert '--branch "$SOURCE_BRANCH"' in run
    assert '--git-sha "$SOURCE_SHA"' in run


def test_accuracy_run_is_published_with_the_compilation_it_measured() -> None:
    steps = load_workflow()["jobs"]["publish-dispatched-benchmark"]["steps"]
    suffix = "${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    publication = step(steps, "Publish the accuracy run")["run"]

    assert step(steps, "Download accuracy evidence")["with"] == {"name": f"accuracy-{suffix}", "path": "_accuracy"}
    assert step(steps, "Download the registered compilation the accuracy measured")["with"] == {
        "name": f"benchmark-input-{suffix}",
        "path": "_compiled",
    }
    assert "model-registry publish-accuracy" in publication
    assert "--repository _source" in publication
    assert "--compiled-dir _compiled" in publication
    assert "--result _accuracy/accuracy-result.json" in publication
    assert "--log _accuracy/accuracy.log" in publication
    assert '--git-sha "$SOURCE_SHA"' in publication


def test_cleanup_removes_input_and_both_evidence_artifacts() -> None:
    cleanup = load_workflow()["jobs"]["cleanup-dispatched-benchmark-artifacts"]
    script = cleanup["steps"][0]["run"]

    assert cleanup["needs"] == ["benchmark-dispatched-model", "publish-dispatched-benchmark"]
    assert cleanup["if"] == (
        "!cancelled() && needs.benchmark-dispatched-model.result == 'success' && "
        "needs.publish-dispatched-benchmark.result == 'success'"
    )
    for prefix in ("benchmark-input-", "benchmark-", "accuracy-"):
        assert f"{prefix}$MODEL_ID-$BUILD_PROFILE_ID-$COMPILE_INVOCATION_ID" in script
