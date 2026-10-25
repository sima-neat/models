import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from model_registry.fingerprints import ACCURACY_TOOLING, PERFORMANCE_TOOLING

from tests.test_compile_workflow import evaluate
from tests.test_stage_plan import repository_without_w16a16_accuracy

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-model-benchmarking.yml"
PUBLISH_PERFORMANCE = "${{ !cancelled() && needs.benchmark-dispatched-model.outputs.performance == 'success' }}"
PUBLISH_ACCURACY = "${{ !cancelled() && needs.benchmark-dispatched-model.outputs.accuracy == 'success' }}"
# Model code from the source revision runs only through these steps' scripts.
MODEL_CODE_MARKERS = (
    "scripts/run_model_performance.sh",
    "scripts/run_model_accuracy.sh",
    "scripts/run_model_tests.sh",
    "pip install _source",
)


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
        "stages",
    }
    optional = {"source_run_id", "stages"}
    assert all(inputs[name]["required"] == "false" for name in optional)
    assert all(value["required"] == "true" for name, value in inputs.items() if name not in optional)
    assert inputs["stages"]["default"] == "performance,accuracy"
    assert inputs["environment"]["options"] == ["default", "staging", "production"]
    assert inputs["environment"]["default"] == "default"
    assert "operation" not in inputs
    assert "model-benchmark-" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"


def test_dispatched_benchmark_stages_only_its_compiled_artifact() -> None:
    workflow = load_workflow()
    stage = workflow["jobs"]["stage-dispatched-benchmark"]

    assert stage["needs"] == "resolve-dispatched-benchmark-config"
    assert stage["environment"] == (
        "${{ inputs.environment != 'default' && inputs.environment || vars.VULCAN_ENV || 'staging' }}"
    )
    assert stage["permissions"]["id-token"] == "write"
    assert stage["outputs"]["source_branch"] == "${{ steps.source.outputs.branch }}"
    assert "inputs.source_branch == github.ref_name" in stage["if"]
    assert "inputs.stages != ''" in stage["if"]
    assert "inputs.operation" not in stage["if"]
    verify = step(stage["steps"], "Verify the source run and compiled artifact")
    assert verify["env"]["STAGES"] == "${{ inputs.stages }}"
    assert '[[ "$stage" == performance || "$stage" == accuracy ]]' in verify["run"]
    assert "COMPONENT_IDS" not in BENCHMARK_WORKFLOW.read_text(encoding="utf-8")
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
    restore = step(stage["steps"], "Restore the current immutable compilation from the registry")
    assert restore["if"] == "inputs.source_run_id == ''"
    assert "model-registry stage-registry-build" in restore["run"]
    # A benchmark runs under the profile's published package; the stage never registers a build itself.
    registered = step(stage["steps"], "Verify the registered build and its package")
    assert "if" not in registered
    assert "model-registry model-component-mpks _compiled --repository _source" in registered["run"]
    assert '--model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID"' in registered["run"]
    assert (
        'test -f "_compiled/$MODEL_ID/$BUILD_PROFILE_ID/packages/default/registry-publication.json"'
        in registered["run"]
    )
    assert all("model-registry publish" not in item.get("run", "") for item in stage["steps"])
    credentials = step(stage["steps"], "Configure authenticated production registry reads")
    assert credentials["with"]["output-env-credentials"] == "false"
    assert credentials["with"]["output-credentials"] == "true"
    assert stage["steps"].index(credentials) < stage["steps"].index(restore)
    upload = next(item for item in stage["steps"] if item.get("uses") == "actions/upload-artifact@v7")
    assert upload["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    steps = stage["steps"]
    assert steps.index(download) < steps.index(registered)
    assert steps.index(restore) < steps.index(registered) < steps.index(upload)


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
    assert benchmark["environment"] == (
        "${{ inputs.environment != 'default' && inputs.environment || vars.VULCAN_ENV || 'staging' }}"
    )
    assert benchmark["runs-on"]["group"] == (
        "${{ needs.resolve-dispatched-benchmark-runner.outputs.modalix_runner_group }}"
    )
    assert json.loads(resolver["with"]["runner-labels"]) == [
        "self-hosted", "Linux", "ARM64", "modalix", "model-registry", "vulcan-staging",
    ]
    assert benchmark["runs-on"]["labels"] == (
        "${{ fromJSON(needs.resolve-dispatched-benchmark-runner.outputs.runner_labels) }}"
    )
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
    assert "pip install '.[tests]'" in registry_install["run"]
    assert "|| sudo -n apt-get install -y python3-venv" in registry_install["run"]
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


def test_performance_runs_only_at_the_full_mla_clock() -> None:
    script = (ROOT / "scripts/run_model_performance.sh").read_text(encoding="utf-8")
    clock = script.index('MLA_DEVFREQ=/sys/class/devfreq/5000000.mla')

    assert clock < script.index('"$PYNEAT_PYTHON" "$MODEL_DIR/scripts/performance.py"')
    assert 'if (( $(cat "$MLA_DEVFREQ/cur_freq") < MLA_FREQUENCY_HZ )); then' in script
    assert 'sudo -n tee "$MLA_DEVFREQ/userspace/set_freq"' in script
    assert "if (( mla_frequency_hz != MLA_FREQUENCY_HZ )); then" in script


def test_performance_runs_only_at_the_full_cpu_clock() -> None:
    script = (ROOT / "scripts/run_model_performance.sh").read_text(encoding="utf-8")
    clock = script.index("CPU_CPUFREQ=/sys/devices/system/cpu/cpufreq/policy0")

    assert clock < script.index('"$PYNEAT_PYTHON" "$MODEL_DIR/scripts/performance.py"')
    assert 'if (( $(cat "$CPU_CPUFREQ/scaling_cur_freq") != CPU_FREQUENCY_KHZ )); then' in script
    assert 'sudo -n tee "$CPU_CPUFREQ/scaling_setspeed"' in script
    assert "if (( cpu_frequency_khz != CPU_FREQUENCY_KHZ )); then" in script


def test_performance_runs_the_model_owned_script() -> None:
    script = (ROOT / "scripts/run_model_performance.sh").read_text(encoding="utf-8")
    assert '"$PYNEAT_PYTHON" "$MODEL_DIR/scripts/performance.py"' in script
    assert '"${mpks[@]}" --output "$OUTPUT_DIR"' in script
    assert "model_registry model-component-mpks" in script
    assert script.index('scripts/performance.py"') < script.index("model_registry check-report")

def test_benchmarks_run_only_when_requested() -> None:
    benchmark = load_workflow()["jobs"]["benchmark-dispatched-model"]
    steps = benchmark["steps"]
    artifact_suffix = "${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"

    assert not {"BENCHMARK_PERFORMANCE", "BENCHMARK_ACCURACY"} & set(benchmark["env"])
    performance = step(steps, "Measure performance on DevKit")
    assert performance["if"] == (
        "${{ !cancelled() && steps.core.outcome == 'success' && steps.stages.outputs.performance == 'true' }}"
    )
    assert performance["run"] == (
        'scripts/run_model_performance.sh "_source/models/$MODEL_ID" "$BUILD_PROFILE_ID" _compiled _performance'
    )
    performance_evidence = step(steps, "Retain performance evidence")
    assert performance_evidence["if"] == "always() && steps.stages.outputs.performance == 'true'"
    assert performance_evidence["with"]["name"] == f"performance-{artifact_suffix}"
    assert performance_evidence["with"]["if-no-files-found"] == "error"

    accuracy = step(steps, "Measure accuracy on DevKit")
    assert accuracy["if"] == (
        "${{ !cancelled() && steps.core.outcome == 'success' && steps.dataset-cache.outcome == 'success' "
        "&& steps.stages.outputs.accuracy == 'true' }}"
    )
    assert accuracy["run"] == (
        'scripts/run_model_accuracy.sh "_source/models/$MODEL_ID" "$BUILD_PROFILE_ID" _compiled _accuracy'
    )
    accuracy_evidence = step(steps, "Retain accuracy evidence")
    assert accuracy_evidence["if"] == "always() && steps.stages.outputs.accuracy == 'true'"
    assert accuracy_evidence["with"]["name"] == f"accuracy-{artifact_suffix}"
    assert accuracy_evidence["with"]["if-no-files-found"] == "error"
    for name in ("Install trusted dataset tooling", "Configure dataset reader credentials",
                 "Cache the accuracy dataset on NVMe"):
        assert step(steps, name)["if"] == "steps.stages.outputs.accuracy == 'true'", name


@pytest.mark.parametrize(
    ("profile", "stages", "selected"),
    [
        ("w8a8", "performance,accuracy", "performance=true\naccuracy=true\n"),
        ("w8a8", "performance", "performance=true\naccuracy=false\n"),
        # A manual dispatch's default stages on a profile with no accuracy qualification.
        ("w16a16", "performance,accuracy", "performance=true\naccuracy=false\n"),
    ],
)
def test_benchmarks_run_only_the_requested_stages_the_profile_declares(
    tmp_path: Path, profile: str, stages: str, selected: str
) -> None:
    run = step(load_workflow()["jobs"]["benchmark-dispatched-model"]["steps"],
               "Select the requested stages the profile declares")["run"]
    (tmp_path / "_source").symlink_to(repository_without_w16a16_accuracy(tmp_path))
    output = tmp_path / "github-output"
    env = {**os.environ, "REGISTRY_PYTHON": sys.executable, "MODEL_ID": "resnet_50", "BUILD_PROFILE_ID": profile,
           "STAGES": stages, "GITHUB_OUTPUT": str(output)}
    subprocess.run(["bash", "-c", run], cwd=tmp_path, env=env, check=True)
    assert output.read_text(encoding="utf-8") == selected


def test_each_benchmark_runs_and_publishes_on_its_own_outcome() -> None:
    workflow = load_workflow()
    benchmark = workflow["jobs"]["benchmark-dispatched-model"]
    publish = workflow["jobs"]["publish-dispatched-benchmark"]

    assert benchmark["outputs"] == {
        "performance": "${{ steps.performance-tests.outcome }}",
        "accuracy": "${{ steps.accuracy-tests.outcome }}",
    }
    # Shared setup gates both stages; a failed accuracy dataset step does not skip performance.
    assert step(benchmark["steps"], "Install selected Neat Core")["if"] == (
        "${{ !cancelled() && steps.runtime.outcome == 'success' }}"
    )
    assert "needs.benchmark-dispatched-model.result" not in publish["if"]
    assert "needs.benchmark-dispatched-model.outputs.performance == 'success' ||" in publish["if"]
    assert "needs.benchmark-dispatched-model.outputs.accuracy == 'success'" in publish["if"]


def test_each_benchmark_result_records_its_model_test_outcome() -> None:
    steps = load_workflow()["jobs"]["benchmark-dispatched-model"]["steps"]
    names = [item.get("name") for item in steps]
    performance_tests = step(steps, "Test the performance report with the model's performance tests")
    accuracy_tests = step(steps, "Test the accuracy report with the model's accuracy tests")

    for stage, item in (("performance", performance_tests), ("accuracy", accuracy_tests)):
        assert item["id"] == f"{stage}-tests"
        assert item["if"] == f"${{{{ !cancelled() && steps.{stage}.outcome == 'success' }}}}"
        # The tests read the stage's whole-model report and the build profile it measured.
        assert (
            f'scripts/run_model_tests.sh {stage} "_source/models/$MODEL_ID" _{stage}/model-report.json \\\n'
            f'  "$BUILD_PROFILE_ID" > _{stage}/test-status.txt'
        ) in item["run"]
        # A failed test is an outcome to publish; only an error running the tests fails the step.
        assert "== failed ]]" in item["run"]
    for first, second in (
        ("Measure performance on DevKit", "Test the performance report with the model's performance tests"),
        ("Test the performance report with the model's performance tests", "Retain performance evidence"),
        ("Measure accuracy on DevKit", "Test the accuracy report with the model's accuracy tests"),
        ("Test the accuracy report with the model's accuracy tests", "Retain accuracy evidence"),
    ):
        assert names.index(first) < names.index(second)
    assert step(steps, "Retain performance evidence")["with"]["path"] == "_performance/"
    assert step(steps, "Retain accuracy evidence")["with"]["path"] == "_accuracy/"


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
    assert credentialed == ["Publish each measured stage's run"]


@pytest.mark.parametrize(
    ("name", "condition"),
    [
        ("Download performance evidence", PUBLISH_PERFORMANCE),
        ("Download accuracy evidence", PUBLISH_ACCURACY),
        ("Publish each measured stage's run", "${{ !cancelled() }}"),
    ],
)
def test_publication_steps_follow_the_requested_stages(name, condition) -> None:
    steps = load_workflow()["jobs"]["publish-dispatched-benchmark"]["steps"]

    assert step(steps, name)["if"] == condition


def test_each_measured_stage_is_published_under_the_registered_build() -> None:
    steps = load_workflow()["jobs"]["publish-dispatched-benchmark"]["steps"]
    suffix = "${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    publication = step(steps, "Publish each measured stage's run")
    run = publication["run"]

    for stage in ("performance", "accuracy"):
        assert step(steps, f"Download {stage} evidence")["with"] == {"name": f"{stage}-{suffix}", "path": f"_{stage}"}
    # Both stages publish under the build and package receipt the board measured.
    registered = step(steps, "Download the registered build the stages measured")
    assert "if" not in registered
    assert registered["with"] == {"name": f"benchmark-input-{suffix}", "path": "_compiled"}
    assert publication["env"]["AWS_ACCESS_KEY_ID"] == (
        "${{ steps.benchmark-publisher-credentials.outputs.aws-access-key-id }}"
    )
    assert publication["env"]["SOURCE_BRANCH"] == "${{ inputs.source_branch }}"
    assert "for stage in performance accuracy; do" in run
    # A stage whose tests did not run downloaded no evidence and is not published.
    assert '[[ -f "_$stage/test-status.txt" ]] || continue' in run
    assert run.count("model-registry publish-report") == 1
    for argument in (
        'model-registry publish-report "_$stage"', '--stage "$stage"', "--repository _source",
        "--compiled-dir _compiled", '--model "$MODEL_ID"', '--build-profile "$BUILD_PROFILE_ID"',
        '--branch "$SOURCE_BRANCH"', '--git-sha "$SOURCE_SHA"', '--test-status "$(<"_$stage/test-status.txt")"',
    ):
        assert argument in run, argument
    # One stage's failed publication does not stop the other's, but still fails the step.
    assert '|| failed=1' in run
    assert run.rstrip().endswith('exit "$failed"')


@pytest.mark.parametrize(
    ("statuses", "fails"),
    [
        ({"_performance/test-status.txt": "passed", "_accuracy/test-status.txt": "skipped"}, False),
        ({"_performance/test-status.txt": "failed", "_accuracy/test-status.txt": "passed"}, True),
        ({"_accuracy/test-status.txt": "failed"}, True),
        ({}, False),
    ],
)
def test_run_fails_after_publication_on_a_failed_model_test(
    tmp_path: Path, statuses: dict[str, str], fails: bool
) -> None:
    steps = load_workflow()["jobs"]["publish-dispatched-benchmark"]["steps"]
    check = steps[-1]
    for relative, status in statuses.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(f"{status}\n", encoding="utf-8")

    result = subprocess.run(["bash", "-c", check["run"]], cwd=tmp_path, capture_output=True, text=True, check=False)

    assert check["name"] == "Fail the run on a failed model test"
    assert check["if"] == "${{ !cancelled() }}"
    assert (result.returncode != 0) == fails, result.stderr


def test_cleanup_removes_input_and_both_evidence_artifacts() -> None:
    cleanup = load_workflow()["jobs"]["cleanup-dispatched-benchmark-artifacts"]
    script = cleanup["steps"][0]["run"]

    assert cleanup["needs"] == ["benchmark-dispatched-model", "publish-dispatched-benchmark"]
    assert cleanup["if"] == (
        "!cancelled() && needs.benchmark-dispatched-model.result == 'success' && "
        "needs.publish-dispatched-benchmark.result == 'success'"
    )
    for prefix in ("benchmark-input-", "performance-", "accuracy-"):
        assert f"{prefix}$MODEL_ID-$BUILD_PROFILE_ID-$COMPILE_INVOCATION_ID" in script


@pytest.mark.parametrize("relative", [
    "tools/model_registry/report_publication.py", "tools/model_report/metrics.json",
    "tools/model_registry/benchmark_declarations.py", "tools/model_registry/model.py",
])
def test_board_runs_require_the_measured_revisions_tooling(tmp_path, relative) -> None:
    stage = load_workflow()["jobs"]["stage-dispatched-benchmark"]
    guard = step(stage["steps"], "Require the measured revision's board-stage tooling")
    assert guard["run"] == "scripts/require_source_board_tooling.sh _source"
    source = tmp_path / "_source"
    for item in {*PERFORMANCE_TOOLING, *ACCURACY_TOOLING,
                 "tools/model_registry/benchmark_declarations.py", "tools/model_registry/model.py"}:
        (source / item).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item, source / item)
    python_bin = tmp_path / "bin"
    python_bin.mkdir()
    # The CI step's `python` is the one with model-registry installed.
    (python_bin / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    (python_bin / "python").chmod(0o755)
    env = {**os.environ, "PATH": f"{python_bin}{os.pathsep}{os.environ['PATH']}"}

    def check() -> subprocess.CompletedProcess:
        return subprocess.run(
            [ROOT / "scripts" / "require_source_board_tooling.sh", source],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )

    assert check().returncode == 0
    changed = source / relative
    changed.write_text(changed.read_text(encoding="utf-8") + "# newer\n", encoding="utf-8")
    result = check()
    assert result.returncode != 0
    assert relative in result.stderr


@pytest.mark.parametrize(
    ('override', 'default', 'expected'),
    [('default', 'production', 'production'), ('default', 'staging', 'staging'),
     ('default', '', 'staging'), ('staging', 'production', 'staging'),
     ('production', 'staging', 'production')],
)
def test_registry_selection_is_independent_of_devkit_placement(override, default, expected):
    jobs = load_workflow()['jobs']
    values = {'inputs.environment': override, 'vars.VULCAN_ENV': default}
    for name in ('stage-dispatched-benchmark', 'benchmark-dispatched-model', 'publish-dispatched-benchmark'):
        assert evaluate(jobs[name]['environment'][3:-3], values) == expected
        assert "inputs.environment == 'staging'" not in jobs[name]['if']
    config = jobs['resolve-dispatched-benchmark-config']['with']['vulcan_env']
    assert evaluate(config[3:-3], values) == expected
    board = jobs['benchmark-dispatched-model']
    dataset = step(board['steps'], 'Install trusted dataset tooling')['env']['DATASET_REGISTRY_ENVIRONMENT']
    assert evaluate(dataset[3:-3], values) == expected
    assert 'vulcan-staging' in json.loads(jobs['resolve-dispatched-benchmark-runner']['with']['runner-labels'])
    reader = step(jobs['stage-dispatched-benchmark']['steps'], 'Configure authenticated production registry reads')
    for source_run in ('', '123'):
        assert evaluate(reader['if'], {**values, 'inputs.source_run_id': source_run}) == (
            expected == 'production' and not source_run
        )
    restore = step(jobs['stage-dispatched-benchmark']['steps'],
                   'Restore the current immutable compilation from the registry')
    urls = {**values, 'vars.VULCAN_ARTIFACT_REGISTRY_API_URL': 'https://iam.example',
            'vars.VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL': 'https://public.example'}
    assert evaluate(restore['env']['REGISTRY_API_URL'][3:-3], urls) == 'https://public.example'
    assert evaluate(restore['env']['MODEL_REGISTRY_IAM_READ_API_URL'][3:-3], urls) == (
        'https://iam.example' if expected == 'production' else ''
    )


@pytest.mark.parametrize('environment', ['staging', 'production'])
def test_report_command_publishes_both_stages_to_selected_registry(tmp_path, environment):
    publication = step(load_workflow()['jobs']['publish-dispatched-benchmark']['steps'],
                       "Publish each measured stage's run")
    selected = evaluate(publication['env']['VULCAN_ENV'][3:-3],
                        {'inputs.environment': 'default', 'vars.VULCAN_ENV': environment})
    executable = tmp_path / 'model-registry'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    executable.chmod(0o755)
    for stage_name in ('performance', 'accuracy'):
        directory = tmp_path / f'_{stage_name}'
        directory.mkdir()
        (directory / 'test-status.txt').write_text('passed\n')
    env = {**os.environ, 'PATH': f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
           'CALLS': str(tmp_path / 'calls'), 'VULCAN_ENV': selected,
           **dict.fromkeys(('MODEL_ID', 'BUILD_PROFILE_ID', 'REGISTRY_API_URL', 'AWS_REGION',
                            'ARTIFACT_BUCKET', 'KMS_KEY_ID', 'SOURCE_BRANCH', 'SOURCE_SHA',
                            'GITHUB_SERVER_URL', 'GITHUB_REPOSITORY', 'GITHUB_RUN_ID'), 'fixture')}
    subprocess.run(['bash', '-c', publication['run']], cwd=tmp_path, env=env, check=True)
    calls = (tmp_path / 'calls').read_text().splitlines()
    assert len(calls) == 2
    assert all(f'--environment {environment} ' in call for call in calls)
    assert all('--compiled-dir _compiled ' in call for call in calls)
