import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from model_registry.fingerprints import ACCURACY_TOOLING, LLIMA_PERFORMANCE_TOOLING, PERFORMANCE_TOOLING

from tests.test_compile_workflow import evaluate
from tests.test_stage_plan import repository_without_w16a16_accuracy

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-model-benchmarking.yml"
# A manual dispatch's "default" environment resolves to the repository's VULCAN_ENV.
ENVIRONMENT = "${{ inputs.environment != 'default' && inputs.environment || vars.VULCAN_ENV || 'staging' }}"
PUBLISH_PERFORMANCE = "${{ !cancelled() && needs.benchmark.outputs.performance == 'success' }}"
PUBLISH_ACCURACY = "${{ !cancelled() && needs.benchmark.outputs.accuracy == 'success' }}"
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


def test_manual_benchmarks_keep_their_actions_identity() -> None:
    workflow = load_workflow()

    assert workflow["name"] == "Vulcan Model Benchmarking"
    assert workflow["run-name"] == (
        "Benchmarking (${{ inputs.model_id }}/${{ inputs.compile_invocation_id }}, ${{ inputs.build_profile_id }})"
    )
    assert list(workflow["on"]) == ["workflow_dispatch", "workflow_call"]
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
    assert inputs["environment"]["type"] == "choice"
    assert inputs["environment"]["options"] == ["default", "staging", "production"]
    assert inputs["environment"]["default"] == "default"
    assert "operation" not in inputs
    assert workflow["concurrency"]["group"].startswith(
        "model-benchmark-${{ inputs.environment != 'default' && inputs.environment || vars.VULCAN_ENV || 'staging' }}-"
    )
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert list(workflow["jobs"]) == [
        "resolve-vulcan-config", "resolve-benchmark-runner", "stage", "benchmark", "publish", "check",
    ]


def test_manual_dispatch_takes_the_called_inputs() -> None:
    triggers = load_workflow()["on"]
    inputs, call = triggers["workflow_dispatch"]["inputs"], triggers["workflow_call"]["inputs"]

    # The resolvers' outputs are the only inputs a person does not type.
    assert set(inputs) == set(call) - {"vulcan_config", "devkit_runner"}
    assert all(value["type"] == "string" for value in call.values())
    optional = {"source_run_id", "vulcan_config", "devkit_runner"}
    assert {name for name, value in call.items() if value["required"] == "false"} == optional
    assert all(call[name]["default"] == "" for name in optional)
    assert all("default" not in value for name, value in call.items() if name not in optional)


def test_only_a_manual_dispatch_resolves_the_config_and_runner() -> None:
    jobs = load_workflow()["jobs"]
    stage = jobs["stage"]

    # The orchestrator passes the config and the DevKit runner it resolved once for the run.
    assert jobs["resolve-vulcan-config"]["if"] == "inputs.vulcan_config == ''"
    assert jobs["resolve-benchmark-runner"]["if"] == "inputs.devkit_runner == ''"
    assert stage["needs"] == ["resolve-vulcan-config", "resolve-benchmark-runner"]
    # Without a status function a skipped resolver would skip the stage too.
    assert stage["if"].startswith("!cancelled() && !contains(needs.*.result, 'failure') &&")
    assert stage["outputs"]["vulcan_config"] == (
        "${{ inputs.vulcan_config || toJSON(needs.resolve-vulcan-config.outputs) }}"
    )
    assert stage["outputs"]["devkit_runner"] == (
        "${{ inputs.devkit_runner || toJSON(needs.resolve-benchmark-runner.outputs) }}"
    )


def test_benchmark_stages_only_its_compiled_artifact() -> None:
    workflow = load_workflow()
    stage = workflow["jobs"]["stage"]

    assert stage["environment"] == ENVIRONMENT
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
    # An orchestrator run holds every build's artifacts, more than one listing page shows.
    assert "/artifacts?name=$artifact_name" in verify["run"]
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
    assert "if" not in restore
    assert restore["run"] == "scripts/stage_benchmark_inputs.sh _source _compiled"
    script = (ROOT / "scripts/stage_benchmark_inputs.sh").read_text()
    assert 'if [[ -z "$SOURCE_RUN_ID" ]]' in script
    assert 'model-registry stage-registry-build "${args[@]}" --benchmark-baseline' in script
    # A benchmark runs under the profile's published package; the stage never registers a build itself.
    registered = step(stage["steps"], "Verify the registered build and its package")
    assert "if" not in registered
    assert "model-registry model-component-artifacts _compiled --repository _source" in registered["run"]
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


def test_benchmark_runs_on_the_devkit_its_caller_resolved() -> None:
    resolver = load_workflow()["jobs"]["resolve-benchmark-runner"]
    benchmark = load_workflow()["jobs"]["benchmark"]
    steps = benchmark["steps"]

    assert resolver["uses"].endswith("/vulcan-resolve-modalix-runner.yml@main")
    assert json.loads(resolver["with"]["runner-labels"]) == [
        "self-hosted", "Linux", "ARM64", "modalix", "modalix-som-16gig", "model-registry", "vulcan-staging",
    ]
    assert benchmark["needs"] == "stage"
    assert benchmark["environment"] == ENVIRONMENT
    # The stage passes the resolver's outputs as JSON; its runner_labels output is itself a JSON array.
    assert benchmark["runs-on"] == {
        "group": "${{ fromJSON(needs.stage.outputs.devkit_runner).modalix_runner_group }}",
        "labels": "${{ fromJSON(fromJSON(needs.stage.outputs.devkit_runner).runner_labels) }}",
    }
    assert benchmark["env"]["SIMA_CLI_AUTO_ACCEPT_UPDATE"] == "1"
    # One orchestrator run benchmarks many builds on the same board under one run ID.
    assert benchmark["env"]["WORK_DIR"] == (
        "/media/nvme/neat-tests/model-benchmark-${{ github.run_id }}-${{ github.run_attempt }}"
        "-${{ inputs.model_id }}-${{ inputs.build_profile_id }}"
    )
    assert benchmark["env"]["SOURCE_BRANCH"] == "${{ needs.stage.outputs.source_branch }}"
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
    assert install == (
        '"$REGISTRY_PYTHON" -m model_registry ensure-neat-core --repository _source --environment production'
    )
    assert "tools/model_registry/neat_core.py" in ACCURACY_TOOLING
    assert "tools/model_registry/neat_core.py" in PERFORMANCE_TOOLING
    assert "kerrigan" not in BENCHMARK_WORKFLOW.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(("branch", "expected"), [("develop", True), ("feature/other", False)])
def test_devkit_job_runs_only_on_a_staged_build(branch, expected) -> None:
    benchmark = load_workflow()["jobs"]["benchmark"]
    condition = benchmark["if"]

    # A skipped resolver would skip the board under the implicit success(), so the board names the stage's result.
    assert benchmark["needs"] == "stage"
    values = {"inputs.source_branch": branch, "github.ref_name": "develop", "needs.stage.result": "success",
              "cancelled()": False}
    assert evaluate(condition[3:-3], values) is expected
    assert evaluate(condition[3:-3], {**values, "needs.stage.result": "skipped"}) is False


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
    assert '"${mpks[@]}" "${dataset[@]}" "${resources[@]}" --output "$OUTPUT_DIR"' in script
    # A model's CPU work gets its pinned resources and its pinned wheels, beside PyNeat's environment.
    assert "model_registry fetch-resources" in script
    assert 'resources=(--resources "$resource_dir")' in script
    assert script.index("fetch-resources") < script.index('scripts/performance.py"')
    assert 'REQUIREMENTS="$MODEL_DIR/scripts/requirements-performance.txt"' in script
    assert '--target "$site" -r "$REQUIREMENTS"' in script
    assert 'PYTHONPATH="$REPOSITORY/tools:$site"' in script
    # The script picks its benchmark input from the model's first accuracy dataset, unpacked.
    assert "model_registry resolve-devkit-accuracy" in script
    assert 'dataset=(--dataset "$name=$directory")' in script
    assert script.index("resolve-devkit-accuracy") < script.index('scripts/performance.py"')
    assert "model_registry model-component-artifacts" in script
    assert script.index('scripts/performance.py"') < script.index("model_registry check-report")

def test_benchmarks_run_only_when_requested() -> None:
    benchmark = load_workflow()["jobs"]["benchmark"]
    steps = benchmark["steps"]
    artifact_suffix = "${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"

    assert not {"BENCHMARK_PERFORMANCE", "BENCHMARK_ACCURACY"} & set(benchmark["env"])
    performance = step(steps, "Measure performance on DevKit")
    assert performance["if"] == (
        "${{ !cancelled() && ((steps.core.outcome == 'success' && steps.dataset-cache.outcome == 'success') || "
        "steps.llima.outcome == 'success') && steps.stages.outputs.performance == 'true' }}"
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
        assert step(steps, name)["if"] == (
            "env.BENCHMARK_BACKEND == 'core' && (steps.stages.outputs.accuracy == 'true' || "
            "steps.stages.outputs.performance == 'true')"
        ), name


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
    run = step(load_workflow()["jobs"]["benchmark"]["steps"],
               "Select the requested stages the profile declares")["run"]
    (tmp_path / "_source").symlink_to(repository_without_w16a16_accuracy(tmp_path))
    output = tmp_path / "github-output"
    env = {**os.environ, "REGISTRY_PYTHON": sys.executable, "MODEL_ID": "resnet_50", "BUILD_PROFILE_ID": profile,
           "STAGES": stages, "GITHUB_OUTPUT": str(output)}
    subprocess.run(["bash", "-c", run], cwd=tmp_path, env=env, check=True)
    assert output.read_text(encoding="utf-8") == selected


def test_each_benchmark_runs_and_publishes_on_its_own_outcome() -> None:
    workflow = load_workflow()
    benchmark = workflow["jobs"]["benchmark"]
    publish = workflow["jobs"]["publish"]

    assert benchmark["outputs"] == {
        "performance": "${{ steps.performance-tests.outcome }}",
        "accuracy": "${{ steps.accuracy-tests.outcome }}",
    }
    # Shared setup gates both stages; each stage checks the dataset step itself.
    assert step(benchmark["steps"], "Install selected Neat Core")["if"] == (
        "${{ !cancelled() && steps.runtime.outcome == 'success' && env.BENCHMARK_BACKEND == 'core' }}"
    )
    assert "needs.benchmark.result" not in publish["if"]
    assert "needs.benchmark.outputs.performance == 'success' ||" in publish["if"]
    assert "needs.benchmark.outputs.accuracy == 'success'" in publish["if"]


def test_each_benchmark_result_records_its_model_test_outcome() -> None:
    steps = load_workflow()["jobs"]["benchmark"]["steps"]
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
    benchmark = load_workflow()["jobs"]["benchmark"]
    steps = benchmark["steps"]
    names = [item.get("name") for item in steps]
    credentials = step(steps, "Configure dataset reader credentials")
    cache = step(steps, "Cache the accuracy dataset on NVMe")

    # Credentials stay in step outputs, so only steps that name them receive them.
    assert credentials["id"] == "dataset-credentials"
    assert credentials["with"]["output-credentials"] == "true"
    assert credentials["with"]["output-env-credentials"] == "false"
    assert credentials["with"]["role-to-assume"] == (
        "${{ fromJSON(needs.stage.outputs.vulcan_config).dataset_reader_role_arn }}"
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
    steps = load_workflow()["jobs"]["benchmark"]["steps"]
    runs = "\n".join(item.get("run", "") for item in steps)
    scripts = sorted({word.strip('"') for word in runs.split() if word.startswith("scripts/") and word.endswith(".sh")})

    assert "scripts/cache_accuracy_dataset.sh" in scripts
    # The DevKit image has no jq; JSON is read with the registry Python.
    for text in [runs, *((ROOT / script).read_text(encoding="utf-8") for script in scripts)]:
        assert "jq " not in text


def test_accuracy_dataset_is_cached_on_nvme_under_a_shared_lock() -> None:
    steps = load_workflow()["jobs"]["benchmark"]["steps"]
    names = [item.get("name") for item in steps]
    order = [
        names.index(name)
        for name in (
            "Resolve and verify benchmark runtime",
            "Install trusted dataset tooling",
            "Configure dataset reader credentials",
            "Cache the accuracy dataset on NVMe",
            "Measure performance on DevKit",
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
    performance_script = (ROOT / "scripts" / "run_model_performance.sh").read_text(encoding="utf-8")
    assert 'NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"' in cache_script
    assert 'mountpoint -q "$NVME_MOUNT"' in cache_script
    assert "coco" not in cache_script.lower()
    # Every script serializes resolution through the board-wide dataset lock, under a timeout.
    assert 'DATASET_LOCK="${DATASET_LOCK:-$NVME_MOUNT/datasets/locks/prepare.lock}"' in cache_script
    for script in (accuracy_script, performance_script):
        assert 'DATASET_LOCK="${DATASET_LOCK:-/media/nvme/datasets/locks/prepare.lock}"' in script
        assert "/media/nvme/datasets/artifacts" in script
    for script in (cache_script, accuracy_script, performance_script):
        assert 'flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK"' in script
        assert 'timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS"' in script
    assert "$NVME_MOUNT/datasets/artifacts" in cache_script


def test_benchmark_runs_only_on_its_source_branch() -> None:
    workflow = load_workflow()

    for name in ("stage", "benchmark", "publish"):
        condition = workflow["jobs"][name]["if"]
        assert "inputs.source_branch == github.ref_name" in condition, name
        assert "refs/heads/develop" not in condition, name


def test_completed_stages_are_published_from_a_hosted_runner() -> None:
    publish = load_workflow()["jobs"]["publish"]
    steps = publish["steps"]

    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["needs"] == ["stage", "benchmark"]
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
    steps = load_workflow()["jobs"]["publish"]["steps"]

    assert step(steps, name)["if"] == condition


def test_each_measured_stage_is_published_under_the_registered_build() -> None:
    steps = load_workflow()["jobs"]["publish"]["steps"]
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
    ("statuses", "failed"),
    [
        ({"_performance/test-status.txt": "passed", "_accuracy/test-status.txt": "skipped"}, ""),
        ({"_performance/test-status.txt": "failed", "_accuracy/test-status.txt": "passed"}, "performance"),
        ({"_accuracy/test-status.txt": "failed"}, "accuracy"),
        ({"_performance/test-status.txt": "failed", "_accuracy/test-status.txt": "failed"}, "performance accuracy"),
        ({}, ""),
    ],
)
def test_run_fails_after_publication_on_a_failed_model_test(
    tmp_path: Path, statuses: dict[str, str], failed: str
) -> None:
    jobs = load_workflow()["jobs"]
    publish, check_job = jobs["publish"], jobs["check"]
    record = publish["steps"][-1]
    check = check_job["steps"][0]
    for relative, status in statuses.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(f"{status}\n", encoding="utf-8")
    output = tmp_path / "github-output"

    subprocess.run(["bash", "-c", record["run"]], cwd=tmp_path, env={**os.environ, "GITHUB_OUTPUT": str(output)},
                   check=True)
    recorded = output.read_text(encoding="utf-8").splitlines()
    result = subprocess.run(["bash", "-c", check["run"]], env={**os.environ, "FAILED_STAGES": failed},
                            capture_output=True, text=True, check=False)

    assert record["name"] == "Record the stages whose model tests failed"
    assert record["id"] == "tests"
    assert publish["outputs"]["failed_tests"] == "${{ steps.tests.outputs.failed }}"
    assert recorded == [f"failed={failed}"]
    # A separate job fails the run, so re-running failed jobs never publishes the same results again.
    assert check_job["needs"] == ["benchmark", "publish"]
    assert check_job["runs-on"] == "ubuntu-latest"
    # Deleting the build's artifacts is the only write the check job makes.
    assert check_job["permissions"] == {"actions": "write", "contents": "read"}
    assert check["name"] == "Fail the run on a failed model test"
    assert check["env"]["FAILED_STAGES"] == "${{ needs.publish.outputs.failed_tests }}"
    assert (result.returncode != 0) == bool(failed), result.stderr


@pytest.mark.parametrize(
    ("cancelled", "publish", "expected"),
    [(False, "success", True), (False, "failure", False), (False, "skipped", False), (True, "success", False)],
)
def test_model_tests_are_checked_only_after_publication(cancelled, publish, expected) -> None:
    condition = load_workflow()["jobs"]["check"]["if"]

    assert evaluate(condition[3:-3], {
        "!cancelled()": not cancelled, "needs.publish.result": publish,
    }) is expected


@pytest.mark.parametrize(
    ("benchmark", "expected"),
    [
        ("success", True),
        # A failed stage keeps the evidence it failed on, though the other stage published.
        ("failure", False),
    ],
)
def test_cleanup_waits_for_the_model_test_check(benchmark, expected) -> None:
    check = load_workflow()["jobs"]["check"]
    names = [item["name"] for item in check["steps"]]
    cleanup = step(check["steps"], "Delete benchmark input and evidence from GitHub")

    # A failed model test fails the step before it, and without a status function a failed step skips the cleanup.
    assert names == ["Fail the run on a failed model test", "Delete benchmark input and evidence from GitHub"]
    assert not any(function in cleanup["if"] for function in ("always()", "cancelled()", "failure()"))
    assert evaluate(cleanup["if"], {"needs.benchmark.result": benchmark}) is expected


def test_cleanup_removes_input_and_both_evidence_artifacts(tmp_path: Path) -> None:
    script = step(load_workflow()["jobs"]["check"]["steps"], "Delete benchmark input and evidence from GitHub")["run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    # The run lists only one artifact per name; each lookup answers with that artifact's ID.
    (bin_dir / "gh").write_text(
        f'#!/bin/sh\necho "$*" >> {calls}\n'
        'case "$*" in *"--method DELETE"*) ;; *name=benchmark-input-*) echo 1;; *name=performance-*) echo 2;; '
        '*name=accuracy-*) echo 3;; esac\n',
        encoding="utf-8",
    )
    (bin_dir / "gh").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r",
           "GITHUB_RUN_ID": "9", "MODEL_ID": "m", "BUILD_PROFILE_ID": "p", "COMPILE_INVOCATION_ID": "c"}

    subprocess.run(["bash", "-c", script], env=env, check=True)

    # The lookups and deletions run as one pipeline, so their calls interleave.
    assert sorted(calls.read_text(encoding="utf-8").splitlines()) == sorted([
        *(f"api /repos/o/r/actions/runs/9/artifacts?name={prefix}-m-p-c --jq .artifacts[].id"
          for prefix in ("benchmark-input", "performance", "accuracy")),
        *(f"api --method DELETE /repos/o/r/actions/artifacts/{artifact_id}" for artifact_id in (1, 2, 3)),
    ])


@pytest.mark.parametrize("relative", [
    "tools/model_registry/report_publication.py",
    "tools/model_registry/benchmark_declarations.py", "tools/model_registry/model.py",
])
def test_board_runs_require_the_measured_revisions_tooling(tmp_path, relative) -> None:
    stage = load_workflow()["jobs"]["stage"]
    guard = step(stage["steps"], "Require the measured revision's board-stage tooling")
    assert guard["run"] == "scripts/require_source_board_tooling.sh _source"
    source = tmp_path / "_source"
    for item in {*PERFORMANCE_TOOLING, *ACCURACY_TOOLING, *LLIMA_PERFORMANCE_TOOLING,
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
    # A manual dispatch may pass "default"; the orchestrator passes staging or production.
    assert evaluate(ENVIRONMENT[3:-3], values) == expected
    assert jobs['resolve-vulcan-config']['with']['vulcan_env'] == ENVIRONMENT
    assert 'vulcan-staging' in json.loads(jobs['resolve-benchmark-runner']['with']['runner-labels'])
    resolved = {'inputs.environment': expected, 'vars.VULCAN_ENV': default}
    for name in ('stage', 'benchmark', 'publish'):
        assert jobs[name]['environment'] == ENVIRONMENT
        assert "inputs.environment == 'staging'" not in jobs[name]['if']
    board = jobs['benchmark']
    dataset = step(board['steps'], 'Install trusted dataset tooling')['env']['DATASET_REGISTRY_ENVIRONMENT']
    assert dataset == ENVIRONMENT
    reader = step(jobs['stage']['steps'], 'Configure authenticated production registry reads')
    for source_run in ('', '123'):
        assert evaluate(reader['if'], {**resolved, 'inputs.source_run_id': source_run}) == (
            expected == 'production'
        )
    restore = step(jobs['stage']['steps'],
                   'Restore the current immutable compilation from the registry')
    urls = {**resolved, 'vars.VULCAN_ARTIFACT_REGISTRY_API_URL': 'https://iam.example',
            'vars.VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL': 'https://public.example'}
    assert evaluate(restore['env']['REGISTRY_API_URL'][3:-3], urls) == 'https://public.example'
    assert evaluate(restore['env']['MODEL_REGISTRY_IAM_READ_API_URL'][3:-3], urls) == (
        'https://iam.example' if expected == 'production' else ''
    )


@pytest.mark.parametrize('environment', ['staging', 'production'])
def test_report_command_publishes_both_stages_to_selected_registry(tmp_path, environment):
    publication = step(load_workflow()['jobs']['publish']['steps'],
                       "Publish each measured stage's run")
    assert publication['env']['VULCAN_ENV'] == ENVIRONMENT
    selected = evaluate(ENVIRONMENT[3:-3], {'inputs.environment': 'default', 'vars.VULCAN_ENV': environment})
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


@pytest.mark.parametrize("supporting_files", [False, True])
def test_accuracy_runner_passes_only_verified_local_resources(tmp_path, supporting_files):
    import hashlib
    import io
    import tarfile

    compiled = tmp_path / "compiled with spaces"
    profile = compiled / "clip_vit_b_16/w8a16"
    profile.mkdir(parents=True)
    manifest = {}
    if supporting_files:
        asset = profile / "tokenizer.json"
        asset.write_text("{}")
        manifest["runtime_files"] = [{"path": asset.name, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}]
    (profile / "compile-outputs.json").write_text(json.dumps(manifest))
    dataset = tmp_path / "dataset.tgz"
    with tarfile.open(dataset, "w:gz") as archive:
        info = tarfile.TarInfo("val2012/image.JPEG")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"img"))
    registry = tmp_path / "registry-python"
    registry.write_text(f'''#!{sys.executable}
import json, os, sys
if sys.argv[1] == '-c' or sys.argv[3] == 'model-runtime-directory':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
command = sys.argv[3]
if command == 'resolve-devkit-accuracy':
    print(json.dumps({{"name": "imagenet-2012-val", "path": {str(dataset)!r}}}))
elif command == 'model-component-mpks':
    print('image_encoder=image.tar.gz\\ntext_encoder=text.tar.gz')
elif command != 'check-report':
    raise AssertionError(command)
''')
    board = tmp_path / "board-python"
    captured = tmp_path / "arguments.json"
    board.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
if sys.argv[1:3] != ['-m', 'pip']:
    Path({str(captured)!r}).write_text(json.dumps(sys.argv[2:]))
''')
    registry.chmod(0o755)
    board.chmod(0o755)
    environment = {
        **os.environ, "PYTHONPATH": str(ROOT / "tools"), "REGISTRY_PYTHON": str(registry),
        "PYNEAT_PYTHON": str(board), "WORK_DIR": str(tmp_path / "scratch"),
        "DATASET_CACHE": str(tmp_path / "cache"), "DATASET_LOCK": str(tmp_path / "prepare.lock"),
    }
    run = subprocess.run(
        [ROOT / "scripts/run_model_accuracy.sh", ROOT / "models/clip_vit_b_16", "w8a16", compiled, tmp_path / "report"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    arguments = json.loads(captured.read_text())
    assert ("--resources" in arguments) == supporting_files
    if supporting_files:
        assert arguments[arguments.index("--resources") + 1] == str(profile)
    assert arguments.count("--mpk") == 2
