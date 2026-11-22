import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
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
    assert "inputs.promoted_missing" in workflow["run-name"]
    assert "format('Benchmarking ({0}/{1}, {2})'" in workflow["run-name"]
    assert list(workflow["on"]) == ["workflow_dispatch", "workflow_call"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "environment",
        "source_sha",
        "source_branch",
        "deployment_branch",
        "model_id",
        "build_profile_id",
        "compile_invocation_id",
        "stages",
        "promoted_missing",
    }
    optional = set(inputs) - {"environment"}
    assert all(inputs[name]["required"] == "false" for name in optional)
    assert all(value["required"] == "true" for name, value in inputs.items() if name not in optional)
    assert inputs["stages"]["default"] == "performance,accuracy"
    assert inputs["environment"]["type"] == "choice"
    assert inputs["environment"]["options"] == ["default", "staging", "production"]
    assert inputs["environment"]["default"] == "default"
    assert inputs["promoted_missing"]["type"] == "boolean"
    assert inputs["promoted_missing"]["default"] == "false"
    assert "operation" not in inputs
    assert workflow["concurrency"]["group"].startswith(
        "model-benchmark-${{ inputs.environment != 'default' && inputs.environment || vars.VULCAN_ENV || 'staging' }}-"
    )
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert list(workflow["jobs"]) == [
        "resolve-vulcan-config", "resolve-benchmark-runner", "promoted", "stage", "benchmark", "publish", "check",
    ]


def test_manual_dispatch_takes_the_called_inputs() -> None:
    triggers = load_workflow()["on"]
    inputs, call = triggers["workflow_dispatch"]["inputs"], triggers["workflow_call"]["inputs"]

    # The resolvers' outputs are the only inputs a person does not type.
    assert set(inputs) - {"promoted_missing"} == set(call) - {"vulcan_config", "devkit_runner"}
    assert all(value["type"] == "string" for value in call.values())
    optional = {"deployment_branch", "vulcan_config", "devkit_runner"}
    assert {name for name, value in call.items() if value["required"] == "false"} == optional
    assert all(call[name]["default"] == "" for name in optional)
    assert all("default" not in value for name, value in call.items() if name not in optional)


def test_benchmark_inputs_outlive_long_speculative_runs():
    jobs = load_workflow()["jobs"]
    stage = jobs["stage"]
    retained = step(stage["steps"], "Retain the staged build metadata for the DevKit and publication")
    assert int(retained["with"]["retention-days"]) * 24 * 60 > 4320
    assert jobs["benchmark"]["timeout-minutes"] == "${{ fromJSON(needs.stage.outputs.timeout_minutes) }}"
    assert stage["outputs"]["timeout_minutes"] == "${{ steps.settings.outputs.timeout_minutes }}"
    assert jobs["benchmark"]["env"]["BENCHMARK_BACKEND"] == "${{ needs.stage.outputs.benchmark_backend }}"
    settings = step(stage["steps"], "Resolve benchmark job settings")
    reader = step(stage["steps"], "Configure authenticated production registry reads")
    assert stage["steps"].index(settings) < stage["steps"].index(reader)


def test_only_a_manual_dispatch_resolves_the_config_and_runner() -> None:
    jobs = load_workflow()["jobs"]
    stage = jobs["stage"]

    # The orchestrator passes the config and the DevKit runner it resolved once for the run.
    assert jobs["resolve-vulcan-config"]["if"] == "inputs.vulcan_config == ''"
    assert jobs["resolve-benchmark-runner"]["if"] == "inputs.devkit_runner == '' && !inputs.promoted_missing"
    assert stage["needs"] == ["resolve-vulcan-config", "resolve-benchmark-runner"]
    # Without a status function a skipped resolver would skip the stage too.
    assert stage["if"].startswith("!cancelled() && !contains(needs.*.result, 'failure') &&")
    assert "!inputs.promoted_missing" in stage["if"]
    assert stage["outputs"]["vulcan_config"] == (
        "${{ inputs.vulcan_config || toJSON(needs.resolve-vulcan-config.outputs) }}"
    )
    assert stage["outputs"]["devkit_runner"] == (
        "${{ inputs.devkit_runner || toJSON(needs.resolve-benchmark-runner.outputs) }}"
    )


def test_promoted_batch_is_manual_and_does_not_claim_a_devkit():
    job = load_workflow()["jobs"]["promoted"]
    assert job["if"] == "github.event_name == 'workflow_dispatch' && inputs.promoted_missing"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {"contents": "read", "actions": "write", "id-token": "write"}
    dispatch = step(job["steps"], "Queue missing measurements through this workflow")
    assert dispatch["run"] == "scripts/benchmark_promoted_models.py"
    assert dispatch["env"]["GH_TOKEN"] == "${{ github.token }}"


@pytest.mark.parametrize("case", ["valid", "missing_model", "wrong_branch"])
def test_single_model_dispatch_validates_fields_before_reading_the_registry(tmp_path, case):
    source = step(load_workflow()["jobs"]["stage"]["steps"], "Validate the benchmark request")
    environment = {**os.environ, "SOURCE_SHA": "a" * 40,
                   "SOURCE_BRANCH": "main" if case == "wrong_branch" else "develop", "GITHUB_REF_NAME": "develop",
                   "MODEL_ID": "" if case == "missing_model" else "resnet_50", "BUILD_PROFILE_ID": "w8a8",
                   "COMPILE_INVOCATION_ID": "w8a8", "STAGES": "performance", "GITHUB_OUTPUT": str(tmp_path / "output")}
    result = subprocess.run(["bash", "-c", source["run"]], env=environment, capture_output=True, text=True)
    assert (result.returncode == 0) == (case == "valid")


@pytest.mark.parametrize("missing", [False, True])
def test_promoted_batch_dispatches_single_profile_runs_and_reports_skips(monkeypatch, tmp_path, missing):
    import runpy

    queued = []
    build = {"model_id": "hrnet_w32", "build_profile_id": "w8a8", "compile_invocation_id": "w8a8",
             "stages": ["accuracy"]}
    plan = {"source_branch": "main", "source_sha": "a" * 40, "include": [build] if missing else [],
            "skipped": [{"model_id": "measured", "build_profile_id": "w8a8", "reason": "already measured"}]}

    def read(command, **kwargs):
        assert command[:2] == ["model-registry", "plan-promoted-benchmarks"]
        return json.dumps(plan)

    def dispatch(command, **kwargs):
        assert command[:4] == ["gh", "api", "--method", "POST"]
        queued.append(json.loads(kwargs["input"]))

    environment = {"REGISTRY_API_URL": "https://registry.example.invalid", "GITHUB_REF_NAME": "main",
                   "STAGES": "performance,accuracy", "VULCAN_ENV": "production",
                   "GITHUB_REPOSITORY": "sima-neat/models", "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(subprocess, "check_output", read)
    monkeypatch.setattr(subprocess, "run", dispatch)
    runpy.run_path(str(ROOT / "scripts/benchmark_promoted_models.py"), run_name="__main__")
    assert len(queued) == int(missing)
    if missing:
        assert queued[0] == {"ref": "main", "inputs": {"environment": "production", "source_sha": "a" * 40,
            "source_branch": "main", "model_id": "hrnet_w32", "build_profile_id": "w8a8",
            "compile_invocation_id": "w8a8", "stages": "accuracy"}}
        assert "promoted_missing" not in queued[0]["inputs"]  # Child dispatches use the single-model default.
    summary = (tmp_path / "summary").read_text()
    assert "already measured" in summary and f"Queued {int(missing)}" in summary


def test_benchmark_stages_only_the_build_metadata() -> None:
    workflow = load_workflow()
    stage = workflow["jobs"]["stage"]
    text = BENCHMARK_WORKFLOW.read_text(encoding="utf-8")

    assert stage["environment"] == ENVIRONMENT
    # The stage reads no Actions artifact, so it needs no Actions permission.
    assert stage["permissions"] == {"contents": "read", "id-token": "write"}
    assert stage["outputs"]["source_branch"] == "${{ steps.source.outputs.branch }}"
    assert "inputs.operation" not in stage["if"]
    validate = step(stage["steps"], "Validate the benchmark request")
    assert '[[ "$SOURCE_BRANCH" == "$GITHUB_REF_NAME" ]]' in validate["run"]
    assert '[[ -n "$STAGES" ]]' in validate["run"]
    assert validate["env"]["STAGES"] == "${{ inputs.stages }}"
    assert '[[ "$stage" == performance || "$stage" == accuracy ]]' in validate["run"]
    assert '[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]' in validate["run"]
    assert 'git check-ref-format --branch "$SOURCE_BRANCH"' in validate["run"]
    assert "COMPONENT_IDS" not in text
    # Every build comes from the registry by compile run; no model file passes through an Actions artifact.
    for retired in ("source_run_id", "SOURCE_RUN_ID", "compiled-model-"):
        assert retired not in text, retired
    assert all("gh " not in item.get("run", "") for item in stage["steps"])
    assert all(item.get("uses") != "actions/download-artifact@v7" for item in stage["steps"])
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
    restore = step(stage["steps"], "Stage the current immutable compilation from the registry")
    assert "if" not in restore
    assert restore["run"] == "scripts/stage_benchmark_inputs.sh _source _compiled"
    script = (ROOT / "scripts/stage_benchmark_inputs.sh").read_text()
    assert 'model-registry stage-registry-build "${args[@]}"\n' in script
    assert 'model-registry stage-registry-build "${args[@]}" --benchmark-baseline' in script
    # A benchmark runs under the profile's published package; the stage never registers a build itself.
    package = step(stage["steps"], "Require the profile's published package")
    assert "if" not in package
    assert package["run"] == (
        'test -f "_compiled/$MODEL_ID/$BUILD_PROFILE_ID/packages/default/registry-publication.json"'
    )
    assert all("model-registry publish" not in item.get("run", "") for item in stage["steps"])
    credentials = step(stage["steps"], "Configure authenticated production registry reads")
    assert credentials["with"]["output-env-credentials"] == "false"
    assert credentials["with"]["output-credentials"] == "true"
    assert stage["steps"].index(credentials) < stage["steps"].index(restore)
    upload = next(item for item in stage["steps"] if item.get("uses") == "actions/upload-artifact@v7")
    assert upload["name"] == "Retain the staged build metadata for the DevKit and publication"
    assert upload["with"]["name"] == (
        "benchmark-input-${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"
    )
    steps = stage["steps"]
    assert steps.index(restore) < steps.index(package) < steps.index(upload)


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
    # Both stages read their datasets from the board's dataset store, which outlives the job.
    assert "DATASET_DIR" not in benchmark["env"]
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
    download = step(steps, "Download the staged build metadata")
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
    # The script picks its benchmark input from the model's first accuracy dataset, from the board's dataset store.
    assert 'unpack_benchmark_datasets.sh" "$MODEL_DIR" "$BUILD_PROFILE_ID" --first)' in script
    assert 'dataset=(--dataset "$(json_field name <<< "$resolution")=$(json_field directory <<< "$resolution")")' in (
        script
    )
    assert script.index("unpack_benchmark_datasets.sh") < script.index('scripts/performance.py"')
    assert "model_registry model-component-artifacts" in script
    assert script.index('scripts/performance.py"') < script.index("model_registry check-report")


@pytest.mark.parametrize("extra_wheels", [True, False])
def test_performance_dependencies_are_isolated_and_cleaned(tmp_path: Path, extra_wheels: bool) -> None:
    scripts = tmp_path / "repository/models/fixture/scripts"
    scripts.mkdir(parents=True)
    if extra_wheels:
        (scripts / "requirements-performance.txt").write_text("benchmark_wheel==1.0\n")
    (scripts / "performance.py").write_text(
        "import argparse, json, os\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--mpk'); p.add_argument('--output'); a=p.parse_args()\n"
        "if os.environ['EXTRA_WHEELS']=='true':\n"
        "    from benchmark_wheel import value\n"
        "else: value=0\n"
        "Path(a.output, 'model-report.json').write_text(json.dumps({'value':value}))\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3]==['-m','model_registry']:\n"
        "    if sys.argv[3]=='model-component-artifacts': print('model=fixture_mpk.tar.gz')\n"
        "elif sys.argv[1:2]==['-c'] and 'from model_registry.model import load_model' in sys.argv[2]:\n"
        "    print('core')\n"
        "elif sys.argv[1:3]==['-m','pip']:\n"
        "    Path(os.environ['PIP_CALL']).write_text(json.dumps(sys.argv[1:]))\n"
        "    site=Path(sys.argv[sys.argv.index('--target')+1]); site.mkdir(parents=True)\n"
        "    (site/'benchmark_wheel.py').write_text('value=42\\n')\n"
        "else: os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    cat = binaries / "cat"
    cat.write_text(
        '#!/bin/sh\ncase "$1" in */5000000.mla/cur_freq) echo 1000000000;; '
        '*/policy0/scaling_cur_freq) echo 1400000;; *) exec /bin/cat "$@";; esac\n'
    )
    cat.chmod(0o755)
    work, output, pip_call = (tmp_path / name for name in ("scratch", "report", "pip.json"))
    env = {**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
           "REGISTRY_PYTHON": str(python), "PYNEAT_PYTHON": str(python),
           "WORK_DIR": str(work), "DATASET_CACHE": str(tmp_path / "cache"),
           "DATASET_LOCK": str(tmp_path / "prepare.lock"), "PIP_CALL": str(pip_call),
           "EXTRA_WHEELS": str(extra_wheels).lower()}
    subprocess.run(
        [ROOT / "scripts/run_model_performance.sh", scripts.parent, "profile", tmp_path / "compiled", output],
        env=env, check=True, capture_output=True, text=True,
    )
    assert json.loads((output / "model-report.json").read_text())["value"] == (42 if extra_wheels else 0)
    assert not work.exists()
    assert pip_call.exists() == extra_wheels
    if extra_wheels:
        arguments = json.loads(pip_call.read_text())
        assert "--no-deps" in arguments and "--only-binary=:all:" in arguments
        assert arguments[arguments.index("--target") + 1] == str(work / "site-packages")
        assert arguments[arguments.index("-r") + 1] == str(scripts / "requirements-performance.txt")


@pytest.mark.parametrize("installed", [None, "0.6.0+develop.abcfe372a575", "0.6.0+develop.111111111111", "broken"])
def test_native_setup_reuses_matching_mole_on_nvme(tmp_path, installed):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "install.json"
    expected = "0.6.0+develop.abcfe372a575"
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"version": expected}))
    state = tmp_path / "installed-version"
    if installed is not None:
        state.write_text(installed)
    venv = tmp_path / "nvme/mole-venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text(
        '#!/bin/sh\n[ -f "$MOLE_STATE" ] && [ "$(cat "$MOLE_STATE")" != broken ] || exit 1\n'
        'cat "$MOLE_STATE"\n'
    )
    python.chmod(0o755)
    for name, body in {
        "sima-cli": '''import json, os, sys
from pathlib import Path
if "--json" in sys.argv:
    print("sima-cli is up-to-date")
    print("Environment: elxr")
    print(json.dumps({"metadata_url": os.environ["METADATA_URL"]}))
else:
    Path(os.environ["INSTALL_CALLS"]).write_text(json.dumps({
        "args": sys.argv[1:], "venv": os.environ["SIMA_MOLE_VENV_PATH"],
        "cache": os.environ["PIP_CACHE_DIR"], "tmp": os.environ["TMPDIR"],
    }))
    Path(os.environ["MOLE_STATE"]).write_text(os.environ["EXPECTED_VERSION"])
''',
        "dpkg-query": "pass",
    }.items():
        command = binaries / name
        command.write_text(f"#!{sys.executable}\n{body}\n")
        command.chmod(0o755)
    runtime = {"llima": {"install_target": "llima@develop:abcfe372a575"}}
    env = {**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
           "REGISTRY_PYTHON": sys.executable, "SIMA_MOLE_VENV_PATH": str(venv),
           "PIP_CACHE_DIR": str(tmp_path / "nvme/pip-cache"), "TMPDIR": str(tmp_path / "nvme/tmp"),
           "WORK_DIR": str(tmp_path / "nvme/work"), "MOLE_STATE": str(state),
           "METADATA_URL": metadata.as_uri(), "INSTALL_CALLS": str(calls), "EXPECTED_VERSION": expected}
    env.pop("LLIMA_MOLE_PYTHON", None)
    result = subprocess.run(
        [ROOT / "scripts/install_mole_benchmark_runtime.sh", json.dumps(runtime)],
        env=env, check=True, capture_output=True, text=True,
    )
    assert calls.exists() == (installed != expected)
    if installed == expected:
        assert "Skipping install" in result.stdout
    else:
        assert json.loads(calls.read_text()) == {
            "args": ["neat", "install", "--env", "production", "llima/mole@develop:abcfe372a575",
                     "--install-dir", str(tmp_path / "nvme/work/mole")],
            "venv": str(venv), "cache": env["PIP_CACHE_DIR"], "tmp": env["TMPDIR"],
        }
    assert state.read_text() == expected


def test_benchmarks_run_only_when_requested() -> None:
    benchmark = load_workflow()["jobs"]["benchmark"]
    steps = benchmark["steps"]
    artifact_suffix = "${{ inputs.model_id }}-${{ inputs.build_profile_id }}-${{ inputs.compile_invocation_id }}"

    assert not {"BENCHMARK_PERFORMANCE", "BENCHMARK_ACCURACY"} & set(benchmark["env"])
    performance = step(steps, "Measure performance on DevKit")
    assert performance["if"] == (
        "${{ !cancelled() && steps.build.outcome == 'success' && steps.core.outcome == 'success' && "
        "(steps.dataset-cache.outcome == 'success' || steps.mole.outcome == 'success') && "
        "steps.stages.outputs.performance == 'true' }}"
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
        "${{ !cancelled() && steps.build.outcome == 'success' && steps.core.outcome == 'success' && "
        "steps.dataset-cache.outcome == 'success' && steps.stages.outputs.accuracy == 'true' }}"
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
        "${{ !cancelled() && steps.runtime.outcome == 'success' }}"
    )
    mole = step(benchmark["steps"], "Install selected MoLE")
    assert mole["if"] == (
        "${{ !cancelled() && steps.core.outcome == 'success' && env.BENCHMARK_BACKEND == 'llima' }}"
    )
    assert mole["run"] == 'scripts/install_mole_benchmark_runtime.sh "$BENCHMARK_RUNTIME"'
    assert mole["env"] == {
        "SIMA_MOLE_VENV_PATH": "/media/nvme/llima/mole-venv",
        "PIP_CACHE_DIR": "/media/nvme/llima/pip-cache",
        "TMPDIR": "/media/nvme/llima/tmp",
    }
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

    # The credentialed step runs the cache script, which fills the dataset store through the unpack script.
    cache_script = (ROOT / "scripts" / "cache_accuracy_dataset.sh").read_text(encoding="utf-8")
    unpack_script = (ROOT / "scripts" / "unpack_benchmark_datasets.sh").read_text(encoding="utf-8")
    assert "unpack_benchmark_datasets.sh" in cache_script
    for script in (cache_script, unpack_script):
        for model_code in ("accuracy.py", "performance.py", "compile.py", "pip install"):
            assert model_code not in script


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
    unpack_script = (ROOT / "scripts" / "unpack_benchmark_datasets.sh").read_text(encoding="utf-8")
    assert 'NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"' in cache_script
    assert 'mountpoint -q "$NVME_MOUNT"' in cache_script
    assert "coco" not in cache_script.lower()
    # The cache step fills the board's dataset store through the script both stages read it with.
    assert 'unpack_benchmark_datasets.sh" "$MODEL_DIR" "$BUILD_PROFILE_ID")"' in cache_script
    # That one script owns the store, the archive cache, the board-wide lock, and the share's location.
    for line in (
        'NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"',
        'DATASET_STORE="${DATASET_STORE:-$NVME_MOUNT/datasets/unpacked}"',
        'DATASET_CACHE="${DATASET_CACHE:-$NVME_MOUNT/datasets/artifacts}"',
        'DATASET_LOCK="${DATASET_LOCK:-$NVME_MOUNT/datasets/locks/prepare.lock}"',
        'SHARED_DATASET_ARCHIVES="${SHARED_DATASET_ARCHIVES:-/mnt/dataset-private/current}"',
        'flock --wait "$DATASET_LOCK_SECONDS" 9',
        'timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS"',
    ):
        assert line in unpack_script, line
    for runner in ("cache_accuracy_dataset.sh", "run_model_accuracy.sh", "run_model_performance.sh"):
        text = (ROOT / "scripts" / runner).read_text(encoding="utf-8")
        assert "unpack_benchmark_datasets.sh" in text
        for owned in ("resolve-accuracy-dataset", "devkit-accuracy-datasets", "tar -xzf", "flock", "DATASET_STORE"):
            assert owned not in text, (runner, owned)


def _dataset(directory: Path, name: str, member: str, content: bytes, source: str) -> tuple[dict, dict]:
    """A one-file dataset archive: its devkit-accuracy-datasets pin and its resolve-accuracy-dataset line."""
    path = directory / f"{name}.tgz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    pin = {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
    return pin, {"path": str(path), "source": source}


def _board_paths(tmp_path: Path) -> dict[str, str]:
    """The dataset store, archive cache, and lock under tmp_path, and a share that is not mounted."""
    return {"DATASET_STORE": str(tmp_path / "store"), "DATASET_CACHE": str(tmp_path / "cache"),
            "DATASET_LOCK": str(tmp_path / "locks/prepare.lock"),
            "SHARED_DATASET_ARCHIVES": str(tmp_path / "unmounted/current")}


def _stand_in_registry(tmp_path: Path, pins: list[dict], archives: dict[str, dict]) -> Path:
    """A registry Python whose devkit-accuracy-datasets prints ``pins`` and whose resolve-accuracy-dataset prints
    ``archives[name]``, logging each resolved name to tmp_path/resolved."""
    registry = tmp_path / "registry-python"
    registry.write_text(f"""#!{sys.executable}
import json, os, sys
if sys.argv[1] == '-c':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
command = sys.argv[3]
if command == 'devkit-accuracy-datasets':
    for pin in {pins!r}:
        print(json.dumps(pin))
elif command == 'resolve-accuracy-dataset':
    name = sys.argv[sys.argv.index('--dataset') + 1]
    with open({str(tmp_path / "resolved")!r}, 'a') as log:
        log.write(name + '\\n')
    print(json.dumps({archives!r}[name]))
else:
    raise AssertionError(sys.argv)
""")
    registry.chmod(0o755)
    return registry


def _unpack(tmp_path: Path, pins: list[dict], archives: dict[str, dict], *flags: str) -> list[dict]:
    """Run unpack_benchmark_datasets.sh with a stand-in registry Python; return its JSON lines."""
    env = {**os.environ, "REGISTRY_PYTHON": str(_stand_in_registry(tmp_path, pins, archives)), **_board_paths(tmp_path)}
    result = subprocess.run(
        [ROOT / "scripts/unpack_benchmark_datasets.sh", ROOT / "models/resnet_50", "w8a8", *flags],
        env=env, check=True, capture_output=True, text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def _resolved(tmp_path: Path) -> list[str]:
    log = tmp_path / "resolved"
    return log.read_text().splitlines() if log.exists() else []


def test_a_share_that_gives_no_listing_is_left_out(tmp_path: Path) -> None:
    version = subprocess.run(["timeout", "--version"], capture_output=True, text=True, check=False).stdout
    if "coreutils" not in version:
        pytest.skip("needs coreutils timeout, which ends the listing")
    pin, archive = _dataset(tmp_path, "first", "a.jpg", b"one", "registry")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # A stale share blocks the listing; this one blocks for far longer than the test allows.
    (binaries / "ls").write_text("#!/bin/sh\nsleep 60\n")
    (binaries / "ls").chmod(0o755)
    env = {**os.environ, "REGISTRY_PYTHON": str(_stand_in_registry(tmp_path, [pin], {"first": archive})),
           **_board_paths(tmp_path), "SHARE_LISTING_SECONDS": "1",
           "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run([ROOT / "scripts/unpack_benchmark_datasets.sh", ROOT / "models/resnet_50", "w8a8"],
                            env=env, check=True, capture_output=True, text=True, timeout=30)

    assert "The shared dataset archive share gave no listing in 1 s." in result.stderr
    assert json.loads(result.stdout)["source"] == "registry"
    assert not (tmp_path / "store/.share-copy").exists()


def test_the_board_unpacks_each_archive_once_for_every_later_job(tmp_path: Path) -> None:
    first, first_archive = _dataset(tmp_path, "first", "val/a.jpg", b"one", "share")
    second, second_archive = _dataset(tmp_path, "second", "b.jpg", b"two", "registry")
    archives = {"first": first_archive, "second": second_archive}
    store = tmp_path / "store"

    # The performance stage reads only the first dataset, unpacked looking through its single top-level folder.
    first_directory = str(store / first["sha256"] / "dataset/val")
    assert _unpack(tmp_path, [first, second], archives, "--first") == [
        {**first, "directory": first_directory, "source": "share"},
    ]
    # The accuracy stage finds it in the store and unpacks only the dataset performance did not read.
    assert [(item["directory"], item["source"]) for item in _unpack(tmp_path, [first, second], archives)] == [
        (first_directory, "board"), (str(store / second["sha256"] / "dataset"), "registry"),
    ]
    # A later job reads no archive, so it works with the archives gone.
    for archive in archives.values():
        Path(archive["path"]).unlink()
    assert [item["source"] for item in _unpack(tmp_path, [first, second], archives)] == ["board", "board"]
    assert _resolved(tmp_path) == ["first", "second"]
    assert sorted(path.name for path in store.iterdir()) == sorted([first["sha256"], second["sha256"]])


def test_an_interrupted_unpack_is_redone(tmp_path: Path) -> None:
    pin, archive = _dataset(tmp_path, "data", "val/a.jpg", b"one", "registry")
    # An unpack that stopped early left part of the tree, already read-only, and no marker.
    partial = tmp_path / "store" / pin["sha256"] / "dataset/val"
    partial.mkdir(parents=True)
    (partial / "half.jpg").write_bytes(b"o")
    partial.chmod(0o555)

    assert _unpack(tmp_path, [pin], {"data": archive})[0]["source"] == "registry"
    assert sorted(path.name for path in partial.iterdir()) == ["a.jpg"]
    assert (tmp_path / "store" / pin["sha256"] / "complete").is_file()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into read-only folders")
def test_the_unpacked_dataset_is_read_only(tmp_path: Path) -> None:
    pin, archive = _dataset(tmp_path, "data", "val/a.jpg", b"one", "share")
    directory = Path(_unpack(tmp_path, [pin], {"data": archive})[0]["directory"])

    # A model script that writes into its dataset fails, so it cannot change a later job's input.
    with pytest.raises(PermissionError):
        (directory / "written-by-a-model.txt").write_text("")
    with pytest.raises(PermissionError):
        (directory / "a.jpg").write_bytes(b"changed")
    with pytest.raises(PermissionError):
        (directory / "a.jpg").unlink()
    assert (directory / "a.jpg").read_bytes() == b"one"


def test_the_store_removes_copies_unused_for_30_days(tmp_path: Path) -> None:
    store = tmp_path / "store"
    pin, archive = _dataset(tmp_path, "data", "val/a.jpg", b"one", "registry")

    def stored_copy(sha256: str, unused_days: int | None) -> None:
        tree = store / sha256 / "dataset"
        tree.mkdir(parents=True)
        (tree / "a.jpg").write_bytes(b"one")
        tree.chmod(0o555)
        if unused_days is not None:
            marker = store / sha256 / "complete"
            marker.touch()
            used = time.time() - unused_days * 24 * 3600
            os.utime(marker, (used, used))

    stored_copy("a" * 64, 31)
    stored_copy("b" * 64, 29)
    stored_copy("c" * 64, None)
    stored_copy(pin["sha256"], 31)

    # The job uses its copy before the removal runs, so that copy stays.
    assert _unpack(tmp_path, [pin], {"data": archive})[0]["source"] == "board"
    assert sorted(path.name for path in store.iterdir()) == sorted(["b" * 64, pin["sha256"]])
    assert time.time() - (store / pin["sha256"] / "complete").stat().st_mtime < 600
    assert _resolved(tmp_path) == []


def test_a_profile_without_accuracy_datasets_unpacks_nothing(tmp_path: Path) -> None:
    assert _unpack(tmp_path, [], {}) == []
    assert not (tmp_path / "store").exists()


def test_the_job_summary_names_where_each_dataset_came_from(tmp_path: Path) -> None:
    first, first_archive = _dataset(tmp_path, "first", "val/a.jpg", b"one", "share")
    second, second_archive = _dataset(tmp_path, "second", "b.jpg", b"two", "registry")
    registry = _stand_in_registry(tmp_path, [first, second], {"first": first_archive, "second": second_archive})
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "mountpoint").write_text("#!/bin/sh\nexit 0\n")
    (binaries / "mountpoint").chmod(0o755)
    summary = tmp_path / "summary.md"
    env = {**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}", "REGISTRY_PYTHON": str(registry),
           "NVME_MOUNT": str(tmp_path), "GITHUB_STEP_SUMMARY": str(summary), **_board_paths(tmp_path)}
    store = tmp_path / "store"

    def rows(source: dict[str, str]) -> list[str]:
        return [
            f"| first | `{first['sha256']}` | {first['size_bytes']} | {source['first']} | "
            f"`{store / first['sha256'] / 'dataset/val'}` |",
            f"| second | `{second['sha256']}` | {second['size_bytes']} | {source['second']} | "
            f"`{store / second['sha256'] / 'dataset'}` |",
        ]

    def cache() -> list[str]:
        subprocess.run([ROOT / "scripts/cache_accuracy_dataset.sh", ROOT / "models/resnet_50", "w8a8"], env=env,
                       check=True, capture_output=True, text=True)
        return summary.read_text(encoding="utf-8").splitlines()

    assert cache() == [
        "### Accuracy datasets", "", "| Dataset | SHA-256 | Bytes | Source | Directory |", "|---|---|---|---|---|",
        *rows({"first": "shared dataset archive share", "second": "Dataset Registry"}),
    ]
    # The next job finds both datasets on the board.
    assert cache()[-2:] == rows({"first": "board's unpacked copy", "second": "board's unpacked copy"})


@pytest.mark.parametrize("on_share", [True, False], ids=["from_the_share", "from_the_dataset_registry"])
def test_the_store_fills_from_the_share_or_else_the_dataset_registry(tmp_path: Path, on_share: bool) -> None:
    """Run the unpack with the real model-registry commands and a stand-in dataset-registry on PATH."""
    archive = tmp_path / "imagenet-val.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("val2012/image.JPEG")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"img"))
    digest, size = hashlib.sha256(archive.read_bytes()).hexdigest(), archive.stat().st_size
    model_dir = repository_without_w16a16_accuracy(tmp_path) / "models" / "resnet_50"
    model_yaml = model_dir / "model.yaml"
    model = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
    model["accuracy"]["datasets"] = [{"id": "imagenet-2012", "variant": "val", "sha256": digest, "size_bytes": size}]
    model_yaml.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    board = _board_paths(tmp_path)
    if on_share:
        share = Path(board["SHARED_DATASET_ARCHIVES"])
        (share / "imagenet").mkdir(parents=True)
        shutil.copyfile(archive, share / "imagenet/val.tgz")
        (share / "_release.json").write_text(json.dumps({"archives": [
            {"key": "datasets/artifactory/imagenet/val.tgz", "sha256": digest, "size": size},
        ]}))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "dataset-registry").write_text(
        f"#!/bin/sh\necho \"$*\" >> '{tmp_path / 'registry-calls'}'\necho '{json.dumps({'path': str(archive)})}'\n"
    )
    (binaries / "dataset-registry").chmod(0o755)
    env = {**os.environ, **board, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
           "REGISTRY_PYTHON": sys.executable,
           "PYTHONPATH": os.pathsep.join([str(ROOT / "tools"), *filter(None, [os.environ.get("PYTHONPATH")])])}

    def unpack() -> tuple[dict, str]:
        result = subprocess.run([ROOT / "scripts/unpack_benchmark_datasets.sh", model_dir, "w8a8"], env=env,
                                check=True, capture_output=True, text=True)
        [line] = result.stdout.splitlines()
        return json.loads(line), result.stderr

    resolution, messages = unpack()
    directory = tmp_path / "store" / digest / "dataset" / "val2012"
    assert resolution == {"name": "imagenet-2012-val", "sha256": digest, "size_bytes": size,
                          "directory": str(directory), "source": "share" if on_share else "registry"}
    assert (directory / "image.JPEG").read_bytes() == b"img"
    # A machine without the share says nothing about it and resolves through the Dataset Registry.
    assert "shared dataset archive share" not in messages
    calls = tmp_path / "registry-calls"
    assert calls.exists() != on_share
    if not on_share:
        assert calls.read_text().split()[:4] == ["resolve", "imagenet-2012", "--variant", "val"]
    # The next job takes the board's copy and reads neither source.
    archive.unlink()
    shutil.rmtree(tmp_path / "unmounted", ignore_errors=True)
    calls.unlink(missing_ok=True)
    assert unpack()[0]["source"] == "board"
    assert not calls.exists()


def test_the_devkit_downloads_the_build_files_without_credentials() -> None:
    steps = load_workflow()["jobs"]["benchmark"]["steps"]
    names = [item.get("name") for item in steps]
    download = step(steps, "Download the build files from the registry")

    assert download["id"] == "build"
    assert download["run"] == (
        '"$REGISTRY_PYTHON" -m model_registry download-build _compiled --model "$MODEL_ID" '
        '--build-profile "$BUILD_PROFILE_ID"'
    )
    assert download["if"] == (
        "steps.stages.outputs.performance == 'true' || steps.stages.outputs.accuracy == 'true'"
    )
    # The public registry serves the files, so the step takes no credentials.
    assert "env" not in download
    order = [names.index(name) for name in (
        "Download the staged build metadata",
        "Install registry tooling in an isolated environment",
        "Select the requested stages the profile declares",
        "Download the build files from the registry",
        "Measure performance on DevKit",
        "Measure accuracy on DevKit",
    )]
    assert order == sorted(order)
    for name in ("Measure performance on DevKit", "Measure accuracy on DevKit"):
        assert "steps.build.outcome == 'success'" in step(steps, name)["if"], name


def test_the_devkit_removes_the_build_files_however_the_job_ends_and_keeps_the_dataset_store(
    tmp_path: Path,
) -> None:
    benchmark = load_workflow()["jobs"]["benchmark"]
    cleanup = benchmark["steps"][-1]

    assert cleanup["name"] == "Remove the build files and scratch folder from the DevKit"
    assert cleanup["if"] == "always()"
    assert benchmark["env"]["WORK_DIR"].startswith("/media/nvme/neat-tests/")
    work = tmp_path / "scratch"
    for path in (tmp_path / "_compiled/resnet_50/w8a8/model", work / "site-packages"):
        path.mkdir(parents=True)
    subprocess.run(["bash", "-c", cleanup["run"]], cwd=tmp_path, check=True, env={**os.environ, "WORK_DIR": str(work)})
    assert sorted(path.name for path in tmp_path.iterdir()) == []
    # Only the unpack script removes a copy from the board's dataset store, after 30 days without use.
    workflow = BENCHMARK_WORKFLOW.read_text(encoding="utf-8")
    assert "datasets/unpacked" not in workflow and "DATASET_STORE" not in workflow


def test_benchmark_runs_only_on_its_source_branch() -> None:
    workflow = load_workflow()

    for name in ("benchmark", "publish"):
        condition = workflow["jobs"][name]["if"]
        assert "inputs.source_branch == github.ref_name" in condition, name
        assert "refs/heads/develop" not in condition, name
    validate = step(workflow["jobs"]["stage"]["steps"], "Validate the benchmark request")
    assert '[[ "$SOURCE_BRANCH" == "$GITHUB_REF_NAME" ]]' in validate["run"]


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
    registered = step(steps, "Download the receipts of the build the stages measured")
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


@pytest.mark.parametrize("relative,backend,rejected", [
    ("tools/model_registry/report_publication.py", "core", True),
    ("tools/model_registry/benchmark_declarations.py", "core", True),
    ("tools/model_registry/model.py", "core", True),
    ("tools/model_report/generation_benchmark.py", "core", False),
    ("tools/model_report/generation_benchmark.py", "llima", True),
])
def test_board_runs_require_the_measured_revisions_tooling(tmp_path, relative, backend, rejected) -> None:
    stage = load_workflow()["jobs"]["stage"]
    guard = step(stage["steps"], "Require the measured revision's board-stage tooling")
    assert guard["run"] == 'scripts/require_source_board_tooling.sh _source "${{ steps.settings.outputs.backend }}"'
    assert stage["steps"].index(step(stage["steps"], "Resolve benchmark job settings")) < stage["steps"].index(guard)
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
            [ROOT / "scripts" / "require_source_board_tooling.sh", source, backend],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )

    assert check().returncode == 0
    changed = source / relative
    changed.write_text(changed.read_text(encoding="utf-8") + "# newer\n", encoding="utf-8")
    result = check()
    assert (result.returncode != 0) == rejected
    if rejected:
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
    # Every benchmark stages its build from the registry, so every production benchmark signs its reads.
    assert evaluate(reader['if'], resolved) == (expected == 'production')
    restore = step(jobs['stage']['steps'],
                   'Stage the current immutable compilation from the registry')
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


def accuracy_runner(tmp_path: Path, *, supporting_files: bool = False, samples: int | None = None) -> list[str]:
    """Run run_model_accuracy.sh on clip_vit_b_16/w8a16 with stand-in tools; return accuracy.py's arguments."""
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
if command == 'devkit-accuracy-datasets':
    print(json.dumps({{"name": "imagenet-2012-val", "sha256": "a" * 64, "size_bytes": 1,
                      **({{"samples": {samples!r}}} if {samples!r} else {{}})}}))
elif command == 'resolve-accuracy-dataset':
    print(json.dumps({{"path": {str(dataset)!r}, "source": "registry"}}))
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
        "PYNEAT_PYTHON": str(board), "WORK_DIR": str(tmp_path / "scratch"), **_board_paths(tmp_path),
    }
    run = subprocess.run(
        [ROOT / "scripts/run_model_accuracy.sh", ROOT / "models/clip_vit_b_16", "w8a16", compiled, tmp_path / "report"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    assert not (tmp_path / "scratch").exists()
    return json.loads(captured.read_text())


@pytest.mark.parametrize("supporting_files", [False, True])
def test_accuracy_runner_passes_only_verified_local_resources(tmp_path, supporting_files):
    arguments = accuracy_runner(tmp_path, supporting_files=supporting_files)
    profile = tmp_path / "compiled with spaces/clip_vit_b_16/w8a16"
    assert ("--resources" in arguments) == supporting_files
    if supporting_files:
        assert arguments[arguments.index("--resources") + 1] == str(profile)
    assert arguments.count("--mpk") == 2
    # From the board's dataset store, looking through the archive's single top-level directory. The store outlives
    # the runner's scratch folder.
    dataset = tmp_path / "store" / ("a" * 64) / "dataset/val2012"
    assert arguments[arguments.index("--dataset") + 1] == f"imagenet-2012-val={dataset}"
    assert (dataset / "image.JPEG").is_file()


@pytest.mark.parametrize("samples", [None, 500])
def test_accuracy_runner_caps_samples_only_when_model_yaml_does(tmp_path, samples):
    arguments = accuracy_runner(tmp_path, samples=samples)
    # accuracy.samples reaches accuracy.py as --limit; without it the script evaluates every sample.
    assert ("--limit" in arguments) == (samples is not None)
    if samples is not None:
        assert arguments[arguments.index("--limit") + 1] == str(samples)
