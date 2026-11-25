from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "vulcan-model-orchestrator.yml"
BUILD_WORKFLOW = WORKFLOWS / "vulcan-model-compile.yml"
BENCHMARK_WORKFLOW = WORKFLOWS / "vulcan-model-benchmarking.yml"


def load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def step(job: dict, name: str) -> dict:
    return next(item for item in job["steps"] if item.get("name") == name)


def step_index(job: dict, name: str) -> int:
    return next(index for index, item in enumerate(job["steps"]) if item.get("name") == name)


def evaluate(condition: str, values: dict[str, object]) -> object:
    """Evaluate a GitHub Actions expression after substituting every context reference it uses."""
    expression = " ".join(condition.splitlines())
    for name in sorted(values, key=len, reverse=True):
        expression = expression.replace(name, repr(values[name]))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return eval(expression, {"__builtins__": {}, "fromJSON": json.loads, "format": str.format}, {})


LOCAL_BENCHMARK_WORKFLOW = "./.github/workflows/vulcan-model-benchmarking.yml"
PERMISSION_LEVELS = {"none": 0, "read": 1, "write": 2}
BUILD_NAME = "${{ matrix.model_id }}/${{ matrix.compile_invocation_id }} (${{ matrix.build_profile_id }})"
# What a job that needs both resolvers passes the benchmark workflow.
RESOLVED_JSON = {
    "vulcan_config": "${{ toJSON(needs.resolve-vulcan-config.outputs) }}",
    "devkit_runner": "${{ toJSON(needs.resolve-benchmark-runner.outputs) }}",
}


# Every job that calls the benchmark workflow, keyed by workflow and job ID.
BENCHMARK_CALLERS = {
    "compile.benchmark": (BUILD_WORKFLOW, "benchmark"),
    "orchestrator.benchmark": (WORKFLOW, "benchmark"),
    "orchestrator.benchmark-without-compiles": (WORKFLOW, "benchmark-without-compiles"),
}


def benchmark_caller(caller: str) -> dict:
    """Return the job that calls the benchmark workflow under the caller's key."""
    workflow, job_id = BENCHMARK_CALLERS[caller]
    return load(workflow)["jobs"][job_id]


def excess_permissions(caller: dict, called: Path) -> list[str]:
    """Return each permission a called workflow's job requests above what its caller job grants."""
    workflow = load(called)
    granted = caller["permissions"]
    excess = []
    for name, job in workflow["jobs"].items():
        requested = job.get("permissions", workflow.get("permissions", {}))
        for scope, level in requested.items():
            if PERMISSION_LEVELS[level] > PERMISSION_LEVELS[granted.get(scope, "none")]:
                excess.append(f"{name}: {scope}: {level}")
    return excess


def jq_program(script: str, prefix: str) -> str:
    """Return the single-quoted jq program that follows ``prefix`` in a shell script."""
    return re.search(re.escape(prefix) + r"\s*'(.*?)'", script, re.DOTALL).group(1)


def run_jq(program: str, document: object) -> object:
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not installed")
    output = subprocess.run(
        [jq, "-c", program], input=json.dumps(document), capture_output=True, text=True, check=True
    ).stdout
    return json.loads(output)


# Triggers, inputs and concurrency


def test_every_push_runs_so_every_branch_revision_gets_a_catalog() -> None:
    on = load(WORKFLOW)["on"]

    # A push that changes no compile input still publishes its catalog; its fingerprints select no work.
    assert "push" in on and not on["push"]


def test_build_environment_is_owned_by_vulcan_env() -> None:
    inputs = load(WORKFLOW)["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"models", "profiles", "components", "base_branch", "force_rebuild_all"}
    assert inputs["force_rebuild_all"]["type"] == "boolean"


@pytest.mark.parametrize("environment", ["staging", "production", "", "invalid"])
def test_environment_selection_and_runner_mapping(tmp_path, environment) -> None:
    workflow = load(WORKFLOW)
    resolve = step(workflow["jobs"]["select"], "Resolve Vulcan environment")
    assert resolve["env"]["REQUESTED_ENVIRONMENT"] == "${{ vars.VULCAN_ENV }}"
    assert "${{ vars.VULCAN_ENV }}" in workflow["run-name"]
    assert "${{ vars.VULCAN_ENV }}" in workflow["concurrency"]["group"]
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-c", resolve["run"]],
        env={**os.environ, "REQUESTED_ENVIRONMENT": environment, "GITHUB_OUTPUT": str(output)},
        capture_output=True, text=True,
    )
    if environment not in {"staging", "production"}:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        runner = "vulcan-stg" if environment == "staging" else "vulcan-production"
        assert output.read_text().splitlines() == [
            f"vulcan_environment={environment}", f"runner_environment={runner}",
        ]


@pytest.mark.parametrize(("event", "ref", "cancels"), [
    ("push", "refs/heads/feat/new-model", True),
    # A merge waits for the running run of develop or main instead of cancelling its compiles.
    ("push", "refs/heads/develop", False),
    ("push", "refs/heads/main", False),
    ("workflow_dispatch", "refs/heads/feat/new-model", False),
    ("workflow_dispatch", "refs/heads/develop", False),
])
def test_feature_pushes_supersede_runs_while_develop_main_and_dispatches_run_to_completion(
    event: str, ref: str, cancels: bool
) -> None:
    concurrency = load(WORKFLOW)["concurrency"]

    assert evaluate(concurrency["cancel-in-progress"].strip()[3:-3], {
        "github.event_name": event, "github.ref": ref,
    }) is cancels
    # Only dispatches get a per-run group, so they never cancel or replace one another.
    assert "github.event_name == 'workflow_dispatch' && format('-{0}', github.run_id)" in concurrency["group"]
    assert "github.ref" in concurrency["group"]
    assert "vars.VULCAN_ENV" in concurrency["group"]
    # GitHub keeps one waiting run per group; a longer queue would compile revisions a newer push replaces.
    assert "queue" not in concurrency


# Job graph and pull-request isolation


def test_compile_workflow_has_independent_build_stages() -> None:
    jobs = load(WORKFLOW)["jobs"]

    assert set(jobs) == {
        "select",
        "portal",
        "resolve-vulcan-config",
        "resolve-benchmark-runner",
        "publish-model-browser",
        "resolve-staging-browser-config",
        "plan",
        "authorize-runner-classes",
        "initialize-catalog",
        "compile",
        "publish-catalog",
        "publish-candidates",
        "benchmark",
        "benchmark-without-compiles",
        "publish-selection",
        "report-partial-publication",
    }


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
@pytest.mark.parametrize("branch, allowed", [("develop", True), ("main", True), ("feature/example", False)])
def test_frontend_staging_publication_is_independent_of_model_environment(environment, event, branch, allowed):
    jobs = load(WORKFLOW)["jobs"]
    values = {
        "github.ref": f"refs/heads/{branch}",
        "github.event_name": event,
        "vars.VULCAN_ENV": environment,
        "needs.select.outputs.vulcan_environment": environment,
    }
    for name in ("resolve-staging-browser-config", "publish-model-browser"):
        assert evaluate(jobs[name]["if"], values) is allowed


def test_json_catalog_and_package_publication_are_retired() -> None:
    workflow = load(WORKFLOW)
    obsolete_jobs = {"package", "publish", "publish-latest-index", "promote-latest", "cleanup-github-artifacts"}
    assert obsolete_jobs.isdisjoint(workflow["jobs"])
    text = WORKFLOW.read_text(encoding="utf-8")
    for obsolete in ("latest.json", "latest.tag", "metadata.json", "models-dist", "previous-model-index"):
        assert obsolete not in text
    assert "vulcan-publish-artifacts" not in text
    assert "vulcan-update-latest-artifacts" not in text


def test_compilation_no_longer_evaluates_accuracy_or_layers_an_exporter() -> None:
    for path in (WORKFLOW, BUILD_WORKFLOW):
        text = path.read_text(encoding="utf-8")
        assert "model_registry evaluate" not in text
        assert "accuracy_samples" not in text
        assert "ACCURACY_SAMPLES" not in text
        assert "python_packages" not in text


def test_workflow_never_schedules_pull_requests_on_vulcan() -> None:
    workflow = load(WORKFLOW)
    jobs = workflow["jobs"]

    # Pull requests run Target Branch Checks; pull-request code never starts this workflow.
    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in jobs.items():
        if "uses" not in job:
            assert job["runs-on"] == "ubuntu-latest", name

    # The self-hosted compiler is reachable only through the build job, and only as a workflow call.
    build_workflow = load(BUILD_WORKFLOW)
    assert list(build_workflow["on"]) == ["workflow_call"]
    assert jobs["compile"]["uses"] == "./.github/workflows/vulcan-model-compile.yml"
    self_hosted = [
        name for name, job in build_workflow["jobs"].items()
        if isinstance(job.get("runs-on"), list) and "self-hosted" in job["runs-on"]
    ]
    assert self_hosted == ["compile"]
    # The DevKits are reachable only through the benchmark workflow, which a pull request never starts.
    assert [name for name, job in jobs.items() if job.get("uses") == LOCAL_BENCHMARK_WORKFLOW] == [
        "benchmark", "benchmark-without-compiles",
    ]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_trusted_events_reach_the_build(event) -> None:
    condition = load(WORKFLOW)["jobs"]["compile"]["if"]
    values = {
        "!cancelled()": True,
        "github.event_name": event,
        "needs.select.outputs.publishable": "true",
        "needs.resolve-vulcan-config.result": "success",
        "needs.resolve-benchmark-runner.result": "success",
        "needs.authorize-runner-classes.result": "success",
        "needs.initialize-catalog.result": "success",
    }
    assert evaluate(condition, values) is True


@pytest.mark.parametrize(("runner_result", "expected"), [
    ("success", True), ("failure", False), ("skipped", False), ("cancelled", False),
])
def test_builds_wait_for_the_shared_benchmark_runner(runner_result, expected) -> None:
    jobs = load(WORKFLOW)["jobs"]
    values = {
        "!cancelled()": True,
        "needs.select.outputs.publishable": "true",
        "needs.resolve-vulcan-config.result": "success",
        "needs.resolve-benchmark-runner.result": runner_result,
        "needs.authorize-runner-classes.result": "success",
        "needs.initialize-catalog.result": "success",
    }

    assert evaluate(jobs["compile"]["if"], values) is expected


# Selection and planning


def test_catalog_taxonomy_is_validated_before_compiler_selection() -> None:
    select = load(WORKFLOW)["jobs"]["select"]
    validation = step_index(select, "Validate every model and the taxonomy")
    compiler = next(index for index, item in enumerate(select["steps"]) if item.get("id") == "compiler")

    assert select["steps"][validation]["run"] == "model-registry validate-models --repository ."
    assert validation < compiler


def test_model_compiler_is_resolved_before_requesting_a_runner() -> None:
    jobs = load(WORKFLOW)["jobs"]
    select = jobs["select"]
    compiler = next(item for item in select["steps"] if item.get("id") == "compiler")

    assert "model-registry resolve-compiler --repository ." in compiler["run"]
    assert select["outputs"]["model_compiler_version"] == "${{ steps.compiler.outputs.version }}"
    assert select["outputs"]["model_compiler_label"] == "${{ steps.compiler.outputs.label }}"
    assert select["outputs"]["model_compiler_identity"] == "${{ steps.compiler.outputs.identity }}"
    plan = step(jobs["plan"], "Compare fingerprints with published runs")
    assert plan["env"]["COMPILER_IDENTITY"] == "${{ needs.select.outputs.model_compiler_identity }}"
    build = jobs["compile"]["with"]
    assert build["model_compiler_version"] == "${{ needs.select.outputs.model_compiler_version }}"
    assert build["model_compiler_label"] == "${{ needs.select.outputs.model_compiler_label }}"

    compile_job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    assert compile_job["env"]["EXPECTED_COMPILER_VERSION"] == "${{ inputs.model_compiler_version }}"
    assert "${{ inputs.model_compiler_label }}" in compile_job["runs-on"]
    verify = step(compile_job, "Verify the pinned Model Compiler environment")
    assert "/etc/vulcan/model-compiler-artifact.json" in verify["run"]
    assert step_index(compile_job, "Verify the pinned Model Compiler environment") < step_index(
        compile_job, "Compile the invocation with its model's compile script"
    )


def test_targeted_profile_or_component_dispatch_skips_catalog_updates() -> None:
    select = load(WORKFLOW)["jobs"]["select"]
    scope = next(item for item in select["steps"] if item.get("id") == "scope")

    assert select["outputs"]["publishable"] == "${{ steps.scope.outputs.publishable }}"
    assert scope["env"] == {
        "REQUESTED_PROFILES": "${{ inputs.profiles }}",
        "REQUESTED_COMPONENTS": "${{ inputs.components }}",
    }
    assert "publishable=false" in scope["run"]


@pytest.mark.parametrize(
    ("manual", "models", "profiles", "expected"),
    [(False, "resnet_50", "modalix_bf16", False), (True, "", "", False),
     (True, "resnet_50", "", True), (True, "resnet_50", "modalix_bf16", True)],
)
def test_only_explicit_manual_model_selection_enables_on_demand_builds(
    manual: bool, models: str, profiles: str, expected: bool,
) -> None:
    plan = step(load(WORKFLOW)["jobs"]["plan"], "Compare fingerprints with published runs")
    selection = plan["run"].split('plan="$(model-registry plan-stages', 1)[0]
    result = subprocess.run(
        ["bash", "-c", selection + '\nprintf "%s\\n" "${scope_args[@]}"'],
        env={**os.environ, "REGISTRY_READ_API_URL": "https://registry.invalid", "MANUAL_BUILD": str(manual).lower(),
             "REQUESTED_MODELS": models, "REQUESTED_PROFILES": profiles, "REQUESTED_COMPONENTS": "",
             "FORCE_REBUILD_ALL": "true"},
        capture_output=True, text=True, check=True,
    )

    assert ("--include-on-demand" in result.stdout.splitlines()) is expected
    assert "--force-rebuild-all" in result.stdout.splitlines()


def test_branches_reuse_builds_from_the_resolved_base_branch_catalogs() -> None:
    jobs = load(WORKFLOW)["jobs"]
    base = next(item for item in jobs["select"]["steps"] if item.get("id") == "base")

    assert "model-registry resolve-base" in base["run"]
    assert ".base_branches | join(\",\")" in base["run"]
    assert base["env"]["PREFERRED_BRANCH"] == "${{ inputs.base_branch }}"
    plan = step(jobs["plan"], "Compare fingerprints with published runs")
    assert plan["env"]["BASE_BRANCHES"] == "${{ needs.select.outputs.base_branches }}"
    assert '--base-branches "$BASE_BRANCHES"' in plan["run"]
    assert jobs["compile"]["with"]["base_branches"] == "${{ needs.select.outputs.base_branches }}"


def test_portal_checks_gate_browser_publication_independently_of_model_processing() -> None:
    jobs = load(WORKFLOW)["jobs"]

    def ancestors(job_id: str) -> set[str]:
        dependencies = jobs[job_id].get("needs", [])
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        return set(dependencies).union(*(ancestors(dependency) for dependency in dependencies))

    assert jobs["portal"]["uses"] == "./.github/workflows/developer-docs.yml"
    assert "portal" in ancestors("publish-model-browser")
    for job_id in (
        "plan", "authorize-runner-classes", "initialize-catalog", "compile", "publish-candidates",
        "publish-catalog", "benchmark", "benchmark-without-compiles", "publish-selection",
        "report-partial-publication",
    ):
        assert "portal" not in ancestors(job_id), job_id
    # Model validation and registry configuration must still succeed before planning.
    assert "if" not in jobs["plan"]
    assert load(WORKFLOWS / "target-branch-checks.yml")["jobs"]["portal"]["uses"] == (
        "./.github/workflows/developer-docs.yml"
    )


def test_plan_compares_fingerprints_with_the_environment_registry() -> None:
    plan_job = load(WORKFLOW)["jobs"]["plan"]
    plan = step(plan_job, "Compare fingerprints with published runs")

    assert plan_job["needs"] == ["select", "resolve-vulcan-config"]
    assert plan_job["environment"] == "${{ needs.select.outputs.vulcan_environment }}"
    assert plan_job["permissions"] == {"contents": "read", "id-token": "write"}
    credentials = step(plan_job, "Configure registry planning credentials")
    assert "if" not in credentials
    assert plan["env"]["VULCAN_ARTIFACT_REGISTRY_API_URL"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_API_URL }}"
    assert plan["env"]["REGISTRY_READ_API_URL"] == (
        "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL || vars.VULCAN_ARTIFACT_REGISTRY_API_URL }}"
    )
    assert "model-registry plan-stages" in plan["run"]
    assert '--read-api-url "$REGISTRY_READ_API_URL"' in plan["run"]
    assert "--offline" not in plan["run"]
    for name in ("REQUESTED_MODELS", "REQUESTED_PROFILES", "REQUESTED_COMPONENTS", "FORCE_REBUILD_ALL"):
        assert name in plan["env"]
    assert "--force-rebuild-all" in plan["run"]
    assert plan_job["outputs"] == {
        "conversion_sources": "${{ steps.plan.outputs.conversion_sources }}",
        "compile_count": "${{ steps.plan.outputs.compile_count }}",
        "compile_matrix": "${{ steps.plan.outputs.compile_matrix }}",
        "benchmark_count": "${{ steps.plan.outputs.benchmark_count }}",
        "benchmark_matrix": "${{ steps.plan.outputs.benchmark_matrix }}",
        "candidate_count": "${{ steps.plan.outputs.candidate_count }}",
        "candidate_matrix": "${{ steps.plan.outputs.candidate_matrix }}",
    }


def test_reused_builds_are_benchmarked_on_modalix_only() -> None:
    plan = step(load(WORKFLOW)["jobs"]["plan"], "Compare fingerprints with published runs")
    program = jq_program(plan["run"], 'benchmark_matrix="$(jq -c')
    document = {
        "compile": {"include": []},
        "benchmark": {"include": [{"model_id": "a", "target": "modalix"}, {"model_id": "b", "target": "davinci"}]},
    }

    assert run_jq(program, document) == {"include": [{"model_id": "a", "target": "modalix"}]}


# Runner authorization and catalog initialization


def test_runner_classes_are_authorized_before_builds_are_queued() -> None:
    jobs = load(WORKFLOW)["jobs"]
    admission = jobs["authorize-runner-classes"]
    build = jobs["compile"]

    assert admission["needs"] == ["select", "plan", "resolve-vulcan-config"]
    assert "needs.plan.outputs.compile_count != '0'" in admission["if"]
    assert admission["permissions"]["id-token"] == "write"
    authorization = next(item for item in admission["steps"] if item.get("id") == "authorization")
    assert "authorize-runner-matrix" in authorization["run"]
    assert authorization["env"]["SELECTED_MATRIX"] == "${{ needs.plan.outputs.compile_matrix }}"
    assert authorization["env"]["VULCAN_CONFIG_S3_URI"] == "${{ needs.resolve-vulcan-config.outputs.config_s3_uri }}"

    assert build["needs"] == [
        "select", "plan", "resolve-vulcan-config", "resolve-benchmark-runner", "authorize-runner-classes",
        "initialize-catalog",
    ]
    assert build["strategy"]["matrix"] == "${{ fromJSON(needs.authorize-runner-classes.outputs.matrix) }}"
    assert build["strategy"]["fail-fast"] == "false"
    assert build["with"]["entry"] == "${{ toJSON(matrix) }}"
    assert build["with"]["vulcan_environment"] == "${{ needs.select.outputs.vulcan_environment }}"
    assert build["with"]["publishable"] == "${{ needs.select.outputs.publishable == 'true' }}"
    assert build["secrets"] == "inherit"
    assert set(build["with"]) == set(load(BUILD_WORKFLOW)["on"]["workflow_call"]["inputs"])
    assert {name: build["with"][name] for name in RESOLVED_JSON} == RESOLVED_JSON


def test_catalog_marks_authorized_builds_pending_before_they_run() -> None:
    initialize = load(WORKFLOW)["jobs"]["initialize-catalog"]
    publish = step(initialize, "Publish the pending invocation plan")

    assert initialize["needs"] == ["select", "plan", "resolve-vulcan-config", "authorize-runner-classes"]
    assert "needs.select.outputs.publishable == 'true'" in initialize["if"]
    assert "needs.authorize-runner-classes.result == 'success'" in initialize["if"]
    assert publish["env"]["SELECTED_MATRIX"] == "${{ needs.authorize-runner-classes.outputs.matrix }}"
    assert "model-registry publish-catalog" in publish["run"]
    assert '--selected-matrix "$matrix_file"' in publish["run"]
    assert publish["env"]["FORCE_REBUILD_ALL"] == "${{ inputs.force_rebuild_all }}"
    assert "force_args+=(--force-rebuild-all)" in publish["run"]
    assert '"${force_args[@]}"' in publish["run"]
    assert '--base-branches "$BASE_BRANCHES"' in publish["run"]


@pytest.mark.parametrize(("branch", "on_demand", "publishable", "expected"), [
    ("feature/llima-models", True, "false", True),
    ("feature/llima-models", False, "false", False),
    ("develop", True, "false", False),
    ("main", True, "false", False),
    ("master", True, "false", False),
    ("develop", False, "true", True),
])
def test_targeted_candidates_publish_feature_requirements_before_compiling(branch, on_demand, publishable, expected):
    condition = load(WORKFLOW)["jobs"]["initialize-catalog"]["if"]
    assert evaluate(condition, {
        "!cancelled()": True,
        "github.ref": f"refs/heads/{branch}",
        "needs.select.outputs.publishable": publishable,
        "needs.resolve-vulcan-config.result": "success",
        "needs.authorize-runner-classes.result": "success",
        "contains(fromJSON(needs.authorize-runner-classes.outputs.matrix).include.*.build_on_demand, true)": on_demand,
    }) is expected


def test_native_candidates_are_not_published_again_by_the_multi_invocation_job():
    plan = step(load(WORKFLOW)["jobs"]["plan"], "Compare fingerprints with published runs")
    program = jq_program(plan["run"], 'candidate_matrix="$(jq -c')
    native = {"model_id": "qwen", "build_profile_id": "w4a16", "build_on_demand": True,
              "runner_script": "scripts/run_llima_compilation.sh"}
    generic = {"model_id": "vision", "build_profile_id": "w8a8", "build_on_demand": True,
               "runner_script": "scripts/run_model_compilation.sh"}
    assert run_jq(program, {"include": [native, generic, generic, {**generic, "build_on_demand": False}]}) == {
        "include": [{"model_id": "vision", "build_profile_id": "w8a8"}],
    }


@pytest.mark.parametrize(
    ("publishable", "initialize_result", "expected"),
    [
        ("false", "skipped", True),
        ("false", "success", True),
        ("false", "failure", False),
        ("false", "cancelled", False),
        ("true", "success", True),
        ("true", "skipped", False),
        ("true", "failure", False),
    ],
)
def test_build_allows_only_intentional_catalog_initialization_skips(
    publishable, initialize_result, expected,
) -> None:
    condition = load(WORKFLOW)["jobs"]["compile"]["if"]
    values = {
        "!cancelled()": True,
        "github.event_name": "push",
        "needs.select.outputs.publishable": publishable,
        "needs.resolve-vulcan-config.result": "success",
        "needs.resolve-benchmark-runner.result": "success",
        "needs.authorize-runner-classes.result": "success",
        "needs.initialize-catalog.result": initialize_result,
    }
    assert evaluate(condition, values) is expected


# Catalog snapshot, reused-build stages and failure report


@pytest.mark.parametrize(
    ("cancelled", "publishable", "compile_count", "initialize_result", "expected"),
    [
        # The snapshot publishes whether or not individual builds failed; build is not in the condition.
        (False, "true", "2", "success", True),
        (False, "true", "0", "skipped", True),
        (False, "true", "2", "skipped", False),
        (False, "false", "2", "skipped", False),
        (True, "true", "2", "success", False),
    ],
)
def test_catalog_snapshot_publishes_even_when_builds_fail(
    cancelled, publishable, compile_count, initialize_result, expected,
) -> None:
    job = load(WORKFLOW)["jobs"]["publish-catalog"]
    values = {
        "!cancelled()": not cancelled,
        "github.event_name": "push",
        "needs.select.outputs.publishable": publishable,
        "needs.resolve-vulcan-config.result": "success",
        "needs.plan.result": "success",
        "needs.plan.outputs.compile_count": compile_count,
        "needs.initialize-catalog.result": initialize_result,
    }

    assert "compile" in job["needs"]
    assert "needs.compile.result" not in job["if"]
    assert evaluate(job["if"], values) is expected


@pytest.mark.parametrize("branch,catalog_result,expected", [
    ("main", "success", True), ("develop", "success", False), ("main", "failure", False),
])
def test_main_publishes_its_selection_after_the_catalog_and_benchmarks(branch, catalog_result, expected) -> None:
    job = load(WORKFLOW)["jobs"]["publish-selection"]
    publish = step(job, "Publish the selection and the platform default")
    values = {"!cancelled()": True, "github.ref_name": branch, "needs.publish-catalog.result": catalog_result}

    assert {"publish-catalog", "benchmark", "benchmark-without-compiles"} <= set(job["needs"])
    assert evaluate(job["if"], values) is expected
    assert job["permissions"]["id-token"] == "write"
    assert "model-registry publish-selection" in publish["run"]
    assert '--git-sha "$GITHUB_SHA"' in publish["run"]


def test_catalog_snapshot_carries_this_revision_without_build_outputs() -> None:
    job = load(WORKFLOW)["jobs"]["publish-catalog"]
    publish = step(job, "Publish and promote the catalog snapshot")

    assert job["permissions"]["id-token"] == "write"
    assert "model-registry publish-catalog" in publish["run"]
    assert '--git-sha "$GITHUB_SHA"' in publish["run"]
    assert '--base-branches "$BASE_BRANCHES"' in publish["run"]
    assert '--read-api-url "$REGISTRY_READ_API_URL"' in publish["run"]
    # Each build published its own outputs; the snapshot neither downloads nor republishes them.
    assert "--compiled-dir" not in publish["run"]
    assert "--package-dir" not in publish["run"]
    assert "--selected-matrix" not in publish["run"]
    assert all(not item.get("uses", "").startswith("actions/download-artifact") for item in job["steps"])


@pytest.mark.parametrize(
    ("event", "environment", "publishable", "catalog_result", "runner_result", "benchmark_count", "expected"),
    [
        ("push", "staging", "true", "success", "success", "1", True),
        ("workflow_dispatch", "staging", "true", "success", "success", "3", True),
        ("push", "production", "true", "success", "success", "1", True),
        # A feature branch publishes a pending catalog for on-demand candidates only.
        ("push", "staging", "false", "success", "success", "1", False),
        ("push", "staging", "true", "failure", "success", "1", False),
        ("push", "staging", "true", "skipped", "success", "1", False),
        ("push", "staging", "true", "success", "failure", "1", False),
        ("push", "staging", "true", "success", "success", "0", False),
    ],
)
def test_reused_builds_start_benchmarks_once_the_pending_catalog_publishes(
    event, environment, publishable, catalog_result, runner_result, benchmark_count, expected,
) -> None:
    job = load(WORKFLOW)["jobs"]["benchmark"]
    values = {
        "!cancelled()": True,
        "github.event_name": event,
        "needs.initialize-catalog.result": catalog_result,
        "needs.resolve-benchmark-runner.result": runner_result,
        "needs.select.outputs.publishable": publishable,
        "needs.select.outputs.vulcan_environment": environment,
        "needs.plan.outputs.benchmark_count": benchmark_count,
    }

    # The pending catalog already links reused builds, so their benchmarks do not wait for compiles.
    assert job["needs"] == ["select", "plan", "resolve-vulcan-config", "resolve-benchmark-runner", "initialize-catalog"]
    # Without a status function GitHub adds success(), which also requires the
    # indirectly needed jobs and skips reused builds when one of them is skipped.
    assert job["if"].startswith("!cancelled() &&")
    assert evaluate(job["if"], values) is expected


@pytest.mark.parametrize(
    ("environment", "compile_count", "catalog_result", "runner_result", "benchmark_count", "expected"),
    [
        ("staging", "0", "success", "success", "1", True),
        # With compiles, the benchmark job already started them after the pending catalog.
        ("staging", "2", "success", "success", "1", False),
        ("production", "0", "success", "success", "1", True),
        ("staging", "0", "failure", "success", "1", False),
        ("staging", "0", "skipped", "success", "1", False),
        ("staging", "0", "success", "failure", "1", False),
        ("staging", "0", "success", "success", "0", False),
    ],
)
def test_reused_builds_start_benchmarks_after_the_snapshot_when_nothing_compiles(
    environment, compile_count, catalog_result, runner_result, benchmark_count, expected,
) -> None:
    job = load(WORKFLOW)["jobs"]["benchmark-without-compiles"]
    values = {
        "!cancelled()": True,
        "needs.plan.outputs.compile_count": compile_count,
        "needs.publish-catalog.result": catalog_result,
        "needs.resolve-benchmark-runner.result": runner_result,
        "needs.select.outputs.vulcan_environment": environment,
        "needs.plan.outputs.benchmark_count": benchmark_count,
    }

    assert job["needs"] == ["select", "plan", "resolve-vulcan-config", "resolve-benchmark-runner", "publish-catalog"]
    assert job["if"].startswith("!cancelled() &&")
    assert evaluate(job["if"], values) is expected


@pytest.mark.parametrize("job_id", ["benchmark", "benchmark-without-compiles"])
def test_reused_builds_are_benchmarked_one_call_per_build(job_id) -> None:
    job = load(WORKFLOW)["jobs"][job_id]

    # GitHub prefixes each benchmark job with this name, so the build's name alone tells the builds apart.
    assert job["name"] == BUILD_NAME
    assert job["uses"] == LOCAL_BENCHMARK_WORKFLOW
    assert "runs-on" not in job
    assert job["permissions"] == {"actions": "write", "contents": "read", "id-token": "write"}
    # One build's failed benchmark does not cancel the others.
    assert job["strategy"] == {"fail-fast": "false", "matrix": "${{ fromJSON(needs.plan.outputs.benchmark_matrix) }}"}
    assert job["secrets"] == "inherit"
    assert job["with"] == {
        "environment": "${{ needs.select.outputs.vulcan_environment }}",
        # Each benchmark stages its build from the registry at this catalog revision.
        "source_sha": "${{ github.sha }}",
        "source_branch": "${{ github.ref_name }}",
        "model_id": "${{ matrix.model_id }}",
        "build_profile_id": "${{ matrix.build_profile_id }}",
        "compile_invocation_id": "${{ matrix.compile_invocation_id }}",
        "stages": "${{ join(matrix.stages, ',') }}",
        **RESOLVED_JSON,
    }


def test_the_run_resolves_one_devkit_runner_for_every_benchmark() -> None:
    jobs, dispatch = load(WORKFLOW)["jobs"], load(BENCHMARK_WORKFLOW)["jobs"]
    resolver, manual = jobs["resolve-benchmark-runner"], dispatch["resolve-benchmark-runner"]

    assert resolver["uses"] == manual["uses"]
    assert jobs["resolve-vulcan-config"]["uses"] == dispatch["resolve-vulcan-config"]["uses"]
    assert "if" not in resolver
    # An orchestrator run and a manual dispatch benchmark a build on the same DevKits.
    assert json.loads(resolver["with"]["runner-labels"]) == json.loads(manual["with"]["runner-labels"])


@pytest.mark.parametrize("caller", BENCHMARK_CALLERS)
def test_every_benchmark_caller_passes_the_json_of_its_resolvers_outputs(caller) -> None:
    workflow, _ = BENCHMARK_CALLERS[caller]
    job = benchmark_caller(caller)
    if workflow == BUILD_WORKFLOW:
        # The build workflow forwards what the orchestrator's compile job passed it.
        assert {name: job["with"][name] for name in RESOLVED_JSON} == {
            name: f"${{{{ inputs.{name} }}}}" for name in RESOLVED_JSON
        }
        workflow, job = WORKFLOW, load(WORKFLOW)["jobs"]["compile"]

    assert {"resolve-vulcan-config", "resolve-benchmark-runner"} <= set(job["needs"])
    assert {"resolve-vulcan-config", "resolve-benchmark-runner"} <= set(load(workflow)["jobs"])
    assert {name: job["with"][name] for name in RESOLVED_JSON} == RESOLVED_JSON


@pytest.mark.parametrize("caller", BENCHMARK_CALLERS)
def test_every_benchmark_caller_passes_declared_inputs_and_every_required_one(caller) -> None:
    job = benchmark_caller(caller)
    inputs = load(BENCHMARK_WORKFLOW)["on"]["workflow_call"]["inputs"]

    assert job["uses"] == LOCAL_BENCHMARK_WORKFLOW
    assert set(job["with"]) <= set(inputs)
    assert {name for name, value in inputs.items() if value["required"] == "true"} <= set(job["with"])


@pytest.mark.parametrize("caller", BENCHMARK_CALLERS)
def test_benchmark_callers_grant_every_permission_the_benchmark_jobs_request(caller) -> None:
    assert excess_permissions(benchmark_caller(caller), BENCHMARK_WORKFLOW) == []


def test_orchestrator_compile_grants_every_permission_its_build_jobs_request() -> None:
    # The build workflow's benchmark job passes its own grant on to the benchmark workflow.
    assert excess_permissions(load(WORKFLOW)["jobs"]["compile"], BUILD_WORKFLOW) == []


def test_failed_builds_are_reported_after_the_snapshot() -> None:
    workflow = load(WORKFLOW)
    jobs = workflow["jobs"]
    report = jobs["report-partial-publication"]
    script = step(report, "List the builds that did not publish")["run"]
    program = jq_program(script, "jq")

    assert report["needs"] == ["plan", "compile", "publish-catalog"]
    # 100 jobs per page answers HTTP 502 on a full catalog rebuild.
    assert "jobs?per_page=50" in script
    assert "for attempt in 1 2 3" in script
    assert report["if"].startswith("always() &&")
    assert report["permissions"] == {"actions": "read", "contents": "read"}
    # GitHub names a called job "<caller job name> / <called job name>", and the build calls are named by build.
    assert jobs["compile"]["name"] == BUILD_NAME
    # A build's own jobs are the build workflow's Compile and Publish; its benchmark is a nested call.
    assert [job["name"] for job in load(BUILD_WORKFLOW)["jobs"].values()] == ["Compile", "Publish", "Benchmark"]
    benchmark_names = [job["name"] for job in load(BENCHMARK_WORKFLOW)["jobs"].values()]
    assert benchmark_names == [
        "Resolve Vulcan config", "Resolve the benchmark DevKit runner",
        "Queue promoted profiles missing results (excluding text generation)", "Stage benchmark input",
        "Benchmark on DevKit", "Publish benchmarks to Vulcan", "Check model tests and clean up",
    ]

    def job(name: str, conclusion: str) -> dict:
        return {"name": name, "conclusion": conclusion, "html_url": f"https://example.test/{name}"}

    pages = [{"jobs": [
        job("a/b (p) / Compile", "failure"),
        job("a/b (p) / Publish", "skipped"),
        job("e/f (p) / Publish", "cancelled"),
        job("g/h (p) / Publish", "success"),
        job("i/j (p) / Publish", "failure"),
        job("Publish the catalog at this revision", "failure"),
    ]}, {"jobs": [
        # A published build's failed benchmark reports itself; the build's catalog entry is not pending.
        *(job(f"a/b (p) / Benchmark / {name}", "failure") for name in benchmark_names),
        # A reused build's benchmark is called by the orchestrator itself, under the same build name.
        *(job(f"c/d (p) / {name}", "failure") for name in benchmark_names),
    ]}]
    failed = run_jq(program, pages)
    assert [item["name"] for item in failed] == [
        "a/b (p) / Compile", "e/f (p) / Publish", "i/j (p) / Publish",
    ]


# Reusable build workflow: compile on the self-hosted compiler


def test_compile_job_requests_authorized_runner_labels() -> None:
    runs_on = load(BUILD_WORKFLOW)["jobs"]["compile"]["runs-on"]

    assert {"self-hosted", "model-compiler", "${{ inputs.runner_environment }}"} <= set(runs_on)
    assert "${{ fromJSON(inputs.entry).runner_class }}" in runs_on
    assert "model-id-${{ fromJSON(inputs.entry).model_id }}" in runs_on
    assert "build-profile-id-${{ fromJSON(inputs.entry).build_profile_id }}" in runs_on
    assert "component-id-${{ fromJSON(inputs.entry).component_id }}" in runs_on
    assert all(not label.startswith("variant-id-") for label in runs_on)


def test_self_hosted_compile_holds_no_publication_credentials() -> None:
    compile_job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    text = yaml.safe_dump(compile_job)

    assert compile_job["permissions"] == {"contents": "read", "id-token": "write"}
    credentials = step(compile_job, "Configure read-only conversion registry credentials")
    assert credentials["if"] == "${{ inputs.vulcan_environment == 'production' && inputs.conversion_sources != '{}' }}"
    assert credentials["with"]["output-env-credentials"] == "false"
    assert credentials["with"]["output-credentials"] == "true"
    compile_env = step(compile_job, "Compile the invocation with its model's compile script")["env"]
    assert set(compile_env) == {
        "MODEL_REGISTRY_READ_AWS_ACCESS_KEY_ID", "MODEL_REGISTRY_READ_AWS_SECRET_ACCESS_KEY",
        "MODEL_REGISTRY_READ_AWS_SESSION_TOKEN",
    }
    policy = json.loads(credentials["with"]["inline-session-policy"])
    assert policy["Statement"] == [{
        "Effect": "Allow", "Action": "execute-api:Invoke",
        "Resource": "arn:aws:execute-api:${{ inputs.aws_region }}:*:*/*/GET/v1/runs/*/artifacts/onnx",
    }]
    assert compile_job["env"]["MODEL_REGISTRY_IAM_READ_API_URL"] == (
        "${{ inputs.vulcan_environment == 'production' && vars.VULCAN_ARTIFACT_REGISTRY_API_URL || '' }}"
    )
    assert "${{ secrets." not in text
    for item in compile_job["steps"]:
        run = item.get("run", "")
        assert "model-registry publish" not in run
        assert "model_registry publish" not in run
        assert "gh workflow run" not in run


def test_conversion_fetch_steps_receive_isolated_reader_credentials() -> None:
    compile_job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    for name, reader in (
        ("Compile the invocation with its model's compile script", "conversion-reader"),
        ("Verify every MPK and record its provenance", "conversion-verification-reader"),
    ):
        expected = {
            "MODEL_REGISTRY_READ_AWS_ACCESS_KEY_ID": "${{ steps." + reader + ".outputs.aws-access-key-id }}",
            "MODEL_REGISTRY_READ_AWS_SECRET_ACCESS_KEY": "${{ steps." + reader + ".outputs.aws-secret-access-key }}",
            "MODEL_REGISTRY_READ_AWS_SESSION_TOKEN": "${{ steps." + reader + ".outputs.aws-session-token }}",
        }
        assert step(compile_job, name).get("env", {}) == expected
    # Dataset downloads must keep the runner's default AWS credential chain.
    for env in [compile_job["env"], *(item.get("env", {}) for item in compile_job["steps"])]:
        assert not {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"} & env.keys()


def test_verification_refreshes_the_restricted_reader_after_compilation() -> None:
    job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    initial = step(job, "Configure read-only conversion registry credentials")
    refresh_name = "Refresh read-only conversion registry credentials for verification"
    refresh = step(job, refresh_name)

    # A six-hour compilation must not leave verification using the original one-hour session.
    assert job["timeout-minutes"] == "360"
    assert refresh["id"] == "conversion-verification-reader"
    assert refresh["if"] == initial["if"]
    assert refresh["uses"] == initial["uses"]
    assert {key: value for key, value in refresh["with"].items() if key != "role-session-name"} == {
        key: value for key, value in initial["with"].items() if key != "role-session-name"
    }
    assert refresh["with"]["role-session-name"] == "models-conversion-verifier-${{ github.run_id }}"
    compilation_index = step_index(job, "Compile the invocation with its model's compile script")
    assert step_index(job, refresh_name) == compilation_index + 1
    assert step_index(job, "Verify every MPK and record its provenance") == step_index(job, refresh_name) + 1


def test_compile_script_runs_in_activated_model_compiler_shell() -> None:
    compile_job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    script = step(compile_job, "Compile the invocation with its model's compile script")["run"]

    assert "bash --noprofile --norc" in script
    assert "source /etc/profile.d/vulcan-model-compiler.sh" in script
    assert "type activate-model-compiler" in script
    assert "command -v mla-masm" in script
    assert compile_job["env"]["COMPILE_RUNNER"] == (
        "${{ fromJSON(inputs.entry).runner_script || 'scripts/run_model_compilation.sh' }}"
    )
    assert script.index("activate-model-compiler") < script.index('"$COMPILE_RUNNER"')
    assert (
        '"$COMPILE_RUNNER" "$MODEL_DIR" "$BUILD_PROFILE_ID" "$COMPILE_INVOCATION_ID" "$ARTIFACT_ROOT"'
        in script
    )
    order = [
        step_index(compile_job, name)
        for name in (
            "Check out the trusted revision",
            "Verify the pinned Model Compiler environment",
            "Install dataset-registry from the main branch artifact",
            "Install shared registry tooling outside the read-only compiler image",
            "Install pytest for the model's compile tests",
            "Compile the invocation with its model's compile script",
            "Verify every MPK and record its provenance",
            "Test the build with its model's compile tests",
            "Retain the complete compiler output",
        )
    ]
    assert order == sorted(order)


def test_compile_tests_check_the_build_before_its_publication() -> None:
    workflow = load(BUILD_WORKFLOW)
    compile_job = workflow["jobs"]["compile"]
    script = step(compile_job, "Test the build with its model's compile tests")["run"]

    assert step_index(compile_job, "Test the build with its model's compile tests") < step_index(
        compile_job, "Retain the complete compiler output"
    )
    assert "if" not in step(compile_job, "Retain the complete compiler output")
    assert workflow["jobs"]["publish"]["needs"] == "compile"
    # pytest is installed before compiling, and only the test step puts it on PYTHONPATH.
    install = step(compile_job, "Install pytest for the model's compile tests")["run"]
    assert "--target \"$RUNNER_TEMP/model-test-tools\"" in install
    assert "GITHUB_ENV" not in install
    assert "pip install" not in script
    assert 'PYTHONPATH="$RUNNER_TEMP/model-test-tools' in script
    assert 'scripts/run_model_tests.sh compile "$MODEL_DIR" "$PROFILE_OUTPUT_DIR"' in script


def test_sima_cli_install_retries_only_transient_package_manager_lock_failures() -> None:
    script = step(
        load(BUILD_WORKFLOW)["jobs"]["compile"], "Install dataset-registry from the main branch artifact"
    )["run"]

    assert "for attempt in 1 2 3" in script
    assert "Unable to acquire the dpkg frontend lock" in script
    assert "Could not get lock .*(apt|dpkg)" in script
    assert "attempt == 3" in script
    assert "non-retryable error" in script
    assert 'sleep "$delay"' in script


def test_compile_worker_installs_main_dataset_registry_artifact() -> None:
    install = step(load(BUILD_WORKFLOW)["jobs"]["compile"], "Install dataset-registry from the main branch artifact")

    assert install["env"]["DATASET_REGISTRY_ENVIRONMENT"] == "${{ inputs.vulcan_environment }}"
    assert "dataset/dataset-registry@main:latest" in install["run"]
    assert "--download-only" in install["run"]
    assert "sima_neat_dataset_registry-*.whl" in install["run"]
    assert '--target "$dataset_tools"' in install["run"]
    assert '"$MODEL_COMPILER_PYTHON" -m dataset_registry --version' in install["run"]


def test_registry_tool_install_does_not_replace_compiler_dependencies() -> None:
    script = step(
        load(BUILD_WORKFLOW)["jobs"]["compile"], "Install shared registry tooling outside the read-only compiler image"
    )["run"]

    assert "--no-deps" in script
    assert "--target" in script
    assert "import certifi, jsonschema, yaml" in script


def test_every_component_mpk_records_runner_and_input_provenance() -> None:
    compile_job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    verify = step(compile_job, "Verify every MPK and record its provenance")["run"]

    assert compile_job["env"]["COMPONENT_IDS"] == "${{ join(fromJSON(inputs.entry).component_ids, ',') }}"
    assert compile_job["env"]["RUNNER_CLASS"] == "${{ fromJSON(inputs.entry).runner_class }}"
    assert compile_job["env"]["RUNNER_APPROVAL_SPEC_REVISION"] == (
        "${{ fromJSON(inputs.entry).runner_approval_spec_revision }}"
    )
    assert 'for component in "${components[@]}"' in verify
    assert compile_job["env"]["RUNNER_APPROVAL_ID"] == "${{ fromJSON(inputs.entry).runner_approval_id }}"
    assert 'verify-artifact "$PROFILE_OUTPUT_DIR/$component"' in verify
    for option in (
        "--write-manifest",
        "--source-sha256",
        "--calibration-sha256",
        "--compile-fingerprint",
        "--requested-runner-class",
        "--runner-class",
        "--runner-approval-id",
        "--runner-approval-spec-revision",
    ):
        assert option in verify


# Reusable build workflow: publication and retained outputs


@pytest.mark.parametrize("runner_script", ["scripts/run_model_compilation.sh", "scripts/run_llima_compilation.sh"])
@pytest.mark.parametrize("runner_environment", ["vulcan-stg", "vulcan-production"])
def test_build_publication_runner_and_steps(runner_script: str, runner_environment: str) -> None:
    jobs = load(BUILD_WORKFLOW)["jobs"]
    compile_job, publish = jobs["compile"], jobs["publish"]
    upload = step(compile_job, "Retain the complete compiler output")
    download = step(publish, "Download this compiler output")

    assert publish["needs"] == "compile"
    assert publish["if"] == (
        "${{ !fromJSON(inputs.entry).build_on_demand || "
        "fromJSON(inputs.entry).runner_script == 'scripts/run_llima_compilation.sh' }}"
    )
    runner = evaluate(publish["runs-on"].removeprefix("${{").removesuffix("}}"), {
        "fromJSON(inputs.entry).runner_script": runner_script,
        "inputs.runner_environment": runner_environment,
    })
    assert runner == (
        ["self-hosted", "linux", "arm64", runner_environment, "ubuntu", "default"]
        if runner_script == "scripts/run_llima_compilation.sh" else "ubuntu-latest"
    )
    assert publish["permissions"] == {"actions": "write", "contents": "read", "id-token": "write"}
    assert publish["outputs"] == {"package_count": "${{ steps.packages.outputs.count }}"}
    assert upload["with"]["name"].startswith("compiler-output-")
    assert download["with"]["name"] == upload["with"]["name"]
    assert "continue-on-error" not in download
    verify = step(publish, "Verify the complete component set")["run"]
    assert "validate-compile-outputs" in verify
    assert "compile-result.json" in verify
    credentials = step(publish, "Configure artifact-registry publisher credentials")
    assert credentials["with"]["role-to-assume"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN }}"
    assert "model-registry publish" in step(publish, "Publish the complete component set")["run"]
    order = [
        step_index(publish, name)
        for name in (
            "Download this compiler output",
            "Verify the complete component set",
            "Configure artifact-registry publisher credentials",
            "Publish the complete component set",
            "Assemble the packages this build completes",
            "Publish the finalized packages",
            "Publish this build in the catalog",
        )
    ]
    assert order == sorted(order)


def test_no_model_file_reaches_the_benchmark_through_an_actions_artifact() -> None:
    workflow = load(BUILD_WORKFLOW)
    publish = workflow["jobs"]["publish"]
    uploads = [item for item in publish["steps"] if item.get("uses", "").startswith("actions/upload-artifact")]

    # The benchmark stages the build from the registry catalog this job publishes, so the job retains only the
    # candidate receipts an on-demand rerun resumes from.
    assert [item["name"] for item in uploads] == ["Retain candidate publication receipts"]
    assert all('"_compiled/$receipt"' not in item.get("run", "") for item in publish["steps"])
    assert "compiled-model-" not in BUILD_WORKFLOW.read_text(encoding="utf-8")
    assert "source_run_id" not in workflow["jobs"]["benchmark"]["with"]


@pytest.mark.parametrize(
    ("publishable", "on_demand", "packages", "environment", "target", "stages", "expected"),
    [
        (True, False, "1", "staging", "modalix", "performanceaccuracy", True),
        (True, False, "1", "staging", "modalix", "accuracy", True),
        (True, False, "1", "staging", "modalix", "", False),
        (True, False, "1", "production", "modalix", "performance", True),
        (True, False, "1", "staging", "davinci", "performance", False),
        # A benchmark runs under the profile's published package.
        (False, False, "1", "staging", "modalix", "performance", False),
        (True, True, "1", "staging", "modalix", "performance", False),
        (True, False, "0", "staging", "modalix", "performance", False),
    ],
)
def test_new_builds_start_only_their_applicable_benchmarks(
    publishable, on_demand, packages, environment, target, stages, expected
) -> None:
    benchmark = load(BUILD_WORKFLOW)["jobs"]["benchmark"]
    values = {
        "inputs.publishable": publishable,
        "fromJSON(inputs.entry).build_on_demand": on_demand,
        "needs.publish.outputs.package_count": packages,
        "inputs.vulcan_environment": environment,
        "fromJSON(inputs.entry).target": target,
        "join(fromJSON(inputs.entry).stages, '')": stages,
    }

    assert evaluate(benchmark["if"], values) is expected


def test_new_build_benchmarks_its_own_compile_output() -> None:
    workflow = load(BUILD_WORKFLOW)
    benchmark = workflow["jobs"]["benchmark"]

    # It starts once this build publishes, while other builds may still compile.
    assert benchmark["needs"] == "publish"
    assert benchmark["name"] == "Benchmark"
    assert benchmark["uses"] == LOCAL_BENCHMARK_WORKFLOW
    assert "runs-on" not in benchmark
    assert benchmark["permissions"] == {"actions": "write", "contents": "read", "id-token": "write"}
    assert benchmark["secrets"] == "inherit"
    assert benchmark["with"] == {
        "environment": "${{ inputs.vulcan_environment }}",
        # The benchmark stages this build from the catalog the publish job wrote at this revision.
        "source_sha": "${{ github.sha }}",
        "source_branch": "${{ github.ref_name }}",
        "model_id": "${{ fromJSON(inputs.entry).model_id }}",
        "build_profile_id": "${{ fromJSON(inputs.entry).build_profile_id }}",
        "compile_invocation_id": "${{ fromJSON(inputs.entry).compile_invocation_id }}",
        "stages": "${{ join(fromJSON(inputs.entry).stages, ',') }}",
        "vulcan_config": "${{ inputs.vulcan_config }}",
        "devkit_runner": "${{ inputs.devkit_runner }}",
    }
    inputs = workflow["on"]["workflow_call"]["inputs"]
    for name in ("vulcan_config", "devkit_runner"):
        assert (inputs[name]["required"], inputs[name]["type"]) == ("true", "string")


def test_native_candidates_finalize_and_select_in_the_existing_publish_job() -> None:
    publish = load(BUILD_WORKFLOW)["jobs"]["publish"]
    assemble = step(publish, "Assemble the packages this build completes")
    packages = step(publish, "Publish the finalized packages")
    catalog = step(publish, "Publish this build in the catalog")

    finalize = step(publish, "Publish and finalize the complete candidate")
    select = step(publish, "Publish this candidate in the feature-branch catalog")
    assert finalize["if"] == "fromJSON(inputs.entry).build_on_demand"
    assert "model-registry publish-candidate --repository . --dist-dir _compiled" in finalize["run"]
    assert "model-registry publish-branch-deployment --candidate _candidate/candidate.json" in select["run"]
    assert step_index(publish, finalize["name"]) < step_index(publish, select["name"])
    assert "inputs.publishable" not in select["if"]
    for branch in ("develop", "main", "master"):
        assert f"github.ref != 'refs/heads/{branch}'" in select["if"]
    assert step(publish, "Publish the complete component set")["if"] == "${{ !fromJSON(inputs.entry).build_on_demand }}"
    assert assemble["if"] == "inputs.publishable && !fromJSON(inputs.entry).build_on_demand"
    assert "--all" in assemble["run"]
    assert packages["if"] == (
        "inputs.publishable && !fromJSON(inputs.entry).build_on_demand && steps.packages.outputs.count != '0'"
    )
    assert "model-registry publish-package" in packages["run"]
    assert catalog["if"] == "inputs.publishable && !fromJSON(inputs.entry).build_on_demand"
    assert "--compiled-dir _compiled" in catalog["run"]
    assert '--base-branches "$BASE_BRANCHES"' in catalog["run"]


def test_candidates_join_every_invocation_and_never_promote_catalog() -> None:
    job = load(WORKFLOW)["jobs"]["publish-candidates"]
    assert "compile" in job["needs"]
    for result, cancelled, expected in (
        ("success", False, True), ("failure", False, True),
        ("skipped", False, False), ("cancelled", False, False), ("failure", True, False),
    ):
        assert evaluate(job["if"], {
            "cancelled()": cancelled, "needs.plan.outputs.candidate_count": "2", "needs.compile.result": result,
        }) is expected
    assert job["runs-on"] == "ubuntu-latest"
    assert job["environment"] == "${{ needs.select.outputs.vulcan_environment }}"
    steps = job["steps"]
    download = next(item for item in steps if item.get("uses") == "actions/download-artifact@v7")
    assert download["with"]["pattern"].endswith("-${{ matrix.build_profile_id }}-*")
    assert not download["with"].get("merge-multiple", False)
    text = "\n".join(item.get("run", "") for item in steps)
    assert "model-registry publish-candidate" in text
    assert "model-registry publish-branch-deployment" in text
    assert "publish-catalog" not in text
    assert "--summary" in text



def test_concurrent_builds_retry_catalog_snapshot_conflicts() -> None:
    catalog = step(load(BUILD_WORKFLOW)["jobs"]["publish"], "Publish this build in the catalog")["run"]

    # publish-catalog retries the conflict in process; the shell loop only gives it a second run.
    assert "for attempt in 1 2; do" in catalog
    assert "grep -q 'catalog_snapshot_conflict'" in catalog
    assert "(( attempt == 2 ))" in catalog
    assert "ceiling=$(( 2 ** attempt > 30 ? 30 : 2 ** attempt ))" in catalog
    assert 'sleep "$(( 1 + RANDOM % ceiling ))"' in catalog


# Scripts the workflows invoke


def test_directly_invoked_scripts_exist_and_are_executable() -> None:
    texts = "\n".join(path.read_text(encoding="utf-8") for path in WORKFLOWS.glob("*.yml"))
    scripts = sorted(set(re.findall(r"(?<![\w/])scripts/[\w.-]+\.sh", texts)))

    assert "scripts/run_model_compilation.sh" in scripts
    for script in scripts:
        path = ROOT / script
        assert path.is_file(), script
        assert os.access(path, os.X_OK), f"{script} is not executable"


FAKE_REGISTRY_PYTHON = """#!/usr/bin/env bash
if [[ "$1" == "-m" && "$2" == "model_registry" ]]; then
  case "$3" in
    matrix) echo '{"include": [{"compile_invocation_id": "model", "component_ids": ["model"]}]}' ;;
    fetch) echo '{"path": "/sources/model.onnx"}' ;;
    fetch-calibration | fetch-fidelity-dataset) echo '{"path": "/datasets/coco"}' ;;
  esac
  exit 0
fi
exec python3 "$@"
"""

# Like the RF-DETR and YOLO recipes, refuse an output directory that already exists.
STRICT_RECIPE = """import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
args, _ = parser.parse_known_args()
if args.output_dir.exists():
    raise FileExistsError(f"output directory already exists: {args.output_dir}")
args.output_dir.mkdir(parents=True)
(args.output_dir / "compile-outputs.json").write_text("{}")
"""


def test_compile_runner_leaves_the_output_directory_to_the_recipe(tmp_path: Path) -> None:
    model_dir = tmp_path / "repository" / "models" / "alpha"
    (model_dir / "scripts").mkdir(parents=True)
    (model_dir / "scripts" / "compile.py").write_text(STRICT_RECIPE, encoding="utf-8")
    python = tmp_path / "python"
    python.write_text(FAKE_REGISTRY_PYTHON, encoding="utf-8")
    python.chmod(0o755)
    output_root = tmp_path / "output"
    stale = output_root / "alpha" / "fixture" / "stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")

    subprocess.run(
        [ROOT / "scripts" / "run_model_compilation.sh", model_dir, "fixture", "model", output_root],
        check=True,
        env={**os.environ, "MODEL_COMPILER_PYTHON": str(python), "MODEL_REGISTRY_CACHE_DIR": str(tmp_path / "cache")},
    )

    assert sorted(path.name for path in stale.parent.iterdir()) == ["compile-outputs.json", "compile-timing.json"]


def test_resnet_recipe_is_python_and_uses_the_standard_cli_contract() -> None:
    recipe = ast.parse((ROOT / "models" / "resnet_50" / "scripts" / "compile.py").read_text(encoding="utf-8"))
    imported = {
        (node.module, alias.name)
        for node in ast.walk(recipe)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    # The recipe's command line is the shared parser's.
    parser = ast.parse((ROOT / "tools" / "model_registry" / "compile_script.py").read_text(encoding="utf-8"))
    option_strings = {
        value.value
        for node in ast.walk(parser)
        if isinstance(node, ast.Call)
        for value in node.args
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith("--")
    }
    assert ("model_registry.compile_script", "parse_arguments") in imported
    assert {"--model", "--output-dir", "--build-profile", "--dataset"} <= option_strings
    assert "--accuracy-samples" not in option_strings
    assert "--accuracy-workers" not in option_strings


def test_compile_dataset_mirror_uses_the_selected_environment_config() -> None:
    caller = load(WORKFLOW)["jobs"]["compile"]
    build = load(BUILD_WORKFLOW)
    assert caller["with"]["dataset_bucket"] == "${{ needs.resolve-vulcan-config.outputs.dataset_bucket }}"
    assert build["on"]["workflow_call"]["inputs"]["dataset_bucket"]["required"] == "true"
    assert build["jobs"]["compile"]["env"]["SIMA_DATASET_S3_URI"] == "s3://${{ inputs.dataset_bucket }}"


def test_on_demand_planner_does_not_restrict_the_selected_environment() -> None:
    plan = step(load(WORKFLOW)["jobs"]["plan"], "Compare fingerprints with published runs")["run"]
    assert '"$VULCAN_ENVIRONMENT" != "staging"' not in plan
    assert 'enabled only in staging' not in plan
    assert '--include-on-demand' in plan


@pytest.mark.parametrize("environment,public_url", [
    ("production", ""), ("production", "https://public.example.test"),
    ("staging", "https://public.example.test"),
])
def test_mpk_candidate_selects_its_branch_with_private_or_public_registry_reads(
    tmp_path, monkeypatch, environment, public_url,
):
    from model_registry import fetch, registry
    from model_registry.candidates import publish_branch_deployment

    private_url = "https://private.example.test"
    values = {
        "vars.VULCAN_ARTIFACT_REGISTRY_API_URL": private_url,
        "vars.VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL": public_url,
        "needs.select.outputs.vulcan_environment": environment,
    }
    job_env = load(WORKFLOW)["jobs"]["publish-candidates"]["env"]
    read_url = evaluate(job_env["REGISTRY_READ_API_URL"][3:-3], values)
    iam_url = evaluate(job_env["MODEL_REGISTRY_IAM_READ_API_URL"][3:-3], values)
    monkeypatch.setenv("MODEL_REGISTRY_IAM_READ_API_URL", iam_url)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    reads, writes = [], []
    catalog = {"catalog": {"canonical_sha256": "b" * 64, "document": {"models": []}}}

    def signed(client, method, path, body, key):
        assert method == "GET" and body is None
        reads.append(("signed", client.api_url + path))
        return catalog

    def unsigned(request, timeout):
        reads.append(("public", request.full_url))
        return io.BytesIO(json.dumps(catalog).encode())

    monkeypatch.setattr(registry.SigV4RegistryClient, "_curl_request", signed)
    monkeypatch.setattr(fetch, "urlopen", unsigned)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "deployment_id": "11111111-1111-4111-8111-111111111111", "manifest_digest": "a" * 64,
        "git_ref": "refs/heads/feature/mpk-model", "state": "finalized",
    }))

    def write(method, path, body, key):
        assert method == "POST" and path.endswith("/branch-deployments")
        assert body["expected_snapshot"] == "b" * 64
        assert body["git_ref"] == "refs/heads/feature/mpk-model"
        writes.append(body)
        return {"selection": {"deployment_id": body["deployment_id"]}, "result_snapshot": "c" * 64}

    result = publish_branch_deployment(
        candidate, api_url=private_url, read_api_url=read_url,
        aws_region="us-west-2", request=write,
    )
    assert len(reads) == len(writes) == 1
    assert reads[0][0] == ("signed" if environment == "production" and not public_url else "public")
    assert reads[0][1].startswith((public_url or private_url) + "/v1/")
    assert result["selection"]["deployment_id"] == json.loads(candidate.read_text())["deployment_id"]
    assert json.loads((tmp_path / "branch-publication.json").read_text()) == result
