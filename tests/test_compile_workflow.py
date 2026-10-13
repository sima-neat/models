from __future__ import annotations

import ast
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


def evaluate(condition: str, values: dict[str, object]) -> bool:
    """Evaluate a GitHub Actions condition after substituting every context reference it uses."""
    expression = condition
    for name in sorted(values, key=len, reverse=True):
        expression = expression.replace(name, repr(values[name]))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval(expression, {"__builtins__": {}}, {})


def dispatched_inputs(script: str) -> set[str]:
    return set(re.findall(r"-f (\w+)=", script))


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

    # A push that changes no compile input still publishes its catalog; its stage keys select no work.
    assert "push" in on and not on["push"]


def test_production_is_an_explicit_dispatch_choice() -> None:
    inputs = load(WORKFLOW)["on"]["workflow_dispatch"]["inputs"]

    assert set(inputs) == {"environment", "models", "profiles", "components", "base_branch", "force_rebuild_all"}
    assert inputs["environment"]["options"] == ["staging", "production"]
    assert inputs["environment"]["default"] == "staging"
    assert inputs["force_rebuild_all"]["type"] == "boolean"


def test_newer_pushes_supersede_runs_but_manual_dispatches_run_to_completion() -> None:
    concurrency = load(WORKFLOW)["concurrency"]

    assert concurrency["cancel-in-progress"] == "${{ github.event_name != 'workflow_dispatch' }}"
    # Only dispatches get a per-run group, so they never cancel or replace one another.
    assert "github.event_name == 'workflow_dispatch' && format('-{0}', github.run_id)" in concurrency["group"]
    assert "github.ref" in concurrency["group"]
    assert "inputs.environment" in concurrency["group"]


# Job graph and pull-request isolation


def test_compile_workflow_has_independent_build_stages() -> None:
    jobs = load(WORKFLOW)["jobs"]

    assert set(jobs) == {
        "select",
        "portal",
        "resolve-vulcan-config",
        "publish-model-browser",
        "plan",
        "authorize-runner-classes",
        "initialize-catalog",
        "compile",
        "publish-catalog",
        "measure",
        "report-partial-publication",
    }


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
        if isinstance(job["runs-on"], list) and "self-hosted" in job["runs-on"]
    ]
    assert self_hosted == ["compile"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_trusted_events_reach_the_build(event) -> None:
    condition = load(WORKFLOW)["jobs"]["compile"]["if"]
    values = {
        "!cancelled()": True,
        "github.event_name": event,
        "needs.select.outputs.publishable": "true",
        "needs.resolve-vulcan-config.result": "success",
        "needs.authorize-runner-classes.result": "success",
        "needs.initialize-catalog.result": "success",
    }
    assert evaluate(condition, values) is True


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
    plan = step(jobs["plan"], "Compare stage keys with published runs")
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


def test_targeted_profile_or_component_dispatch_is_compile_only() -> None:
    select = load(WORKFLOW)["jobs"]["select"]
    scope = next(item for item in select["steps"] if item.get("id") == "scope")

    assert select["outputs"]["publishable"] == "${{ steps.scope.outputs.publishable }}"
    assert scope["env"] == {
        "REQUESTED_PROFILES": "${{ inputs.profiles }}",
        "REQUESTED_COMPONENTS": "${{ inputs.components }}",
    }
    assert "publishable=false" in scope["run"]


def test_branches_reuse_builds_from_the_resolved_base_branch_catalogs() -> None:
    jobs = load(WORKFLOW)["jobs"]
    base = next(item for item in jobs["select"]["steps"] if item.get("id") == "base")

    assert "model-registry resolve-base" in base["run"]
    assert ".base_branches | join(\",\")" in base["run"]
    assert base["env"]["PREFERRED_BRANCH"] == "${{ inputs.base_branch }}"
    plan = step(jobs["plan"], "Compare stage keys with published runs")
    assert plan["env"]["BASE_BRANCHES"] == "${{ needs.select.outputs.base_branches }}"
    assert '--base-branches "$BASE_BRANCHES"' in plan["run"]
    assert jobs["compile"]["with"]["base_branches"] == "${{ needs.select.outputs.base_branches }}"


def test_a_portal_that_fails_its_check_stops_the_run_before_planning() -> None:
    jobs = load(WORKFLOW)["jobs"]

    assert jobs["portal"]["uses"] == "./.github/workflows/developer-docs.yml"
    assert "portal" in jobs["plan"]["needs"]
    # Without a status function, plan runs only when portal succeeded; every later job needs plan.
    assert "if" not in jobs["plan"]


def test_plan_compares_stage_keys_with_the_environment_registry() -> None:
    plan_job = load(WORKFLOW)["jobs"]["plan"]
    plan = step(plan_job, "Compare stage keys with published runs")

    assert plan_job["needs"] == ["select", "portal"]
    assert plan_job["environment"] == "${{ needs.select.outputs.vulcan_environment }}"
    assert "permissions" not in plan_job
    assert plan["env"]["REGISTRY_READ_API_URL"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL }}"
    assert "model-registry plan-stages" in plan["run"]
    assert '--read-api-url "$REGISTRY_READ_API_URL"' in plan["run"]
    assert "--offline" not in plan["run"]
    for name in ("REQUESTED_MODELS", "REQUESTED_PROFILES", "REQUESTED_COMPONENTS", "FORCE_REBUILD_ALL"):
        assert name in plan["env"]
    assert "--force-rebuild-all" in plan["run"]
    assert plan_job["outputs"] == {
        "compile_count": "${{ steps.plan.outputs.compile_count }}",
        "compile_matrix": "${{ steps.plan.outputs.compile_matrix }}",
        "measure_count": "${{ steps.plan.outputs.measure_count }}",
        "measure_matrix": "${{ steps.plan.outputs.measure_matrix }}",
    }


def test_reused_build_stages_are_measured_on_modalix_only() -> None:
    plan = step(load(WORKFLOW)["jobs"]["plan"], "Compare stage keys with published runs")
    program = jq_program(plan["run"], 'measure_matrix="$(jq -c')
    document = {
        "compile": {"include": []},
        "measure": {"include": [{"model_id": "a", "target": "modalix"}, {"model_id": "b", "target": "davinci"}]},
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

    assert build["needs"] == ["select", "resolve-vulcan-config", "authorize-runner-classes", "initialize-catalog"]
    assert build["strategy"]["matrix"] == "${{ fromJSON(needs.authorize-runner-classes.outputs.matrix) }}"
    assert build["strategy"]["fail-fast"] == "false"
    assert build["with"]["entry"] == "${{ toJSON(matrix) }}"
    assert build["with"]["publishable"] == "${{ needs.select.outputs.publishable == 'true' }}"
    assert build["secrets"] == "inherit"
    assert set(build["with"]) == set(load(BUILD_WORKFLOW)["on"]["workflow_call"]["inputs"])


def test_catalog_marks_authorized_builds_pending_before_they_run() -> None:
    initialize = load(WORKFLOW)["jobs"]["initialize-catalog"]
    publish = step(initialize, "Publish the pending invocation plan")

    assert initialize["needs"] == ["select", "resolve-vulcan-config", "authorize-runner-classes"]
    assert "needs.select.outputs.publishable == 'true'" in initialize["if"]
    assert "needs.authorize-runner-classes.result == 'success'" in initialize["if"]
    assert publish["env"]["SELECTED_MATRIX"] == "${{ needs.authorize-runner-classes.outputs.matrix }}"
    assert "model-registry publish-catalog" in publish["run"]
    assert '--selected-matrix "$matrix_file"' in publish["run"]
    assert '--base-branches "$BASE_BRANCHES"' in publish["run"]


@pytest.mark.parametrize(
    ("publishable", "initialize_result", "expected"),
    [
        ("false", "skipped", True),
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
    ("event", "environment", "catalog_result", "measure_count", "expected"),
    [
        ("push", "staging", "success", "1", True),
        ("workflow_dispatch", "staging", "success", "3", True),
        ("push", "production", "success", "1", False),
        ("push", "staging", "failure", "1", False),
        ("push", "staging", "skipped", "1", False),
        ("push", "staging", "success", "0", False),
    ],
)
def test_reused_builds_start_devkit_stages_after_the_catalog_snapshot(
    event, environment, catalog_result, measure_count, expected,
) -> None:
    job = load(WORKFLOW)["jobs"]["measure"]
    values = {
        "!cancelled()": True,
        "github.event_name": event,
        "needs.publish-catalog.result": catalog_result,
        "needs.select.outputs.vulcan_environment": environment,
        "needs.plan.outputs.measure_count": measure_count,
    }

    assert job["needs"] == ["select", "plan", "publish-catalog"]
    # Without a status function GitHub adds success(), which also requires the
    # indirectly needed build jobs and skips reused builds when one build fails.
    assert job["if"].startswith("!cancelled() &&")
    assert evaluate(job["if"], values) is expected


def test_reused_builds_start_their_stages_from_one_job() -> None:
    job = load(WORKFLOW)["jobs"]["measure"]
    start = step(job, "Start the stages reused builds still lack")

    assert job["name"] == "Dispatch DevKit stages for reused builds"
    assert job["permissions"] == {"actions": "write", "contents": "read"}
    assert "strategy" not in job
    assert job["steps"][0]["uses"] == "actions/checkout@v6"
    assert start["env"]["MEASURE_MATRIX"] == "${{ needs.plan.outputs.measure_matrix }}"
    # An empty source run restores each reused build from the registry at this catalog revision.
    assert start["run"] == 'scripts/start_devkit_stages.sh "$MEASURE_MATRIX"'


def test_failed_builds_are_reported_after_the_snapshot() -> None:
    workflow = load(WORKFLOW)
    jobs = workflow["jobs"]
    report = jobs["report-partial-publication"]
    script = step(report, "List the builds that did not publish")["run"]
    program = jq_program(script, "| jq")

    assert report["needs"] == ["plan", "compile", "publish-catalog"]
    assert report["if"].startswith("always() &&")
    assert report["permissions"] == {"actions": "read", "contents": "read"}
    # The filter matches the compile job's name, and no other job shares its prefix.
    assert jobs["compile"]["name"].startswith("Compile ")
    assert [name for name, job in jobs.items() if job["name"].startswith("Compile ")] == ["compile"]

    def job(name: str, conclusion: str) -> dict:
        return {"name": name, "conclusion": conclusion, "html_url": f"https://example.test/{name}"}

    pages = [{"jobs": [
        job("Compile a/b (p) / Compile", "failure"),
        job("Compile a/b (p) / Publish", "skipped"),
        job("Compile c/d (p) / Publish", "cancelled"),
        job("Compile e/f (p) / Publish", "success"),
        job("Publish the catalog at this revision", "failure"),
    ]}]
    failed = run_jq(program, pages)
    assert [item["name"] for item in failed] == ["Compile a/b (p) / Compile", "Compile c/d (p) / Publish"]


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

    assert compile_job["permissions"] == {"contents": "read"}
    assert all("aws-actions/configure-aws-credentials" not in item.get("uses", "") for item in compile_job["steps"])
    assert "${{ secrets." not in text
    assert "${{ vars." not in text
    for item in compile_job["steps"]:
        run = item.get("run", "")
        assert "model-registry publish" not in run
        assert "model_registry publish" not in run
        assert "gh workflow run" not in run


def test_compile_script_runs_in_activated_model_compiler_shell() -> None:
    compile_job = load(BUILD_WORKFLOW)["jobs"]["compile"]
    script = step(compile_job, "Compile the invocation with its model's compile script")["run"]

    assert "bash --noprofile --norc" in script
    assert "source /etc/profile.d/vulcan-model-compiler.sh" in script
    assert "type activate-model-compiler" in script
    assert "command -v mla-masm" in script
    assert script.index("activate-model-compiler") < script.index("scripts/run_model_compilation.sh")
    assert (
        'scripts/run_model_compilation.sh "$MODEL_DIR" "$BUILD_PROFILE_ID" "$COMPILE_INVOCATION_ID" "$ARTIFACT_ROOT"'
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
            "Retain the complete compiler output",
            "Test the build with its model's compile tests",
        )
    ]
    assert order == sorted(order)


def test_compile_tests_check_the_build_before_its_publication() -> None:
    workflow = load(BUILD_WORKFLOW)
    compile_job = workflow["jobs"]["compile"]
    script = step(compile_job, "Test the build with its model's compile tests")["run"]

    assert step_index(compile_job, "Test the build with its model's compile tests") == len(compile_job["steps"]) - 1
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
    assert 'for index in "${!components[@]}"' in verify
    assert 'verify-mpk "$PROFILE_OUTPUT_DIR/$component"' in verify
    for option in (
        "--write-manifest",
        "--source-sha256",
        "--calibration-sha256",
        "--build-fingerprint",
        "--requested-runner-class",
        "--runner-class",
        "--runner-approval-id",
        "--runner-approval-spec-revision",
    ):
        assert option in verify


# Reusable build workflow: publish from a hosted runner


def test_build_is_published_from_a_hosted_runner() -> None:
    jobs = load(BUILD_WORKFLOW)["jobs"]
    compile_job, publish = jobs["compile"], jobs["publish"]
    upload = step(compile_job, "Retain the complete compiler output")
    download = step(publish, "Download this compiler output")

    assert publish["needs"] == "compile"
    assert "if" not in publish
    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["permissions"] == {"actions": "write", "contents": "read", "id-token": "write"}
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
            "Retain the published model output for the DevKit stages",
            "Start the DevKit stages",
            "Assemble the packages this build completes",
            "Publish the finalized packages",
            "Publish this build in the catalog",
        )
    ]
    assert order == sorted(order)


def test_published_output_is_the_artifact_the_benchmark_verifies() -> None:
    publish = load(BUILD_WORKFLOW)["jobs"]["publish"]
    retained = step(publish, "Retain the published model output for the DevKit stages")
    stage = load(BENCHMARK_WORKFLOW)["jobs"]["stage-dispatched-benchmark"]
    verify = step(stage, "Verify the source run and compiled artifact")["run"]

    assert retained["with"]["name"] == (
        "compiled-model-${{ fromJSON(inputs.entry).model_id }}-${{ fromJSON(inputs.entry).build_profile_id }}"
        "-${{ fromJSON(inputs.entry).compile_invocation_id }}"
    )
    assert 'artifact_name="compiled-model-$MODEL_ID-$BUILD_PROFILE_ID-$COMPILE_INVOCATION_ID"' in verify


@pytest.mark.parametrize(
    ("environment", "target", "stages", "expected"),
    [
        ("staging", "modalix", "performanceaccuracy", True),
        ("staging", "modalix", "accuracy", True),
        ("staging", "modalix", "", False),
        ("production", "modalix", "performance", False),
        ("staging", "davinci", "performance", False),
    ],
)
def test_new_builds_start_only_their_applicable_devkit_stages(environment, target, stages, expected) -> None:
    start = step(load(BUILD_WORKFLOW)["jobs"]["publish"], "Start the DevKit stages")
    values = {
        "inputs.vulcan_environment": environment,
        "fromJSON(inputs.entry).target": target,
        "join(fromJSON(inputs.entry).stages, '')": stages,
    }

    assert "inputs.publishable" not in start["if"]
    assert evaluate(start["if"], values) is expected


def test_new_build_starts_its_stages_with_the_shared_script() -> None:
    start = step(load(BUILD_WORKFLOW)["jobs"]["publish"], "Start the DevKit stages")

    # A failed dispatch leaves the build published; the next run plans the missing stages.
    assert start["continue-on-error"] == "true"
    assert start["env"]["ENTRY"] == "${{ inputs.entry }}"
    # The benchmark downloads this compile run's output.
    assert start["run"] == 'scripts/start_devkit_stages.sh "$ENTRY" "$GITHUB_RUN_ID"'


START_SCRIPT = ROOT / "scripts" / "start_devkit_stages.sh"


def run_start_script(tmp_path: Path, gh: str, *args: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run the start script with a fake gh that records each call's arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "gh").write_text(f'#!/bin/sh\necho "$*" >> {calls}\n{gh}\n', encoding="utf-8")
    (bin_dir / "gh").chmod(0o755)
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_REPOSITORY": "sima-neat/models",
        "GITHUB_SHA": "0" * 40,
        "GITHUB_REF_NAME": "develop",
    }
    result = subprocess.run(["bash", str(START_SCRIPT), *args], env=env, capture_output=True, text=True)
    return result, summary, calls


def build(model_id: str, components: list[str], stages: list[str]) -> dict:
    return {
        "model_id": model_id,
        "build_profile_id": "int8",
        "compile_invocation_id": "default",
        "benchmark_component_ids": components,
        "stages": stages,
    }


def test_start_script_dispatches_every_benchmark_input() -> None:
    script = START_SCRIPT.read_text(encoding="utf-8")
    benchmark_inputs = load(BENCHMARK_WORKFLOW)["on"]["workflow_dispatch"]["inputs"]

    assert "gh workflow run vulcan-model-benchmarking.yml" in script
    assert dispatched_inputs(script) == set(benchmark_inputs)
    # A checkout-free caller has no git remote for gh to infer the repository from.
    assert '--repo "$GITHUB_REPOSITORY"' in script
    # The benchmark only runs when dispatched on its source branch.
    assert '--ref "$GITHUB_REF_NAME"' in script
    assert '-f source_branch="$GITHUB_REF_NAME"' in script


@pytest.mark.parametrize(
    ("gh", "link"),
    [
        ("echo https://github.com/sima-neat/models/actions/runs/123", "https://github.com/sima-neat/models/actions/runs/123"),
        ("true", "link unavailable"),
    ],
)
def test_start_script_logs_one_devkit_run_link_per_build(tmp_path: Path, gh: str, link: str) -> None:
    matrix = {"include": [build("alpha", ["model"], ["performance", "accuracy"]), build("beta", [], ["accuracy"])]}

    result, summary, calls = run_start_script(tmp_path, gh, json.dumps(matrix))

    lines = (
        f"DevKit run for alpha/int8/default (performance,accuracy): {link}\n"
        f"DevKit run for beta/int8/default (accuracy): {link}\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == lines
    assert summary.read_text(encoding="utf-8") == lines
    alpha, beta = calls.read_text(encoding="utf-8").splitlines()
    assert "-f source_run_id= " in alpha
    assert "-f component_ids=model " in alpha
    # An accuracy-only build names no performance component.
    assert "-f component_ids= -f stages=accuracy" in beta


def test_start_script_accepts_one_build_with_its_source_run(tmp_path: Path) -> None:
    result, _, calls = run_start_script(tmp_path, "true", json.dumps(build("alpha", ["model"], ["performance"])), "77")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "DevKit run for alpha/int8/default (performance): link unavailable\n"
    assert "-f source_run_id=77 " in calls.read_text(encoding="utf-8")


def test_start_script_starts_the_other_builds_after_a_failure(tmp_path: Path) -> None:
    gh = 'case "$*" in *model_id=alpha*) echo "HTTP 404: not found" >&2; exit 1;; esac'
    matrix = {"include": [build("alpha", ["model"], ["performance"]), build("beta", ["model"], ["performance"])]}

    result, summary, calls = run_start_script(tmp_path, gh, json.dumps(matrix))

    assert result.returncode == 1
    assert "HTTP 404: not found" in result.stderr
    assert result.stdout == (
        "DevKit run for alpha/int8/default (performance): not started\n"
        "DevKit run for beta/int8/default (performance): link unavailable\n"
    )
    assert summary.read_text(encoding="utf-8") == result.stdout
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 2


def test_packages_and_catalog_update_only_for_publishable_builds() -> None:
    publish = load(BUILD_WORKFLOW)["jobs"]["publish"]
    assemble = step(publish, "Assemble the packages this build completes")
    packages = step(publish, "Publish the finalized packages")
    catalog = step(publish, "Publish this build in the catalog")

    assert assemble["if"] == "inputs.publishable"
    assert "--all" in assemble["run"]
    assert packages["if"] == "inputs.publishable && steps.packages.outputs.count != '0'"
    assert "model-registry publish-package" in packages["run"]
    assert catalog["if"] == "inputs.publishable"
    assert "--compiled-dir _compiled" in catalog["run"]
    assert '--base-branches "$BASE_BRANCHES"' in catalog["run"]


def test_concurrent_builds_retry_catalog_snapshot_conflicts() -> None:
    catalog = step(load(BUILD_WORKFLOW)["jobs"]["publish"], "Publish this build in the catalog")["run"]

    assert "for attempt in 1 2 3 4 5 6 7 8; do" in catalog
    assert "grep -q 'catalog_snapshot_conflict'" in catalog
    assert "(( attempt == 8 ))" in catalog
    assert 'sleep "$(( delay + RANDOM % 5 ))"' in catalog


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
    recipe = ROOT / "models" / "resnet_50" / "scripts" / "compile.py"
    tree = ast.parse(recipe.read_text(encoding="utf-8"))
    option_strings = {
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for value in node.args
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith("--")
    }
    assert {"--model", "--output-dir", "--build-profile", "--dataset"} <= option_strings
    assert "--accuracy-samples" not in option_strings
    assert "--accuracy-workers" not in option_strings
