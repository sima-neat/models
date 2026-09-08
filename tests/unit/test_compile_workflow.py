from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Compile.yml"
INVOCATION_WORKFLOW = ROOT / ".github" / "workflows" / "model-compile-invocation.yml"


def _workflow():
    parent = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    child = yaml.load(INVOCATION_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    parent["jobs"].update(child["jobs"])
    return parent


@pytest.mark.parametrize(
    ("compile_result", "event", "cancelled", "authorized", "expected"),
    [
        ("success", "push", False, True, True),
        ("failure", "push", False, True, False),
        ("failure", "workflow_dispatch", False, True, False),
        ("failure", "pull_request", False, True, False),
        ("failure", "push", True, True, False),
        ("failure", "push", False, False, False),
        ("cancelled", "push", False, True, False),
        ("skipped", "push", False, True, False),
    ],
)
def test_publication_is_independent_of_other_compile_failures(
    compile_result,
    event,
    cancelled,
    authorized,
    expected,
) -> None:
    workflow = _workflow()
    publisher = workflow["jobs"]["publish-compilations"]
    condition = publisher["if"]
    replacements = {
        "!cancelled()": not cancelled,
        "github.event_name": event,
        "needs.select.outputs.count": "2",
        "needs.resolve-vulcan-config.result": "success",
        "needs.authorize-runner-classes.result": "success" if authorized else "failure",
        "needs.compile.result": compile_result,
    }
    for name, value in replacements.items():
        condition = condition.replace(name, repr(value))
    condition = condition.replace("&&", "and").replace("||", "or")
    assert eval(condition, {"__builtins__": {}}, {}) is expected
    assert "strategy" not in publisher
    assert publisher["needs"] == "compile"
    assert workflow["jobs"]["build-invocations"]["strategy"]["fail-fast"] == "false"
    download = next(step for step in publisher["steps"] if step.get("uses", "").startswith("actions/download-artifact"))
    assert download["with"]["name"] == (
        "compiler-output-${{ fromJSON(inputs.invocation).model_id }}-"
        "${{ fromJSON(inputs.invocation).build_profile_id }}-"
        "${{ fromJSON(inputs.invocation).compile_invocation_id }}"
    )
    assert "continue-on-error" not in download
    verify = next(step for step in publisher["steps"] if step["name"] == "Verify the complete component set")
    publish = next(step for step in publisher["steps"] if step["name"] == "Publish the complete component set")
    assert publisher["steps"].index(download) < publisher["steps"].index(verify) < publisher["steps"].index(publish)
    assert "validate-compile-outputs" in verify["run"]


def test_workflow_never_schedules_pull_requests_on_vulcan() -> None:
    workflow = _workflow()
    assert "pull_request_target" not in workflow["on"]
    compile_job = workflow["jobs"]["compile"]
    assert "github.event_name != 'pull_request'" in compile_job["if"]
    assert "model-compiler" in compile_job["runs-on"]
    assert "${{ fromJSON(inputs.selection).model_compiler_label }}" in compile_job["runs-on"]
    assert "${{ fromJSON(inputs.invocation).runner_class }}" in compile_job["runs-on"]
    assert "model-id-${{ fromJSON(inputs.invocation).model_id }}" in compile_job["runs-on"]
    assert "variant-id-${{ fromJSON(inputs.invocation).build_profile_id }}" in compile_job["runs-on"]
    assert all(not label.startswith("build-profile-id-") for label in compile_job["runs-on"])
    assert all(not label.startswith("component-id-") for label in compile_job["runs-on"])


def test_changed_models_compile_on_pushes_to_every_branch() -> None:
    workflow = _workflow()
    push_trigger = workflow["on"]["push"]
    assert "branches" not in push_trigger
    assert "branches-ignore" not in push_trigger
    assert "models/**" in push_trigger["paths"]
    assert "deps/**" in push_trigger["paths"]
    assert "training/yolo26_tiny_drone/**" in push_trigger["paths"]
    assert "training/yolo26_tiny_drone/**" in workflow["on"]["pull_request"]["paths"]
    assert "taxonomies/**" in push_trigger["paths"]


def test_model_compiler_is_resolved_before_requesting_a_runner() -> None:
    workflow = _workflow()
    select = workflow["jobs"]["select"]
    compile_job = workflow["jobs"]["compile"]
    compiler_step = next(step for step in select["steps"] if step.get("id") == "compiler")

    assert "model-registry resolve-compiler --repository ." in compiler_step["run"]
    assert select["outputs"]["model_compiler_version"] == "${{ steps.compiler.outputs.version }}"
    assert select["outputs"]["model_compiler_label"] == "${{ steps.compiler.outputs.label }}"
    assert compile_job["env"]["EXPECTED_COMPILER_VERSION"] == (
        "${{ fromJSON(inputs.selection).model_compiler_version }}"
    )
    assert compile_job["env"]["MODEL_COMPILER_LABEL"] == "${{ fromJSON(inputs.selection).model_compiler_label }}"


def test_sima_cli_install_retries_only_transient_package_manager_lock_failures() -> None:
    workflow = _workflow()
    install = next(
        step
        for step in workflow["jobs"]["compile"]["steps"]
        if step.get("name") == "Install dataset-registry from the main branch artifact"
    )
    script = install["run"]

    assert "for attempt in 1 2 3" in script
    assert "sima-cli-installer.log" in script
    assert "Unable to acquire the dpkg frontend lock" in script
    assert "Could not get lock .*(apt|dpkg)" in script
    assert "attempt == 3" in script
    assert "non-retryable error" in script
    assert 'sleep "$delay"' in script


def test_runner_classes_are_authorized_before_compile_matrix_is_queued() -> None:
    workflow = _workflow()
    admission = workflow["jobs"]["authorize-runner-classes"]

    assert admission["runs-on"] == "ubuntu-latest"
    assert admission["needs"] == ["select", "resolve-vulcan-config"]
    assert "github.event_name != 'pull_request'" in admission["if"]
    assert admission["permissions"]["id-token"] == "write"
    authorization = next(step for step in admission["steps"] if step.get("id") == "authorization")
    assert "authorize-runner-matrix" in authorization["run"]
    assert authorization["env"]["VULCAN_CONFIG_S3_URI"] == ("${{ needs.resolve-vulcan-config.outputs.config_s3_uri }}")
    assert workflow["jobs"]["build-invocations"]["needs"] == [
        "select",
        "resolve-vulcan-config",
        "authorize-runner-classes",
        "publish-catalog",
    ]
    assert workflow["jobs"]["build-invocations"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.authorize-runner-classes.outputs.matrix) }}"
    )


def test_catalog_taxonomy_is_validated_before_compiler_selection() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["select"]["steps"]
    validation_index = next(
        index for index, step in enumerate(steps) if step.get("name") == "Validate complete model catalog and taxonomy"
    )
    compiler_index = next(index for index, step in enumerate(steps) if step.get("id") == "compiler")

    assert steps[validation_index]["run"] == "model-registry validate-catalog --repository ."
    assert validation_index < compiler_index


def test_production_is_an_explicit_dispatch_choice() -> None:
    workflow = _workflow()
    options = workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["options"]
    assert options == ["staging", "production"]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["default"] == "staging"


def test_dispatch_can_request_full_mini_accuracy() -> None:
    workflow = _workflow()
    accuracy = workflow["on"]["workflow_dispatch"]["inputs"]["accuracy_samples"]

    assert accuracy["default"] == "500"
    assert accuracy["options"] == ["500", "5000"]
    assert workflow["jobs"]["compile"]["env"]["ACCURACY_SAMPLES"] == "${{ inputs.accuracy_samples || '500' }}"


def test_dispatch_can_select_expensive_variants_and_passes_private_asset_token() -> None:
    workflow = _workflow()
    assert "profiles" in workflow["on"]["workflow_dispatch"]["inputs"]
    assert "components" in workflow["on"]["workflow_dispatch"]["inputs"]
    compile_steps = workflow["jobs"]["compile"]["steps"]
    source_step = next(step for step in compile_steps if step.get("id") == "source")
    recipe_step = next(
        step for step in compile_steps if step.get("name") == "Run model-specific surgery and compilation recipe"
    )
    assert source_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert recipe_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert workflow["jobs"]["select"]["outputs"]["publishable"] == "${{ steps.matrix.outputs.publishable }}"


def test_dispatch_without_model_filters_is_the_explicit_full_rebuild_path() -> None:
    workflow = _workflow()
    matrix_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")

    assert 'if [[ "$EVENT_NAME" == "workflow_dispatch" ]]' in matrix_script
    assert 'if [[ -n "${REQUESTED_MODELS:-}" ]]' in matrix_script
    assert "model-registry matrix --repository . --all" in matrix_script
    assert workflow["on"]["workflow_dispatch"]["inputs"]["force_rebuild_all"]["type"] == "boolean"
    assert '"${FORCE_REBUILD_ALL:-false}" == "true"' in matrix_script


def test_json_catalog_and_package_publication_are_retired() -> None:
    workflow = _workflow()
    obsolete_jobs = {"package", "publish", "publish-latest-index", "promote-latest", "cleanup-github-artifacts"}
    assert obsolete_jobs.isdisjoint(workflow["jobs"])
    text = WORKFLOW.read_text(encoding="utf-8")
    for obsolete in ("latest.json", "latest.tag", "metadata.json", "models-dist", "previous-model-index"):
        assert obsolete not in text
    assert "vulcan-publish-artifacts" not in text
    assert "vulcan-update-latest-artifacts" not in text


def test_compiler_output_is_published_from_a_hosted_runner() -> None:
    workflow = _workflow()
    compile_job = workflow["jobs"]["compile"]
    publish_job = workflow["jobs"]["publish-compilations"]

    assert workflow["jobs"]["build-invocations"]["needs"] == [
        "select",
        "resolve-vulcan-config",
        "authorize-runner-classes",
        "publish-catalog",
    ]
    assert "id-token" not in compile_job["permissions"]
    assert all("aws-actions/configure-aws-credentials" not in step.get("uses", "") for step in compile_job["steps"])
    assert all("model_registry publish" not in step.get("run", "") for step in compile_job["steps"])
    compiler_output = next(
        step for step in compile_job["steps"] if step.get("name") == "Retain the complete compiler output"
    )
    assert compiler_output["with"]["name"].startswith("compiler-output-")

    assert publish_job["runs-on"] == "ubuntu-latest"
    assert publish_job["needs"] == "compile"
    assert publish_job["permissions"]["id-token"] == "write"
    download = next(
        step for step in publish_job["steps"] if step.get("name") == "Download this complete compiler output"
    )
    assert download["with"]["name"].startswith("compiler-output-")
    verify = next(step for step in publish_job["steps"] if step.get("name") == "Verify the complete component set")
    assert "validate-compile-outputs" in verify["run"]
    assert "compile-result.json" in verify["run"]
    credentials = next(
        step for step in publish_job["steps"] if step.get("name") == "Configure artifact-registry publisher credentials"
    )
    assert credentials["with"]["role-to-assume"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN }}"
    publish = next(step for step in publish_job["steps"] if step.get("name") == "Publish the complete component set")
    assert publish["env"]["REGISTRY_API_URL"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_API_URL }}"
    assert "model-registry publish" in publish["run"]
    assert "--dist-dir _compiled" in publish["run"]
    assert "path.write_text" not in publish["run"]
    published_output = next(
        step
        for step in publish_job["steps"]
        if step.get("name") == "Retain the published model output for benchmark handoff"
    )
    assert published_output["with"]["name"].startswith("compiled-model-")


def test_catalog_publication_does_not_require_a_model_compile() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["publish-catalog"]

    assert job["needs"] == ["select", "resolve-vulcan-config", "validation-gate"]
    assert "needs.select.outputs.count" not in job["if"]
    assert "needs.validation-gate.result == 'success'" in job["if"]
    assert job["permissions"]["id-token"] == "write"
    publish = next(
        step for step in job["steps"] if step.get("name") == "Publish and promote the complete catalog snapshot"
    )
    assert "model-registry publish-catalog" in publish["run"]
    assert publish["env"]["BASE_BRANCH"] == "${{ needs.select.outputs.base_branch }}"
    assert '--base-branch "$BASE_BRANCH"' in publish["run"]
    assert '--git-sha "$GITHUB_SHA"' in publish["run"]
    assert publish["env"]["REGISTRY_READ_API_URL"] == ("${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL }}")
    assert '--read-api-url "$REGISTRY_READ_API_URL"' in publish["run"]


def test_validation_only_changes_are_reported_and_block_catalog_publication() -> None:
    workflow = _workflow()
    select = workflow["jobs"]["select"]
    gate = workflow["jobs"]["validation-gate"]

    assert select["outputs"]["validation_count"] == "${{ steps.matrix.outputs.validation_count }}"
    assert select["outputs"]["validation_matrix"] == "${{ steps.matrix.outputs.validation_matrix }}"
    assert gate["needs"] == "select"
    assert gate["env"]["VALIDATION_COUNT"] == "${{ needs.select.outputs.validation_count }}"
    assert gate["env"]["VALIDATION_MATRIX"] == "${{ needs.select.outputs.validation_matrix }}"
    script = gate["steps"][0]["run"]
    assert "Validation-only changes require evaluation against retained MPKs" in script
    assert 'jq . <<< "$VALIDATION_MATRIX"' in script
    assert "exit 1" in script


def test_benchmark_only_changes_dispatch_against_durable_registry_artifacts() -> None:
    workflow = _workflow()
    select = workflow["jobs"]["select"]
    dispatch = workflow["jobs"]["dispatch-reused-benchmarks"]

    assert select["outputs"]["benchmark_matrix"] == "${{ steps.matrix.outputs.benchmark_matrix }}"
    assert dispatch["needs"] == ["select", "publish-catalog"]
    assert "needs.select.outputs.benchmark_count != '0'" in dispatch["if"]
    assert dispatch["strategy"]["matrix"] == "${{ fromJSON(needs.select.outputs.benchmark_matrix) }}"
    step = dispatch["steps"][0]
    assert "gh workflow run Vulcan-Model-Benchmarking.yml" in step["run"]
    assert "-f source_run_id=" in step["run"]


def test_changed_selection_does_not_fetch_or_carry_forward_a_previous_catalog() -> None:
    workflow = _workflow()
    select = workflow["jobs"]["select"]
    matrix_script = next(step["run"] for step in select["steps"] if step.get("id") == "matrix")
    assert all(step.get("id") != "previous" for step in select["steps"])
    assert "fetch-previous" not in matrix_script
    assert "--previous-index" not in matrix_script
    assert '--compiler-version "$COMPILER_VERSION"' in matrix_script


def test_feature_branch_pushes_always_use_merge_base_for_retryable_selection() -> None:
    workflow = _workflow()
    matrix_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")
    base_step = next(step for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "base")
    base_script = base_step["run"]

    assert "model-registry resolve-base" in base_script
    assert 'comparison_base="$merge_base"' in base_script
    assert base_step["env"]["EVENT_NAME"] == "${{ github.event_name }}"
    assert '[[ "$EVENT_NAME" == "push" && "$CURRENT_BRANCH" == "$DEFAULT_BRANCH" ]]' in base_script
    assert base_script.index('comparison_base="$merge_base"') < base_script.index(
        '[[ "$EVENT_NAME" == "push" && "$CURRENT_BRANCH" == "$DEFAULT_BRANCH" ]]'
    )
    assert "model-registry matrix --repository ." in matrix_script
    assert '--base "$COMPARISON_BASE" --head "$HEAD_SHA"' in matrix_script
    assert 'elif [[ "$BASE_SHA" =~ ^0+$ ]]' not in matrix_script


def test_develop_pushes_compare_only_the_push_event_range() -> None:
    workflow = _workflow()
    base_step = next(step for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "base")
    base_script = base_step["run"]

    assert base_step["env"]["DEFAULT_BRANCH"] == "develop"
    assert base_step["env"]["EVENT_BEFORE"] == "${{ github.event.before }}"
    develop_guard = '[[ "$EVENT_NAME" == "push" && "$CURRENT_BRANCH" == "$DEFAULT_BRANCH" ]]'
    assert develop_guard in base_script
    assert 'git cat-file -e "${EVENT_BEFORE}^{commit}"' in base_script
    assert 'comparison_base="$EVENT_BEFORE"' in base_script
    assert base_script.index(develop_guard) < base_script.index('comparison_base="$EVENT_BEFORE"')


def test_manual_dispatch_keeps_explicit_model_profile_and_component_selection() -> None:
    workflow = _workflow()
    matrix_step = next(step for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")
    matrix_script = matrix_step["run"]

    assert 'if [[ "$EVENT_NAME" == "workflow_dispatch" ]]' in matrix_script
    assert matrix_step["env"]["REQUESTED_MODELS"] == "${{ inputs.models }}"
    assert matrix_step["env"]["REQUESTED_PROFILES"] == "${{ inputs.profiles }}"
    assert matrix_step["env"]["REQUESTED_COMPONENTS"] == "${{ inputs.components }}"
    assert '--models "$REQUESTED_MODELS"' in matrix_script
    assert 'profile_args=(--profiles "$REQUESTED_PROFILES")' in matrix_script
    assert 'component_args=(--components "$REQUESTED_COMPONENTS")' in matrix_script


def test_resnet_recipe_is_python_and_uses_the_standard_cli_contract() -> None:
    recipe = ROOT / "models" / "resnet_50" / "compile.py"
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


def test_compile_recipe_runs_in_activated_model_compiler_shell() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    compile_script = next(
        step["run"] for step in steps if step.get("name") == "Run model-specific surgery and compilation recipe"
    )
    assert "bash --noprofile --norc" in compile_script
    assert "source /etc/profile.d/vulcan-model-compiler.sh" in compile_script
    assert "type activate-model-compiler" in compile_script
    assert "activate-model-compiler" in compile_script
    assert "command -v mla-masm" in compile_script
    assert 'archives=("$OUTPUT_DIR"/*_mpk.tar.gz)' in compile_script
    assert '--compile-invocation "$COMPILE_INVOCATION_ID"' in compile_script
    assert '--dataset "$CALIBRATION_PATH"' in compile_script
    assert 'if [[ "$CALIBRATION_KIND" == "dataset" ]]' in compile_script
    assert "python -m model_registry evaluate" not in compile_script


def test_verified_exporter_is_layered_on_the_model_compiler_interpreter() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    exporter = next(step for step in steps if step.get("name") == "Prepare the verified model exporter")

    assert "json.loads(sys.argv[3])" in exporter["run"]
    assert 'package["path"]' in exporter["run"]
    assert 'package["imports"]' in exporter["run"]
    assert '"--no-deps"' in exporter["run"]
    assert '"--target"' in exporter["run"]
    assert "--system-site-packages" not in exporter["run"]
    assert 'importlib.import_module("afe")' in exporter["run"]
    assert 'echo "PYTHONPATH=$exporter_packages' in exporter["run"]


def test_grouped_recipe_receives_the_declared_correctness_dataset() -> None:
    workflow = _workflow()
    recipe = next(
        step
        for step in workflow["jobs"]["compile"]["steps"]
        if step.get("name") == "Run model-specific surgery and compilation recipe"
    )

    assert recipe["env"]["CORRECTNESS_DATASET_PATH"] == "${{ steps.accuracy_dataset.outputs.path }}"
    assert 'correctness_args=(--correctness-dataset "$CORRECTNESS_DATASET_PATH")' in recipe["run"]


def test_grouped_compile_validates_and_verifies_each_component_output() -> None:
    workflow = _workflow()
    compile_job = workflow["jobs"]["compile"]
    verify = next(step for step in compile_job["steps"] if step.get("name") == "Validate and checksum every MPK")
    retain = next(step for step in compile_job["steps"] if step.get("uses") == "actions/upload-artifact@v7")

    assert "validate-compile-outputs" in verify["run"]
    assert 'for index in "${!components[@]}"' in verify["run"]
    assert 'verify-mpk "$PROFILE_OUTPUT_DIR/$component"' in verify["run"]
    assert retain["with"]["name"].startswith("compiler-output-")


def test_compile_workflow_resolves_registered_calibration_input() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    calibration = next(step for step in steps if step.get("id") == "calibration")
    assert "model_registry fetch-calibration" in calibration["run"]
    assert '--build-profile "$BUILD_PROFILE_ID"' in calibration["run"]
    assert 'get("kind", "none")' in calibration["run"]


def test_compile_workflow_resolves_accuracy_independently_of_calibration() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    accuracy_dataset = next(step for step in steps if step.get("id") == "accuracy_dataset")
    accuracy_step = next(step for step in steps if step.get("name") == "Measure model accuracy")

    assert "model_registry fetch-accuracy-dataset" in accuracy_dataset["run"]
    assert '--build-profile "$BUILD_PROFILE_ID"' in accuracy_dataset["run"]
    assert accuracy_dataset["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert (
        accuracy_step["if"]
        == "steps.accuracy_dataset.outputs.kind == 'dataset' && fromJSON(inputs.invocation).explicit_invocation != true"
    )
    assert accuracy_step["env"]["SOURCE_PATH"] == "${{ steps.source.outputs.path }}"
    assert accuracy_step["env"]["ACCURACY_DATASET_PATH"] == "${{ steps.accuracy_dataset.outputs.path }}"
    assert "source /etc/profile.d/vulcan-model-compiler.sh" in accuracy_step["run"]
    assert "activate-model-compiler" in accuracy_step["run"]
    assert "python -m model_registry evaluate" in accuracy_step["run"]
    assert '--dataset "$ACCURACY_DATASET_PATH"' in accuracy_step["run"]
    assert '--samples "$ACCURACY_SAMPLES"' in accuracy_step["run"]
    assert "--workers 6" in accuracy_step["run"]
    assert ".dataset.evaluated_samples" in accuracy_step["run"]
    assert ".evaluation.status" in accuracy_step["run"]


def test_compile_result_records_runner_authorization_provenance() -> None:
    workflow = _workflow()
    compile_job = workflow["jobs"]["compile"]
    verify = next(step for step in compile_job["steps"] if step.get("name") == "Validate and checksum every MPK")

    assert compile_job["env"]["REQUESTED_RUNNER_CLASS"] == "${{ fromJSON(inputs.invocation).requested_runner_class }}"
    assert compile_job["env"]["RUNNER_CLASS"] == "${{ fromJSON(inputs.invocation).runner_class }}"
    assert compile_job["env"]["RUNNER_APPROVAL_SPEC_REVISION"] == (
        "${{ fromJSON(inputs.invocation).runner_approval_spec_revision }}"
    )
    assert "--requested-runner-class" in verify["run"]
    assert "--runner-class" in verify["run"]
    assert "--runner-approval-id" in verify["run"]
    assert "--runner-approval-spec-revision" in verify["run"]


def test_registry_tool_install_does_not_replace_compiler_dependencies() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    install_script = next(
        step["run"]
        for step in steps
        if step.get("name") == "Install shared registry tooling outside the read-only compiler image"
    )
    assert "--no-deps" in install_script
    assert "import certifi, jsonschema, yaml" in install_script


def test_compile_worker_installs_main_dataset_registry_artifact() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    install_index, install = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Install dataset-registry from the main branch artifact"
    )
    calibration_index = next(index for index, step in enumerate(steps) if step.get("id") == "calibration")

    assert install_index < calibration_index
    assert install["env"]["DATASET_REGISTRY_ENVIRONMENT"] == "${{ fromJSON(inputs.selection).vulcan_environment }}"
    assert "sima-cli-installer.sh" in install["run"]
    assert 'sima_cli="${HOME}/.sima-cli/.venv/bin/sima-cli"' in install["run"]
    assert "dataset/dataset-registry@main:latest" in install["run"]
    assert "--download-only" in install["run"]
    assert "sima_neat_dataset_registry-*.whl" in install["run"]
    assert '--target "$dataset_tools"' in install["run"]
    assert '"$MODEL_COMPILER_PYTHON" -m dataset_registry --version' in install["run"]
    assert 'echo "PYTHONPATH=$dataset_tools${PYTHONPATH:+:$PYTHONPATH}"' in install["run"]


def test_ready_calibration_is_fetched_and_bound_to_compile_provenance() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["compile"]["steps"]
    calibration = next(step for step in steps if step.get("id") == "calibration")
    compile_step = next(
        step for step in steps if step.get("name") == "Run model-specific surgery and compilation recipe"
    )
    verify_step = next(step for step in steps if step.get("name") == "Validate and checksum every MPK")

    assert "model_registry fetch-calibration" in calibration["run"]
    assert calibration["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert compile_step["env"]["CALIBRATION_PATH"] == "${{ steps.calibration.outputs.path }}"
    assert compile_step["env"]["CALIBRATION_KIND"] == "${{ steps.calibration.outputs.kind }}"
    assert "--calibration-sha256" in verify_step["run"]


def test_published_compile_dispatches_one_grouped_benchmark() -> None:
    workflow = _workflow()
    publish_job = workflow["jobs"]["publish-compilations"]
    dispatch = next(step for step in publish_job["steps"] if step.get("name") == "Start the component benchmarks")

    assert publish_job["permissions"]["actions"] == "write"
    assert dispatch["continue-on-error"] == "true"
    assert "needs.select.outputs.publishable" not in dispatch["if"]
    assert "fromJSON(inputs.selection).vulcan_environment == 'staging'" in dispatch["if"]
    assert "fromJSON(inputs.invocation).target == 'modalix'" in dispatch["if"]
    assert "github.ref_name" not in dispatch["if"]
    assert "gh workflow run" not in dispatch["run"]
    assert 'method="POST"' in dispatch["run"]
    assert "/actions/workflows/Vulcan-Model-Benchmarking.yml/dispatches" in dispatch["run"]
    assert '"operation"' not in dispatch["run"]
    assert '"source_run_id": os.environ["GITHUB_RUN_ID"]' in dispatch["run"]
    assert '"source_sha": os.environ["GITHUB_SHA"]' in dispatch["run"]
    assert '"model_id": os.environ["BENCHMARK_MODEL_ID"]' in dispatch["run"]
    assert '"build_profile_id": os.environ["BENCHMARK_BUILD_PROFILE_ID"]' in dispatch["run"]
    dispatch_python = dispatch["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(dispatch_python, "<benchmark-dispatch>", "exec")
    retain_index = next(
        index
        for index, step in enumerate(publish_job["steps"])
        if step.get("name") == "Retain the published model output for benchmark handoff"
    )
    assert retain_index < publish_job["steps"].index(dispatch)
    assert "discover-benchmark-matrix" not in workflow["jobs"]
    assert "inputs.operation" not in workflow["concurrency"]["group"]
    assert "stage-dispatched-benchmark" not in workflow["jobs"]


def test_each_invocation_publishes_before_other_matrix_entries_finish() -> None:
    workflow = _workflow()
    parent = workflow["jobs"]["build-invocations"]
    assert parent["uses"] == "./.github/workflows/model-compile-invocation.yml"
    assert parent["with"]["invocation"] == "${{ toJSON(matrix) }}"
    assert "compile" not in parent["needs"]
    assert "strategy" not in workflow["jobs"]["compile"]
    publish = workflow["jobs"]["publish-compilations"]
    assert publish["needs"] == "compile"
    catalog_step = next(
        step for step in publish["steps"] if step.get("name") == "Publish this invocation in the catalog"
    )
    assert "--compiled-dir _compiled --incremental" in catalog_step["run"]
    assert "--initialize" not in catalog_step["run"]
    seed = workflow["jobs"]["publish-catalog"]
    assert "build-invocations" not in seed["needs"]
    assert any("--incremental --initialize" in step.get("run", "") for step in seed["steps"])
