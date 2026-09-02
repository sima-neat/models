from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Compile.yml"


def test_workflow_never_schedules_pull_requests_on_vulcan() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert "pull_request_target" not in workflow["on"]
    compile_job = workflow["jobs"]["compile"]
    assert "github.event_name != 'pull_request'" in compile_job["if"]
    assert "model-compiler" in compile_job["runs-on"]
    assert "${{ needs.select.outputs.model_compiler_label }}" in compile_job["runs-on"]
    assert "${{ matrix.runner_class }}" in compile_job["runs-on"]
    assert "model-id-${{ matrix.model_id }}" in compile_job["runs-on"]
    assert "build-profile-id-${{ matrix.build_profile_id }}" in compile_job["runs-on"]
    assert "component-id-${{ matrix.component_id }}" in compile_job["runs-on"]


def test_changed_models_compile_on_pushes_to_every_branch() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    push_trigger = workflow["on"]["push"]
    assert "branches" not in push_trigger
    assert "branches-ignore" not in push_trigger
    assert "models/**" in push_trigger["paths"]
    assert "deps/**" in push_trigger["paths"]
    assert "training/yolo26_tiny_drone/**" in push_trigger["paths"]
    assert "training/yolo26_tiny_drone/**" in workflow["on"]["pull_request"]["paths"]
    assert "taxonomies/**" in push_trigger["paths"]


def test_model_compiler_is_resolved_before_requesting_a_runner() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    select = workflow["jobs"]["select"]
    compile_job = workflow["jobs"]["compile"]
    compiler_step = next(step for step in select["steps"] if step.get("id") == "compiler")

    assert "model-registry resolve-compiler --repository ." in compiler_step["run"]
    assert select["outputs"]["model_compiler_version"] == "${{ steps.compiler.outputs.version }}"
    assert select["outputs"]["model_compiler_label"] == "${{ steps.compiler.outputs.label }}"
    assert compile_job["env"]["EXPECTED_COMPILER_VERSION"] == ("${{ needs.select.outputs.model_compiler_version }}")
    assert compile_job["env"]["MODEL_COMPILER_LABEL"] == "${{ needs.select.outputs.model_compiler_label }}"


def test_runner_classes_are_authorized_before_compile_matrix_is_queued() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    admission = workflow["jobs"]["authorize-runner-classes"]
    compile_job = workflow["jobs"]["compile"]

    assert admission["runs-on"] == "ubuntu-latest"
    assert admission["needs"] == ["select", "resolve-vulcan-config"]
    assert "github.event_name != 'pull_request'" in admission["if"]
    assert admission["permissions"]["id-token"] == "write"
    authorization = next(step for step in admission["steps"] if step.get("id") == "authorization")
    assert "authorize-runner-matrix" in authorization["run"]
    assert authorization["env"]["VULCAN_CONFIG_S3_URI"] == (
        "${{ needs.resolve-vulcan-config.outputs.config_s3_uri }}"
    )
    assert compile_job["needs"] == ["select", "resolve-vulcan-config", "authorize-runner-classes"]
    assert compile_job["strategy"]["matrix"] == (
        "${{ fromJSON(needs.authorize-runner-classes.outputs.matrix) }}"
    )


def test_catalog_taxonomy_is_validated_before_compiler_selection() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["select"]["steps"]
    validation_index = next(
        index for index, step in enumerate(steps) if step.get("name") == "Validate complete model catalog and taxonomy"
    )
    compiler_index = next(index for index, step in enumerate(steps) if step.get("id") == "compiler")

    assert steps[validation_index]["run"] == "model-registry validate-catalog --repository ."
    assert validation_index < compiler_index


def test_production_is_an_explicit_dispatch_choice() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    options = workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["options"]
    assert options == ["staging", "production"]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["default"] == "staging"


def test_dispatch_can_request_full_mini_accuracy() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    accuracy = workflow["on"]["workflow_dispatch"]["inputs"]["accuracy_samples"]

    assert accuracy["default"] == "500"
    assert accuracy["options"] == ["500", "5000"]
    assert workflow["jobs"]["compile"]["env"]["ACCURACY_SAMPLES"] == "${{ inputs.accuracy_samples || '500' }}"


def test_dispatch_can_select_expensive_variants_and_passes_private_asset_token() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
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
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    matrix_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")

    assert 'if [[ "$EVENT_NAME" == "workflow_dispatch" ]]' in matrix_script
    assert 'if [[ -n "${REQUESTED_MODELS:-}" ]]' in matrix_script
    assert "model-registry matrix --repository . --all" in matrix_script


def test_json_catalog_and_package_publication_are_retired() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    obsolete_jobs = {"package", "publish", "publish-latest-index", "promote-latest", "cleanup-github-artifacts"}
    assert obsolete_jobs.isdisjoint(workflow["jobs"])
    text = WORKFLOW.read_text(encoding="utf-8")
    for obsolete in ("latest.json", "latest.tag", "metadata.json", "models-dist", "previous-model-index"):
        assert obsolete not in text
    assert "vulcan-publish-artifacts" not in text
    assert "vulcan-update-latest-artifacts" not in text


def test_each_successful_matrix_item_publishes_directly_to_registry() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    compile_job = workflow["jobs"]["compile"]

    assert compile_job["needs"] == ["select", "resolve-vulcan-config", "authorize-runner-classes"]
    assert compile_job["permissions"]["id-token"] == "write"
    credentials = next(
        step for step in compile_job["steps"] if step.get("name") == "Configure artifact-registry publisher credentials"
    )
    assert "if" not in credentials
    assert credentials["with"]["role-to-assume"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN }}"
    publish = next(step for step in compile_job["steps"] if step.get("name") == "Publish this model compilation")
    assert "if" not in publish
    assert publish["env"]["REGISTRY_API_URL"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_API_URL }}"
    assert "-m model_registry publish" in publish["run"]
    assert '--dist-dir "$ARTIFACT_ROOT"' in publish["run"]
    assert 'result["count"] == 1' in publish["run"]
    publish_index = compile_job["steps"].index(publish)
    retain_index = next(
        index
        for index, step in enumerate(compile_job["steps"])
        if step.get("name") == "Retain the validated model output for benchmark handoff"
    )
    assert publish_index < retain_index


def test_taxonomy_publication_does_not_require_a_model_compile() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["publish-taxonomy"]

    assert job["needs"] == ["select", "resolve-vulcan-config"]
    assert "needs.select.outputs.count" not in job["if"]
    assert "needs.select.outputs.vulcan_environment == 'staging'" in job["if"]
    assert job["permissions"]["id-token"] == "write"
    publish = next(step for step in job["steps"] if step.get("name") == "Publish canonical taxonomy")
    assert "model-registry publish-taxonomy" in publish["run"]


def test_changed_selection_does_not_fetch_or_carry_forward_a_previous_catalog() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    select = workflow["jobs"]["select"]
    matrix_script = next(step["run"] for step in select["steps"] if step.get("id") == "matrix")
    assert all(step.get("id") != "previous" for step in select["steps"])
    assert "fetch-previous" not in matrix_script
    assert "--previous-index" not in matrix_script
    assert '--compiler-version "$COMPILER_VERSION"' in matrix_script


def test_new_branch_uses_merge_base_instead_of_full_catalog_fallback() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    matrix_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")
    base_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "base")

    assert "model-registry resolve-base" in base_script
    assert 'comparison_base="$merge_base"' in base_script
    assert 'git cat-file -e "${EVENT_BEFORE}^{commit}"' in base_script
    assert "model-registry matrix --repository ." in matrix_script
    assert '--base "$COMPARISON_BASE" --head "$HEAD_SHA"' in matrix_script
    assert 'elif [[ "$BASE_SHA" =~ ^0+$ ]]' not in matrix_script


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
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
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
    assert '--dataset "$CALIBRATION_PATH"' in compile_script
    assert 'if [[ "$CALIBRATION_KIND" == "dataset" ]]' in compile_script
    assert "python -m model_registry evaluate" not in compile_script


def test_compile_workflow_resolves_registered_calibration_input() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["compile"]["steps"]
    calibration = next(step for step in steps if step.get("id") == "calibration")
    assert "model_registry fetch-calibration" in calibration["run"]
    assert '--build-profile "$BUILD_PROFILE_ID"' in calibration["run"]
    assert 'get("kind", "none")' in calibration["run"]


def test_compile_workflow_resolves_accuracy_independently_of_calibration() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["compile"]["steps"]
    accuracy_dataset = next(step for step in steps if step.get("id") == "accuracy_dataset")
    accuracy_step = next(step for step in steps if step.get("name") == "Measure model accuracy")

    assert "model_registry fetch-accuracy-dataset" in accuracy_dataset["run"]
    assert '--build-profile "$BUILD_PROFILE_ID"' in accuracy_dataset["run"]
    assert accuracy_dataset["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert accuracy_step["if"] == "steps.accuracy_dataset.outputs.kind == 'dataset'"
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
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    compile_job = workflow["jobs"]["compile"]
    verify = next(step for step in compile_job["steps"] if step.get("id") == "mpk")

    assert compile_job["env"]["REQUESTED_RUNNER_CLASS"] == "${{ matrix.requested_runner_class }}"
    assert compile_job["env"]["RUNNER_CLASS"] == "${{ matrix.runner_class }}"
    assert compile_job["env"]["RUNNER_APPROVAL_SPEC_REVISION"] == (
        "${{ matrix.runner_approval_spec_revision }}"
    )
    assert "--requested-runner-class" in verify["run"]
    assert "--runner-class" in verify["run"]
    assert "--runner-approval-id" in verify["run"]
    assert "--runner-approval-spec-revision" in verify["run"]


def test_registry_tool_install_does_not_replace_compiler_dependencies() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["compile"]["steps"]
    install_script = next(
        step["run"]
        for step in steps
        if step.get("name") == "Install shared registry tooling outside the read-only compiler image"
    )
    assert "--no-deps" in install_script
    assert "import certifi, jsonschema, yaml" in install_script


def test_compile_worker_installs_main_dataset_registry_artifact() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["compile"]["steps"]
    install_index, install = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Install dataset-registry from the main branch artifact"
    )
    calibration_index = next(index for index, step in enumerate(steps) if step.get("id") == "calibration")

    assert install_index < calibration_index
    assert install["env"]["DATASET_REGISTRY_ENVIRONMENT"] == "${{ needs.select.outputs.vulcan_environment }}"
    assert "sima-cli-installer.sh" in install["run"]
    assert 'sima_cli="${HOME}/.sima-cli/.venv/bin/sima-cli"' in install["run"]
    assert "dataset/dataset-registry@main:latest" in install["run"]
    assert "--download-only" in install["run"]
    assert "sima_neat_dataset_registry-*.whl" in install["run"]
    assert '--target "$dataset_tools"' in install["run"]
    assert '"$MODEL_COMPILER_PYTHON" -m dataset_registry --version' in install["run"]
    assert 'echo "PYTHONPATH=$dataset_tools${PYTHONPATH:+:$PYTHONPATH}"' in install["run"]


def test_ready_calibration_is_fetched_and_bound_to_compile_provenance() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["compile"]["steps"]
    calibration = next(step for step in steps if step.get("id") == "calibration")
    compile_step = next(
        step for step in steps if step.get("name") == "Run model-specific surgery and compilation recipe"
    )
    verify_step = next(step for step in steps if step.get("name") == "Validate and checksum the MPK")

    assert "model_registry fetch-calibration" in calibration["run"]
    assert calibration["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert compile_step["env"]["CALIBRATION_PATH"] == "${{ steps.calibration.outputs.path }}"
    assert compile_step["env"]["CALIBRATION_KIND"] == "${{ steps.calibration.outputs.kind }}"
    assert "--calibration-sha256" in verify_step["run"]


def test_each_successful_compile_dispatches_its_benchmark_immediately() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    compile_job = workflow["jobs"]["compile"]
    dispatch = next(
        step
        for step in compile_job["steps"]
        if step.get("name") == "Start this model benchmark without waiting for the compile matrix"
    )

    assert compile_job["permissions"]["actions"] == "write"
    assert dispatch["continue-on-error"] == "true"
    assert "needs.select.outputs.publishable" not in dispatch["if"]
    assert "needs.select.outputs.vulcan_environment == 'staging'" in dispatch["if"]
    assert "matrix.target == 'modalix'" in dispatch["if"]
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
        for index, step in enumerate(compile_job["steps"])
        if step.get("name") == "Retain the validated model output for benchmark handoff"
    )
    assert retain_index < compile_job["steps"].index(dispatch)
    assert "discover-benchmark-matrix" not in workflow["jobs"]
    assert "inputs.operation" not in workflow["concurrency"]["group"]
    assert "stage-dispatched-benchmark" not in workflow["jobs"]
