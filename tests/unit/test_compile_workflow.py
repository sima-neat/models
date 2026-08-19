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


def test_changed_models_compile_on_pushes_to_every_branch() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    push_trigger = workflow["on"]["push"]
    assert "branches" not in push_trigger
    assert "branches-ignore" not in push_trigger
    assert "models/**" in push_trigger["paths"]
    assert "deps/**" in push_trigger["paths"]
    assert "training/yolo26_tiny_drone/**" in push_trigger["paths"]
    assert "training/yolo26_tiny_drone/**" in workflow["on"]["pull_request"]["paths"]


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


def test_production_is_an_explicit_dispatch_choice() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    options = workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["options"]
    assert options == ["staging", "production"]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["default"] == "staging"


def test_dispatch_can_select_expensive_variants_and_passes_private_asset_token() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert "variants" in workflow["on"]["workflow_dispatch"]["inputs"]
    compile_steps = workflow["jobs"]["compile"]["steps"]
    source_step = next(step for step in compile_steps if step.get("id") == "source")
    recipe_step = next(
        step for step in compile_steps if step.get("name") == "Run model-specific surgery and compilation recipe"
    )
    assert source_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert recipe_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert workflow["jobs"]["select"]["outputs"]["publishable"] == "${{ steps.matrix.outputs.publishable }}"
    assert "needs.select.outputs.publishable == 'true'" in workflow["jobs"]["package"]["if"]


def test_dispatch_without_model_filters_is_the_explicit_full_rebuild_path() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    matrix_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")

    assert 'if [[ "$EVENT_NAME" == "workflow_dispatch" ]]' in matrix_script
    assert 'if [[ -n "${REQUESTED_MODELS:-}" ]]' in matrix_script
    assert "model-registry matrix --repository . --all" in matrix_script


def test_publication_uses_standard_branch_manifest_and_latest_workflows() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert (
        workflow["jobs"]["publish"]["uses"] == "sima-neat/.github/.github/workflows/vulcan-publish-artifacts.yml@main"
    )
    assert workflow["jobs"]["publish"]["with"]["artifact_folder"] == "."
    assert workflow["jobs"]["publish"]["with"]["publish_manifest"] == "true"
    assert workflow["jobs"]["publish"]["with"]["cleanup_github_artifacts"] == "false"
    assert (
        workflow["jobs"]["promote-latest"]["uses"]
        == "sima-neat/.github/.github/workflows/vulcan-update-latest-artifacts.yml@main"
    )
    assert "latest_tag" not in workflow["jobs"]["promote-latest"]["with"]
    assert workflow["jobs"]["promote-latest"]["with"]["validate_manifest"] == "true"
    assert workflow["jobs"]["publish-latest-index"]["needs"] == [
        "select",
        "package",
        "publish",
        "resolve-vulcan-config",
    ]
    assert workflow["jobs"]["promote-latest"]["needs"] == ["select", "publish-latest-index", "resolve-vulcan-config"]
    cleanup = workflow["jobs"]["cleanup-github-artifacts"]
    assert cleanup["needs"] == ["publish", "publish-latest-index", "promote-latest"]
    assert "needs.publish-latest-index.result == 'success'" in cleanup["if"]
    assert "needs.promote-latest.result == 'success'" in cleanup["if"]
    assert "actions: write" in WORKFLOW.read_text(encoding="utf-8")
    assert "actions/artifacts/$artifact_id" in cleanup["steps"][0]["run"]
    upload_script = workflow["jobs"]["publish-latest-index"]["steps"][2]["run"]
    assert "dist/latest.json" in upload_script
    assert "--cache-control no-store" in upload_script


def test_feature_branch_publication_can_seed_a_partial_catalog() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    package_script = next(
        step["run"]
        for step in workflow["jobs"]["package"]["steps"]
        if step.get("name") == "Generate root and per-model package metadata"
    )

    assert 'if [[ "$GITHUB_REF_NAME" != "main" ]]' in package_script
    assert "package_args+=(--allow-partial-catalog)" in package_script


def test_staging_publication_records_new_compilations_in_registry() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    compile_job = workflow["jobs"]["compile"]

    assert compile_job["needs"] == ["select", "resolve-vulcan-config"]
    assert compile_job["permissions"]["id-token"] == "write"
    credentials = next(
        step for step in compile_job["steps"] if step.get("name") == "Configure artifact-registry publisher credentials"
    )
    assert "needs.select.outputs.publishable == 'true'" in credentials["if"]
    assert "needs.select.outputs.vulcan_environment == 'staging'" in credentials["if"]
    assert credentials["with"]["role-to-assume"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN }}"
    publish = next(step for step in compile_job["steps"] if step.get("name") == "Publish this model compilation")
    assert publish["env"]["REGISTRY_API_URL"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_API_URL }}"
    assert "-m model_registry publish" in publish["run"]
    assert '--dist-dir "$ARTIFACT_ROOT"' in publish["run"]
    assert 'result["count"] == 1' in publish["run"]
    publish_index = compile_job["steps"].index(publish)
    retain_index = next(
        index
        for index, step in enumerate(compile_job["steps"])
        if step.get("name") == "Retain the validated model output for packaging"
    )
    assert publish_index < retain_index


def test_shared_latest_metadata_waits_for_the_complete_matrix() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["jobs"]["package"]["needs"] == ["select", "compile"]
    assert "needs.compile.result == 'success'" in workflow["jobs"]["package"]["if"]
    assert "publish-registry" not in workflow["jobs"]


def test_new_branches_carry_forward_the_published_develop_catalog() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    previous_script = next(
        step["run"]
        for step in workflow["jobs"]["select"]["steps"]
        if step.get("name") == "Resolve immutable previous model catalog"
    )
    assert '--fallback-branch "$FALLBACK_BRANCH"' in previous_script
    previous_step = next(step for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "previous")
    assert "steps.base.outputs.base_branch" in previous_step["env"]["FALLBACK_BRANCH"]


def test_new_branch_uses_merge_base_instead_of_full_catalog_fallback() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    matrix_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")
    base_script = next(step["run"] for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "base")

    assert "model-registry resolve-base" in base_script
    assert 'comparison_base="$merge_base"' in base_script
    assert "model-registry matrix --repository ." in matrix_script
    assert '--base "$COMPARISON_BASE" --head "$HEAD_SHA"' in matrix_script
    assert 'elif [[ "$BASE_SHA" =~ ^0+$ ]]' not in matrix_script


def test_zero_compile_selection_still_packages_and_publishes_catalog() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    package = workflow["jobs"]["package"]
    steps = package["steps"]

    assert "needs.compile.result == 'skipped'" in package["if"]
    assert "needs.select.outputs.count == '0'" in package["if"]
    assert next(step for step in steps if step.get("name") == "Prepare empty compiled-output directory")
    download = next(step for step in steps if step.get("name") == "Download compiled model outputs")
    assert download["if"] == "needs.select.outputs.count != '0'"
    assert workflow["jobs"]["publish"]["with"]["min_artifact_count"] == "3"


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
    assert {"--model", "--output-dir", "--variant"} <= option_strings


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
    assert "--calibration-sha256" in verify_step["run"]
