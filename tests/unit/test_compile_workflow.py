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
    assert "${{ matrix.compiler_label }}" in compile_job["runs-on"]


def test_changed_models_compile_on_pushes_to_every_branch() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    push_trigger = workflow["on"]["push"]
    assert "branches" not in push_trigger
    assert "branches-ignore" not in push_trigger
    assert "models/**" in push_trigger["paths"]


def test_production_is_an_explicit_dispatch_choice() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    options = workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["options"]
    assert options == ["staging", "production"]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["environment"]["default"] == "staging"


def test_publication_uses_standard_branch_manifest_and_latest_workflows() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert (
        workflow["jobs"]["publish"]["uses"] == "sima-neat/.github/.github/workflows/vulcan-publish-artifacts.yml@main"
    )
    assert workflow["jobs"]["publish"]["with"]["artifact_folder"] == "."
    assert workflow["jobs"]["publish"]["with"]["publish_manifest"] == "true"
    assert (
        workflow["jobs"]["promote-latest"]["uses"]
        == "sima-neat/.github/.github/workflows/vulcan-update-latest-artifacts.yml@main"
    )
    assert "latest_tag" not in workflow["jobs"]["promote-latest"]["with"]
    assert workflow["jobs"]["promote-latest"]["with"]["validate_manifest"] == "true"
    assert workflow["jobs"]["publish-latest-index"]["needs"] == [
        "select", "package", "publish", "resolve-vulcan-config"
    ]
    assert workflow["jobs"]["promote-latest"]["needs"] == [
        "select", "publish-latest-index", "resolve-vulcan-config"
    ]
    upload_script = workflow["jobs"]["publish-latest-index"]["steps"][2]["run"]
    assert "dist/latest.json" in upload_script
    assert "--cache-control no-store" in upload_script


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
        step["run"] for step in steps
        if step.get("name") == "Run model-specific surgery and compilation recipe"
    )
    assert 'bash --noprofile --rcfile "${HOME}/.bashrc" -i' in compile_script
    assert "type activate-model-compiler" in compile_script
    assert "activate-model-compiler" in compile_script
    assert "command -v mla-masm" in compile_script
    assert 'archives=("$OUTPUT_DIR"/*_mpk.tar.gz)' in compile_script


def test_registry_tool_install_does_not_replace_compiler_dependencies() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["compile"]["steps"]
    install_script = next(
        step["run"] for step in steps
        if step.get("name") == "Install shared registry tooling outside the read-only compiler image"
    )
    assert "--no-deps" in install_script
    assert "import certifi, jsonschema, yaml" in install_script
