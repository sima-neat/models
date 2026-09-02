from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    failures = []
    for document in markdown_files:
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            path = target.split("#", 1)[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            resolved = (document.parent / path).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not failures, "broken local Markdown links:\n" + "\n".join(failures)


def test_developer_index_covers_the_contribution_path() -> None:
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for guide in ("adding-a-model.md", "metadata-schema.md", "local-development.md", "contributing.md"):
        assert f"]({guide})" in text


def test_developer_sidebar_includes_every_article() -> None:
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")

    for article in sorted((ROOT / "docs").glob("*.md")):
        doc_id = article.stem
        assert f"'{doc_id}'" in sidebar, f"docs/{article.name} is missing from the developer sidebar"


def test_contributor_docs_distinguish_pr_checks_from_trusted_automation() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "Vulcan-Model-Compile.yml").read_text(encoding="utf-8")
    contributing_words = " ".join(contributing.split())

    assert "Pull requests do **not** run recipe or surgery code" in overview
    assert "## After merge: trusted staging automation" in contributing
    assert "A successful compile does not by itself mean the dispatched benchmark succeeded" in contributing_words
    assert "if: github.event_name != 'pull_request'" in workflow


def test_local_qualification_is_required_before_submission() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    overview_words = " ".join(overview.split())

    assert "submission prerequisites, not optional CI substitutes" in overview_words
    assert "complete the entire loop before submitting a model pull request" in local
    assert "Do not start submission until every supported variant" in contributing
    assert "checks not run locally" not in contributing


def test_contribution_guides_have_single_workflow_owners() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    authoring = (ROOT / "docs" / "adding-a-model.md").read_text(encoding="utf-8")
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    overview_words = " ".join(overview.split())

    assert "owns the tool installation and complete test sequence" in overview_words
    assert "sima-cli neat install --stg models-utils/model-registry" not in overview
    assert "model-registry matrix --repository" not in authoring
    assert "model-registry verify-mpk" not in authoring
    assert "does not repeat the local qualification procedure" in contributing
    assert "model-registry validate models/<model-id>" not in contributing
    assert "## Local-first qualification loop" in local


def test_core_page_titles_match_the_journey_navigation() -> None:
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    expected = {
        "adding-a-model.md": ("1", "Define the model contract"),
        "local-development.md": ("2", "Qualify locally"),
        "contributing.md": ("3", "Submit for review"),
    }

    for filename, (step, title) in expected.items():
        article = (ROOT / "docs" / filename).read_text(encoding="utf-8")
        assert article.startswith(f"# {title}\n")
        assert f"label: '{step}. {title}'" in sidebar
        assert f"guideLabel: '{title}'" in component


def test_dataset_guide_covers_artifactory_access_and_registration() -> None:
    text = (ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")

    for expected in (
        "## Set up Artifactory access first",
        "dataset-registry resolve imagenet-2012",
        'jq -e \'.provider == "artifactory"\'',
        "## Introduce a dataset or variant that does not exist",
        "catalog/datasets.yaml",
        "artifactory_sources",
        "Dataset pull-request CI is intentionally credential-free",
        "## Troubleshooting access",
    ):
        assert expected in text

    assert "Do not add a one-off download path to the model recipe" in text


def test_contribution_path_wraps_without_horizontal_scrolling() -> None:
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    css = (ROOT / "website" / "src" / "css" / "custom.css").read_text(encoding="utf-8")

    assert "<ol" in component
    assert "<svg" not in component
    assert "'six five four'" in css
    assert "@container (max-width: 700px)" in css
    assert "@container (max-width: 480px)" in css


def test_local_guide_passes_variant_calibration_to_the_recipe() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    assert "fetch-calibration \\" in text
    assert "models/<model-id> --variant <variant-id> --json" in text
    assert 'recipe_args=(--dataset "$calibration_path")' in text
    assert '"${recipe_args[@]}"' in text
    assert "## Measure declared accuracy locally" in text


def test_local_guide_uses_supported_validation_commands() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    assert "sima-cli neat install --stg models-utils/model-registry@develop:latest" in text
    assert "python -m pip install -e" not in text
    for command in (
        "model-registry validate ",
        "model-registry validate-catalog ",
        "model-registry fetch ",
        "model-registry fetch-calibration ",
        "model-registry matrix ",
        "model-registry resolve-compiler ",
        "model-registry verify-mpk ",
    ):
        assert command in text


def test_local_guide_covers_the_complete_local_qualification_loop() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    for expected in (
        "Compile locally from `model.yaml`",
        "Model Compiler Python APIs available",
        "compile.py --model <verified-source>",
        "model-registry verify-mpk",
        "Benchmark locally on a Modalix DevKit",
        "type dk || type devkit-run",
        "benchmarker \\",
        "performance-result.json",
        "Finish the local review",
    ):
        assert expected in text

    assert "Compilation is intentionally model-owned" in text
    assert "does not provide a public compile command" in text.replace("\n", " ")
    assert "commit either file" in text
    assert "Vulcan" not in text
    assert "What CI repeats" not in text
    assert "GitHub-hosted runner" not in text


def test_bounded_runner_documents_dynamic_image_geometry() -> None:
    text = (ROOT / "docs" / "benchmarking.md").read_text(encoding="utf-8")

    assert "original_width: 1280" in text
    assert "original_height: 720" in text
    assert "For image inputs with dynamic spatial dimensions" in text
    assert "Input geometry overrides are not supported" not in text


def test_docs_keep_author_tools_separate_from_ci_publication() -> None:
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    benchmarking = (ROOT / "docs" / "benchmarking.md").read_text(encoding="utf-8")

    assert "models-utils/model-registry@develop:latest" in local
    assert "Compilation is intentionally model-owned" in local
    assert "Model authors do not publish benchmark results directly" in benchmarking
    assert "reserved for CI" in contributing
    assert "git push --set-upstream origin HEAD" in contributing
    assert "authors do not invoke those publication steps themselves" in contributing

    author_guides = local + benchmarking
    for command in (
        "model-registry publish ",
        "model-registry publish-benchmark ",
        "model-registry publish-taxonomy ",
        "model-registry package ",
    ):
        assert command not in author_guides
