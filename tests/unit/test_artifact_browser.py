from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
BROWSER = ROOT / "artifact-browser"
DEPLOY_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-model-browser.yml"


def test_artifact_browser_assets_exist() -> None:
    for name in (
        "index.html",
        "app.js",
        "benchmark-adapters.js",
        "styles.css",
        "config.js",
        "favicon.svg",
        "README.md",
    ):
        assert (BROWSER / name).is_file()


def test_browser_uses_the_official_sima_faceted_mark() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    icon = (BROWSER / "favicon.svg").read_text(encoding="utf-8")

    assert '<link rel="icon" type="image/svg+xml" href="favicon.svg"' in html
    assert '<img class="brand-mark" src="favicon.svg"' in html
    for color in ("#D26728", "#5998DD", "#D1CEE6", "#A4CC2D", "#2A9C4F", "#C0E8DB", "#010F0E"):
        assert color in icon


def test_browser_calls_registry_directly_without_backend_proxy() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    config = (BROWSER / "config.js").read_text(encoding="utf-8")

    assert 'src="config.js"' in html
    assert "MODEL_REGISTRY_CONFIG" in script
    assert "https://models-registry.stg.neat.sima.ai" in config
    assert "fetch(`${state.apiBase}${path}`" in script
    assert "fetch(`/api${path}`" not in script
    assert not (BROWSER / "serve.py").exists()


def test_browser_links_home_relative_to_the_deployed_subpath() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")

    assert 'class="brand" href="./"' in html
    assert 'class="brand" href="/"' not in html


def test_browser_redirects_the_extensionless_models_route_before_loading_assets() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")

    redirect = 'if (location.pathname === "/models")'
    assert redirect in html
    assert "location.replace(`/models/${location.search}${location.hash}`)" in html
    assert html.index(redirect) < html.index('href="styles.css"')
    assert html.index(redirect) < html.index('src="config.js"')


def test_staging_deployment_publishes_models_to_docs_origin() -> None:
    workflow = yaml.load(DEPLOY_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["on"]["push"]["branches"] == ["main", "develop", "feat/artifact-browser"]
    assert workflow["jobs"]["resolve-vulcan-config"]["with"]["vulcan_env"] == "staging"

    publish = workflow["jobs"]["publish"]["with"]
    assert publish["bucket"] == "${{ needs.resolve-vulcan-config.outputs.docs_bucket }}"
    assert publish["role_to_assume"] == "${{ needs.resolve-vulcan-config.outputs.docs_publisher_role_arn }}"
    assert publish["cloudfront_distribution_id"] == (
        "${{ needs.resolve-vulcan-config.outputs.sysdoc_cloudfront_distribution_id }}"
    )
    assert publish["s3_prefix"] == "models"
    assert publish["cloudfront_paths"] == "/models /models/*"


def test_staging_deployment_fingerprints_mutable_browser_assets() -> None:
    workflow = yaml.load(DEPLOY_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assemble = next(
        step["run"]
        for step in workflow["jobs"]["build"]["steps"]
        if step.get("name") == "Assemble static site"
    )

    assert 'revision="${GITHUB_SHA:0:12}"' in assemble
    for asset in ("app", "benchmark-adapters", "styles", "config"):
        assert f"{asset}.${{revision}}" in assemble
    assert "sed -i" in assemble


def test_catalog_avoids_per_run_enrichment_and_uses_local_cache() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "enrichRuns" not in script
    assert "mapConcurrent" not in script
    assert "readCachedCatalog" in script
    assert "writeCachedCatalog" in script
    assert "state.runs[0]?.id" not in script


def test_concurrent_requests_do_not_overwrite_newer_selections() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "const requestId = ++state.runsRequestId" in script
    assert "if (requestId !== state.runsRequestId) return" in script
    assert "if (state.selectedId === runId)" in script


def test_linked_run_loads_independently_of_the_current_catalog_page() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "if (state.selectedId) await selectRun(state.selectedId, { updateUrl: false });" in script
    assert "state.runs.some((run) => run.id === state.selectedId)" not in script


def test_detail_error_can_be_retried_and_closed() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'errorPanel(error.message, "Could not load this build", { closable: true })' in script
    assert "event.target.closest('[data-action=\"retry\"]') && state.selectedId" in script
    assert "selectRun(state.selectedId, { updateUrl: false });" in script


def test_branch_filter_defaults_to_main_and_persists_user_selection() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'const DEFAULT_BRANCH = "main"' in script
    assert "sima-neat-model-branch:" in script
    assert "readBranchPreference" in script
    assert "writeBranchPreference(state.preferredBranch)" in script
    assert "branches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH" in script
    assert 'knownBranches.has(DEFAULT_BRANCH) ? DEFAULT_BRANCH : ""' in script
    assert "limit: fallbackBranch ? 50 : 200" in script


def test_earlier_builds_pagination_tracks_the_selected_branch() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'query.set("git_ref", `refs/heads/${branch}`)' in script
    assert "availableBranches" in script
    assert "await loadRuns({ branch: state.preferredBranch })" in script
    assert 'elements["load-more"].hidden = !state.hasMore' in script


def test_browser_has_persistent_dual_theme_and_accessibility_controls() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'data-theme="dark"' in styles
    assert "prefers-color-scheme: dark" in html
    assert "sima-neat-artifact-theme" in html
    assert "sima-neat-artifact-theme" in script
    assert "Storage is optional; use the system preference when unavailable." in html
    assert "Storage may be disabled; theme selection still works for this page load." in script
    assert 'class="skip-link"' in html
    assert 'aria-live="polite"' in html
    assert "prefers-reduced-motion: reduce" in styles


def test_browser_uses_refined_search_first_layout() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "Manrope:wght@400;500;600;700" in html
    assert "IBM+Plex+Mono" in html
    assert 'id="page-title">Find a model' in html
    assert "Search models, tasks, or architectures" in html
    assert 'class="page-header"' in html
    assert "Trace every" not in html
    assert "--font-display" not in styles
    assert "font-family: var(--font-body)" in styles


def test_secondary_filters_are_collapsed_and_status_is_not_presented() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'id="filter-toggle"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="filter-drawer" class="filter-drawer" hidden' in html
    assert 'id="filter-count"' in html
    assert 'id="status-filter"' not in html
    assert 'class="run-status' not in script
    assert 'class="status-label' not in script
    assert 'query.set("status", "completed")' in script
    assert 'run.status === "completed"' in script


def test_latest_model_build_per_compiler_is_the_default_with_history_opt_in() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="history-filter" type="checkbox" role="switch"' in html
    assert "Latest per compiler is shown by default." in html
    assert 'query.set("run_type", "model_compile")' in script
    assert 'if (!elements["history-filter"].checked) query.set("latest_per_model_compiler", "true")' in script
    assert 'initialQuery.get("history") === "1"' in script
    assert 'url.searchParams.set("history", "1")' in script
    assert 'elements["history-filter"].checked = false' in script
    assert ".switch-row input:checked + i" in styles


def test_browser_loads_the_published_taxonomy_and_exposes_category_facets() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "model-type-filter",
        "task-filter",
        "architecture-filter",
        "paradigm-filter",
        "input-modality-filter",
    ):
        assert f'id="{element_id}"' in html
    assert "/taxonomies/latest" in script
    assert "payload.taxonomy?.document" in script
    assert '"task-filter": "task"' in script
    assert '"architecture-filter": "architecture_id"' in script
    assert "setCategoryFiltersDisabled(true)" in script
    assert "Category filters are temporarily unavailable" in script


def test_category_filters_are_server_side_and_shareable() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "query.set(name, value)" in script
    assert "url.searchParams.set(queryName, value)" in script
    assert "url.searchParams.delete(queryName)" in script
    assert "run.architecture_id === current.architecture_id" in script
    assert "(run.tasks || []).includes(current.task)" in script


def test_smart_search_uses_names_and_taxonomy_labels_on_client_and_server() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "normalizedSearchTerms" in script
    assert 'taxonomyLabel("tasks", id)' in script
    assert "terms.every((term) => searchableText.includes(term))" in script
    assert 'query.set("q", elements.search.value.trim())' in script
    assert 'url.searchParams.set("q", query)' in script
    assert "setTimeout(() => loadRuns" in script


def test_model_card_links_to_the_exact_compiled_github_source() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "function githubModelUrl(run, modelId)" in script
    assert 'const runRepository = String(run.repository || "").toLowerCase()' in script
    assert "runRepository !== configuredRepository" in script
    assert "/tree/${commit}/models/${encodeURIComponent(modelId)}`" in script
    assert r"/^[a-f0-9]{40}$/i.test(commit)" in script
    assert 'class="model-source-link"' in script
    assert "View ${model} source on GitHub at commit ${shortSha(run.commit_sha)}" in script
    assert 'target="_blank" rel="noreferrer"' in script
    assert ".model-source-link:focus-visible" in styles


def test_benchmark_rendering_is_schema_versioned_and_charted() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    adapters = (BROWSER / "benchmark-adapters.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'import { adaptBenchmarkResult } from "./benchmark-adapters.js"' in script
    assert "new Map([[1, adaptV1], [2, adaptV2]])" in adapters
    assert "/benchmarks/latest" in script
    assert 'class="trial-chart"' in script
    assert 'role="img"' in script
    assert 'class="sr-only"' in script
    assert ".chart-points circle" in styles


def test_detail_layout_can_focus_the_selected_model() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="workspace"' in html
    assert "sima-neat-model-detail-layout" in script
    assert 'class="detail-layout-toggle"' in script
    assert 'aria-pressed="${state.detailLayout === "focused"}"' in script
    assert '.workspace[data-detail-layout="focused"]' in styles
    assert '.workspace[data-detail-layout="focused"] .result-list { width: calc(100% + 30px); }' in styles
    assert "@media (max-width: 840px)" in styles
    assert ".detail-layout-toggle { display: none; }" in styles


def test_model_metadata_tags_use_semantic_icons() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    for icon in ("branch", "precision", "compiler", "modelType", "architecture", "task"):
        assert f"{icon}: `<svg" in script
    assert "detailTag(icons.branch, branchName(run.git_ref), \"branch\")" in script
    assert 'class="detail-tag detail-tag-${kind}"' in script
    assert ".detail-tag svg" in styles
    assert ".detail-tag-branch svg { stroke: var(--blue); }" in styles


def test_local_preview_fixture_is_schema_v2_and_server_is_same_origin() -> None:
    fixture_path = BROWSER / "fixtures" / "performance-result-v2.json"
    schema_path = ROOT / "schemas" / "performance-result-v2.schema.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    server = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    Draft202012Validator(schema, format_checker=FormatChecker()).validate(fixture)
    assert fixture["schema_version"] == 2
    assert len(fixture["trials"]) == 3
    assert 'apiBase: location.origin' in server
    assert "/benchmarks/latest" in server
    assert "127\\.0\\.0\\.1|localhost" in script
