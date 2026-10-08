from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
BROWSER = ROOT / "artifact-browser"
DEPLOY_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-model-browser.yml"
DOCS_WORKFLOW = ROOT / ".github" / "workflows" / "developer-docs.yml"
PORTAL_ASSEMBLER = ROOT / "website" / "scripts" / "assemble-portal.mjs"
DOCUSAURUS_CONFIG = ROOT / "website" / "docusaurus.config.ts"
DOCS_STYLES = ROOT / "website" / "src" / "css" / "custom.css"
DOCS_THEME_TOGGLE = ROOT / "website" / "src" / "theme" / "ColorModeToggle" / "index.tsx"


def test_artifact_browser_assets_exist() -> None:
    for name in (
        "index.html",
        "app.js",
        "benchmark-adapters.js",
        "branch-builds.js",
        "downloads.js",
        "markdown.js",
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
    assert '<nav class="product-nav" aria-label="Model Registry">' in html
    assert '<span aria-current="page">Models</span>' in html
    assert '<a href="./docs">Developer docs</a>' in html
    assert 'href="./docs/metadata-schema">YAML reference</a>' not in html
    assert '>GitHub</a>' not in html


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

    build = workflow["jobs"]["build"]
    assert build["steps"][-1]["with"]["path"] == "website/portal-build/"
    assert any(step.get("run") == "npm run build:portal" for step in build["steps"])


def test_combined_portal_fingerprints_mutable_browser_assets() -> None:
    assemble = PORTAL_ASSEMBLER.read_text(encoding="utf-8")

    assert "process.env.GITHUB_SHA" in assemble
    for asset in ("app", "benchmark-adapters", "branch-builds", "downloads", "markdown", "styles", "config"):
        assert f"'{asset}'" in assemble
    assert "cpSync(docsBuild, output, {recursive: true})" in assemble
    assert "fingerprintedAssets" in assemble


def test_model_cards_use_lazy_integrity_checked_progressive_disclosure() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    renderer = (BROWSER / "markdown.js").read_text(encoding="utf-8")

    assert "documentation?.overview_markdown" in script
    assert 'headers: { Accept: "text/markdown" }' in script
    assert "/artifacts/${encodeURIComponent(documentation.artifact_name" in script
    assert 'crypto.subtle.digest("SHA-256"' in script
    assert "state.documentationCache.set(cacheKey, markdown)" in script
    assert 'aria-expanded="${expanded}"' in script
    assert "Read full model card" in script
    assert "Show overview only" in script
    assert "documentationHeroUrl" in script
    assert "documentation.hero?.markdown_path" in script
    assert "linear-gradient(to bottom, transparent" in styles
    assert ".model-card-hero" in styles
    assert "SAFE_LINK" in renderer
    assert "escapeHtml" in renderer
    assert "resolveImage" in renderer
    assert "resolveDocumentationLink" in renderer
    assert 'loading="lazy"' in renderer
    assert "export function renderMarkdown" in renderer


def test_model_card_markdown_resolves_safe_relative_links_at_the_source_commit() -> None:
    script = """
      import {readFileSync} from 'node:fs';
      const moduleSource = readFileSync('./artifact-browser/markdown.js', 'utf8');
      const moduleUrl = 'data:text/javascript;base64,' + Buffer.from(moduleSource).toString('base64');
      const {renderMarkdown, resolveDocumentationLink} = await import(moduleUrl);
      const documentation = {
        source_commit: 'a'.repeat(40),
        source_path: 'models/ssd_mobilenet_v3/README.md',
      };
      const resolveLink = (source) => resolveDocumentationLink(documentation, source);
      const markdown = '[QAT](qat/README.md) [section](#usage) [unsafe](javascript:alert(1))';
      console.log(JSON.stringify({
        html: renderMarkdown(markdown, {resolveLink}),
        traversal: resolveLink('../../../../outside.md'),
        external: resolveLink('//example.invalid/steal'),
      }));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = json.loads(result.stdout)
    source_url = "https://github.com/sima-neat/models/blob/" + "a" * 40
    assert f'{source_url}/models/ssd_mobilenet_v3/qat/README.md' in rendered["html"]
    assert f'{source_url}/models/ssd_mobilenet_v3/README.md#usage' in rendered["html"]
    assert "javascript:" not in rendered["html"]
    assert rendered["traversal"] is None
    assert rendered["external"] is None


def test_model_card_markdown_renders_every_validator_supported_fence() -> None:
    script = """
      import {readFileSync} from 'node:fs';
      const moduleSource = readFileSync('./artifact-browser/markdown.js', 'utf8');
      const moduleUrl = 'data:text/javascript;base64,' + Buffer.from(moduleSource).toString('base64');
      const {renderMarkdown} = await import(moduleUrl);
      console.log(JSON.stringify({
        tilde: renderMarkdown('~~~~bash\\n## Setup\\necho <ready>\\n~~~~'),
        longBacktick: renderMarkdown('````python\\nprint("ready")\\n````'),
        heading: renderMarkdown('   ### Conversion'),
        closedHeading: renderMarkdown('### Conversion ###'),
        indented: renderMarkdown('    first-command\\n\\tsecond-command\\n\\n    printf "<ready>"\\nnext'),
        blockquoteFence: renderMarkdown('> ```bash\\n> ### configure\\n> echo ready\\n> ````'),
        blockquoteLiteral: renderMarkdown('> ```bash\\n> > output.txt\\n> ````'),
        listFence: renderMarkdown('- ```bash\\n  ### configure\\n  echo ready\\n  ````'),
        mixedFence: renderMarkdown('- > ```bash\\n  > echo ready\\n  > ````'),
        nestedListFence: renderMarkdown('- - ```bash\\n    echo ready\\n    ````'),
        continuationFence: renderMarkdown('- step\\n\\n    ```bash\\n\\n    ## configure\\n    echo ready\\n    ```'),
        endedContainer: renderMarkdown('> ~~~text\\n## Unexpected\\n~~~'),
        listIndentedCode: renderMarkdown('-     echo one \\\\\\n      two'),
      }));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = json.loads(result.stdout)

    assert rendered["tilde"] == '<pre><code class="language-bash">## Setup\necho &lt;ready&gt;</code></pre>'
    assert rendered["longBacktick"] == '<pre><code class="language-python">print(&quot;ready&quot;)</code></pre>'
    assert rendered["heading"] == "<h4>Conversion</h4>"
    assert rendered["closedHeading"] == "<h4>Conversion</h4>"
    assert rendered["indented"] == (
        "<pre><code>first-command\nsecond-command\n\nprintf &quot;&lt;ready&gt;&quot;</code></pre>\n"
        "<p>next</p>"
    )
    assert rendered["blockquoteFence"] == (
        '<pre><code class="language-bash">### configure\necho ready</code></pre>'
    )
    assert rendered["blockquoteLiteral"] == (
        '<pre><code class="language-bash">&gt; output.txt</code></pre>'
    )
    assert rendered["listFence"] == (
        '<pre><code class="language-bash">### configure\necho ready</code></pre>'
    )
    assert rendered["mixedFence"] == '<pre><code class="language-bash">echo ready</code></pre>'
    assert rendered["nestedListFence"] == '<pre><code class="language-bash">echo ready</code></pre>'
    assert rendered["continuationFence"] == (
        '<ul><li>step</li></ul>\n<pre><code class="language-bash">\n## configure\necho ready</code></pre>'
    )
    assert "<h3>Unexpected</h3>" in rendered["endedContainer"]
    assert rendered["listIndentedCode"] == (
        '<ul><li><pre><code>echo one \\\ntwo</code></pre></li></ul>'
    )


def test_developer_docs_workflow_only_validates_the_combined_portal() -> None:
    workflow = yaml.load(DOCS_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    build = workflow["jobs"]["build"]

    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"build"}
    assert any(step.get("run") == "npm run build:portal" for step in build["steps"])
    portal_build = next(step for step in build["steps"] if step.get("run") == "npm run build:portal")
    assert portal_build["env"] == {
        "DOCS_SITE_URL": "https://developer-stg.sima.ai",
        "DOCS_BASE_URL": "/models/",
    }
    assert not any("pages" in step.get("uses", "") for step in build["steps"])


def test_combined_portal_shares_theme_storage_and_section_navigation() -> None:
    config = DOCUSAURUS_CONFIG.read_text(encoding="utf-8")

    assert "trailingSlash: true" in config
    assert "trailingSlash: false" not in config
    assert "namespace: false" in config
    assert "className: 'navbar__catalog-item'" in config
    assert 'href="${baseUrl}">Models</a>' in config
    assert "label: 'YAML reference'" not in config
    assert "label: 'GitHub'" not in config
    assert 'Staging registry' in config
    assert "showLastUpdateAuthor" not in config
    assert "showLastUpdateTime" not in config


def test_docs_catalog_navigation_reloads_the_assembled_portal() -> None:
    placeholder = (ROOT / "website" / "src" / "pages" / "index.tsx").read_text(encoding="utf-8")
    browser_mark = (BROWSER / "favicon.svg").read_text(encoding="utf-8")
    docs_mark = (ROOT / "website" / "static" / "img" / "mark.svg").read_text(encoding="utf-8")
    docs_favicon = (ROOT / "website" / "static" / "img" / "favicon.svg").read_text(encoding="utf-8")

    assert "window.location.replace(catalogUrl)" in placeholder
    assert browser_mark == docs_mark
    assert browser_mark == docs_favicon


def test_developer_docs_use_the_catalog_visual_system() -> None:
    browser_styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    docs_styles = DOCS_STYLES.read_text(encoding="utf-8")
    config = DOCUSAURUS_CONFIG.read_text(encoding="utf-8")

    for color in ("#f7f8f9", "#f1f3f5", "#ffffff", "#20262c", "#5f6972", "#dfe3e6", "#376f9f"):
        assert color in browser_styles
        assert color in docs_styles
    assert "font-size: clamp(2rem, 3vw, 2.65rem)" in docs_styles
    assert "style: 'light'" in config


def test_developer_docs_use_the_catalog_theme_toggle_behavior() -> None:
    toggle = DOCS_THEME_TOGGLE.read_text(encoding="utf-8")
    styles = DOCS_STYLES.read_text(encoding="utf-8")

    assert "document.documentElement.dataset.theme === 'dark'" in toggle
    assert "const nextMode = dark ? 'light' : 'dark'" in toggle
    assert "onChange(nextMode)" in toggle
    assert "Switch to ${nextMode} mode" in toggle
    assert "portal-theme-icon--sun" in toggle
    assert "portal-theme-icon--moon" in toggle
    assert ".portal-theme-toggle" in styles


def test_catalog_avoids_per_run_enrichment_and_uses_local_cache() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "enrichRuns" not in script
    assert "mapConcurrent" not in script
    assert "readCachedCatalog" in script
    assert "writeCachedCatalog" in script
    assert "state.runs[0]?.id" not in script


def test_branch_search_and_categories_filter_the_cached_snapshot() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'const branchSnapshot = history === "latest" && branch !== "all"' in script
    assert 'const categories = branchSnapshot ? [] :' in script
    assert 'const search = branchSnapshot ? "" :' in script
    assert 'const compiler = branchSnapshot ? "any" :' in script
    assert 'if (branchSnapshotMode()) return;' in script
    assert 'if (branchSnapshotMode()) {\n    renderRuns();\n    return;\n  }' in script


def test_branch_switch_loads_builds_even_if_taxonomy_refresh_fails() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    branch_change = script.split('elements["branch-filter"].addEventListener("change", async () => {', 1)[1]
    branch_change = branch_change.split('elements["compiler-filter"].addEventListener', 1)[0]

    assert 'await loadTaxonomy(state.preferredBranch || DEFAULT_BRANCH);\n  } catch (error) {' in branch_change
    assert 'setCategoryFiltersDisabled(true);' in branch_change
    assert 'if (navigationId !== state.navigationId) return;\n  try {\n    await loadRuns({ branch: state.preferredBranch });' in branch_change
    assert 'elements[elementId].disabled ? "" :' in script


def test_cached_build_history_preserves_run_mode() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'catalogMode: "runs"' in script
    assert 'state.catalogMode = modelsMode ? "models" : "runs"' in script
    assert 'mode: state.catalogMode' in script
    assert 'cached.models?.length || !cached.runs?.length ? "models" : "runs"' in script
    assert 'payload.mode !== "runs" && Object.hasOwn(payload, "models")' in script
    assert "applyCatalog(cached, { cache: false })" in script
    assert 'return state.catalogMode === "models"' in script
    assert 'elements["branch-filter"].value && !elements["history-filter"].checked' not in script


def test_concurrent_requests_do_not_overwrite_newer_selections() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "const requestId = ++state.runsRequestId" in script
    assert "if (requestId !== state.runsRequestId) return" in script
    assert "if (state.selectedId === runId)" in script


def test_linked_run_loads_independently_of_the_current_catalog_page() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert "navigationId === state.navigationId && state.selectedId) await selectRun(state.selectedId, { updateUrl: false });" in script
    assert "state.runs.some((run) => run.id === state.selectedId)" not in script
    assert "state.detailCache.get(runId)?.summary?.run" in script


def test_historical_model_cards_load_from_the_run_catalog_revision() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    select_run = script.split("async function selectRun(", 1)[1].split("function keyValue(", 1)[0]

    assert "historicalCatalogModels: new Map()" in script
    assert "async function loadHistoricalCatalogModel(run)" in script
    assert "/catalogs/${encodeURIComponent(run.commit_sha)}?${query}" in script
    assert "payload.catalog?.document?.models" in script
    assert 'state.catalogMode === "runs"' in select_run
    assert "await loadHistoricalCatalogModel(detail.summary.run)" in select_run


def test_cached_run_detail_backfills_package_when_catalog_context_changes() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    select_run = script.split("async function selectRun(", 1)[1].split("function keyValue(", 1)[0]

    assert 'const currentPackage = fullPackageForRun(detail.summary.run);' in select_run
    assert 'detail.packageSummary?.run?.id !== currentPackage.run_id' in select_run
    assert 'currentPackage.run_id !== requestedPackageId' in select_run
    assert 'detail.packageSummary = await api(`/v1/runs/${currentPackage.run_id}`).catch(() => null);' in select_run


def test_run_selection_keeps_the_url_branch_in_sync_with_the_filter() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    select_run = script.split("async function selectRun(", 1)[1].split("function keyValue(", 1)[0]

    assert 'url.searchParams.set("branch", elements["branch-filter"].value);' in select_run
    assert 'url.searchParams.set("branch", branch);' not in select_run
    assert 'state.urlBranch = link.has("branch")' in script


def test_all_branches_selection_has_an_explicit_url_value() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'url.searchParams.set("branch", state.preferredBranch);' in script
    assert 'resetUrl.searchParams.set("branch", resetBranch);' in script
    assert 'searchParams.delete("branch")' not in script


def test_browser_history_restores_category_filters_and_empty_detail() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    initialize = script.split("async function initialize()", 1)[1].split('elements["theme-toggle"]', 1)[0]

    assert 'Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, link.get(queryName) || ""])' in initialize
    assert "state.taxonomy = null" in initialize
    assert "if (state.selectedId) detailSkeleton();\n  else showEmptyDetail();" in initialize
    assert "if (state.taxonomy) populateCategoryFilters();" in initialize
    assert 'window.addEventListener("popstate", () => {' in script


def test_detail_error_can_be_retried_and_closed() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'errorPanel(error.message, "Could not load this build", { closable: true })' in script
    assert "event.target.closest('[data-action=\"retry\"]') && state.selectedId" in script
    assert "selectRun(state.selectedId, { updateUrl: false });" in script


def test_branch_filter_defaults_to_develop_and_persists_user_selection() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'const DEFAULT_BRANCH = "develop"' in script
    assert "sima-neat-model-branch:" in script
    assert "readBranchPreference" in script
    assert "writeBranchPreference(state.preferredBranch)" in script
    assert "branches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH" in script


def test_branch_filter_uses_an_independent_cached_registry_catalog() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert "/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/branches" in script
    assert "branchCatalogCacheKey" in script
    assert "readCachedBranches" in script
    assert "writeCachedBranches" in script
    assert "loadTaxonomy(elements[\"branch-filter\"].value || DEFAULT_BRANCH)" in script
    assert "The registry returned an empty branch catalog; keeping saved branches." in script
    assert "knownBranches.delete(branch)" not in script
    assert "...(payload.branches || [])" not in script
    assert 'path == "/v1/repositories/sima-neat/models/branches"' in preview


def test_earlier_builds_pagination_tracks_the_selected_branch() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'query.set("git_ref", `refs/heads/${branch}`)' in script
    assert "availableBranches" in script
    assert 'await loadRuns({ branch: elements["branch-filter"].value })' in script
    assert 'elements["load-more"].hidden = !state.hasMore' in script


def test_browser_has_persistent_dual_theme_and_accessibility_controls() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'data-theme="dark"' in styles
    assert "prefers-color-scheme: dark" in html
    assert "sima-neat-artifact-theme" in html
    assert "sima-neat-artifact-theme" in script
    assert 'localStorage.getItem("theme")' in html
    assert 'localStorage.setItem("theme", theme)' in script
    assert "Storage is optional; use the system preference when unavailable." in html
    assert "Storage may be disabled; theme selection still works for this page load." in script
    assert 'class="skip-link"' in html
    assert 'aria-live="polite"' in html
    assert "prefers-reduced-motion: reduce" in styles


def test_browser_uses_refined_search_first_layout() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "Roboto:wght@400;500;600;700" in html
    assert "IBM+Plex+Mono" in html
    assert '<h1 class="sr-only">Model Registry</h1>' in html
    assert "Search models, tasks, or architectures" in html
    assert 'class="page-header"' not in html
    assert "Trace every" not in html
    assert "--font-display" not in styles
    assert "font-family: var(--font-body)" in styles


def test_secondary_filters_are_collapsed_and_status_is_not_presented() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="filter-toggle"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="filter-drawer" class="filter-drawer" hidden' in html
    assert 'id="filter-count"' in html
    assert "--filter-surface: color-mix(in srgb, var(--surface) 88%, var(--blue-soft))" in styles
    assert "background: var(--filter-surface)" in styles
    assert "box-shadow: 0 0 0 100vmax var(--filter-surface)" in styles
    assert 'id="status-filter"' not in html
    assert 'class="run-status' not in script
    assert 'class="status-label' not in script
    assert 'query.set("status", "completed")' in script
    assert 'run.status === "completed"' in script


def test_model_catalog_is_default_with_explicit_build_history_mode() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="history-filter" type="checkbox" role="switch"' in html
    assert "Switches from models to chronological build runs." in html
    assert "Show build history" in html
    assert 'if (historical) {' in script
    assert 'allPages(api, `${base}/runs`, compileQuery, "runs")' in script
    assert 'allPages(api, `${base}/runs`, packageQuery, "runs")' in script
    assert 'allPages(api, `${base}/models`, modelQuery, "models")' in script
    assert 'query.set("run_type", "model_compile")' in script
    assert 'if (historical && !elements["history-filter"].checked) query.set("latest_per_model_compiler", "true")' in script
    assert 'link.get("history") === "1"' in script
    assert 'url.searchParams.set("history", "1")' in script
    assert 'elements["history-filter"].checked = false' in script
    assert ".switch-row input:checked + i" in styles


def test_browser_loads_taxonomy_from_the_current_catalog_and_exposes_category_facets() -> None:
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
    assert "/catalogs/current?git_ref=" in script
    assert "payload.catalog?.document?.taxonomy" in script
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


def test_browser_formats_v1_and_v2_build_identities_without_a_false_default() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert "function buildIdentity(run)" in script
    assert "metadata.build_profile_id || metadata.variant_id" in script
    assert 'component !== "model"' in script
    assert '`${profile} · ${component}`' in script
    assert '"Build identity unavailable"' in script
    assert '"default artifact"' not in script
    assert "const build = buildIdentity(run);" in script
    assert "run.metadata?.build_profile_id, run.metadata?.component_id" in script
    assert '"build_profile_id": variant_id' in preview
    assert '"component_id": "model"' in preview


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


def test_mpk_validation_is_a_compact_accessible_status_mark() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'test.name === "mpk-validation" && test.status === "passed"' in script
    assert 'class="mpk-validation-mark" tabindex="0" role="img"' in script
    assert "MPK package integrity verified: archive structure, checksum, and build provenance match." in script
    assert '<h3>Validation</h3>' not in script
    assert ".detail-title-actions" in styles
    compact_controls = (
        ".model-source-link, .huggingface-source-link, .benchmark-shortcut, "
        ".artifact-download-shortcut, .mpk-validation-mark { width: 38px; height: 38px"
    )
    assert compact_controls in styles
    assert ".detail-title-actions > [data-tooltip]:hover::after" in styles


def test_existing_benchmark_has_an_accessible_scroll_shortcut() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "const hasBenchmark = Boolean(detail.benchmark && benchmarkMarkup)" in script
    assert (
        'data-scroll-benchmark aria-label="Jump to benchmark results" data-tooltip="Jump to benchmark results"'
        in script
    )
    assert 'id="benchmark-section" class="benchmark-section" tabindex="-1"' in script
    assert 'querySelector("#benchmark-section")' in script
    assert 'benchmark.scrollIntoView({' in script
    assert 'matchMedia("(prefers-reduced-motion: reduce)").matches' in script
    assert ".benchmark-shortcut:focus-visible" in styles


def test_download_is_promoted_and_artifacts_show_only_size_and_checksum() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'class="artifact-download-shortcut"' in script
    assert 'aria-label="Download full package"' in script
    assert 'data-tooltip="Download full package"' in script
    assert 'aria-label="Download individual components"' in script
    assert 'data-tooltip="Download individual components"' in script
    assert '<dl class="artifact-metadata">' in script
    assert '<dt>File size</dt>' in script
    assert '<dt>SHA-256</dt>' in script
    assert 'class="download-button"' not in script
    assert 'class="artifact-main"' not in script
    assert ".artifact-download-shortcut:focus-visible" in styles


def test_package_download_dialog_uses_one_consistent_label_value_grid() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="download-kicker"' not in html
    assert '"download-kicker"' not in script
    assert '<dt>File name</dt>' in script
    assert '<dt>Attribution</dt>' in script
    assert 'Attribution: ${license.attribution}' not in script
    assert ".download-package-info dt, .download-package-info dd" in styles
    assert ".download-package-info dt { border-right: 1px solid var(--line)" in styles
    assert ".download-package-info code { font: inherit" in styles
    assert 'font-family: "IBM Plex Mono"' not in styles.split(".download-package-info code", 1)[1].split("}", 1)[0]


def test_only_model_packs_are_presented_as_artifacts() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    assert 'artifact.artifact_type === "model-pack"' in script
    assert "artifacts: modelPackArtifacts(summary?.artifacts)" in script
    assert "const artifacts = modelPackArtifacts(detail.summary.artifacts)" in script
    assert "const primaryArtifact = artifacts[0]" in script
    assert '|| artifacts[0]' not in script


def test_source_input_links_to_the_original_hugging_face_model_card() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert "function trustedHuggingFaceUrl" in script
    assert 'url.hostname !== "huggingface.co"' in script
    assert "function huggingFaceSourceLink" in script
    assert 'sourceReference.startsWith("hf://")' in script
    assert "input.metadata?.model_card" in script
    assert "hf:\\/\\/(models|datasets|spaces)" in script
    assert 'class="huggingface-source-link"' in script
    assert 'target="_blank" rel="noreferrer"' in script
    assert "Open source ONNX model card on Hugging Face" in script
    assert 'data-tooltip="Open source ONNX model card on Hugging Face"' in script
    assert "Official no-border logo from https://huggingface.co/brand" in script
    assert 'viewBox="0 0 95 88"' in script
    assert 'fill="#FFD21E"' in script
    assert ".huggingface-source-link:focus-visible" in styles
    assert ".huggingface-source-link svg { width: 20px; height: 19px; }" in styles
    assert "https://huggingface.co/onnx-community/resnet-50-ONNX" in preview


def test_benchmark_rendering_is_schema_versioned_and_charted() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    adapters = (BROWSER / "benchmark-adapters.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'import { adaptBenchmarkResult } from "./benchmark-adapters.js"' in script
    assert "new Map([[1, adaptV1], [2, adaptV2], [3, adaptV3], [4, adaptV4], [5, adaptV4]])" in adapters
    assert "Object.entries(result.metrics || {})" in adapters
    assert "Object.entries(trial)" in adapters
    assert "/benchmarks/latest" in script
    assert 'class="trial-chart benchmark-chart"' in script
    assert "data-trial-metric" in script
    assert "availableTrialMetrics" in script
    assert "data-temperature-sensor" not in script
    assert 'item.sensor.toLowerCase() === "temp1"' in script
    assert 'class="temperature-line"' in script
    assert 'role="img"' in script
    assert 'class="sr-only"' in script
    assert ".chart-points circle" in styles
    assert ".temperature-line" in styles
    assert "temperature_samples" in adapters
    assert "sensor_id" in adapters
    assert "fallbackMetricLabel" in script
    assert 'class="benchmark-measurements"' in script
    assert "mla_memory_peak_delta_mib" in script
    assert "time_to_first_inference_ms" in script


def test_accuracy_result_is_rendered_with_scope_and_reference_comparison() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert 'api(`/v1/runs/${runId}/accuracy/latest`)' in script
    assert 'payload?.result || fallback?.summary' in script
    assert 'item.suite === "accuracy"' in script
    assert 'id="accuracy-section"' in script
    assert "Candidate top-1" in script
    assert "Reference ${percentage(reference.top1_accuracy)}" in script
    assert "dataset.evaluated_samples.toLocaleString()" in script
    assert 'class="accuracy-explainer"' in script
    assert 'summary class="accuracy-status ${statusClass}"' in script
    assert '>?</summary>' not in script
    assert "run the qualification profile for final accuracy reporting" in script
    assert 'const coverageLabel = fullCoverage ? "Full coverage" : "Sampled coverage"' in script
    assert 'const status = qualified ? "Qualification" : "Provisional"' in script
    assert 'report.validation?.status !== "provisional"' in script
    assert 'class="accuracy-status ${statusClass}"' in script
    assert ".accuracy-summary" in styles
    assert ".accuracy-status.provisional" in styles
    assert ".accuracy-status.qualification" in styles
    assert ".accuracy-explainer-card" in styles
    assert ".accuracy-explainer:hover > .accuracy-explainer-card" in styles
    assert ".accuracy-explainer[open] > .accuracy-explainer-card" in styles
    assert '"evaluated_samples": 500' in preview
    assert '"top1_accuracy": 0.762' in preview
    assert '"suite": "accuracy"' in preview
    assert 'parts[3:] == ["accuracy", "latest"]' in preview


def test_devkit_accuracy_is_read_from_the_benchmark_run_and_rendered_per_metric() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert "api(`/v1/runs/${benchmark.benchmark_run.id}/test-results`)" in script
    assert "item.summary?.schema_version === 2" in script
    assert "Object.entries(report.metrics || {})" in script
    assert 'coco_map: "mAP 50–95"' in script
    assert "ACCURACY_METRIC_LABELS[name] || name" in script
    assert '"engine": "devkit_script"' in preview
    assert '"suite": "accuracy"' in preview


def test_numerical_fidelity_is_rendered_from_correctness_results() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert 'item.name === "numerical-fidelity"' in script
    assert "fidelityPanel(tests, run.id)" in script
    assert 'data-download="numerical-fidelity"' in script
    panel = script.split("function fidelityPanel(", 1)[1].split("function licensePanel(", 1)[0]
    assert '<div class="benchmark-summary">${metrics}</div>' in panel
    assert panel.count('class="benchmark-metric"') == 2
    assert '<span>${escapeHtml(metric.name)} · Cosine similarity</span>' in panel
    assert '<span>${escapeHtml(metric.name)} · Normalized RMSE</span>' in panel
    assert '<table>' not in panel
    assert panel.index('class="benchmark-summary"') < panel.index('class="muted-message"')
    assert "<details><summary>Evaluation details</summary>" in panel
    assert '<details open' not in panel
    assert 'Number(metric.cosine_similarity).toFixed(4)' in panel
    assert 'Number(metric.normalized_rmse).toFixed(4)' in panel
    assert "Mean absolute error" not in panel
    assert "Maximum absolute error" not in panel
    assert "Output similarity, not task accuracy" in panel
    assert 'keyValue("Reference", report.reference)' in panel
    assert 'keyValue("Candidate", report.candidate)' in panel
    assert "report.dataset?.id" in panel
    assert ".fidelity-output" not in styles
    assert ".fidelity-summary" not in styles
    assert '"name": "numerical-fidelity"' in preview


def test_models_can_be_selected_and_compared_in_an_accessible_dialog() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="compare-button"' in html
    assert 'id="compare-dialog" class="compare-dialog"' in html
    assert 'aria-labelledby="compare-title" aria-describedby="compare-description"' in html
    assert "compareSelection: new Map()" in script
    assert 'data-compare-run-id="${run.id}"' in script
    assert "return [...state.compareSelection.values()]" in script
    assert 'elements["compare-button"].disabled = selectedCount < 2' in script
    assert "mapWithConcurrency(runs, 6, loadComparisonEntry)" in script
    assert 'api(`/v1/runs/${run.id}/benchmarks/latest`)' in script
    assert "adaptBenchmarkResult(detail.benchmark.result)" in script
    assert '<th scope="col">Artifact size</th>' in script
    assert "data-comparison-metric" in script
    assert "benchmark?.measurements[field]" in script
    assert 'elements["compare-dialog"].showModal()' in script
    assert ".comparison-table-wrap" in styles
    assert 'class="comparison-measurement-control"' in script
    assert ".comparison-measurement-bar { display: flex; flex: 0 0 auto; justify-content: flex-end" in styles
    assert ".compare-content { min-height: 0; flex: 1; display: flex; overflow: hidden" in styles
    assert ".comparison-table thead th { position: sticky; top: 0; z-index: 2" in styles
    assert "height: min(820px, calc(100vh - 50px))" in styles
    assert "position: sticky; left: 0" in styles
    assert ".compare-select input:focus-visible + span" in styles


def test_model_catalog_uses_clean_model_navigation_and_detail_build_switching() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert "models: []" in script
    assert "function profileGroups(model)" in script
    assert 'class="model-row' in script
    assert 'class="detail-variant-tabs" role="tablist"' in script
    assert 'class="detail-variant-tab" type="button" role="tab"' in script
    assert '<strong>Variants</strong>' in script
    assert '<span>Components</span>' in script
    assert '"<b>Current</b>"' in script
    assert 'class="detail-component-tabs" role="group"' in script
    assert 'class="detail-compare-select compare-select"' in script
    assert "activeProfileByModel: new Map()" in script
    assert "activeComponentByProfile: new Map()" in script
    assert "function selectedCatalogBuild(" in script
    assert "function compilerFilteredBuilds(model)" in script
    assert "compilerFilteredBuilds(model).filter" in script
    assert "build.toolchain_version === compiler" in script
    assert 'branchModels(state.branchRuns, state.packageRuns, state.catalogDefinitions, elements["compiler-filter"].value)' in script
    assert "...state.branchRuns.map((run) => run.toolchain_version)" in script
    assert "payload.models.flatMap" not in script
    assert ".detail-variant-tab[aria-selected=\"true\"]::before" in styles
    assert ".model-row.selected" in styles
    render_model_row = script.split("function renderModelRow(", 1)[1].split("function renderModels(", 1)[0]
    assert 'role="tablist"' not in render_model_row
    assert "data-compare-run-id" not in render_model_row
    assert "const [activeProfile, activeBuilds] = profiles.find(" in script
    assert "componentId(build) === requestedComponent" in script
    assert '("rf_detr_small", "modalix_int8_bf16", "backbone"' in preview
    assert '("rf_detr_small", "modalix_int8_bf16", "transformer"' in preview


def test_focused_model_layout_collapses_list_metadata_without_overflow() -> None:
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '.workspace[data-detail-layout="focused"] .results-pane { overflow: hidden; padding-right: 0; }' in styles
    assert '.workspace[data-detail-layout="focused"] .section-heading { gap: 8px; padding-right: 16px; }' in styles
    assert '.workspace[data-detail-layout="focused"] .compare-button #compare-label { display: none; }' in styles
    assert (
        '.workspace[data-detail-layout="focused"] .run-entry { grid-template-columns: 34px minmax(0, 1fr); }' in styles
    )
    assert (
        '.workspace[data-detail-layout="focused"] .run-row { grid-template-columns: minmax(0, 1fr); gap: 0; }' in styles
    )
    assert '.run-row { width: 100%; min-width: 0;' in styles
    assert 'width: calc(100% + 30px)' not in styles
    assert '.workspace[data-detail-layout="focused"] .run-date,' in styles
    assert '.workspace[data-detail-layout="focused"] .row-arrow { display: none; }' in styles


def test_detail_layout_can_focus_the_selected_model() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="workspace"' in html
    assert "sima-neat-model-detail-layout" in script
    assert 'class="detail-layout-toggle"' in script
    assert 'aria-pressed="${state.detailLayout === "focused"}"' in script
    assert '.workspace[data-detail-layout="focused"]' in styles
    assert '.workspace[data-detail-layout="focused"] .results-pane { overflow: hidden; padding-right: 0; }' in styles
    assert "@media (max-width: 840px)" in styles
    assert ".detail-layout-toggle { display: none; }" in styles


def test_model_metadata_tags_use_semantic_icons() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    for icon in ("branch", "precision", "compiler", "license", "modelType", "architecture", "task"):
        assert f"{icon}: `<svg" in script
    assert "detailTag(icons.branch, branchName(run.git_ref), \"branch\")" in script
    assert 'class="detail-tag detail-tag-${kind}"' in script
    assert ".detail-tag svg" in styles
    assert ".detail-tag-branch svg { stroke: var(--blue); }" in styles


def test_model_license_metadata_is_rendered_with_redistribution_and_attribution() -> None:
    script = (BROWSER / "app.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")

    assert "const license = run.metadata?.license" in script
    assert "function licensePanel(license)" in script
    assert 'detailTag(icons.license, license.spdx, "license")' in script
    assert 'keyValue("SPDX", license.spdx' in script
    assert 'keyValue("Redistribution", redistribution)' in script
    assert 'class="license-attribution"' in script
    assert ".detail-tag-license svg" in styles
    assert ".license-attribution" in styles
    assert '"spdx": "Apache-2.0"' in preview
    assert '"redistribution": "allowed"' in preview


def test_local_preview_fixture_is_extensible_schema_v4_and_server_is_same_origin() -> None:
    fixture_path = BROWSER / "fixtures" / "performance-result-v4.json"
    schema_path = ROOT / "schemas" / "performance-result-v4.schema.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    server = (ROOT / "scripts" / "preview_artifact_browser.py").read_text(encoding="utf-8")
    script = (BROWSER / "app.js").read_text(encoding="utf-8")

    Draft202012Validator(schema, format_checker=FormatChecker()).validate(fixture)
    prior_v4 = json.loads(json.dumps(fixture))
    prior_v4["metrics"].pop("model_load_ms")
    prior_v4["metrics"].pop("time_to_first_inference_ms")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(prior_v4)
    assert fixture["schema_version"] == 4
    assert fixture["metrics"]["mla_memory_peak_delta_mib"]["value"] == 136.5
    assert fixture["metrics"]["time_to_first_inference_ms"]["value"] == 348.7
    assert fixture["metrics"]["preview_only_extension_metric"]["unit"] == "mock-units"
    assert fixture["metrics"]["preview_only_extension_metric"]["details"]["synthetic"] is True
    assert len(fixture["trials"]) == 3
    assert 'apiBase: location.origin' in server
    assert 'parts[3:] == ["benchmarks", "latest"]' in server
    assert "PREVIEW_RUNS = (" in server
    assert "range(len(PREVIEW_RUNS))" in server
    assert "127\\.0\\.0\\.1|localhost" in script


def test_branch_cards_use_only_coherent_branch_runs_and_matching_packages() -> None:
    script = r"""
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source = readFileSync('./artifact-browser/branch-builds.js', 'utf8');
const {allPages, branchModels} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

const run = (id, model, component, workflow, version = '3.0', time = '2026-10-01T12:00:00Z') => ({
  id, run_type: 'model_compile', status: 'completed',
  git_ref: 'refs/heads/perf/rfdetr-attention-layout', commit_sha: workflow,
  toolchain_version: version, target_platform: 'modalix', finished_at: time,
  metadata: {model_id: model, build_profile_id: 'modalix_int8_bf16', component_id: component, github_run_id: workflow},
});
const current = [run('backbone', 'rf_detr_small', 'backbone', 'workflow-2'),
                 run('transformer', 'rf_detr_small', 'transformer', 'workflow-2')];
const old = [run('old-backbone', 'rf_detr_small', 'backbone', 'workflow-1', '2.1', '2026-09-01T12:00:00Z'),
             run('old-transformer', 'rf_detr_small', 'transformer', 'workflow-1', '2.1', '2026-09-01T12:00:00Z')];
const catalog = [{model_id: 'rf_detr_small', display_name: 'RF-DETR', builds: [
  {run_id: 'stale-backbone', build_profile_id: 'modalix_int8_bf16', component_id: 'backbone', artifact_suffix: 'backbone_int8'},
  {run_id: 'stale-transformer', build_profile_id: 'modalix_int8_bf16', component_id: 'transformer', artifact_suffix: 'transformer_bf16'},
]},
                 {model_id: 'ssd_mobilenet', display_name: 'SSD', builds: [{run_id: 'inherited'}]}];
const packageRun = (id, components) => ({id, run_type: 'model_package', status: 'completed',
  git_ref: 'refs/heads/perf/rfdetr-attention-layout', commit_sha: 'workflow-2', metadata: {
  model_id: 'rf_detr_small', build_profile_id: 'modalix_int8_bf16', component_run_ids: components,
}});
const packages = [packageRun('good', {backbone: 'backbone', transformer: 'transformer'}),
                  packageRun('wrong', {backbone: 'backbone', transformer: 'old-transformer'})];
const models = branchModels([...old, ...current], packages, catalog);
assert.deepEqual(models.map(model => model.model_id), ['rf_detr_small']);
assert.deepEqual(models[0].builds.map(build => build.run_id), ['backbone', 'transformer']);
assert.deepEqual(models[0].builds.map(build => build.artifact_suffix), ['backbone_int8', 'transformer_bf16']);
assert.equal(models[0].builds[0].run.git_ref, 'refs/heads/perf/rfdetr-attention-layout');
assert.deepEqual(models[0].packages.map(item => item.run_id), ['good']);
assert.deepEqual(branchModels([...old, ...current], packages, catalog, '2.1')[0].builds.map(build => build.run_id),
                 ['old-backbone', 'old-transformer']);
const sameCompilerOld = [run('same-old-backbone', 'rf_detr_small', 'backbone', 'workflow-0', '3.0', '2026-08-01T12:00:00Z'),
                         run('same-old-transformer', 'rf_detr_small', 'transformer', 'workflow-0', '3.0', '2026-08-01T12:00:00Z')];
assert.deepEqual(branchModels([...sameCompilerOld, ...current], packages, catalog, '3.0')[0].builds.map(build => build.run_id),
                 ['backbone', 'transformer']);
assert.deepEqual(branchModels([], packages, catalog), []);
assert.deepEqual(branchModels([{...run('benchmark', 'inherited', 'model', 'workflow-2'), run_type: 'model_benchmark'}], [], catalog), []);
const develop = ['ssd_mobilenet', 'resnet50', 'yolov8', 'mobilenet', 'rf_detr_small', 'unet']
  .map(model => run(model, model, 'model', model));
assert.equal(branchModels(develop, [], catalog).length, 6);
const calls = [];
const definitions = await allPages(async path => {
  calls.push(path);
  return calls.length === 1
    ? {models: [{model_id: 'first'}], has_more: true, next_cursor: 'page-2'}
    : {models: [{model_id: 'second'}], has_more: false};
}, '/models', new URLSearchParams({git_ref: 'refs/heads/develop', limit: '100'}), 'models');
assert.deepEqual(definitions.map(model => model.model_id), ['first', 'second']);
assert.equal(calls.length, 2);
assert.match(calls[1], /cursor=page-2/);
"""
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
