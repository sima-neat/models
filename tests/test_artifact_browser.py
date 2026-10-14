from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from model_registry.model import load_models

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
ORCHESTRATOR_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-model-orchestrator.yml"
DOCS_WORKFLOW = ROOT / ".github" / "workflows" / "developer-docs.yml"
PORTAL_ASSEMBLER = ROOT / "website" / "scripts" / "assemble-portal.mjs"
DOCUSAURUS_CONFIG = ROOT / "website" / "docusaurus.config.ts"
DOCS_STYLES = ROOT / "website" / "src" / "css" / "custom.css"
DOCS_THEME_TOGGLE = ROOT / "website" / "src" / "theme" / "ColorModeToggle" / "index.tsx"
BROWSER_MODULES = (
    "app.js", "state.js", "ui.js", "metrics.js", "api.js", "catalog-view.js", "compare-view.js", "model-view.js",
)


def _browser_source(*modules: str) -> str:
    """Source of the named browser modules, or of every module when none is named."""
    return "\n".join((BROWSER / module).read_text(encoding="utf-8") for module in modules or BROWSER_MODULES)


def test_artifact_browser_assets_exist() -> None:
    for name in (
        "index.html",
        "app.js",
        "state.js",
        "ui.js",
        "metrics.js",
        "api.js",
        "catalog-view.js",
        "compare-view.js",
        "model-view.js",
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
    script = _browser_source("state.js", "api.js")
    browser = _browser_source()
    config = (BROWSER / "config.js").read_text(encoding="utf-8")

    assert 'src="config.js"' in html
    assert "MODEL_REGISTRY_CONFIG" in script
    assert "https://models-registry.stg.neat.sima.ai" in config
    assert "fetch(`${state.apiBase}${path}`" in script
    assert "fetch(`/api${path}`" not in browser
    assert not (BROWSER / "serve.py").exists()


def test_browser_links_home_relative_to_the_deployed_subpath() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")

    assert 'class="brand" href="./"' in html
    assert 'class="brand" href="/"' not in html
    assert '<nav class="product-nav" aria-label="Model Registry">' in html
    assert '<a id="nav-models" href="./" aria-current="page">Models</a>' in html
    assert '<a id="nav-compare" href="./?view=compare">Compare</a>' in html
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
    workflow = yaml.load(ORCHESTRATOR_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["publish-model-browser"]

    assert job["if"] == (
        "(github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop') && "
        "needs.select.outputs.vulcan_environment == 'staging'"
    )
    assert job["uses"] == "sima-neat/.github/.github/workflows/vulcan-publish-static-site.yml@main"
    assert job["permissions"] == {"contents": "read", "actions": "write", "id-token": "write"}
    # One group for every branch, so publishes from main and develop never overlap.
    assert job["concurrency"] == {
        "group": "${{ github.repository }}-model-browser-staging",
        "cancel-in-progress": "false",
    }

    publish = job["with"]
    assert publish["bucket"] == "${{ needs.resolve-vulcan-config.outputs.docs_bucket }}"
    assert publish["role_to_assume"] == "${{ needs.resolve-vulcan-config.outputs.docs_publisher_role_arn }}"
    assert publish["cloudfront_distribution_id"] == (
        "${{ needs.resolve-vulcan-config.outputs.sysdoc_cloudfront_distribution_id }}"
    )
    assert publish["s3_prefix"] == "models"
    assert publish["cloudfront_paths"] == "/models /models/*"

    # The deployment publishes the site the run's portal check built and checked.
    assert workflow["jobs"]["portal"]["uses"] == "./.github/workflows/developer-docs.yml"
    assert job["needs"] == ["select", "portal", "resolve-vulcan-config"]
    docs = yaml.load(DOCS_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    upload = docs["jobs"]["build"]["steps"][-1]
    assert upload["with"]["path"] == "website/portal-build/"
    assert publish["artifact_pattern"] == upload["with"]["name"]


def test_portal_is_uploaded_only_after_its_checks_pass() -> None:
    workflow = yaml.load(DOCS_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["build"]["steps"]
    position = {step.get("run") or step.get("uses", "").split("@")[0]: index for index, step in enumerate(steps)}

    assert position["npm run build:portal"] < position["npm run smoke:portal"] < position["actions/upload-artifact"]
    assert position["node --test tests/browser/*.test.mjs"] < position["actions/upload-artifact"]
    assert position["npx playwright install --with-deps chromium"] < position["npm run smoke:portal"]


@contextmanager
def _preview(root: Path) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, str(BROWSER / "preview.py"), "--root", str(root), "--port", str(port)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        server.stdout.readline()
        yield f"http://127.0.0.1:{port}"
    finally:
        server.kill()


def _get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def test_preview_serves_the_models_this_repository_declares(tmp_path: Path) -> None:
    models = load_models(ROOT)
    api = "/v1/repositories/sima-neat/models"
    with _preview(tmp_path) as base:
        entries = _get_json(f"{base}{api}/models")["models"]
        catalog = _get_json(f"{base}{api}/catalogs/current?git_ref=refs/heads/develop")
        build = entries[0]["builds"][0]
        provenance = _get_json(f"{base}/v1/runs/{build['run_id']}/provenance")

    assert sorted(entry["model_id"] for entry in entries) == sorted(models)
    for entry in entries:
        model = models[entry["model_id"]]
        assert entry["display_name"] == model["display_name"]
        assert entry["description"] == model["description"]
        assert entry["builds"]
        assert entry["documentation"]["hero"]["alt_text"]
    assert sorted(model["id"] for model in catalog["catalog"]["document"]["models"]) == sorted(models)
    assert provenance["run"]["id"] == build["run_id"]
    assert provenance["run"]["metadata"]["build_profile_id"] == build["build_profile_id"]


def test_preview_serves_full_packages_as_the_publisher_records_them(tmp_path: Path) -> None:
    api = "/v1/repositories/sima-neat/models"
    with _preview(tmp_path) as base:
        catalog = _get_json(f"{base}{api}/catalogs/current?git_ref=refs/heads/develop")
        package = next(
            package for model in catalog["catalog"]["document"]["models"] for package in model.get("packages", [])
        )
        artifacts = _get_json(f"{base}/v1/runs/{package['run_id']}")["artifacts"]

    # The browser's Download finds the full package by this type, as the registry publisher writes it.
    assert [artifact["artifact_type"] for artifact in artifacts] == ["model-package"]
    assert artifacts[0]["media_type"] == "application/zip"
    assert artifacts[0]["metadata"]["component_run_ids"] == package["component_run_ids"]
    assert 'item.artifact_type === "model-package"' in _browser_source("app.js")


def test_preview_serves_fixture_paths_only_inside_the_fixtures(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("portal", encoding="utf-8")
    with _preview(tmp_path) as base:
        status = {}
        for request in (
            "/fixtures/performance-result-v4.json",
            "/fixtures/../../README.md",
            "/fixtures/%2e%2e/%2e%2e/README.md",
            "/fixtures/../../../../../../etc/hosts",
        ):
            # Raw request lines keep the ".." that an HTTP client would normalize away.
            response = subprocess.run(
                ["curl", "-s", "--path-as-is", "-o", os.devnull, "-w", "%{http_code}", f"{base}{request}"],
                capture_output=True, text=True, check=True,
            )
            status[request] = response.stdout

    assert status == {
        "/fixtures/performance-result-v4.json": "200",
        "/fixtures/../../README.md": "404",
        "/fixtures/%2e%2e/%2e%2e/README.md": "404",
        "/fixtures/../../../../../../etc/hosts": "404",
    }


def test_combined_portal_fingerprints_mutable_browser_assets() -> None:
    assemble = PORTAL_ASSEMBLER.read_text(encoding="utf-8")

    assert "process.env.GITHUB_SHA" in assemble
    for asset in (
        "app", "state", "ui", "metrics", "api", "catalog-view", "compare-view", "model-view",
        "benchmark-adapters", "branch-builds", "downloads", "markdown", "styles", "config",
    ):
        assert f"'{asset}'" in assemble
    assert "cpSync(docsBuild, output, {recursive: true})" in assemble
    assert "fingerprintedAssets" in assemble


def assemble_portal(root: Path) -> subprocess.CompletedProcess[str]:
    (root / "website" / "scripts").mkdir(parents=True)
    (root / "website" / "build").mkdir()
    shutil.copy(PORTAL_ASSEMBLER, root / "website" / "scripts" / PORTAL_ASSEMBLER.name)
    for source in ("models", "schemas", "tools"):
        (root / source).symlink_to(ROOT / source)
    return subprocess.run(
        ["node", str(root / "website" / "scripts" / PORTAL_ASSEMBLER.name)],
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_SHA": "a" * 40, "PYTHON": sys.executable},
    )


def test_assembled_portal_modules_import_only_deployed_files(tmp_path: Path) -> None:
    shutil.copytree(BROWSER, tmp_path / "artifact-browser")
    assert assemble_portal(tmp_path).returncode == 0

    portal = tmp_path / "website" / "portal-build"
    modules = sorted(portal.glob("*.aaaaaaaaaaaa.js"))
    assert {path.name.split(".")[0] for path in modules} >= {
        "app", "state", "ui", "metrics", "api", "catalog-view", "compare-view", "model-view",
        "downloads", "branch-builds",
    }
    for module in modules:
        for specifier in re.findall(r"[\"'](\.\.?/[^\"'\s]*\.js)[\"']", module.read_text(encoding="utf-8")):
            assert ".aaaaaaaaaaaa.js" in specifier, f"{module.name} imports {specifier}"
            assert (portal / specifier).is_file(), f"{module.name} imports missing {specifier}"


def test_assembled_portal_ships_the_component_pipelines_api_fetches(tmp_path: Path) -> None:
    shutil.copytree(BROWSER, tmp_path / "artifact-browser")
    assert assemble_portal(tmp_path).returncode == 0

    portal = tmp_path / "website" / "portal-build"
    pipelines = json.loads((portal / "component-pipelines.aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    assert [stage.get("component") for stage in pipelines["rf_detr_nano"]] == ["backbone", None, "transformer"]
    api = (portal / "api.aaaaaaaaaaaa.js").read_text(encoding="utf-8")
    assert '"./component-pipelines.aaaaaaaaaaaa.json"' in api
    assert '"./component-pipelines.json"' not in api


def test_portal_assembly_rejects_relative_imports_it_does_not_deploy(tmp_path: Path) -> None:
    for specifier in ("./helpers/util.js", "../shared.js", "./unlisted.js"):
        root = tmp_path / specifier.strip("./").replace("/", "-")
        shutil.copytree(BROWSER, root / "artifact-browser")
        downloads = root / "artifact-browser" / "downloads.js"
        downloads.write_text(f'import "{specifier}";\n' + downloads.read_text(encoding="utf-8"), encoding="utf-8")

        result = assemble_portal(root)

        assert result.returncode != 0
        assert f"downloads.js imports {specifier}" in result.stderr


def test_model_cards_use_lazy_integrity_checked_progressive_disclosure() -> None:
    script = _browser_source("api.js", "model-view.js")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    renderer = (BROWSER / "markdown.js").read_text(encoding="utf-8")

    assert "documentation?.overview_markdown" in script
    assert 'headers: { Accept: "text/markdown" }' in script
    assert "/artifacts/${encodeURIComponent(documentation.artifact_name" in script
    assert 'crypto.subtle.digest("SHA-256"' in script
    assert "state.documentationCache.set(documentationKey(documentation), markdown)" in script
    assert 'disclosure("model-card", "Full model card", modelCardBody(documentation)' in script
    assert "documentationHeroUrl" in script
    assert "documentation.hero?.markdown_path" in script
    assert ".model-card-hero" in styles
    assert "SAFE_LINK" in renderer
    assert "escapeHtml" in renderer
    assert "resolveImage" in renderer
    assert "resolveDocumentationLink" in renderer
    assert 'loading="lazy"' in renderer
    assert "export function renderMarkdown" in renderer


def test_full_model_card_starts_after_the_overview_the_page_already_shows() -> None:
    script = _browser_source("model-view.js")
    start = script.index("function modelCardRemainder(")
    end = script.index("function modelCardBody(")
    readme = "# Model\n\n![Model hero](hero.png)\n\n## Overview\n\nShown above.\n\n## Usage\n\nRun it.\n"
    cases = json.dumps([readme, "# Model\n\nText"])
    probe = f"console.log(JSON.stringify({cases}.map(modelCardRemainder)));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script[start:end] + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == ["## Usage\n\nRun it.\n", None]


def test_int8_variants_are_listed_first() -> None:
    script = _browser_source("api.js")
    start = script.index("export function buildProfileId(")
    end = script.index("export function profileGroups(")
    helper = script[start:script.index("\n}\n", start) + 3]
    groups = script[end:script.index("export function activeModelBuild(")]
    builds = [{"build_profile_id": profile} for profile in ("modalix_bf16", "modalix_int8", "a_fp32", "modalix_int8_bf16")]
    probe = f"console.log(JSON.stringify(profileGroups({{builds: {json.dumps(builds)}}}).map(([id]) => id)));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", helper + groups + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == ["modalix_int8", "modalix_int8_bf16", "a_fp32", "modalix_bf16"]


def test_every_expandable_area_is_the_shared_disclosure_row() -> None:
    browser = _browser_source()
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    # One helper renders every expandable row on the model page; the catalog's More filters copies its markup.
    assert browser.count("<details") == 1
    assert '<details class="disclosure" data-disclosure=' in browser
    assert '<details id="more-filters" class="more-filters disclosure">' in html
    assert html.count("<details") == 1
    assert '<span class="disclosure-label">More filters</span><svg class="chevron"' in html
    assert ".disclosure[open] > summary > .chevron { transform: rotate(90deg); }" in styles
    # Re-rendering keeps the rows the reader opened, keyed by the row rather than by its section.
    assert "state.openDisclosures.has(key)" in browser
    assert "if (event.target.open) state.openDisclosures.add(key);" in browser
    assert "state.openDisclosures.add(`component:${runId}`)" in browser
    for removed in ["expandedRuns", "expandedBenchmarks", "expandedDocumentationModelId", "data-documentation-toggle"]:
        assert removed not in browser


def test_execution_graph_opens_its_tensors_in_the_shared_disclosure_row() -> None:
    script = _browser_source("model-view.js")
    adapters = (BROWSER / "benchmark-adapters.js").read_text(encoding="utf-8")

    assert "executionGraph: executionGraph(result.core_report?.execution_plan)" in adapters
    assert "${executionGraphSection(benchmark.executionGraph, runId)}" in script
    assert 'disclosure(`tensors:${runId}`, "Tensors", table, `${graph.tensors.length} tensors`)' in script


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


def test_developer_docs_workflow_runs_browser_tests_and_smoke_tests_the_portal() -> None:
    workflow = yaml.load(DOCS_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    runs = [step.get("run") for step in workflow["jobs"]["build"]["steps"]]

    # The orchestrator, Target Branch Checks, and the deployment call this; it has no push or pull-request trigger.
    assert set(workflow["on"]) == {"workflow_call", "workflow_dispatch"}
    assert "node --test tests/browser/*.test.mjs" in runs
    assert runs.index("npm run smoke:portal") > runs.index("npm run build:portal")
    assert "npx playwright install --with-deps chromium" in runs


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
    script = _browser_source("app.js")
    browser = _browser_source()

    assert "enrichRuns" not in browser
    assert "mapConcurrent" not in browser
    assert "readCachedCatalog" in script
    assert "writeCachedCatalog" in script
    assert "state.runs[0]?.id" not in browser


def test_branch_search_and_categories_filter_the_cached_snapshot() -> None:
    script = _browser_source("app.js", "api.js")

    assert 'const branchSnapshot = history === "latest" && branch !== "all"' in script
    assert 'const categories = branchSnapshot ? [] :' in script
    assert 'const search = branchSnapshot ? "" :' in script
    assert 'const compiler = branchSnapshot ? "any" :' in script
    assert 'if (branchSnapshotMode()) return;' in script
    assert 'if (branchSnapshotMode()) {\n    renderRuns();\n    return;\n  }' in script


def test_branch_switch_loads_builds_even_if_taxonomy_refresh_fails() -> None:
    script = _browser_source("app.js", "state.js")
    branch_change = script.split('elements["branch-filter"].addEventListener("change", async () => {', 1)[1]
    branch_change = branch_change.split('elements["compiler-filter"].addEventListener', 1)[0]

    assert 'await loadTaxonomy(state.preferredBranch || DEFAULT_BRANCH);\n  } catch (error) {' in branch_change
    assert 'setCategoryFiltersDisabled(true);' in branch_change
    assert 'if (navigationId !== state.navigationId) return;\n  try {\n    await loadRuns({ branch: state.preferredBranch });' in branch_change
    assert 'elements[elementId].disabled ? "" :' in script


def test_cached_build_history_preserves_run_mode() -> None:
    script = _browser_source("app.js", "state.js", "api.js", "catalog-view.js")
    browser = _browser_source()

    assert 'catalogMode: "runs"' in script
    assert 'state.catalogMode = modelsMode ? "models" : "runs"' in script
    assert 'mode: state.catalogMode' in script
    assert 'cached.models?.length || !cached.runs?.length ? "models" : "runs"' in script
    assert 'payload.mode !== "runs" && Object.hasOwn(payload, "models")' in script
    assert "applyCatalog(cached, { cache: false })" in script
    assert 'return state.catalogMode === "models"' in script
    assert 'elements["branch-filter"].value && !elements["history-filter"].checked' not in browser


def test_concurrent_requests_do_not_overwrite_newer_selections() -> None:
    script = _browser_source("app.js")

    assert "const requestId = ++state.runsRequestId" in script
    assert "if (requestId !== state.runsRequestId) return" in script
    assert "if (state.selectedId === runId)" in script


def test_linked_run_loads_independently_of_the_current_catalog_page() -> None:
    script = _browser_source("app.js", "compare-view.js", "model-view.js")
    browser = _browser_source()

    assert "if (navigationId !== state.navigationId) return;" in script
    assert "if (state.selectedId) await selectRun(state.selectedId, { updateUrl: false, expand: linkedRun });" in script
    assert "state.runs.some((run) => run.id === state.selectedId)" not in browser


def test_historical_model_cards_load_from_the_run_catalog_revision() -> None:
    script = _browser_source("app.js", "state.js", "api.js")
    select_run = script.split("async function selectRun(", 1)[1].split("let pendingDownloads", 1)[0]

    assert "historicalCatalogModels: new Map()" in script
    assert "async function loadHistoricalCatalogModel(run)" in script
    assert "/catalogs/${encodeURIComponent(run.commit_sha)}?${query}" in script
    assert "payload.catalog?.document?.models" in script
    assert 'state.catalogMode === "runs"' in select_run
    assert "await loadHistoricalCatalogModel(anchor)" in select_run


def test_cached_run_detail_backfills_package_when_catalog_context_changes() -> None:
    script = _browser_source("app.js")
    select_run = script.split("async function selectRun(", 1)[1].split("let pendingDownloads", 1)[0]

    assert 'const currentPackage = variantPackage(variant.runs);' in select_run
    assert 'variant.packageSummary?.run?.id !== currentPackage.run_id' in select_run
    assert '(cached || currentPackage.run_id !== variant.requestedPackageId)' in select_run
    assert 'variant.packageSummary = await api(`/v1/runs/${currentPackage.run_id}`).catch(() => null);' in select_run


def test_run_selection_keeps_the_url_branch_in_sync_with_the_filter() -> None:
    script = _browser_source("app.js")
    write_url = script.split("function writeVariantUrl(", 1)[1].split("async function selectRun(", 1)[0]

    assert 'url.searchParams.set("branch", elements["branch-filter"].value);' in write_url
    assert 'url.searchParams.set("branch", branch);' not in write_url
    assert 'state.urlBranch = link.has("branch")' in script


def test_all_branches_selection_has_an_explicit_url_value() -> None:
    script = _browser_source("app.js")
    browser = _browser_source()

    assert 'url.searchParams.set("branch", state.preferredBranch);' in script
    assert 'resetUrl.searchParams.set("branch", resetBranch);' in script
    assert 'searchParams.delete("branch")' not in browser


def test_browser_history_restores_category_filters_and_empty_detail() -> None:
    script = _browser_source("app.js")
    initialize = script.split("async function initialize()", 1)[1].split('elements["theme-toggle"]', 1)[0]

    assert 'Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, link.get(queryName) || ""])' in initialize
    assert "state.taxonomy = null" in initialize
    assert "if (state.selectedId || linkedVariant) detailSkeleton();\n  else showEmptyDetail();" in initialize
    assert "if (state.taxonomy) populateCategoryFilters();" in initialize
    assert 'window.addEventListener("popstate", () => {' in script


def test_detail_error_can_be_retried_and_closed() -> None:
    script = _browser_source("app.js")

    assert 'errorPanel(error.message, "Could not load this build", { closable: true })' in script
    assert "event.target.closest('[data-action=\"retry\"]') && state.selectedId" in script
    assert "selectRun(state.selectedId, { updateUrl: false });" in script


def test_branch_filter_defaults_to_develop_and_persists_user_selection() -> None:
    script = _browser_source("app.js", "state.js", "api.js", "catalog-view.js")

    assert 'const DEFAULT_BRANCH = "develop"' in script
    assert "sima-neat-model-branch:" in script
    assert "readBranchPreference" in script
    assert "writeBranchPreference(state.preferredBranch)" in script
    assert "branches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH" in script


def test_branch_filter_uses_an_independent_cached_registry_catalog() -> None:
    script = _browser_source("app.js", "api.js", "catalog-view.js")
    browser = _browser_source()

    assert "/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/branches" in script
    assert "branchCatalogCacheKey" in script
    assert "readCachedBranches" in script
    assert "writeCachedBranches" in script
    assert "loadTaxonomy(elements[\"branch-filter\"].value || DEFAULT_BRANCH)" in script
    assert "The registry returned an empty branch catalog; keeping saved branches." in script
    assert "knownBranches.delete(branch)" not in browser
    assert "...(payload.branches || [])" not in browser


def test_earlier_builds_pagination_tracks_the_selected_branch() -> None:
    script = _browser_source("app.js", "catalog-view.js")

    assert 'query.set("git_ref", `refs/heads/${branch}`)' in script
    assert "availableBranches" in script
    assert 'await loadRuns({ branch: elements["branch-filter"].value })' in script
    assert 'elements["load-more"].hidden = !state.hasMore' in script


def test_browser_has_persistent_dual_theme_and_accessibility_controls() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js")
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
    assert '<header class="catalog-intro">' in html
    assert "Pick a model, check its speed and accuracy, and download it for Neat." in html
    assert "--font-display" not in styles
    assert "font-family: var(--font-body)" in styles


def test_secondary_filters_are_collapsed_and_status_is_not_presented() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js", "catalog-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="filter-toggle"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="filter-drawer" class="filter-drawer" data-open="false"' in html
    assert '<details id="more-filters" class="more-filters disclosure">' in html
    assert html.index('id="task-filter"') < html.index('id="more-filters"') < html.index('id="branch-filter"')
    assert 'if (moreFiltersActive()) elements["more-filters"].open = true;' in script
    assert 'id="filter-count"' in html
    assert "--filter-surface: color-mix(in srgb, var(--surface) 88%, var(--blue-soft))" in styles
    assert "background: var(--filter-surface)" in styles
    assert '.filter-drawer[data-open="false"] { display: none; }' in styles
    assert 'id="status-filter"' not in html
    assert 'class="run-status' not in browser
    assert 'class="status-label' not in browser
    assert 'query.set("status", "completed")' in script
    assert 'run.status === "completed"' in script


def test_model_catalog_is_default_with_explicit_build_history_mode() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js")
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
    script = _browser_source("app.js", "state.js", "catalog-view.js")

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
    script = _browser_source("app.js", "catalog-view.js")

    assert "query.set(name, value)" in script
    assert "url.searchParams.set(queryName, value)" in script
    assert "url.searchParams.delete(queryName)" in script
    assert "run.architecture_id === current.architecture_id" in script
    assert "(run.tasks || []).includes(current.task)" in script


def test_smart_search_uses_names_and_taxonomy_labels_on_client_and_server() -> None:
    script = _browser_source("app.js", "catalog-view.js")

    assert "normalizedSearchTerms" in script
    assert 'taxonomyLabel("tasks", id)' in script
    assert "terms.every((term) => searchableText.includes(term))" in script
    assert 'query.set("q", elements.search.value.trim())' in script
    assert 'url.searchParams.set("q", query)' in script
    assert "setTimeout(() => loadRuns" in script


def test_browser_formats_v1_and_v2_build_identities_without_a_false_default() -> None:
    script = _browser_source("ui.js", "catalog-view.js")
    browser = _browser_source()

    assert "function profileLabel(run)" in script
    assert 'run.metadata?.build_profile_id || run.metadata?.variant_id || "Build identity unavailable"' in script
    assert 'components.length > 1 || components[0] !== "model" ? components.join(" · ") : ""' in script
    assert '<small>${escapeHtml(profileLabel(run))}</small>' in script
    assert '"default artifact"' not in browser
    assert "run.metadata?.build_profile_id, run.metadata?.component_id" in script


def test_model_card_links_to_the_exact_compiled_github_source() -> None:
    script = _browser_source("model-view.js")
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
    script = _browser_source("model-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'test.name === "mpk-validation" && test.status === "passed"' in script
    assert 'class="mpk-validation-mark" tabindex="0" role="img"' in script
    assert "MPK package integrity verified: archive structure, checksum, and build provenance match." in script
    assert '<h3>Validation</h3>' not in browser
    assert ".detail-title-actions" in styles
    compact_controls = ".model-source-link, .original-model-link, .mpk-validation-mark { width: 38px; height: 38px"
    assert compact_controls in styles
    assert ".detail-links > [data-tooltip]:hover::after" in styles
    # Narrow screens anchor tooltips to the links row so a hidden tooltip never widens the page.
    assert ".detail-links { position: relative; grid-column: 1 / -1;" in styles


def test_model_page_sections_have_accessible_jump_links() -> None:
    script = _browser_source("app.js", "model-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '<nav class="subnav" aria-label="Model page sections">' in script
    assert '<button type="button" data-scroll-to="${id}">${label}</button>' in script
    assert '["components-section", "Performance"]' in script
    assert 'accuracyMarkup && ["accuracy-section", "Accuracy"]' in script
    assert 'id="components-section" class="component-section" tabindex="-1"' in script
    assert 'id="details-section" class="details-section" tabindex="-1"' in script
    assert 'id="license-section"' in script
    assert 'event.target.closest("[data-scroll-to]")' in script
    assert "section.scrollIntoView({" in script
    assert 'matchMedia("(prefers-reduced-motion: reduce)").matches' in script
    assert ".subnav button:focus-visible" in styles
    assert "data-scroll-benchmark" not in browser
    assert "benchmark-shortcut" not in styles


def test_download_is_promoted_and_artifacts_show_only_size_and_checksum() -> None:
    script = _browser_source("model-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '<button class="download-action" type="button" data-download="package"' in script
    assert '<button class="download-action" type="button" data-download-variant=' in script
    assert '<span>Copy CLI command</span>' in script
    assert "sima-cli models download --id ${variant.modelId} --variant ${variant.profile} --branch" in script
    # The CLI fetches the catalog build, so an older build from history offers no command.
    assert '${catalogModelOfVariant(variant) ? `<button class="cli-action"' in script
    # Details lead with the package card: size, checksum, and both downloads as icon buttons.
    assert "${icons.download}<span>Download package</span>" in script
    assert "${icons.download}<span>Download components</span>" in script
    assert '<dl class="package-facts">' in script
    assert '<dt>Size</dt>' in script
    assert '<dt>SHA-256</dt>' in script
    assert "Download individual components" not in browser
    assert 'class="download-button"' not in browser
    assert 'class="artifact-main"' not in browser
    assert "artifact-download-shortcut" not in browser
    assert ".download-action:focus-visible, .cli-action:focus-visible" in styles


def test_package_download_dialog_uses_one_consistent_label_value_grid() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="download-kicker"' not in html
    assert '"download-kicker"' not in browser
    assert '<dt>File name</dt>' in script
    assert '<dt>Attribution</dt>' in script
    assert 'Attribution: ${license.attribution}' not in browser
    assert ".download-package-info dt, .download-package-info dd" in styles
    assert ".download-package-info dt { border-right: 1px solid var(--line)" in styles
    assert ".download-package-info code { font: inherit" in styles
    assert 'font-family: "IBM Plex Mono"' not in styles.split(".download-package-info code", 1)[1].split("}", 1)[0]


def test_only_model_packs_are_presented_as_artifacts() -> None:
    script = _browser_source("api.js", "compare-view.js", "model-view.js")
    browser = _browser_source()

    assert 'artifact.artifact_type === "model-pack"' in script
    assert "const artifacts = modelPackArtifacts(detail.summary.artifacts)" in script
    assert 'artifacts.map((artifact) => keyValue("SHA-256", artifact.sha256, { copy: true, mono: true }))' in script
    assert '|| artifacts[0]' not in browser


def test_detail_links_to_the_original_model_whatever_was_compiled() -> None:
    script = _browser_source("ui.js", "model-view.js")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    original_model_link = script.split("function originalModelLink(", 1)[1].split("\n}\n", 1)[0]

    # The declared card decides the link, so converted models link to their checkpoint's page too.
    assert 'url.protocol === "https:" && !url.username && !url.password' in script
    assert "trustedHttpsUrl(input.metadata?.model_card) || huggingFaceRepositoryUrl(input.source_reference)" in original_model_link
    assert "startsWith(\"hf://\")" not in original_model_link
    assert "hf:\\/\\/(models|datasets|spaces)" in script
    assert 'url.hostname === "huggingface.co"' in original_model_link
    assert 'Open the original model on ${huggingFace ? "Hugging Face" : url.hostname}' in original_model_link
    assert "originalModel.huggingFace ? icons.huggingFace : icons.external" in script
    assert 'target="_blank" rel="noreferrer"' in script
    assert 'data-tooltip="${escapeHtml(originalModel.label)}"' in script
    assert "Official no-border logo from https://huggingface.co/brand" in script
    assert 'viewBox="0 0 95 88"' in script
    assert 'fill="#FFD21E"' in script
    assert ".original-model-link:focus-visible" in styles
    assert ".original-model-link svg { width: 20px; height: 19px; }" in styles
    assert ".external-model-link svg { fill: none; stroke: currentColor;" in styles


def test_benchmark_rendering_is_schema_versioned_and_charted() -> None:
    script = _browser_source("metrics.js", "api.js", "model-view.js")
    browser = _browser_source()
    adapters = (BROWSER / "benchmark-adapters.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'import { adaptBenchmarkResult } from "./benchmark-adapters.js"' in script
    assert "new Map([[1, adaptV1], [2, adaptV2], [3, adaptV3], [4, adaptV4], [5, adaptV4]])" in adapters
    assert "Object.entries(result.metrics || {})" in adapters
    assert "Object.entries(trial)" in adapters
    assert "/benchmarks/latest" in script
    # Benchmark details group the measurements into cards; unknown metrics land in Other measurements.
    assert 'disclosure(`benchmark:${runId}`, "Benchmark details", benchmarkDetails(benchmark, payload)' in script
    for group in ['["speed", "Speed"', '["heat", "Power & heat"', '["memory", "MLA memory"', '["startup", "Startup"']:
        assert group in script
    assert 'metricGroup("other", "Other measurements"' in script
    assert "mlaFrequencyHz: result.environment?.mla_frequency_hz" in adapters
    assert "cpuFrequencyKhz: result.environment?.cpu_frequency_khz" in adapters
    assert 'clockRow("MLA clock", benchmark.mlaFrequencyHz / 1e6)' in script
    assert 'clockRow("CPU clock", benchmark.cpuFrequencyKhz / 1e3)' in script
    assert "Not reported by this runtime:" in script
    assert "data-trial-metric" not in browser
    assert "trialChart" not in browser
    assert 'item.sensor.toLowerCase() === "temp1"' in script
    assert 'class="sparkline"' in script
    assert 'role="img"' in script
    assert ".sparkline-line" in styles
    assert '.metric-group[data-group="heat"]' in styles
    assert "temperature_samples" in adapters
    assert "sensor_id" in adapters
    assert "fallbackMetricLabel" in script
    assert "mla_memory_peak_delta_mib" in script
    assert "time_to_first_inference_ms" in script


def test_accuracy_result_is_rendered_with_scope_and_reference_comparison() -> None:
    script = _browser_source("api.js", "model-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'api(`/v1/runs/${runId}/accuracy/latest`)' in script
    assert 'payload?.result || fallback?.summary' in script
    # The DevKit run wins over the legacy benchmark-run result, which wins over compile-run accuracy.
    assert "devkitPayload?.result || legacyDevkit?.summary || payload?.result" in script
    assert 'item.suite === "accuracy"' in script
    assert 'id="accuracy-section"' in script
    assert "Candidate top-1" in script
    assert "Reference ${percentage(reference.top1_accuracy)}" in script
    assert "dataset.evaluated_samples.toLocaleString()" in script
    # The status is a plain badge; its explanation sits in the Evaluation details row.
    assert '<span class="status-badge ${statusClass}">${status}</span>' in script
    assert 'disclosure("accuracy", "Evaluation details", details, coverageLabel)' in script
    assert "accuracy-explainer" not in browser
    assert "run the qualification profile for final accuracy reporting" in script
    assert 'const coverageLabel = fullCoverage ? "Full coverage" : "Sampled coverage"' in script
    assert 'const status = qualified ? "Qualification" : "Provisional"' in script
    assert 'report.validation?.status !== "provisional"' in script
    assert ".accuracy-summary" in styles
    assert ".status-badge.provisional" in styles
    assert ".status-badge.qualification" in styles
    assert ".accuracy-explainer" not in styles


def test_devkit_accuracy_is_resolved_from_the_catalog_accuracy_key_and_rendered_per_metric() -> None:
    script = _browser_source("api.js", "model-view.js")

    # One accuracy key per variant: the component build that carries it names every MPK it measured.
    assert "accuracyKeyForRuns(runs).then(devkitAccuracy)" in script
    # History mode has no catalog loaded, so the key comes from the catalog at the build's own revision.
    assert 'const current = state.catalogMode === "runs" ? null : catalogModelForRun(runs[0]);' in script
    assert "return keyIn(await loadHistoricalCatalogModel(runs[0]));" in script
    assert "runIds.has(build.run_id) && build.validation_fingerprint" in script
    assert 'run_type: "model_accuracy", fingerprint: accuracyKey, fingerprint_version: "1"' in script
    assert "return await api(`/v1/runs/${run.id}/accuracy/latest`)" in script
    assert "api(`/v1/runs/${benchmark.benchmark_run.id}/test-results`)" in script
    assert "item.summary?.schema_version === 2" in script
    assert "Object.entries(report.metrics || {})" in script
    assert 'coco_map: "mAP 50–95"' in script
    assert 'coco_ar100: "AR@100"' in script
    assert "ACCURACY_METRIC_LABELS[name] || name" in script
    # Results with more metrics than tiles list the rest in a table; each shows its published score.
    assert ".slice(0, ACCURACY_SUMMARY_TILES)" in script
    assert ".slice(ACCURACY_SUMMARY_TILES)" in script
    assert "Published ${published}" in script
    assert '<th scope="col">Published</th>' in script
    # Units come from the published result, so the formatted value is escaped before insertion.
    assert "<strong>${escapeHtml(accuracyMetricValue(metric))}</strong>" in script


def test_accuracy_metrics_follow_the_label_order_whatever_order_the_registry_returns() -> None:
    script = _browser_source("model-view.js")
    start = script.index("const ACCURACY_METRIC_LABELS = {")
    end = script.index("function accuracyPanel(")
    # The registry returned RF-DETR Small's published metrics in this order.
    metrics = ["coco_ar1", "coco_map", "coco_ap50", "coco_ap75", "coco_ar10", "coco_ar100", "coco_ap_large",
               "coco_ap_small", "coco_ar_large", "coco_ar_small", "coco_ap_medium", "coco_ar_medium"]
    report = {"metrics": {name: {"value": 0.5} for name in ["custom_b", *metrics, "custom_a"]}}
    order = f"console.log(JSON.stringify(orderedAccuracyMetrics({json.dumps(report)}).map(([name]) => name)));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script[start:end] + order],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        "coco_map", "coco_ap50", "coco_ap75", "coco_ap_small", "coco_ap_medium", "coco_ap_large",
        "coco_ar1", "coco_ar10", "coco_ar100", "coco_ar_small", "coco_ar_medium", "coco_ar_large",
        "custom_b", "custom_a",
    ]
    assert "orderedAccuracyMetrics(report).slice(0, ACCURACY_SUMMARY_TILES)" in script
    assert "orderedAccuracyMetrics(report).slice(ACCURACY_SUMMARY_TILES)" in script


def test_pose_accuracy_metrics_lead_with_keypoint_map() -> None:
    script = _browser_source("model-view.js")
    start = script.index("const ACCURACY_METRIC_LABELS = {")
    end = script.index("function accuracyPanel(")
    metrics = ["coco_person_box_ap50", "coco_keypoint_ar", "coco_keypoint_ap_large", "coco_person_box_map",
               "coco_keypoint_ar75", "coco_keypoint_map", "coco_keypoint_ar_medium", "coco_keypoint_ap75",
               "coco_keypoint_ar_large", "coco_keypoint_ap50", "coco_keypoint_ar50", "coco_keypoint_ap_medium"]
    report = {"metrics": {name: {"value": 0.5} for name in metrics}}
    order = f"console.log(JSON.stringify(orderedAccuracyMetrics({json.dumps(report)}).map(([name]) => name)));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script[start:end] + order],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        "coco_keypoint_map", "coco_keypoint_ap50", "coco_keypoint_ap75", "coco_keypoint_ap_medium",
        "coco_keypoint_ap_large", "coco_keypoint_ar", "coco_keypoint_ar50", "coco_keypoint_ar75",
        "coco_keypoint_ar_medium", "coco_keypoint_ar_large", "coco_person_box_map", "coco_person_box_ap50",
    ]
    assert 'coco_keypoint_map: "Keypoint mAP 50–95"' in script


def test_segmentation_accuracy_metrics_lead_with_mask_map() -> None:
    script = _browser_source("model-view.js")
    start = script.index("const ACCURACY_METRIC_LABELS = {")
    end = script.index("function accuracyPanel(")
    metrics = ["coco_box_ap50", "coco_mask_ar100", "coco_mask_ap_large", "coco_box_map", "coco_mask_ar1",
               "coco_mask_map", "coco_mask_ar_medium", "coco_mask_ap75", "coco_mask_ar_large", "coco_mask_ap50",
               "coco_mask_ar10", "coco_mask_ap_medium", "coco_mask_ap_small", "coco_mask_ar_small"]
    report = {"metrics": {name: {"value": 0.5} for name in metrics}}
    order = f"console.log(JSON.stringify(orderedAccuracyMetrics({json.dumps(report)}).map(([name]) => name)));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script[start:end] + order],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        "coco_mask_map", "coco_mask_ap50", "coco_mask_ap75", "coco_mask_ap_small", "coco_mask_ap_medium",
        "coco_mask_ap_large", "coco_mask_ar1", "coco_mask_ar10", "coco_mask_ar100", "coco_mask_ar_small",
        "coco_mask_ar_medium", "coco_mask_ar_large", "coco_box_map", "coco_box_ap50",
    ]
    assert 'coco_mask_map: "Mask mAP 50–95"' in script


def test_numerical_fidelity_is_rendered_from_correctness_results() -> None:
    script = _browser_source("model-view.js")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

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
    assert "<details" not in panel
    assert "${icons.download}<span>Download JSON</span>" in panel
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


def test_compare_is_a_page_of_three_pickers_over_one_table() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("state.js", "compare-view.js", "app.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '<section id="compare-pane" class="compare-pane" aria-label="Compare builds"' in html
    # Builds are picked on the Compare page, not selected while browsing.
    for removed in ("compare-dialog", "compare-tray", "catalog-compare", "data-compare-variant", "compareSelection"):
        assert removed not in html + browser + styles
    assert "export const MAX_COMPARED_BUILDS = 3;" in script
    assert 'comparePicks: ["", "", ""]' in script
    assert '<select data-pick="${index}"><option value="">Choose a build</option>' in script
    # The picks live in the link, so a comparison can be shared and survives reload and Back.
    assert 'params.getAll("build")' in script
    assert 'url.searchParams.set("view", "compare");' in script
    assert 'if (compareLink) state.comparePicks = picksFromUrl(link);' in script
    # The model page the reader came from fills the first empty picker.
    assert "if (viewed && !state.comparePicks.includes(viewed) && empty >= 0)" in script
    rows = ("Throughput", "Latency", "Board power", "Energy per inference", "Headline metric", "Published")
    for row in (f'"{label}"' for label in rows):
        assert row in script
    assert '<span class="sr-only"> (best)</span>' in script
    # Screen-reader labels are absolutely positioned; the table box must clip them on a phone.
    assert ".compare-table-wrap { position: relative; overflow-x: auto; }" in styles
    assert '.workspace:not([data-view="compare"]) .compare-pane { display: none; }' in styles


def test_compare_bolds_a_value_only_against_the_same_component() -> None:
    metrics = _browser_source("metrics.js").replace("export ", "")
    script = _browser_source("compare-view.js")
    # measuredText and bestValues, without the benchmark adapter between them.
    measured = script[script.index("function measuredText("):script.index("// Each component is measured on its own")]
    best = script[script.index("// A single-component build competes"):script.index("function performanceCell(")]

    def build(**latencies: float | None) -> dict:
        return {"components": [
            {"component": component, "measurements": {} if value is None else {
                "latency_avg_ms": {"status": "measured", "value": value, "unit": "ms"},
            }}
            for component, value in latencies.items()
        ]}

    columns = [
        build(backbone=31.6, transformer=22.7),
        build(backbone=17.9, transformer=39.7),
        build(model=12.9),
        build(model=8.4),
        build(encoder=5.0, decoder=None),
    ]
    probe = f"console.log(JSON.stringify(Object.fromEntries(bestValues({json.dumps(columns)}, 'latency_avg_ms'))));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", metrics + measured + best + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    # Lower latency wins within each component; an encoder no other build has gets no mark.
    assert json.loads(result.stdout) == {"backbone": 17.9, "transformer": 22.7, "model": 8.4}


def test_catalog_orders_a_family_from_smallest_to_largest() -> None:
    script = _browser_source("catalog-view.js")
    ordering = script[script.index("// Size words order a family"):script.index("function modelCardImage(")]
    names = [
        "YOLO26x-det", "RF-DETR Large", "YOLO26l-det", "YOLO26n-det", "RF-DETR Nano",
        "YOLO26s-det", "ResNet-50", "YOLO26m-det", "RF-DETR Medium", "RF-DETR Small",
    ]
    models = [{"display_name": name} for name in names]
    probe = (
        f"console.log(JSON.stringify({json.dumps(models)}.sort(compareModels).map((model) => model.display_name)));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", ordering.replace("export ", "") + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == [
        "ResNet-50", "RF-DETR Nano", "RF-DETR Small", "RF-DETR Medium", "RF-DETR Large",
        "YOLO26n-det", "YOLO26s-det", "YOLO26m-det", "YOLO26l-det", "YOLO26x-det",
    ]


def test_model_catalog_uses_clean_model_navigation_and_detail_build_switching() -> None:
    script = _browser_source("app.js", "state.js", "api.js", "catalog-view.js", "model-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "models: []" in script
    assert "function profileGroups(model)" in script
    assert '<button class="model-card" type="button" data-model-card=' in script
    assert 'class="detail-variant-tabs" role="tablist"' in script
    assert 'class="detail-variant-tab" type="button" role="tab"' in script
    assert '<strong>${profiles.length > 1 ? "Variants" : "Variant"}</strong>' in script
    assert "<b>Current</b>" not in browser
    assert "detail-component-tabs" not in browser
    assert "detail-component-tabs" not in styles
    assert "Add variant to comparison" not in browser
    assert "activeProfileByModel: new Map()" in script
    assert "activeComponentByProfile" not in browser
    assert "function selectedCatalogBuild(" in script
    assert "function compilerFilteredBuilds(model)" in script
    assert "compilerFilteredBuilds(model).filter" in script
    assert "build.toolchain_version === compiler" in script
    assert 'branchModels(state.compileRuns, state.packageRuns, state.catalogDefinitions, elements["compiler-filter"].value)' in script
    assert "...state.compileRuns.map((run) => run.toolchain_version)" in script
    assert "branchName(anchor.git_ref) !== linkedBranch && !listed" in script
    assert "payload.models.flatMap" not in browser
    assert '.detail-variant-tab[aria-selected="true"] {' in styles
    assert ".model-card:focus-visible" in styles
    assert "model-row" not in browser
    render_model_card = script.split("function renderModelCard(", 1)[1].split("function renderModels(", 1)[0]
    assert 'role="tablist"' not in render_model_card
    assert "data-compare-run-id" not in render_model_card
    # A card says what the model is; numbers and build state belong on the model page.
    for marker in ("metricDisplay", "Not measured", "profileLabel", "precision"):
        assert marker not in render_model_card
    assert "taxonomyLabel(\"tasks\", id)" in render_model_card
    assert "compareModels" in script.split("function renderModels(", 1)[1].split("\nfunction ", 1)[0]
    assert "const selected = profile === variant.profile;" in script
    assert "profiles.find(([id]) => id === requestedProfile) || profiles[0]" in script


def test_catalog_and_model_page_are_full_page_views() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("state.js", "model-view.js", "app.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '<section id="workspace" class="workspace" data-view="catalog">' in html
    view_rule = (
        '.workspace:not([data-view="catalog"]) .catalog-view, .workspace:not([data-view="model"]) .detail-pane, '
        '.workspace:not([data-view="compare"]) .compare-pane { display: none; }'
    )
    assert view_rule in styles
    assert "export function showView(view)" in script
    assert "elements.workspace.dataset.view = view;" in script
    # Returning to the catalog restores where the reader was in the list.
    assert 'if (current === "catalog") catalogScrollY = window.scrollY;' in script
    assert 'window.scrollTo({ top: view === "catalog" ? catalogScrollY : 0 });' in script
    assert '<nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">' in script
    assert "detail-layout" not in browser
    assert "detail-layout" not in styles
    # Focus follows the view, and Back restores the catalog position rather than the browser's own guess.
    assert 'if (focusWasHidden) elements[VIEWS[view][1]].focus({ preventScroll: true });' in script
    assert 'elements[navigation[0]].setAttribute("aria-current", "page");' in script
    assert '<h2 id="result-title" tabindex="-1">Models</h2>' in html
    assert 'history.scrollRestoration = "manual";' in script
    assert 'if (!elements["result-list"].querySelector(".model-card, .run-entry")) {' in script
    assert 'event.key === "/" && elements.workspace.dataset.view === "catalog"' in script
    assert ".run-row { width: 100%; min-width: 0;" in styles
    assert "width: calc(100% + 30px)" not in styles
    assert "@media (max-width: 840px)" in styles


def test_model_metadata_tags_use_semantic_icons() -> None:
    script = _browser_source("ui.js", "model-view.js")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    for icon in ("branch", "precision", "compiler", "license", "modelType", "architecture", "task"):
        assert f"{icon}: `<svg" in script
    assert "detailTag(icons.branch, branchName(run.git_ref), \"branch\")" in script
    assert 'class="detail-tag detail-tag-${kind}"' in script
    assert ".detail-tag svg" in styles
    assert ".detail-tag-branch svg { stroke: var(--blue); }" in styles


def test_model_license_metadata_is_rendered_with_redistribution_and_attribution() -> None:
    script = _browser_source("model-view.js")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "const license = run.metadata?.license" in script
    assert "function licensePanel(license)" in script
    assert 'detailTag(icons.license, license.spdx, "license")' in script
    assert '<p class="license-line"><strong>${escapeHtml(license.spdx)}</strong>' in script
    assert "<span>${escapeHtml(redistribution)}</span></p>" in script
    assert "`Redistribution ${license.redistribution}`" in script
    assert 'class="license-attribution"' in script
    assert ".detail-tag-license svg" in styles
    assert ".license-attribution" in styles


def test_local_preview_fixture_is_extensible_schema_v4_and_server_is_same_origin() -> None:
    # The registry still serves published schema-4 results, so the browser keeps a fixture of that shape.
    fixture_path = BROWSER / "fixtures" / "performance-result-v4.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    server = (ROOT / "artifact-browser" / "preview.py").read_text(encoding="utf-8")
    script = _browser_source("app.js")

    assert fixture["schema_version"] == 4
    assert fixture["metrics"]["mla_memory_peak_delta_mib"]["value"] == 136.5
    assert fixture["metrics"]["time_to_first_inference_ms"]["value"] == 348.7
    assert fixture["metrics"]["preview_only_extension_metric"]["unit"] == "mock-units"
    assert fixture["metrics"]["preview_only_extension_metric"]["details"]["synthetic"] is True
    assert len(fixture["trials"]) == 3
    assert 'apiBase: location.origin' in server
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
  {run_id: 'stale-transformer', build_profile_id: 'modalix_int8_bf16', component_id: 'transformer', artifact_suffix: 'transformer_bf16',
   validation_fingerprint: 'stale-accuracy-key'},
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
// A profile/component fallback supplies static metadata, never another build's accuracy key.
assert.deepEqual(models[0].builds.map(build => build.validation_fingerprint), [undefined, undefined]);
const matched = [{...catalog[0], builds: [{run_id: 'transformer', build_profile_id: 'modalix_int8_bf16',
  component_id: 'transformer', validation_fingerprint: 'accuracy-key'}]}];
assert.deepEqual(branchModels(current, [], matched)[0].builds.map(build => build.validation_fingerprint),
                 [undefined, 'accuracy-key']);
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


def test_variant_page_shows_build_facts_once_and_one_row_per_component() -> None:
    script = _browser_source("model-view.js")
    render_detail = script.split("function renderDetail(variant)", 1)[1].split("function errorPanel(", 1)[0]
    component_row = script.split("function componentRow(detail, single)", 1)[1].split("function ", 1)[0]

    # Build-level sections render once per variant, in the page order.
    order = [
        "detailBuildSwitcher(variant)", "${overviewMarkup}", 'id="components-section"', "${accuracyMarkup}",
        "packageCard(variant,", 'disclosure("provenance", "Build provenance"', "sourceInputPanel(inputs)",
        "licensePanel(license)",
    ]
    positions = [render_detail.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert all(render_detail.count(marker) == 1 for marker in order)
    # One accuracy panel for the variant, from whichever component carries it.
    assert "accuracyPanel(variantAccuracyReport(variant))" in render_detail
    assert "details.flatMap((detail) => detail.provenance.inputs || [])" in render_detail
    assert "(input) => input.sha256 || input.name" in render_detail
    assert "details.map((detail) => componentRow(detail, singleComponent))" in render_detail
    assert "component latencies do not add up to end-to-end latency" in render_detail
    # A declared pipeline orders and numbers the components under a strip; otherwise they keep run order.
    assert "componentPerformance(detail, !singleComponent, order && index + 1)" in render_detail
    assert '${order ? pipelineStrip(variant.pipeline) : ""}' in render_detail
    # Per-component facts live in the row, never on the page.
    for marker in ['keyValue("Run ID"', 'keyValue("Fingerprint"', "fidelityPanel(tests, run.id)"]:
        assert marker in component_row
        assert marker not in render_detail
    # Each component's benchmark sits in the Performance section, not in its file row.
    assert "componentBenchmark(" not in component_row
    component_performance = script.split("function componentPerformance(", 1)[1].split("function ", 1)[0]
    assert "componentBenchmark(detail.benchmark, run.id)" in component_performance
    # A component row is a disclosure keyed by its run, with compile parameters nested inside it.
    assert 'disclosure(`component:${run.id}`, single ? "Files and checksums" : component, body, meta)' in component_row
    assert 'disclosure(`parameters:${run.id}`, "Compile parameters"' in component_row
