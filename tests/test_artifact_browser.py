from __future__ import annotations

import json
import os
import re
import runpy
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from model_registry.fingerprints import BENCHMARK_RUN_TYPES
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
    "app.js", "state.js", "audience.js", "ui.js", "metrics.js", "api.js", "catalog-view.js", "compare-view.js",
    "model-view.js", "page-parts.js", "performance-section.js", "execution-section.js", "accuracy-section.js",
    "download-dialog.js",
)
# The model page: its assembly and one module per section.
MODEL_PAGE_MODULES = (
    "page-parts.js", "performance-section.js", "execution-section.js", "accuracy-section.js", "model-view.js",
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
        "model-report.js",
        "mla-layer-stats.js",
        "mla-utilization.js",
        "branch-builds.js",
        "downloads.js",
        "markdown.js",
        "brand.css",
        "styles.css",
        "config.js",
        "portal-shell.js",
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

    assert 'aria-label="Developer Portal home"' in html
    assert '<nav class="product-nav" aria-label="Models">' in html
    assert '<a id="nav-models" href="./" aria-current="page">Browse models</a>' in html
    assert '<a id="nav-compare" href="./?view=compare">Compare</a>' in html
    assert (
        '<a class="developer-only" href="./docs"><span class="nav-long">Developer docs</span>'
        '<span class="nav-short">Docs</span></a>' in html
    )
    assert 'class="sectionbar"' in html
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
        "github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'"
    )
    assert job["uses"] == "sima-neat/.github/.github/workflows/vulcan-publish-static-site.yml@main"
    assert job["permissions"] == {"contents": "read", "actions": "write", "id-token": "write"}
    # One group for every branch, so publishes from main and develop never overlap.
    assert job["concurrency"] == {
        "group": "${{ github.repository }}-model-browser-staging",
        "cancel-in-progress": "false",
    }

    resolver = workflow["jobs"]["resolve-staging-browser-config"]
    assert resolver["if"] == job["if"]
    assert "needs" not in resolver
    assert resolver["uses"] == "sima-neat/.github/.github/workflows/vulcan-resolve-config.yml@main"
    assert resolver["with"]["vulcan_env"] == "staging"
    assert workflow["jobs"]["portal"]["with"]["environment"] == "staging"
    # Model operations retain their independently selected environment.
    assert workflow["jobs"]["resolve-vulcan-config"]["with"]["vulcan_env"] == (
        "${{ needs.select.outputs.vulcan_environment }}"
    )

    publish = job["with"]
    for field, output in {
        "environment_name": "vulcan_env",
        "aws_region": "aws_region",
        "sse_kms_key_id": "docs_kms_key_id",
    }.items():
        assert publish[field] == "${{ needs.resolve-staging-browser-config.outputs." + output + " }}"
    assert publish["bucket"] == "${{ needs.resolve-staging-browser-config.outputs.docs_bucket }}"
    assert publish["role_to_assume"] == "${{ needs.resolve-staging-browser-config.outputs.docs_publisher_role_arn }}"
    assert publish["cloudfront_distribution_id"] == (
        "${{ needs.resolve-staging-browser-config.outputs.sysdoc_cloudfront_distribution_id }}"
    )
    assert publish["s3_prefix"] == "models"
    assert publish["cloudfront_paths"] == "/models /models/*"

    # The deployment publishes the site the run's portal check built and checked.
    assert workflow["jobs"]["portal"]["uses"] == "./.github/workflows/developer-docs.yml"
    assert job["needs"] == ["portal", "resolve-staging-browser-config"]
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


@pytest.fixture(scope="module")
def preview(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The address of one preview server for the tests that only read from it: starting one loads every model."""
    root = tmp_path_factory.mktemp("preview")
    (root / "index.html").write_text("portal", encoding="utf-8")
    with _preview(root) as base:
        yield base


def _get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _get_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_preview_serves_the_models_this_repository_declares(preview: str) -> None:
    models = load_models(ROOT)
    api = "/v1/repositories/sima-neat/models"
    entries = _get_json(f"{preview}{api}/models")["models"]
    catalog = _get_json(f"{preview}{api}/catalogs/current?git_ref=refs/heads/develop")
    build = entries[0]["builds"][0]
    provenance = _get_json(f"{preview}/v1/runs/{build['run_id']}/provenance")

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


def test_preview_serves_full_packages_as_the_publisher_records_them(preview: str) -> None:
    api = "/v1/repositories/sima-neat/models"
    catalog = _get_json(f"{preview}{api}/catalogs/current?git_ref=refs/heads/develop")
    packages = {
        model["id"]: package
        for model in catalog["catalog"]["document"]["models"]
        for package in model.get("packages", [])
    }
    artifacts = {
        model_id: _get_json(f"{preview}/v1/runs/{package['run_id']}")["artifacts"]
        for model_id, package in packages.items()
        if model_id in {"resnet_50", "rf_detr_nano"}
    }
    links = {
        model_id: _get_json(f"{preview}/v1/runs/{packages[model_id]['run_id']}/artifacts/package")["artifact"]
        for model_id in artifacts
    }

    # The browser's Download finds the full package by this type, as the registry publisher writes it.
    for model_id, items in artifacts.items():
        assert [(item["name"], item["artifact_type"]) for item in items] == [("package", "model-package")]
        assert items[0]["metadata"]["component_run_ids"] == packages[model_id]["component_run_ids"]
    # Every archive package is a ZIP, whatever its component count.
    assert artifacts["resnet_50"][0]["media_type"] == "application/zip"
    assert links["resnet_50"]["s3_key"].endswith("/resnet_50_w8a8_default.zip")
    assert artifacts["rf_detr_nano"][0]["media_type"] == "application/zip"
    assert links["rf_detr_nano"]["s3_key"].endswith("/rf_detr_nano_w8a8_mixed_w16a16_default.zip")
    assert 'item.artifact_type === "model-package"' in _browser_source("app.js", "download-dialog.js")


def _benchmark_run(base: str, model_id: str, profile: str, stage: str) -> dict | None:
    """What the browser loads for one stage of a build profile: the catalog's benchmark fingerprint, resolved to its
    run's report."""
    api = "/v1/repositories/sima-neat/models"
    catalog = _get_json(f"{base}{api}/catalogs/current?git_ref=refs/heads/develop")
    model = next(model for model in catalog["catalog"]["document"]["models"] if model["id"] == model_id)
    fingerprint = model.get("benchmark_fingerprints", {}).get(profile, {}).get(stage)
    if fingerprint is None:
        return None
    query = f"run_type={BENCHMARK_RUN_TYPES[stage]}&fingerprint={fingerprint}"
    try:
        run = _get_json(f"{base}{api}/resolve?{query}")["run"]
    except urllib.error.HTTPError as error:
        assert error.code == 404
        return {"fingerprint": fingerprint, "run": None}
    summary = _get_json(f"{base}/v1/runs/{run['id']}")["run"]
    parameters = _get_json(f"{base}/v1/runs/{run['id']}/provenance")["parameters"]["parameters"]
    tests = _get_json(f"{base}/v1/runs/{run['id']}/test-results")["test_results"]
    return {"fingerprint": fingerprint, "run": summary, "parameters": parameters, "tests": tests, "model": model}


def test_preview_resolves_each_benchmark_fingerprint_to_a_model_report(preview: str) -> None:
    validator = Draft202012Validator(json.loads((ROOT / "schemas" / "model-report.schema.json").read_text("utf-8")))
    performance = _benchmark_run(preview, "resnet_50", "w8a8", "performance")
    accuracy = _benchmark_run(preview, "resnet_50", "w8a8", "accuracy")
    wrong_type = _get_status(
        f"{preview}/v1/repositories/sima-neat/models/resolve?run_type=model_accuracy"
        f"&fingerprint={performance['fingerprint']}"
    )

    for stage, measured in (("performance", performance), ("accuracy", accuracy)):
        run = measured["run"]
        package = next(item for item in measured["model"]["packages"] if item["build_profile_id"] == "w8a8")
        assert run["run_type"] == BENCHMARK_RUN_TYPES[stage]
        assert run["run_fingerprint"] == measured["fingerprint"]
        assert run["metadata"] == {
            "model_id": "resnet_50", "build_profile_id": "w8a8", "stage": stage, "package_run_id": package["run_id"],
        }
        assert re.fullmatch(r"[a-f0-9]{40}", run["commit_sha"]) and run["ci_run_url"]
        assert set(measured["parameters"]) == {stage, "builds", "runtime"}
        [test] = measured["tests"]
        assert (test["suite"], test["name"], test["status"]) == (stage, "model", "passed")
        validator.validate(test["summary"])
        assert test["summary"]["stage"] == stage
    metrics = accuracy["tests"][0]["summary"]["datasets"]["imagenet-2012-val"]["metrics"]
    assert set(metrics) == {"top1_accuracy", "top5_accuracy"}
    assert accuracy["parameters"]["accuracy"]["datasets"]["imagenet-2012-val"]["id"] == "imagenet-2012"
    assert list(performance["tests"][0]["summary"]["components"]) == ["model"]
    assert wrong_type == 404


def test_preview_measures_a_multi_component_model_as_one_report(preview: str) -> None:
    measured = _benchmark_run(preview, "rf_detr_nano", "w8a8_mixed_w16a16", "performance")

    report = measured["tests"][0]["summary"]
    builds = [build for build in measured["model"]["builds"] if build["build_profile_id"] == "w8a8_mixed_w16a16"]
    # One report for the whole model, its components in the jsonb key order the registry returns.
    assert list(report["components"]) == ["decoder", "backbone"]
    assert all(component["mla_layer_stats"]["layers"] and component["mla_stage_stats"]["stages"]
               for component in report["components"].values())
    assert report["workload"] == "Backbone, Top 300 proposals on the CPU, decoder"
    assert measured["parameters"]["builds"] == {
        build["component_id"]: {"run_id": build["run_id"], "artifact_sha256": build["artifact_sha256"]}
        for build in builds
    }


def test_preview_leaves_a_profile_without_a_benchmark_run(preview: str) -> None:
    unmeasured = _benchmark_run(preview, "yolo26n", "w16a16", "performance")
    on_demand = _benchmark_run(preview, "whisper_small", "w8a16", "performance")

    # A fingerprint without a run reads as not measured at this revision.
    assert unmeasured["run"] is None
    assert on_demand is None


def test_preview_package_runs_carry_numerical_fidelity_and_mla_stage_simulation(preview: str) -> None:
    api = "/v1/repositories/sima-neat/models"
    catalog = _get_json(f"{preview}{api}/catalogs/current?git_ref=refs/heads/develop")
    model = next(model for model in catalog["catalog"]["document"]["models"] if model["id"] == "resnet_50")
    package = model["packages"][0]
    tests = _get_json(f"{preview}/v1/runs/{package['run_id']}/test-results")["test_results"]
    compile_tests = _get_json(f"{preview}/v1/runs/{model['builds'][0]['run_id']}/test-results")["test_results"]

    fidelity, simulation = tests
    assert (fidelity["suite"], fidelity["name"]) == ("correctness", "numerical-fidelity")
    assert (simulation["suite"], simulation["name"]) == ("performance", "mla-stage-simulation")
    schema = json.loads((ROOT / "schemas/mla-stage-simulation.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(simulation["summary"])
    assert [stage["stage_index"] for stage in simulation["summary"]["components"]["model"]["stages"]] == [0, 2]
    assert fidelity["summary"]["build_profile_id"] == package["build_profile_id"]
    assert fidelity["summary"]["samples"] == fidelity["summary"]["selection"]["samples"]
    assert "numerical-fidelity" not in {test["name"] for test in compile_tests}


def test_preview_serves_fixture_paths_only_inside_the_fixtures(preview: str) -> None:
    status = {}
    for request in (
        "/fixtures/model-report-performance.json",
        "/fixtures/../../README.md",
        "/fixtures/%2e%2e/%2e%2e/README.md",
        "/fixtures/../../../../../../etc/hosts",
    ):
        # Raw request lines keep the ".." that an HTTP client would normalize away.
        response = subprocess.run(
            ["curl", "-s", "--path-as-is", "-o", os.devnull, "-w", "%{http_code}", f"{preview}{request}"],
            capture_output=True, text=True, check=True,
        )
        status[request] = response.stdout

    assert status == {
        "/fixtures/model-report-performance.json": "200",
        "/fixtures/../../README.md": "404",
        "/fixtures/%2e%2e/%2e%2e/README.md": "404",
        "/fixtures/../../../../../../etc/hosts": "404",
    }


def test_combined_portal_fingerprints_mutable_browser_assets() -> None:
    assemble = PORTAL_ASSEMBLER.read_text(encoding="utf-8")

    assert "process.env.GITHUB_SHA" in assemble
    for asset in (
        "app", "state", "ui", "metrics", "api", "catalog-view", "compare-view", "model-view",
        "model-report", "mla-layer-stats", "mla-utilization", "branch-builds", "downloads", "markdown", "brand",
        "styles", "config",
    ):
        assert f"'{asset}'" in assemble
    assert "cpSync(docsBuild, output, {recursive: true})" in assemble
    assert "fingerprintedAssets" in assemble


def assemble_portal(root: Path) -> subprocess.CompletedProcess[str]:
    (root / "website" / "scripts").mkdir(parents=True)
    (root / "website" / "build").mkdir()
    shutil.copy(PORTAL_ASSEMBLER, root / "website" / "scripts" / PORTAL_ASSEMBLER.name)
    shutil.copy(PORTAL_ASSEMBLER.with_name("portal-config.mjs"), root / "website" / "scripts" / "portal-config.mjs")
    for source in ("models", "datasets", "schemas", "tools"):
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


def test_portal_config_assets_preserve_audience_and_backend_on_rollback(tmp_path: Path, monkeypatch) -> None:
    deployments = []
    for index, (environment, audience) in enumerate([
        ("production", "developer"), ("production", "customer"),
        ("staging", "developer"), ("production", "developer"),
    ]):
        root = tmp_path / str(index)
        shutil.copytree(BROWSER, root / "artifact-browser")
        monkeypatch.setenv("MODEL_BROWSER_ENVIRONMENT", environment)
        monkeypatch.setenv("MODEL_BROWSER_UI_MODE", audience)
        result = assemble_portal(root)
        assert result.returncode == 0, result.stderr
        portal = root / "website" / "portal-build"
        html = (portal / "index.html").read_text()
        filename = re.search(r'src="(config\.[a-f0-9]+\.js)"', html).group(1)
        deployments.append((filename, (portal / filename).read_text()))
    assert len({name for name, _ in deployments[:3]}) == 3
    assert deployments[0] == deployments[3]
    # Simulate retained S3 assets: each old entry point must retain its exact config.
    retained = dict(deployments)
    for name, content in deployments:
        assert retained[name] == content


def test_assembled_portal_ships_the_model_data_api_fetches(tmp_path: Path) -> None:
    shutil.copytree(BROWSER, tmp_path / "artifact-browser")
    assert assemble_portal(tmp_path).returncode == 0

    portal = tmp_path / "website" / "portal-build"
    pipelines = json.loads((portal / "component-pipelines.aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    assert [stage.get("component") for stage in pipelines["rf_detr_nano"]] == ["backbone", None, "decoder"]
    reference = json.loads((portal / "reference-metrics.aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    assert reference["rf_detr_nano"][0]["metrics"] == {"coco_ap50": 0.676, "coco_map": 0.484}
    licenses = json.loads((portal / "licenses.aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    assert licenses == json.loads((BROWSER / "licenses.json").read_text(encoding="utf-8"))
    api = (portal / "api.aaaaaaaaaaaa.js").read_text(encoding="utf-8")
    names = json.loads((portal / "dataset-names.aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    assert names["imagenet-2012"] == "ImageNet 2012"
    apps = json.loads((portal / "example-apps.aaaaaaaaaaaa.json").read_text(encoding="utf-8"))
    assert apps["rf_detr_nano"] == "https://developer.sima.ai/examples/app/object-detection%2Frfdetr-detection-segmentation"
    model_data = ("component-pipelines", "reference-metrics", "dataset-names", "example-apps", "licenses")
    for name in model_data:
        assert f'"./{name}.aaaaaaaaaaaa.json"' in api
        assert f'"./{name}.json"' not in api


def _report_metrics(stage: str) -> dict[str, dict]:
    metrics = json.loads((ROOT / "tools" / "model_report" / "metrics.json").read_text(encoding="utf-8"))
    return {name: metric for name, metric in metrics.items() if metric["stage"] == stage}


def test_browser_accuracy_labels_match_the_model_report_metrics() -> None:
    # model_report accepts these metrics; the browser must label each of them, in the same order.
    source = (BROWSER / "accuracy-metrics.js").read_text(encoding="utf-8")
    table = re.search(r"ACCURACY_METRIC_LABELS = \{(.*?)\n\};", source, re.S)
    assert table is not None
    labels = re.findall(r'^\s*(\w+): "([^"]*)",$', table.group(1), re.M)
    assert labels == [(name, metric["label"]) for name, metric in _report_metrics("accuracy").items()]


def test_browser_accuracy_scales_match_the_model_report_metrics() -> None:
    # The browser preserves metric units/directions and applies the same explicit display overrides.
    scales = subprocess.run(
        [
            "node", "--input-type=module", "--eval",
            "const {ACCURACY_METRIC_SCALES} = await import('./artifact-browser/accuracy-metrics.js');"
            "console.log(JSON.stringify(ACCURACY_METRIC_SCALES));",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )  # fmt: skip
    expected = {
        name: {"unit": metric["unit"], "direction": metric["direction"],
               **({"display": metric["display"]} if "display" in metric else {})}
        for name, metric in _report_metrics("accuracy").items()
        if (metric["unit"], metric["direction"]) != ("ratio", "higher") or "display" in metric
    }
    assert json.loads(scales.stdout) == expected


def test_browser_performance_metrics_match_the_model_report_metrics() -> None:
    # Each performance metric shows in its display unit and scale, when metrics.json gives one, else its own.
    definitions = subprocess.run(
        [
            "node", "--input-type=module", "--eval",
            "const {METRIC_DEFINITIONS} = await import('./artifact-browser/metrics.js');"
            "console.log(JSON.stringify(METRIC_DEFINITIONS));",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )  # fmt: skip
    expected = {
        name: {
            "label": metric["label"],
            "unit": metric.get("display", {}).get("unit", metric["unit"]),
            "digits": metric["digits"],
            "scale": metric.get("display", {}).get("scale", 1),
            "direction": metric["direction"],
        }
        for name, metric in _report_metrics("performance").items()
    }
    assert json.loads(definitions.stdout) == expected


def test_browser_telemetry_readings_match_the_model_report_metrics() -> None:
    # The browser takes each telemetry metric's minimum, maximum and samples from the reading metrics.json names.
    source = (BROWSER / "model-report.js").read_text(encoding="utf-8")
    table = re.search(r"TELEMETRY_METRICS = \{(.*?)\n\};", source, re.S)
    assert table is not None
    readings = {name: [field, statistic]
                for name, field, statistic in re.findall(r'(\w+): \["(\w+)", "(\w+)"', table.group(1))}
    metrics = _report_metrics("performance")
    assert readings == {name: metric["telemetry"][:2] for name, metric in metrics.items() if "telemetry" in metric}


def test_browser_benchmark_run_types_match_the_registry() -> None:
    source = (BROWSER / "model-report.js").read_text(encoding="utf-8")
    table = re.search(r"BENCHMARK_RUN_TYPES = \{(.*?)\n\};", source, re.S)
    assert table is not None
    assert dict(re.findall(r'(\w+): "(\w+)"', table.group(1))) == BENCHMARK_RUN_TYPES


def test_portal_assembly_rejects_relative_imports_it_does_not_deploy(tmp_path: Path) -> None:
    for specifier in ("./helpers/util.js", "../shared.js", "./unlisted.js"):
        root = tmp_path / specifier.strip("./").replace("/", "-")
        shutil.copytree(BROWSER, root / "artifact-browser")
        downloads = root / "artifact-browser" / "downloads.js"
        downloads.write_text(f'import "{specifier}";\n' + downloads.read_text(encoding="utf-8"), encoding="utf-8")

        result = assemble_portal(root)

        assert result.returncode != 0
        assert f"downloads.js imports {specifier}" in result.stderr


def test_model_instructions_load_inline_with_integrity_checks() -> None:
    script = _browser_source("api.js", *MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    renderer = (BROWSER / "markdown.js").read_text(encoding="utf-8")

    assert "documentation?.overview_markdown" in script
    # The Overview text opens its section without a title.
    assert 'id="overview-title"' not in script
    assert 'headers: { Accept: "text/markdown" }' in script
    assert "/artifacts/${encodeURIComponent(documentation.artifact_name" in script
    assert 'crypto.subtle.digest("SHA-256"' in script
    assert "state.documentationCache.set(documentationKey(documentation), markdown)" in script
    assert '<div data-model-card-body>${modelCardBody(documentation)}</div>' in script
    assert 'disclosure("model-card"' not in script
    assert "loadModelCard();" in script
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
    script = _browser_source(*MODEL_PAGE_MODULES)
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


def test_w8a8_variants_are_listed_first() -> None:
    script = _browser_source("api.js")
    start = script.index("export function profileGroups(")
    groups = script[start:script.index("export function activeModelBuild(")]
    builds = [{"build_profile_id": profile} for profile in ("w16a16", "w8a8", "a_fp32", "w8a8_w16a16")]
    probe = f"console.log(JSON.stringify(profileGroups({{builds: {json.dumps(builds)}}}).map(([id]) => id)));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", groups + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == ["w8a8", "w8a8_w16a16", "a_fp32", "w16a16"]


def test_catalog_view_prefers_the_current_catalog_over_cached_history() -> None:
    script = _browser_source("api.js")
    start = script.index("export function catalogModelForRun(")
    helper = script[start:script.index("\n}\n", start) + 3]
    run = {"id": "reused", "git_ref": "refs/heads/develop", "commit_sha": "old", "metadata": {"model_id": "yolo26n"}}
    probe = f"""
const state = {{
  models: [{{model_id: "yolo26n", family: "yolo", builds: [{{run_id: "reused"}}]}}],
  historicalCatalogModels: new Map([["refs/heads/develop:old:yolo26n", {{model_id: "yolo26n", family: "yolo_26"}}]]),
}};
const families = ["models", "runs"].map((mode) => {{
  state.catalogMode = mode;
  return catalogModelForRun({json.dumps(run)}).family;
}});
console.log(JSON.stringify(families));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", helper + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == ["yolo", "yolo_26"]


def test_every_expandable_area_is_the_shared_disclosure_row() -> None:
    browser = _browser_source()
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    # One helper renders every expandable row on the model page.
    assert browser.count("<details") == 1
    assert '<details class="disclosure" data-disclosure=' in browser
    assert "<details" not in html
    assert ".disclosure[open] > summary > .chevron { transform: rotate(90deg); }" in styles
    # Re-rendering keeps the rows the reader opened, keyed by the row rather than by its section.
    assert "state.openDisclosures.has(key)" in browser
    assert "if (event.target.open) state.openDisclosures.add(key);" in browser
    assert "state.openDisclosures.add(`component:${runId}`)" in browser
    for removed in ["expandedRuns", "expandedBenchmarks", "expandedDocumentationModelId", "data-documentation-toggle"]:
        assert removed not in browser


def test_execution_graph_opens_its_tensors_from_details_beside_the_diagram() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)
    app = _browser_source("app.js", "download-dialog.js")
    reports = (BROWSER / "model-report.js").read_text(encoding="utf-8")
    design = (BROWSER / "DESIGN.md").read_text(encoding="utf-8")

    assert "executionGraph: executionGraph(component?.execution_plan)," in reports
    # Every model's execution graph is a disclosure; a pipeline's switches between its components.
    assert (
        "${executionGraphSection(components.map(({ id, label, executionGraph }) => "
        "({ id: `${run.id}:${id}`, label, graph: executionGraph })), run.id)}"
    ) in script
    assert (
        "${layerAnalysisSection(components.map(({ label, layerAnalysis }) => "
        "({ label, analysis: layerAnalysis })), run.id)}"
    ) in script
    # Details sits beside the diagram and controls the tensor table, which starts closed.
    assert '<div class="execution-graph-row"><ol>${steps}</ol>${details}</div>' in script
    assert 'aria-controls="${escapeHtml(key)}" aria-expanded="${open}">Details${icons.chevron}</button>' in script
    assert '<div id="${escapeHtml(key)}"${open ? "" : " hidden"}>${table}</div>' in script
    assert '"Tensors"' not in script
    # Like a disclosure row, the table stays open across re-renders.
    assert "const open = state.openDisclosures.has(key);" in script
    assert "if (open) state.openDisclosures.add(key);" in app
    assert "The one exception is the execution graph" in design


def test_model_card_markdown_resolves_safe_relative_links_at_the_source_commit() -> None:
    script = """
      import {readFileSync} from 'node:fs';
      const moduleSource = readFileSync('./artifact-browser/markdown.js', 'utf8');
      const moduleUrl = 'data:text/javascript;base64,' + Buffer.from(moduleSource).toString('base64');
      const {renderMarkdown, resolveDocumentationLink} = await import(moduleUrl);
      const documentation = {
        source_commit: 'a'.repeat(40),
        source_path: 'models/rf_detr_nano/README.md',
      };
      const resolveLink = (source) => resolveDocumentationLink(documentation, source);
      const markdown = '[surgery](scripts/surgery.py) [section](#usage) [unsafe](javascript:alert(1))';
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
    assert f'{source_url}/models/rf_detr_nano/scripts/surgery.py' in rendered["html"]
    assert f'{source_url}/models/rf_detr_nano/README.md#usage' in rendered["html"]
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
        "DOCS_SITE_URL": "${{ inputs.environment == 'production' && 'https://developer.sima.ai'"
        " || 'https://developer-stg.sima.ai' }}",
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
    assert 'href="${baseUrl}">Browse models</a>' in config
    assert 'href="${baseUrl}?view=compare">Compare</a>' in config
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

    for color in ("#f5f5f7", "#f1f3f5", "#ffffff", "#0b1220", "#5f6972", "#dfe3e6"):
        assert color in browser_styles
        assert color in docs_styles
    assert "font-size: clamp(2rem, 3vw, 2.65rem)" in docs_styles
    assert "style: 'light'" in config


def test_browser_and_docs_draw_accents_from_the_sima_brand_palette() -> None:
    brand = (BROWSER / "brand.css").read_text(encoding="utf-8")
    browser_styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    docs_styles = DOCS_STYLES.read_text(encoding="utf-8")

    for name, color in (
        ("blue", "#5998DD"), ("teal", "#C0E8DB"), ("green", "#2A9C4F"),
        ("lime", "#A4CC2D"), ("purple", "#D1CEE6"), ("orange", "#D26728"),
    ):
        assert f"--sima-{name}: {color};" in brand
    assert '<link rel="stylesheet" href="brand.css" />' in (BROWSER / "index.html").read_text(encoding="utf-8")
    assert "@import '../../../artifact-browser/brand.css';" in docs_styles
    accent = re.compile(r"--(?:portal-)?(?:blue|green|amber|violet|brand|ifm-color-primary)[\w-]*:\s*([^;]+);")
    for styles in (browser_styles, docs_styles):
        values = accent.findall(styles)
        assert values
        assert not [value for value in values if "#" in value]


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
    script = _browser_source("app.js", "download-dialog.js")
    browser = _browser_source()

    assert "enrichRuns" not in browser
    assert "mapConcurrent" not in browser
    assert "readCachedCatalog" in script
    assert "writeCachedCatalog" in browser
    assert "state.runs[0]?.id" not in browser


def test_branch_search_and_categories_filter_the_cached_snapshot() -> None:
    script = _browser_source("app.js", "download-dialog.js", "api.js")

    assert 'const branchSnapshot = history === "latest" && branch !== "all"' in script
    assert 'const categories = branchSnapshot ? [] :' in script
    assert 'const search = branchSnapshot ? "" :' in script
    assert 'const compiler = branchSnapshot ? "any" :' in script
    assert 'if (branchSnapshotMode()) return;' in script
    assert 'if (branchSnapshotMode()) {\n    renderRuns();\n    return;\n  }' in script


def test_cached_build_history_preserves_run_mode() -> None:
    script = _browser_source("app.js", "download-dialog.js", "state.js", "api.js", "catalog-view.js")
    browser = _browser_source()

    assert 'catalogMode: "runs"' in script
    assert 'state.catalogMode = modelsMode ? "models" : "runs"' in script
    assert 'mode: state.catalogMode' in script
    assert 'cached && ["models", "runs"].includes(cached.mode) ? cached : null' in script
    assert "applyCatalog(cached, { cache: false })" in script
    assert 'return state.catalogMode === "models"' in script
    assert 'elements["branch-filter"].value && !elements["history-filter"].checked' not in browser


def test_concurrent_requests_do_not_overwrite_newer_selections() -> None:
    script = _browser_source("app.js", "download-dialog.js")

    assert "const requestId = ++state.runsRequestId" in script
    assert "if (requestId !== state.runsRequestId) return" in script
    assert "if (state.selectedId === runId)" in script


def test_linked_run_loads_independently_of_the_current_catalog_page() -> None:
    script = _browser_source("app.js", "download-dialog.js", "compare-view.js", *MODEL_PAGE_MODULES)
    browser = _browser_source()

    assert "if (navigationId !== state.navigationId) return;" in script
    assert "if (state.selectedId) await selectRun(state.selectedId, { updateUrl: false, expand: linkedRun });" in script
    assert "state.runs.some((run) => run.id === state.selectedId)" not in browser


def test_historical_model_cards_load_from_the_run_catalog_revision() -> None:
    script = _browser_source("app.js", "download-dialog.js", "state.js", "api.js")
    select_run = script.split("async function selectRun(", 1)[1].split("let pendingDownloads", 1)[0]

    assert "historicalCatalogModels: new Map()" in script
    assert "async function loadHistoricalCatalogModel(run)" in script
    assert "/catalogs/${encodeURIComponent(run.commit_sha)}?${query}" in script
    assert "payload.catalog?.document?.models" in script
    assert 'state.catalogMode === "runs"' in select_run
    assert "await loadHistoricalCatalogModel(anchor)" in select_run


def test_cached_run_detail_backfills_package_when_catalog_context_changes() -> None:
    script = _browser_source("app.js", "download-dialog.js")
    select_run = script.split("async function selectRun(", 1)[1].split("let pendingDownloads", 1)[0]

    assert 'const currentPackage = variantPackage(variant.runs);' in select_run
    assert 'variant.packageSummary?.run?.id !== currentPackage.run_id' in select_run
    assert '(cached || currentPackage.run_id !== variant.requestedPackageId)' in select_run
    assert "[variant.packageSummary, variant.packageTests] = await loadPackageRun(currentPackage.run_id);" in select_run


def test_run_selection_keeps_the_url_branch_in_sync_with_the_filter() -> None:
    script = _browser_source("app.js", "download-dialog.js")
    write_url = script.split("function writeVariantUrl(", 1)[1].split("async function selectRun(", 1)[0]

    assert 'url.searchParams.set("branch", elements["branch-filter"].value);' in write_url
    assert 'url.searchParams.set("branch", branch);' not in write_url
    assert 'state.urlBranch = !branchSelectionEnabled || link.has("branch")' in script


def test_all_branches_selection_has_an_explicit_url_value() -> None:
    script = _browser_source("app.js", "download-dialog.js")
    browser = _browser_source()

    assert 'url.searchParams.set("branch", state.preferredBranch);' in script
    assert 'resetUrl.searchParams.set("branch", resetBranch);' in script
    assert 'searchParams.delete("branch")' not in browser


def test_browser_history_restores_category_filters_and_empty_detail() -> None:
    script = _browser_source("app.js", "download-dialog.js")
    initialize = script.split("async function initialize()", 1)[1].split('elements["theme-toggle"]', 1)[0]

    assert (
        'Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => '
        '[elementId, link.get(queryName) || ""])' in initialize
    )
    assert "state.taxonomy = null" in initialize
    assert "if (state.selectedId || linkedVariant) detailSkeleton();\n  else showEmptyDetail();" in initialize
    assert "applyTaxonomy(cached.taxonomy || null, null);" in initialize
    assert 'window.addEventListener("popstate", () => {' in script


def test_detail_error_can_be_retried_and_closed() -> None:
    script = _browser_source("app.js", "download-dialog.js")

    assert 'errorPanel(error.message, "Could not load this build", { closable: true })' in script
    assert "event.target.closest('[data-action=\"retry\"]') && state.selectedId" in script
    assert "selectRun(state.selectedId, { updateUrl: false });" in script


def test_branch_filter_defaults_to_develop_and_persists_user_selection() -> None:
    script = _browser_source("app.js", "download-dialog.js", "state.js", "api.js", "catalog-view.js")

    assert 'const DEFAULT_BRANCH = branches.defaultBranch' in script
    assert "sima-neat-model-branch:" in script
    assert "readBranchPreference" in script
    assert "writeBranchPreference(state.preferredBranch)" in script
    assert "branches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH" in script


def test_branch_filter_uses_an_independent_cached_registry_catalog() -> None:
    script = _browser_source("app.js", "download-dialog.js", "api.js", "catalog-view.js")
    browser = _browser_source()

    assert "/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/branches" in script
    assert "branchCatalogCacheKey" in script
    assert "readCachedBranches" in script
    assert "writeCachedBranches" in script
    assert "The registry returned an empty branch catalog; keeping saved branches." in script
    assert "knownBranches.delete(branch)" not in browser
    assert "...(payload.branches || [])" not in browser


def test_earlier_builds_pagination_tracks_the_selected_branch() -> None:
    script = _browser_source("app.js", "download-dialog.js", "catalog-view.js")

    assert 'query.set("git_ref", gitRef)' in script
    assert "availableBranches" in script
    assert 'await loadRuns({ branch: elements["branch-filter"].value })' in script
    assert 'elements["load-more"].hidden = !state.hasMore' in script


def test_browser_has_persistent_dual_theme_and_accessibility_controls() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js", "download-dialog.js")
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


def test_browser_never_stores_a_theme_from_the_url() -> None:
    # sima-neat/models#1246: ?theme=dark|light (tests/browser/theme.test.mjs covers index.html)
    # is shown without being stored; the toggle and a stored header theme still store it.
    script = _browser_source("app.js", "download-dialog.js")
    initialize = script.split("async function initialize()", 1)[1].split("const link", 1)[0]

    assert "function setTheme(theme, { persist = true } = {})" in script
    set_theme = script.split("function setTheme(", 1)[1].split("\n}\n", 1)[0]
    assert set_theme.index("if (persist) {") < set_theme.index('localStorage.setItem("theme", theme)')
    assert set_theme.index("if (persist) {") < set_theme.index("document.cookie = `sima-neat-theme=")
    assert "delete document.documentElement.dataset.themeSource;" in set_theme
    assert 'persist: document.documentElement.dataset.themeSource !== "url"' in initialize
    assert "setTheme(theme, { persist: event.detail.persist !== false })" in script
    assert 'setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"))' in script


def test_browser_uses_refined_search_first_layout() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "Roboto:wght@400;500;600;700" in html
    assert "IBM+Plex+Mono" in html
    assert '<h1 class="sr-only">Model Browser</h1>' in html
    assert "Search models, tasks, or families" in html
    assert 'class="page-header"' not in html
    assert "Trace every" not in html
    assert '<header class="catalog-intro">' in html
    assert "Pick a model, compare performance and accuracy, and download it for Palette." in html
    assert "--font-display" not in styles
    assert "font-family: var(--font-body)" in styles


def test_developer_filters_are_directly_visible_and_status_is_not_presented() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js", "download-dialog.js", "catalog-view.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="filter-toggle"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="filter-drawer" class="filter-drawer" data-open="false"' in html
    assert 'id="more-filters"' not in html
    assert "moreFiltersActive" not in script
    assert html.index('id="task-filter"') < html.index('id="architecture-filter"') < html.index('id="branch-filter"')
    assert '<label class="developer-only branch-filter"><span>Branch</span>' in html
    assert '<label class="developer-only"><span>Compiler</span>' in html
    assert 'id="filter-count"' in html
    assert "--filter-surface: color-mix(in srgb, var(--surface) 88%, var(--blue-soft))" in styles
    assert "background: var(--filter-surface)" in styles
    assert '.filter-drawer[data-open="false"] { display: none; }' in styles
    assert 'id="status-filter"' not in html
    assert 'class="run-status' not in browser
    assert 'class="status-label' not in browser
    assert 'status: "completed"' in script
    assert 'run.status === "completed"' in script


def test_model_catalog_is_default_with_explicit_build_history_mode() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js", "download-dialog.js")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="history-filter" type="checkbox" role="switch"' in html
    assert "Switches from models to chronological build runs." in html
    assert "Show build history" in html
    assert 'if (historical) {' in script
    assert '["model_compile", "model_package"]' in script
    assert 'allPages(api, `${base}/runs`, query, "runs")' in script
    assert 'run_type: "model_compile"' in script
    assert (
        'if (!elements["history-filter"].checked) query.set("latest_per_model_compiler", "true")'
        in script
    )
    assert 'link.get("history") === "1"' in script
    assert 'url.searchParams.set("history", "1")' in script
    assert 'elements["history-filter"].checked = false' in script
    assert ".switch-row input:checked + i" in styles


def test_browser_loads_taxonomy_from_the_current_catalog_and_exposes_category_facets() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js", "download-dialog.js", "state.js", "catalog-view.js")

    for element_id in ("task-filter", "architecture-filter"):
        assert f'id="{element_id}"' in html
    for removed in ("model-type-filter", "paradigm-filter", "input-modality-filter"):
        assert f'id="{removed}"' not in html
        assert removed not in script
    assert "/catalogs/current?${query}" in script
    assert "{ git_ref: `refs/heads/${branch}` }" in script
    assert "applyTaxonomy(document?.taxonomy || null, snapshotBranch)" in script
    assert '"task-filter": "task"' in script
    assert '"architecture-filter": "architecture_id"' in script
    assert "setCategoryFiltersDisabled(!taxonomy)" in script
    assert "Category filters are temporarily unavailable" in script


def test_filters_list_only_used_values_and_hide_without_a_choice() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("catalog-view.js")

    assert '<span>Model family</span><select id="architecture-filter">' in html
    assert ".filter(underOtherSelection).flatMap((record) => record[field] ?? [])" in script
    assert "return { ...build.run, ...catalogCategories(model) };" in _browser_source("api.js")
    model_view = _browser_source(*MODEL_PAGE_MODULES)
    assert "const categories = catalogModel?.categories || run.model_categories || {};" in model_view
    assert 'element.closest("label").hidden = element.options.length < 3 && !element.value' in script
    assert 'hideFilterWithoutChoice(elements["compiler-filter"])' not in script
    assert 'hideFilterWithoutChoice(elements["branch-filter"])' not in script


def test_category_filters_are_server_side_and_shareable() -> None:
    script = _browser_source("app.js", "download-dialog.js", "catalog-view.js")

    assert "query.set(name, value)" in script
    assert "url.searchParams.set(queryName, value)" in script
    assert "url.searchParams.delete(queryName)" in script
    assert "run.architecture_id === current.architecture_id" in script
    assert "(run.tasks || []).includes(current.task)" in script


def test_smart_search_uses_names_and_taxonomy_labels_on_client_and_server() -> None:
    script = _browser_source("app.js", "download-dialog.js", "catalog-view.js")

    assert "normalizedSearchTerms" in script
    assert 'taxonomyLabel("tasks", id)' in script
    assert "terms.every((term) => searchableText.includes(term))" in script
    assert 'query.set("q", elements.search.value.trim())' in script
    assert 'url.searchParams.set("q", query)' in script
    assert "setTimeout(() => loadRuns" in script


def test_browser_formats_build_identities_without_a_false_default() -> None:
    script = _browser_source("ui.js", "catalog-view.js")
    browser = _browser_source()

    assert "function profileLabel(run)" in script
    assert "return variantLabel(run.metadata.build_profile_id);" in script
    assert 'components.length > 1 || components[0] !== "model" ? components.join(" · ") : ""' in script
    assert '<small>${escapeHtml(profileLabel(run))}</small>' in script
    assert '"default artifact"' not in browser
    assert "run.metadata?.build_profile_id, run.metadata?.component_id" in script


def test_model_source_link_opens_the_model_card_commit() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "function githubModelUrl(run, modelId, sourceCommit = run.commit_sha)" in script
    assert 'const runRepository = String(run.repository || "").toLowerCase()' in script
    assert "runRepository !== configuredRepository" in script
    assert "/tree/${commit}/models/${encodeURIComponent(modelId)}`" in script
    assert r"/^[a-f0-9]{40}$/i.test(commit)" in script
    assert 'class="model-source-link"' in script
    # CI reuses a build until its compile inputs change, so the link follows the model card, not the build.
    assert "const sourceCommit = catalogModel?.documentation?.source_commit || run.commit_sha;" in script
    assert "githubModelUrl(run, model, sourceCommit)" in script
    assert "View ${model} source on GitHub at commit ${shortSha(sourceCommit)}" in script
    # Build provenance keeps the commit the build was compiled at.
    assert 'keyValue("Commit", run.commit_sha, { copy: true, mono: true })' in script
    assert 'target="_blank" rel="noreferrer"' in script
    assert ".model-source-link:focus-visible" in styles


def test_performance_and_accuracy_name_the_commit_and_workflow_that_measured_them() -> None:
    api = _browser_source("api.js")
    script = _browser_source(*MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    measurement_source = script.split("function measurementSource(run, modelId)", 1)[1].split("\nfunction ", 1)[0]

    # A reused build is measured by later runs, whose commit and workflow the build run does not carry.
    assert "api(`/v1/runs/${resolved.id}`).catch(() => null)," in api
    assert "run: summary?.run || resolved," in api
    # Each result links to the model source and the workflow of its own run, never the build's.
    assert "githubModelUrl(run, modelId)" in measurement_source
    assert "run.ci_run_url" in measurement_source
    assert r"/^[a-f0-9]{40}$/i.test(commit)" in measurement_source
    assert 'target="_blank" rel="noreferrer"' in measurement_source
    assert "${measurementSource(run, modelId)}</p>" in script
    assert "performanceSummary(result.value, run, modelId)" in script
    assert "measurementSource(variant.accuracy.run, modelId)" in script
    # The accuracy line holds the measurement date and, in the developer view, its source.
    assert '<p class="accuracy-scope">${measured}</p>' in script
    assert ".measurement-link {" in styles


def test_archive_validation_is_a_compact_accessible_status_mark() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '["mpk-validation", "deployment-validation"].includes(test.name) && test.status === "passed"' in script
    assert 'class="mpk-validation-mark" tabindex="0" role="img"' in script
    assert '"Model deployment" : "MPK package"' in script
    assert "integrity verified: file structure, checksum, and build provenance match." in script
    assert '<h3>Validation</h3>' not in browser
    assert ".detail-title-actions" in styles
    compact_controls = (
        ".model-source-link, .original-model-link, .example-app-link, .mpk-validation-mark { width: 38px; height: 38px"
    )
    assert compact_controls in styles
    assert ".detail-links > [data-tooltip]:hover::after" in styles
    # A hovered link lifts, which traps its tooltip's z-index, so the link itself sits above the section bar.
    assert ".detail-links > [data-tooltip] { position: relative; z-index: 11; }" in styles
    assert ".subnav { position: sticky;" in styles and "z-index: 10; display: flex;" in styles
    assert ".detail-links > [data-tooltip]:hover { position: static; transform: none; }" in styles
    # Narrow screens anchor tooltips to the links row so a hidden tooltip never widens the page.
    assert ".detail-title-actions { position: relative;" in styles


def test_model_page_sections_have_accessible_jump_links() -> None:
    script = _browser_source("app.js", "download-dialog.js", *MODEL_PAGE_MODULES)
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
    script = _browser_source(*MODEL_PAGE_MODULES)
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '<button class="download-action" type="button" data-download="package"' in script
    assert '<button class="download-action" type="button" data-download-variant=' in script
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    app = _browser_source("app.js", "download-dialog.js")
    assert 'id="download-copy-cli"' in html
    assert 'class="cli-action"' not in script
    assert "modelDownloadCommand(run.metadata.model_id, runProfile(run)" in app
    # The CLI fetches the catalog build, so an older build from history offers no command.
    assert "const current = state.models.some" in app
    # Details lead with the package card: size, checksum, and the download as an icon button.
    assert "${icons.download}<span>Download package</span>" in script
    assert "${icons.download}<span>Download components</span>" not in script
    assert '<dl class="package-facts">' in script
    assert '<dt>Size</dt>' in script
    assert 'Manifest SHA-256' in script
    assert "Download individual components" not in browser
    assert 'class="download-button"' not in browser
    assert 'class="artifact-main"' not in browser
    assert "artifact-download-shortcut" not in browser
    assert ".download-action:focus-visible" in styles


def test_package_download_dialog_uses_one_consistent_label_value_grid() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("app.js", "download-dialog.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'id="download-kicker"' not in html
    assert '"download-kicker"' not in browser
    assert '<dt>File name</dt>' in script
    assert '<dt>Attribution</dt>' not in script
    assert '<dt>SHA-256</dt>' not in script
    assert 'Attribution: ${license.attribution}' not in browser
    assert ".download-package-info dt, .download-package-info dd" in styles
    assert ".download-package-info dt { border-right: 1px solid var(--line)" in styles
    assert ".download-package-info code { font: inherit" in styles
    assert 'font-family: "IBM Plex Mono"' not in styles.split(".download-package-info code", 1)[1].split("}", 1)[0]


def test_only_model_packs_are_presented_as_artifacts() -> None:
    script = _browser_source("api.js", "compare-view.js", *MODEL_PAGE_MODULES)
    browser = _browser_source()

    assert 'artifact.artifact_type === "model-pack"' in script
    assert "const artifacts = modelPackArtifacts(detail.summary.artifacts)" in script
    assert (
        'artifacts.map((artifact) => keyValue(artifactDelivery(artifact) === "archive" ? "SHA-256" : '
        '"Manifest SHA-256", artifact.sha256, { copy: true, mono: true }))' in script
    )
    assert '|| artifacts[0]' not in browser


def test_detail_links_to_the_original_model_whatever_was_compiled() -> None:
    script = _browser_source("ui.js", *MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")
    original_model_link = script.split("function originalModelLink(", 1)[1].split("\n}\n", 1)[0]

    # The declared card decides the link, so converted models link to their checkpoint's page too.
    assert 'url.protocol === "https:" && !url.username && !url.password' in script
    assert (
        "trustedHttpsUrl(input.metadata?.model_card) || huggingFaceRepositoryUrl(input.source_reference)"
        in original_model_link
    )
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
    assert ".external-model-link svg, .example-app-link svg { fill: none; stroke: currentColor;" in styles


def test_model_page_links_the_example_application_model_yaml_names() -> None:
    api = _browser_source("api.js")
    script = _browser_source("model-view.js")
    example_app_link = script.split("function exampleAppLink(", 1)[1].split("\n}\n", 1)[0]

    # The page reaches the browser as a file built from model.yaml, so a model without one shows no link.
    assert 'fetch(new URL("./example-apps.json", import.meta.url))' in api
    assert "exampleApps.then((apps) => apps[modelId] || null)" in api
    assert "${exampleAppLink(variant.exampleApp)}</div>" in script
    assert "const url = trustedHttpsUrl(exampleApp);" in example_app_link
    assert 'if (!url) return "";' in example_app_link
    assert 'target="_blank" rel="noreferrer"' in example_app_link
    assert "${icons.apps}" in example_app_link


def test_performance_rendering_reads_the_model_report_and_is_charted() -> None:
    script = _browser_source("metrics.js", "api.js", *MODEL_PAGE_MODULES)
    browser = _browser_source()
    reports = (BROWSER / "model-report.js").read_text(encoding="utf-8")
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'import { readPerformance } from "./model-report.js";' in script
    assert 'import { readAccuracy } from "./model-report.js";' in script
    # Version 1 of the model report is the one format; any other reads as unsupported.
    assert 'report?.schema === "sima.model-report" && report.schema_version === 1 && report.stage === stage' in reports
    assert 'item.suite === stage && item.name === "model"' in script
    assert "This result format is not supported by this browser." in script
    assert '<p class="muted-message">Not measured at this revision.</p>' in script
    # Each headline tile shows one number; Performance Details has three cards on the left and two on the right.
    assert 'KPI_METRICS.map((key) => kpiTile(key, performance.measurements[key])).join("") + socPowerTiles(' in script
    assert '"Performance Details", details.markup' in script
    for card in ['["performance", "Performance"', '["heat", "Heat"', '["cpu", "CPU"', '["power", "Power"',
                 '["memory", "Memory"']:
        assert card in script
    assert 'const RIGHT_COLUMN_CARDS = new Set(["power", "memory", "other"]);' in script
    assert '["Throughput", ["throughput_fps"]' in script
    assert "sync_throughput_fps" not in browser
    assert "batchSize" not in browser
    assert "cpuCores: cpuCores(report.telemetry)" in reports
    assert "telemetry?.cpu_cores_usage_pct" in reports
    assert "coreLoad(performance.cpuCores)" in script
    assert '<p class="metric-caption core-legend"><span><b></b>Average</span><span><i></i>Peak</span></p>' in script
    assert ".core-load {" in styles
    assert '<span class="qualifier">' in script
    assert '["Latency (p50 / p95 / p99)", ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]' in script
    assert "`${performance.cpuCores.length} cores`" in script
    assert 'const rows = [["Section", "Metric", "Value", "Unit", "Key", "Note"]];' in script
    assert 'download="${escapeHtml(csvFileName(performance))}">${icons.download}Download CSV' in script
    assert "data:text/csv;charset=utf-8," in script
    # SoC power is the sum of its MLA and Non-MLA rails, nested under SoM; its peak is an upper bound.
    assert '["SoC", "soc", "MLA plus Non-MLA.' in script
    assert '["Non-MLA", "power_rail_soc_ddr_vdd_watts",' in script
    assert 'const SOC_RAILS = ["power_rail_mla_watts", "power_rail_soc_ddr_vdd_watts"];' in script
    assert 'return [reading(sum("mean")), reading(sum("maximum"))];' in script
    assert "≤" not in script
    # The other SoM rails are rows at the SoC's level, and the bar above the table replaces the legend.
    assert (
        "...Object.keys(RAILS).filter((key) => !SOC_RAILS.includes(key))"
        ".map((key) => [RAILS[key][0], key, RAILS[key][1], null, 1])" in script
    )
    assert "rail-legend" not in script and "rail-legend" not in styles
    assert 'title="Average SoM power by rail"' in script
    assert '</h4>${lead || ""}<table>' in script
    assert ".rail-split > .rail-soc {" in styles
    assert "repeating-linear-gradient(135deg, var(--red) 0 2px" in script
    assert ".detail-card tbody tr.depth-2 th" in styles
    assert '["SoM", ["avg_board_temperature_c", "peak_board_temperature_c"]' in script
    assert '["EV74", ["ev74_memory_avg_mib", "ev74_memory_peak_mib"]' in script
    assert '["Linux (board)", ["avg_linux_memory_mib", "peak_linux_memory_mib"]' in script
    assert "cma_memory" not in script
    assert "power: { lead: railSplit(measurements) }" in script
    assert 'power_rail_mla_watts: ["MLA core (0.68 V)"' in script
    assert ".rail-split" in styles
    assert ".detail-cards" in styles
    assert ".info-mark::after { content: attr(data-tooltip);" in styles
    assert "telemetryText(performance.telemetry)" in script
    assert "metric-group" not in browser
    assert "Other measurements</h4>" in script
    assert "FrequencyHz" not in browser and "FrequencyKhz" not in browser
    assert 'tool: pyneatVersion(report),' in reports
    assert '["Runtime", performance.tool]' in script
    assert '<footer class="performance-footer"><dl class="performance-provenance">' in script
    assert ".performance-action {" in styles
    # Raw result downloads the report the performance run published.
    assert 'data-download="model-report" data-run-id="${escapeHtml(run.id)}">${icons.download}Raw result' in script
    assert "data-trial-metric" not in browser
    assert "trialChart" not in browser
    assert "fallbackMetricLabel" in script
    for removed in ("energy_joules:", "model_load_ms", "time_to_first_inference_ms", "mla_memory_peak_delta_mib"):
        assert removed not in browser


def test_preview_accuracy_uses_the_regression_metrics_without_an_authored_inventory() -> None:
    preview = runpy.run_path(str(BROWSER / "preview.py"))
    report = preview["_accuracy_report"](load_models(ROOT)["resnet_50"], "w8a8")
    assert set(report["datasets"]["imagenet-2012-val"]["metrics"]) == {"top1_accuracy", "top5_accuracy"}


def test_accuracy_result_is_rendered_with_coverage_and_reference_comparison() -> None:
    script = _browser_source("api.js", *MODEL_PAGE_MODULES)
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    # The accuracy run's report is the only accuracy source.
    assert 'loadBenchmarkRun("accuracy", fingerprints.accuracy)' in script
    assert 'id="accuracy-section"' in script
    assert "dataset.evaluated_samples.toLocaleString()" in script
    # The dataset line states the sample count; the coverage explanation sits in Evaluation details.
    assert "? `using ${evaluated} samples`" in script
    assert "status-badge" not in script
    assert 'disclosure(`accuracy:${name}`, "Evaluation details", evaluation, coverageLabel)' in script
    assert 'const coverageLabel = full ? "Full coverage" : "Sampled coverage"' in script
    # One section holds every dataset's result, and each dataset line says when it was measured.
    assert 'data-accuracy-dataset="${escapeHtml(name)}"' in script
    assert "`Measured ${escapeHtml(formatDate(timestamp, false))}`" in script
    # The dataset and its sample count sit behind the info mark of the title they describe.
    assert 'datasets.length === 1 ? evaluatedOn(datasets[0].dataset, target) : "")}' in script
    assert "<h4>${escapeHtml(name)}${infoMark(evaluatedOn(dataset, target))}</h4>" in script
    assert "performanceMethod(variant))}" in script
    assert "accuracy-explainer" not in browser
    for removed in ("Qualification", "Provisional", "Candidate top-1", "validation?.status"):
        assert removed not in script
    # A build without an accuracy run still shows what the model's authors published.
    assert 'if (result.kind === "missing") return referenceAccuracySection(reference, variant);' in script
    assert ".accuracy-summary" in styles
    assert ".status-badge" not in styles
    assert ".accuracy-explainer" not in styles


def test_benchmark_runs_are_resolved_from_catalog_fingerprints_and_accuracy_is_rendered_per_metric() -> None:
    script = _browser_source("api.js", *MODEL_PAGE_MODULES, "accuracy-metrics.js")

    # The catalog records each build profile's benchmark fingerprints on the model; each names every MPK it measured.
    assert "const fingerprints = await benchmarkFingerprints(runs, runProfile(anchor));" in script
    assert "return lists(model) ? model.benchmark_fingerprints?.[profile] || {} : {};" in script
    # History mode has no catalog loaded, so the fingerprints come from the catalog at the build's own revision.
    assert 'const current = state.catalogMode === "runs" ? null : catalogModelForRun(runs[0]);' in script
    assert "const model = lists(current) ? current : await loadHistoricalCatalogModel(runs[0]);" in script
    assert "run_type: BENCHMARK_RUN_TYPES[stage], fingerprint: benchmarkFingerprint" in script
    assert "if (error.status === 404) return null;" in script
    assert "api(`/v1/runs/${resolved.id}/provenance`)" in script
    assert "api(`/v1/runs/${resolved.id}/test-results`)" in script
    assert "orderedMetricNames(Object.keys(metrics))" in script
    assert 'coco_map: "mAP 50–95"' in script
    assert 'coco_ar100: "AR@100"' in script
    assert "ACCURACY_METRIC_LABELS[name] || name" in script
    # Results with more metrics than tiles list the rest in a table; each shows the reference score.
    assert ".slice(0, ACCURACY_SUMMARY_TILES)" in script
    assert ".slice(ACCURACY_SUMMARY_TILES)" in script
    assert "`${label} ${score}`" in script
    assert '<th scope="col">${escapeHtml(REFERENCE_ORIGINS[origin].label)}</th>' in script
    # Reference scores sit beside a measurement of the same dataset only, each origin apart; the headline is the
    # first dataset's published score.
    assert "datasetReferences(reference, entry.dataset.id)" in script
    assert "referenceScores(reference, datasetId, origin)" in script
    assert "const { metrics, dataset } = result.value.datasets[0];" in script
    assert "metric.reference" not in script
    assert "<strong>${escapeHtml(formatAccuracy(name, value))}</strong>" in script


def test_numerical_fidelity_is_rendered_from_correctness_results() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert 'item.name === "numerical-fidelity"' in script
    assert "fidelityPanel(variant.packageTests?.test_results || [], fullPackage.run_id)" in script
    assert 'data-download="numerical-fidelity"' in script
    panel = script.split("function fidelityPanel(", 1)[1].split("function licensePanel(", 1)[0]
    assert '<div class="fidelity-metrics">${metrics}</div>' in panel
    assert panel.count('class="fidelity-metric"') == 2
    assert '<span>${escapeHtml(metric.name)} · Cosine similarity</span>' in panel
    assert '<span>${escapeHtml(metric.name)} · Normalized RMSE</span>' in panel
    assert '<table>' not in panel
    assert panel.index('class="fidelity-metrics"') < panel.index('class="muted-message"')
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


def test_compare_explores_one_task_with_a_chart_and_selected_table() -> None:
    html = (BROWSER / "index.html").read_text(encoding="utf-8")
    script = _browser_source("state.js", "compare-view.js", "app.js", "download-dialog.js")
    browser = _browser_source()
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert '<section id="compare-pane" class="compare-pane" aria-label="Compare builds"' in html
    # Builds are picked on the Compare page, not selected while browsing.
    for removed in ("compare-dialog", "compare-tray", "catalog-compare", "data-compare-variant", "compareSelection"):
        assert removed not in html + browser + styles
    assert "export const MAX_COMPARED_BUILDS = 3;" in script
    assert 'comparePicks: ["", "", ""]' in script
    assert 'selectControl("Task", "task"' in script
    assert 'const options = taskOptions;' in script
    assert 'class="compare-tooltip"' in script
    assert 'class="compare-chart"' in script
    assert 'data-compare-toggle=' in script
    assert "filter(task => comparableTask(task, all))" in script
    # The picks live in the link, so a comparison can be shared and survives reload and Back.
    assert 'params.getAll("build")' in script
    assert 'url.searchParams.set("view", "compare");' in script
    assert 'if (compareLink) state.comparePicks = picksFromUrl(link);' in script
    # The model page the reader came from fills the first empty picker.
    assert "if (viewed && !state.comparePicks.includes(viewed) && empty >= 0)" in script
    rows = ("Throughput", "Latency", "SoM power", "Energy per inference", "Headline metric", "Published")
    for row in (f'"{label}"' for label in rows):
        assert row in script
    assert '<span class="sr-only"> (best)</span>' in script
    # Screen-reader labels are absolutely positioned; the table box must clip them on a phone.
    assert ".compare-table-wrap { position: relative; overflow-x: auto; }" in styles
    assert '.workspace:not([data-view="compare"]) .compare-pane { display: none; }' in styles


def test_compare_loads_only_the_results_it_shows() -> None:
    compare = _browser_source("compare-view.js")
    api = _browser_source("api.js")

    # A model page's compile and package runs cost five requests per variant that Compare never shows.
    assert "loadBenchmarkResults(option.model, option.profile)" in compare
    assert "loadVariant" not in compare and "loadRunDetail" not in compare
    loader = api.split("export async function loadBenchmarkResults(", 1)[1].split("\n}\n", 1)[0]
    assert "loadRunDetail" not in loader and "loadPackageRun" not in loader


def test_compare_lists_every_build_profile_of_a_multi_component_model() -> None:
    api = _browser_source("api.js")
    script = _browser_source("compare-view.js")
    grouping = "".join(
        api[api.index(f"export function {name}("):api.index(end)].replace("export ", "")
        for name, end in (
            ("anchorBuild", "function modelCategory("),
            ("profileGroups", "export function activeModelBuild("),
        )
    )
    variants = script[script.index("function catalogVariants("):script.index("function measuredText(")]
    builds = [
        {"model_id": "rf_detr_small", "build_profile_id": profile, "component_id": component,
         "run_id": f"{profile}-{component}"}
        for profile in ("w8a8_w16a16", "w8a16_w16a16") for component in ("backbone", "decoder")
    ]
    stubs = (
        f"const state = {{models: [{{model_id: 'rf_detr_small', builds: {json.dumps(builds)}}}]}};"
        "const compareModels = () => 0; const compilerFilteredBuilds = (model) => model.builds;"
        "const pickKey = (modelId, profile) => `${modelId}.${profile}`;"
    )
    probe = "console.log(JSON.stringify(catalogVariants().map(({key, build}) => [key, build.run_id])));"
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", stubs + grouping + variants + probe],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    # Each profile is one pick, anchored on its first component; the variant then reads its whole-model result.
    assert json.loads(result.stdout) == [
        ["rf_detr_small.w8a8_w16a16", "w8a8_w16a16-backbone"],
        ["rf_detr_small.w8a16_w16a16", "w8a16_w16a16-backbone"],
    ]


def test_compare_bolds_the_best_whole_model_value() -> None:
    metrics = _browser_source("metrics.js").replace("export ", "")
    script = _browser_source("compare-view.js")
    # measuredText and bestValue, without the model report reader between them.
    measured = script[script.index("function measuredText("):script.index("function variantMeasurements(")]
    best = script[script.index("function bestValue("):script.index("function performanceCell(")]

    def build(latency: float | None, throughput: float | None = None) -> dict:
        measure = lambda value: {"status": "measured", "value": value}  # noqa: E731
        return {"measurements": {
            **({} if latency is None else {"latency_avg_ms": measure(latency)}),
            **({} if throughput is None else {"throughput_fps": measure(throughput)}),
        }}

    def best_value(columns: list[dict], key: str) -> float | None:
        probe = f"console.log(JSON.stringify(bestValue({json.dumps(columns)}, '{key}')));"
        result = subprocess.run(
            ["node", "--input-type=module", "--eval", metrics + measured + best + probe],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        return json.loads(result.stdout)

    columns = [build(31.6, 120.0), build(12.9, 410.5), build(None, 380.0)]
    # Lower latency and higher throughput win across whole models; one measured value has nothing to beat.
    assert best_value(columns, "latency_avg_ms") == 12.9
    assert best_value(columns, "throughput_fps") == 410.5
    assert best_value([build(8.4), build(None)], "latency_avg_ms") is None


@pytest.mark.parametrize("family", ["YOLO26", "YOLO11", "YOLOv8"])
def test_catalog_orders_a_family_from_smallest_to_largest(family: str) -> None:
    script = _browser_source("catalog-view.js")
    ordering = script[script.index("// Size words order a family"):script.index("function modelCardImage(")]
    names = [
        f"{family}x-det", "RF-DETR Large", f"{family}l-det", f"{family}n-det", "RF-DETR Nano",
        f"{family}s-det", "ResNet-50", f"{family}m-det", "RF-DETR Medium", "RF-DETR Small",
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
        f"{family}n-det", f"{family}s-det", f"{family}m-det", f"{family}l-det", f"{family}x-det",
    ]


def test_model_catalog_uses_clean_model_navigation_and_detail_build_switching() -> None:
    script = _browser_source(
        "app.js", "download-dialog.js", "state.js", "api.js", "catalog-view.js", *MODEL_PAGE_MODULES,
    )
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
    assert 'branchModels(state.compileRuns, state.packageRuns, state.catalogDefinitions,' in script
    assert 'state.catalogDefinitions,\n      elements["compiler-filter"].value);' in script
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
    script = _browser_source("state.js", *MODEL_PAGE_MODULES, "app.js", "download-dialog.js")
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
    assert 'if (!elements["result-list"].querySelector(".model-card, .run-entry")) renderCatalogLoading();' in script
    assert 'event.key === "/" && elements.workspace.dataset.view === "catalog"' in script
    assert ".run-row { width: 100%; min-width: 0;" in styles
    assert "width: calc(100% + 30px)" not in styles
    assert "@media (max-width: 840px)" in styles


def test_model_metadata_tags_use_semantic_icons() -> None:
    script = _browser_source("ui.js", *MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    for icon in ("branch", "compiler", "license", "modelType", "architecture", "task"):
        assert f"{icon}: `<svg" in script
    assert "detailTag(icons.branch, branchName(run.git_ref), \"branch\")" in script
    assert 'class="detail-tag detail-tag-${kind}"' in script
    assert ".detail-tag svg" in styles
    assert ".detail-tag-branch svg { stroke: var(--blue); }" in styles


def test_model_license_metadata_is_rendered_with_redistribution_and_attribution() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)
    styles = (BROWSER / "styles.css").read_text(encoding="utf-8")

    assert "const license = catalogModel?.license || run.metadata?.license" in script
    assert "function licensePanel(license, listed)" in script
    assert 'detailTag(icons.license, license.spdx, "license")' in script
    # The section names the license once; an unlisted license falls back to its identifier and model.yaml.
    assert '<p class="license-line"><strong>${escapeHtml(listed?.name || license.spdx)}</strong></p>' in script
    assert "listed?.summary || (license.redistribution ? `Redistribution ${license.redistribution}.` : \"\")" in script
    assert 'class="license-attribution"' in script
    assert ".detail-tag-license svg" in styles
    assert ".license-attribution" in styles


def test_model_page_names_the_license_and_what_it_asks_from_the_license_list() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)

    assert "licensePanel(license, variant.licenses?.[license?.spdx])" in script
    assert '<p class="license-summary">${escapeHtml(terms)}</p>' in script
    assert ".license-summary" in (BROWSER / "styles.css").read_text(encoding="utf-8")


def test_model_page_offers_artifact_downloads_only_under_a_downloadable_license() -> None:
    script = _browser_source(*MODEL_PAGE_MODULES)
    render_detail = script.split("export function renderDetail(", 1)[1]

    assert "const offered = downloadOffered(license?.spdx, variant.licenses)" in render_detail
    # The notice names the reason and jumps to the License section.
    notice = (
        '`<button class="download-unavailable" type="button" data-scroll-to="license-section">'
        "${icons.license}Download restricted by license</button>`"
    )
    assert f"const downloadAction = !offered ? {notice}" in render_detail
    assert "${offered ? packageCard(fullPackage," in render_detail
    # Every artifact download on the page sits in the header action or the package card.
    assert script.count("data-download-variant=") == 1
    assert script.count('data-download="package"') == 2


def test_local_preview_fixtures_are_current_model_reports_and_server_is_same_origin() -> None:
    server = (ROOT / "artifact-browser" / "preview.py").read_text(encoding="utf-8")
    script = _browser_source("app.js", "download-dialog.js")
    validator = Draft202012Validator(json.loads((ROOT / "schemas" / "model-report.schema.json").read_text("utf-8")))

    # The fixtures follow the current report format, so publication accepts them unchanged.
    for stage in ("performance", "accuracy"):
        report = json.loads((BROWSER / "fixtures" / f"model-report-{stage}.json").read_text(encoding="utf-8"))
        validator.validate(report)
        assert report["stage"] == stage
    assert not (BROWSER / "fixtures" / "performance-result.json").exists()
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
  metadata: {model_id: model, build_profile_id: 'w8a8_w16a16', component_id: component, github_run_id: workflow},
});
const current = [run('backbone', 'rf_detr_small', 'backbone', 'workflow-2'),
                 run('decoder', 'rf_detr_small', 'decoder', 'workflow-2')];
const old = [run('old-backbone', 'rf_detr_small', 'backbone', 'workflow-1', '2.1', '2026-09-01T12:00:00Z'),
             run('old-decoder', 'rf_detr_small', 'decoder', 'workflow-1', '2.1', '2026-09-01T12:00:00Z')];
const catalog = [{model_id: 'rf_detr_small', display_name: 'RF-DETR', builds: [
  {run_id: 'stale-backbone', build_profile_id: 'w8a8_w16a16', component_id: 'backbone',
   finished_at: '2026-08-01T00:00:00Z'},
  {run_id: 'stale-decoder', build_profile_id: 'w8a8_w16a16', component_id: 'decoder',
   finished_at: '2026-08-01T00:00:00Z'},
], benchmark_fingerprints: {w8a8_w16a16: {performance: 'stale-performance-fingerprint'}}},
                 {model_id: 'ssd_mobilenet', display_name: 'SSD',
                  builds: [{run_id: 'inherited', build_profile_id: 'w8a8_w16a16'}]}];
const packageRun = (id, components) => ({id, run_type: 'model_package', status: 'completed',
  git_ref: 'refs/heads/perf/rfdetr-attention-layout', commit_sha: 'workflow-2', metadata: {
  model_id: 'rf_detr_small', build_profile_id: 'w8a8_w16a16', component_run_ids: components,
}});
const packages = [packageRun('good', {backbone: 'backbone', decoder: 'decoder'}),
                  packageRun('wrong', {backbone: 'backbone', decoder: 'old-decoder'})];
const models = branchModels([...old, ...current], packages, catalog);
assert.deepEqual(models.map(model => model.model_id), ['rf_detr_small']);
assert.deepEqual(models[0].builds.map(build => build.run_id), ['backbone', 'decoder']);
assert.equal(models[0].builds[0].run.git_ref, 'refs/heads/perf/rfdetr-attention-layout');
// Only a profile showing the catalog's own build keeps its benchmark fingerprints.
assert.deepEqual(models[0].benchmark_fingerprints, {});
const matched = [{...catalog[0], builds: ['backbone', 'decoder'].map(id => ({run_id: id,
  build_profile_id: 'w8a8_w16a16', component_id: id})),
  benchmark_fingerprints: {w8a8_w16a16: {accuracy: 'accuracy-fingerprint'}}}];
assert.deepEqual(branchModels(current, [], matched)[0].benchmark_fingerprints,
                 {w8a8_w16a16: {accuracy: 'accuracy-fingerprint'}});
assert.deepEqual(models[0].packages.map(item => item.run_id), ['good']);
assert.deepEqual(branchModels([...old, ...current], packages, catalog, '2.1')[0].builds.map(build => build.run_id),
                 ['old-backbone', 'old-decoder']);
const sameCompilerOld = [
  run('same-old-backbone', 'rf_detr_small', 'backbone', 'workflow-0', '3.0', '2026-08-01T12:00:00Z'),
  run('same-old-decoder', 'rf_detr_small', 'decoder', 'workflow-0', '3.0', '2026-08-01T12:00:00Z')];
assert.deepEqual(
  branchModels([...sameCompilerOld, ...current], packages, catalog, '3.0')[0].builds.map(build => build.run_id),
  ['backbone', 'decoder']);
const [snapshot] = branchModels([], packages, catalog);
assert.deepEqual(snapshot.builds.map(build => build.run_id), ['stale-backbone', 'stale-decoder']);
assert.deepEqual(snapshot.benchmark_fingerprints, {w8a8_w16a16: {performance: 'stale-performance-fingerprint'}});
assert.deepEqual(snapshot.packages, []);
const performanceRun = {...run('benchmark', 'inherited', 'model', 'workflow-2'), run_type: 'model_performance'};
assert.deepEqual(branchModels([performanceRun], [], catalog), branchModels([], [], catalog));
const develop = ['ssd_mobilenet', 'resnet50', 'yolov8', 'mobilenet', 'rf_detr_small', 'unet']
  .map(model => run(model, model, 'model', model));
// Only models the catalog lists show; a removed model's runs stay in the registry.
assert.deepEqual(branchModels(develop, [], catalog).map(model => model.model_id), ['rf_detr_small', 'ssd_mobilenet']);
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
    script = _browser_source(*MODEL_PAGE_MODULES)
    render_detail = script.split("function renderDetail(variant)", 1)[1].split("function errorPanel(", 1)[0]
    component_row = script.split("function componentRow(detail, single)", 1)[1].split("function ", 1)[0]

    # Build-level sections render once per variant, in the page order.
    order = [
        "detailBuildSwitcher(variant)", "${overviewMarkup}", 'id="components-section"',
        "${accuracyMarkup}", "packageCard(fullPackage,", 'disclosure("provenance", "Build provenance"',
        "sourceInputPanel(inputs)", "licensePanel(license,",
    ]
    positions = [render_detail.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert all(render_detail.count(marker) == 1 for marker in order)
    # One accuracy panel for the variant, from its accuracy run.
    assert "accuracyPanel(variant, model)" in render_detail
    assert "details.flatMap((detail) => detail.provenance.inputs || [])" in render_detail
    assert "(input) => input.sha256 || input.name" in render_detail
    assert "details.map((detail) => componentRow(detail, singleComponent))" in render_detail
    # The whole model is measured once, whatever its components; a declared pipeline adds a strip above it.
    assert "This model runs as ${details.length} components, measured running together." in render_detail
    assert "section-note" not in script
    strip, performance = (render_detail.index(marker) for marker in (
        '${order ? pipelineStrip(variant.pipeline) : ""}', "${modelPerformance(variant, model)}",
    ))
    assert strip < performance
    model_performance = script.split("function modelPerformance(", 1)[1].split("\nfunction ", 1)[0]
    # One layer analysis and one execution graph disclosure, each switching between the components.
    assert (
        "layerAnalysisSection(components.map(({ label, layerAnalysis }) => "
        "({ label, analysis: layerAnalysis })), run.id)"
    ) in model_performance
    assert "executionGraphSection(components.map(" in model_performance
    # MLA utilization follows the layer analysis, from the package run's simulation.
    assert (model_performance.index("layerAnalysisSection(") < model_performance.index("renderMlaUtilization(")
            < model_performance.index("executionGraphSection("))
    assert "mlaStageSimulation(variant.packageTests?.test_results)" in model_performance
    utilization = script.split("function renderMlaUtilization(", 1)[1].split("\nfunction ", 1)[0]
    # It rests on a simulator estimate, so customers never see it.
    assert utilization.split("\n", 2)[1].strip().startswith("if (!shows.mlaUtilization ||")
    assert "performanceSummary(result.value, run, modelId)" in model_performance
    graphs = script.split("function executionGraphSection(", 1)[1].split("\nexport function ", 1)[0]
    assert 'disclosure(`graphs:${key}`, "Execution Graph"' in graphs
    # Execution Graph and MLA Utilization switch between components the same way.
    assert "componentPanels(graphed.map(({ label }) => label)," in graphs
    assert "componentPanels(results.map(({ label }) => label), panels)" in utilization
    assert "<h4>" not in utilization
    panels = script.split("function componentPanels(", 1)[1].split("\n}\n", 1)[0]
    assert 'componentSwitch(labels, 0, "data-component-switch")' in panels
    select = script.split("export function selectComponentPanel(", 1)[1].split("\n}\n", 1)[0]
    assert 'item.hidden = item.dataset.componentPanel !== button.value;' in select
    assert 'elements["detail-pane"].addEventListener("click", selectComponentPanel);' in _browser_source("app.js")
    # Switching component re-renders the panel and restarts the minimum time on the new scale.
    update = script.split("export function updateLayerAnalysis(", 1)[1].split("\nexport function ", 1)[0]
    assert 'if (field === "component") {' in update
    assert "entry.view.minUs = 0;\n    panel.innerHTML = layerAnalysisPanel(entry);" in update
    # Per-component facts live in the row, never on the page.
    for marker in ['keyValue("Run ID"', 'keyValue("Fingerprint"']:
        assert marker in component_row
        assert marker not in render_detail
    # Numerical fidelity belongs to the package run, beside the package; no performance sits in a file row.
    assert "fidelityPanel(" not in component_row and "Performance(" not in component_row
    assert render_detail.index("packageCard(fullPackage,") < render_detail.index("fidelityPanel(")
    assert render_detail.index("fidelityPanel(") < render_detail.index('disclosure("provenance"')
    # A component row is a disclosure keyed by its run, with compile parameters nested inside it.
    assert 'disclosure(`component:${run.id}`, single ? "Files and checksums" : component, body, meta)' in component_row
    assert 'disclosure(`parameters:${run.id}`, "Compile parameters"' in component_row


def test_preview_native_deployments_use_manifest_and_path_preserving_files(preview: str) -> None:
    api = "/v1/repositories/sima-neat/models"
    catalog = _get_json(f"{preview}{api}/catalogs/current?git_ref=refs/heads/develop")
    # A LLiMa model; a vision-language model such as CLIP can be an ONNX model with MPKs instead.
    model = next(model for model in catalog["catalog"]["document"]["models"] if model["builds"] and _get_json(
        f"{preview}/v1/runs/{model['builds'][0]['run_id']}")["run"]["metadata"]["artifact_kind"] == "llima")
    for record in [model["builds"][0], model["packages"][0]]:
        artifacts = _get_json(f"{preview}/v1/runs/{record['run_id']}")["artifacts"]
        manifest, *files = artifacts
        assert manifest["media_type"] == "application/json"
        assert manifest["metadata"]["layout"] == "directory"
        assert manifest["metadata"]["storage_backend"] == "s3"
        assert manifest["metadata"]["file_count"] == len(files)
        assert manifest["metadata"]["deployment_size_bytes"] == sum(file["size_bytes"] for file in files)
        assert manifest["size_bytes"] < manifest["metadata"]["deployment_size_bytes"]
        assert all(file["artifact_type"] == "model-file" and file["metadata"]["path"] for file in files)
