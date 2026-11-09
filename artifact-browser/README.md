# Model Artifact Browser

A client-only web interface for the public SiMa.ai Model Registry read APIs.
The browser calls the registry directly; there is no application backend,
credential handling, or proxy middleware. Archive downloads use the temporary URL
returned by the registry directly. Folder deployments show CLI retrieval
instructions, and Hugging Face deployments link to the resolved, pinned revision.

For an app-only development loop, run from the Models repository:

```bash
npx --yes serve artifact-browser --listen 4173
```

Then open <http://127.0.0.1:4173>. Any static file server works. To use another
registry environment, change `apiBase`, `environment`, and `repository` in
`config.js` before serving the directory:

```js
window.MODEL_REGISTRY_CONFIG = Object.freeze({
  apiBase: "https://example.test",
  environment: "Production",
  repository: Object.freeze({ organization: "sima-neat", name: "models" }),
});
```

To test the catalog together with the Docusaurus developer documentation, use
the unified build:

```bash
npm ci --prefix website
npm run typecheck --prefix website
npm run build:portal --prefix website
npm run serve:portal --prefix website
```

Open <http://localhost:4173> for the catalog and
<http://localhost:4173/docs> for the developer guide.

The deployed catalog is the assembled `website/portal-build/`, where every
module is renamed with the revision. `preview.py` serves the unrenamed sources,
so check the assembled build before a pull request. CI runs the same check:

```bash
npx --prefix website playwright install chromium
npm run smoke:portal --prefix website
```

It serves `portal-build/` against the local preview API in Chromium and fails
when a module or stylesheet does not load, the page throws or logs an error, or
the model list or a variant page does not render. The preview API needs the
registry tooling, so run it with a Python that has the repository installed
(`python -m pip install .`), or point `PYTHON` at one.

To preview the browser without a registry or DevKit connection, run:

```bash
python3 artifact-browser/preview.py
```

Open the URL printed by the script. The local API builds its catalog from
`models/` with the registry tooling, so it shows the models, build profiles,
packages, READMEs, heroes, provenance, and accuracy and fidelity declarations
this checkout defines; a model added or removed there appears or disappears
without changes to the preview. Only what a board or the registry would produce
is synthetic: run IDs, timestamps, sizes, and measured values. Two catalog
revisions give every variant an older build in its history.
The preview resolves each benchmark fingerprint the catalog records to a DevKit run
whose model report starts from a fixture. `fixtures/model-report-performance.json`
is a ResNet-50 DevKit Core benchmark converted to a model report, so its board
telemetry and power rails are real readings; the preview varies only speed,
energy, and MLA memory per build, and generates illustrative layer timings
covering all ten recognized operation families in each profiling panel. A
multi-component model, such as RF-DETR, reports each component's layers and
execution graph in the order its pipeline runs them. These synthetic
layer names include fused operations, nested reshapes, segmentation helpers,
outputs and zero-time entries. Input placeholders are omitted, matching the
benchmark API. These names do not describe the selected model's actual graph.
Each preview package run also carries a generated MLA stage simulation for the
same stages, at illustrative rates, so developer mode shows MLA Utilization.
`fixtures/model-report-accuracy.json` gives each accuracy run its shape; the
preview fills in the metrics the model's tests expect. YOLO26n W16A16
keeps its benchmark fingerprint without a run, so its page shows **Not measured at
this revision.** Edit the fixtures to exercise other metric values or
availability states.

The browser reads version 1 of the model report (`schemas/model-report.schema.json`)
in `model-report.js`. Known metrics receive curated labels, formatting, and a
place in a benchmark card; unknown additive metrics appear automatically under
Other measurements. Any other report format shows **This result format is not
supported by this browser** instead of guessing at an incompatible structure.

## Staging deployment

Every push to `main` or `develop` runs `vulcan-model-orchestrator.yml`, whose
`portal` job assembles one static portal with `developer-docs.yml` and
browser-tests and smoke-tests it. Its `publish-model-browser` job then
publishes that portal to the staging docs bucket under `models/`. The
Developer Center CloudFront distribution routes that prefix to the docs bucket,
making the browser available at <https://developer-stg.sima.ai/models>.

A manual dispatch of the orchestrator on either branch also republishes it.
The frontend build and publication always use staging, independently of
`VULCAN_ENV` and the selected model build/registry environment. The separate
`resolve-staging-browser-config` job resolves the bucket, publisher role, KMS key,
and CloudFront distribution from the shared staging Vulcan configuration.
Production frontend deployment remains a manual, owner-gated
**Deploy Model Browser & Docs** dispatch.

`website/scripts/portal-config.mjs` selects the UI at build time from `GITHUB_REF`:
Main builds the customer view of the main catalog; Develop, feature branches and
pull requests build the developer view. Both use the same staging API and
publication URL. The last publication wins; the catalog branch selector does not
change the site's UI. Customer view hides contributor navigation, branch/compiler
filters, build history and the Details section while retaining browsing, comparison,
measurements and downloads.

For a local customer build, use `MODEL_BROWSER_UI_MODE=customer` with
`npm run build:portal --prefix website`; pass the same variable to
`npm run smoke:portal --prefix website` or `preview.py`. The default local mode
is `developer`. `MODEL_REGISTRY_API_URL` optionally overrides the API for local
builds; neither mode requires production infrastructure.

The default is the WAF-protected public staging registry at
`https://models-registry.stg.neat.sima.ai`. The API must allow cross-origin
`GET` requests from the static app's origin. The separate
`artifacts-registry.stg.neat.sima.ai` endpoint is reserved for IAM-signed
publisher traffic and rejects direct public reads.

The initial view loads the repository branch catalog independently from build
runs and caches both so repeat visits can render immediately while refreshing.
The selected branch's model cards show what the branch delivers: the builds in
its catalog, including builds it inherits from its base branch. A completed
build compiled on the branch that is newer than the catalog's build of the same
model and build profile replaces it, so a build the catalog has not recorded
still shows. Each build keeps the provenance of its own compile run, so an
inherited build names the branch that compiled it. A branch without a catalog
shows an empty state until its first push completes. Components are grouped
only within the same workflow build, and a full package appears only when its
recorded component run IDs match the shown build.
Build history lists one entry per variant build, newest first, naming its
components; an entry gains components as **Load more builds** brings in the rest of its
build. If branch discovery is
temporarily unavailable, the browser keeps saved branches and can infer
additional branches from returned runs. Full
provenance, validation, and metric views load only after a build is selected.
Public discovery requests completed builds only. The primary search matches
model, build-profile, and component names together with category names. Task and
model family filters sit beside the cards. Developer view also shows Branch,
Compiler and Build history directly. Category filters list the values the loaded
builds use and hide when fewer than two remain; developer controls stay visible.
On narrow screens the filters move into a drawer whose button shows the
active-filter count.
The branch filter defaults to `develop`. A branch selected by the user is saved per
registry and repository in browser storage. Shared links include `branch` and
`model` and `variant` (the build profile), which open the latest build of that
variant on the branch. Links from build history also carry `run`, one component
run of the pinned build. URL state takes precedence over that preference. Links
containing only `run` resolve the run's actual branch and open that run's
variant build with its component row expanded, so every component's link opens
the same page. The dedicated branch catalog, not a
limited runs page, determines whether a saved branch falls back to `develop` (or all
branches when `develop` is unavailable). Build-history pagination remains
branch-aware; the model view fetches all pages needed to group the latest
completed builds.

## Variant pages

Selecting a model card opens its page in place of the catalog; the breadcrumb
or the browser's Back button returns to the catalog at the same scroll position.
A page shows one variant build: the component runs of one model and build
profile compiled by one workflow. Its header carries the name, description,
category tags and **Download**, with validation and source-link icons beside it.
**Copy CLI command** is available inside the download dialog for current catalog builds.
The model image sits beside the header content on wide screens and stacks below
it on narrow screens. Both UI modes use this layout. Variant tabs and section
links to **Overview**, **Performance**, **Accuracy**, **Details**, and **License**
follow the header. A section without
data is left out.

The page is read at a glance first: each section shows its key facts, and
everything longer sits in one expandable row.

- **Overview** shows the README's `## Overview`, then loads the remaining input/output
  and application instructions inline after checking their SHA-256 against the
  catalog. It renders the README from the section after Overview, because the title,
  hero, and Overview are already on the page.
- **Performance** shows throughput, latency, SoM power, and energy per
  inference for the whole model, one number per tile, from the build profile's
  performance run. A multi-component model is measured with its components
  running together, and its layer analysis and execution graph switch between
  the components. **Performance Details** is
  performance, heat, and CPU cards, with power (SoM; SoC as the sum of its
  MLA and Non-MLA rails; and each other power rail) and memory cards beside
  them,
  then any other measurements and a footer with the telemetry source and
  runtime beside **Raw result** (the run's model report) and **Download CSV**
  (every reading of the model in one table).
- **Accuracy** shows the coverage, the first four metrics, and the dataset,
  from the build profile's accuracy run. In the developer UI, **Evaluation
  details** explains the coverage and lists the context and remaining metrics.
  **Published scores** and **FP32 baseline** list the `reference_metrics`
  entries of `model.yaml` of each origin, with their dataset, conditions, and
  source or script; a reference score sits beside a measured one only when both
  name the same dataset. A build without an accuracy result shows only the
  reference scores.
- **Details**, in the developer UI only, shows the package card (size, SHA-256, **Download package**,
  and **Download components** for a multi-component build) and the package's numerical
  fidelity, then rows for build provenance, source models (one
  per distinct digest), and each component: checksum, run identity, and a nested
  **Compile parameters** row.
- **License** is one line with the SPDX identifier and redistribution, plus
  the attribution.

The catalog records each build profile's benchmark fingerprints. A fingerprint
without a run, or a profile without one, shows **Not measured at this
revision.**

## Compare page

**Compare** explores the selected branch's model variants by task. All
precisions appear together; measured variants appear in a scatter
plot of accuracy against throughput in inferences/s. Accuracy plots keep dataset
and metric identities separate. A multi-component model shows its whole-model result, with all its components running as one pipeline. Hovering or focusing a point
shows the model name, variant, throughput and accuracy.

Select up to three points or use **Add model** to inspect exact measurements
in a table. It includes throughput, latency, power, energy, peak MLA and EV74 memory and
accuracy. Bold marks the best value.
Models without measurements remain available in **Add model**. LLM/VLM comparisons show **TBD**.
Links preserve the task and selected builds with `compare_task` and repeated
`build` parameters. Build history is not compared.

## How the page is built

The browser is plain ES modules with no build step and no framework. Each view
renders HTML strings into its container and `app.js` handles events by
delegation.

| Module | Owns |
| --- | --- |
| `app.js` | Startup, routing from URL state, and event handlers |
| `portal-shell.js` | Load and mount the shared Developer Portal header |
| `state.js` | Shared page state, configuration, and DOM element lookup |
| `api.js` | Registry requests, caches, and catalog build selection |
| `branch-builds.js` | Which builds a branch shows, and grouping by workflow |
| `catalog-view.js` | Model cards, build history, search, and filters |
| `model-view.js` | The model page |
| `compare-view.js` | The Compare page: pickers and table |
| `model-report.js` | Reading performance and accuracy model reports |
| `mla-layer-stats.js` | Analyzing a component's MLA layer stats |
| `mla-utilization.js` | Comparing measured MLA stages with their simulation |
| `metrics.js` | Metric labels, units, and formatting |
| `downloads.js` | Component discovery for downloads |
| `markdown.js` | The README renderer |
| `ui.js` | Icons, escaping, toasts, and the `disclosure()` row |

Every expandable area is a `disclosure(key, label, body, meta)` row from
`ui.js`: a full-width row with the label on the left, an optional summary
(`meta`) and a › that turns down when open. The open rows are kept in
`state.openDisclosures` by key, so a rerender keeps them open; the set is
cleared when the page moves to another model. A link opened with `run`
opens that component's row (`component:<run id>`). Opening the
model page loads its README instructions automatically.

Buttons with an icon start actions (download, copy). Blue text with ↗ is
reserved for links that leave the browser. Cards group related numbers only
(the KPI tiles, benchmark cards, and the package); the rest of the page uses
sections and thin rules.

`website/scripts/assemble-portal.mjs` renames every module with the revision
for the deployed portal and fails when a module imports a file missing from its
`fingerprintedAssets` list, so add a new module there.
It also writes `component-pipelines.<revision>.json` from the `pipeline` field
of each `model.yaml`, which draws the component order on a multi-component
model page, and `reference-metrics.<revision>.json` from each
`reference_metrics`; `preview.py` serves the same data from `models/`.
`metrics.js` and `accuracy-metrics.js` keep the labels, units, and directions
of `tools/model_report/metrics.json`, and tests keep them in step. Run
`build:portal` with the repository installed or `PYTHON` set, as for
`preview.py`.

## Downloading model variants

Native LLiMa deployments use a JSON index (`model-pack`, or `model-package`
for the complete package) with `metadata.layout: directory` and separate
`model-file` payload artifacts. The browser shows the aggregate
`metadata.deployment_size_bytes`, not the small index size. **Download folder**
opens instructions for `sima-cli models download`, which preserves paths and
checks each file. Folder downloads require sima-cli 2.1.19 or later. The command
uses the selected branch and adds `--stg` for staging; historical builds point
to the current catalog variant instead of offering a command for a different
deployment. The browser never downloads a folder's JSON index as a ZIP.

Delivery is determined from the backend-resolved artifact. A resolved
`metadata.storage_backend: huggingface` has
`metadata.huggingface: {repo_id, revision, subdirectory?}`; `revision` is an
immutable 40-character commit. The dialog links to that exact repository
revision and folder. This location describes the compiled deployment and is
independent of the source checkpoint links. Storage selection and approval
remain backend responsibilities.

Selecting a feature branch shows its finalized `branch_deployments`, including
candidates that have not been promoted. They take precedence over inherited
`approved_deployments`. Metadata and unselected runs do not create on-demand
cards or replace a selection. Develop and Main show approved selections.
Completed candidates also remain in build history.
Approved package membership comes from the profile's `build_requirements`. Only
packages covering every shown component are offered as full-model downloads.

The following confirmation flow applies to archive deployments, including
previously published LLiMa ZIPs.

The model download button includes every model-pack component from the selected
variant's workflow build, commit, branch, target, and compiler. All
variants show a confirmation dialog listing component names, actual archive filenames, and file sizes.
Choose **Download N files** to start the transfers in parallel, or **Cancel**
(or Escape) to leave without starting any downloads. Cancel also works while
links are being prepared. Single-file variants require confirmation with **Download 1 file**.
Numerical-fidelity **Download JSON** remains a separate report download.

**Download package** saves the published package under its registry filename:
the MPK itself for a single-component model, or a ZIP of the component MPKs
and declared resources otherwise. The dialog labels the file MPK or ZIP.

All temporary URLs are prepared before any transfer starts. If preparation
fails, the dialog reports the error and allows retrying. Files stream directly
to the browser's download manager, which may request permission for multiple
downloads; use that manager to monitor or cancel transfers once started.

For a local preview using live staging data:

```bash
python3 -m http.server 4174 --bind 127.0.0.1 --directory artifact-browser
```

Open <http://127.0.0.1:4174/?run=53bfaa91-8b75-4bdb-afc2-83019a61f514>
and click **Download** to review the RF-DETR component dialog.

Run the download and variant-build checks with `node --test tests/browser/*.test.mjs`
and the existing UI checks with `python -m pytest tests/test_artifact_browser.py`.
The dialog shows each stored MPK's own filename, `<model_id>_<profile_id>_mpk.tar.gz`
or `<model_id>_<profile_id>_<component_id>_mpk.tar.gz`, and downloads it unchanged.

## Duration reporting

Registry run timestamps cover artifact publication, which happens after compilation.
The UI labels this interval **Registry publication duration**. **Compile recipe
duration** comes from `metadata.compile_timing`, captured around the compilation
recipe by the compile workflow. It includes surgery and any validation performed
inside that recipe; it excludes runner queue/setup, separate accuracy jobs,
publication, and subsequent benchmarks. Components compiled together share this
invocation timing.

The publisher requires `compile-timing.json` alongside `compile-outputs.json`,
validates its timezone-aware start/end timestamps before any registry writes,
and publishes it as run metadata.

The preview's layer timings are illustrative generated data, including zero-time
entries and multiple MLA stages to exercise the analysis controls. Production
pages use only the compiled layer timings in each published model report.

### Staging hero CDN

Staging and production developer builds use their own public-reader hostname's
`/images/sha256/<hero artifact SHA-256>.png` path. The hash comes from catalog
metadata; no signed URLs or extra per-image API resolution are needed. Vulcan
publishes validated hero derivatives into a private S3 origin when documentation
runs finish. Deploy that infrastructure and backfill current images **before**
merging/deploying this browser change. See the companion Vulcan
`docs/artifact-registry-hero-cdn.md` runbook.

`MODEL_HERO_CDN_BASE_URL` is an optional build setting. Its default is the matching environment
CDN path for developer builds using either standard registry API; customer builds and
custom API origins default to disabled. Set it to an empty string and rebuild to
roll back to API image URLs. The source preview `config.js` selects staging CDN.
Image loading retries twice on the CDN and then uses the existing API URL with
bounded retries. Removed cards stop retrying, and retries preserve immutable URLs
so they do not bypass CloudFront caching. Model downloads and README text keep
their existing routes. No model definitions, fingerprints or compilation behavior
change.

After deployment, test a fresh catalog load, a model detail page, and an image
missing from the CDN (API fallback). Inspect image request paths, verify that no
repeated failure loops occur, and check staging reader throttles. Backend tests
and local browser tests do not replace this staging acceptance check.

## Preview production data from staging

The staging browser uses staging by default. Open
[the production preview](https://developer-stg.sima.ai/models?env=prod)
to run the same staged UI against the public read-only production Models registry.
The registry label changes to **Production registry**, downloads use production,
and catalog/branch caches stay separate by API origin. Model navigation retains
`env=prod`; remove that parameter to return to staging. Select the catalog branch
you want to inspect; changing backend does not populate or promote a catalog.

The preview uses only the fixed `https://models-registry.neat.sima.ai` reader.
Other `env` values retain the default backend; query strings cannot set arbitrary
API origins. Production images use production API reads, never the staging image
CDN. The preview is enabled only by staging configuration, including customer UI
builds staged for review. Local custom APIs and production configurations do not
opt into this override. No AWS credentials are sent by the browser. CORS permits
cross-origin reads, while registry policy enforces repository and artifact access.

## Hide branch selection

Add `?branches=0` to browse Main with the branch selector hidden. This overrides
saved branch preferences and any `branch=` parameter, without changing the UI
audience or the selected backend. Other developer filters remain available.
Clear filters and in-page navigation keep Main selected and retain `branches=0`.
The preview does not replace your saved branch preference. Omit the parameter
(or use `branches=1`) to restore the normal branch behavior; customer UI remains
Main-only. Combine with `env=prod` as `/models/?env=prod&branches=0` to preview
production Main. That catalog must be published before it can display models.

## Production deployment

`MODEL_BROWSER_ENVIRONMENT=production` selects the production public registry,
Developer Center origin and environment label. Audience remains independent.
See [production frontend deployment](../docs/model-browser.md#production-frontend)
for the tested, prefix-scoped operator deployment and rollback procedure.
