# Model Browser

A client-only web interface for the public read APIs of the SiMa.ai model registry.
The browser calls the registry directly; there is no application backend,
credential handling, or proxy middleware. Archive downloads use the temporary URL
returned by the registry directly. Folder deployments show CLI retrieval
instructions, and Hugging Face deployments link to the resolved, pinned revision.

Read this file before you change `artifact-browser/` or `website/`. It holds the code
map, the rules for a change, and how the browser is deployed. Two other files own the
rest:

- [docs/model-browser.md](../docs/model-browser.md) owns what a visitor sees and does.
- [DESIGN.md](DESIGN.md) owns the visual design.

## Run the browser locally

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
outputs, zero-time entries and several MLA stages, to exercise the analysis
controls. Input placeholders are omitted, matching the
benchmark API. These names do not describe the selected model's actual graph.
A deployed page uses only the layer timings in each published model report.
Each preview package run also carries a generated MLA stage simulation for the
same stages, at illustrative rates, so developer mode shows MLA Utilization.
`fixtures/model-report-accuracy.json` gives each accuracy run its shape; the
preview fills in the metrics the model's tests expect. YOLO26n W16A16
keeps its benchmark fingerprint without a run, so its page shows **Not measured at
this revision.** Edit the fixtures to exercise other metric values or
availability states.

GenAI previews use `fixtures/genai-model-report.json`;
`preview.py --synthetic-genai` enables examples for unmeasured models.
Synthetic readings are labeled.

The preview supplies deterministic synthetic conversion identities for its catalog.
It resolves no real conversion and needs no registry credentials.

The browser reads version 1 of the model report (`schemas/model-report.schema.json`)
in `model-report.js`. Known metrics receive curated labels, formatting, and a
place in a benchmark card; unknown additive metrics appear automatically under
Other measurements. Any other report format shows **This result format is not
supported by this browser** instead of guessing at an incompatible structure.

## Rules for changing the browser

Read [DESIGN.md](DESIGN.md), [How the page is built](#how-the-page-is-built), and
[model browser code](../knowledge/model-browser.md) before you change a file here. Update
[docs/model-browser.md](../docs/model-browser.md) in the same change as anything a
visitor can see.

**Page structure**

- Render every expandable area with `disclosure()` from `ui.js`.
  `tests/test_artifact_browser.py` fails when a module writes its own `<details>`
  element.
- Keep the page glance-first: key facts in the section, the rest in one row.
- Make each action an icon button. Keep blue text with ↗ for links that leave the
  browser.
- Put each difference between the customer and the developer view in `audienceFacts`
  in `audience.js`, as a named fact such as `shows.buildDetails`. Test that fact where
  the markup is built. Page code never tests the audience itself.
  Three customer-view rules live outside that list: `styles.css` hides `.developer-only`
  markup and the branch and compiler tags, and `environment.js` keeps customers on Main.
- Give each section of the model page its own module: `performance-section.js`,
  `execution-section.js`, and `accuracy-section.js`. A section module never imports
  `model-view.js` or another section. Markup that sections share goes in
  `page-parts.js`, and `model-view.js` assembles the sections.

**Catalog**

- Group the Cards overview by each model's architecture ID. Take a family's title
  and description from `taxonomies/model-taxonomy.yaml`. Derive its members, tasks,
  and hero from the filtered models.
- Give each architecture in the taxonomy a short `description` that covers the whole
  family, with no claim about one size. Validation rejects an empty one.
- A family card selects the existing Model family filter. Search and Table show
  individual models.
- Taxonomy and browser files are catalog metadata. Keep them out of the conversion,
  compile, and benchmark fingerprints.
- `profileGroups` in `api.js` orders the build profiles. A profile with a `w8a8`
  component comes first and opens first, whatever order `model.yaml` declares.

**Measurements**

- Show one whole-model result per build profile by default. When
  `performance.scope` is `components`, keep each component's measurements separate.
- Show layer timings and execution plans per component. Never add component
  numbers together, and never mark one component best against a different component.
- Render the current model report version, version 1. A metric with no tile or card
  goes under Other measurements, so every reported metric stays visible.
- Add a new metric to `tools/model_report/metrics.json` and to `metrics.js` or
  `accuracy-metrics.js`. Tests keep their labels, units, and directions in step.

**Downloads**

- Offer a model's downloads only when `downloadOffered` in `downloads.js` allows
  them. It reads the license from the branch catalog and `licenses.json`. An
  unlisted license, or a list that did not load, offers none.
- Keep `licenses.json` in `artifact-browser/`, so that a plain static server serves
  it with the page.

**Preview and build**

- Keep the preview data-driven. `preview.py` builds its catalog from `models/`, so
  add no model, name, or README to it by hand.
- List each new module in `fingerprintedAssets` in
  `website/scripts/assemble-portal.mjs` and in the
  [module table](#how-the-page-is-built).

### Checks before a push

1. Run `python -m pytest tests/test_artifact_browser.py`.
2. Run `node --test tests/browser/*.test.mjs`.
3. Run `npm run typecheck --prefix website`.
4. Run `npm run build:portal --prefix website`.
5. Run `npm run smoke:portal --prefix website`.
6. Open each changed page in `preview.py` at desktop and phone width, in light and
   dark.

`build:portal` and `smoke:portal` need a Python with the repository installed, as
[Run the browser locally](#run-the-browser-locally) describes.

## How the page is built

The browser is plain ES modules with no build step and no framework. Each view
renders HTML strings into its container and `app.js` handles events by
delegation.

| Module | Owns |
| --- | --- |
| `app.js` | Startup, routing from URL state, and event handlers |
| `config.js` | The registry, environment, and view settings for a source preview. Assembly writes its own for a deployed build |
| `environment.js` | The `env=prod` production preview and the `branches=0` branch policy |
| `portal-shell.js` | Load and mount the shared Developer Portal header |
| `analytics.js` | The cookie consent banner and consent-gated GA4 page views, shared with the developer docs |
| `state.js` | Shared page state, configuration, and DOM element lookup |
| `api.js` | Registry requests, caches, catalog build selection, and build profile order |
| `branch-builds.js` | Which builds a branch shows, and grouping by workflow |
| `catalog-view.js` | Model and family cards, the table, build history, search, and filters |
| `hero-images.js` | Hero image URLs, and retries from the CDN to the registry API |
| `audience.js` | What the customer and the developer view each show |
| `model-view.js` | The model page: header, variants, overview, details, and license |
| `performance-section.js` | Performance: tiles, detail cards, power rails, and the CSV |
| `execution-section.js` | Execution Graph, Runtime Layer Analysis, and MLA Utilization |
| `accuracy-section.js` | Accuracy: measured scores, coverage, and reference scores |
| `page-parts.js` | Markup and helpers the model page's sections share |
| `download-dialog.js` | The download dialog |
| `compare-view.js` | The Compare page: chart, pickers, and table |
| `model-report.js` | Reading performance and accuracy model reports |
| `mla-layer-stats.js` | Analyzing a component's MLA layer stats |
| `mla-utilization.js` | Comparing measured MLA stages with their simulation |
| `metrics.js` | Performance metric labels, units, and formatting |
| `accuracy-metrics.js` | Accuracy metric labels, scales, and order, and reference scores |
| `downloads.js` | Whether a license permits downloads, how an artifact is delivered, and component discovery for downloads |
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
for the deployed portal, preloads the entry point and its modules, and fails when
a module imports a file missing from its `fingerprintedAssets` list.
It also writes `component-pipelines.<revision>.json` from the `pipeline` field
of each `model.yaml`, which draws the component order on a multi-component
model page, `reference-metrics.<revision>.json` from each
`reference_metrics`, `low-accuracy-variants.<revision>.json`, the variants
the customer view leaves out, `dataset-names.<revision>.json` from the `name`
of each `datasets/<id>.yaml`, and `example-apps.<revision>.json` from each
`example_app`; `preview.py` serves the same data from the checkout.
It copies `licenses.json` to `licenses.<revision>.json`. That file names each
license and decides whether a model page offers downloads, and it sits in this
directory so that a static file server serves it with the page.
`metrics.js` and `accuracy-metrics.js` keep the labels, units, and directions
of `tools/model_report/metrics.json`, and tests keep them in step. Run
`build:portal` with the repository installed or `PYTHON` set, as for
`preview.py`.

## How the catalog loads

[The catalog](../docs/model-browser.md#the-catalog) describes the cards, the table,
search, and the filters a visitor uses. This section holds what the code does behind
them.

The customer catalog reads Main's `/catalogs/current` snapshot in one registry
request. The snapshot supplies model definitions, build and package IDs, and the
taxonomy. The browser applies the customer visibility rules and renders cards
from those definitions. Table and Compare retrieve measurements for their
views, and the model page loads full run data on demand.

The developer view discovers branches first, then requests the selected
branch's snapshot and its compile and package lists together. Cards render
as soon as the snapshot arrives. When the lists finish, the browser resolves
missing build IDs through `/runs?summary_ids=…` in batches of up to 40 and
refreshes the catalog with compiler, source branch, and timing metadata.
A reader that lacks batch support leaves those fields unavailable for the
affected builds. The compiler filter includes only builds with a matching
known compiler. Opening a model still retrieves its exact run data.

Local storage saves the branch names and each catalog's definitions and
taxonomy for the next visit. It keeps the live compile and package lists
in memory. A package appears only when its recorded component IDs match
the displayed build.

Build history lists one entry per variant build, newest first, naming its
components; an entry gains components as **Load more builds** brings in the rest of its
build. If branch discovery is
temporarily unavailable, the browser keeps saved branches and can infer
additional branches from returned runs. Full
provenance, validation, and metric views load only after a build is selected.
Public discovery requests completed builds only.
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

## Variant pages and Compare

[The model page](../docs/model-browser.md#the-model-page) describes each section of a
variant page, and [The catalog](../docs/model-browser.md#the-catalog) describes
**Compare**. This section holds what the code does behind them.

A page shows one variant build: the component runs of one model and build
profile compiled by one workflow. Selecting a model card opens its page in place of
the catalog, and the breadcrumb or the browser's Back button returns to the catalog
at the same scroll position. Variant tabs and links to each section follow the
header.

**Overview** shows the README's `## Overview`, then loads the remaining sections
inline after checking their SHA-256 against the catalog. It renders the README from
the section after Overview, because the title, hero, and Overview are already on the
page.

The catalog records each build profile's benchmark fingerprints, and the browser
resolves each to its run. A fingerprint without a run, or a profile without a
fingerprint, shows **Not measured at this revision.**

**Details** lists one source model row per distinct digest. **Compare** reads the
selected branch's catalog and does not compare build history. For each variant it loads only the
performance and accuracy results (`loadBenchmarkResults` in `api.js`), not the compile and package
runs a model page shows. Its links preserve the task, the workload, and the selected builds.

## Downloading model variants

Native LLiMa deployments use a JSON index (`model-pack`, or `model-package`
for the complete package) with `metadata.layout: directory` and separate
`model-file` payload artifacts. The browser shows the aggregate
`metadata.deployment_size_bytes`, not the small index size. **Download folder**
opens the dialog that
[Model folder downloads](../docs/model-browser.md#model-folder-downloads) describes.
The browser never downloads a folder's JSON index as a ZIP.

Delivery is determined from the backend-resolved artifact. A resolved
`metadata.storage_backend: huggingface` has
`metadata.huggingface: {repo_id, revision, subdirectory?}`; `revision` is an
immutable 40-character commit. The dialog links to that exact repository
revision and folder. This location describes the compiled deployment and is
independent of the source checkpoint links. Storage selection and approval
remain backend responsibilities.

Selecting a feature branch shows its finalized `branch_deployments`, including
candidates that have not been promoted. They take precedence over inherited
`approved_deployments`, and they keep their inherited compilation runs and matching
package. Metadata and unselected runs do not create on-demand
cards or replace a selection. Develop and Main show approved selections.
Completed candidates also remain in build history.
Approved package membership comes from the profile's `build_requirements`. Only
packages covering every shown component are offered as full-model downloads.
[knowledge/on-demand-builds.md](../knowledge/on-demand-builds.md) defines these
catalog fields.

The following confirmation flow applies to archive deployments, including
previously published LLiMa ZIPs.

A build with no package falls back to a download button that includes every
model-pack component from the selected variant's workflow build, commit, branch,
target, and compiler. All
variants show a confirmation dialog listing component names, actual archive filenames, and file sizes.
Choose **Download N files** to start the transfers in parallel, or **Cancel**
(or Escape) to leave without starting any downloads. Cancel also works while
links are being prepared. Single-file variants require confirmation with **Download 1 file**.
Numerical-fidelity **Download JSON** remains a separate report download.

**Download package** saves the published package under its registry filename:
a ZIP of the component MPKs and declared resources. A single-component build
published before packages became ZIPs stays a bare MPK until its next compile.
The dialog labels the file MPK or ZIP.

All temporary URLs are prepared before any transfer starts. If preparation
fails, the dialog reports the error and allows retrying. Files stream directly
to the browser's download manager, which may request permission for multiple
downloads; use that manager to monitor or cancel transfers once started.

For a local preview using live staging data:

```bash
python3 -m http.server 4174 --bind 127.0.0.1 --directory artifact-browser
```

Open `http://127.0.0.1:4174/?run=<run ID>` with the ID of one component run of a
multi-component build, and click **Download** to review the component dialog.

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

## Build settings

`website/scripts/portal-config.mjs` reads these settings when it builds the portal:

| Setting | Values | Default | Selects |
| --- | --- | --- | --- |
| `MODEL_BROWSER_UI_MODE` | `developer`, `customer` | `developer` | The view. Customer shows Main only, and developer shows the branch selector |
| `MODEL_BROWSER_ENVIRONMENT` | `staging`, `production` | `staging` | The registry API, the site origin, and the environment label |
| `MODEL_REGISTRY_API_URL` | An HTTPS origin, or `http://localhost` for a preview | The selected environment's public registry | Another registry API |
| `MODEL_HERO_CDN_BASE_URL` | An HTTPS URL, or empty | See [Hero image CDN](#hero-image-cdn) | Where hero images load from |
| `MODEL_BROWSER_GA_MEASUREMENT_ID` | A GA4 measurement ID, or empty | The selected environment's Developer Center stream: `G-T1D371DW1K` for staging, `G-Y6YREB4EG5` for production | Where page views report. Empty disables analytics and the consent banner |
| `MODEL_BROWSER_ANALYTICS_HOST` | A hostname | The selected environment's site host | The only host where analytics runs. Set `127.0.0.1` to test the banner on a local preview |

The view and the environment are independent. Customer view hides contributor
navigation, branch/compiler filters, build history and the Details section while
retaining browsing, comparison, measurements and downloads.
`customerPerformanceReport` in `model-report.js` also
removes the board's Linux memory from each performance report as it is loaded,
so no view of the customer page shows it. The registry still serves that reading;
the customer page only leaves it out.

For a local customer build, use `MODEL_BROWSER_UI_MODE=customer` with
`npm run build:portal --prefix website`; pass the same variable to
`npm run smoke:portal --prefix website` or `preview.py`. Neither view requires
production infrastructure.

The staging default is the WAF-protected public staging registry at
`https://models-registry.stg.neat.sima.ai`, and the production default is
`https://models-registry.neat.sima.ai`. The API must allow cross-origin
`GET` requests from the static app's origin. The separate
`artifacts-registry.stg.neat.sima.ai` endpoint is reserved for IAM-signed
publisher traffic and rejects direct public reads.

Analytics follows the rest of the Developer Center. The browser and the developer
docs share one cookie decision for the developer.sima.ai origin with the core
documentation, stored in `localStorage` as `sima-developer-center-cookie-consent`.
A visitor who already chose on another Developer Center page is not asked again.
Without a stored decision, the page shows the consent banner. GA4 loads only after
analytics is accepted, and a page view is sent on load, on each `pushState`
navigation, and on Back and Forward; `replaceState` URL updates are not page views.
**Cookie preferences** in the footer reopens the settings. Local previews and the
smoke test run on another host, so they show no banner and send nothing.

The assembled configuration file is named after a hash of its contents. Changing the
view or the environment at one source revision therefore keeps the earlier settings
for cached pages and for an index rollback.

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
[knowledge/ci-flow.md](../knowledge/ci-flow.md#model-browser-publication) lists the
jobs of both publication paths.

Every push and pull request builds the developer view,
so staging always shows it after a push to Main or Develop. Only a manual
[**Deploy Model Browser & Docs**](#deploy-from-github-actions) dispatch with the
`customer` audience builds the
customer view of the main catalog. Both use the same staging API and
publication URL. The last publication wins; the catalog branch selector does not
change the site's view.

### Staging release notice

Staging builds show an unofficial-release notice above the browser navigation,
including the connected backend environment and the frontend's source branch and
commit. These are the browser build's details, independent of the selected model
catalog branch. A staging browser previewing production data keeps the notice
and identifies the production backend. Local builds with uncommitted changes
say so beside the commit. Production builds omit the staging notice.

### Hero image CDN

Staging and production developer builds use their own public-reader hostname's
`/images/sha256/<hero artifact SHA-256>.png` path. The hash comes from catalog
metadata; no signed URLs or extra per-image API resolution are needed. Vulcan
publishes validated hero derivatives into a private S3 origin when documentation
runs finish. An image published before the CDN held it needs a backfill, which the
Vulcan repository's `docs/artifact-registry-hero-cdn.md` runbook describes. Without
one, the browser loads that image from the registry API.

`MODEL_HERO_CDN_BASE_URL` is an optional build setting. Its default is the matching environment
CDN path for developer builds using either standard registry API; customer builds and
custom API origins default to disabled. Set it to an empty string and rebuild to
roll back to API image URLs. The source preview `config.js` selects staging CDN.
Image loading retries twice on the CDN and then uses the existing API URL with
bounded retries. Removed cards stop retrying, and retries preserve immutable URLs
so they do not bypass CloudFront caching. Model downloads and README text keep
their existing routes. No model definitions, fingerprints or compilation behavior
change.

After a deployment that changes image loading, test a fresh catalog load, a model
detail page, and an image missing from the CDN (API fallback). Inspect image request
paths, verify that no repeated failure loops occur, and check staging reader
throttles. Backend tests and local browser tests do not replace this staging
acceptance check.

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

The production browser is hosted at <https://developer.sima.ai/models/> and reads
`https://models-registry.neat.sima.ai`. Deploying the frontend does not publish,
promote, or rebuild model catalogs. An unpublished branch remains empty, and the
browser never substitutes staging data. No push publishes the production frontend.
Deploy it from GitHub Actions or with the operator script.

### Deploy from GitHub Actions

Run **Deploy Model Browser & Docs** from the protected `develop` or `main`
branch. Select `staging` or `production` and the developer or customer audience.
The selected workflow commit is the frontend source revision. This workflow
builds and tests the browser and bundled docs, then publishes only `models/`
using that environment's Terraform-managed bucket, KMS key, identity and CDN.
It neither builds models nor publishes the registry tooling package. Production
Basic Auth is managed by infrastructure, and this workflow leaves it enabled.

The dispatching user must belong to `owners-models` and approve their own
`model-browser-deploy` environment gate. GitHub's Run button is visible to other
writers, but their runs cannot deploy, even if a different owner approves them.
The gate allows only that team, disables administrator bypass, and restricts
workflow refs to `develop` and `main`. The authorization job verifies the gate
and the dispatcher's approval before build/configuration/AWS jobs. Reruns are
rejected to avoid reusing historical authorization; dispatch a fresh run instead.
No additional organization-read token is needed.

The environment protection payload is versioned in
`config/model-browser-deployment-environment.json`. Its two deployment branch
policies are exact branch names `develop` and `main` (no tags or wildcard refs).
The environment gate is configured separately through repository administration and
checked again at runtime. GitHub shows the **Run workflow** button only for a
workflow that exists on the default branch, `main`.

Tooling publication (**Build and Publish Model Registry**) follows repository
`VULCAN_ENV` on both push and manual invocation, and uses staging when it is unset.
Its destination is independent of this workflow's environment selector.

### Deploy with the operator script

From a clean committed checkout, with AWS production operator credentials,
Node dependencies, Chromium and the Python registry dependencies available:

```bash
AWS_PROFILE=<production operator profile> PRODUCTION_AWS_ACCOUNT_ID=<account> \
  PRODUCTION_DOCS_BUCKET=<bucket> PRODUCTION_CLOUDFRONT_DISTRIBUTION_ID=<distribution> \
  scripts/deploy-production-model-browser.sh developer
```

The script refuses to publish unless the active credentials belong to that
account and the bucket is encrypted with one of its keys.

Use `customer` for Main-only presentation. The script builds and smoke-tests the
portal, publishes only the production docs bucket's `models/` prefix, and
invalidates only `/models` paths. It retains old assets and uploads the entry
point last.

Before deployment, record the prior root index object version and
source revision. Roll back by restoring that index version and invalidating
`/models` and `/models/*`; for a first deployment with no prior page, remove only
the newly created root index to disable the entry point. Restore the previous
full build from its source revision if documentation routes also need rollback.
Model registry contents and publisher environment selections are unchanged.
