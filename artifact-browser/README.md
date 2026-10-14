# Model Artifact Browser

A client-only web interface for the public SiMa.ai Model Registry read APIs.
The browser calls the registry directly; there is no application backend,
credential handling, or proxy middleware. Downloads use the temporary URL
returned by the registry directly.

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
The preview's benchmark results start from
`artifact-browser/fixtures/performance-result-v5.json`, a ResNet-50 result
that `import-core-benchmark` produced from a DevKit Core benchmark, so its
board telemetry, rails, and sensors are real readings; the preview varies only
speed, energy, and MLA memory per build. Edit it to exercise other metric
values or availability states. The three-trial
`artifact-browser/fixtures/performance-result-v4.json` keeps the schema-4
shape the registry still serves. Its `preview_only_extension_metric` is
deliberately synthetic and exists only to verify that unknown additive metrics
render gracefully.

Benchmark adapters preserve every metric that follows the result contract.
Known metrics receive curated labels, formatting, and a place in a benchmark
group; unknown additive metrics appear automatically under Other measurements.
Unsupported schema versions still
link to the raw result instead of guessing at incompatible structures.

## Staging deployment

Every push to `main` or `develop` runs `vulcan-model-orchestrator.yml`, whose
`portal` job assembles one static portal with `developer-docs.yml` and
browser-tests and smoke-tests it. Its `publish-model-browser` job then
publishes that portal to the staging docs bucket under `models/`. The
Developer Center CloudFront distribution routes that prefix to the docs bucket,
making the browser available at <https://developer-stg.sima.ai/models>.

A manual staging dispatch of the orchestrator on either branch republishes it. The job resolves the bucket, publisher
role, KMS key, and CloudFront distribution from the shared staging Vulcan
configuration; those deployment values are not hard-coded in this repository.

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
architecture filters sit beside the cards; branch, compiler, type, paradigm, and
input filters stay under **More filters**, which opens when one of them is active.
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
category tags, **Download**, and **Copy CLI command**. Below it, the variant
tabs, then section links to **Overview**,
**Performance**, **Accuracy**, **Details**, and **License**. A section without
data is left out.

The page is read at a glance first: each section shows its key facts, and
everything longer sits in one expandable row.

- **Overview** shows the hero and the README's `## Overview`. **Full model
  card** loads the README on first open, checks its SHA-256 against the
  catalog, and renders it from the section after Overview, because the title,
  hero, and Overview are already on the page.
- **Performance** shows throughput, latency, SoM power, and energy per
  inference for each component. **Benchmark details** shows the speed, power
  and energy, heat and load, memory, and startup groups, metrics the runtime
  did not report, the board telemetry source, and the result schema, tool, and
  raw-result link. **Power rails, sensors, and CPU cores** lists each rail,
  temperature sensor, and CPU core when the result carries them.
- **Accuracy** shows the status, the first four metrics, and the scope.
  **Evaluation details** explains status and coverage and lists the context
  and remaining metrics.
- **Details** shows the package card (size, SHA-256, **Download package**,
  **Download components**), then rows for build provenance, source models (one
  per distinct digest), and each component: checksum, run identity, numerical
  fidelity, and a nested **Compile parameters** row.
- **License** is one line with the SPDX identifier and redistribution, plus
  the attribution.

Components are measured on their own, so their latencies do not add up to
end-to-end latency.

## Compare page

**Compare** in the top bar opens a page with three build pickers over one
table. Each picker lists the variants of the selected branch's catalog and
resolves to the build the model page would show. Opening Compare from a model
page fills the first empty picker with that variant. The link records the picks
as `view=compare&build=<model id>.<build profile>`, one `build` per picker, so
a shared link reopens the same comparison.

The table shows task, component count, throughput, latency, SoM power,
energy per inference, and the headline accuracy metric with its published
score. A multi-component build lists one line per component and never adds
them up. Bold marks the best value among single-component builds, or among the
same component of different builds; a row with one value has no mark. Builds
from build history and metrics outside these rows are not comparable here;
their model pages show them.

## How the page is built

The browser is plain ES modules with no build step and no framework. Each view
renders HTML strings into its container and `app.js` handles events by
delegation.

| Module | Owns |
| --- | --- |
| `app.js` | Startup, routing from URL state, and event handlers |
| `state.js` | Shared page state, configuration, and DOM element lookup |
| `api.js` | Registry requests, caches, and catalog build selection |
| `branch-builds.js` | Which builds a branch shows, and grouping by workflow |
| `catalog-view.js` | Model cards, build history, search, and filters |
| `model-view.js` | The model page |
| `compare-view.js` | The Compare page: pickers and table |
| `benchmark-adapters.js` | One shape for every supported result schema |
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
`model-card` row is what triggers the README fetch.

Buttons with an icon start actions (download, copy). Blue text with ↗ is
reserved for links that leave the browser. Cards group related numbers only
(the KPI tiles, benchmark groups, and the package); the rest of the page uses
sections and thin rules.

`website/scripts/assemble-portal.mjs` renames every module with the revision
for the deployed portal and fails when a module imports a file missing from its
`fingerprintedAssets` list, so add a new module there.
It also writes `component-pipelines.<revision>.json` from the `pipeline` field
of each `model.yaml`, which draws the component order on a multi-component
model page; `preview.py` serves the same data from `models/`. Run
`build:portal` with the repository installed or `PYTHON` set, as for
`preview.py`.

## Downloading model variants

The model download button includes every model-pack component from the selected
variant's workflow build, commit, branch, target, and compiler. All
variants show a confirmation dialog listing component names, actual archive filenames, and file sizes.
Choose **Download N files** to start the transfers in parallel, or **Cancel**
(or Escape) to leave without starting any downloads. Cancel also works while
links are being prepared. Single-file variants require confirmation with **Download 1 file**.
Numerical-fidelity **Download JSON** remains a separate report download.

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
When a catalog build declares `artifact_suffix`, the dialog and download use
`<model_id>_<artifact_suffix>_mpk.tar.gz`. The browser sends `download_filename` to
the registry artifact endpoint; deploy registry support before this consumer.
Existing objects, checksums, and run metadata stay unchanged. Builds without a
suffix and history downloads retain their existing filenames.

## Duration reporting

Registry run timestamps cover artifact publication, which happens after compilation.
The UI labels this interval **Registry publication duration**. **Compile recipe
duration** comes from `metadata.compile_timing`, captured around the compilation
recipe by the compile workflow. It includes surgery and any validation performed
inside that recipe; it excludes runner queue/setup, separate accuracy jobs,
publication, and subsequent benchmarks. Components compiled together share this
invocation timing. Older builds display **Not recorded** instead of presenting
publication time as compile time; the workflow link provides historical job timing.

The publisher accepts optional `compile-timing.json` alongside `compile-outputs.json`,
validates its timezone-aware start/end timestamps before any registry writes,
and publishes it as additive run metadata. Existing outputs remain publishable.
No registry API or database schema changes are required.
