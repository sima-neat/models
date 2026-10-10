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

It serves `portal-build/` against the preview API fixture in Chromium and fails
when a module or stylesheet does not load, the page throws or logs an error, or
the model list or a variant page does not render.

To preview the complete ResNet-50 accuracy and schema-v4 benchmark UI without a
registry or DevKit connection, run:

```bash
python3 artifact-browser/preview.py
```

Open the URL printed by the script. The local server provides a realistic model
catalog with fourteen builds spanning multiple models, build profiles, branches,
artifact sizes, accuracy evidence, and benchmark availability states. The
initial ResNet-50 INT8 build uses the 500-image local accuracy result so the
sample scope, reference comparison, and provisional status can be reviewed.
The catalog intentionally overflows
the comparison dialog so sticky headers and vertical scrolling can be checked.
The base three-trial result fixture is stored in
`artifact-browser/fixtures/performance-result-v4.json` and can be edited to
exercise other metric values or availability states. Its
`preview_only_extension_metric` is deliberately synthetic and exists only to
verify that unknown additive metrics render gracefully.

Benchmark adapters preserve every metric that follows the result contract.
Known metrics receive curated labels, formatting, and summary placement;
unknown additive metrics appear automatically under Additional measurements and
in the comparison measurement selector. Unsupported schema versions still link
to the raw result instead of guessing at incompatible structures.
Schema-v3 previews include model-load and time-to-first-inference measurements
alongside steady-state latency, throughput, power, energy, and MLA memory.

## Staging deployment

Changes to `artifact-browser/**`, `docs/**`, or `website/**` on `main` are
assembled into one static portal and published by
`vulcan-model-browser.yml` to the staging docs bucket under `models/`. The
Developer Center CloudFront distribution routes that prefix to the docs bucket,
making the browser available at <https://developer-stg.sima.ai/models>.

The workflow can also be started manually. It resolves the bucket, publisher
role, KMS key, and CloudFront distribution from the shared staging Vulcan
configuration; those deployment values are not hard-coded in this repository.

The default is the WAF-protected public staging registry at
`https://models-registry.stg.neat.sima.ai`. The API must allow cross-origin
`GET` requests from the static app's origin. The separate
`artifacts-registry.stg.neat.sima.ai` endpoint is reserved for IAM-signed
publisher traffic and rejects direct public reads.

The initial view loads the repository branch catalog independently from build
runs and caches both so repeat visits can render immediately while refreshing.
The selected branch's model cards come from completed compile runs with that
exact `git_ref`; the model catalog supplies names and categories, not build
membership. Components are grouped only within the same workflow build, and a
full package appears only when its recorded component run IDs match the group.
Build history lists one entry per variant build, newest first, naming its
components; an entry gains components as **Load more builds** brings in the rest of its
build. If branch discovery is
temporarily unavailable, the browser keeps saved branches and can infer
additional branches from returned runs. Full
provenance, validation, and metric views load only after a build is selected.
Public discovery requests completed builds only. The primary search matches
model, build-profile, and component names together with category names; secondary branch,
compiler, type, task, architecture, paradigm, and input facets are collapsed by
default and display an active-filter count when used.
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

A page shows one variant build: the component runs of one model and build
profile compiled by one workflow. Build-level facts appear once: the model card,
full package, provenance, source inputs (one per distinct digest), license, and
DevKit accuracy. The **Components** section has one row per component with its
MPK size, latency, throughput, and power; expanding a row shows its checksum,
run identity, benchmark detail, numerical fidelity, and compile parameters.
Components are measured on their own, so their latencies do not add up to
end-to-end latency. Comparison selects whole variant builds and lists each
component under its variant without combining measurements.

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
and click the model download icon to review the RF-DETR component dialog.

Run the download and variant-build checks with `node --test tests/browser/*.test.mjs`
and the existing UI checks with `python -m pytest tests/unit/test_artifact_browser.py`.
When a catalog build declares `artifact_suffix`, the dialog and download use
`<model_id>_<artifact_suffix>_mpk.tar.gz`. The browser sends `download_filename` to
the registry artifact endpoint; deploy registry support before this consumer.
Existing objects, checksums, and run metadata stay unchanged. Builds without a
suffix and history downloads retain their existing filenames.

Changing the trial comparison metric updates the chart in place, preserving
scroll position, keyboard focus, and expanded detail sections.

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
