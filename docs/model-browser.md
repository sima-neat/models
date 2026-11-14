# Model browser

The Model Browser is the public catalog of SiMa Neat models. People use it to
find a model, check how its builds perform, compare builds, and download a
package. This page explains where each part of the browser gets its content, so
you can check how your model will appear before you submit it.

## The catalog

Models uses the shared Developer Portal header on the deployed site. Its own
navigation underneath contains **Browse models**, **Compare** and **Developer docs**.
The section navigation matches Software documentation: full-width, left-aligned
links with a blue active label. The catalog and contributor documentation share
the portal's theme preference.
Standalone previews retain local navigation when the shared header is unavailable.

CI builds the customer UI from `main` and the developer UI from `develop` and
feature branches. Both currently use the staging registry and share the staging
publication URL; the most recently published build determines its UI. The customer
view uses the main catalog and hides branch/compiler filters, build history,
the Details section and contributor navigation. It also leaves out SoM power,
energy per inference, the board's Linux memory, MLA Utilization, the **Raw
result** download, and the commit and workflow links of each result. The
developer view keeps all of them. Selecting a branch in the developer UI changes the catalog, not the UI.

While the catalog loads, the list shows a spinner. A message that a branch has
no models appears only after the registry has answered.

The catalog opens as a grid of cards, one per available **Model family**, sorted
by its existing architecture ID. Selecting a family applies its sidebar filter
and shows individual model cards in the same grid. A family with one available
model opens that model directly. **All model families** clears only the family
selection; other filters remain. Family selection is shareable in the URL, and
returning from a model or using browser Back preserves the browsing context.
Branch, compiler and task filters apply before grouping, so families without a
usable build do not appear. Search always shows matching individual models.

**Cards** and **Table** switch between the two views and remember the choice.
The table always shows individual models,
selected precision, measured accuracy and whole-model performance. Changing
the precision updates its measurements; opening the model shows the full results.
LLM and VLM rows show generation speed in tokens/s and TTFT (time to first
token) as ranges across measured workloads, with the workload count. Hover over
the reading for the workload names. Text Generation rows use text workloads;
Visual Question Answering rows use image workloads. Open the model for each
workload's exact result.
Both views share search and filters. The table groups models by task, with the
same model-family ordering inside each group. Models with multiple tasks appear
in every applicable group; selecting a task filter shows only that group. The
model count counts unique models, and precision changes apply to every copy of
the model. Build history keeps its existing list.
Task labels use a single blue accent in light and dark mode; model names and
table task headings stay neutral.
Model cards show a blue outline and soft blue glow on hover in both themes.
Model metadata uses small tinted chips: blue for tasks, orange for precision
and purple for model type; other metadata stays neutral.
Dropdown menus use matching text and backgrounds in both themes.
Filter dropdowns stay neutral after mouse selection; keyboard use shows a neutral focus outline.
The selected precision tab has a soft blue fill and border.

A family card shows four things:

| Card element | Source |
| --- | --- |
| Image | First available model hero in the existing model order |
| Name | Architecture `display_name` in `taxonomies/model-taxonomy.yaml` |
| One-line description | Architecture `description` in the same taxonomy |
| Task labels | Union of tasks of the family's currently available models |

Descriptions cover the whole family. Historical catalogs without them omit the
description. No extra family image or maintained model list is needed.

Individual model cards use the model directory:

| Card element | Source |
| --- | --- |
| Image | `hero.png`, 1200x480 |
| Name | `display_name` in `model.yaml` |
| One-line description | `description` in `model.yaml` |
| Task labels | `tasks` in `model.yaml` |

Keep the name short and leave input sizes out of it; the build profile carries
them. Write the description as one plain sentence about what the model does.
Changing the name, description, or hero does not recompile or remeasure any
build, because none of them is part of a fingerprint.

The search field matches model, build-profile, and component names and the taxonomy
labels and family descriptions. Task and model family filters sit beside the cards. Developer view also
shows Branch, Compiler and Build history directly in the filter panel. Customer
view keeps only Task and Model family. Category filters list the values the
loaded builds use and are hidden when fewer than two remain; developer controls
stay visible even when only one branch or compiler is available.

**Compare** in the top bar explores all model variants for one task. The
chart shows all measured precisions together. Hover over a point to see the model,
variant, throughput and accuracy. Accuracy is the vertical axis and throughput
in inferences/s is the horizontal axis. The dataset, variant and split appear below
the chart title. Accuracy is grouped by dataset and metric; when more than one
group is available, choose it with **Dataset & metric**. The axis title gives the
metric's unit and says "lower is better" for an error metric such as EPE.
A multi-component model shows its [whole-model result](benchmarking.md#multi-component-models), with all its components running as one pipeline. Select up to three points or use the compact
**Add model** selector to show detailed measurements, including memory and, in the developer UI, SoM power.
Remove a selected variant using the × in its table heading or its chart label.
The chart labels only selected variants plotted against its current dataset and metric.
Selections without matching accuracy or throughput appear only in the comparison table.
The graph, table and model selector update together. The link
preserves the task and selected variants. Text Generation and Visual Question Answering
compare parameters in billions against generation speed for matching workloads.
Image comparisons use each model's encoder resolution, listed in its tooltip and table.
A workload selector appears when several text buckets or datasets are available.
Audio comparisons show **TBD**. Multimodal detectors can still join a vision-task comparison.

On a phone, the section navigation shortens **Developer docs** to **Docs**.

## The model page

Selecting an individual model opens its page. The page is built to be read at a glance:
each section shows its key facts directly, and longer detail sits in a row
marked with › that opens in place. The header places the model image beside its
name, tags, description, and the validation and source-link icons in both UI
modes; narrow screens stack them. The tags name what holds for every variant:
tasks, model family, model type and license. The download dialog offers an outlined **Copy CLI command**
button with a copy icon alongside browser downloads where available. Customer
package dialogs name the model, summarize the variant and size, and show a quiet
filename block and license line with a **Download model** action. Developer dialogs
use the same layout with extra fields for package identifiers, component counts,
format, checksums and attribution. Dialogs and action buttons use softly rounded corners.
The model's License section retains attribution in both views. Dialog actions
place Cancel on the left, Copy CLI command in the middle and Download on the right.
A model with several build profiles shows
one tab per profile, labelled with its variant ID in uppercase, such as
**W8A8**. Profiles with a W8A8 component come first and the page opens on the
first tab, whatever order `model.yaml` declares. **Download** sits at the end of
the variant row, because it downloads the selected variant, and the Performance
and Accuracy titles name that variant.
A model whose license does not permit redistribution has no download action. Its
variant row shows **Not available for download** in that place, and its Details section
has no package card. The GitHub link in the header opens
the model's folder at the commit the README comes from. The original-model link
beside it opens the page `upstream.model_card` names. It shows the Hugging Face
logo for a Hugging Face page and an external-link icon for any other host, such
as a GitHub release page. LLM and VLM pages show
the selected build's maximum context length (input plus output tokens) and image
input resolution directly below the variant selector. Native deployments supply
their verified settings, including fixed image resolutions without compiler
overrides; other builds show their recorded compile parameters. Sections appear
in this order:

1. **Overview.** The `## Overview` section of the README, including
   any model-specific note about changing context length or image resolution
   during compilation. The **Inputs and outputs** and **Building applications**
   sections appear inline from the
   rest of the README after its checksum is verified, so the title, hero, and
   overview are never shown twice. The README follows the
   [README requirements](https://github.com/sima-neat/models/blob/develop/AGENTS.md#model-readme-requirements).
2. **Performance.** One whole-model result for the selected build profile,
   read from the profile's [model report](benchmarking.md#workload-and-measurements).
   Core models show four tiles: throughput, average latency, average SoM
   power, and energy per inference. The customer UI shows the first two. The
   line below them names the workload the
   report records, such as **YoloV26 box decoder, top 100**, and when it was
   measured. In the developer UI it also links the commit and GitHub workflow
   of the run that measured the profile. A benchmark run is reused until its
   [benchmark fingerprint](benchmarking.md#ci-and-registry-publication) changes, so that
   commit can be later than the build's. A profile with no result shows **Not measured at
   this revision.**
   LLMs and VLMs show TTFT, prefill and generation tables for text, with image timings
   in a separate table. TTFT headings explain time to first token in a tooltip.
   Speculative models compare **Baseline** with **MTP**, **DFlash** or **EAGLE3**
   when the compiled build identifies its method. The baseline is the target
   model without speculative decoding. Generation speedup excludes prompt
   processing and the first token; draft contribution is the percentage of
   generated tokens accepted from the draft model. Measurement details show
   each workload's telemetry for both deployments. Each CSV identifies its own
   model, build profile and decoding role; filenames include the workload and
   role to distinguish the exports.
   When the report includes MLA layer stats, **Runtime Layer Analysis**
   shows one full-width table with **Compiled layer / Execution time (µs) / Share (%)**
   headings. Each execution-time cell shows its exact value above a thin bar,
   colored by operation family with a compact legend. Families are inferred from
   compiled names: Convolution, Matmul / Einsum, Activation, Pooling, Elementwise,
   Tensor layout, Normalization, Resize, Reduction, Input / output and Other.
   Matching follows the named operation, rather than searching the whole string.
   Compiler-generated slice/concat vertices are Tensor layout even when their names
   reference another computation. Segments retain their computation's family;
   placeholders and `_output` vertices are Input / output. Fused `conv2d_add_relu`
   is Convolution, while `add_relu` is Elementwise. Nested reshape transformations
   stay Tensor layout. Unrecognized operations use Other. This is a display
   classification, not a source-graph operator mapping or consumer-time attribution.
   The benchmark API filters out input placeholders but retains `_output` entries;
   the Input / output family in those reports therefore describes output handling.
   Local previews use synthetic layer profiles covering the ten recognized families for UI
   review; those profiles do not represent the selected model's actual graph.
   The legend shows families present in the component and stays
   unchanged when filtering. All bars share a time axis from zero to the rounded-up
   maximum execution time, with a midpoint tick on desktop. Desktop columns have equal
   widths and aligned starting points; narrow screens reserve space for percentages.
   Rounded bars, a quiet axis line, soft row separators and a hover highlight help
   follow each layer's measurements across the row.
   Hovering anywhere on a layer's row emphasizes its matching family in the legend,
   without shifting the labels. Moving between cells preserves the emphasis; it
   clears when the pointer leaves the row.
   **Runtime order / Slowest layers first** changes the row order; Runtime order
   follows the reported layer sequence, and Slowest layers first is the default.
   The bar scale stays fixed when filtering or reordering. Name
   search and a minimum-time slider preserve the selected order.
   Each layer also identifies its stage and compiler position. The summary shows
   profiled MLA time, the layer count (including zero-time entries), and the ten
   slowest layers' share. Filtering preserves the totals, bar scale and percentage
   denominator. These timings come from a separate profiling pass; they are not
   a partition of headline latency. Compiled layer names may differ from ONNX
   operators. Historical reports without stats omit this section.
   In developer mode, **MLA Utilization** follows Runtime Layer Analysis. Like the
   layer summary, it shows each component's totals as tiles, each explained behind
   an (i) mark: utilization (effective TOPS as a share of the MLA's peak: 50
   TOPS for a w8a8 component, and a quarter of that, 12.5 TOPS, for one with
   BF16 activations, whose multiplies take four times the cycles), effective
   TOPS (the simulator's count of operations that contribute to the result,
   divided by the MLA time Core measured on the board), packing (their share of all operations the MLA executes), and the
   measured MLA time. **Details** opens a table of each MLA stage's measured and
   simulated latency, effective operations, packing, effective TOPS, and
   utilization. For a single-component model, the summary gives the effective TOPS
   the model sustains at its measured throughput, counting one MLA run per batch in
   the execution plan.
   A component whose stages were not
   measured or simulated, or whose board and simulator stages differ, says so instead.
   Customers do not see this section, because its rates rest on a simulator estimate.
   **Performance Details** has three cards on the left:
   **Performance** with the throughput, the mean latency, and the p50, p95,
   and p99 latency in one row; **Heat** with the SoC and SoM temperatures;
   and **CPU** with the core count, the load over all cores and on the
   busiest core, and a bar per core with its average and peak. On the right,
   **Power** opens with a bar that splits the average SoM power by rail; the
   two SoC rails sit in one block outlined like the SoC row's mark and
   coloured like the MLA and Non-MLA rows. Below it are the average and peak
   of SoM power, then SoC power indented under it, then the two rails that
   add up to SoC power indented under SoC: MLA and Non-MLA (the rest of the
   SoC). Rails peak at different moments, so the SoC peak is the sum of the
   two peaks, an upper bound. The other SoM rails follow at the SoC's level,
   each coloured like its part of the bar. **Memory** has MLA and
   EV74 memory, which add up, and the board's Linux memory. The customer UI
   keeps MLA power and MLA and EV74 memory, and leaves out SoM power, the
   rail bar, the other rails, and Linux memory, here and in the CSV. Readings other than latency and throughput cover
   the throughput pass. The (i) mark beside a reading or
   rail explains what it measures. The footer lists the telemetry (the
   Sentinel version and the samples behind the board readings, or why they
   are missing) and the runtime. Beside it, **Raw result** downloads the
   model report and **Download CSV** saves one table of the result's context,
   every metric, every SoM rail, every CPU core, and all compiled layer timings.
   A metric the browser does not place in the tiles or the cards appears under **Other measurements**,
   so a new metric shows up without a browser change.
   Runtime Layer Analysis sits below Performance Details, followed by an **Execution Graph** disclosure that
   draws the steps the component runs, from input to output, each tagged with the
   processor it runs on; the step on the MLA is the **Compiled Model**. Its summary
   lists those processors. The diagram uses the available width, with equal-width
   Input and Output boxes and **Details** beside it. This button opens a
   table of the tensors passed between steps with their data type, shape,
   scale, and zero point. Hover, focus or tap a processing stage for a short explanation of
   what it does, with its processor named in a badge at the bottom. The top accent
   line and badge use the same colour as the graph. Input and Output have no tooltips.
   Tensor information stays under Details. A
   multi-component model whose `model.yaml` declares a `pipeline` first shows
   the order its components run in, from input to output: numbered components,
   with any CPU step between them, such as RF-DETR's **Top 300 proposals**.
   The tiles, context line, and **Performance Details** are the
   [whole-model result](benchmarking.md#multi-component-models), with the
   components and CPU step running together. **Runtime Layer Analysis**,
   **MLA Utilization**, and **Execution Graph** each switch between the
   components, and **Download CSV** lists each component's layer timings.
3. **Accuracy.** For each dataset the run evaluated, in `model.yaml` order:
   the coverage (full when the run evaluated every sample of its split,
   sampled otherwise), the first four metrics in their units (ratios as
   percentages, EPE in pixels, D1-all in percent), the dataset, such as
   **coco-2017 val**, the evaluated sample count when coverage is sampled, and,
   in the developer UI, the commit and GitHub workflow of the run that measured them. With more than
   one dataset, each result is titled by its dataset name. The catalog table
   and **Compare** use the first dataset. Under each of the four metrics, both
   views show its published score and its FP32 baseline when `reference_metrics`
   records them for the measured dataset. A published score links to the
   source `reference_metrics` names.
   In the developer UI, each dataset's **Evaluation details** explains the coverage and lists
   the remaining metrics, each with its published score and its FP32 baseline
   when `reference_metrics` in `model.yaml` records them for the measured
   dataset. The **Published scores** and **FP32 baseline** rows list each
   origin's entries with their dataset, conditions, and source or script. A
   build with no accuracy result yet shows only the reference scores, each
   origin under its own explanation.
4. **Details (developer UI only).** The package with its size, checksum, and download buttons.
   A package downloads as a ZIP of its MPKs. A single-component build published before
   packages became ZIPs downloads as its MPK.
   Then come rows for build provenance (the commit and workflow that compiled the
   build), source models, the package's numerical fidelity, and each
   component's files, checksum, and compile parameters.
5. **License.** The license identifier and name, whether redistribution is
   allowed, one sentence on what the license asks, and the attribution.

A section that has no data, such as Accuracy for a model with neither an
accuracy result nor published scores, is left out.

## Preview your model locally

`artifact-browser/preview.py` serves the browser with a local registry API built
from `models/`, so it shows your model's card, README, hero, build profiles,
packages, provenance, and accuracy and fidelity declarations as this checkout
defines them. Run IDs, timestamps, sizes, and measured values are synthetic. It
needs a Python environment with the repository installed:

```bash
python -m pip install .
```

```bash
python artifact-browser/preview.py
```

Open the URL the script prints, find your model in the catalog, and open its
page. The preview is for checking content and layout; the real numbers come
from CI after merge.

### Model folder downloads

A directory package opens a **Download model folder**
dialog explaining that the model is provided as individual files in a folder,
similar to a Hugging Face repository. This applies to any directory package,
including Whisper, LLMs, and VLMs. The dialog shows its file count, total download
size, and a copyable `sima-cli models download` command. The command uses the selected branch and adds `--stg` for
the staging registry. Commands are offered for the current catalog deployment;
historical builds point to the current variant instead. Folder downloads
require sima-cli 2.1.19 or later to download the files in parallel and assemble
the model folder.
The CLI verifies the manifest and each file and recreates their relative paths.
The displayed size covers all deployment files; **Manifest SHA-256** identifies
the index. MPK and ZIP packages keep their browser download flow.

When the registry resolves a deployment to Hugging Face, the dialog links to
its pinned repository revision and optional subdirectory. This location is
separate from the original source checkpoint linked in the page header.

Branch views honor explicit `branch_deployments` (or `approved_deployments`)
selections, including inherited compilation runs and their matching package.
Newer unselected runs do not replace an explicitly selected deployment.
Feature branches show finalized on-demand candidates before promotion; metadata
and unselected runs do not create a card. Develop and Main show approved deployments.

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

## Production frontend

The production browser is hosted at [developer.sima.ai/models](https://developer.sima.ai/models/) and
uses `https://models-registry.neat.sima.ai`. Build with
`MODEL_BROWSER_ENVIRONMENT=production` to select both production origins and
the Production label. Audience selection is independent: developer mode exposes
branch selection; customer mode shows Main only. Deploying the frontend does
not publish, promote, or rebuild model catalogs. An unpublished branch remains
empty; the browser never substitutes staging data.

From a clean committed checkout, with AWS production operator credentials,
Node dependencies, Chromium and the Python registry dependencies available:

```bash
AWS_PROFILE=vulcan-prod scripts/deploy-production-model-browser.sh developer
```

Use `customer` for Main-only presentation. The script builds and smoke-tests the
portal, publishes only the production docs bucket's `models/` prefix, and
invalidates only `/models` paths. It retains old assets and uploads the entry
point last. The generated configuration filename hashes its contents, so changing
audience or backend at the same source revision preserves the previous settings
for cached pages and index rollback. Before deployment, record the prior root index object version and
source revision. Roll back by restoring that index version and invalidating
`/models` and `/models/*`; for a first deployment with no prior page, remove only
the newly created root index to disable the entry point. Restore the previous
full build from its source revision if documentation routes also need rollback.
Model registry contents and publisher environment selections are unchanged.

## Staging release notice

Staging builds show an unofficial-release notice above the browser navigation,
including the connected backend environment and the frontend's source branch and
commit. These are the browser build's details, independent of the selected model
catalog branch. A staging browser previewing production data keeps the notice
and identifies the production backend. Local builds with uncommitted changes
say so beside the commit. Production builds omit the staging notice.

## Manual frontend deployment in GitHub Actions

Run **Deploy Model Browser & Docs** from the protected `develop` or `main`
branch. Select `staging` or `production` and the developer or customer audience.
The selected workflow commit is the frontend source revision. This workflow
builds and tests the browser and bundled docs, then publishes only `models/`
using that environment's Terraform-managed bucket, KMS key, identity and CDN.
It neither builds models nor publishes the registry tooling package. Existing
production Basic Auth is infrastructure-managed and stays enabled.

The dispatching user must belong to `owners-models` and approve their own
`model-browser-deploy` environment gate. GitHub's Run button is visible to other
writers, but their runs cannot deploy, even if a different owner approves them.
The gate allows only that team, disables administrator bypass, and restricts
workflow refs to `develop` and `main`. The authorization job verifies the gate
and the dispatcher's approval before build/configuration/AWS jobs. Reruns are
rejected to avoid reusing historical authorization; dispatch a fresh run instead.
No additional organization-read token is needed.

Automatic staging frontend publication remains as configured in the model
orchestrator; production frontend publication is manual. Tooling publication
(**Build and Publish Model Registry**) follows repository `VULCAN_ENV` on both
push and manual invocation, defaulting to staging only when unset. Its destination
is independent of this frontend workflow's environment selector.

The environment protection payload is versioned in
`config/model-browser-deployment-environment.json`. Its two deployment branch
policies are exact branch names `develop` and `main` (no tags or wildcard refs).
The workflow must reach the default branch (`main`) before GitHub exposes its
Run workflow button; merge through the normal reviewed branch promotion. The environment gate is configured
separately through repository administration and checked again at runtime.

Developer deployments serve model hero illustrations from the matching registry
environment’s CDN. Existing production illustrations are backfilled before browser
cutover; image loading retains bounded retries and the registry API fallback.

For models declaring `performance.scope: components`, the Performance section
shows a component selector. Each selection has its own throughput, latency,
power, memory, workload and telemetry, with no combined score. The catalog table
lists the components' measurements separately. Models using the default
whole-model scope, such as RF-DETR, keep their pipeline-level result.
