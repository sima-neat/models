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

The browser has two views. The customer view shows the Main catalog and hides the
branch and compiler filters, build history, the Details section and contributor
navigation. It also leaves out the board's Linux memory, MLA Utilization, the
**Raw result** download, and the commit and workflow links of each result. The
developer view keeps all of them. Selecting a branch in the developer view changes
the catalog, not the view.

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
loaded builds use and are hidden when fewer than two remain. Selecting a task
narrows Model family to the families with that task, and selecting a family
narrows Task to that family's tasks; developer controls
stay visible even when only one branch or compiler is available.

**Compare** in the top bar explores all model variants for one task. The
chart shows all measured precisions together. Hover over a point to see the model,
variant, throughput and accuracy. Accuracy is the vertical axis and throughput
in inferences/s is the horizontal axis. The dataset and its variant appear below
the chart title. Accuracy is grouped by dataset and metric; **Dataset & metric**
at the top right names the group on every task and selects one when a task has several. The axis title gives the
metric's unit and says "lower is better" for an error metric such as EPE.
A multi-component model shows its [whole-model result](benchmarking.md#multi-component-models), with all its components running as one pipeline. Select up to three points or use the compact
**Add model** selector to show detailed measurements, including power and memory.
Remove a selected variant using the × in its table heading or its chart label.
The chart labels only selected variants plotted against its current dataset and metric.
Selections without matching accuracy or throughput appear only in the comparison table.
The graph, table and model selector update together. The link
preserves the task and selected variants. Text Generation and Visual Question Answering
compare parameters in billions against generation speed for matching workloads.
Image comparisons use each model's encoder resolution, listed in its tooltip and table.
A workload selector appears when several text buckets or datasets are available.
The Task menu leaves out tasks with no comparison yet, such as the audio tasks. Multimodal detectors can still join a vision-task comparison.

On a phone, the section navigation shortens **Developer docs** to **Docs**.

## The model page

Selecting an individual model opens its page. The page is built to be read at a glance:
each section shows its key facts directly, and longer detail sits in a row
marked with › that opens in place. The header places the model image beside its
name, tags, description, download action, and the validation and source-link
icons in both UI modes; narrow screens stack them. The tags name what holds for every variant:
tasks, model family, model type and license. The download dialog offers an outlined **Copy CLI command**
button with a copy icon alongside browser downloads where available. The copied
command names the selected branch in the developer view. In the customer view it
names no branch, because `sima-cli` downloads from `main` by default. A package dialog names the model, summarizes the variant
and size, and shows a quiet filename block and license line with a **Download**
action, in both views. Dialogs and action buttons use softly rounded corners.
The model's License section retains attribution in both views. Dialog actions
place Cancel on the left, Copy CLI command in the middle and Download on the right.
A model with several build profiles shows
one tab per profile, labelled with its variant ID in uppercase, such as
**W8A8**. Profiles with a W8A8 component come first and the page opens on the
first tab, whatever order `model.yaml` declares. **Download** in the header
downloads the selected variant, and the Performance and Accuracy titles name that variant.
A model whose license does not permit redistribution has no download action. Its
header shows **Not available for download** in that place, and its Details section
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
   [README requirements](model-readme.md).
2. **Performance.** One whole-model result for the selected build profile,
   read from the profile's [model report](benchmarking.md#workload-and-measurements).
   Core models show four tiles: throughput, average latency, average SoC
   power (the MLA rail plus the rail for the rest of the SoC), and inferences
   per watt, which is throughput in inferences/s divided by that SoC power
   in watts and is shown without a unit. Energy per
   inference, which uses SoM power, is in the CSV. The line below them says when the result was
   measured; the workload the report records is in the CSV. In the developer UI it also links the commit and GitHub workflow
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
   leaves out Linux memory, here and in the CSV. Readings other than latency and throughput cover
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

For models declaring `performance.scope: components`, the Performance section
shows a component selector. Each selection has its own throughput, latency,
power, memory, workload and telemetry, with no combined score. The catalog table
lists the components' measurements separately. Models using the default
whole-model scope, such as RF-DETR, keep their pipeline-level result.

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

A feature branch shows its finished on-demand builds before they are promoted.
Develop and Main show only promoted on-demand builds. See
[Model promotion](model-promotion.md).

## Preview your model locally

`artifact-browser/preview.py` serves the browser with a local registry API built
from `models/`. It shows your model's card, README, hero, build profiles,
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
page. The preview is for checking content and layout. The real numbers come
from CI after merge.
