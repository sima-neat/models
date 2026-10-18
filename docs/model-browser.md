# Model browser

The model browser is the public catalog of the Model Registry. People use it to
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
the Details section and contributor navigation. The developer view keeps those
controls. Selecting a branch in the developer UI changes the catalog, not the UI.

The catalog opens as a grid of cards, one per model. **Cards** and **Table**
switch between the two views and remember the choice. The table shows the model,
selected precision, measured accuracy and component performance. Changing
the precision updates its measurements; opening the model shows the full results.
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

A card shows four things, all
taken from the model directory:

| Card element | Source |
| --- | --- |
| Image | `hero.png`, 1200x480 |
| Name | `display_name` in `model.yaml` |
| One-line description | `description` in `model.yaml` |
| Task labels | `tasks` in `model.yaml` |

Keep the name short and leave input sizes out of it; the build profile carries
them. Write the description as one plain sentence about what the model does.
Changing the name, description, or hero does not recompile or remeasure any
build, because none of them is part of a stage key.

The search field matches model, build-profile, and component names and the taxonomy
labels. Task and model family filters sit beside the cards. Developer view also
shows Branch, Compiler and Build history directly in the filter panel. Customer
view keeps only Task and Model family. Category filters list the values the
loaded builds use and are hidden when fewer than two remain; developer controls
stay visible even when only one branch or compiler is available.

**Compare** in the top bar explores all model variants for one task. The
chart shows all measured precisions together. Hover over a point to see the model,
variant, throughput and accuracy. Accuracy is the vertical axis and throughput
in inferences/s is the horizontal axis. The dataset, variant and split appear below
the chart title. Accuracy is grouped by dataset and metric; when more than one
group is available, choose it with **Dataset & metric**.
Multi-component models are excluded for now. Select up to three points or use the compact
**Add model** selector to show detailed measurements, including power and memory.
Remove a selected variant using the × in its table heading or its chart label.
The chart labels only selected variants plotted against its current dataset and metric.
Selections without matching accuracy or throughput appear only in the comparison table.
The graph, table and model selector update together. The link
preserves the task and selected variants. LLM/VLM comparison shows **TBD**.

On a phone, the section navigation shortens **Developer docs** to **Docs**.

## The model page

Selecting a card opens the model page. The page is built to be read at a glance:
each section shows its key facts directly, and longer detail sits in a row
marked with › that opens in place. The header places the model image beside its
name, description and download action in both UI modes; narrow screens stack them.
The validation and source-link icons form a separate group beside Download, with
space between the groups. The download dialog offers an outlined **Copy CLI command**
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
first tab, whatever order `model.yaml` declares. The GitHub link in the header opens
the model's folder at the commit the README comes from. LLM and VLM pages show
the selected build's maximum context length (input plus output tokens) and image
input resolution directly below the variant selector. Native deployments supply
their verified settings, including fixed image resolutions without compiler
overrides; older builds fall back to recorded compile parameters. Sections appear
in this order:

1. **Overview.** The `## Overview` section of the README, including
   any model-specific note about changing context length or image resolution
   during compilation. The **Inputs and outputs** and **Building applications**
   sections appear inline from the
   rest of the README after its checksum is verified, so the title, hero, and
   overview are never shown twice. The README follows the
   [README requirements](https://github.com/sima-neat/models/blob/develop/AGENTS.md#model-readme-requirements).
2. **Performance.** For each component, four tiles with one number each:
   throughput, average latency, average SoM power, and energy per inference.
   The line below them names the workload: **Core default MPK**
   when the benchmark script uses Core's defaults, or the script's own
   selection, such as **YoloV26 box decoder, top 100**, followed by
   preprocessing, postprocessing, and the MLA and CPU clocks. It ends with
   the commit and GitHub workflow of the run that measured the component.
   CI reuses a build until its compile inputs change, so that commit can be
   later than the build's.
   When the report includes layer stats, **Runtime Layer Analysis**
   opens a **Chart / Table** selector. **Chart** shows
   measured execution times in microseconds with a **Slowest layers first** caption and a fixed time
   axis and a legend of layer families inferred from compiled names. **Table**
   shows names, execution time and share without bars, with a **Compiler order** caption.
   Its compact table is left-aligned, with fixed-width numeric columns; the chart
   uses the full available width. Name
   search and a minimum-time slider apply to either view; only one is displayed.
   Each layer also identifies its stage and compiler position. The summary shows
   profiled MLA time, the layer count (including zero-time entries), and the ten
   slowest layers' share. Filtering preserves the totals, time axis and percentage
   denominator. These timings come from a separate profiling pass; they are not
   a partition of headline latency. Compiled layer names may differ from ONNX
   operators. Historical reports without stats omit this section.
   **Benchmark details** has three cards on the left:
   **Performance** with asynchronous and synchronous throughput in one row
   (synchronous from the mean latency), the mean latency, and the p50, p95,
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
   each coloured like its part of the bar. **Memory** has MLA, CMA
   pool, and Linux memory. Readings other than latency and throughput cover
   the throughput pass. The (i) mark beside a reading or
   rail explains what it measures. The footer lists the telemetry (the
   Sentinel version and the samples behind the board readings, or why they
   are missing) and the runtime. Beside it, **Raw result** opens the result
   JSON and **Download CSV** saves one table of the component's context,
   every metric, every SoM rail, every CPU core, and all compiled layer timings.
   A metric the browser does not place in the tiles or the cards appears under **Other measurements**,
   so a new metric shows up without a browser change.
   Runtime Layer Analysis sits below Benchmark details, followed by an **Execution graph** that
   draws the steps the component runs, from input to output, each tagged with the
   processor it runs on; the step on the MLA is the **Compiled Model**. The diagram
   uses the available width, with equal-width Input and Output boxes and **Details**
   beside it. Its heading matches Benchmark details in size and aligns with the diagram. This button opens a
   table of the tensors passed between steps with their data type, shape,
   scale, and zero point. Hover, focus or tap a processing stage for a short explanation of
   what it does, with its processor named in a badge at the bottom. The top accent
   line and badge use the same colour as the graph. Input and Output have no tooltips.
   Tensor information stays under Details. A
   multi-component model whose `model.yaml` declares a `pipeline` first shows
   the order its components run in, from input to output: numbered components,
   with any CPU step between them, such as RF-DETR's **Top 300 proposals**.
   Each component's measurements then appear under the same number.
3. **Accuracy.** The status, the first four metrics, the evaluated sample
   count, and the commit and GitHub workflow of the run that measured them.
   In the developer UI, **Evaluation details** explains the status and coverage and lists
   the remaining metrics, each with its published score when
   `published_metrics` in `model.yaml` records one for the measured dataset.
   Its **Published scores** row lists every published entry with its dataset,
   conditions, and source. A build with no accuracy result yet shows only the
   published scores.
4. **Details (developer UI only).** The package with its size, checksum, and download buttons.
   A single-component package downloads as its MPK. A multi-component package
   downloads as a ZIP of its MPKs and also offers **Download components**.
   Then come rows for build provenance (the commit and workflow that compiled the
   build), source models, and each component's files,
   checksum, numerical fidelity, and compile parameters.
5. **License.** The SPDX identifier, whether redistribution is allowed, and the
   attribution.

A section that has no data, such as Accuracy for a model with neither an
accuracy result nor published scores, is left out. Components are measured on their own, so the latencies of a
multi-component model do not add up to end-to-end latency; the page says so.

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
