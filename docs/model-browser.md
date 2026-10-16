# Model browser

The model browser is the public catalog of the Model Registry. People use it to
find a model, check how its builds perform, compare builds, and download a
package. This page explains where each part of the browser gets its content, so
you can check how your model will appear before you submit it.

## The catalog

The catalog is a grid of cards, one per model. A card shows four things, all
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

Search matches model, build-profile, and component names and the taxonomy
labels. Task and model family filters sit beside the cards. Architecture, type,
input, branch, and compiler filters are under **More filters**. A filter lists
only the values the loaded builds use and is hidden when fewer than two remain.

To compare builds, choose **Compare** in the top bar. The page has three
pickers, one per build; opening it from a model page fills the first picker
with that build. The table shows each build's task, component count, Modalix
performance, and headline accuracy. A model with several components lists each
component on its own line, and the best value is marked in bold only against
the same component of another build, or between single-component builds.

**Buy**, at the right of the top bar, opens the DevKit store at
[devkit.sima.ai](https://devkit.sima.ai/) in a new tab. On a phone, the top
bar shortens **Developer docs** to **Docs**.

## The model page

Selecting a card opens the model page. The page is built to be read at a glance:
each section shows its key facts directly, and longer detail sits in a row
marked with › that opens in place. A model with several build profiles shows
one tab per profile; INT8 profiles come first and the page opens on the first
tab, whatever order `model.yaml` declares. The GitHub link in the header opens
the model's folder at the commit the README comes from. Sections appear in this
order:

1. **Overview.** The hero and the `## Overview` section of the README. The
   **Full model card** row opens the rest of the README, its **Inputs and
   outputs** and **Building applications** sections, so the title, hero, and
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
   **Execution graph** then draws the steps the component runs, from input to
   output, each tagged with the processor it runs on; the step on the MLA is
   the **Compiled Model**. **Details**, at the right of its heading, opens a
   table of the tensors passed between steps with their data type, shape,
   scale, and zero point. **Benchmark details** has three cards on the left:
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
   every metric, every SoM rail, and every CPU core. A metric the browser does
   not place in the tiles or the cards appears under **Other measurements**,
   so a new metric shows up without a browser change. A
   multi-component model whose `model.yaml` declares a `pipeline` first shows
   the order its components run in, from input to output: numbered components,
   with any CPU step between them, such as RF-DETR's **Top 300 proposals**.
   Each component's measurements then appear under the same number.
3. **Accuracy.** The status, the first four metrics, the evaluated sample
   count, and the commit and GitHub workflow of the run that measured them.
   **Evaluation details** explains the status and coverage and lists
   the remaining metrics, each with its published score when `model.yaml`
   records one.
4. **Details.** The package with its size, checksum, and download buttons,
   then rows for build provenance (the commit and workflow that compiled the
   build), source models, and each component's files,
   checksum, numerical fidelity, and compile parameters.
5. **License.** The SPDX identifier, whether redistribution is allowed, and the
   attribution.

A section that has no data, such as Accuracy for a model without an accuracy
entry, is left out. Components are measured on their own, so the latencies of a
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
