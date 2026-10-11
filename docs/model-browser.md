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
labels. Task and architecture filters sit beside the cards. Branch, compiler,
type, paradigm, and input filters are under **More filters**.

To compare builds, choose **Compare** in the top bar. The page has three
pickers, one per build; opening it from a model page fills the first picker
with that build. The table shows each build's task, component count, Modalix
performance, and headline accuracy. A model with several components lists each
component on its own line, and the best value is marked in bold only against
the same component of another build, or between single-component builds.

## The model page

Selecting a card opens the model page. The page is built to be read at a glance:
each section shows its key facts directly, and longer detail sits in a row
marked with › that opens in place. Sections appear in this order:

1. **Overview.** The hero and the `## Overview` section of the README. The
   **Full model card** row opens the rest of the README, starting at the
   section after Overview, so the title, hero, and overview are never shown
   twice. The README follows the
   [README requirements](https://github.com/sima-neat/models/blob/develop/AGENTS.md#model-readme-requirements).
2. **Performance.** For each component: throughput, average latency, board
   power, and energy per inference from its DevKit benchmark. **Benchmark
   details** groups every measurement into speed (latency per trial),
   heat (power and board temperature), memory (MLA memory before loading, at
   peak, and after the run), and startup (model load and time to first
   inference). A metric the browser does not recognize appears under **Other
   measurements**, so a new metric shows up without a browser change.
3. **Accuracy.** The status, the first four metrics, and the evaluated sample
   count. **Evaluation details** explains the status and coverage and lists
   the remaining metrics, each with its published score when `model.yaml`
   records one.
4. **Details.** The package with its size, checksum, and download buttons,
   then rows for build provenance, source models, and each component's files,
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
