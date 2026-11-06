# Model browser design

## Users

Model engineers and application developers use the Model Registry to find a
suitable verified model quickly, compare completed builds, inspect provenance,
and download deployable artifacts. Search and model compatibility are the
primary browsing tasks; operational publication state is not part of the
public discovery experience.

## Brand Personality

Calm, precise, and technical. The interface should create confidence through
clear hierarchy, restrained visual language, and trustworthy metadata rather
than decoration or dashboard density.

## Aesthetic Direction

The shared Developer Portal shell supplies global navigation, search, language
selection and theme switching. Models mounts it with Models active. Browse models,
Compare and Developer docs sit in a separate section navigation row underneath.
The section row follows Software documentation: full-width and left-aligned,
with bold links and a blue active label.
Standalone previews keep a local header when the shared shell is unavailable.

The build selects the audience: Main uses a customer view of its catalog;
Develop and feature branches keep the developer view, with branch/compiler filters,
build history and contributor navigation. Both default to the staging registry
and publish to the same staging site; `?env=prod` explicitly previews production
data with the same staged UI and a Production registry label. Selecting a catalog branch changes data,
never the audience of the built site. `?branches=0` hides only the branch selector
and keeps the selected catalog on Main without changing the audience.

Refined minimalism with generous space and progressive disclosure. Light and
dark themes remain supported. Search is the dominant action; Task and Model family
are the catalog filters. Developer view also exposes Branch, Compiler and Build history.
Avoid crowded toolbars, repeated status
indicators, and decorative effects that compete with model information.

The catalog opens as a grid of model cards because people recognize a model by what
it does before they know its name. A Cards/Table switch also offers a compact
table for scanning model names, tasks, precisions, accuracy, and performance.
Both views use the same filters; the table groups models by task and keeps the
model-family ordering within each group. Multi-task models appear in each applicable
group, while the total counts unique models. It names the selected precision and
shows one whole-model measurement per variant. Each card carries one thing per role: the
model's hero illustration, its name, a one-line description, and its tasks. A
card states what the model is, not how a build measured; numbers, precision,
and build state belong on the model page, where they have context. Card
borders and fills mark the cards as the objects to choose; the model page
itself uses sections and rules instead of nested cards.

The catalog follows the developer examples portal: blue task labels, neutral
model names and section headings, and quiet card surfaces in both themes.
Small metadata chips use pale blue for tasks, orange for precision and purple
for model type. Other metadata stays neutral. Additional palette colors identify
chart selections or carry a specific status; tasks do not have individual colors.
Filter dropdowns stay neutral when selected, with a neutral outline for keyboard focus only.
The selected precision tab uses a soft blue fill and border while keeping its
text neutral and readable in both themes.

Compare starts with a task, then shows measured vision variants
in one scatter plot: accuracy vertically, throughput horizontally. The dataset is
shown below the chart title, with a Dataset & metric selector when multiple groups
are available. A point popup identifies its model, variant and measurements. Muted points become colored when selected; a
compact table below aligns up to three selections. A compact Add model selector supports keyboard and mobile selection, including
variants with missing measurements. The graph legend includes plotted selections only;
variants without matching measurements remain in the table. Selection colors use the shared SiMa blue,
green and orange palette. GenAI comparisons plot all variants by parameters
(in billions) versus decode speed. Text Generation uses one selected text input
length, including text-only measurements from VLMs. Visual Question Answering uses
each model's image-plus-text benchmark at its encoder resolution, with no workload selector.

The model page reads at a glance first and expands on request.
The header pairs its title, description and Download action with a compact model
image on the right; narrow screens stack the content and image. The image appears
once, in the header. Performance and accuracy keep separate stacked sections, each
measuring the whole model once per build profile.
Validation and source-link icons sit beside Download; CLI commands live inside
the download dialog.
Each section shows its key facts directly; longer detail sits behind one kind of control: a
full-width row between thin rules, the label on the left, an optional short
summary, and a › on the right that turns down when the row opens, like a
settings list. No other expand control exists, so anything that opens looks
and behaves the same. The one exception is the execution graph: a Details
button beside the diagram opens the tensor table under the graph, so
the details stay attached to the graph they describe.

Actions and links look different. An action (download, copy) is a
button with an icon. Blue text ending in ↗ is a link that leaves the browser.

## Design Principles

1. Make finding a model the single primary action.
2. Show only completed, usable builds in public discovery.
3. Keep filters focused on model discovery, with developer controls visible only
   in developer view. Make active filters obvious and easy to clear.
4. Search across names and sanctioned category labels using plain language.
5. Preserve accessible labels, keyboard operation, responsive behavior, and
   reduced-motion support while simplifying the interface.
6. Show the answer before the evidence: key numbers first, then one row to
   open for the rest.
