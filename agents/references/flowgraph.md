# Drawing a model's flowgraph

A pull request description draws the flowgraph of each model it changes, as
[`models-write-pr-description`](../skills/models-write-pr-description/SKILL.md) says, and a review compares
it with the model's files. Every model gets one. The graph answers one question, written as its
first line: "How does an input become a result in `<model_id>`?". It shows the path one input
takes through the package, at the altitude of `pipeline` in `model.yaml`, never the operators
inside a component.

## Parts

- One box per model component, named by its ID in `compilation.components`, or `model` for a
  single-component model.
- One box per host step between components, named as its `cpu` entry in `pipeline` names it,
  such as `Top 300 proposals`.
- One box for the Neat decoder when the README's Postprocessing names one, such as
  `BoxDecode YoloV26`.
- One input box first and one result box last, named from `modalities` in `model.yaml`. A
  single-component model is a short graph: input, `model`, the decoder if any, and the result.
- Label every arrow with what moves: tensor names, shapes, and format, such as
  `proposals: 300×4` or `640×640×3 RGB, 0 to 1`. Names and shapes come from `model.yaml` and
  the README's Inputs, Between the components, and Outputs sections. Leave out a shape that
  neither states. Write a shape as `H×W×C` or `rows×values`, in the README's order.
- Draw a boundary, `╭┄╮ ┆ ╰┄╯`, labelled `MLA`, around the compiled components. The input, host
  steps, the Neat decoder, and the result sit outside it, so a host step between two
  components splits the boundary in two.
- A tensor that passes from one component to a later one without a host step gets its own line
  beside the host step, such as RF-DETR's `backbone_features`.
- Put a change marker after a box name or an arrow label: `(new)`, `(changed)`, or `(removed)`.
  A changed shape, name, or format marks its arrow. A changed recipe, surgery, or precision marks
  its component.

## Characters and layout

- Put the graph in a fenced `text` code block, at most 100 columns wide, running top to bottom
  from the input.
- Boxes use `┌ ┐ └ ┘ ─ │`, lines use `─ │`, joins use `┬ ┴ ├ ┤ ┼`, and arrowheads `► ◄ ▼ ▲`
  touch the box they point into. Use spaces only, no tabs, emoji, or ASCII `+-|` boxes.
- Draw a graph of more than four boxes with a short script that places boxes by row and column.
  The graph is done when every line of each box has the same width and no line passes 100
  columns.

Under the graph, write one line per part or path worth the reviewer's attention, using the
graph's names.

## Examples

[`models/yolo26n`](../../models/yolo26n) has one component and Neat's box decoder:

```text
How does an input become a result in yolo26n?

   ┌───────────────────────────┐
   │ RGB image                 │
   └───────────────────────────┘
                 │ letterboxed to 640×640×3 RGB, 0 to 1
╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ MLA            ▼                                            ┆
┆  ┌───────────────────────────┐                              ┆
┆  │ model                     │                              ┆
┆  └───────────────────────────┘                              ┆
╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                 │ bbox_0 to bbox_2: 80×80, 40×40, 20×20 cells × 4 distances
                 │ class_logit_0 to class_logit_2: the same cells × 80 logits
                 ▼
   ┌───────────────────────────┐
   │ BoxDecode YoloV26         │
   └───────────────────────────┘
                 │ boxes, classes, scores
                 ▼
   ┌───────────────────────────┐
   │ Boxes and class scores    │
   └───────────────────────────┘
```

- `model` ends at the raw one-to-one heads, at strides 8, 16, and 32. The surgery cut the
  in-graph decode and TopK.
- `BoxDecode YoloV26` runs the README's Postprocessing steps for the application.

[`models/rf_detr_nano`](../../models/rf_detr_nano) has two components and one host step:

```text
How does an input become a result in rf_detr_nano?

   ┌───────────────────────────┐
   │ RGB image                 │
   └───────────────────────────┘
                 │ stretched to 384×384×3 RGB, ImageNet mean and standard deviation
╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ MLA            ▼                                                      ┆
┆  ┌───────────────────────────┐                                        ┆
┆  │ backbone                  │────────────────┐                       ┆
┆  └───────────────────────────┘                │                       ┆
╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                 │ proposal_scores: 576         │ backbone_features
                 │ proposal_boxes: 576×4        │
                 ▼                              │
   ┌───────────────────────────┐                │
   │ Top 300 proposals         │                │
   └───────────────────────────┘                │
                 │ proposals: 300×4             │
╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ MLA            ▼                              ▼                       ┆
┆  ┌─────────────────────────────────────────────────┐                  ┆
┆  │ decoder                                         │                  ┆
┆  └─────────────────────────────────────────────────┘                  ┆
╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                 │ dets: 300×4 boxes as (cx, cy, w, h), fractions of the image
                 │ labels: 300×91 class logits
                 ▼
   ┌───────────────────────────┐
   │ Boxes and class scores    │
   └───────────────────────────┘
```

- `Top 300 proposals` keeps the 300 highest `proposal_scores`, ties to the lower index, and
  gathers those rows of `proposal_boxes`.
- `backbone_features` reaches the decoder without a host step.
- No Neat decoder runs. The application turns `dets` and `labels` into boxes with the README's
  Postprocessing.
