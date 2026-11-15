# Write the model README

Every model folder holds a `README.md` and a `hero.png`. Follow this page when you add a
model, or when a change alters what a model detects, its input, its outputs, or its
components. At the end, both files pass the check in step 7.

The model page shows what it reads from `model.yaml` and from the published results:
metrics, build profiles, components, license, and source links. Below that it shows the
README. The README therefore holds what an application developer needs and the page
cannot show: what the model does, how to prepare its input, and how to read its outputs.

Two checks apply to each rule on this page. "The validator" is
`model-registry validate-models`, which rejects a README or hero that breaks a rule.
"A reviewer" is the person who reviews your pull request. The examples use
[`models/yolo11m`](../models/yolo11m/model.yaml).

## Before you start

| You need | Check it |
| --- | --- |
| `model-registry` installed, as [Qualify locally](local-development.md) describes. | `model-registry --version` |
| A model folder with a valid `model.yaml`. | `model-registry validate models/yolo11m` prints `Valid model: yolo11m`. |
| The model's display name, which the README repeats. | `grep display_name models/yolo11m/model.yaml` prints `display_name: YOLO11m-det`. |

## Steps

### 1. Start from the template

Create `README.md` in the model folder from this template:

````markdown
# <display_name>

![<display_name> model hero](hero.png)

## Overview

<50 to 120 words of prose: what the model finds, the input it takes, what it
returns, and the components it runs as.>

## Inputs and outputs

**Input.** <Each preparation step with its exact values: resize, padding,
color order, scaling, and normalization.>

**Between the components.** <Only for a model with several components: what
the application computes on the CPU and which tensors it passes on.>

**Outputs.** <Each actual output tensor by name, and what it holds.>

1. <Each decode step, in order, one line per step.>

<The class-index mapping, and the Neat decoder for these outputs, if one exists.>

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
````

A README of the same kind is the fastest start:
[`yolo26n`](https://github.com/sima-neat/models/blob/develop/models/yolo26n/README.md)
for a detector with one component,
[`rf_detr_nano`](https://github.com/sima-neat/models/blob/develop/models/rf_detr_nano/README.md)
for a CPU step between two components, and
[`resnet_50`](https://github.com/sima-neat/models/blob/develop/models/resnet_50/README.md)
for a classifier.

The validator checks the frame of the file:

- Line 1 is `# ` followed by the exact `display_name` from `model.yaml`, such as `# YOLO11m-det`.
- Line 3 is the hero line, `![YOLO11m-det model hero](hero.png)` for this model, with a blank line before and after it.
- Line 5 is `## Overview`.
- The H2 headings are `Overview`, `Inputs and outputs`, and `Building applications`, in this order.
- The file has no other heading of any level and no HTML block.
- The file is UTF-8 text, ends with a newline, and is no larger than 128 KiB.
- `README.md` is a regular file, and no part of its path is a symbolic link.

### 2. Write the overview

Say what the model finds, what input it takes, what it returns, and which components it
runs as. A reviewer checks that the section answers these four points. The Model Browser
shows this section as the model's summary and searches its text.

The validator checks its form:

- The section has one or two paragraphs, with 50 to 120 words in total.
- It holds prose only: no list, table, code block, image, quote, or HTML.

### 3. Write the inputs and outputs

Describe what the application does with the model. A reviewer checks each point:

- Give each input preparation step with its exact values, such as the resize, the padding, and the scaling.
- For a model with several components, give each step the application runs between them.
- Name each actual output tensor and say what it holds.
- Write the decode steps as a numbered list, in the order the application performs them.
- Give the class-index mapping.
- Name the Neat decoder that handles the outputs when one exists, as the `yolo11m` README names `YoloV8`.
- Write each list item on one line, because the model page ends a list item at its line break.
- Take each value from `model.yaml` and the current scripts.

`python -m pytest tests/test_documentation.py` checks one more rule. A `bash`, `sh`, or
`shell` code block holds no placeholder such as `<path>`, because a shell reads it as a
redirection.

### 4. Keep the building applications text

The validator checks that the last section holds exactly this text. The line breaks are
free.

```markdown
Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
```

A language model, whose `model.yaml` sets `compilation.format: huggingface`, uses this
text instead:

```markdown
Run the package with the [Neat GenAI Model API](https://developer.sima.ai/software/develop-apps/development-workflow/genai-model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
```

### 5. Use plain words and leave out what the model page shows

The validator checks two limits on the whole file:

- The README has at most 350 words from `## Overview` to its end.
- No line uses one of the terms in this table.

| Kind of term | Rejected terms |
| --- | --- |
| Internal hardware and tool terms | MPK, A65, MLA, AFE, ELF, tessellation |
| Metric and tool names | mAP, AP50, FPS, latency, benchmark, sima-cli, model-registry, Vulcan |

The validator also rejects the plural of an acronym, such as MPKs. It rejects the
lowercase terms in any capitalization and at the start of a longer word, such as
Benchmarked.

A reviewer checks the rest:

- Name hardware and tools in plain words, such as "the CPU" and "the package".
- Leave out scores and metrics, benchmark settings, build profiles and compile settings, and versions.
- Leave out the license and redistribution terms, checksums, and download commands.
- Leave out links to the original model or its source repository.
- Use the same wording for every size in a model family, so the READMEs differ only in values.
- Keep explanations for maintainers out of the README.

Put an explanation for maintainers in a `model.yaml` comment, or in the docstring of
`convert.py`, `performance.py`, or `accuracy.py`. For an ONNX model, every other `.py`
and `.sh` file under `scripts/`, except `fp32_baseline_accuracy.py`, is a compile input,
so a comment there recompiles the model. An edit to `convert.py` reruns the conversion,
and an edit to `performance.py` or `accuracy.py` reruns that benchmark. A procedure that
applies to every model belongs in `docs/`.

### 6. Add the hero

Create `hero.png` in the model folder with the [illustration guide](illustration-guide.md),
which gives the styles, the palette, the typography, and the export commands.

The validator checks the file:

- `hero.png` exists in the model folder as a regular file, and no part of its path is a symbolic link.
- It is a PNG of exactly 1200x480 pixels.
- It is no larger than 2 MiB.
- It holds only the chunks `IHDR`, `PLTE`, `IDAT`, and `IEND`, plus at most one valid `tIME` and one valid `tRNS` chunk.
- Its checksums are valid and its image data decodes to the full image.

Any other chunk fails the check, including text metadata and color profiles. Export the
PNG without metadata, as the guide's export commands do.

A reviewer checks the picture:

- It shows the model's output on an input, such as labeled boxes on a scene.
- Its scene appears on no other model's hero.
- It shares its style and accent color with the other models of the same architecture.
- Its labels have the same size as the labels on every other hero.
- It stays legible in the light and the dark theme.
- It carries no instruction a reader needs. Instructions belong in the README text.
- It is finished artwork, not a placeholder.
- It is the `hero.png` file in the model folder, not a remote image URL and not an image from another folder.

### 7. Check the README and the hero

```bash
model-registry validate-models --repository .
```

The command checks the README and hero of every model. It prints `Valid models:` and the
number of models, or one error that names the file and the broken rule.
`model-registry validate models/yolo11m` does not read the README or the hero.

Then run the README tests, from a Python environment with this repository's `dev` extra
installed:

```bash
python -m pytest tests/test_documentation.py
```

The pull request checks run these tests, which apply the validator to every model.

Update the README and the hero in the same change that alters what they describe. A
change to only these two files reruns no stage, because no fingerprint includes them.

## If something fails

Each error ends with the path of the file that failed.

| What you see | Cause | Fix |
| --- | --- | --- |
| `model README H1 must exactly match display_name` | Line 1 differs from `display_name` in `model.yaml`. | Copy the display name into the H1. |
| `model README must begin with H1, blank line, ...` | The hero line or `## Overview` is missing, changed, or on another line. | Restore lines 1 to 5 of the template. |
| `model README H2 sections must be Overview, Inputs and outputs, Building applications in order` | A section is missing or renamed, or the file has another heading or an HTML block. | Keep the three H2 headings and replace other headings with bold lead-ins. |
| `model README Overview must contain 50 to 120 words, found ...` | The overview is too short or too long. | Shorten or extend the overview. |
| `model README Overview must contain one or two prose paragraphs` | The overview has more than two paragraphs. | Merge the paragraphs into one or two. |
| `model README Overview must contain prose paragraphs only` | The overview holds a list, table, code, image, quote, or HTML. | Move that content to `Inputs and outputs`. |
| `model README Building applications must be exactly ...` | The last section differs from the fixed text. | Paste the text from step 4. |
| `model README line ... uses 'MPKs'; write it in plain words` | The line holds a rejected term. | Use plain words, such as "the package". |
| `model README must contain at most 350 words after the hero, found ...` | The README is too long. | Cut what the model page already shows. |
| `model README must end with a newline` | The last line has no line ending. | Add a newline at the end of the file. |
| `model hero image is required` | The model folder has no `hero.png`. | Add the hero. |
| `model hero image must be 1200x480, found ...` | The PNG has another size. | Export the hero at 1200x480. |
| `model hero image must contain between 1 and 2097152 bytes` | The file is empty or larger than 2 MiB. | Export it as an indexed-palette PNG, as the illustration guide's export command does. |
| `model hero image contains an unsupported PNG ancillary chunk` | The PNG carries metadata, such as text or a color profile. | Export the PNG without metadata. |

## Next

Return to [Define the model contract](adding-a-model.md) to validate the whole model
folder.
