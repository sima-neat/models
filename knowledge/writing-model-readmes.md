# Writing model READMEs

The writing standard for `models/<model_id>/README.md`: who reads it, what each part must
answer, and how to check it. Read it before you write or review a model README. The
required frame, the limits, and the validator errors are in
[the model README page](../docs/model-readme.md), and this file does not repeat them.

## Who reads a model README

An application developer who opens a model page to decide two things:

1. Does this model do what my application needs?
2. What do I have to write to use it?

This reader may never have heard of the model. They know what a camera frame and a
tensor are. They do not know the model's architecture, its training, or this
repository's scripts, and they do not need to. The model page links the original model
page for anyone who wants that depth, so the README does not retell it.

Three words decide every choice below: brief, consistent, precise. Clearer never means
more text. A README gets clearer by putting one thing in one place, in the same place on
every model, with its exact value.

## What went wrong before

The first READMEs were accurate but written from the model's side. Reviewers of the model
pages found them hard to understand, for these reasons:

| Problem | Example |
|---|---|
| The overview opened with who made the model and which size it is. | "YOLO26n-det is the Nano size of the Ultralytics YOLO26 object detector" |
| Most overviews named no use. | Every classifier and detector said what it returns, and not what to build with it. |
| The overview repeated the next section. | "returns raw box and class outputs at three scales" |
| Inputs and outputs mixed several things. | One paragraph held the input, its preparation, script details, and Neat options. |
| Tensors came before the result. | The reader met `bbox_0` and strides before "a box, a class, and a score". |
| Maintainer notes leaked in. | "The reference scripts pass this image", how a speed was measured, internal precision. |
| Two ways to run the model were mixed into every section. | "With Neat: ..." and "Yourself: ..." items, followed by advice on which to prefer. |
| Headings were inconsistent. | A section "Inputs and outputs" with bold lead-ins "Input.", "Image input.", and "Outputs." under it. |
| Sentences told the reader nothing. | "The larger sizes are more accurate and slower.", "It comes from Roboflow." |

## The order of every README

Each part answers one question, in the order the reader asks them.

| Part | Question it answers |
|---|---|
| Overview, sentence 1 | What does the model do? |
| Overview, next | Where would I use it? |
| Overview, next | What do I get back? |
| Overview, last | What would surprise me? |
| `## Inputs`, first sentence or table | What does the model take? |
| `Preprocessing:` steps | How do I prepare it? |
| `## Outputs`, table | What does the model return? |
| `Postprocessing:` steps | How do I turn that into a result? |

The section names and these four parts are the same on every model page, so a reader
finds the same structure everywhere. The sections are always `## Inputs` and `## Outputs`,
in the plural, also for a model with one input or one output. There is no
"Inputs and outputs" section above them, and no bold lead-in stands in for a heading. A part that has nothing
to say is left out: an embedding model whose output needs no decoding has no
postprocessing list.

## Overview

Open with the task in words a non-specialist uses. Then name two or three applications.

| Kind of model | Opening | Uses to name |
|---|---|---|
| Classifier | "sorts an image into one of the 1,000 ImageNet classes" | tagging, sorting, a base for a custom classifier |
| Detector | "finds everyday objects in an image and returns a box, a class, and a score for each" | counting, tracking, monitoring |
| Segmentation, pose, oriented boxes | what it adds to a box: a pixel mask, body keypoints, a rotated box | cutting objects out, activity analysis, aerial images |
| Embedding | "turns an image into embeddings, lists of numbers that describe what it shows" | search, grouping, a base for a small trained head |
| Depth and stereo | "estimates how far away each pixel is" | obstacle avoidance, 3D measurement |
| Keypoints and matching | "finds the same scene points in two images" | visual odometry, mapping, image alignment |
| Language and vision-language | "answers questions and writes text", "answers questions about an image" | assistants, summaries, image description |
| Speech | "turns speech into text", "speaks text" | voice interfaces, dictation |

Rules for the rest of the overview:

- Give the result as the application sees it, not as tensors.
- Do not say who made the model. The model page links the original model and shows its license.
- Do not restate what the name already says: the size, the family, or the version.
- Do not compare sizes. "The larger sizes are more accurate and slower" is true of every family, so it tells the reader nothing, and it reads wrongly on the largest size.
- Before keeping a sentence, ask what the reader would lose without it. Remove it when the answer is nothing.
- Name a surprise when there is one: the model needs another model first, its result is relative and not metric, it is trained on a fixed class list (write "It is trained on the 80 COCO classes", not "It knows"), it handles a fixed number of items.
- Leave out the architecture, the training method, and the number of components, unless the application has to act on them. A model with a CPU step between two components says so, because the application writes that step.
- Do not make a speed or accuracy claim. The model page shows the measured values.
- The validator wants at least 50 words. An overview that falls short after these cuts needs a use, a result, or a limit the reader lacks, never a filler sentence.

## Inputs and Outputs sections

These two sections describe the model alone: what it takes, how to prepare it, what it
returns, and how to decode it. They are brief, precise, and the same shape on every model.

- Do not name Neat, its options, or its decoders here. `Building applications` links the Neat API, and the first version of this standard was rejected for mixing a "With Neat" path and a "Yourself" path into every section.
- Describe one path: the preprocessing of the original model, which `calibration.py` and `accuracy.py` implement and the published accuracy uses. When the two scripts differ, follow `accuracy.py`, as `siglip2_vit_b_16` does with its 256×256 resize and 224×224 crop.
- State steps, not advice. Leave out which path to prefer, why a step exists, and what happens otherwise, unless the application must act on it.
- Never mention the scripts, how a result was measured, or how the model was converted or compiled.
- Keep the exact values: sizes, means, thresholds, counts. Clarity comes from order and layout, never from dropping a value the application needs.
- Write sizes one way: `640×640×3 (height, width, channel)` for an input and plain counts in a table, such as "196 rows of 768 values".
- One step per line, in the order the application performs it. A step that needs a formula gives the formula.
- Write each step so that it works for every input, not only the usual one. Do not tell the reader to cut a token list to the input length, because that drops the end token of a long prompt. Say to set the tokenizer's truncation instead, as `clip_vit_b_16` does.
- Do not squeeze steps to save words. "Repeat step 5's drop across windows with 0.3" is shorter than the full rule and useless to someone implementing it. The limit is 400 words so that dense models can state each step in full.
- A language or speech model has no tensors to show. It keeps the two sections and gives one short list under each: what to supply, and what comes back.

## Keeping a family consistent

Write one README of a family first, check it, then derive the other sizes from it by
changing values only: the name, the input size, the grid sizes, the counts.
Compare two sizes with `diff` before a pull request. Any difference that is not a value is
a mistake in one of them.

## Checks before a README merges

1. Check every value against `model.yaml` and the current scripts.
2. Read only the overview and answer: what does it do, where would I use it, what do I get? If one answer is missing, rewrite it.
3. Follow the preprocessing and postprocessing lists as if implementing them. Fix each step where you would have to guess.
4. Search the README for "Neat", "script", "reference", and "measured". Only `Building applications` may hold one of them.
5. Look at the model page in the local preview, `artifact-browser/preview.py`, before changing the format of many READMEs.
6. Run `model-registry validate-models --repository .` and `python -m pytest tests/test_documentation.py`.

## Examples

| Kind | README |
|---|---|
| Detector | [`yolo26n`](../models/yolo26n/README.md) |
| Classifier | [`resnet_50`](../models/resnet_50/README.md) |
| CPU step between two components | [`rf_detr_nano`](../models/rf_detr_nano/README.md) |
| Outputs that need no decoding | [`dinov3_vit_b_16`](../models/dinov3_vit_b_16/README.md) |
| Two inputs with their own preprocessing | [`clip_vit_b_16`](../models/clip_vit_b_16/README.md) |
| Language model | [`qwen3_4b`](../models/qwen3_4b/README.md) |
