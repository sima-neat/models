---
name: models-write-pr-description
description: Write or update the description of a sima-neat/models pull request, after writing what the work taught into agents/, skills/, or docs/. Use for every pull request, model or not, when opening it, after a push that changes what the description says, and after CI publishes the results its table reports.
---

# Write a Models pull request description

A person reads it top to bottom in a minute and knows what changed, what it measured, what
reran, and what nobody checked. Write for that reader:

- Open with the change in one or two plain sentences, before any block or table.
- Report what changed. For a change to an existing model, a field line that did not change is
  left out. A new model gets every line.
- Use the words of the model's README. Give a number with its unit, and a link with a short
  label where a bare URL or hash would run long.
- Keep each line to one fact a reviewer acts on. Commands, file hashes, and tool versions
  stay out, because the repository already pins them.
- Leave out how the work went, what was tried, and anything the diff already shows.

## Steps

1. Read the linked issue, every commit, and the full diff. List each changed model and each
   changed file outside `models/`. A family that shares one recipe is one block.
2. List each lesson the work taught: a statement in `AGENTS.md`, `agents/`, `skills/`, or
   `docs/` that the change proves wrong, and a fact, rule, or co-change it established that no
   file holds, including what a review comment or a failed check showed. Write each one into
   its file as [`rules.md` "Where a lesson goes"](../../rules.md#where-a-lesson-goes) says, and
   commit it with the change. The step is done when each lesson is in its file and named on
   the `Rule, skill, and knowledge updates` line, or the line says `none` with the reason the
   work taught nothing new.
3. Write the Summary, then each model's block from the sources under [Fields](#fields), and
   draw its graph with [`flowgraph.md`](../../references/flowgraph.md), with change markers.
4. First pass, when the pull request opens: write each stage cell as `running`, `not run`, or
   `reused`, and leave out measured values.
5. Second pass, after CI publishes: fill each cell from the published results and name the
   commit they came from. Repeat after each push that changes a result.
6. Check every claim against a file or a run. Remove what you cannot check, or move it under
   `Not verified`.
7. Read the draft as the reviewer: cut each line that tells them nothing they would act on.
8. Publish with `gh pr edit <number> --body-file <file>`, then read the published body back.

The description is done when each lesson is in its file, each changed model has its block,
each published stage has its cell, and each stage nobody ran is under `Not verified`.

## Model changes

````markdown
## Summary
<What changes and why, in one or two sentences.>

## <model_id>
- **Source:** [<repository>@<short revision>](<repository>/tree/<revision>)
- **License:** <spdx>, redistribution <allowed, restricted, prohibited, or unknown>, <downloadable or not available for download>. <Which license file ships in the package and where it is pinned, or why none ships.>
- **Surgery:** <each rewrite and its purpose, or none>
- **Quantization:** <weights, activations, and calibration, per build profile when they differ>
- **Builds:** every push | on demand

```text
<flowgraph drawn with agents/references/flowgraph.md, with change markers>
```

| Build | Compile | Performance | Accuracy | Reference | Compile time | Benchmark time | Runner |
|---|---|---|---|---|---|---|---|
| <profile> | passed | passed, 32.1 fps (+0.4%) | passed, coco_map 0.442 (-0.001) | 0.451 published, 0.447 FP32 | 41 min | 12 min + 38 min | expensive, spot |

Results from commit <sha>.

## Other changes
- <One line per changed file outside the model folders: the file and why it changes.>

## Reruns
<Which stages rerun for which models, and the input that causes each.>

## Not verified
<What did not run, and why.>

Tested locally: <the stages that passed, and any that failed or did not run>

Rule, skill, and knowledge updates: <file and lesson, or none with the reason>
Refs #<issue>
````

Leave out `Other changes` when the pull request touches only model folders. Add a `Links:` line above `Refs` only when the change has a conversion pull request or a
Dataset Registry change. Input hashes, tool versions, and the commands that ran stay out:
`model.yaml` and `deps/manifest.json` pin the first two, and the runner scripts fix the third.

`Results from commit` names one commit per group of builds when the builds came from different
commits.

A family that shares one recipe gets one block headed by every model ID, and one graph whose
question names every model ID. A new model's component boxes carry `(new)`. Its table has one row
per `<model_id>/<profile>`, and a field that differs per size gives each size's value on its
line.

## Fields

| Field | Source |
|---|---|
| Source | `upstream.repository` and `upstream.revision` in `model.yaml`. When `upstream` is absent, the file-set source's `repository` as `https://huggingface.co/<repository>` and its `revision` ([product identity](../../../docs/metadata-schema.md#product-identity-and-provenance)) |
| License | `license.spdx` and `license.redistribution` in `model.yaml`, and the `downloadable` value of that ID in `artifact-browser/licenses.json`. How it is handled: each `license` resource under `resources`, with the repository and commit its `uri` pins and its `install_path` under `licenses/`. Say so when the weights carry a second license, when the text is pinned from another repository, when the change adds the license to `licenses.json`, and when no file ships because the license forbids redistribution ([license](../../../docs/metadata-schema.md#license), [license files in the package](../../../docs/metadata-schema.md#license-files-in-the-package)) |
| Reference | the `reference_metrics` values for the metric and dataset the Accuracy cell reports: the `published` score of the source model's authors and the `fp32_baseline` score on the package's own inputs, each labelled ([reference metrics](../../../docs/metadata-schema.md#reference-metrics)). Give both when the model declares both. Write `none` when it declares neither, and say so under `Not verified`, because the accuracy then has nothing to be judged against |
| Other changes | each changed file outside `models/`, from the diff |
| Surgery | the rewrites `prepare_model` in `surgery.py` runs, each with the purpose its docstring or comment states |
| Quantization | each component's `parameters` in the build profile and the `QuantizationParams` that `compile.py` builds from them, with the profile's `calibration_dataset` |
| Builds | `build-on-demand` in `model.yaml`: `true` is `on demand`, absent or `false` is `every push` |
| Compile time | **Compile recipe duration** in the build's Details in the [model browser](../../../docs/model-browser.md)'s developer view. It shows the compile run's `compile_timing`, which `scripts/run_model_compilation.sh` records around the recipe alone ([duration reporting](../../../artifact-browser/README.md#duration-reporting)). Components compiled together share one value. |
| Runner | the build's Compile job in the Actions tab: its runner labels name the class, and its **Compile runner** summary names the requested, selected, and actual market. The package manifest's `provenance` keeps the same fields ([verify the result](../../../docs/runner-class-requests.md#verify-the-result-and-diagnose-failures)). Write `<class>, <actual market>`, and add the requested one in parentheses when it differs. |
| Performance | the published `throughput_fps` in the model browser's developer view with the branch selected, or one value per component for `scope: components` |
| Accuracy | the published value of the first metric `datasets/<id>.yaml` requires for the model's task, on its first accuracy dataset, with the digits `tools/model_report/metrics.json` gives |
| Benchmark time | the durations of the `Measure performance on DevKit` and `Measure accuracy on DevKit` steps of the build's `Benchmark on DevKit` job, as `<performance> + <accuracy>`. No registry run records them. The job's own duration adds the DevKit setup: tooling, the build's download from the registry, the dataset unpacked for the job, Core, and MoLE for a model it measures ([benchmark workflow](../../references/ci-flow.md#benchmark-workflow)). |
| Reruns | [`rules.md` "Stages and fingerprints"](../../rules.md#stages-and-fingerprints) and [what a change reruns](../../../docs/adding-a-model.md#9-know-what-a-change-reruns), per model and build profile, with the input that causes each. When a shared file reruns models the pull request is not about, the line opens with that: the file, the stages, and all models or how many ([`rules.md` "Before a push"](../../rules.md#before-a-push)) |
| Tested locally | the stages run before the push and the result of each: compile, `verify-artifact`, the compile tests, performance, and accuracy |
| Links | the pull request that published the conversion when that was a separate one, and the Dataset Registry change when the change adds data |

[CI flow](../../references/ci-flow.md) and [publication](../../references/publication.md) say
which job publishes each result and what a run records.

The existing description is a claim to check, never a source. A value taken from it is checked
against a file or a run, or goes under `Not verified`. When the Actions runs of a build can no
longer be read, take the runner from the package manifest's `provenance`, and write `unknown`
for what no source gives.

## Cells

- A stage cell reads `passed`, `failed`, `skipped`, `running`, `not run`, `reused`, or `unknown`.
  `not run` states that the stage did not run. A stage whose state no source shows reads
  `unknown`, and `Not verified` says what could not be read.
  `skipped` means the stage ran and published its result, but `tests/tests.yaml` sets
  `run_tests: false` for it, so no stage test checked the result (`TEST_OUTCOMES` in
  `tools/model_registry/model_tests.py`). `reused` means the build came from the base branch
  and this change did not rerun it ([the prior
  catalog](../../references/branch-catalogs.md#the-prior-catalog)). A `failed` cell names the
  failing job or test: `failed, compile test`.
- The time cell of a stage that is `reused`, `running`, or `not run` repeats that word. A stage
  the build does not declare reads `none`.
- An accuracy value means little alone. The Reference cell beside it gives what the unquantized
  model scores, so the reader sees the cost of quantization at a glance.
- A measured value sits beside its change from the base branch's published value: in percent
  for throughput, in the metric's own unit for accuracy. Write `new` when the base branch has
  no value.
- Only published values enter the table. A local DevKit measurement goes on its own line under
  the table, starting `Local:`, with the DevKit type, the MLA clock, and the commit it ran, in the digits
  the table uses. A local value without its clock and commit goes under `Not verified`.
- A model with `build-on-demand: true` keeps `not run` until someone requests its build and its
  benchmark ([on-demand builds](../../../docs/adding-a-model.md#on-demand-builds)).

## Shared tooling or docs

A pull request that changes no model folder draws no graph:

```markdown
## Summary
<What changes, for whom, in one or two lines.>

## What changed
- <One line per change, naming the file, command, or workflow it touches.>

## Reruns
<Stages and models, or none, with the rerun-table row that says so.>

## Tests
| Test | Checks |
|---|---|
| `<file>` | <the behavior it asserts> |

## Not verified
<What did not run, and why.>

Rule, skill, and knowledge updates: <file and lesson, or none with the reason>
Refs #<issue>
```

## Example

A change that raises `calibration_samples` of the `w8a8` profile. Only the changed field line
appears, because the model already exists. The numbers are illustrative.

````markdown
## Summary
Doubles the calibration images of the yolo26n `w8a8` build from 10 to 20, which raises its COCO mAP by 0.002.

## yolo26n
- **Quantization:** `w8a8` now calibrates on 20 coco-2017-cal images (was 10). `w8a16` and `w16a16` are unchanged.

```text
How does an input become a result in yolo26n?

   ┌───────────────────────────┐
   │ RGB image                 │
   └───────────────────────────┘
                 │ letterboxed to 640×640×3 RGB, 0 to 1
╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ MLA            ▼                                            ┆
┆  ┌───────────────────────────┐                              ┆
┆  │ model (changed)           │                              ┆
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

| Build | Compile | Performance | Accuracy | Reference | Compile time | Benchmark time | Runner |
|---|---|---|---|---|---|---|---|
| w8a8 | passed | skipped, 966.4 fps (+0.2%) | passed, coco_map 0.383 (+0.002) | 0.409 published | 19 min | 6 min + 24 min | default, spot |
| w8a16 | reused | reused, 610.2 fps | reused, coco_map 0.386 | 0.409 published | reused | reused | default, spot |
| w16a16 | reused | reused, 315.5 fps | reused, coco_map 0.403 | 0.409 published | reused | reused | default, spot |

Local: w8a8 on a Modalix DevKit, MLA at 1 GHz, commit 0123abc: 964.9 fps, coco_map 0.383
Results from commit 0123abc.

## Reruns
yolo26n `w8a8` compile, performance, and accuracy, because `calibration_samples` is a compile input. No other build or model reruns.

## Not verified
`w8a8` performance and accuracy on a second DevKit.

Tested locally: compile, artifact verification, compile tests, and accuracy passed for `w8a8`.

Rule, skill, and knowledge updates: none, because the change applies the existing calibration guidance.
Refs #<issue>
````

## Rules

- Name runners by class and DevKit type, never by host name.
- Each list item, field line, and table row stays on one source line.
- Leave out AI attribution, internal host names, private links, board addresses, and customer
  information. The repository is public.
