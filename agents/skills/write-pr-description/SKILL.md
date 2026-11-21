---
name: write-pr-description
description: Write or update the description of a sima-neat/models pull request. Use when opening a pull request, after a push that changes what the description says, and after CI publishes the results its table reports.
---

# Write a Models pull request description

Write it compact: every line carries a fact, and no paragraph narrates. A reviewer who reads
each model's block and the closing lines knows what changed, what reran, what it measured, and
what nobody checked. [Pull request
description](../../../docs/contributing.md#pull-request-description) lists the evidence
reviewers expect, and this format carries it.

## Steps

1. Read the linked issue, every commit, and the full diff. List each changed model and each
   changed file outside `models/`. A family that shares one recipe is one block.
2. Fill each model's block from the sources under [Fields](#fields), and draw its graph with
   [`flowgraph.md`](../../references/flowgraph.md), with change markers.
3. First pass, when the pull request opens: write each stage cell as `running`, `not run`, or
   `reused`, leave out measured values, and name the pushed commit in the commit line.
4. Second pass, after CI publishes: fill each cell from the published results and set the
   commit line to the commit they came from. Repeat after each push that changes a result.
5. Check every claim against a file or a run. Remove what you cannot check, or move it under
   `Not verified`.
6. Publish with `gh pr edit <number> --body-file <file>`, then read the published body back.

The description is done when each changed model has its block, each published stage has its
cell, and each stage nobody ran is under `Not verified`.

## Model changes

One block per changed model:

````markdown
### <model_id>
Upstream: <repository>/tree/<revision> · License: <spdx>, downloadable or not
Surgery: <each rewrite and its purpose, or none>
Quantization: <weights, activations, calibration method and sample count, per build profile when they differ>
Scheduling: every push | on demand

```text
<flowgraph drawn with agents/references/flowgraph.md, with change markers>
```

| Build | Compile | Compile time | Compile runner | Performance | Accuracy | Benchmark time |
|---|---|---|---|---|---|---|
| <profile> | passed | 41 min | expensive, spot | passed, 32.1 fps (+0.4%) | passed, coco_map 0.442 (-0.001) | 12 min + 38 min |
````

Then, once per pull request:

```markdown
Results from commit <sha>.
Reruns: <which stages for which models>
Local qualification: Model Compiler <version>, DevKit platform <version>, Core <version>
Local qualification: <source file or dataset source ID> sha256 <hash>
Local qualification: `<exact command>`: <passed or failed, and its summary>
Links: <conversion pull request or Dataset Registry change, or none with the reason>
Rule, skill, and knowledge updates: <file and lesson, or none with the reason>
Not verified: <what did not run, and why>
Refs #<issue>
```

Each `Local qualification:` and `Links:` line holds one fact. Write one line per source file,
dataset, command, and link.

A family that shares one recipe gets one block headed by every model ID. Its table has one row
per `<model_id>/<profile>`, and a field that differs per size gives each size's value on its
line.

## Fields

| Field | Source |
|---|---|
| Upstream | `upstream.repository` and `upstream.revision` in `model.yaml`. When `upstream` is absent, the file-set source's `repository` as `https://huggingface.co/<repository>` and its `revision` ([product identity](../../../docs/metadata-schema.md#product-identity-and-provenance)) |
| License | `license.spdx` in `model.yaml`, and its `downloadable` value in `artifact-browser/licenses.json` |
| Surgery | the rewrites `prepare_model` in `surgery.py` runs, each with the purpose its docstring or comment states |
| Quantization | each component's `parameters` in the build profile and the `QuantizationParams` that `compile.py` builds from them, with the profile's `calibration_dataset` |
| Scheduling | `build-on-demand` in `model.yaml`: `true` is `on demand`, absent or `false` is `every push` |
| Compile time | **Compile recipe duration** in the build's Details in the [model browser](../../../docs/model-browser.md)'s developer view. It shows the compile run's `compile_timing`, which `scripts/run_model_compilation.sh` records around the recipe alone ([duration reporting](../../../artifact-browser/README.md#duration-reporting)). Components compiled together share one value. |
| Compile runner | the build's Compile job in the Actions tab: its runner labels name the class, and its **Compile runner** summary names the requested, selected, and actual market. The package manifest's `provenance` keeps the same fields ([verify the result](../../../docs/runner-class-requests.md#verify-the-result-and-diagnose-failures)). Write `<class>, <actual market>`, and add the requested one in parentheses when it differs. |
| Performance | the published `throughput_fps` in the model browser's developer view with the branch selected, or one value per component for `scope: components` |
| Accuracy | the published value of the first metric `datasets/<id>.yaml` requires for the model's task, on its first accuracy dataset, with the digits `tools/model_report/metrics.json` gives |
| Benchmark time | the durations of the `Measure performance on DevKit` and `Measure accuracy on DevKit` steps of the build's `Benchmark on DevKit` job, as `<performance> + <accuracy>`. No registry run records them. The job's own duration adds the DevKit setup: tooling, the dataset cache, and Core ([benchmark workflow](../../references/ci-flow.md#benchmark-workflow)). |
| Reruns | [`rules.md` "Stages and fingerprints"](../../rules.md#stages-and-fingerprints) and [what a change reruns](../../../docs/adding-a-model.md#9-know-what-a-change-reruns), per model and build profile, with the input that causes each. When a shared file reruns models the pull request is not about, the line opens with that: the file, the stages, and all models or how many ([`rules.md` "Before a push"](../../rules.md#before-a-push)) |
| Local qualification | the versions the local runs used, which `deps/manifest.json` pins as `model-compiler.version`, `platform-version`, and `neat-core.install_target`. Each source file and dataset the runs read, with its `sha256` in `model.yaml`. Each local command exactly as run, with its result: compile, `verify-artifact`, the compile tests, performance, and accuracy. The compile line also names the fidelity report when `model.yaml` declares `numerical_fidelity`. |
| Links | the conversion pull request when the change converts, and the Dataset Registry change when it adds data |

[CI flow](../../references/ci-flow.md) and [publication](../../references/publication.md) say
which job publishes each result and what a run records.

## Cells

- A stage cell reads `passed`, `failed`, `skipped`, `running`, `not run`, or `reused`.
  `skipped` means the stage ran and published its result, but `tests/tests.yaml` sets
  `run_tests: false` for it, so no stage test checked the result (`TEST_OUTCOMES` in
  `tools/model_registry/model_tests.py`). `reused` means the build came from the base branch
  and this change did not rerun it ([the prior
  catalog](../../references/branch-catalogs.md#the-prior-catalog)). A `failed` cell names the
  failing job or test: `failed, compile test`.
- The time cell of a stage that is `reused`, `running`, or `not run` repeats that word. A stage
  the build does not declare reads `none`.
- A measured value sits beside its change from the base branch's published value: in percent
  for throughput, in the metric's own unit for accuracy. Write `new` when the base branch has
  no value.
- Only published values enter the table. A local DevKit measurement goes on its own line under
  the table, starting `Local:`, with the DevKit type, the MLA clock, and the commit it ran.
- A model with `build-on-demand: true` keeps `not run` until someone requests its build and its
  benchmark ([on-demand builds](../../../docs/adding-a-model.md#on-demand-builds)).

## Shared tooling or docs

A pull request that changes no model folder draws no graph:

```markdown
## Summary
<What changes, for whom, in one or two lines.>

## What changed
- <One line per change, naming the file, command, or workflow it touches.>

Reruns: <stages and models, or nothing, with the rerun-table row that says so>

## Tests
| Test | Checks |
|---|---|
| `<file>` | <the behavior it asserts> |

Rule, skill, and knowledge updates: <file and lesson, or none with the reason>
Not verified: <what did not run, and why>
Refs #<issue>
```

## Example

A change that raises `calibration_samples` of the `w8a8` profile. The numbers are illustrative.

````markdown
### yolo26n
Upstream: https://huggingface.co/Ultralytics/YOLO26/tree/070ac3c51435984d992ba92eb3f234daf4e4500d · License: AGPL-3.0-only, downloadable
Surgery: cut at the six raw one-to-one Detect heads, before the in-graph box decode and TopK, so Neat's BoxDecode YoloV26 decodes them
Quantization: w8a8 INT8 per-channel weights, INT8 asymmetric activations, histogram MSE with 2048 bins on 100 coco-2017-cal images (was 50) · w8a16 INT8 per-channel weights, BF16 activations with per-token scales, no calibration · w16a16 BF16 weights and activations, no calibration
Scheduling: every push

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

| Build | Compile | Compile time | Compile runner | Performance | Accuracy | Benchmark time |
|---|---|---|---|---|---|---|
| w8a8 | passed | 19 min | default, spot | skipped, 966.4 fps (+0.2%) | passed, coco_map 0.383 (+0.002) | 6 min + 24 min |
| w8a16 | reused | reused | default, spot | reused, 610.2 fps | reused, coco_map 0.386 | reused |
| w16a16 | reused | reused | default, spot | reused, 315.5 fps | reused, coco_map 0.403 | reused |

Local: w8a8 on a Modalix DevKit, MLA at 1 GHz, commit 0123abc: 964.9 fps, coco_map 0.383

Results from commit 0123abc.
Reruns: yolo26n w8a8 compile, performance, and accuracy, because calibration_samples is a compile input. w8a16 and w16a16 rerun nothing.
Local qualification: Model Compiler 3.0.0.2, DevKit platform 3.0.1, Core v0.6.2
Local qualification: yolo26n.pt sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
Local qualification: coco-2017-cal sha256 54c64d84122eba371ac014d7b43849a461551c2e6c5c430b5ead90ec1dec12b1
Local qualification: coco-2017-val sha256 a40de1caf42b47c5a8976eb97efe3c83f3a41337c4d16d318eb15d65c293f2da
Local qualification: `scripts/run_model_compilation.sh models/yolo26n w8a8 w8a8 build`: passed
Local qualification: `model-registry verify-artifact build/yolo26n/w8a8/model`: passed
Local qualification: `scripts/run_model_tests.sh compile models/yolo26n build/yolo26n/w8a8`: passed
Local qualification: `PYTHONPATH=tools ~/pyneat/bin/python models/yolo26n/scripts/performance.py --mpk build/yolo26n/w8a8/model/yolo26n_w8a8_mpk.tar.gz --dataset coco-2017-val=/media/nvme/datasets/coco-2017-val --output results/performance`: passed
Local qualification: `scripts/run_model_accuracy.sh models/yolo26n w8a8 build results/accuracy`: passed
Links: none, because the change adds no data and converts nothing new
Rule, skill, and knowledge updates: none, because the change applies the existing calibration guidance.
Not verified: w8a8 performance and accuracy on a second DevKit.
Refs #<issue>
````

## Rules

- Name runners by class and DevKit type, never by host name.
- Each list item, field line, and table row stays on one source line.
- Leave out AI attribution, internal host names, private links, board addresses, and customer
  information. The repository is public.
