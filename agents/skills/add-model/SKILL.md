---
name: add-model
description: Add a model to sima-neat/models, or change an existing model, from model.yaml to a qualified pull request. Use for a new model or family member, a new build profile or component, and a change to a model's sources, conversion, surgery, compile recipe, benchmark scripts, or tests.
---

# Add or change a model

The references are [`models/yolo26n`](../../../models/yolo26n) for one component and
[`models/rf_detr_nano`](../../../models/rf_detr_nano) for two components with a host step
between them. Before step 1, read [`rules.md`](../../rules.md),
[`coding-style.md`](../../coding-style.md), and the [`change-map.md`](../../change-map.md) rows
of every file the change will touch.

## Steps

1. Decide what the product contains, as
   [step 1](../../../docs/adding-a-model.md#1-decide-what-the-product-contains) lists: the
   source and its owner, the license, the components in run order, the build profiles, and the
   data each stage needs. For a change, also list what the model's README and
   `model.yaml` promise today. The step is done when each item has a value, or a reason the pull
   request will state.
2. Declare `model.yaml` as [steps 2 to 8](../../../docs/adding-a-model.md#2-create-the-directory)
   and the [`model.yaml` reference](../../../docs/metadata-schema.md) describe, including the
   license resource and the source model's published scores in `reference_metrics`. The step is
   done when `model-registry validate models/<model_id>` passes.
3. When upstream publishes a checkpoint, write `convert.py` and its requirements as
   [`coding-style.md` "convert.py"](../../coding-style.md#convertpy) and [what `convert.py`
   exports](../../../docs/conversion-artifacts.md#what-convertpy-exports) describe, and run
   `scripts/run_model_conversion.sh`. Then run **Vulcan Model Convert** in each environment
   that builds the model ([publish a
   conversion](../../../docs/conversion-artifacts.md#publish-a-conversion)). The step is done
   when the conversion tests pass and a published conversion matches the current conversion
   inputs.
4. Write `surgery.py`, `calibration.py`, and `compile.py` as `coding-style.md` describes. Judge
   each rewrite and each quantization choice with
   [`neat-compile-model`](../../../skills/neat-compile-model/SKILL.md). The step is done when
   each rewrite has a purpose the description can state under `Surgery`, and each choice away
   from the compiler default has its comment. An LLM or VLM writes `compile.sh` as [LLiMa
   models](../../../docs/adding-a-model.md#llima-language-and-vision-language-models)
   describes instead.
5. Write `performance.py`, and `accuracy.py` when `model.yaml` declares accuracy, following
   [DevKit performance](../../../docs/benchmarking.md) and [accuracy and
   fidelity](../../../docs/local-accuracy-development.md). Judge the workload and the
   measurement settings with
   [`neat-benchmark-model`](../../../skills/neat-benchmark-model/SKILL.md). The step is done
   when each script runs with only the arguments those pages name.
6. Write the tests and `tests/tests.yaml` as [the tests
   step](../../../docs/adding-a-model.md#10-write-the-models-tests) and `coding-style.md`
   "Tests" describe. A measurement stage with no published report to take values from sets
   `run_tests: false` and a `skip_reason`. The step is done when
   `python -m pytest models/<model_id>/tests` passes and each behavior the change adds or
   changes has a test.
7. Write or update the README and hero as [model README](../../../docs/model-readme.md) and
   [writing model READMEs](../../references/writing-model-readmes.md) describe, and draw the
   model's graph with [`flowgraph.md`](../../references/flowgraph.md). The step is done when
   the README's Inputs, Between the components, and Outputs match the surgery constants and
   `decode` in `accuracy.py`, and `model-registry validate-models --repository .` passes.
8. Run `ruff check .`, `python -m pytest`, and `model-registry validate models/<model_id>`.
   `tests/test_target_branch_checks.py` rejects a committed file that `.gitignore` excludes or
   that is larger than [`rules.md` "Limits"](../../rules.md#limits) allows, unless
   `ALLOWED_FILES` lists it. The step is done when all three pass.
9. Compile each changed build in the pinned Model Compiler with the backend's runner,
   `scripts/run_model_compilation.sh` or `scripts/run_llima_compilation.sh`, and check each
   artifact with `model-registry verify-artifact` ([qualify
   locally](../../../docs/local-development.md)). The step is done when every changed build
   compiled and verified, and `scripts/run_model_tests.sh compile` passed on its output.
10. On a DevKit prepared as [set up a DevKit](../../../docs/benchmarking.md#set-up-a-devkit)
    describes, run the model's `scripts/performance.py` with `PYTHONPATH=tools` and
    `--mpk <archive>`. A multi-component model takes one `--mpk <component>=<archive>` per
    component instead. Run `scripts/run_model_accuracy.sh` for each build that declares accuracy. The step is done when each declared stage wrote
    `model-report.json` and passed its stage test, or its failure is recorded for the
    description.
11. List the stages the change reruns, per model and build profile, with
    [`rules.md` "Stages and fingerprints"](../../rules.md#stages-and-fingerprints) and [what a
    change reruns](../../../docs/adding-a-model.md#9-know-what-a-change-reruns). A model with
    `build-on-demand: true` builds only on request ([on-demand
    builds](../../../docs/adding-a-model.md#on-demand-builds)). The step is done when every
    changed file maps to a row of that table.
12. List what the work taught, and place each lesson as [`rules.md` "Where a lesson
    goes"](../../rules.md#where-a-lesson-goes) says. The step is done when each lesson is
    written into its file, or the description records `Rule, skill, and knowledge updates: none`
    with the reason.
13. Write the description with [`write-pr-description`](../write-pr-description/SKILL.md). It
    reports which stages ran and where. CI publishes the results. The step is done when every
    local run appears in the description labelled local, and every stage not run is under
    `Not verified`.
