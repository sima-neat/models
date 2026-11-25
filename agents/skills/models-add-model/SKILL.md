---
name: models-add-model
description: Add a model to sima-neat/models, or change an existing model, from model.yaml to a qualified pull request. Use for a new model or family member, a new build profile or component, and a change to a model's sources, conversion, surgery, compile recipe, benchmark scripts, or tests.
---

# Add or change a model

The references are [`models/yolo26n`](../../../models/yolo26n) for one component and
[`models/rf_detr_nano`](../../../models/rf_detr_nano) for two components with a host step
between them. Before step 1, read [`rules.md`](../../rules.md),
[`coding-style.md`](../../coding-style.md), and the [`change-map.md`](../../change-map.md) rows
of every file the change will touch.

Keep the change inside the model's folder. A shared file such as
`scripts/run_model_compilation.sh` is an input of every model, so an edit to it reruns models
this work has nothing to do with ([`rules.md` "Before a push"](../../rules.md#before-a-push)).

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
4. Write `surgery.py`, `calibration.py`, and `compile.py` as `coding-style.md` describes.
   `compile.py` imports the shared helpers from `model_registry.compile_script` and
   `model_registry.fidelity`, `calibration.py` lists and reads images with
   `model_inputs.images`, and both keep the recipe, and each component calibrates on at most 20 samples, as [`rules.md`
   "Compile"](../../rules.md#compile) says. Judge each rewrite and each quantization choice with
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
   `run_tests: false` and a `skip_reason`. When `model-registry validate` fails a profile whose
   expected accuracy is more than 8% below its reference score, comment the profile out as
   [comment out a variant far below its reference
   score](../../../docs/adding-a-model.md#comment-out-a-variant-far-below-its-reference-score)
   describes. While `accuracy` sets `run_tests: false`, neither `model-registry validate` nor
   the DevKit test applies that rule, so compare each profile's measured accuracy with its
   reference score by hand and comment the profile out when it is more than 8% below. The
   step is done when
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
    describes, run `scripts/run_model_performance.sh <model_dir> <profile>
    <compile-output-root> <result-dir>` for each changed build, and
    `scripts/run_model_accuracy.sh` with the same arguments for each build that declares
    accuracy ([benchmark
    locally](../../../docs/local-development.md#benchmark-locally-on-a-modalix-devkit)). A
    direct `performance.py` run sets no clocks and runs no `check-report`, and it needs
    `--output`, and `--dataset` for a model with an accuracy dataset, beside `--mpk`. The
    step is done when each declared stage wrote
    `model-report.json` and passed its stage test, or its failure is recorded for the
    description.
11. Before a push, list the stages the change reruns, per model and build profile, with
    [`rules.md` "Stages and fingerprints"](../../rules.md#stages-and-fingerprints) and [what a
    change reruns](../../../docs/adding-a-model.md#9-know-what-a-change-reruns). A model with
    `build-on-demand: true` builds only on request ([on-demand
    builds](../../../docs/adding-a-model.md#on-demand-builds)). When the list holds a
    model the change is not about, stop and say so as [`rules.md` "Before a
    push"](../../rules.md#before-a-push) says. The step is done when every changed file maps to
    a row of that table, and the user has approved any rerun beyond the changed models.
12. Write the description with [`models-write-pr-description`](../models-write-pr-description/SKILL.md).
    Its lesson step writes what the work taught into the owning files first. The description
    reports which stages ran and where, and CI publishes the results. The step is done when
    each lesson is in its file, every local run appears in the description labelled local, and
    every stage not run is under `Not verified`.
