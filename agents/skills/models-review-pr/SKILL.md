---
name: models-review-pr
description: Review a sima-neat/models pull request for merge readiness. Use for the automatic review, for a full review, for a review of one changed model, and for judging whether a change should update agents/, skills/, or docs/.
---

# Review a Models pull request

Review each changed model whole: its source pins, conversion, surgery, compile recipe,
benchmark scripts, tests, and README. A changed line can break a stage it does not touch, and a
stage the change does not rerun keeps its old result. [`rules.md`](../../rules.md), with its
[rules no check catches](../../rules.md#rules-no-check-catches),
[`coding-style.md`](../../coding-style.md), [`change-map.md`](../../change-map.md), and
[`docs/`](../../../docs) own the rules. This skill says what to check and in which order, and
each finding cites the owning file and section.

## Steps

1. Read the linked issue and the description. List every claim: each changed model and build
   profile, each rewrite and quantization choice, the reruns, each measured value, and each
   local run. Check the description against
   [`models-write-pr-description`](../models-write-pr-description/SKILL.md), and report each block, graph,
   table row, or closing line it lacks.
2. Read `rules.md`, `coding-style.md`, and the `change-map.md` rows of every changed file. The
   step is done when each changed file maps to its rows and you have listed the files each row
   requires.
3. For each changed model, trace source pin, conversion, surgery, compile recipe, benchmark
   scripts, tests, and README, in that order. Judge the graph and recipe choices with
   [`sima-compile-model`](../../../skills/sima-compile-model/SKILL.md), and the measurement
   with [`sima-benchmark-model`](../../../skills/sima-benchmark-model/SKILL.md). The step is
   done when every claim maps to code you read, or is reported missing.
4. Work out which stages the change reruns with [`rules.md` "Stages and
   fingerprints"](../../rules.md#stages-and-fingerprints) and [what a change
   reruns](../../../docs/adding-a-model.md#9-know-what-a-change-reruns), and compare the result
   with the description's `Reruns` line. Report a stage that reruns without a stated reason, and
   a change meant to move a result that reruns nothing.
5. Check every item under Shape and under Evidence, and report each one that fails.
6. Compare the change with [`rules.md` "Where a lesson goes"](../../rules.md#where-a-lesson-goes).
   Name each lesson the work established that no file holds, and each statement in `agents/`,
   `skills/`, or `docs/` the change proves wrong, with the file and the text to add or correct.
7. Report the two verdicts, then the findings:
   - **Shape:** whether the change follows `rules.md`, `coding-style.md`, and `change-map.md`.
   - **Rule, skill, and knowledge updates:** the file and the text to add or correct, or
     `none` with the reason.

## Shape

- **Scope.** The change does what its issue and description claim and adds only stated scope.
  A new model has one owner, an unambiguous source, and a license
  ([step 1](../../../docs/adding-a-model.md#1-decide-what-the-product-contains)).
- **Provenance.** Every source, resource, and dataset is pinned as `rules.md` and [pin every
  input](../../../docs/adding-a-model.md#3-pin-every-input) say. `license.spdx` is listed in
  `artifact-browser/licenses.json`, and a downloadable ONNX model carries its [license
  files](../../../docs/metadata-schema.md#license-files-in-the-package). A `published` entry in
  `reference_metrics` holds only numbers its pinned page prints for these weights, and an
  `fp32_baseline` entry holds what `scripts/fp32_baseline_accuracy.py` measured ([reference
  metrics](../../../docs/metadata-schema.md#reference-metrics)).
- **Conversion and surgery.** `convert.py` exports and `surgery.py` rewrites as `rules.md`
  splits them. Each rewrite has a purpose in the description and a test.
- **Recipe.** Compiler parameters live in the build profile, and profile IDs follow
  [`naming-conventions.yaml`](../../../naming-conventions.yaml). Each choice away from the
  compiler default carries its reason, and no `**kwargs` compiler argument is passed at its
  default ([`coding-style.md` "compile.py"](../../coding-style.md#compilepy)). `compile.py`
  takes the shared helpers from `model_registry.compile_script` and `model_registry.fidelity`, passes
  `compiler_log_level()` to every `quantize` and `compile` call, keeps its recipe,
  and calibrates each component on at most 20 samples ([`rules.md`
  "Compile"](../../rules.md#compile)). A `runner_class` or `runner_market` request links its
  approval ([request additional compile
  resources](../../../docs/contributing.md#request-additional-compile-resources)).
- **Benchmark scripts.** They import only `model_report` and `model_inputs` as `rules.md` says, follow the [canonical
  script](../../../docs/benchmarking.md#the-performance-script), and name what ran in
  `workload`. Each accuracy report holds the metrics its `datasets/<id>.yaml` requires.
- **Code.** Each changed script follows `coding-style.md`. Model-specific code sits under
  `models/<model_id>/scripts/` and shared code under `tools/` ([review and merge
  expectations](../../../docs/contributing.md#review-and-merge-expectations)). Report each
  comment that restates the code, runs past two lines, tells how the code got this way, or names
  compiler or runtime source that is not public, and each chosen value that lacks its reason
  ([`coding-style.md` "A comment leads with its
  point"](../../coding-style.md#a-comment-leads-with-its-point)).
- **Tests.** They follow `coding-style.md` "Tests" and the DevKit import rule in `rules.md`.
  `tests/tests.yaml` has one entry per stage, and its `expected` values follow published
  results under the shared limits in [`rules.md` "Limits"](../../rules.md#limits).
- **README and graph.** The README follows [model README](../../../docs/model-readme.md). Its
  Inputs, Between the components, and Outputs match the surgery constants and `decode` in
  `accuracy.py`. The description's graph follows [`flowgraph.md`](../../references/flowgraph.md)
  and matches the components and `pipeline` in `model.yaml`.
- **Structure.** Every change-map row the change touches is complete in the same commit,
  including the fingerprint lists, rerun tables, and tests a shared tooling change needs.
- **Variants.** A profile is commented out, as `rules.md` says, only after it is proven
  unusable ([review and merge
  expectations](../../../docs/contributing.md#review-and-merge-expectations)), or because its
  accuracy is more than 8% below its reference score. Its comment then gives the metric, both
  values, and the gap. While `accuracy` in `tests/tests.yaml` sets `run_tests: false`, no
  check applies the 8% rule, so compare each profile's measured accuracy with its reference
  score yourself.

## Evidence

- **Claims.** Each changed build produces what the description claims.
- **Local compile.** Each changed build compiled with its runner script in the pinned Model
  Compiler, `verify-artifact` passed, and the compile tests passed on the output. Pull request
  checks compile nothing ([CI and trust
  boundaries](../../../docs/contributing.md#ci-and-trust-boundaries)).
- **Local DevKit.** Each declared stage ran on a DevKit with the pinned platform and Core and
  the clocks that [set up a DevKit](../../../docs/benchmarking.md#set-up-a-devkit) requires. Its
  numbers are labelled local and kept apart from published values.
- **CI.** Read the **Vulcan Model Orchestrator** run of the branch's head commit, each build's
  Compile, Publish, and benchmark jobs ([CI flow](../../references/ci-flow.md)). A successful
  compile says nothing about its benchmarks, and a skipped stage test proves nothing.
- **Published values.** Each value in the table matches the model browser's developer view for
  the branch at the commit the description names. Its change is against the base branch's
  published value.
- **On demand.** A model with `build-on-demand: true` has a requested build and benchmark for
  each result it claims ([on-demand builds](../../../docs/adding-a-model.md#on-demand-builds)).
- **Conversion.** When a conversion input changed, **Vulcan Model Convert** ran again in each
  environment that builds the model ([when to run Convert
  again](../../../docs/conversion-artifacts.md#when-to-run-convert-again)).
- **FP32 baseline.** A `reference_metrics` entry with `origin: fp32_baseline` comes with its
  command and output ([FP32 baseline](../../../docs/local-accuracy-development.md#fp32-baseline)).
- **Runner.** An elevated class or the on-demand market links the first run's **Compile
  runner** summary ([verify the
  result](../../../docs/runner-class-requests.md#verify-the-result-and-diagnose-failures)).

## Findings

- Anchor each finding to the smallest changed line range. State the condition, the breakage,
  its impact, and what should change, in plain English.
- Report one finding per root cause. Separate blockers, such as a wrong pin, a missing
  companion file, a broken stage, or a result the evidence does not support, from optional
  naming, style, and performance follow-ups. A blocker can stand while CI passes.
- Report style only where `coding-style.md` states the rule, and a test request only with the
  behavior that could regress.
- Report anything not run or observed as `not verified`.
