# SiMa Neat Models

This repository is the source of truth for model definitions and for how each
model is compiled, benchmarked, and published. `model.yaml` declares a model, the
model's `scripts/` hold code specific to that model, and `tools/model_registry`
holds the shared lifecycle.

## Vocabulary

A **build** is one build profile of one model, compiled in one compile
invocation named after the profile. Use these names in code, CI, and docs:

| Concept | Name |
|---|---|
| stage | `compile`, `performance`, or `accuracy`; each build passes through up to three |
| benchmark | the performance and accuracy stages, which run on a DevKit |
| model component | a separately compiled part of a model; a single-component model has one, with ID `model` |
| model package | what a build profile ships |
| registry runs | `model_compile`, `model_package`, `model_performance`, `model_accuracy`; a package's `component_run_ids` hold its compile run IDs |
| fingerprint | a SHA-256 of every input that decides a result: `compile_fingerprint`, `benchmark_fingerprint` (Vulcan stores it as `run_fingerprint`) |
| sha256 | the SHA-256 of one file |
| metric | what is measured, with its unit and better direction (`tools/model_report/metrics.json`) |
| measurement | one measured value of a metric |

These names are exceptions, because a file format or another system owns them:

- MPK names only the file format: `--mpk`, `mpk_path`, `mpk_sha256`,
  `mpk_filename`, `compiled_mpks`, `verify-mpk`, and the Vulcan artifact
  `mpk`. Call what an MPK holds a model component.
- Vulcan's `/measurements` API and its per-value `measurement_key`.
- Vulcan's compile-run parameter `build_fingerprint`, which carries the
  compile fingerprint.
- `package_fingerprint`, which equals the package manifest's sha256 (a Vulcan
  contract).
- Core's speed-only `Model.benchmark`, `BenchmarkReport`, `BenchmarkOptions`,
  and `core-benchmark-report`.

Each stage's registry run carries its fingerprint, and CI runs a stage only
when no completed run carries the current fingerprint. A change therefore
reruns exactly the stages whose inputs it touches.

When an application needs supporting files derived from the converted source,
write them under the profile output and declare their relative `path` and
`sha256` in `compile-outputs.json`'s optional `runtime_files` list. The shared
packager includes them beside the component MPKs, and retained-build restoration
restores them too. Accuracy scripts receive the verified profile root as
`--resources`; they must use those local files instead of downloading checkpoints.
See [Ship supporting files](docs/adding-a-model.md#ship-supporting-files).

A tooling change reruns a stage only when the changed file is in that stage's
tooling list in `fingerprints.py`. No shared file is a conversion input: a
conversion reruns only when the model's checkpoint, `convert.py` and the
modules it imports, conversion requirements, or `conversion.outputs`
change. Raise `FINGERPRINT_VERSION` in `conversion.py` in the change that
makes shared tooling convert the same inputs to a different model. Planning
then fails until Vulcan Model Convert reruns for every conversion-backed model.

| Stage | Venue | Model entry point | Runner script |
|---|---|---|---|
| compile (ONNX) | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| compile (LLiMa) | Model Compiler | `scripts/compile.sh` | `scripts/run_llima_compilation.sh` |
| performance | DevKit | `scripts/performance.py` + `requirements-performance.txt` | `scripts/run_model_performance.sh` |
| accuracy | DevKit | `scripts/accuracy.py` + `requirements-accuracy.txt` | `scripts/run_model_accuracy.sh` |

`compile_fingerprint` and `benchmark_fingerprint` in
`tools/model_registry/fingerprints.py` compute the fingerprints. Their inputs, and what
each kind of change reruns, are listed once in
[what a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns);
read both before changing what a stage depends on. Conversion
(`scripts/convert.py`, run by `scripts/run_model_conversion.sh`) produces the
ONNX consumed as either a pinned source or `compilation.source: conversion`.
It writes `model.onnx` unless `conversion.outputs` lists other files: several
ONNX models, or a `.pt`/`.pth` checkpoint that `surgery.py` exports to ONNX.
`compile.py` receives the first as `--model`, with the rest beside it.
`convert.py` exports the upstream forward pass at the shipped input shape, from
its input to the final outputs the package keeps. ONNX simplification with
`onnxsim.simplify`, including constant folding, is allowed in `convert.py` when it
preserves the upstream inputs, outputs and computation. Other changes for the
compiler, even exact ones, belong in `surgery.py`. That includes a cut: an
export that ends at an inner tensor, such as raw heads before the upstream
decode, or starts after an upstream step.
Registry sources resolve during planning; only resolved content enters the compile
fingerprint. See [conversion artifacts](docs/conversion-artifacts.md).

Prefer Hugging Face whenever the required checkpoint is available there, pinned
to a full repository commit. Use an official upstream GitHub release asset only
as a fallback, declared in `conversion.source` with its SHA-256 and byte count.
Explain the fallback in a source comment in `model.yaml`.

The shared tooling is one module per job. The
[module map](design-doc/model-registry-architecture.md#shared-tooling) says
which module owns what; put new shared code in the module that owns its job,
and give a new job a new module named after what it does.

Authored definitions use the shared loader in `model.py`: directory-derived ID,
`architecture: <id>`, conventional recipes, `modalix` target, and one compile
invocation and one `default` package per build profile, derived rather than
declared. Each stage pins the Dataset Registry archive it uses, named by its
source ID `<id>-<variant>`; Hugging Face file sets can share a
repository/revision. Compile recipes call `load_model` instead of parsing
YAML independently. Add a variant that should not appear in the Model Browser
yet as a commented-out profile in `model.yaml`: CI neither compiles nor
benchmarks it, and its tests skip until it is uncommented. A multi-component
model lists its components in run order
and may add a display-only `pipeline`. Preprocessing, decoding and surgery
belong in executable code and the README; ownership belongs in `CODEOWNERS`.
Do not add maturity compatibility fields to new authored YAML.

Performance defaults to each build profile's whole model: every component's
MPK and the CPU work between them. Independent components such as CLIP's image
and text encoders may declare `performance: {scope: components}` instead. Their
script benchmarks each component separately with Core's benchmark API and merges
its single-component `from_performance()` reports using
`model_report.from_component_performance()`. Each component keeps its own
metrics, workload, telemetry, execution plan and MLA stats; the report has no
combined metrics. Every component must satisfy the required performance metrics,
and `tests.yaml` uses `performance.expected.<profile>.components.<component>`.
Composed pipelines such as RF-DETR keep `performance: {}` and one whole-model
result. Accuracy remains model-level and declares its evaluated datasets.
`model.yaml` declares only
`performance`, `accuracy.datasets` (Dataset Registry pins) and optional
`build_profiles`. Every performance report holds the metrics
`tools/model_report/metrics.json` marks `required`. The metrics an accuracy
dataset requires live in `datasets/<registry id>.yaml`, per task; a model
reports, on each dataset, the metrics listed for its `tasks`. The benchmark input, decoder
selection, top-k, score threshold, sample count, duration and MLA layer and stage stats belong
in each model's `scripts/performance.py`. Every benchmarked model must include that standalone
script so customers can copy it and reproduce CI in the PyNeat environment,
with `tools/model_report` on `PYTHONPATH` and without `model_registry` imports
or YAML configuration. Keep its `--mpk` and `--output` CLI. A model with accuracy datasets
also takes `--dataset <source id>=<directory>`, as `accuracy.py` does, and benchmarks one
`random.Random(0)` image from it, letterboxed into a 1280x720 RGB frame by code inside the
script and passed in `BenchmarkOptions.inputs` on the image route, so Neat preprocessing is
timed. Set `preprocess.input_max_width` and `input_max_height` to the frame size, or the
execution plan reports Neat's 1920x1080 default as the input. A stereo model benchmarks one
dataset pair as a side-by-side frame instead. A model whose
inputs are not images benchmarks one `random.Random(0)` validation sample as tensors on the
tensor route, and each timed sample runs the CPU steps that produce the model's result.
`run_model_performance.sh` passes only the first dataset, and only for a profile
accuracy measures. A model with `resources` also takes `--resources <directory>`,
laid out as its package installs them. A model without an accuracy dataset builds a fixed input in its script;
do not benchmark Core's synthetic inputs ([Benchmark
input](docs/benchmarking.md#benchmark-input)). `performance.py`
and `accuracy.py` write `model-report.json` with `model_report.from_performance()`
or `model_report.from_accuracy()`, the latter with one result per dataset source ID;
`schemas/model-report.schema.json` is the report
contract, and `tools/model_report/metrics.json` names every metric a report may
hold. Each benchmarked profile needs finite `tests.yaml` expectations covering its
required metrics (performance: `throughput_fps`); optional expectations and
the shared regression thresholds remain gates.

When no published score matches a package's setup, the model may add a standalone
`scripts/fp32_baseline_accuracy.py`, as `models/superglue_outdoor` does. It scores the
FP32 ONNX the model compiles on a host with the model's own `accuracy.py` scoring, without
`model_registry` imports, credentials or YAML. Its output becomes the
`reference_metrics` entry with `origin: fp32_baseline`. No stage runs it and no
fingerprint includes it ([FP32 baseline](docs/local-accuracy-development.md#fp32-baseline)).

## LLiMa models

LLMs and VLMs declare the complete model as one component, with a pinned Hugging
Face file set, `format: huggingface`, `entrypoint: generate`, and the conventional
`scripts/compile.sh`. Whisper uses `entrypoint: transcribe`. The
recipe only compiles; use adjacent `compile.py` or `config.py` when needed.
`scripts/run_llima_compilation.sh` downloads, compiles, and runs `llima-deploy`
into the component's `deployment/` folder, then writes a direct reproduction
command with download instructions and the resolved graph arguments, using four
processes. Include any Python helpers alongside it.
`model-registry verify-artifact --write-manifest` verifies that folder and writes
`deployment-manifest.json` beside it. Publication preserves the folder as
individually registered files; native builds never create a deployment ZIP.
Do not enumerate generated layers as components or nest model archives in packages.
Only declared performance and accuracy stages run; omit their sections until
implemented. See [LLiMa model definitions](docs/adding-a-model.md#llima-language-and-vision-language-models).
Speculative pairs stay one component: pin both checkpoints in named file-set
subfolders, use the target's `config.json` as `compilation.source.main_file`, and declare the draft
subfolder in `parameters.draft_model_path`. Deploy and publish the complete pair.
An optional profile-level `benchmark_baseline` links to a normal-decoding model
and build profile; it affects performance fingerprints, never compilation.
See [LLiMa performance](docs/benchmarking.md#llima-text-and-image-performance).

Start with one 4K profile per LLM/VLM. Declare `max_num_tokens: 4096` and
`language_group_size` in its component parameters. Image dimensions are optional
overrides for configurable VLMs; omit them for LLMs and fixed-resolution VLMs.
Recipes forward the runner's compiler options rather than duplicating these values.
LLiMa compile fingerprints cover checkpoint content, the target, graph parameters and
the pinned compiler identity. Keep graph-changing options in component
`parameters`; script edits, job counts, logging and runner classes do not
invalidate LLiMa builds.

Any expensive model may set top-level `build-on-demand: true` (default `false`).
Pushes still validate it and publish catalog metadata, but never compile or
benchmark it. A manual **Vulcan Model Orchestrator** dispatch must select its
`models` and `profiles`; the planner also requires `--include-on-demand`.
The flag changes scheduling and catalog metadata only, never fingerprints.
Publication in the selected `VULCAN_ENV` uses `publish-candidate` once per complete model/profile;
it binds every required compile/package run before registering on-demand
artifacts. Ordinary catalog documentation remains branch-retained. Publish
target `build_requirements` using existing compile fingerprints and consume server-owned
`approved_deployments`. After finalization, `publish-branch-deployment` selects
the candidate in its feature catalog; readers consume server-owned
`branch_deployments` before inherited approvals. Never synthesize approval. See
[on-demand builds](design-doc/model-registry-architecture.md#on-demand-builds).

## Tests

Repository-wide tests sit in `tests/`. Each model's tests sit in its folder,
named by what they check:

```text
models/<model_id>/tests/
  tests.yaml                    which stage tests run, and the expected DevKit values
  build/test_conversion.py      when the model has scripts/convert.py
  build/test_compile.py         every model
  benchmark/test_accuracy.py    when model.yaml declares accuracy
  benchmark/test_performance.py when model.yaml declares performance
  fixtures/                     optional test data
```

`tests.yaml` has one entry per stage the model has (`conversion`, `compile`,
`accuracy`, `performance`), each with `run_tests: true`, or `false` with a
`skip_reason`. A running `accuracy` or `performance` entry also sets
`expected`: the latest published `develop` values of every benchmarked build
profile (`performance.expected.<profile>.throughput_fps`,
`accuracy.expected.<profile>.<dataset>.<metric>`). Each benchmark test asserts
`measured_shortfalls(model_dir, "<stage>", build_profile, report) == []`;
it fails when a
value falls more than `ACCURACY_ALLOWED_DROP` (5%) or
`PERFORMANCE_ALLOWED_DROP` (10%) in `model_tests.py` below its expected
value, or rises by as much for a metric that `metrics.json` marks lower is
better. A performance test also asserts that each component's
`execution_plan` in the report is a `sima.neat.execution_plan` version 1 plan
with nodes and edges, the plan the model browser draws. These limits are shared by every model. When a reviewed
change moves a published value, update `expected` to the new published result.
`model-registry validate` enforces the entries, the expected values, and the
layout; [set up
`tests/tests.yaml`](docs/adding-a-model.md#set-up-teststestsyaml) has the
format. A model test imports a script with the root `conftest.py` fixture
`load_script("<name>")`, which stubs the Model Compiler and DevKit modules a
host lacks.

The benchmark tests under `tests/benchmark/` also run on the DevKit, in
PyNeat, which has no `onnx`, `onnxruntime` or `torch`. A test that needs one
imports it inside the test with `pytest.importorskip("onnx")`. A module-level
import stops pytest at collection on the DevKit, and the benchmark then
publishes no result for that stage. Pull request CI has these packages, so it
cannot catch the mistake.

A test's fixtures decide where it runs:

| A test that takes | Runs | A failure |
|---|---|---|
| no stage output | `python -m pytest`, on every pull request and push | blocks the merge |
| `converted` | after `convert.py`, in `vulcan-model-convert.yml` | stops the ONNX upload |
| `compiled` | after compilation, in `vulcan-model-compile.yml` | leaves the build unpublished |
| `accuracy` or `performance` | after the benchmark, in `vulcan-model-benchmarking.yml` | publishes the report with test status `failed`, then fails the run |

`scripts/run_model_tests.sh <stage> <model_dir> <stage_output> [<profile>]`
runs one stage's file on that output (the benchmarks also take the build
profile their `model-report.json` measured) and prints `passed`, `failed`, or `skipped` (the
stage's `run_tests` is `false`); a plain `pytest` deselects every test that
takes stage output. Editing `models/<id>/tests/` or `run_model_tests.sh` reruns nothing,
because no fingerprint includes them. The changed test runs the next time its stage runs. To run it now, dispatch `Vulcan Model Orchestrator`
with `models` set to the model and `force_rebuild_all`, which recompiles and
remeasures it. If its package is already published in the current catalog
revision, use a new Git commit; published packages are immutable within a revision.

## Keep the structure consistent

Each row is a change that is incomplete without the files beside it. Make them
in the same commit.

| When you change | Also change |
|---|---|
| what a stage depends on: a `model.yaml` field its fingerprint reads, or a shared file it runs | the fingerprint function or the `COMPILE_TOOLING`/`BOARD_STAGE_TOOLING`/`PERFORMANCE_TOOLING`/`ACCURACY_TOOLING` list in `fingerprints.py`, the tables in [what a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns), and a test in `tests/test_compile_matrix.py` or `tests/test_stage_plan.py` |
| an approved source host | its entry in `HTTPS_HOST_CHECKS` in `source.py`, accept and reject cases in `tests/test_source.py`, and the host table in [pin every input](docs/adding-a-model.md#3-pin-every-input). This reruns no stage |
| the set of model scripts a benchmark or conversion runs | `SCRIPTS` in `benchmark_declarations.py` or `CONVERSION_FILES` in `model.py`, which make up `NON_COMPILE_SCRIPTS` in `fingerprints.py` with `FP32_BASELINE_SCRIPT`; every other `.py`/`.sh` under `scripts/` is an ONNX compile input |
| board-only code | keep it inside `performance.py` or `accuracy.py`; a helper module under `scripts/` is an ONNX compile input, so editing it recompiles and then remeasures the model |
| how `performance.py` gets its input | `scripts/run_model_performance.sh`, the dataset steps in `vulcan-model-benchmarking.yml`, [Benchmark input](docs/benchmarking.md#benchmark-input), and `tests/test_model_report.py` and `tests/test_benchmark_workflow.py` |
| what a package holds besides MPKs: `resources` (pinned files with no `main_file`) or a compile's `runtime_files` | `package.py`, `candidates.py` and `runtime_files.py`, `validate_compile_outputs` in `artifact.py`, the compile fingerprint in `fingerprints.py`, `source_identity` in `source.py`, `resourceSource` in `model.schema.json`, [Packages and pipeline](docs/metadata-schema.md#packages-and-pipeline), and `tests/test_package.py`, `tests/test_artifact.py`, `tests/test_compile_matrix.py` and `tests/test_source.py` |
| a custom `performance.py` | start from the [canonical script](docs/benchmarking.md#the-performance-script), and for a multi-component model the [multi-component script](docs/benchmarking.md#multi-component-models): hand Core's `BenchmarkReport`, board telemetry, and each component's Model to `model_report.from_performance()` |
| a metric: a Core benchmark reading or an accuracy score the registry should publish | a row in `tools/model_report/metrics.json` with its Vulcan measurement name (`registry_name`), and its entry in the browser's `metrics.js` or `accuracy-metrics.js`, and for a telemetry reading in `TELEMETRY_METRICS` in `model-report.js`, which tests keep in step; for a performance metric, its row in `DETAIL_CARDS` in `model-view.js` when a Performance details card should show it, and the metric table in [benchmarking](docs/benchmarking.md#workload-and-measurements) |
| a `datasets/<id>.yaml` requirement | `dataset-requirements.schema.json` when its shape changes, and the `accuracy.py` and `tests.yaml` expectations of every model with that task that evaluates the dataset |
| a field of the model report | `schemas/model-report.schema.json`, `tools/model_report/__init__.py`, the checks in `report_publication.py`, and the browser code that reads the report |
| a model's category: `model_type`, `tasks`, `modalities`, or the family in `architecture.id` | pick existing IDs from `taxonomies/model-taxonomy.yaml`; when none fits, add the entry there and follow [Model categorization taxonomy](docs/model-taxonomy.md) |
| a model's `license` | set `spdx` to an ID in `artifact-browser/licenses.json`. When the upstream license is not listed, add it there, with `downloadable: false` if it forbids redistribution. No fingerprint includes the license, so the change reruns nothing ([license](docs/metadata-schema.md#license)) |
| the model folder layout | `MODEL_FOLDER_ENTRIES` in `model.py` and [the directory step](docs/adding-a-model.md#2-create-the-directory) |
| a model script | the model's test of that behavior under `tests/build/` or `tests/benchmark/` |
| what a model converts, compiles, or measures on a DevKit | its `tests/tests.yaml` entries, their `expected` values, and the stage test files they run |
| the layout of a model's `tests/` | `TESTS_DIR_ENTRIES` and `TEST_GROUPS` in `model_tests.py`, `schemas/model-tests.schema.json`, and [the tests step](docs/adding-a-model.md#10-write-the-models-tests) |
| how a stage runs its model tests or what they report | `scripts/run_model_tests.sh`, the stage's workflow step, and `TEST_OUTCOMES` in `model_tests.py` |
| a `model-registry` command | `cli.py` (and `AUTHOR_COMMANDS` when authors run it), every workflow and script that calls it, `packaging/model-registry/install_model_registry_package.sh`, the docs, and `tests/test_cli.py` |
| a module that only CI may run | the removal list in `packaging/model-registry/build.sh` |
| a module's job, or a new module | the [module map](design-doc/model-registry-architecture.md#shared-tooling) |
| a published result's shape | its schema under `schemas/` and `artifact-browser/` |
| a CI job, trigger, or concurrency rule | [CI flow](design-doc/model-registry-architecture.md#ci-flow) and the [after-merge steps](docs/contributing.md#after-merge-trusted-staging-automation) |

## CI

- **Pull request.** `Target Branch Checks` is the only workflow that builds or
  tests a pull request, on the merge result and GitHub-hosted runners:
  `model-registry-package.yml` runs `ruff check .` and `python -m pytest`,
  collects every model's stage-output tests, and builds the `model-registry`
  package, and `developer-docs.yml` checks the model browser.
  `PR Sanitization` checks the pull request's description. Contributor code
  never runs on self-hosted runners for a pull request.
- **Push.** Every push runs `Vulcan Model Orchestrator`. Its `portal` job builds and
  smoke-tests the model browser (`developer-docs.yml`) independently of model processing.
  Planning waits for model validation and registry configuration. On `main` and `develop`,
  `publish-model-browser` publishes the portal to staging only after its checks pass.
  The `plan` job runs `model-registry plan-stages`, which reads the
  branch's prior catalog (its own, merged with its base branches' on a feature
  branch; see `design-doc/model-registry-architecture.md`) and resolves each
  fingerprint against the registry. `compile` calls
  `.github/workflows/vulcan-model-compile.yml` once per build to compile,
  run the model's compile tests, publish, and benchmark the build.
  `benchmark` benchmarks each reused build that still lacks a benchmark
  once `initialize-catalog` marks the selected builds pending, or after
  `publish-catalog` writes the final catalog snapshot when nothing compiles.
  Every benchmark is a job of the orchestrator run, grouped under its build's
  name: both paths call `vulcan-model-benchmarking.yml` with the Vulcan config
  and the DevKit runner the orchestrator resolves once per run.
- **Benchmarks.** `vulcan-model-benchmarking.yml` runs the requested
  `stages` of one build profile on a DevKit, runs the model's benchmark tests on
  each `model-report.json`, and publishes performance (`model_performance` runs)
  and accuracy (`model_accuracy` runs) separately with `model-registry
  publish-report`, each under the profile's `model_package` run with its test
  outcome. A failed test fails its check job after publication; the completed
  run is not benchmarked again. Pushes do not benchmark on-demand builds.
  The manual `promoted_missing` checkbox queues missing stages except `text_generation`/`visual_question_answering` models.
  A manual dispatch of the same workflow resolves its own config and runner.
- Benchmark setup uses `neat-core.install_target` from the measured source's
  `deps/manifest.json`. `ensure-neat-core` resolves floating targets once and
  skips installation only when Core's channel/tag/environment and the artifact's
  Runner/Scheduler versions match
  `neat --json`. Unverifiable state installs;
  an unresolved floating target fails. Both paths report PyNeat and packages.
  Core's production artifact source is independent of the Model Registry environment.
- Workflow YAML only orchestrates: triggers, jobs, permissions, inputs, and
  the order of steps. A step that needs more than one command calls a named,
  executable script with a usage header: under `scripts/`, or beside the code
  it serves, as `packaging/model-registry/build.sh` does. Registry logic
  belongs in a `model-registry` command. A new script under `scripts/` is not
  a fingerprint input unless `fingerprints.py` lists it.
- A newer push cancels the branch's running orchestrator, its compiles and its
  benchmarks; builds and benchmark runs that already published stay and are
  reused. How branch catalogs, merges, and the
  model browser interact is in
  [branches and merges](design-doc/model-registry-architecture.md#branches-merges-and-concurrent-pushes).

## Contributing

1. Declare the change in `model.yaml` and the model's `scripts/` and
   `tests/`, following [adding a model](docs/adding-a-model.md) and the
   [metadata schema](docs/metadata-schema.md). Record the source model's
   published scores in `reference_metrics`.
2. Run `ruff check .`, `python -m pytest`, and `model-registry validate models/<model_id>`.
   `tests/test_target_branch_checks.py` rejects committed files that
   `.gitignore` excludes or that exceed 2 MiB, except its `ALLOWED_FILES`.
3. Compile each changed build in the pinned Model Compiler with
   the backend's runner (`run_model_compilation.sh` or `run_llima_compilation.sh`)
   and check each compiled artifact with
   `model-registry verify-artifact` ([local development](docs/local-development.md)).
4. On a DevKit, run the model's standalone `scripts/performance.py --mpk
   --output` with `PYTHONPATH=tools`, with one `--mpk component=<archive>` per
   component for a [multi-component model](docs/benchmarking.md#multi-component-models),
   and, for builds that declare accuracy, `scripts/run_model_accuracy.sh` on
   the compile output ([benchmarking](docs/benchmarking.md),
   [local accuracy](docs/local-accuracy-development.md)).
5. Report which stages you ran and where; CI publishes the results.

Conventions the code relies on:

- Model scripts receive verified local paths (`--model`, `--dataset`,
  `--fidelity-dataset`, `--mpk`, and `--resources` when supporting files are
  declared). Evaluation datasets come through the Dataset Registry declared
  in `model.yaml`; supporting files travel with the verified model package.
  Benchmark scripts run without registry credentials.
- When a model declares `conversion.outputs`, every script reads those names
  with `conversion_outputs()` and never hard-codes one, so renaming a file only
  changes `model.yaml`.
- Schemas under `schemas/` reject unknown fields. Published reports keep
  their `schema_version` (model report 1, numerical fidelity 1) because the
  artifact browser and `sima-cli` choose a renderer by it. Registry
  documents that only this repository's tooling reads (`model.yaml`, compile
  outputs, package manifests) carry no version; change their readers and
  writers together instead.

## Model browser

The default Cards overview groups available models by their existing architecture
ID. Each architecture in `taxonomies/model-taxonomy.yaml` requires a short
`description` that covers the whole family, without size-specific claims. Keep
family text there; derive membership, tasks and hero selection from filtered
models. Family cards select the existing Model family filter; search and Table
continue to show individual models. This is catalog metadata: do not add taxonomy
or browser files to conversion, compile or benchmark fingerprints.

Before changing `artifact-browser/`, read its
[design context](artifact-browser/DESIGN.md) and
[README](artifact-browser/README.md#how-the-page-is-built). What users see is in
[docs/model-browser.md](docs/model-browser.md); update it with any visible
change.

- Make every expandable area a `disclosure()` row from `ui.js`. Raw
  `<details>` elements fail `tests/test_artifact_browser.py`.
- The browser lists INT8 build profiles first and opens a model on INT8;
  `profileGroups` in `api.js` sets this, not the `model.yaml` profile order.
- Keep the glance-first page: key facts in the section, the rest in one row.
  Actions are icon buttons; blue text with ↗ is only for external links.
- Benchmarks default to the whole model, per build profile. Independent-component
  performance keeps separate measurements when `performance.scope: components`.
  Show layer timings and execution plans per component; never add component
  numbers up or mark one component best against a different component.
- Render the current model report version, and let unknown metrics fall
  through to Other measurements rather than hiding them.
- Keep the preview data-driven. `preview.py` builds its catalog from
  `models/`; never add a model, name, or README to it by hand. It needs the
  repository installed (`python -m pip install .`) or `PYTHON` pointing at
  such an interpreter.
- Add a new module to `fingerprintedAssets` in
  `website/scripts/assemble-portal.mjs`. Assembly also writes
  `component-pipelines.<revision>.json` and `reference-metrics.<revision>.json`
  from each `model.yaml`, so `build:portal` needs the same Python as
  `preview.py`. It copies the browser's `licenses.json` to
  `licenses.<revision>.json`.
- Offer a model's downloads only when `downloadOffered` in `downloads.js`
  allows them. It reads the license from the branch catalog and
  `licenses.json`, and an unlisted license offers none. Keep the list in
  `artifact-browser/`, so a plain static server serves it with the page.
- A new metric goes into `tools/model_report/metrics.json` and the browser's
  `metrics.js` or `accuracy-metrics.js`; a test keeps their labels, units and
  directions in step.
- Before a push, run `python -m pytest tests/test_artifact_browser.py`,
  `node --test tests/browser/*.test.mjs`, and `npm run typecheck`,
  `npm run build:portal`, and `npm run smoke:portal` with `--prefix website`,
  then look at the changed pages in `preview.py` at desktop and phone width,
  light and dark.

## Model README requirements

Every model under `models/<model_id>` has a `README.md`. The model page shows
the model's metrics, variants, components, license, and source links, then the
README. The README holds what an application developer needs and the page
cannot show: what the model does, how to prepare its input, and how to read its
outputs. Start from the template in
[docs/adding-a-model.md](docs/adding-a-model.md#11-write-the-readme).

`model-registry validate-models` enforces this structure:

1. `# <display name>`, matching `display_name` in `model.yaml`.
2. `![<display name> model hero](hero.png)`
3. `## Overview`: one or two prose paragraphs of 50 to 120 words. Say what the
   model finds, what input it takes, what it returns, and which components it
   runs as. The registry publishes this section as the catalog summary and
   search text.
4. `## Inputs and outputs`: what the application does with the model. Give the
   input preparation with its exact values, each step between components, and
   each actual output tensor by name. Write the decode steps as a
   numbered list, give the class-index mapping, and name the Neat decoder that
   handles the outputs when one exists.
5. `## Building applications`: the `BUILDING_APPLICATIONS` text from
   `tools/model_registry/documentation.py`, unchanged. LLiMa deployments
   (`interface.format: huggingface`) use `GENAI_BUILDING_APPLICATIONS` instead.

Write the README for an application developer:

- Keep it within 350 words after the hero, with H1 and H2 headings only.
- Name hardware and tools in plain words, such as "the CPU" and "the package".
  The validator rejects internal terms (MPK, A65, MLA, AFE, ELF, tessellation)
  and metric and tool names (mAP, AP50, FPS, latency, benchmark, sima-cli,
  model-registry, Vulcan).
- Leave to the model page what it reads from `model.yaml` and registry runs:
  scores and metrics, benchmark settings, variants and compile settings,
  versions, license and redistribution terms, checksums, download commands, and
  links to the original model or its source repository.
- Use the same wording for every size in a model family, so the READMEs differ
  only in values.
- Take each value from `model.yaml` and the current scripts, and update the
  README in the change that alters it.
- Write each list item on one line, because the model browser ends a list item
  at its line break.

Keep maintainer explanations next to what they explain: in `model.yaml`
comments, or in the docstrings of `convert.py`, `performance.py`, and
`accuracy.py`. For ONNX models, every other file under `models/<id>/scripts/`
is part of the compile fingerprint, so a comment there recompiles the model. Procedures
that apply to every model belong in `docs/`. Maintainer guides inside a model directory, such
as `scripts/qat/README.md`, use whatever structure suits their workflow.

Every model has a hero: a regular (non-symlinked) 1200x480 PNG no
larger than 2 MiB. The registry's strict PNG profile permits `IHDR`, optional
`PLTE`, `IDAT`, `IEND`, and optional validated `tIME` or `tRNS` ancillary
chunks. Keep it legible in light and dark themes, avoid embedding essential
instructions in the pixels, and use the exact accessible Markdown alt text
shown above. The registry publishes the image as a checksum-bound Vulcan
artifact alongside the README; do not use a placeholder, remote image URL, or
an image outside the model directory.

Create a hero for each model and keep heroes consistent across the catalog.
Each hero shows the model's output on an input, such as labeled boxes on a
scene, in a scene no other model uses. Models of one architecture share a
style and accent color, and labels use the same size on every hero.

For new or refreshed artwork, follow the [illustration guide](docs/illustration-guide.md),
the single source for the two geometric comic styles, exact palette, Roboto
typography, prediction overlays and reusable prompts. Use existing model heroes
as visual references.

When a change alters what the model detects, its input, its outputs, or its
components, update the README and the hero in the same change. A README- or
hero-only edit is documentation or catalog metadata; it must not change
a compile or benchmark fingerprint.
