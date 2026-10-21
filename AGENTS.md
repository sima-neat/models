# SiMa Neat Models

This repository is the source of truth for model definitions and for how each
model is compiled, measured, and published. `model.yaml` declares a model, the
model's `scripts/` hold code specific to that model, and `tools/model_registry`
holds the shared lifecycle.

## Stages and stage keys

A **build** is one compile invocation of one build profile of one model. Each
build passes through up to three **stages**. Each stage has a **stage key**, a
SHA-256 of everything its result depends on. The registry run of a stage
carries its key, and CI runs a stage only when no completed run carries the
current key. A change therefore reruns exactly the stages whose inputs it
touches.

| Stage | Venue | Model entry point | Runner script |
|---|---|---|---|
| compile (ONNX) | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| compile (LLiMa) | Model Compiler | `scripts/compile.sh` | `scripts/run_llima_compilation.sh` |
| performance | DevKit | `scripts/performance.py` | `scripts/run_model_performance.sh` |
| accuracy | DevKit | `scripts/accuracy.py` + `requirements-accuracy.txt` | `scripts/run_model_accuracy.sh` |

`compile_key`, `performance_key` and `accuracy_key` in
`tools/model_registry/stage_keys.py` compute the keys. Their inputs, and what
each kind of change reruns, are listed once in
[what a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns);
read both before changing what a stage depends on. Conversion
(`scripts/convert.py`, run by `scripts/run_model_conversion.sh`) produces the
ONNX that `model.yaml` pins as the source; only that pin enters the compile
key.

The shared tooling is one module per job. The
[module map](design-doc/model-registry-architecture.md#shared-tooling) says
which module owns what; put new shared code in the module that owns its job,
and give a new job a new module named after what it does.

Authored definitions use the shared loader in `model.py`: directory-derived ID,
`architecture: <id>`, conventional recipes, `modalix` target, optional empty
sections and generated single-component packages/compositions. Dataset aliases
retain immutable pins; Hugging Face file sets can share a repository/revision.
Compile recipes call `load_model` instead of parsing YAML independently. Keep
multi-component boundary contracts, execution order and CPU steps explicit.
Preprocessing, decoding and surgery belong in executable code and the README;
ownership belongs in `CODEOWNERS`. Do not add maturity compatibility fields to
new authored YAML.

`benchmark: true` schedules `modalix_latency` for a single ONNX component across
its supported profiles. Explicit entries retain workload IDs, scope, profile
restrictions and required metrics. Decoder selection, top-k, sample count,
duration and layer statistics belong in each model's `scripts/performance.py`.
Every benchmarked model must include that standalone script so customers can
copy it and reproduce CI in the PyNeat environment, without registry imports
or YAML configuration. Keep its `--mpk` and `--output` CLI.
Executable accuracy keeps `engine: devkit_script` and nonempty required metrics;
missing `engine` remains recorded-only. Returned accuracy names and regression
expectations use `taxonomies/accuracy-metrics.yaml`. Each running profile needs
nonempty finite expectations covering required metrics; optional expectations
and the shared regression thresholds remain gates.

## LLiMa models

LLMs and VLMs declare the complete model as one component, with a pinned Hugging
Face file set, `format: huggingface`, `entrypoint: generate`, and the conventional
`scripts/compile.sh`. Whisper uses `entrypoint: transcribe`. The
recipe calls `llima-compile` and saves a standalone `compile.sh` with its complete
arguments; `scripts/run_llima_compilation.sh` downloads,
compiles, and runs `llima-deploy` into the component's `deployment/` folder.
`model-registry verify-artifact --write-manifest` verifies that folder and writes
`deployment-manifest.json` beside it. Publication preserves the folder as
individually registered files; native builds never create a deployment ZIP.
Do not enumerate generated layers as components or nest model archives in packages.
Only declared performance and accuracy stages run; omit their sections until
implemented. See [LLiMa model definitions](docs/adding-a-model.md#llima-language-and-vision-language-models).
Speculative pairs stay one component: pin both checkpoints in named file-set
subfolders, use the target's `config.json` as entrypoint, and declare the draft
subfolder in `parameters.draft_model_path`. Deploy and publish the complete pair.
An optional profile-level `benchmark_baseline` links to a normal-decoding model
and build profile; it affects performance keys, never compilation.

Start with one 4K profile per LLM/VLM. Declare `max_num_tokens: 4096` and
`language_group_size` in its component parameters. Image dimensions are optional
overrides for configurable VLMs; omit them for LLMs and fixed-resolution VLMs.
Recipes forward the runner's compiler options rather than duplicating these values.
LLiMa compile keys cover checkpoint content, the target, graph parameters and
the pinned compiler identity. Keep graph-changing options in component
`parameters`; script edits, job counts, logging and runner classes do not
invalidate LLiMa builds.

Any expensive model may set top-level `build-on-demand: true` (default `false`).
Pushes still validate it and publish catalog metadata, but never compile or
measure it. A manual **Vulcan Model Orchestrator** dispatch must select its
`models` and `profiles`; the planner also requires `--include-on-demand`.
The flag changes scheduling and catalog metadata only, never stage keys.
Staging publication uses `publish-candidate` once per complete model/profile;
it binds every required compile/package run before registering on-demand
artifacts. Ordinary catalog documentation remains branch-retained. Publish
target `build_requirements` using existing compile keys and consume server-owned
`approved_deployments`. After finalization, `publish-branch-deployment` selects
the candidate in its feature catalog; readers consume server-owned
`branch_deployments` before inherited approvals. Never synthesize approval. See
[on-demand builds](design-doc/model-registry-architecture.md#on-demand-builds).

## Tests

Repository-wide tests sit in `tests/`. Each model's tests sit in its folder,
named by what they check:

```text
models/<model_id>/tests/
  tests.yaml                  which stage tests run, and the expected DevKit values
  build/test_conversion.py    when the model has scripts/convert.py
  build/test_compile.py       every model
  measure/test_accuracy.py    when model.yaml declares DevKit accuracy
  measure/test_performance.py when model.yaml declares a benchmark
  fixtures/                   optional test data
```

`tests.yaml` has one entry per stage the model has (`conversion`, `compile`,
`accuracy`, `performance`), each with `run_tests: true`, or `false` with a
`skip_reason`. A running `accuracy` or `performance` entry also sets
`expected`: the latest published `develop` value of every measured build, per
build profile for its accuracy regression metrics, and per build profile and
component for `throughput_fps`. Each measure test asserts
`measured_shortfalls(model_dir, "<stage>", result) == []`; it fails when a
value falls more than `ACCURACY_ALLOWED_DROP` (5%) or
`PERFORMANCE_ALLOWED_DROP` (10%) in `model_tests.py` below its expected
value. A performance test also asserts that `execution_plan` is a
`sima.neat.execution_plan` version 1 plan with nodes and edges, the plan the
model browser draws. These limits are shared by every model. When a reviewed
change moves a published value, update `expected` to the new published result.
`model-registry validate` enforces the entries, the expected values, and the
layout; [set up
`tests/tests.yaml`](docs/adding-a-model.md#set-up-teststestsyaml) has the
format. A model test imports a script with the root `conftest.py` fixture
`load_script("<name>")`, which stubs the Model Compiler and DevKit modules a
host lacks.

A test's fixtures decide where it runs:

| A test that takes | Runs | A failure |
|---|---|---|
| no stage output | `python -m pytest`, on every pull request and push | blocks the merge |
| `converted` | after `convert.py`, in `vulcan-model-convert.yml` | stops the ONNX upload |
| `compiled` | after compilation, in `vulcan-model-compile.yml` | leaves the build unpublished |
| `accuracy` or `performance` | after measurement, in `vulcan-model-benchmarking.yml` | publishes the result with test status `failed`, then fails the run |

`scripts/run_model_tests.sh <stage> <model_dir> <stage_output>` runs one
stage's file on that output and prints `passed`, `failed`, or `skipped` (the
stage's `run_tests` is `false`); a plain `pytest` deselects every test that
takes stage output. No stage key reads `models/<id>/tests/` or
`run_model_tests.sh`, so a changed test reruns nothing and applies the next
time its stage runs. To apply it now, dispatch `Vulcan Model Orchestrator`
with `models` set to the model and `force_rebuild_all`, which recompiles and
remeasures it.

## Keep the structure consistent

Each row is a change that is incomplete without the files beside it. Make them
in the same commit.

| When you change | Also change |
|---|---|
| what a stage depends on: a `model.yaml` field its key reads, or a shared file it runs | the key function or the `PERFORMANCE_TOOLING`/`ACCURACY_TOOLING` list in `stage_keys.py`, the tables in [what a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns), and a test in `tests/test_compile_matrix.py` or `tests/test_stage_plan.py` |
| the set of model scripts a DevKit stage or conversion runs | `NON_COMPILE_SCRIPTS` in `stage_keys.py`; every other `.py`/`.sh` under `scripts/` is an ONNX compile input |
| board-only code | keep it inside `performance.py` or `accuracy.py`; a helper module under `scripts/` is an ONNX compile input, so editing it recompiles and then remeasures the model |
| a custom `performance.py` | start from the [canonical script](docs/benchmarking.md#the-performance-script): save the whole Core `BenchmarkReport`, including board telemetry |
| a benchmark workload or implementation | hash its declaration and model-owned script in the performance key; preserve explicit decoders and required telemetry |
| registered accuracy vocabulary or its loader | include it in accuracy tooling dependencies and retain required-metric and regression checks |
| a Core benchmark reading the registry should publish | a row with its Vulcan measurement name in `REPORT_METRICS` or `TELEMETRY_METRICS` in `benchmark_metrics.py` and the same table in the browser's `benchmark-adapters.js`, its label in the browser's `metrics.js`, its row in `DETAIL_CARDS` in `model-view.js` when a Benchmark details card should show it, and the metric table in [benchmarking](docs/benchmarking.md#workload-and-measurements) |
| a field of the performance result | `performance-result.schema.json`, `import_core_benchmark` in `core_benchmark.py`, and the browser's `benchmark-adapters.js`; a package result's `components` also feed `packageBenchmark`, which the model page and catalog read |
| a compile input outside `models/<id>/`, such as declared training code | `MODEL_COMPILE_INPUT_PATHS` in `stage_keys.py` |
| a model's category: `model_type`, `tasks`, `modalities`, or the family in `architecture.id` | pick existing IDs from `taxonomies/model-taxonomy.yaml`; when none fits, add the entry there and follow [Model categorization taxonomy](docs/model-taxonomy.md) |
| the model folder layout | `MODEL_FOLDER_ENTRIES` in `model.py` and [the directory step](docs/adding-a-model.md#2-create-the-directory) |
| a model script | the model's test of that behavior under `tests/build/` or `tests/measure/` |
| what a model converts, compiles, or measures on a DevKit | its `tests/tests.yaml` entries, their `expected` values, and the stage test files they run |
| the layout of a model's `tests/` | `TESTS_DIR_ENTRIES` and `TEST_GROUPS` in `model_tests.py`, `schemas/model-tests.schema.json`, and [the tests step](docs/adding-a-model.md#10-write-the-models-tests) |
| how a stage runs its model tests or what they report | `scripts/run_model_tests.sh`, the stage's workflow step, and `TEST_OUTCOMES` in `model_tests.py` |
| a `model-registry` command | `cli.py` (and `AUTHOR_COMMANDS` when authors run it), every workflow and script that calls it, `packaging/model-registry/install_model_registry_package.sh`, the docs, and `tests/test_cli.py` |
| a module that only CI may run | the removal list in `packaging/model-registry/build.sh` |
| a module's job, or a new module | the [module map](design-doc/model-registry-architecture.md#shared-tooling) |
| a published result's shape | its schema under `schemas/` and `artifact-browser/`, which must keep rendering results already in the registry |
| a CI job, trigger, or concurrency rule | [CI flow](design-doc/model-registry-architecture.md#ci-flow) and the [after-merge steps](docs/contributing.md#after-merge-trusted-staging-automation) |

## CI

- **Pull request.** `Target Branch Checks` is the only workflow that builds or
  tests a pull request, on the merge result and GitHub-hosted runners:
  `model-registry-package.yml` runs `python -m pytest`, collects every
  model's stage-output tests, and builds the `model-registry` package, and
  `developer-docs.yml` checks the model browser.
  `PR Sanitization` checks the pull request's description. Contributor code
  never runs on self-hosted runners for a pull request.
- **Push.** Every push runs `Vulcan Model Orchestrator`. Its `portal` job builds and smoke-tests the model browser (`developer-docs.yml`); `plan` waits for it, so a portal failure stops the run. On `main` and `develop`, `publish-model-browser` publishes that checked portal to staging. The `plan` job runs `model-registry plan-stages`, which reads the
  branch's prior catalog (its own, merged with its base branches' on a feature
  branch; see `design-doc/model-registry-architecture.md`) and resolves each
  stage key against the registry. `compile` calls
  `.github/workflows/vulcan-model-compile.yml` once per build to compile,
  run the model's compile tests, publish, and start the build's board
  stages. `publish-catalog` writes the final catalog snapshot, and `measure`
  starts the board stages a reused build still lacks.
- **Board stages.** `vulcan-model-benchmarking.yml` runs the requested
  `stages` on a DevKit, runs the model's measure tests on each result, and
  publishes performance (`model_benchmark` runs) and accuracy
  (`model_accuracy` runs) separately, each with its test outcome. A failed
  test fails the run after publication; the completed run is not measured
  again.
- Workflow YAML only orchestrates: triggers, jobs, permissions, inputs, and
  the order of steps. A step that needs more than one command calls a named,
  executable script with a usage header: under `scripts/`, or beside the code
  it serves, as `packaging/model-registry/build.sh` does. Registry logic
  belongs in a `model-registry` command. A new script under `scripts/` is not
  a stage-key input unless `stage_keys.py` lists it.
- A newer push cancels the branch's running orchestrator; builds that
  already published stay and are reused. How branch catalogs, merges, and the
  model browser interact is in
  [branches and merges](design-doc/model-registry-architecture.md#branches-merges-and-concurrent-pushes).

## Contributing

1. Declare the change in `model.yaml` and the model's `scripts/` and
   `tests/`, following [adding a model](docs/adding-a-model.md) and the
   [metadata schema](docs/metadata-schema.md). Record the source model's
   published scores in `published_metrics`.
2. Run `python -m pytest` and `model-registry validate models/<model_id>`.
3. Compile each changed build in the pinned Model Compiler with
   the backend's runner (`run_model_compilation.sh` or `run_llima_compilation.sh`)
   and check each compiled artifact with
   `model-registry verify-artifact` ([local development](docs/local-development.md)).
4. On a DevKit, run the model's standalone
   `scripts/performance.py --mpk --output`, with one `--mpk component=<archive>` per component for a
   [package benchmark](docs/benchmarking.md#package-benchmark), and, for builds that declare accuracy,
   `scripts/run_model_accuracy.sh` on the compile output.
   `scripts/run_model_performance.sh` needs a registered build, so CI runs it
   ([benchmarking](docs/benchmarking.md),
   [local accuracy](docs/local-accuracy-development.md)).
5. Report which stages you ran and where; CI publishes the results.

Conventions the code relies on:

- Model scripts receive verified local paths (`--model`, `--dataset`,
  `--fidelity-dataset`, `--mpk`); data enters only through the dataset registry
  declared in `model.yaml`. Board stage scripts run without registry
  credentials.
- Schemas under `schemas/` reject unknown fields. Published results keep their
  `schema_version` (performance 6, accuracy 2, numerical fidelity 1) because
  the artifact browser and `sima-cli` choose a renderer by it. Registry
  documents that only this repository's tooling reads (`model.yaml`, compile
  outputs, package manifests) carry no version; change their readers and
  writers together instead.

## Model browser

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
- Compare and show component measurements per component; never add them up
  or mark one component best against a different component.
- Keep rendering every published benchmark schema version, and let unknown
  metrics fall through to Other measurements rather than hiding them.
- Keep the preview data-driven. `preview.py` builds its catalog from
  `models/`; never add a model, name, or README to it by hand. It needs the
  repository installed (`python -m pip install .`) or `PYTHON` pointing at
  such an interpreter.
- Add a new module to `fingerprintedAssets` in
  `website/scripts/assemble-portal.mjs`. Assembly also writes
  `component-pipelines.<revision>.json` and `published-metrics.<revision>.json`
  from each `model.yaml`, so `build:portal` needs the same Python as
  `preview.py`.
- A new accuracy metric goes into both `taxonomies/accuracy-metrics.yaml` and
  `artifact-browser/accuracy-metrics.js`.
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
   `tools/model_registry/documentation.py`, unchanged.

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
is part of the compile key, so a comment there recompiles the model. Procedures
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

When a change alters what the model detects, its input, its outputs, or its
components, update the README and the hero in the same change. A README- or
hero-only edit is documentation or catalog metadata; it must not change
compilation, validation, benchmark, or artifact fingerprints.
