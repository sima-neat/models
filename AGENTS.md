# SiMa Neat Models

This repository is the source of truth for model definitions and for how each
model is compiled, benchmarked, and published. `model.yaml` declares a model, the
model's `scripts/` hold code specific to that model, and `tools/model_registry`
holds the shared lifecycle.

This file holds the names, the rules for every change, and an [index](#index) of
everything else. `docs/` is the public site for people who reproduce, add, or
update a model. `knowledge/` explains how the shared tooling, CI, publication,
and operations work. Open the indexed file before you work on its topic.

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
- `package_fingerprint`, the SHA-256 of the package manifest's canonical JSON
  (a Vulcan contract).
- Core's speed-only `Model.benchmark`, `BenchmarkReport`, `BenchmarkOptions`,
  and `core-benchmark-report`.

## Stages and fingerprints

| Stage | Venue | Model entry point | Runner script |
|---|---|---|---|
| compile (ONNX) | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| compile (LLiMa) | Model Compiler | `scripts/compile.sh` | `scripts/run_llima_compilation.sh` |
| performance | DevKit | `scripts/performance.py` + `requirements-performance.txt` | `scripts/run_model_performance.sh` |
| accuracy | DevKit | `scripts/accuracy.py` + `requirements-accuracy.txt` | `scripts/run_model_accuracy.sh` |

Conversion comes before the stages when a model declares it: `scripts/convert.py`,
run by `scripts/run_model_conversion.sh`, produces the ONNX that `compile.py`
receives as `--model`.

Each stage's registry run carries its fingerprint, and CI runs a stage only
when no completed run carries the current fingerprint. A change therefore
reruns exactly the stages whose inputs it touches. `compile_fingerprint` and
`benchmark_fingerprint` in `tools/model_registry/fingerprints.py` and
`conversion_fingerprint` in `tools/model_registry/conversion.py` compute them.
[What a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns)
lists their inputs. Read the code and that table before you change what a stage
depends on.

- A tooling change reruns a stage only when the changed file is in that stage's
  tooling list in `fingerprints.py`. No shared file is a conversion input.
- Raise `FINGERPRINT_VERSION` in `conversion.py` in the change that makes shared
  tooling convert the same inputs to a different model. Planning then fails
  until Vulcan Model Convert reruns for every conversion-backed model.
- For an ONNX model, every `.py` and `.sh` file under `models/<id>/scripts/` is a
  compile input, except `convert.py`, `performance.py`, `accuracy.py`, and
  `fp32_baseline_accuracy.py` (`NON_COMPILE_SCRIPTS`). Editing a comment in a
  compile input recompiles and then remeasures the model.
- No fingerprint reads a model's `README.md`, `hero.png`, `tests/`, `license`,
  the taxonomy, `docs/`, `knowledge/`, or this file. Editing them reruns nothing.

## Rules no check catches

Pull request CI passes when one of these is broken, and the failure shows up
later or never. Apply them while you write.

- **DevKit tests import lazily.** Tests under `tests/benchmark/` also run on the
  DevKit, in PyNeat, which has no `onnx`, `onnxruntime`, or `torch`. Import such
  a package inside the test with `pytest.importorskip("onnx")`. A module-level
  import stops collection on the DevKit, and the stage publishes no result
  ([the tests step](docs/adding-a-model.md#10-write-the-models-tests)).
- **`convert.py` exports, `surgery.py` rewrites.** `convert.py` exports the
  upstream forward pass at the shipped input shape, from its input to the final
  outputs the package keeps. `onnxsim.simplify`, including constant folding, is
  allowed there when it preserves the upstream inputs, outputs, and computation.
  Every other change for the compiler belongs in `surgery.py`, even an exact
  one. That includes a cut: an export that ends at an inner tensor, such as raw
  heads before the upstream decode, or starts after an upstream step
  ([what `convert.py` exports](docs/conversion-artifacts.md#what-convertpy-exports)).
- **Read conversion output names from `model.yaml`.** When a model declares
  `conversion.outputs`, every script reads the names with
  `conversion_outputs()`, so renaming a file changes only `model.yaml`.
- **Board-only code stays in `performance.py` or `accuracy.py`.** A helper
  module under `scripts/` is a compile input. Put maintainer explanations in
  `model.yaml` comments or in the docstrings of `convert.py`, `performance.py`,
  and `accuracy.py`.
- **Benchmark scripts stand alone.** `performance.py` and `accuracy.py` run
  with `tools/model_report` on `PYTHONPATH`, without `model_registry` imports,
  YAML configuration, registry credentials, or downloads. They take verified
  local paths: `--mpk`, `--dataset`, `--output`, and `--resources` when the
  model declares supporting files.
- **Pin sources by content.** Prefer a Hugging Face checkpoint pinned to a full
  repository commit. Use an official upstream GitHub release asset only as a
  fallback, with its SHA-256 and byte count, and explain the fallback in a
  comment in `model.yaml`.
- **Hide an unready variant by commenting it out.** A commented-out profile in
  `model.yaml` is neither compiled nor benchmarked, and its tests skip.
- **An LLM or VLM is one component.** Declare the complete model as one
  component with one 4K profile (`max_num_tokens: 4096` and
  `language_group_size`). Keep graph-changing options in component
  `parameters`. Leave generated layers out of `components`, and ship the
  deployment folder as files, never as a nested archive. A speculative pair is
  still one component: deploy and publish the target and its draft together.
- **Approval comes from the server.** Read `approved_deployments` and
  `branch_deployments` as Vulcan returns them. Never write or infer one.
- **Workflow YAML only orchestrates.** It holds triggers, jobs, permissions,
  inputs, and the order of steps. A step that needs more than one command calls
  a named, executable script with a usage header, under `scripts/` or beside the
  code it serves. Registry logic belongs in a `model-registry` command.
- **Expected values follow published results.** When a reviewed change moves a
  published value, set `expected` in `tests/tests.yaml` to the new published
  result. Leave `ACCURACY_ALLOWED_DROP` and `PERFORMANCE_ALLOWED_DROP` alone:
  every model shares them.
- **Schemas reject unknown fields.** Published reports keep their
  `schema_version` (model report 1, numerical fidelity 1), because the model
  browser and `sima-cli` choose a renderer by it. Documents only this
  repository reads (`model.yaml`, compile outputs, package manifests) carry no
  version. Change their readers and writers together.

## Keep the structure consistent

Each row is a change that is incomplete without the files beside it. Make them
in the same commit.

| When you change | Also change |
|---|---|
| what a stage depends on: a `model.yaml` field its fingerprint reads, or a shared file it runs | the fingerprint function or the `COMPILE_TOOLING`/`BOARD_STAGE_TOOLING`/`PERFORMANCE_TOOLING`/`ACCURACY_TOOLING` list in `fingerprints.py`, the tables in [what a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns), and a test in `tests/test_compile_matrix.py` or `tests/test_stage_plan.py` |
| an approved source host | its entry in `HTTPS_HOST_CHECKS` in `source.py`, accept and reject cases in `tests/test_source.py`, and the host table in [pin every input](docs/adding-a-model.md#3-pin-every-input). This reruns no stage |
| the set of model scripts a benchmark or conversion runs | `SCRIPTS` in `benchmark_declarations.py` or `CONVERSION_FILES` in `model.py`, which make up `NON_COMPILE_SCRIPTS` in `fingerprints.py` with `FP32_BASELINE_SCRIPT` |
| how `performance.py` gets its input | `scripts/run_model_performance.sh`, the dataset steps in `vulcan-model-benchmarking.yml`, [Benchmark input](docs/benchmarking.md#benchmark-input), and `tests/test_model_report.py` and `tests/test_benchmark_workflow.py` |
| what a package holds besides MPKs: `resources` (pinned files with no `main_file`) or a compile's `runtime_files` | `package.py`, `candidates.py` and `runtime_files.py`, `validate_compile_outputs` in `artifact.py`, the compile fingerprint in `fingerprints.py`, `source_identity` in `source.py`, `resourceSource` in `model.schema.json`, [Packages and pipeline](docs/metadata-schema.md#packages-and-pipeline), and `tests/test_package.py`, `tests/test_artifact.py`, `tests/test_compile_matrix.py` and `tests/test_source.py` |
| a custom `performance.py` | start from the [canonical script](docs/benchmarking.md#the-performance-script), and for a multi-component model the [multi-component script](docs/benchmarking.md#multi-component-models): hand Core's `BenchmarkReport`, board telemetry, and each component's Model to `model_report.from_performance()` |
| a metric: a Core benchmark reading or an accuracy score the registry should publish | a row in `tools/model_report/metrics.json` with its Vulcan measurement name (`registry_name`), and its entry in the browser's `metrics.js` or `accuracy-metrics.js`, and for a telemetry reading in `TELEMETRY_METRICS` in `model-report.js`, which tests keep in step; for a performance metric, its row in `DETAIL_CARDS` in `performance-section.js` when a Performance details card should show it, and the metric table in [benchmarking](docs/benchmarking.md#workload-and-measurements) |
| a `datasets/<id>.yaml` requirement | `dataset-requirements.schema.json` when its shape changes, and the `accuracy.py` and `tests.yaml` expectations of every model with that task that evaluates the dataset |
| a field of the model report | `schemas/model-report.schema.json`, `tools/model_report/__init__.py`, the checks in `report_publication.py`, and the browser code that reads the report |
| a model's category: `model_type`, `tasks`, `modalities`, or the family in `architecture.id` | pick existing IDs from `taxonomies/model-taxonomy.yaml`; when none fits, add the entry there, with a `description` for a new architecture, and follow [Model categorization taxonomy](docs/model-taxonomy.md) |
| a model's `license` | set `spdx` to an ID in `artifact-browser/licenses.json`. When the upstream license is not listed, add it there, with `downloadable: false` if it forbids redistribution ([license](docs/metadata-schema.md#license)) |
| a downloadable ONNX model, its `upstream` pin, or its license | its `license` resource: the upstream license file pinned by commit and installed under `licenses/`, one resource per license the weights carry, and `--resources` in its `performance.py`. A resource is a compile input, so the pin recompiles the model ([license files in the package](docs/metadata-schema.md#license-files-in-the-package)) |
| the model folder layout | `MODEL_FOLDER_ENTRIES` in `model.py` and [the directory step](docs/adding-a-model.md#2-create-the-directory) |
| a model script | the model's test of that behavior under `tests/build/` or `tests/benchmark/` |
| what a model converts, compiles, or measures on a DevKit | its `tests/tests.yaml` entries, their `expected` values, and the stage test files they run |
| what a model detects, its input, its outputs, or its components | its `README.md` and `hero.png` ([model README](docs/model-readme.md)) |
| a README rule or the fixed `BUILDING_APPLICATIONS` or `GENAI_BUILDING_APPLICATIONS` text in `documentation.py` | [model README](docs/model-readme.md), which quotes the rules and both texts, and the `README.md` of every model the change affects |
| the layout of a model's `tests/` | `TESTS_DIR_ENTRIES` and `TEST_GROUPS` in `model_tests.py`, `schemas/model-tests.schema.json`, and [the tests step](docs/adding-a-model.md#10-write-the-models-tests) |
| how a stage runs its model tests or what they report | `scripts/run_model_tests.sh`, the stage's workflow step, and `TEST_OUTCOMES` in `model_tests.py` |
| a `model-registry` command | `cli.py` (and `AUTHOR_COMMANDS` when authors run it), every workflow and script that calls it, `packaging/model-registry/install_model_registry_package.sh`, the docs, and `tests/test_cli.py` |
| a module that only CI may run | the removal list in `packaging/model-registry/build.sh` |
| a module's job, or a new module | the module map in [architecture](knowledge/architecture.md) |
| a published result's shape | its schema under `schemas/` and `artifact-browser/` |
| a CI job, trigger, or concurrency rule | [CI flow](knowledge/ci-flow.md), and the [after-merge section](docs/contributing.md#after-merge-trusted-staging-automation) when the author sees a different outcome |
| a module under `artifact-browser/` | `fingerprintedAssets` in `website/scripts/assemble-portal.mjs`, the module table in `artifact-browser/README.md`, and [the model browser page](docs/model-browser.md) for a visible change |
| a file under `docs/` or `knowledge/`, or the topics one covers | its row in the [index](#index), and for a docs page `website/sidebars.ts` |

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
5. Report which stages you ran and where. CI publishes the results.

A pull request runs only `Target Branch Checks` and `PR Sanitization`, on
GitHub-hosted runners. Every push to a branch of this repository runs
`Vulcan Model Orchestrator`, which compiles and benchmarks the builds whose
fingerprints changed ([CI flow](knowledge/ci-flow.md)).

## Index

Each row names a file, when to open it, and what it holds. `docs/` pages are
written for people and hold the contributor rules. `knowledge/` files hold what
the code and workflows do.

| When you | Read | It holds |
|---|---|---|
| need the repository's purpose, or the whole path from official model to measured results | [docs/getting-started.md](docs/getting-started.md) | each step in order with its command: datasets, convert, compile, performance, accuracy |
| meet a term such as build profile, model package, or fingerprint | [docs/concepts.md](docs/concepts.md) | plain definitions with one example each, and the stage table |
| add a model or change its folder | [docs/adding-a-model.md](docs/adding-a-model.md) | folder layout, input pinning, compile recipe, model tests and `tests.yaml`, supporting files, what a change reruns, LLiMa definitions |
| set or change a `model.yaml` field | [docs/metadata-schema.md](docs/metadata-schema.md) | every field with its default, packages and pipeline, resources, license |
| write `performance.py` or performance expectations | [docs/benchmarking.md](docs/benchmarking.md) | the canonical script, benchmark input, multi-component and per-component reports, the metric table, LLiMa performance |
| write `accuracy.py` or an FP32 baseline | [docs/local-accuracy-development.md](docs/local-accuracy-development.md) | running accuracy on a DevKit, dataset arguments, the FP32 baseline procedure |
| compile or verify a build locally | [docs/local-development.md](docs/local-development.md) | tool installation, validation commands, calling `compile.py`, `verify-mpk`, the DevKit benchmark run, common errors |
| change `convert.py`, `surgery.py`, or `conversion.outputs` | [docs/conversion-artifacts.md](docs/conversion-artifacts.md) | what `convert.py` exports and what belongs in `surgery.py`, checkpoint outputs, what reruns a conversion |
| pin a dataset or edit `datasets/<id>.yaml` | [docs/datasets.md](docs/datasets.md) | Dataset Registry access and pins, registering a dataset, per-task required metrics |
| write a model README or hero | [docs/model-readme.md](docs/model-readme.md), [docs/illustration-guide.md](docs/illustration-guide.md) | required sections, word limits, banned terms, the template, hero format, styles, palette, and prompts |
| write or review the text of a model README | [knowledge/writing-model-readmes.md](knowledge/writing-model-readmes.md) | who reads a README, the question each part answers, the opening and uses for each kind of model, the Neat-first rule, family consistency, the checks before merge |
| set a model's type, task, or family | [docs/model-taxonomy.md](docs/model-taxonomy.md) | existing category IDs and how to add one |
| need a different CI runner for a model | [docs/runner-class-requests.md](docs/runner-class-requests.md) | runner classes and how to request one |
| prepare a pull request | [docs/contributing.md](docs/contributing.md) | what to submit, what a pull request runs, what the author sees after merge |
| ask for a build to be promoted | [docs/model-promotion.md](docs/model-promotion.md) | what the requester provides and what promotion changes |
| describe what a visitor sees in the model browser | [docs/model-browser.md](docs/model-browser.md) | the page's views, filters, model page sections, and downloads |
| write or change a page under `docs/` | [knowledge/writing-docs.md](knowledge/writing-docs.md) | the two page kinds, step and language rules, the checks before merge |
| change shared tooling under `tools/` | [knowledge/architecture.md](knowledge/architecture.md) | component boundaries, the directory contract, which module owns which job |
| change a workflow, job, trigger, or runner | [knowledge/ci-flow.md](knowledge/ci-flow.md) | the pull request and push flow job by job, concurrency, the trust boundary |
| change what a run publishes or how results are stored | [knowledge/publication.md](knowledge/publication.md) | registry runs, compile outputs, package manifests, reports, catalog snapshots, conversion retention |
| touch branch catalogs or merge behavior | [knowledge/branch-catalogs.md](knowledge/branch-catalogs.md) | how a branch catalog inherits, what a merge does, concurrent pushes |
| set `build-on-demand` or publish a candidate | [knowledge/on-demand-builds.md](knowledge/on-demand-builds.md) | scheduling, manual dispatch, candidate publication, approvals and branch deployments |
| change the LLiMa runner or deployment folder | [knowledge/llima-builds.md](knowledge/llima-builds.md) | what the runner downloads, compiles, and deploys, the deployment manifest, the LLiMa fingerprint |
| promote or withdraw a build | [knowledge/promotion-operations.md](knowledge/promotion-operations.md) | the maintainer runbook |
| add a section to the model page, move browser code between modules, or hide something from customers | [knowledge/model-browser.md](knowledge/model-browser.md) | which module draws what, the import rule for page sections, the audience facts in `audience.js`, and the steps to hide a part from the customer view |
| change `artifact-browser/` or `website/` | [artifact-browser/README.md](artifact-browser/README.md), [artifact-browser/DESIGN.md](artifact-browser/DESIGN.md) | the code map, the rules for changing the browser, checks before a push, deployment, design context |
