# Architecture

What this repository owns, what each folder holds, and which module owns which job. Read
it before you add shared code, move a file, or change what a stage depends on. How CI runs
the stages is in [CI flow](ci-flow.md). What a run records is in
[publication](publication.md).

## Component boundaries

This repository holds reviewable inputs: model definitions, schemas, shared tooling, runner
scripts, workflows, and the model browser. It stores no ONNX file, dataset, MPK, log, or
result. `tests/test_target_branch_checks.py` rejects a committed file over 2 MiB. Compiled
artifacts and results live in the Vulcan registry, which records each object's storage key,
version, size, and SHA-256.

| Component | Owns | Where this repository meets it |
| --- | --- | --- |
| This repository | `models/<model_id>/` definitions and model code, `schemas/`, the shared tooling in `tools/`, the runner scripts in `scripts/`, the workflows, and the model browser | |
| Vulcan | The artifact registry API (runs, artifacts, measurements, catalog snapshots, deployments), object storage, the self-hosted Model Compiler and DevKit runners, and the reusable workflows in `sima-neat/.github` | `registry_http.py` signs each request and sends an idempotency key. The registry owns the catalog fields `approved_deployments` and `branch_deployments` (`SERVER_OWNED_SELECTIONS` in `registry.py`). Its environment specification holds the runner approvals that `authorize-runner-matrix` reads. |
| Dataset Registry (`sima-neat/dataset`) | Dataset identities, archives, and the `dataset-registry` resolver | `model.yaml` pins each archive by ID, variant, SHA-256, and size. `resolve_dataset` in `fetch.py` calls `dataset-registry resolve` and verifies both values. `datasets/<id>.yaml` here holds only the accuracy metrics each task must report. |
| Model Compiler | Quantization, compilation, and the `mla-msim` simulator | `deps/manifest.json` pins one version for every model. `resolve_model_compiler` in `dependencies.py` derives its identity and runner label. A model's `scripts/compile.py` calls the compiler. |
| Core and PyNeat | Loading an MPK, running it on a DevKit, and the benchmark API (`Model.benchmark`, `BenchmarkReport`) | `deps/manifest.json` names the package in `neat-core.install_target`, and `neat_core.py` installs it before a benchmark. A model's `performance.py` and `accuracy.py` call PyNeat and hand the results to `tools/model_report`. |
| LLiMa | `llima-compile`, `llima-deploy`, the LLM and VLM runtime, and the MoLE benchmark | [LLiMa builds](llima-builds.md) |
| `sima-cli` | Package installation on hosts and DevKits | `packaging/model-registry/build.sh` builds the author package with `sima-cli packages build`. CI installs Core and the Dataset Registry wheel with `sima-cli neat install`. |

Three rules follow from these boundaries:

- Model-specific code lives in `models/<model_id>/scripts/`. Download, checksum, cache,
  validation, report, and publication code lives in `tools/`.
- A stage runs the same way locally and in CI, because both call the same runner script in
  `scripts/`.
- A model's `performance.py` and `accuracy.py` import `tools/model_report` and never
  `model_registry`. The report writer in `model_report/__init__.py` needs only the Python
  standard library and PyNeat, so it runs inside PyNeat's environment.
  `generation_benchmark.py` also imports the MoLE environment's packages.

The product name is Model Browser. Commands, code, and paths keep `model-registry` and
`model_registry`. "Registry" alone names the Vulcan service that stores runs and artifacts.

## Directory contract

```text
.github/workflows/         every workflow (ci-flow.md)
agents/                    for agents working in this repository, never shipped: rules.md,
                           change-map.md, coding-style.md, skills/, and references/, which
                           explains tooling, CI, publication, and operations
artifact-browser/          the model browser app, preview.py, licenses.json, fixtures
config/                    model-browser-deployment-environment.json: protection settings of
                           the GitHub environment that gates manual browser deployment
datasets/                  <dataset id>.yaml: the accuracy metrics each task must report
deps/                      manifest.json: platform, Model Compiler, Core, and LLiMa pins
docs/                      the published site, for people who reproduce or add a model
models/<model_id>/         model.yaml, README.md, hero.png, scripts/, tests/
packaging/model-registry/  build, installer, and checks for the author package
schemas/                   JSON Schemas: model, model tests, dataset requirements, package
                           manifest, model report, numerical fidelity, MLA stage
                           simulation, licenses
scripts/                   one runner script per stage, and CI helper scripts
skills/                    skills for customers' agents, which sima-cli ships alone
taxonomies/                model-taxonomy.yaml: model categories and display labels
tests/                     repository-wide tests, browser tests in tests/browser/, fixtures
tools/model_registry/      the model-registry command and the shared lifecycle code
tools/model_report/        the report writer and metric vocabulary used on the DevKit
website/                   the Docusaurus site and the scripts that assemble the portal
```

Root files: `pyproject.toml` installs `tools/` as the `model-registry` command.
`conftest.py` holds the fixtures of each model's tests. `naming-conventions.yaml` is an
author reference for build profile IDs and MPK names, and no code loads it. `state.json`
is the registry release state ([CI flow](ci-flow.md#registry-release-state)).

A model folder may hold only the entries in `MODEL_FOLDER_ENTRIES` (`model.py`), and
`validate` rejects any other entry. A model's `tests/` may hold only `TESTS_DIR_ENTRIES`
(`model_tests.py`).

## Module map

### `tools/model_registry`

One module per job. Put a new operation in the module that owns its job. Give a new job a
new module named after what it does, and add a row here. `build.sh` removes the modules
marked CI only from the author package.

| Module | Owns |
| --- | --- |
| `cli.py` | The `model-registry` command. `AUTHOR_COMMANDS` lists the commands the author package exposes. |
| `model.py` | Loading and validating `model.yaml` and the model folder (`load_models`). Expanding it into components, build profiles with their one compile invocation, and the `default` package. MPK file names (`mpk_filename`). |
| `compile_matrix.py` | The compile matrix: one entry per compile invocation, with its compile fingerprint and the benchmark stages that measure it (`build_matrix`). |
| `fingerprints.py` | `compile_fingerprint`, `benchmark_fingerprint`, the shared files each one hashes, and each benchmark's run type. |
| `stage_plan.py` | Which builds compile and which benchmarks run (`plan_stages`), the registry reads that decision needs, base-branch resolution, the merge of source catalogs, and the check that `main` can take a merge from `develop` without compiling (`find_main_merge_blockers`). |
| `conversion.py` | The conversion fingerprint, declared outputs, bundles, conversion publication, the saved selection a plan records, and exact downloads. |
| `registry_http.py` | The signed HTTP transport, and registry reads that sign only the configured IAM API. |
| `registry.py` (CI only) | Publishing compile runs, packages with their fidelity report and MLA stage simulation, model documentation, and the branch catalog. Restoring a published build for a benchmark. |
| `candidates.py` (CI only) | On-demand candidates: publication, run binding, finalization, and selection in a feature catalog. |
| `promotions.py` (CI only) | Promotion requests, decisions, receipts, and the benchmark reference refresh after a promotion. |
| `report_publication.py` (CI only) | Checking a `model-report.json` against `model.yaml` (`check_report`) and publishing one benchmark run (`publish_report`). |
| `neat_core.py` (CI only) | Resolving the selected Core artifact and installing it only when the installed one differs. |
| `benchmark_declarations.py` | The `performance` and `accuracy` declarations: benchmarked profiles, stage scripts, required metrics, `reference_metrics` validation, and the dataset names the model browser shows. |
| `artifact.py` | `compile-outputs.json` validation, MPK and LLiMa deployment verification, the LLiMa reproduction recipe, and deployment file indexes. |
| `runtime_files.py` | Path and checksum validation of a compile's `runtime_files`, and their restoration from a package ZIP. |
| `mla_stage_simulation.py` | The simulator's operation counts and latency for each MLA stage (`simulate-mla-stages`). |
| `package.py` | Assembling a profile's `default` package from MPKs, pinned `resources`, and `runtime_files`, or from a complete LLiMa deployment folder. |
| `dependencies.py` | Resolving the Model Compiler and DevKit runtime pins in `deps/manifest.json`. |
| `fetch.py` | Verified downloads, the cache, dataset resolution, and unsigned registry reads. |
| `source.py` | The approved source hosts (`HTTPS_HOST_CHECKS`), source identity, and URL redaction. |
| `taxonomy.py` | Loading the taxonomy and validating a model's categories. |
| `documentation.py` | Loading and validating a model's README and hero image, and the example application page the model browser links. |
| `runner_approval.py` | Authorizing requested compile runner classes and markets, and selecting the runner audit fields a compile run records. |
| `model_tests.py` | Each model's test files and `tests/tests.yaml`, the comparison of a report with its expected values, the test outcomes, and the variants the model browser hides for low accuracy. |
| `errors.py` | The error types every module raises. |

`__init__.py` exports the loaders and error types, `__main__.py` runs `cli.main`, and
`py.typed` marks the package as typed. The author package replaces `__main__.py` with
`packaging/model-registry/__main__.py`, which runs `cli.author_main`. Its installer finds
the Model Compiler's Python and installs the wheel there with `--no-deps`, so it replaces
no compiler dependency.

### `tools/model_report`

| Module | Owns |
| --- | --- |
| `__init__.py` | Building `model-report.json` from what Core returned: `from_performance`, `from_component_performance`, `from_accuracy`, and `from_generation` for LLiMa. |
| `board_telemetry.py` | Sentinel traces around measured inference, summarized like Core's telemetry. |
| `generation_benchmark.py` | MoLE timings, dataset comparisons, and telemetry for deployed LLiMa models. |
| `speech.py` | Scoring speech recognition transcripts: one text normalization, then corpus word and character error rates (`wer`, `cer`). Hashed only into the accuracy fingerprints of models that must report them. |
| `metrics.json` | The one metric vocabulary: each metric's stage, label, unit, direction, Vulcan measurement name (`registry_name`), and whether a performance report requires it. |

## Stages and fingerprints

A build is one build profile of one model, compiled in one compile invocation named after
the profile. It passes through up to three stages. Compile runs in the Model Compiler
environment, and performance and accuracy run on a DevKit. Conversion runs before them, on
request.

| Stage | Runner script | Model entry point | Fingerprint | Registry record |
| --- | --- | --- | --- | --- |
| conversion | `scripts/run_model_conversion.sh` | `scripts/convert.py` | `conversion_fingerprint` in `conversion.py` | one `model_conversion` run |
| compile (ONNX) | `scripts/run_model_compilation.sh` | `scripts/compile.py` | `compile_fingerprint` in `fingerprints.py` | one `model_compile` run per component and one `model_package` run per profile |
| compile (LLiMa) | `scripts/run_llima_compilation.sh` | `scripts/compile.sh` | `compile_fingerprint`, LLiMa branch | the same |
| performance | `scripts/run_model_performance.sh` | `scripts/performance.py` | `benchmark_fingerprint` | one `model_performance` run under the package run |
| accuracy | `scripts/run_model_accuracy.sh` | `scripts/accuracy.py` | `benchmark_fingerprint` | one `model_accuracy` run under the package run |

A fingerprint is a SHA-256 of every input that decides a stage's result. CI runs a stage
only when the registry holds no result for the current fingerprint
([branch catalogs](branch-catalogs.md#what-a-push-compiles-and-benchmarks)). The code is
the truth for what each fingerprint hashes:

| Fingerprint | Built from |
| --- | --- |
| ONNX compile | The profile's projection of `model.yaml` (sources, parameters, calibration pin, `numerical_fidelity`, `resources`), the compiler identity, the files in `COMPILE_TOOLING`, and every `.py` and `.sh` file under the model's `scripts/` that is not in `NON_COMPILE_SCRIPTS`. |
| LLiMa compile | Checkpoint content, target, the component's `parameters`, and the compiler identity. No script or shared file. |
| benchmark | Each component's compile run ID, plus its MPK SHA-256 for an ONNX build. The stage's `model.yaml` declaration without `build_profiles`. For accuracy it includes the metrics each dataset's `datasets/<id>.yaml` requires for the model's tasks. The stage script and its optional requirements file. `PERFORMANCE_TOOLING` or `ACCURACY_TOOLING`, plus `LLIMA_PERFORMANCE_TOOLING` for LLiMa performance. The DevKit runtime pins. A performance fingerprint adds the profile's `benchmark_baseline` with that baseline's compile fingerprint. |
| conversion | `conversion.source`, `conversion.outputs`, `FINGERPRINT_VERSION`, and the model's conversion files under `scripts/`. No shared file. |

[What a change reruns](../../docs/adding-a-model.md#9-know-what-a-change-reruns) lists each
kind of change and the stages it reruns. Keep it in step with the code.

Rules for changing what a stage depends on:

- A shared file reruns a stage only when it is in that stage's tooling list in
  `fingerprints.py`. Add the file there, update the docs table, and add a test in
  `tests/test_compile_matrix.py` or `tests/test_stage_plan.py`.
- A new script under the repository's `scripts/` is an input of no stage until a tooling
  list names it. Add it to the list of each stage that runs it.
- A `compilation.source: conversion` source resolves during planning
  ([selection](publication.md#selection)). The compile fingerprint hashes only the
  resolved file's SHA-256 (`source_identity` in `source.py`), never the run that produced
  it. A conversion that reruns and yields the same ONNX therefore recompiles nothing.
- A recompiled build has new compile run IDs, so its benchmark fingerprints change and its
  benchmarks run again.
- No fingerprint includes a model's `tests/`, `tools/model_report/metrics.json`, the
  license, `reference_metrics`, `build-on-demand`, or the runner class and market.
- Keep the conversion fingerprint free of shared files. A fingerprint with no published
  conversion fails `plan-stages`, and with it the whole run, until someone runs Vulcan
  Model Convert for that model. When shared tooling changes converted output, raise
  `FINGERPRINT_VERSION` in the same change.
- `scripts/require_source_board_tooling.sh` reads the same tooling lists. It stops a manual
  benchmark whose branch changed that tooling after the measured revision.
