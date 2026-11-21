# SiMa Neat Models

This repository is the source of truth for model definitions and for how each model is
compiled, benchmarked, and published. `model.yaml` declares a model, the model's `scripts/`
hold code specific to that model, and `tools/model_registry` holds the shared lifecycle.

Each tree has one reader. `docs/` is the public site for people who reproduce, add, or update
a model. `agents/` serves agents working in this repository and never ships. `skills/` serves
customers' agents, and sima-cli ships it alone. Open the indexed file before you work on its
topic.

Read [agents/rules.md](agents/rules.md) before any change, because it holds the rules no check
catches. It also holds the stages and fingerprints, the limits, and where a lesson goes.

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

## Index

| When you | Read | It holds |
|---|---|---|
| change a field, script, schema, command, or workflow that other files depend on | [agents/change-map.md](agents/change-map.md) | each change with the files that change with it in the same commit |
| write or review a model script or model test | [agents/coding-style.md](agents/coding-style.md) | how `convert.py`, `surgery.py`, `compile.py`, `calibration.py`, `performance.py`, `accuracy.py`, and model tests read |
| add or update a model, through to its pull request | [agents/skills/add-model/SKILL.md](agents/skills/add-model/SKILL.md) | the agent workflow for a model change |
| review a pull request | [agents/skills/review-pr/SKILL.md](agents/skills/review-pr/SKILL.md) | the review checks and the two verdicts every review reports |
| write or update a pull request description | [agents/skills/write-pr-description/SKILL.md](agents/skills/write-pr-description/SKILL.md) | the description's sections, including `Rule, skill, and knowledge updates` |
| draw or review a model's flowgraph | [agents/references/flowgraph.md](agents/references/flowgraph.md) | the drawing rules for a model's flowgraph |
| audit operators, rewrite or optimize a graph, quantize, or compile a model | [skills/neat-compile-model/SKILL.md](skills/neat-compile-model/SKILL.md) | the customer skill for operator audit, graph surgery, graph optimization, quantization, and compilation |
| measure a model's performance or accuracy on a DevKit | [skills/neat-benchmark-model/SKILL.md](skills/neat-benchmark-model/SKILL.md) | the customer skill for benchmarks on a DevKit |
| need the repository's purpose, or the whole path from official model to measured results | [docs/getting-started.md](docs/getting-started.md) | each step in order with its command: datasets, convert, compile, performance, accuracy |
| download a published model, or describe the `sima-cli models` commands | [docs/downloading-models.md](docs/downloading-models.md) | finding a model, the model card, downloading a variant, choosing a branch, the printed errors |
| meet a term such as build profile, model package, or fingerprint | [docs/concepts.md](docs/concepts.md) | plain definitions with one example each, and the stage table |
| add a model or change its folder | [docs/adding-a-model.md](docs/adding-a-model.md) | folder layout, input pinning, compile recipe, model tests and `tests.yaml`, supporting files, what a change reruns, LLiMa definitions |
| set or change a `model.yaml` field | [docs/metadata-schema.md](docs/metadata-schema.md) | every field with its default, packages and pipeline, resources, license |
| write `performance.py` or performance expectations | [docs/benchmarking.md](docs/benchmarking.md) | the canonical script, benchmark input, multi-component and per-component reports, the metric table, LLiMa performance |
| write `accuracy.py` or an FP32 baseline | [docs/local-accuracy-development.md](docs/local-accuracy-development.md) | running accuracy on a DevKit, dataset arguments, the FP32 baseline procedure |
| compile or verify a build locally | [docs/local-development.md](docs/local-development.md) | tool installation, validation commands, calling `compile.py`, `verify-mpk`, the DevKit benchmark run, common errors |
| change `convert.py`, `surgery.py`, or `conversion.outputs` | [docs/conversion-artifacts.md](docs/conversion-artifacts.md) | what `convert.py` exports and what belongs in `surgery.py`, checkpoint outputs, what reruns a conversion |
| pin a dataset or edit `datasets/<id>.yaml` | [docs/datasets.md](docs/datasets.md) | Dataset Registry access and pins, registering a dataset, the name the browser shows, per-task required metrics |
| write a model README or hero | [docs/model-readme.md](docs/model-readme.md), [docs/illustration-guide.md](docs/illustration-guide.md) | required sections, word limits, banned terms, the template, hero format, styles, palette, and prompts |
| write or review the text of a model README | [agents/references/writing-model-readmes.md](agents/references/writing-model-readmes.md) | who reads a README, the question each part answers, the opening and uses for each kind of model, the four parts of Inputs and Outputs, family consistency, the checks before merge |
| set a model's type, task, or family | [docs/model-taxonomy.md](docs/model-taxonomy.md) | existing category IDs and how to add one |
| need a different CI runner for a model | [docs/runner-class-requests.md](docs/runner-class-requests.md) | runner classes and how to request one |
| prepare a pull request | [docs/contributing.md](docs/contributing.md) | what to submit, what a pull request runs, what the author sees after merge |
| ask for a build to be promoted | [docs/model-promotion.md](docs/model-promotion.md) | what the requester provides and what promotion changes |
| describe what a visitor sees in the model browser | [docs/model-browser.md](docs/model-browser.md) | the page's views, filters, model page sections, and downloads |
| write or change a page under `docs/` | [agents/references/writing-docs.md](agents/references/writing-docs.md) | the two page kinds, step and language rules, the checks before merge |
| change shared tooling under `tools/` | [agents/references/architecture.md](agents/references/architecture.md) | component boundaries, the directory contract, which module owns which job |
| change a workflow, job, trigger, or runner | [agents/references/ci-flow.md](agents/references/ci-flow.md) | the pull request and push flow job by job, concurrency, the trust boundary |
| change what a run publishes or how results are stored | [agents/references/publication.md](agents/references/publication.md) | registry runs, compile outputs, package manifests, reports, catalog snapshots, conversion retention |
| touch branch catalogs or merge behavior | [agents/references/branch-catalogs.md](agents/references/branch-catalogs.md) | how a branch catalog inherits, what a merge does, concurrent pushes |
| set `build-on-demand` or publish a candidate | [agents/references/on-demand-builds.md](agents/references/on-demand-builds.md) | scheduling, manual dispatch, candidate publication, approvals and branch deployments |
| change the LLiMa runner or deployment folder | [agents/references/llima-builds.md](agents/references/llima-builds.md) | what the runner downloads, compiles, and deploys, the deployment manifest, the LLiMa fingerprint |
| promote or withdraw a build | [agents/references/promotion-operations.md](agents/references/promotion-operations.md) | the maintainer runbook |
| add a section to the model page, move browser code between modules, or hide something from customers | [agents/references/model-browser.md](agents/references/model-browser.md) | which module draws what, the import rule for page sections, the audience facts in `audience.js`, and the steps to hide a part from the customer view |
| change `artifact-browser/` or `website/` | [artifact-browser/README.md](artifact-browser/README.md), [artifact-browser/DESIGN.md](artifact-browser/DESIGN.md) | the code map, the rules for changing the browser, checks before a push, deployment, design context |

## Contributing

Follow [agents/skills/add-model/SKILL.md](agents/skills/add-model/SKILL.md) for a model
change and [docs/contributing.md](docs/contributing.md) for what a pull request submits.

Before a push, run `ruff check .`, `python -m pytest`, and
`model-registry validate models/<model_id>`, and report which stages you ran and where. CI
publishes the results.

A pull request runs only `Target Branch Checks` and `PR Sanitization`, on GitHub-hosted
runners. Every push to a branch of this repository runs `Vulcan Model Orchestrator`, which
compiles and benchmarks the builds whose fingerprints changed
([CI flow](agents/references/ci-flow.md)).

## Every review reports

The automatic review and every reviewer follow
[agents/skills/review-pr/SKILL.md](agents/skills/review-pr/SKILL.md) and report two verdicts:

1. **Shape:** whether the change follows [agents/rules.md](agents/rules.md),
   [agents/coding-style.md](agents/coding-style.md), and
   [agents/change-map.md](agents/change-map.md).
2. **Rule, skill, and knowledge updates:** the text the change should add or correct, as
   [where a lesson goes](agents/rules.md#where-a-lesson-goes) defines, or
   `Rule, skill, and knowledge updates: none` with the reason.

Report one finding per root cause, with the condition, the breakage, and its impact. Report
anything not run or observed as `not verified`.
