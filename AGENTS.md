# SiMa Neat Models

This repository is the source of truth for model definitions and for how each model is
compiled, benchmarked, and published. `model.yaml` declares a model, the model's `scripts/`
hold code specific to that model, and `tools/model_registry` holds the shared lifecycle.

Read [agents/rules.md](agents/rules.md) before any change. It holds the stages and
fingerprints, the rules no check catches, the limits, and where a lesson goes.

## Three trees, three readers

| Tree | Reader | Ships | It holds |
|---|---|---|---|
| `docs/` | people who reproduce, add, or update a model | as the public site | guides and references, indexed by [docs/README.md](docs/README.md) |
| `agents/` | agents that maintain this repository | never | the rules, the change map, the coding style, repository skills in `agents/skills/`, and references in `agents/references/` |
| `skills/` | customers' agents | alone, through sima-cli | customer skills for compiling and benchmarking a model |

Each tree stands alone for its reader: `docs/` links no file under `agents/` and not this
file, and `skills/` links only its own files and public URLs. Tests check both.

## Skills

Claude Code and Codex discover every skill through `.claude/skills/` and `.agents/skills/`,
which hold one link per skill folder and no copies. A new skill goes in its tree and gets a
link in both folders. A repository skill's name starts with `models-` and a customer skill's
with `sima-`, because Claude Code lets a personal skill of the same name hide a project skill.

- Repository skills: [`models-add-model`](agents/skills/models-add-model/SKILL.md) for a
  model change through to its pull request,
  [`models-review-pr`](agents/skills/models-review-pr/SKILL.md) for every review, and
  [`models-write-pr-description`](agents/skills/models-write-pr-description/SKILL.md) for a
  pull request description.
- Customer skills, used here too: [`sima-compile-model`](skills/sima-compile-model/SKILL.md)
  for operator audit, graph surgery, quantization, and compilation, and
  [`sima-benchmark-model`](skills/sima-benchmark-model/SKILL.md) for performance and accuracy
  on a DevKit.

## References

| When you | Read |
|---|---|
| change a field, script, schema, command, or workflow that other files depend on | [agents/change-map.md](agents/change-map.md) |
| write or review a model script or model test | [agents/coding-style.md](agents/coding-style.md) |
| draw or review a model's flowgraph | [agents/references/flowgraph.md](agents/references/flowgraph.md) |
| write or review the text of a model README | [agents/references/writing-model-readmes.md](agents/references/writing-model-readmes.md) |
| write or change a page under `docs/` | [agents/references/writing-docs.md](agents/references/writing-docs.md) |
| change shared tooling under `tools/` | [agents/references/architecture.md](agents/references/architecture.md) |
| change a workflow, job, trigger, or runner | [agents/references/ci-flow.md](agents/references/ci-flow.md) |
| change what a run publishes or how results are stored | [agents/references/publication.md](agents/references/publication.md) |
| add or change a registry request, or ask which endpoints this repository calls | [agents/references/registry-api.md](agents/references/registry-api.md) | the two endpoints, where each setting comes from, how a request is signed, every endpoint with its caller |
| touch branch catalogs or merge behavior | [agents/references/branch-catalogs.md](agents/references/branch-catalogs.md) |
| set `build-on-demand` or publish a candidate | [agents/references/on-demand-builds.md](agents/references/on-demand-builds.md) |
| change the LLiMa runner or deployment folder | [agents/references/llima-builds.md](agents/references/llima-builds.md) |
| promote or withdraw a build | [agents/references/promotion-operations.md](agents/references/promotion-operations.md) |
| add a section to the model page, move browser code between modules, or hide something from customers | [agents/references/model-browser.md](agents/references/model-browser.md) |
| change `artifact-browser/` or `website/` | [artifact-browser/README.md](artifact-browser/README.md), [artifact-browser/DESIGN.md](artifact-browser/DESIGN.md) |

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

## Keep these files true

Every piece of work, whether or not it ends in a pull request, leaves `AGENTS.md`, `agents/`,
`skills/`, and `docs/` true. It corrects each statement that a file, a compile output, or a
measurement proves wrong, and writes each lesson where
[where a lesson goes](agents/rules.md#where-a-lesson-goes) places it. When a statement looks
wrong and nothing proves it, report it to the user and leave the text.

`skills/sima-compile-model` and `skills/sima-benchmark-model` matter most, because customers'
agents act on them without anyone to ask. Every change that compiles, rewrites a graph, or
measures a model ends by asking what it learned about operators, graph surgery, quantization,
compilation, or measurement, and writes each answer into that skill's `references/`. A change
that learned nothing says so with the reason.

This repository is public, and `skills/` ships to customers. Write only what a reader outside
SiMa could check: this repository, a public SiMa repository or documentation page, the
installed Model Compiler's output, or a measurement on a DevKit. A lesson never contains:

- a customer's name, model, data, or request
- compiler or runtime source that is not public, including its file paths, function names,
  and internal option names
- internal hosts, addresses, runners by name, credentials, or private links
- unreleased products, versions, or plans
- a person's name

When a lesson cannot be written without one of these, leave it out and tell the user.

## Contributing

Follow [`models-add-model`](agents/skills/models-add-model/SKILL.md) for a model change and
[docs/contributing.md](docs/contributing.md) for what a pull request submits.

Before a push, run `ruff check .`, `python -m pytest`, and
`model-registry validate models/<model_id>`, and report which stages you ran and where. CI
publishes the results.

A pull request runs only `Target Branch Checks` and `PR Sanitization`, on GitHub-hosted
runners. Every push to a branch of this repository runs `Vulcan Model Orchestrator`, which
compiles and benchmarks the builds whose fingerprints changed
([CI flow](agents/references/ci-flow.md)).

## Code Review Rules

Every review, the automatic one included, follows
[`models-review-pr`](agents/skills/models-review-pr/SKILL.md) and opens with two verdicts:

1. **Shape:** whether the change follows [agents/rules.md](agents/rules.md),
   [agents/coding-style.md](agents/coding-style.md), and
   [agents/change-map.md](agents/change-map.md).
2. **Rule, skill, and knowledge updates:** the text the change should add or correct, as
   [where a lesson goes](agents/rules.md#where-a-lesson-goes) defines, or
   `Rule, skill, and knowledge updates: none` with the reason.

Report each of these as P1, because the next agent acts on the stale text:

- a pull request description without its `Rule, skill, and knowledge updates:` line
- a statement in `AGENTS.md`, `agents/`, `skills/`, or `docs/` that the change makes wrong
- a lesson the change established that no file holds

Report one finding per root cause, with the condition, the breakage, and its impact. Report
anything not run or observed as `not verified`.
