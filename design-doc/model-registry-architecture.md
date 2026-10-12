# Model Registry architecture

The Model Registry repository owns small, reviewable lifecycle inputs:

- versioned JSON Schemas and policy;
- `models/<model-id>/model.yaml` definitions, model-local documentation, and
  model-owned scripts;
- shared lifecycle tooling and the runner scripts for each stage;
- CI orchestration and tests.

It does not store ONNX files, datasets, compiled model packs (MPKs), logs, or
results. Those remain in immutable object storage and are referenced by URI,
byte size, and SHA-256.

## Component boundaries

| Component | Responsibility |
| --- | --- |
| Model-owned `scripts/compile.py` | Reads its build profile from `model.yaml`, emits one MPK per compiled component, and writes the declared fidelity report. |
| Model-owned `scripts/performance.py` | Benchmarks one MPK with Neat Core on a Modalix DevKit. |
| Model-owned `scripts/accuracy.py` | Measures task accuracy with a build's MPKs on a Modalix DevKit, when declared. |
| Runner scripts in `scripts/` | Run each stage the same way locally and in CI: `run_model_compilation.sh`, `run_model_performance.sh`, `run_model_accuracy.sh`, and `run_model_conversion.sh`. |
| `model-registry` | Author-safe `sima-cli` package for validation, verified input handling, the compile matrix, and compile-output and MPK checks. CI installs the full command surface for stage planning, packaging, provenance, and publication. |
| GitHub Actions | Plans stages from their keys, runs them, publishes results, and reports status. Pull requests run only `Target Branch Checks` and the description check, `PR Sanitization`. |
| Model Compiler runner | An ephemeral self-hosted runner with the pinned Model Compiler. Compiles one build with read-only repository access. |
| Modalix DevKit runner | Runs the performance and accuracy stages of one build. Holds no publication credentials. |
| S3/object storage | Stores immutable compiled artifacts, packages, logs, and results. |
| Artifact-registry API | Records runs, provenance, results, and catalog snapshots through authenticated, idempotent requests. |

## Directory contract

```text
.github/workflows/         Compile, build, DevKit-stage, conversion, tool, and branch workflows
artifact-browser/          Model browser and its local preview server (preview.py)
deps/                      Repository-wide toolchain and DevKit runtime pins
design-doc/                Architecture and lifecycle design documents
docs/                      Developer contribution guides
models/<model-id>/         model.yaml, README.md, hero.png, and model code in scripts/
packaging/model-registry/  Build and installer for the author-safe sima-cli package
schemas/                   Model, composition, package, and result JSON Schemas
scripts/                   Stage runner scripts and CI helpers
taxonomies/                Sanctioned model categories and display labels
tests/                     Schema fixtures, unit tests, and contract tests
tools/model_registry/      Shared CLI and lifecycle implementation
training/                  Declared training code that is a compile input of its model
website/                   Developer documentation site
```

Common download, checksum, cache, MPK validation, result import, and
publication logic stays under `tools/`. A model directory owns the code that is
specific to that model: the compile recipe, any surgery that cannot be
expressed as shared parameters, and its performance, accuracy, and conversion
scripts. The recipe's product is one directly consumable `*_mpk.tar.gz` per
component, not an ONNX or compiler intermediate.

The canonical product name is **Model Registry**. Code and paths use
`model-registry` or `model_registry`. The word `modelzoo` is used only when
documenting compatibility with an existing external CLI or legacy service.

Model authors install the non-publishing `model-registry` command through
`sima-cli`; its installer finds the Model Compiler environment and installs
alongside it without replacing compiler-owned dependencies.
`packaging/model-registry/build.sh` removes the publication modules from that
package. Packaging and publication remain automation-only.

## Shared tooling

`tools/model_registry/` holds one module per job. Model-specific code belongs in
the model's `scripts/`, not here. Put a new operation in the module that owns
its job; a new job gets a new module named after what it does, and a row in
this table. Modules marked CI only are removed from the author package by
`packaging/model-registry/build.sh`.

| Module | Owns |
| --- | --- |
| `cli.py` | the `model-registry` command; `AUTHOR_COMMANDS` lists the commands the author package exposes |
| `model.py` | loading and validating `model.yaml` and the model folder (`load_models`), and reading a model's declarations: compile invocations, profiles, components |
| `compile_matrix.py` | the compile matrix: one entry per declared compile invocation (`build_matrix`) |
| `stage_keys.py` | the stage keys (`compile_key`, `performance_key`, `accuracy_key`) and the files each one hashes |
| `stage_plan.py` | which builds compile and which DevKit stages run on a push (`plan_stages`), the registry reads that decision needs, and base-branch resolution |
| `registry.py` | publishing compile runs, packages, model documentation, and the branch catalog the browser reads (CI only) |
| `benchmark_manifest.py` | the DevKit benchmark manifest for a compiled build (CI only) |
| `benchmark_publication.py` | publishing performance and accuracy runs (CI only) |
| `core_benchmark.py` | packaging Core's measurements for publication |
| `accuracy_config.py`, `accuracy_result.py` | finding a build's DevKit accuracy entry; accuracy result documents |
| `fidelity_config.py` | finding a build's fidelity entry and checking its result |
| `artifact.py` | compile outputs and MPK verification |
| `artifact_metadata.py` | publication metadata validation |
| `package.py` | assembling model packages from component MPKs |
| `dependencies.py` | resolving the Model Compiler and DevKit runtime pins in `deps/` |
| `fetch.py` | verified downloads, the cache, and dataset resolution |
| `source.py` | source identity and URL redaction |
| `taxonomy.py` | loading the taxonomy and validating model categories |
| `documentation.py` | loading a model's README and hero image |
| `runner_approval.py` | authorizing requested compile runner classes |
| `errors.py` | the error types every module raises |

## Stages and stage keys

A build is one compile invocation of one build profile of one model. It passes
through up to three stages, each with a stage key: a SHA-256 of everything its
result depends on.

| Stage | Model script | Key | Registry record |
| --- | --- | --- | --- |
| compile | `compile.py` | `compile_key` in `stage_keys.py` | compile run per component, recorded in the branch catalog |
| performance | `performance.py` | `performance_key` in `stage_keys.py` | `model_benchmark` run under the component's compile run |
| accuracy | `accuracy.py` | `accuracy_key` in `stage_keys.py` | `model_accuracy` run under the result component's compile run |

The compile key covers the invocation's `model.yaml` projection, the fidelity
entries, the model's compile scripts, declared training code, and the compiler
version. The performance and accuracy keys cover the compile runs and MPK
digests they measure, their `model.yaml` entries and model script, the shared
tooling of that stage, and the DevKit runtime pin. A recompiled build gets new
DevKit keys and is therefore remeasured. The contributor guide lists every
input in [what a change reruns](../docs/adding-a-model.md#9-know-what-a-change-reruns).

## CI flow

```text
push (Vulcan Model Orchestrator, every push)
  -> select: validate the catalog, resolve the compiler pin and base branches
     portal (in parallel): build and smoke-test the model browser; a failure stops the run here
       -> on main and develop: publish that model browser to staging
  -> plan: compare stage keys with the branch catalog and registry runs
  -> authorize: check requested runner classes
  -> initialize the catalog: mark selected builds pending
  -> compile (Vulcan Model Compile, per selected build)
       compile: run_model_compilation.sh, verify-mpk (self-hosted, read-only)
       publish: compile runs, fidelity, start DevKit stages, packages, catalog entry
  -> publish the catalog snapshot
  -> measure: start the DevKit stages that reused builds still lack
DevKit stages (per build)
  -> performance and accuracy on a DevKit, then publish each result
```

`plan-stages` reads the branch's prior catalog (see below). A build compiles when the catalog records no build with its current
compile key, or when a maintainer forces a rebuild. A reused build is queued
for the DevKit stages whose keys have no completed run. DevKit stages run only
in staging and only for Modalix targets.

### Branches, merges, and concurrent pushes

- **One registry per environment.** Pushes publish to staging. Each branch has
  its own catalog in that registry, keyed by Git ref; runs are shared by every
  branch.
- **Where each decision looks.** The compile decision compares compile keys
  with the branch's prior catalog. The DevKit decision looks for a completed
  run with the stage key from any branch; that key names the compile run and
  MPK digest, so it matches only a measurement of the same build.
- **Base branches.** `develop`, `main`, `release-*` and `integration/*` are
  long-lived. `select` lists the long-lived branches that share history with
  the head, closest first, or only the dispatch base when one is given; Git does not record which branch another was cut from.
- **Prior catalog.** A long-lived branch keeps to its own catalog; on its first
  push it inherits from the closest base branch that has one. A feature branch
  takes each model build profile from the first catalog whose builds carry
  every current compile key of that profile: its own, then its base branches
  in order. A feature branch that merged `develop` therefore reuses both its
  own builds and the builds `develop` recompiled. A profile no catalog matches
  compiles. A sibling feature branch is never a source.
- **Merge.** A merge is a push to `develop`, which compares with its own
  catalog. Builds the merged branch changed compile again on `develop`, and
  their DevKit stages run again; builds the branch compiled are not reused.
- **Newer push.** It cancels the running orchestrator of the same ref
  (manual dispatches run to completion). Builds that already published stay;
  builds left pending are planned again. Dispatched DevKit runs continue. A
  DevKit stage still running when the next plan runs has no completed run yet,
  so it is dispatched again.
- **Model browser.** A branch's view lists its catalog's builds: what the
  branch delivers (#181). A completed build compiled on the branch that is
  newer than the catalog entry for its model and build profile replaces that
  entry, so a build missing from the catalog still shows. Each build keeps its
  own compile run's branch, commit and compiler, so a reused build still names
  the branch that compiled it. Every push publishes a catalog, so a branch that
  changes no stage key still lists every model.

## Compile workflow trust boundary

Pull requests run only `Target Branch Checks` and `PR Sanitization`, on
GitHub-hosted runners with read-only repository access. The orchestrator has no pull-request trigger, so
self-hosted jobs run only for trusted pushes and explicit maintainer
dispatches. Production is manual and uses the protected
GitHub `production` environment.

Before requesting a self-hosted runner, the select job resolves the exact Model
Compiler version from `deps/manifest.json` and derives the compiler runner
label. The compile job then checks that the runner's installed compiler reports
the same version before any model code runs. Model builds do not carry
independent compiler pins. The compile job has only read access to the
repository and no publication credentials; a separate GitHub-hosted job
verifies its output and publishes it.

## Artifact publication contract

Each published build verifies its MPKs and provenance, uploads each archive to
a collision-resistant immutable S3 key with native SHA-256, and records one
compile run per component through IAM-authenticated, idempotent API calls. The
run records the immutable source, the compile key, the compiler identity, the
exact S3 object version and checksum, the MPK validation result, and the
fidelity report when one is declared. A sibling build failure cannot roll back
or hide a completed run.

The registry database is the discovery and lifecycle source of truth. Each
push publishes a catalog snapshot for its revision, and public read APIs serve
the current snapshot, its builds, and presigned download URLs. The model
browser and `sima-cli models` commands consume those APIs directly.

Branch deletion cancels the branch's unfinished workflow runs and calls the
registry purge endpoint; branch creation calls the restore endpoint. Retention
operates on registered runs and exact S3 object versions.
