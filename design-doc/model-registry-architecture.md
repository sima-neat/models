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
| Model-owned `scripts/compile.py` | Reads its build profile from `model.yaml`, emits one MPK per compiled component, and writes the profile's numerical fidelity report when declared. |
| Model-owned `scripts/compile.sh` | Calls `llima-compile`; receives checkpoint/output paths, the profile's compiler options and `BUILD_PROFILE_ID`. |
| Model-owned `scripts/performance.py` | Measures a build profile's whole model, or each independent component when `performance.scope: components`, on a Modalix DevKit with Neat Core and writes its report with `tools/model_report`, when declared. |
| Model-owned `scripts/accuracy.py` | Measures a build profile's whole-model task accuracy on a Modalix DevKit and writes its report with `tools/model_report`, when declared. |
| `tools/model_report` | Board-side report writer, standard library only: builds `model-report.json` from what Core returned, with the metric vocabulary in `metrics.json`. |
| Runner scripts in `scripts/` | Run each stage the same way locally and in CI: `run_model_compilation.sh`, `run_llima_compilation.sh`, `run_model_performance.sh`, `run_model_accuracy.sh`, and `run_model_conversion.sh`. |
| `model-registry` | Author-safe `sima-cli` package for validation, verified input handling, the compile matrix, and compile-output and MPK checks. CI installs the full command surface for stage planning, packaging, provenance, and publication. |
| GitHub Actions | Plans stages from their fingerprints, runs them, publishes results, and reports status. Pull requests run only `Target Branch Checks` and the description check, `PR Sanitization`. |
| Model Compiler runner | An ephemeral self-hosted runner with the pinned Model Compiler. Compiles one build with read-only repository access. |
| Modalix DevKit runner | Runs the performance and accuracy stages of one build. Holds no publication credentials. |
| S3/object storage | Stores immutable compiled artifacts, packages, logs, and results. |
| Artifact-registry API | Records runs, provenance, results, and catalog snapshots through authenticated, idempotent requests. |

## Directory contract

```text
.github/workflows/         Compile, build, DevKit-stage, conversion, tool, and branch workflows
artifact-browser/          Model browser, its local preview server (preview.py), and licenses.json
deps/                      Repository-wide toolchain and DevKit runtime pins
design-doc/                Architecture and lifecycle design documents
docs/                      Developer contribution guides
models/<model-id>/         model.yaml, README.md, hero.png, model code in scripts/, and its tests in tests/
packaging/model-registry/  Build and installer for the author-safe sima-cli package
schemas/                   Model, tests, package, model report, and fidelity JSON Schemas
scripts/                   Stage runner scripts and CI helpers
taxonomies/                Sanctioned model categories and display labels
tests/                     Repository-wide unit and contract tests and their fixtures
tools/model_registry/      Shared CLI and lifecycle implementation
tools/model_report/        Board-side model report writer and metric vocabulary
website/                   Developer documentation site
```

Common download, checksum, cache, MPK validation, report writing, and
publication logic stays under `tools/`. A model directory owns the code that is
specific to that model: the compile recipe, any surgery that cannot be
expressed as shared parameters, and its performance, accuracy, and conversion
scripts. ONNX recipes produce one directly consumable `*_mpk.tar.gz` per
component. LLiMa recipes produce compiler output that the shared runner
deploys into a complete model folder.

The canonical product name is **Model Registry**. Code and paths use
`model-registry` or `model_registry`. The word `modelzoo` is used only when
documenting compatibility with an existing external CLI or legacy service.

Model authors install the non-publishing `model-registry` command through
`sima-cli`; its installer finds the Model Compiler environment and installs
alongside it without replacing compiler-owned dependencies.
`packaging/model-registry/build.sh` removes the publication modules from that
package and adds `model_report`, whose metric vocabulary validation reads. Packaging and publication remain automation-only.

## Shared tooling

`tools/model_registry/` holds one module per job. Model-specific code belongs in
the model's `scripts/`, not here. Put a new operation in the module that owns
its job; a new job gets a new module named after what it does, and a row in
this table. Modules marked CI only are removed from the author package by
`packaging/model-registry/build.sh`.

| Module | Owns |
| --- | --- |
| `cli.py` | the `model-registry` command; `AUTHOR_COMMANDS` lists the commands the author package exposes |
| `model.py` | loading and validating `model.yaml` and the model folder (`load_models`), and expanding it into components, build profiles with their one compile invocation, and the `default` package |
| `compile_matrix.py` | the compile matrix: one entry per build profile's compile invocation, with the benchmarks that measure it (`build_matrix`) |
| `fingerprints.py` | the fingerprints (`compile_fingerprint`, `benchmark_fingerprint`), the files each one hashes, and each benchmark's run type |
| `stage_plan.py` | which builds compile and which benchmarks run on a push or explicit on-demand dispatch (`plan_stages`), the registry reads that decision needs, and base-branch resolution |
| `registry_http.py` | HTTP transport and authenticated read-only access shared with the author package. |
| `registry.py` | publishing compile runs, packages (with their MLA stage simulation), model documentation, and the branch catalog the browser reads (CI only) |
| `promotions.py` | pinned environment/target-specific Develop and production Main requests, inspection, decisions, receipts and review summaries (CI only) |
| `candidates.py` | complete model/profile candidate contracts, run publication and binding, finalization, feature selection and retry receipts (CI only) |
| `benchmark_declarations.py` | the `performance` and `accuracy` declarations: the profiles each benchmarks, their scripts and required metrics, and validating `reference_metrics`, which the portal ships |
| `report_publication.py` | the MPKs a benchmark runs, checking a `model-report.json` against `model.yaml`, and publishing one benchmark run under the profile's package run, with whole-model measurements by default or separate component measurements identified by `component_id` (CI only) |
| `artifact.py` | compile outputs, MPK/deployed LLiMa verification, reproduction recipes, and deployment file indexes |
| `mla_stage_simulation.py` | after an MPK compile, the Model Compiler simulator's operation counts and latency for each MLA stage (`simulate-mla-stages`) |
| `package.py` | assembling a profile's `default` package from component MPKs, the model's pinned `resources` and the compile's `runtime_files`, or from a complete native deployment folder, with its numerical fidelity report and MLA stage simulation |
| `dependencies.py` | resolving the Model Compiler and DevKit runtime pins in `deps/` |
| `fetch.py` | verified downloads, the cache, and dataset resolution |
| `conversion.py` | conversion fingerprints, declared outputs, validated bundles, publication, durable plan selections and exact downloads; author CLI exposes only saved-selection fetching |
| `source.py` | the approved source hosts and their URL checks, source identity, and URL redaction |
| `taxonomy.py` | loading the taxonomy and validating model categories |
| `documentation.py` | loading a model's README and hero image |
| `runner_approval.py` | authorizing requested compile runner classes/markets and selecting optional runner audit fields for durable publication |
| `model_tests.py` | each model's test files and `tests/tests.yaml`, the comparison of a report's metrics with its build profile's expected values, and the outcomes a stage's tests report |
| `errors.py` | the error types every module raises |

## Stages and fingerprints

A build is one build profile of one model, compiled in one compile invocation
named after the profile. It passes through up to three stages, each with a
fingerprint: a SHA-256 of every input that decides its result. The performance
and accuracy stages are the build's benchmarks.

| Stage | Model script | Fingerprint | Registry record |
| --- | --- | --- | --- |
| compile | `compile.py` or `compile.sh` | `compile_fingerprint` in `fingerprints.py` | compile run per component and a `model_package` run per profile, recorded in the branch catalog |
| performance | `scripts/performance.py` | `benchmark_fingerprint` in `fingerprints.py` | `model_performance` run under the profile's package run |
| accuracy | `scripts/accuracy.py` | `benchmark_fingerprint` in `fingerprints.py` | `model_accuracy` run under the profile's package run |

The ONNX compile fingerprint covers the profile's `model.yaml` projection, the
`numerical_fidelity` section, the model's compile scripts, shared model loader,
and the compiler version. The LLiMa compile fingerprint
covers checkpoint content, target, declared graph parameters and pinned
compiler identity; scripts and registry helpers are not compile inputs. The
benchmark fingerprints cover the compile run and MPK digest of every component,
the stage's `model.yaml` declaration and model script, the shared benchmark
tooling and that stage's runner scripts, and the DevKit runtime pin. The
catalog lists each profile's benchmark fingerprints under
`benchmark_fingerprints`, and the model browser resolves the runs by those
fingerprints. A recompiled build gets new benchmark fingerprints and is
therefore benchmarked again. Performance fingerprints also include the profile's
optional `benchmark_baseline`. Accuracy and compilation fingerprints do not.
Editing a test reruns nothing, because no fingerprint includes tests. The changed
test runs the next time its stage runs.

The conversion fingerprint covers only files that can change the ONNX. Keep it
that narrow: a fingerprint with no published conversion fails the plan for every
model until someone reruns Convert. The contributor guide lists
every input in [what a change
reruns](../docs/adding-a-model.md#9-know-what-a-change-reruns).

## CI flow

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `Target Branch Checks` | pull request | Runs `python -m pytest`, builds the `model-registry` package, and builds, browser-tests, and smoke-tests the model browser on the merge result. Contributor code never reaches a self-hosted runner. |
| `Vulcan Model Orchestrator` | every push, manual dispatch | Plans each stage against the registry, runs `Vulcan Model Compile` for each changed build and the benchmark workflow for each reused build that lacks a benchmark, and publishes the branch catalog. Each build's jobs are grouped under the build's name. A newer push cancels the running orchestrator with its compiles and benchmarks. A push that changes no fingerprint only publishes the catalog. On `main` and `develop`, it also publishes the model browser it checked to staging, independently of `VULCAN_ENV`. |
| `Vulcan Model Compile` | called by the orchestrator | Compiles one build, runs the model's compile tests on it, publishes it, and benchmarks it. |
| `Vulcan Model Benchmarking` | called by `Vulcan Model Compile` and the orchestrator, manual dispatch | Runs the requested `performance` and `accuracy` stages of one build on a Modalix DevKit, runs the model's benchmark tests on each result, and publishes each result with its test outcome. |
| `Vulcan Model Convert` | manual dispatch | Runs `scripts/run_model_conversion.sh` and the model's conversion tests, publishes a validated ONNX bundle and its provenance to Vulcan. |
| `Build and Publish Model Registry` | push, manual dispatch | Runs `python -m pytest`, builds the `model-registry` package with `packaging/model-registry/build.sh`, publishes it, and moves the branch's `latest` tag. |
| `Model branch lifecycle` | branch create or delete | Cancels a deleted branch's unfinished runs and purges or restores its registry state. |

### Registry release state

[`state.json`](../state.json) defines the release experience consumed by the SDK's
Palette SDK component: `current_state` accepts `GA`, `Beta`, or `Alpha`, and
`default_branch` accepts `develop` or `main`. Initial values are `Alpha` and
`develop`, allowing the consumer to display `Current Release: Alpha`. Palette SDK
integration must fetch this file; it is not implemented here. This file does not
participate in model compilation or fingerprints.

Registry release state is a separate production workflow: changes to `state.json`,
`scripts/publish_registry_state.sh`, or `vulcan-model-state.yml` on `main` publish
`models/state.json`. Manual runs also require `main`; both jobs and the upload
script enforce this restriction. Runs share one production concurrency group
and verify the uploaded file. Merging to `develop` does not deploy this state.
Publication validates both fields and sets JSON content type and `Cache-Control:
no-store`. CDN configuration must honor that cache policy.

### Other workflows

Manual **Deploy Model Browser & Docs** is independent of model builds and tooling
publication. From `main` or `develop`, select `staging` or `production` and the
`developer` or `customer` audience. The `authorize` job waits on the
`model-browser-deploy` environment's sole reviewer team, `owners-models`, with
administrator bypass disabled. The dispatching owner must approve their own run;
reruns must be replaced by a new dispatch. Every downstream job independently
requires attempt 1 and the original triggering actor, so rerunning only failed
jobs cannot reuse a completed authorization job. The named authorization script verifies
the gate and approval before configuration resolution or deployment credentials.
The reusable docs workflow builds and smoke-tests the selected frontend, then the
shared static publisher uploads only `models/`, retains old assets, and invalidates
`/models` and `/models/*`. Its publish concurrency group is
`${{ github.repository }}-model-browser-${{ inputs.environment }}`, with
`cancel-in-progress: false`; staging shares the automatic publisher's group so
manual and automatic uploads cannot run concurrently. Automatic staging publication
on `main`/`develop` uses a separate staging configuration regardless of `VULCAN_ENV`;
production frontend publication requires this manual dispatch. See [browser deployment](../docs/model-browser.md).

**Vulcan Model Registry** tooling publication selects repository `VULCAN_ENV`
(default `staging`) for both pushes and manual runs. It has no separate environment
input and publishes to the selected Vulcan backend independently of frontend deployment.

Manual **Promote model** takes a model ID and build profile, resolves the newest matching
GitHub candidate receipt, and verifies its identity with Vulcan. It runs independently of
compilation: prepare and persist a request plan →
create the protected backend request and issue summary → mention model owners and send
a Slack approval reminder → protected `<VULCAN_ENV>-promote-<target_branch>`
review → approve the exact request → report backend state and notify Slack of the outcome. Only trusted Develop workflow code
runs; a separate approver role decides requests. Status/cancel use publisher credentials; explicit
reject uses the matching target's protected gate. Main is production-only, keeps
S3 storage, and uses a separate Main-only approver identity. It pins a completed
Develop approval for the exact candidate and digest before its own review. The target branch,
environment, endpoints and canonical snapshot are pinned in the immutable plan;
retries cannot reinterpret a Develop request as Main. Publish the target's own
metadata before requesting selection. See [promotion operations](../docs/model-promotion.md) for
configuration, retries, and the staging acceptance boundary.

```text
push (Vulcan Model Orchestrator, every push)
  -> select: validate the catalog, resolve the compiler pin and base branches
     portal (in parallel): build and smoke-test the model browser; a failure stops the run here
       -> on main and develop: resolve-staging-browser-config + checked portal -> publish-model-browser to staging
  -> resolve-vulcan-config: after select, resolve the selected environment configuration
  -> plan (waits for select, portal and resolve-vulcan-config): resolve and retain converted ONNX inputs, then compare fingerprints with the branch catalog and registry runs; skip on-demand models on pushes
  -> authorize: check requested runner classes
  -> initialize the catalog: publish model requirements and mark automatic builds pending
  -> resolve-benchmark-runner: resolve the DevKit runner once; every benchmark of the run gets it and the Vulcan config
  -> benchmark: benchmark each reused build that still lacks a benchmark (after the snapshot when nothing compiles)
  -> compile (Vulcan Model Compile, per selected build)
       compile: backend runner, verify-artifact, the model's compile tests (self-hosted, read-only)
       publish: compile runs, the package with its fidelity report, catalog entry
       benchmark: the benchmark workflow for the new Modalix build, as soon as it publishes
       on-demand LLiMa: finalize the complete candidate, then select it in its feature catalog
  -> publish-candidates (other on-demand models): join all invocations, finalize and select in the feature catalog
  -> publish the catalog snapshot, once every compile call ends, including each new build's benchmarks
benchmarks (vulcan-model-benchmarking.yml, called per build profile inside the orchestrator run, or dispatched by hand)
  -> stage the build, on a GitHub-hosted runner
  -> performance and accuracy on a DevKit, then the model's benchmark tests on each model report
  -> publish each report under the profile's package run with its test outcome; a failed test then fails a check job
```

Pushes and manual build dispatches both use `VULCAN_ENV` as the model build and
registry publication target. Set it to `production` for production builds or
`staging` for staging builds. An unset or invalid value fails before compiler
runners are requested; a dispatch cannot override it. Concurrency and the run
title use the same variable. On-demand candidates use the selected environment
for compilation, publication and feature-branch catalog selection.
Frontend builds explicitly use staging, and `resolve-staging-browser-config`
resolves staging publication settings independently of `select` and
`resolve-vulcan-config`. Frontend publication waits only for this staging resolver
and the checked portal. The selected model configuration
supplies the compiler runner environment, artifact bucket and dataset bucket; compile
workers receive that dataset bucket through `SIMA_DATASET_S3_URI`.

The `plan` job requests `id-token: write` and waits for configuration resolution
as well as validation and the portal check. Only production assumes the selected
GitHub environment's `VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN` via OIDC before
planning. That role must authorize catalog/resolve reads for this repository;
planning uses a publisher-capable identity even though its registry operations
are reads. Missing trust or read permissions fail the job before compilation.
Staging skips this credential step and continues using public registry reads.

Read URLs prefer `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL`, falling back to
`VULCAN_ARTIFACT_REGISTRY_API_URL`. For production, `MODEL_REGISTRY_IAM_READ_API_URL`
is set to the configured API base: only reads below that exact base's `/v1/`
path are SigV4-signed. Public-reader URLs, other hosts and artifact downloads
retain their existing behavior. An unset public-read URL therefore permits
private production planning without enabling public access. Catalog initialization
and final publication use the same read selection alongside their existing write
authentication. Pull-request validation does not acquire this planning identity.

`plan-stages` reads the branch's prior catalog (see below). A build compiles when the catalog records no build with its current
compile fingerprint, or when a maintainer forces a rebuild. A reused build is
queued for the benchmarks whose fingerprints have no completed run. Benchmarks
run on staging DevKits for either registry environment, only for Modalix targets, and only after the profile's
package publishes. On-demand models compile only on the explicit manual
selection described below, and their builds are not benchmarked.

The orchestrator passes its resolved `VULCAN_ENV` to every benchmark it calls.
A manual benchmark dispatch defaults to repository `VULCAN_ENV`, with the same
staging fallback and per-run override. Registry reads, datasets, and report
publication use that selection; DevKit discovery independently requests
`vulcan-staging`, and execution consumes the resolver's labels. Production
registry restoration and report publication use the production GitHub environment
on hosted runners. The board receives staged artifacts and dataset-reader access.
Benchmark concurrency includes the registry environment so staging and production
runs do not share a concurrency group. This routing does not change model or
benchmark fingerprints.

### Branches, merges, and concurrent pushes

- **One registry per environment.** Pushes publish to the environment selected by `VULCAN_ENV`. Each branch has
  its own catalog in that registry, keyed by Git ref; runs are shared by every
  branch.
- **Where each decision looks.** The compile decision compares compile
  fingerprints with the branch's prior catalog. The benchmark decision looks
  for a completed run with the benchmark fingerprint from any branch; that
  fingerprint names every component's compile run and MPK digest, so it
  matches only a benchmark of the same build. A feature branch leaves to its
  closest base branch every build profile that base branch's catalog lists as
  pending at the head fingerprints, with its benchmarks, and every benchmark
  whose fingerprint that catalog records, so the
  work a merge started on `develop` is not repeated on branches that merge
  `develop`. A forced rebuild still compiles.
- **Base branches.** `develop`, `main`, `release-*` and `integration/*` are
  long-lived. `select` lists the long-lived branches that share history with
  the head, closest first, or only the dispatch base when one is given; Git does not record which branch another was cut from.
- **Prior catalog.** A long-lived branch keeps to its own catalog; on its first
  push it inherits from the closest base branch that has one. A feature branch
  takes each model build profile from the first catalog whose builds carry
  every current compile fingerprint of that profile: its own, then its base branches
  in order. A feature branch that merged `develop` therefore reuses both its
  own builds and the builds `develop` recompiled. A profile whose current
  fingerprints a base branch is still compiling takes that base branch's entry,
  so the branch shows the base branch's previous build until its next push. A
  profile that no catalog holds and the closest base branch is not compiling
  compiles on the branch. A sibling
  feature branch is never a source.
- **Merge.** A merge is a push to `develop`, which compares with its own
  catalog. Automatic builds changed by the merge compile again on `develop`,
  and their benchmarks run again; builds the branch compiled are not reused.
- **Newer push.** It cancels the running orchestrator of the same ref
  (manual dispatches run to completion), including its compiles and the
  benchmarks running inside it. Builds and benchmark runs that already
  published stay; builds left pending and benchmarks without a completed run
  are planned again.
- **Model browser.** A branch's view lists its catalog's builds: what the
  branch delivers (#181). A completed build compiled on the branch that is
  newer than the catalog entry for its model and build profile replaces that
  entry, so a build missing from the catalog still shows. Each build keeps its
  own compile run's branch, commit and compiler, so a reused build still names
  the branch that compiled it. Every push publishes a catalog, so a branch that
  changes no fingerprint still lists every model.

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

Each published ONNX build verifies its MPKs and provenance, uploads each archive to
a collision-resistant immutable S3 key with native SHA-256, and records one
compile run per component through IAM-authenticated, idempotent API calls. The
run records the immutable source, the compile fingerprint, the compiler identity, the
exact S3 object version and checksum, and the MPK validation result. The
profile's `model_package` run carries its numerical fidelity report when one
is declared. A sibling build failure cannot roll back or hide a completed
run.

The registry database is the discovery and lifecycle source of truth. Each
push publishes a catalog snapshot for its revision, and public read APIs serve
the current snapshot, its builds, and presigned download URLs. The model
browser and `sima-cli models` commands consume those APIs directly.

Branch deletion cancels the branch's unfinished workflow runs and calls the
registry purge endpoint; branch creation calls the restore endpoint. Retention
operates on registered runs and exact S3 object versions.


## On-demand builds

Any expensive model may set top-level `build-on-demand: true`; the default is
`false`. Pushes validate its contract and publish catalog metadata but never
compile or measure it. Existing automatic models keep their scheduling.
The flag changes scheduling and catalog metadata only, never compile fingerprints.
An explicit **Vulcan Model Orchestrator** dispatch selects both `models` and
`profiles`; `plan-stages --include-on-demand` also requires both selectors.
For Qwen3.5-0.8B use `models=qwen3_5_0_8b`, `profiles=w4a16`.

In staging or production, `publish-candidate` waits for every invocation of the selected
model/profile. A single-component LLiMa build does this in its existing Publish
job; other on-demand models join their invocations in the orchestrator's
`publish-candidates` job.
Each candidate requires its own compile tests to pass; a failure in another
model/profile does not block its finalization.
It registers one versioned contract, binds each compile and
package run before publishing its records, and declares every required file.
New uploads use the returned candidate prefix, with shared payloads reused
within that candidate. Every run is completed before candidate finalization.
Ordinary catalog README and hero publication stays branch-retained and is not
part of the candidate. DevKit qualification is separate; it is not claimed as
frozen candidate evidence in this iteration.

The backend protects assembly for seven days. The newest successful candidate
per source branch/model/profile receives `latest_successful` protection;
older late completions receive bounded protection. Source-branch deletion does
not remove protected candidates. The summary and `candidate.json` report the
actual returned protection, digest and provenance. Stable keys resume the same
GitHub run across attempts, including interrupted publication; expired assembly
requires a new workflow run. Compiler outputs and publication receipts remain
GitHub artifacts for recovery. An on-demand profile is always rebuilt whole,
since older source runs cannot be rebound as new provenance. Ordinary automatic fingerprint reuse is unchanged.

Vulcan limits a candidate to 32 roles, 512 required names, 512 actual S3
dependencies and a 1 MiB frozen manifest. The producer checks the role/name
limits before publication; the backend validates the actual frozen dependency
set. Candidate IDs and scheduling policy do not enter compilation fingerprints.

Catalog publication includes versioned `build_requirements` from the checkout's
existing compile fingerprints, compiler pin, and package topology, even before a build
exists. Vulcan owns `approved_deployments`; readers resolve its original compile
and package roots through the existing artifact APIs. The producer never
manufactures approval or copies candidate requirements into the target.

For feature builds, the workflow publishes these requirements before compiling.
After finalization, `publish-branch-deployment` calls Vulcan's `branch-deployments`
endpoint with the frozen candidate ID/digest and current catalog snapshot. A
snapshot conflict reloads the catalog and retries with a new body-bound key;
failed selection can be retried without recompilation. Vulcan atomically selects
and protects the deployment in `branch_deployments`, without approving it.
Readers prefer that profile's branch selection over inherited approvals and
catalog builds; unselected completed runs cannot replace it.

Vulcan PRs #248, #250 and #253 provide the staging backend. The protected GitHub
approval workflow is tracked separately in [#315](https://github.com/sima-neat/models/issues/315).
Candidate finalization does not approve or promote a build. Approver roles and
end-to-end producer acceptance are required before routine approval. Develop is
supported in staging and production. Main is supported only in production through
its own protected gate and branch-scoped approver role. Both select the same
immutable production candidate in independently pinned target catalogs. Each
selection holds its own retention reference, so replacing Develop cannot release
Main's artifacts or the reverse. Main metadata comes from its chosen source
revision, not from an implicit copy of Develop.

Production Main registry selection retains S3 storage and active S3 downloads.
The earlier combined Main/Hugging Face plan is split: Hugging Face transfer,
immutable revision verification and any later download-host switch remain a
separate release operation under [#293](https://github.com/sima-neat/models/issues/293).
This promotion workflow never uploads to Hugging Face. The backend resolves
approved roots; the CLI downloads and verifies their files.

Manual **Publish Main model metadata** executes trusted Develop workflow code,
checks out Main's exact release revision, installs its matching producer, and
runs `publish-catalog` against production Main with Main as the only base branch.
It publishes metadata and documentation without compiling, benchmarking or
approving artifacts. The job uses publisher credentials and saves the source SHA
and receipt. Its independent metadata-only path supports Main's older source
schema without copying newer Develop model definitions or requiring a build.
Main still needs its own on-demand profile requirements before candidate approval.

## LLiMa builds

The compile matrix selects `scripts/run_llima_compilation.sh` for a Hugging
Face component. The shared downloader assembles its pinned, verified file
set; the recipe only compiles. The runner writes a direct reproduction command
in `compile.sh` with download instructions and the resolved graph arguments,
defaulting to four processes, plus any adjacent
`compile.py` or `config.py`. It then runs
`llima-deploy` into `<model>/<profile>/<component>/deployment/`, containing
`devkit/`, `elf_files/`, any `npy_files/`, and those reproduction files. `verify-artifact --write-manifest`
writes `deployment-manifest.json` beside the folder. Compile outputs point to
that index and its SHA-256. ONNX builds retain their runner and compile-fingerprint
inputs.

Publication registers deployment files as `model-file` artifacts and the
compile index as the `llima` `model-pack` artifact. A native full package is
the deployment folder with root `manifest.json`, published as the `package`
`model-package` JSON artifact. Its manifest references files on that same
package run. Content-addressed S3 objects are reused between compile and
package publication. Retrieval preserves the folder and verifies every file
through public resolved URLs. See the [metadata contract](../docs/metadata-schema.md#llima-checkpoint-and-deployment-formats)
for manifest fields, publication metadata, and the temporary 5 GiB per-file
upload limit. Aggregate deployment size has no limit.

Empty benchmark declarations schedule no benchmarks. Qwen's defaults
and the separate performance and accuracy follow-ups are in the
[author guide](../docs/adding-a-model.md#llima-language-and-vision-language-models).

### Forced rebuilds of published packages

`force_rebuild_all` recompiles the selected builds. Once a selected automatic
build's package is published in the current catalog revision, dispatch the
forced rebuild from a new Git commit. Catalog packages cannot be removed or
replaced within that revision, even by initialization. The orchestrator passes
`--force-rebuild-all` to `publish-catalog`, which rejects this conflict before
compile jobs start instead of silently discarding the rebuild selection.
An ordinary initialization retry preserves packages already published in the
revision. On-demand candidate publication follows its separate deployment flow.

## Converted ONNX inputs

Models may declare `compilation.source: conversion`. The planner resolves and
records the latest validated matching output before deciding build reuse. A
`model_compile_plan` run retains the selection across workflow retries and
protects its conversion for 24 hours; retained compile inputs protect it afterward.
No artifact table or custom resolver is added. See [conversion artifacts](../docs/conversion-artifacts.md)
for the registry contract, rollout and reproduction.

Production conversion-backed compile jobs obtain a session restricted to ONNX
artifact metadata GET requests before compilation. They use it for private registry
reads during compilation, including cache hits. Immediately after compilation,
the job obtains a fresh session with the same restricted policy for provenance
verification, so a long compile cannot exhaust the verification session's lifetime.
Both steps receive isolated reader credentials, preserving the runner credentials used
for calibration datasets; publication remains in the separate hosted publish job. Public reads do not
require signing. See [conversion artifacts](../docs/conversion-artifacts.md#private-production-reads).
