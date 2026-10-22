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
| Model-owned `scripts/performance.py` | Measures a build profile's whole model on a Modalix DevKit with Neat Core and writes its report with `tools/model_report`, when declared. |
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
artifact-browser/          Model browser and its local preview server (preview.py)
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
| `registry.py` | publishing compile runs, packages (with, in staging, their MLA stage simulation), model documentation, and the branch catalog the browser reads (CI only) |
| `promotions.py` | pinned environment-specific Develop requests, inspection, decisions, receipts and review summaries (CI only) |
| `candidates.py` | complete model/profile candidate contracts, run publication and binding, finalization, feature selection and retry receipts (CI only) |
| `benchmark_declarations.py` | the `performance` and `accuracy` declarations: the profiles each benchmarks, their scripts and required metrics, and validating `reference_metrics`, which the portal ships |
| `report_publication.py` | the MPKs a benchmark runs, checking a `model-report.json` against `model.yaml`, and publishing it as one whole-model run under the profile's package run (CI only) |
| `artifact.py` | compile outputs, MPK/deployed LLiMa verification, and deployment file indexes |
| `mla_stage_simulation.py` | after an MPK compile, the Model Compiler simulator's operation counts and latency for each MLA stage (`simulate-mla-stages`) |
| `package.py` | assembling a profile's `default` package from component MPKs or a complete native deployment folder, with its numerical fidelity report and MLA stage simulation |
| `dependencies.py` | resolving the Model Compiler and DevKit runtime pins in `deps/` |
| `fetch.py` | verified downloads, the cache, and dataset resolution |
| `source.py` | source identity and URL redaction |
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
therefore benchmarked again. No fingerprint reads a model's `tests/`, so a
changed test applies the next time its stage runs. The contributor guide lists
every input in [what a change
reruns](../docs/adding-a-model.md#9-know-what-a-change-reruns).

## CI flow

Manual **Promote model** takes a model ID and build profile, resolves the newest matching
GitHub candidate receipt, and verifies its identity with Vulcan. It runs independently of
compilation: prepare and persist a request plan →
create the protected backend request and issue summary → mention model owners and send
a Slack approval reminder → protected `<VULCAN_ENV>-promote-develop`
review → approve the exact request → report backend state and notify Slack of the outcome. Only trusted Develop workflow code
runs; a separate approver role decides requests. Status/cancel use publisher credentials; explicit
reject uses the same protected gate. See [promotion operations](../docs/model-promotion.md) for
configuration, retries, and the staging acceptance boundary.

```text
push (Vulcan Model Orchestrator, every push)
  -> select: validate the catalog, resolve the compiler pin and base branches
     portal (in parallel): build and smoke-test the model browser; a failure stops the run here
       -> on main and develop: publish that model browser to staging
  -> plan: compare fingerprints with the branch catalog and registry runs; skip on-demand models on pushes
  -> authorize: check requested runner classes
  -> initialize the catalog: publish model requirements and mark automatic builds pending
  -> benchmark: start the benchmarks that reused builds still lack (after the snapshot when nothing compiles)
  -> compile (Vulcan Model Compile, per selected build)
       compile: backend runner, verify-artifact, the model's compile tests (self-hosted, read-only)
       publish: compile runs, the package with its fidelity report, start the benchmarks, catalog entry
       on-demand LLiMa: finalize the complete candidate, then select it in its feature catalog
  -> publish-candidates (other on-demand models): join all invocations, finalize and select in the feature catalog
  -> publish the catalog snapshot
benchmarks (Vulcan Model Benchmarking, per build profile)
  -> performance and accuracy on a DevKit, then the model's benchmark tests on each model report
  -> publish each report under the profile's package run with its test outcome; a failed test then fails the run
```

`plan-stages` reads the branch's prior catalog (see below). A build compiles when the catalog records no build with its current
compile fingerprint, or when a maintainer forces a rebuild. A reused build is
queued for the benchmarks whose fingerprints have no completed run. Benchmarks
run only in staging, only for Modalix targets, and only after the profile's
package publishes. On-demand models compile only on the explicit manual
selection described below, and their builds are not benchmarked.

### Branches, merges, and concurrent pushes

- **One registry per environment.** Pushes publish to staging. Each branch has
  its own catalog in that registry, keyed by Git ref; runs are shared by every
  branch.
- **Where each decision looks.** The compile decision compares compile
  fingerprints with the branch's prior catalog. The benchmark decision looks
  for a completed run with the benchmark fingerprint from any branch; that
  fingerprint names every component's compile run and MPK digest, so it
  matches only a benchmark of the same build.
- **Base branches.** `develop`, `main`, `release-*` and `integration/*` are
  long-lived. `select` lists the long-lived branches that share history with
  the head, closest first, or only the dispatch base when one is given; Git does not record which branch another was cut from.
- **Prior catalog.** A long-lived branch keeps to its own catalog; on its first
  push it inherits from the closest base branch that has one. A feature branch
  takes each model build profile from the first catalog whose builds carry
  every current compile fingerprint of that profile: its own, then its base branches
  in order. A feature branch that merged `develop` therefore reuses both its
  own builds and the builds `develop` recompiled. A profile no catalog matches
  compiles. A sibling feature branch is never a source.
- **Merge.** A merge is a push to `develop`, which compares with its own
  catalog. Automatic builds changed by the merge compile again on `develop`,
  and their benchmarks run again; builds the branch compiled are not reused.
- **Newer push.** It cancels the running orchestrator of the same ref
  (manual dispatches run to completion). Builds that already published stay;
  builds left pending are planned again. Dispatched DevKit runs continue. A
  benchmark still running when the next plan runs has no completed run yet,
  so it is dispatched again.
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

In staging, `publish-candidate` waits for every invocation of the selected
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
the only implemented promotion target; Main/HF release remains follow-up work.

Develop approval retains S3 storage. The future Develop → Main flow in
[#315](https://github.com/sima-neat/models/issues/315) and
[#293](https://github.com/sima-neat/models/issues/293) needs a separate release
gate and backend. It must transfer the same approved deployment to Hugging Face,
verify every file at an immutable revision, and only then update Main's selection
and active download location. S3 cleanup must preserve other references,
including Develop. Pushes never upload to Hugging Face. The backend resolves
approved roots; the CLI downloads and verifies files.

## LLiMa builds

The compile matrix selects `scripts/run_llima_compilation.sh` for a Hugging
Face component. The shared downloader assembles its pinned, verified file
set; the recipe calls `llima-compile` and records a standalone `compile.sh` with
the complete arguments it used. The runner then runs
`llima-deploy` into `<model>/<profile>/<component>/deployment/`, containing
`devkit/`, `elf_files/`, any `npy_files/`, and that script. `verify-artifact --write-manifest`
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
