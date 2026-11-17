# Publication

What a run writes to the Vulcan registry: the run types, the files each stage hands to its
publisher, the checks before a write, catalog snapshots, conversions, and what can never
change afterward. Read it before you change a publisher in `tools/model_registry`, a
published document's shape, or a step that passes output between jobs. Which job calls
each publisher is in [CI flow](ci-flow.md).

## Registry runs

Every published result is a run in the registry. Runs are shared by all branches. A
branch's catalog names the runs that branch delivers.

| Run type | Command and function | `run_fingerprint` | Holds |
| --- | --- | --- | --- |
| `model_conversion` | `publish-conversion`, `publish_conversion` in `conversion.py` | the conversion fingerprint | input `upstream`, artifact `onnx` of type `model-onnx`, test result `correctness/conversion` |
| `model_compile_plan` | `plan-stages`, `plan_conversions` in `conversion.py` | a digest of the workflow run ID, commit, model, and conversion fingerprint | input `conversion`: the selected conversion's exact object |
| `model_compile` | `publish`, `publish_compilations` in `registry.py` | a digest of the run's parameters and the artifact SHA-256 | input `source-component`, artifact `mpk` or `llima` of type `model-pack`, test result `compile/mpk-validation` or `compile/deployment-validation` |
| `model_package` | `publish-package`, `publish_packages` in `registry.py` | a digest of the package parameters and the artifact SHA-256 | one input `component-<id>` per compile run, artifact `package` of type `model-package`, the fidelity report, and the MLA stage simulation |
| `model_performance`, `model_accuracy` | `publish-report`, `publish_report` in `report_publication.py` | the stage's benchmark fingerprint | parent: the profile's package run. See [model reports](#model-reports). |
| `model_documentation` | `publish-catalog`, `publish_model_documentation` in `registry.py` | a digest of the README and hero parameters | artifacts `readme` and the hero image |

A compile run stores the compile fingerprint in the parameter `build_fingerprint`, because
Vulcan promotion reads it under that name. A package's `package_fingerprint` is
`canonical_sha256` of the manifest's content, which differs from the SHA-256 of the
`manifest.json` file.

Rules every publisher follows:

- Each request carries an idempotency key derived from what it publishes. Compile and
  package keys include the workflow run ID and not the attempt, so a new attempt of the
  same workflow run resumes the same registry run. The registry rejects a key longer than
  200 characters, so a catalog snapshot key (`catalog_snapshot_key` in `registry.py`)
  replaces a branch name that would overflow it with the first 16 hex digits of the
  name's SHA-256.
- `S3ArtifactStore.ensure_native_sha256` uploads with a native SHA-256 checksum, verifies
  the stored object, and reuses an existing object whose checksum and size match. One
  object may be at most 5 GiB.
- Each artifact record names the storage key, the exact object version, the SHA-256, and
  the size.
- A build publishes on its own. A sibling build's failure neither hides nor rolls back a
  completed run.

An on-demand build publishes the same runs inside a candidate
([on-demand builds](on-demand-builds.md)).

## Compile outputs

A compile leaves this tree. The compile job uploads it, and the publish job reads it.

```text
<model_id>/<build_profile>/
  compile-outputs.json            what the invocation produced
  compile-timing.json             start and end of the recipe
  fidelity.json                   when model.yaml declares numerical_fidelity
  mla-stage-simulation.json       from simulate-mla-stages
  <component_id>/
    <name>_mpk.tar.gz             or deployment-manifest.json and deployment/ for LLiMa
    compile-result.json           from verify-artifact --write-manifest
    registry-publication.json     the compile run ID, written by publish
  packages/default/
    registry-publication.json     the package run, copied in after publish-package
```

`compile-outputs.json` holds `model_id`, `build_profile_id`, `compile_invocation_id`, and
`outputs`. Each output names its `component_id` with `mpk_path` and `mpk_sha256`, or
`artifact_path` and `artifact_sha256` for an LLiMa deployment index. An invocation with
several components also lists `derived_graphs`, one SHA-256 per component. `runtime_files`
is optional. For a single-component model, `run_model_compilation.sh` writes the file when
the recipe does not.

`validate_compile_outputs` in `artifact.py` rejects a missing or extra component, a path
outside its component folder, a wrong SHA-256, and any MPK the file does not declare.

Before any registry write, `_preflight_compilations` in `registry.py` also requires that:

- each `compile-result.json` names this model, profile, component, commit, and environment
- an MPK has the name `mpk_filename` returns, `<model_id>_<profile_id>_mpk.tar.gz`, with
  `_<component_id>` before `_mpk` when the profile has several components
- the compiler version and label equal the pin in `deps/manifest.json`
- `source_sha256` equals the declared source's identity
- the recorded compile fingerprint equals the one computed from the checkout

## Packages

Every build profile publishes one package, `default`, with all its components.
`assemble_package` in `package.py` builds it from the compile output.

| Build | Package | Contents |
| --- | --- | --- |
| ONNX | `<model_id>_<profile_id>_default.zip`, a deterministic, uncompressed ZIP | each MPK at the top level, the compile's `runtime_files`, and each `resources` entry under its `install_path` |
| LLiMa | the deployment folder with `manifest.json` at its root | [LLiMa builds](llima-builds.md) |

`package-result.json` records the package manifest, validated against
`schemas/package-manifest.schema.json`. The ZIP does not contain the manifest. The manifest
lists the components with their compile run IDs, the resources, every installed file with
its SHA-256, the provenance shared by all components, and the license.

`publish_packages` refuses a package when:

- its components differ from the declared ones, or a component has no compile run ID
- the license's `redistribution` is `prohibited` or `unknown`
- `fidelity.json` is missing although `model.yaml` declares `numerical_fidelity`, or
  present although it does not
- the fidelity report does not match the declared dataset, selection, and outputs

The package run carries the fidelity report as the `numerical-fidelity` artifact, a
`correctness/numerical-fidelity` test result, and one measurement per metric and output.
It carries `mla-stage-simulation.json` as an artifact with a
`performance/mla-stage-simulation` test result. That result is `skipped` when the
simulator could not run a component.

## Model reports

`performance.py` and `accuracy.py` write `model-report.json`, which follows
`schemas/model-report.schema.json`. The model, build profile, MPK digests, and board are
not report fields. Publication records them on the run.

`check_report` runs on the DevKit and again before publication. It requires:

| Report | Requirement |
| --- | --- |
| accuracy | exactly the datasets `model.yaml` declares, each with the metrics its `datasets/<id>.yaml` requires for the model's tasks, and every score within its metric's bounds |
| performance, whole model | every metric `metrics.json` marks `required`, and the components in `model.yaml` order |
| performance, `scope: components` | the required metrics for each component |
| LLiMa performance | `generation` workloads with `ttft_ms` and `decode_tps`, plus `prefill_tps` for text workloads and `avg_power_watts` telemetry. A speculative report must use the declared `benchmark_baseline`. |

`publish_report` then publishes one run per stage:

- The run's parent is the profile's package run. The staged input must hold each
  component's compile receipt and a package receipt that links exactly those runs.
- It has one `compiled-model-<component>` input per component, and `baseline-model-…`
  inputs for a speculative LLiMa report.
- Its artifacts are `model-report`, Core's report as `core-benchmark-report` (one per
  component for `scope: components`, `llima-benchmark-report` for LLiMa), and `<stage>-log`.
- Its test result `<stage>/model` holds the whole report as the summary and the model
  test outcome as the status. An LLiMa summary leaves out per-request timings, which stay
  in the artifact.
- It has one measurement per metric, named by the metric's `registry_name`. An accuracy
  measurement has a `dataset` dimension. A component measurement has a `component_id`
  dimension. An LLiMa measurement has workload, role, and decoding dimensions.
- The run environment records the DevKit's clocks, CPU governor, and kernel from
  `environment.json`, which `check-report` wrote on the board.

A run completes even when its model tests failed, so that stage is not measured again. The
`check` job fails the workflow afterward.

The catalog lists each profile's benchmark fingerprints under `benchmark_fingerprints`. The
model browser resolves the runs by those fingerprints. A profile with no completed run
shows "Not measured at this revision."

## Model documentation

`publish-catalog` publishes each model's README and hero as one `model_documentation` run.
When the README, the hero, and their metadata equal the branch's current catalog entry, the
new catalog reuses that run. Its receipt then sets `source_commit` to the current commit
and keeps the first commit as `publication_commit`. Changed documentation gets a new run,
with objects stored under a branch-scoped content hash.

## Catalog snapshots

A catalog is one document per branch and revision. `build_catalog_snapshot` in
`registry.py` builds it, and `publish_catalog` posts it.

| Field | Content |
| --- | --- |
| `git_ref`, `catalog_revision` | the branch and the commit |
| `taxonomy`, `taxonomy_sha256` | the taxonomy of this checkout |
| `models[]` | `id`, `display_name`, `description`, `license`, `maturity`, `categories`, `documentation`, `builds`, `packages`, and `benchmark_fingerprints` |
| `models[].builds[]` | `build_profile_id`, `compile_invocation_id`, `component_id`, `compile_fingerprint`, `artifact_sha256`, `run_id` |
| `models[].packages[]` | `package_id`, `build_profile_id`, `run_id`, `artifact_sha256`, `size_bytes`, `package_fingerprint`, `component_run_ids` |
| `models[].build_on_demand`, `build_requirements` | on-demand models only ([on-demand builds](on-demand-builds.md)) |
| `pending_builds[]` | each build the revision still has to compile, with its compile fingerprint |

The registry adds `approved_deployments` and `branch_deployments` itself. A snapshot may
omit them and may not carry another branch's.

One orchestrator run publishes the catalog at three points:

1. `initialize-catalog` starts the revision. It seeds the document from the prior catalog
   ([branch catalogs](branch-catalogs.md)) and lists each build that will compile as
   pending. A pending profile keeps its prior builds and their benchmark fingerprints, so
   the previous build stays visible.
2. Each build's Publish job adds its builds and package and clears its pending entry.
3. `publish-catalog` writes the final snapshot. It carries metadata-only changes, and when
   nothing compiles it is the catalog the reused builds' benchmarks read.

Publication is a compare-and-swap. Each snapshot names the snapshot it replaces, and the
registry rejects it with `catalog_snapshot_conflict` when another publisher got there
first. `publish_catalog` then reads the winner's catalog, adds its own build, and tries
again for up to 180 seconds (`CATALOG_CONFLICT_RETRY_SECONDS`).

Within a revision, a snapshot may only clear pending entries whose builds it carries
(`_remaining_pending_builds`). A build that the revision never marked pending needs a new
commit.

A model with a conversion source is refreshed only when the run's plan selected its
conversion. An unselected one keeps its previous catalog entry, with fresh documentation
and categories. A new conversion-backed model enters the catalog once a plan selects its
conversion.

On `develop` and `main`, `_refresh_approved_benchmarks` recomputes the benchmark
fingerprints of each approved deployment from its producing runs.

The model browser reads the current snapshot, its runs, and download URLs through the
registry's read API.

## Conversions

### Publication

`publish_conversion` requires passed conversion tests. `bundle_conversion` builds a
deterministic `onnx.tar.gz` with the declared outputs, every external tensor file an ONNX
output references, and a `manifest.json` that lists each file's SHA-256 and size. It fails
when a declared output is missing and runs the ONNX checker on every `.onnx` file. Nothing
shared checks a `.pt` or `.pth` output.

The bundle is stored under `models/conversions/<model_id>/<bundle sha256>/`. The run keeps
the artifact name `onnx` and type `model-onnx` whatever the bundle holds. Its
`run_fingerprint` is the conversion fingerprint, and its parameters record the
`conversion` declaration and outputs.

### Selection

`plan-stages` resolves a conversion for every planned model with
`compilation.source: conversion`, including on-demand models, before it decides reuse.

1. It creates or resumes a `model_compile_plan` run for this workflow run, commit, model,
   and conversion fingerprint.
2. A resumed plan keeps its recorded conversion. A new plan resolves the completed
   `model_conversion` run with the current fingerprint.
3. It requires the artifact to be available, validated, and versioned, then records it as
   the plan's `conversion` input.

A missing conversion fails with "run Vulcan Model Convert first". Planning never falls
back to an older fingerprint. A recorded object that changed fails too, and so does a plan
run that is neither running nor completed. Start a new workflow run in that case.

The selections travel as `MODEL_REGISTRY_CONVERSIONS` to the compile, publish, candidate,
and catalog jobs. They hold exact object versions, not signed URLs. `fetch_conversion`
asks for a fresh download URL for that exact artifact and verifies the bundle and each
member. The compile fingerprint uses the bundle's SHA-256 only, so identical bytes from
another run or environment do not change it. The compile run records the conversion as a
versioned input.

Leave a successful conversion alone. The registry resolves the latest completed conversion
for a fingerprint. A repeated Convert whose bundle bytes differ therefore recompiles that
model on every branch.

To reproduce an old build with the ONNX CI compiled, save the original plan's
`conversion_sources` JSON map and use it with the same checkout and compiler:

```bash
export MODEL_REGISTRY_CONVERSIONS="$(cat conversion-sources.json)"
model-registry fetch models/<model-id> --json
scripts/run_model_compilation.sh models/<model-id> <profile> <profile> /tmp/compiled
```

Without the map, `apply_selection` in `conversion.py` fails with `resolve conversion with
plan-stages before compilation`. The plan run's `/provenance` endpoint also returns the
map: its `conversion` input records the producer, the artifact IDs, and the content
metadata, and retained compile inputs record the same selection. Use the original
environment's reader. Do not resolve the current conversion instead.

### Retention

These are Vulcan behaviors that this repository relies on and cannot check:

- A plan protects its selected conversion for 24 hours, which covers the six-hour compile
  limit and queue time. An abandoned plan holds nothing after that.
- After publication, the compile run's retained input protects the conversion's bytes and
  provenance, also when the branch that converted is deleted.
- A conversion with no consumer follows branch retention (`retention_class: branch` in
  `publish_conversion`). A newer conversion does not delete an older one.
- After the plan window, an unpublished build has no guarantee. Start a new workflow run
  when its conversion is gone.

### Private production reads

When a selection's read URL is the production IAM API, `fetch_conversion` signs its
artifact metadata request, also on a cache hit.
The compile job assumes the registry role with an inline session policy that allows only
`GET /v1/runs/*/artifacts/onnx`. It passes those credentials to the compile step as
`MODEL_REGISTRY_READ_AWS_*`, so the runner's own dataset credentials stay in place. A
compile can outlast the session. The job therefore assumes a fresh session with the same
policy right before `verify-artifact`. The artifact body comes from a version-specific
download URL. Version, checksum, size, and validation checks apply on both paths.

## Restoring a published build

A benchmark of a reused build has no workflow artifact. `stage-registry-build`
(`stage_catalog_build` in `registry.py`) rebuilds the compile output tree from the registry:

1. It reads the branch's current catalog and requires its revision to equal `source_sha`.
   With `deployment_branch`, it reads that branch's catalog and requires the build's
   compile fingerprint to equal the checkout's.
2. It downloads each component's MPK from its compile run and verifies size and SHA-256.
   For LLiMa it downloads the deployment index and every listed file.
3. It writes `compile-result.json`, the component's `registry-publication.json`, and a
   `compile-outputs.json` like the one a compile writes.
4. When the package records `runtime_files`, it downloads the package ZIP, checks it
   against the catalog, and restores only the declared files.
5. It writes the package receipt that `publish-report` needs.

With `--benchmark-baseline` it restores the LLiMa deployment a profile's
`benchmark_baseline` names and writes `benchmark-baseline.json`.

## What is immutable

| Thing | Rule |
| --- | --- |
| A completed run | Immutable in the registry. |
| A stored object | Every record names an exact object version. A publisher reuses an object only when its checksum and size match. |
| A package within a catalog revision | Cannot be removed or replaced, also not by initialization. |
| A selected conversion | A plan refuses a recorded object that changed. |
| Report schema versions | `model-report.json` and the fidelity report keep `schema_version` 1, because the browser and `sima-cli` choose a renderer by it. |

Documents that only this repository's tooling reads carry no version: `model.yaml`,
`compile-outputs.json`, and the package manifest. Change their readers and writers
together.

### Forced rebuilds

`force_rebuild_all` recompiles the selected builds. `initialize-catalog` passes
`--force-rebuild-all` to `publish-catalog`. When the catalog is already at this revision
and holds a package of a selected automatic profile, the command fails before any compile
starts. Dispatch the forced rebuild from a new commit. A plain retry of initialization,
without the flag, keeps the packages the revision already published.

At initialization, a selected profile's prior package is left out of the new revision.
The rebuilt package must link the new compile runs.

### Branch deletion

Deleting a branch calls the purge endpoint of the staging and the production registry,
and creating one calls restore in both
([branch lifecycle](ci-flow.md#branch-lifecycle)). Each purge reports how many runs and
registered object versions it removed. Vulcan decides what a purge keeps, such as a
conversion that a compile input retains or a protected candidate.
