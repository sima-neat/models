# On-demand builds

How CI schedules, publishes, and selects a build of a model that sets
`build-on-demand: true`. Read it before you change the planner, candidate publication, or
the catalog fields that carry on-demand state, and when a manual build does not publish.
To promote or withdraw a finished candidate, follow
[promotion operations](promotion-operations.md). The author's steps are in
[On-demand builds](../docs/adding-a-model.md#on-demand-builds).

A **candidate** is one complete build profile of an on-demand model: every compile run
and package run of that profile, retained by Vulcan as a unit. Its `deployment_id`
identifies it. A **selection** is a catalog entry that points one build profile at one
finalized candidate.

## What the flag changes

`build-on-demand` is a top-level boolean in `model.yaml`, default `false`
(`schemas/model.schema.json`). It changes scheduling and catalog metadata. No fingerprint
reads it (`compile_fingerprint` and `benchmark_fingerprint` in
`tools/model_registry/fingerprints.py`).

| Where | Effect | Source |
|---|---|---|
| compile matrix | each entry of the model carries `build_on_demand: true` | `build_matrix` in `compile_matrix.py` |
| stage plan | the entry compiles only with `--include-on-demand`, and never enters the benchmark plan | `plan_stages` in `stage_plan.py` |
| catalog | the model entry carries `build_on_demand: true` and `build_requirements` | `build_catalog_snapshot` in `registry.py` |
| registry runs | compile and package runs carry `metadata.build_on_demand: true`, and their artifacts use `retention_class: on-demand` instead of `branch` | `publish_compilations`, `publish_packages` in `registry.py` |
| publication | `publish` and `publish-package` fail with `on-demand publication requires publish-candidate` | same functions |

A push validates an on-demand model and publishes its catalog metadata. It compiles and
benchmarks nothing for that model, even when its inputs change.

## Dispatch a build

Dispatch **Vulcan Model Orchestrator** (`.github/workflows/vulcan-model-orchestrator.yml`)
with both `models` and `profiles` set.

- The `plan` job adds `--include-on-demand` to `model-registry plan-stages` when the run
  is a manual dispatch and `models` is not empty.
- With `--include-on-demand`, `plan_stages` fails with `on-demand builds require explicit
  --models and --profiles` when the selection holds an on-demand model and either
  selector is empty. Without the flag it skips on-demand entries and reports nothing.
- `plan_stages` compiles an invocation unless the prior catalog's raw `builds` carry its
  compile fingerprint for every component. The producer adds no new `builds` for an
  on-demand model, so a skip depends on what the catalog Vulcan serves lists there.
  `force_rebuild_all` always compiles.
- The run publishes in the environment `VULCAN_ENV` names, `staging` or `production`. The
  `select` job fails when the variable holds any other value. Candidate publication works
  in both environments (`publish-candidate --environment` in `cli.py`).

A dispatch that sets `profiles` or `components` is not `publishable` (`select` job, step
"Decide whether this run updates the catalog"). The run then skips the ordinary package
and catalog updates. Two things still happen for an on-demand build:

- On a feature branch, `initialize-catalog` publishes the branch catalog before the
  compile, so the catalog holds the model's `build_requirements` when the candidate is
  selected.
- The candidate publishes its own compile runs and packages, as described below.

## Publish a candidate

`model-registry publish-candidate` (`publish_candidate` in
`tools/model_registry/candidates.py`) publishes one model and build profile. Which job
runs it depends on the compile runner:

| Build | Job | Runner |
|---|---|---|
| LLiMa (`runner_script` is `scripts/run_llima_compilation.sh`) | `Publish` in `vulcan-model-compile.yml` | self-hosted ARM64 |
| every other on-demand build | `publish-candidates` in `vulcan-model-orchestrator.yml`, one matrix job per model and profile | `ubuntu-latest` |

The `plan` job builds `candidate_matrix` from the on-demand compile entries that do not
use the LLiMa runner. `publish-candidates` starts after every compile job ends, whether
it passed or failed, and its matrix sets `fail-fast: false`. One candidate therefore
publishes even when another model or profile failed.

The `Compile` job uploads `compiler-output-<model>-<profile>-<invocation>` only after the
model's compile tests pass, and a build profile has one compile invocation. A failed
`Compile` job therefore leaves no output for its profile. For a LLiMa build, `Publish`
needs `compile` and is skipped. For any other build, `publish_candidate` requires the
output directory and every component of the profile, and fails without them.

`publish_candidate` runs these steps in order:

1. Check the inputs. The model must be on-demand, the output directory must sit outside
   the compiler output, and the compiler output must hold every component of the profile
   exactly once.
2. Build the contract, `{"version": 1, "roles": ...}`. Each component gets a role
   `compile-<component_id>` and each package a role `package-<package_id>`. A role lists
   the artifacts and inputs its run registers. Keep the two in step when you change what
   a compile or package run registers. The package role omits `mla-stage-simulation`,
   which `publish_packages` registers when the profile output holds a simulation file.
3. Stop when the contract exceeds 32 roles or 4,096 required names. No registry call has
   happened yet.
4. Create the candidate with `POST /v1/repositories/sima-neat/models/deployments`. The
   body holds the branch ref, commit, model, profile, and contract.
5. Read the candidate back and compare its `contract_digest` with the local contract.
6. Return a candidate that is already `finalized`, provided it is still `registered` and
   protected. Otherwise fail and ask for a new build run.
7. Fail when `assembly_expires_at` has passed. A new workflow run then publishes a new
   candidate.
8. Publish each component with `publish_compilations`. It creates the `model_compile`
   run, binds it to the candidate under its role, and only then registers the run's
   inputs, parameters, and artifacts.
9. Assemble the profile's packages with `assemble_package` and publish them with
   `publish_packages`, which binds each `model_package` run the same way.
10. Finalize with `POST .../deployments/<deployment_id>/finalize` and the contract digest.
11. Read the candidate again. Fail unless it is `finalized`, has a `manifest_digest`, is
    `registered`, and carries `protection`.

New uploads go under the candidate's `upload_prefix`. Files with the same SHA-256 and
file name upload once within a candidate (`_publish_deployment_files` in `registry.py`).

The job summary shows the `deployment_id`, the manifest digest, the source ref and commit,
and the protection Vulcan returned (`_publish_candidate` in `cli.py`).

### What a candidate leaves out

- The model's README and hero. `publish-catalog` publishes them per branch
  (`publish_model_documentation` in `registry.py`), so a candidate's protection does not
  retain them.
- DevKit results. Benchmarks run separately, as described under
  [Benchmarks](#benchmarks). A report artifact always uses `retention_class: branch`
  (`_put_file` in `report_publication.py`), so the candidate's protection does not
  cover it.
- Approval. A finalized candidate is not approved or promoted.

`publish_model_documentation` reuses a completed documentation run when the README, the
hero, and their presentation metadata match the branch's current catalog. The receipt's
`source_commit` then names the current checkout and `publication_commit` the commit that
first published the files.

### Retries and receipts

The candidate's idempotency key is a hash of the GitHub run ID, the environment, and the
request body. A rerun of the same workflow run therefore resumes the same candidate. A
new workflow run creates a new one.

Both publishing jobs upload `candidate-<model>-<profile>-<attempt>` with
`candidate.json`, `publication.json`, `branch-publication.json`, and `uploads/*.json`,
even when publication fails. GitHub keeps it and the compiler output for 14 days. A rerun
downloads the receipts of earlier attempts first. `S3ArtifactStore` then reuses each
saved upload and fails with `start a new candidate` when the saved object changed.

### What Vulcan decides

The registry server sets these, and this repository cannot check them:

- The assembly window lasts seven days.
- The newest successful candidate per source branch, model, and profile receives
  `latest_successful` protection. An older candidate that finishes later receives a
  bounded protection.
- Deleting the source branch does not remove a protected candidate.
- A candidate holds at most 4,096 S3 dependencies and a 4 MiB frozen manifest. The
  producer checks only the role and name limits in step 3.

## Catalog fields

### `build_requirements`

`build_catalog_snapshot` writes `build_requirements` for an on-demand model each time it
builds that model's catalog entry, including before any build exists.

A conversion-backed model gets a new entry only when the run resolved its conversion.
`_catalog_models` in `registry.py` reads the selections from `MODEL_REGISTRY_CONVERSIONS`,
which the orchestrator's `plan` job fills for the models it plans. For an unselected
model, `build_catalog_snapshot` copies the previous entry with its old
`build_requirements`. A model without a previous entry stays out of the catalog.

Each build profile maps to:

```text
version: 1
compiler_version: <model-compiler.version from deps/manifest.json>
invocations: [{id, component_ids, compile_fingerprint}]
packages:    [{id, component_ids, layout, format}]
```

`layout` and `format` are `directory` and `llima` for a LLiMa package, `archive` and `zip`
otherwise. The fingerprints are the checkout's ordinary compile fingerprints. A candidate
ID never enters a fingerprint.

### `approved_deployments` and `branch_deployments`

Vulcan owns both fields (`SERVER_OWNED_SELECTIONS` in `registry.py`). Each maps a build
profile to a selection with the candidate's `deployment_id`, `requirements`, and
`artifacts`. Promotion writes `approved_deployments` on develop and main.
`publish-branch-deployment` asks Vulcan to write `branch_deployments` on a feature branch.

- The producer never writes either field. `publish_catalog` strips them from any model
  entry it carries over from another ref, because the registry rejects a snapshot that
  holds another ref's selections.
- Readers resolve a selection through `_catalog_outputs` in `registry.py`. A branch
  selection wins over an approved one for the same profile, and both replace the
  profile's raw `builds` and `packages`. A completed run that is not selected cannot
  replace a selection.
- The planner reads an inherited `approved_deployments` entry as the profile's published
  fingerprints (`_carries_fingerprints` in `stage_plan.py`).

## Select a candidate on its feature branch

After finalization, both publishing jobs run `model-registry publish-branch-deployment`
on every branch except `develop`, `main`, and `master`. `publish_branch_deployment` in
`candidates.py` does this:

1. Read `candidate.json`. Fail unless it is `finalized` with a `deployment_id` and a
   `manifest_digest`, and its `git_ref` is a feature branch.
2. Read the branch's current catalog. Fail with `publish the feature-branch model
   requirements before selecting its deployment` when the branch has none.
3. `POST .../branch-deployments` with the ref, the `deployment_id`, the `manifest_digest`,
   and the catalog's `canonical_sha256` as `expected_snapshot`.
4. On `catalog_snapshot_conflict`, read the catalog again and retry, up to eight attempts.
   Each attempt derives a new key from its body.
5. Save the result as `branch-publication.json`.

A failed selection can be retried by rerunning the job. It does not recompile. The
selection makes the candidate the branch's download for that profile. It does not approve
the candidate for develop or main.

## Never synthesize approval

Approval exists only as a Vulcan promotion record in state `promoted` with a
`result_snapshot`. Producer code must keep to these rules:

- Write `build_requirements` from the checkout only. Never copy a candidate's
  requirements into a target catalog to make it eligible.
- Leave `approved_deployments` and `branch_deployments` to the server.
- Treat a finalized candidate, a passed compile, and an approved GitHub job as separate
  facts. None of them is an approval.

## Benchmarks

No push or build benchmarks an on-demand build. `plan_stages` removes on-demand entries
from the benchmark plan, and the `Benchmark` job in `vulcan-model-compile.yml` requires
`!build_on_demand`. Dispatch **Vulcan Model Benchmarking** for the model and profile
instead.

Its `promoted_missing` input queues the missing benchmarks of promoted builds on
`develop` or `main` ([manual dispatch](ci-flow.md#manual-dispatch)). It queues a profile
only for stages its catalog entry lists under `benchmark_fingerprints`
(`plan_promoted_benchmarks` in `stage_plan.py`). The promotion report job and
`publish_catalog` write those through `_refresh_approved_benchmarks` in `registry.py`.

## Forced rebuilds

`force_rebuild_all` compiles an on-demand profile whatever the catalog holds, and the run
publishes a new candidate. The rule that a published package of an automatic build needs
a new commit does not apply to on-demand profiles
([forced rebuilds](publication.md#forced-rebuilds)).
