# CI flow

Every workflow in `.github/workflows/`, job by job: what triggers it, what each job runs,
which environment it targets, and which jobs hold credentials. Read it before you change a
workflow, diagnose a run, or reason about what a push will do. What the jobs write to the
registry is in [publication](publication.md). How a branch's catalog decides what compiles
is in [branch catalogs](branch-catalogs.md).

## Workflows

| Workflow | File | Trigger | Does |
| --- | --- | --- | --- |
| Target Branch Checks | `target-branch-checks.yml` | pull request | Lints, tests, builds the author package, and checks the model browser. |
| PR Sanitization | `pr-sanitization.yml` | pull request opened, edited, reopened, or updated | Calls the `pr-sanitization.yml` workflow of `sima-neat/.github`. What it checks is defined there. |
| Vulcan Model Orchestrator | `vulcan-model-orchestrator.yml` | every push, manual dispatch | Plans, compiles, benchmarks, and publishes the branch catalog. |
| Vulcan Model Compile | `vulcan-model-compile.yml` | called by the orchestrator | Compiles, tests, publishes, and benchmarks one build. |
| Vulcan Model Benchmarking | `vulcan-model-benchmarking.yml` | called by the compile workflow and the orchestrator, manual dispatch | Runs the requested stages of one build on a DevKit and publishes each report. |
| Vulcan Model Convert | `vulcan-model-convert.yml` | manual dispatch | Converts one model and publishes the bundle. |
| Build and Publish Model Registry | `vulcan-model-registry.yml` | push that changes tooling, models, schemas, tests, or workflows, manual dispatch | Builds and publishes the author package. |
| Model Registry Package | `model-registry-package.yml` | called | Lints, tests, and builds the author package. Publishes nothing. |
| Developer Docs Validation | `developer-docs.yml` | called, manual dispatch | Builds and smoke-tests the portal. |
| Deploy Model Browser & Docs | `deploy-model-browser.yml` | manual dispatch | Publishes the portal to staging or production after an owner approves. |
| Model branch lifecycle | `model-branch-lifecycle.yml` | branch created or deleted, manual dispatch | Cancels a deleted branch's runs and purges or restores its registry state. |
| Vulcan Model Registry State | `vulcan-model-state.yml` | push to `main` that changes `state.json`, the workflow, or its script, manual dispatch | Publishes `state.json`. |
| Promote model | `model-promotion.yml` | manual dispatch | [Promotion operations](promotion-operations.md) |
| Publish Main model metadata | `model-main-catalog.yml` | manual dispatch | [Promotion operations](promotion-operations.md) |

## Pull request checks

`Target Branch Checks` runs on GitHub-hosted runners with `contents: read` and no secrets.

| Job | Calls | Runs |
| --- | --- | --- |
| `package` | `model-registry-package.yml`, job `build` | `ruff check .`, `python -m pytest`, a collect-only pass over every model's stage-output tests, `packaging/model-registry/build.sh`, and `verify_package.sh` |
| `portal` | `developer-docs.yml`, job `build` | `npm ci`, the TypeScript check, `npm run build:portal`, `node --test tests/browser/*.test.mjs`, and the Chromium smoke test |

A passing pull request shows that the definitions validate, the host tests pass, and the
package and portal build. It compiles nothing, verifies no MPK, and measures nothing.

## Push: Vulcan Model Orchestrator

Every push runs the orchestrator on the pushed revision. A push that changes no fingerprint
compiles and benchmarks nothing and still publishes the catalog.

| Job | Runs when | Does |
| --- | --- | --- |
| `select` | always | Runs `validate-models`. Reads `vars.VULCAN_ENV`. Resolves the Model Compiler pin (`resolve-compiler`) and the base branches (`resolve-base`). Sets `publishable` to false when the dispatch names profiles or components. |
| `portal` | always, in parallel | Calls `developer-docs.yml` with the staging configuration and uploads the checked site. |
| `resolve-vulcan-config` | after `select` | Resolves the selected environment's region, buckets, and role settings with the shared resolver. |
| `resolve-benchmark-runner` | always | Resolves the DevKit runner once. Every benchmark of the run receives it. |
| `resolve-staging-browser-config` | `main` and `develop` | Resolves the staging site settings. |
| `publish-model-browser` | `main` and `develop`, after `portal` and `resolve-staging-browser-config` | Publishes the checked site to staging. |
| `plan` | after `select` and `resolve-vulcan-config` | Assumes the publisher role and runs `plan-stages`. Outputs the compile matrix, the benchmark matrix of reused builds, the candidate matrix, and the conversion selections. It also writes: `plan_conversions` creates and finalizes one `model_compile_plan` run per conversion-backed model, through `--api-url`, which defaults to `VULCAN_ARTIFACT_REGISTRY_API_URL`. |
| `authorize-runner-classes` | the plan compiles something | Reads the environment specification with the config reader role. Runs `authorize-runner-matrix`, which rejects a runner class or market the model has no approval for. |
| `initialize-catalog` | after authorization, when `publishable`, or on any branch except `develop`, `main`, and `master` whose run builds an on-demand model | Runs `publish-catalog --selected-matrix`. The catalog moves to this revision and lists each build whose fingerprint changed as pending. |
| `compile` | one matrix entry per selected build, `fail-fast: false` | Calls `vulcan-model-compile.yml`. |
| `publish-candidates` | the plan holds on-demand builds that are not LLiMa, after `compile` succeeds or fails | [On-demand builds](on-demand-builds.md) |
| `publish-catalog` | `publishable`, after every `compile` call ends | Publishes the final snapshot for the revision. |
| `benchmark` | `publishable`, after `initialize-catalog`, for each reused build that lacks a benchmark | Calls `vulcan-model-benchmarking.yml`. Runs while the changed builds compile. |
| `benchmark-without-compiles` | nothing compiles, after `publish-catalog` | The same call, for a run with no pending catalog. |
| `report-partial-publication` | the plan compiles something | Lists each `Compile` or `Publish` job that did not succeed and fails the run. |

Three orderings matter:

- A reused build's benchmark restores the build from the catalog at this revision. It
  therefore waits for `initialize-catalog`, or for `publish-catalog` when nothing compiles.
- A new build's benchmark starts inside its own `compile` call as soon as the build
  publishes. Other builds may still compile.
- `publish-catalog` waits for every `compile` call, including each new build's benchmarks.
  Each build has already published itself by then. A failed build stays pending.

The benchmark matrix keeps only `modalix` targets. Each build's jobs are grouped under the
name `<model>/<compile invocation> (<build profile>)`. A build's Compile and Publish jobs
can succeed while its benchmark jobs fail, so read each build's benchmark jobs too.

### Dispatch inputs

| Input | Effect |
| --- | --- |
| `models` | Limits planning to those models. Only a dispatch that names models passes `--include-on-demand`. |
| `profiles`, `components` | Limit the plan further. The run publishes compile runs but no package, no catalog entry, and no benchmark. An on-demand model needs both `models` and `profiles`, and still publishes its candidate. |
| `base_branch` | Replaces the resolved base branches with this one branch. |
| `force_rebuild_all` | Recompiles every selected build whatever its fingerprint. [Forced rebuilds](publication.md#forced-rebuilds) need a new commit once a package is published. |

A forced rebuild of one model also applies a changed model test at once. Without it, a
changed test runs the next time its stage runs.

## Compile workflow

`vulcan-model-compile.yml` handles one build in three jobs.

### `compile` (Compile)

Runs on a self-hosted Model Compiler runner. Its labels name the environment, the compiler
label, the runner class, the runner market, and `model-id-…`, `build-profile-id-…`, and
`component-id-…`. The timeout is 360 minutes.

1. `scripts/record_compile_runner_identity.sh` reads the instance identity and fails when
   the instance's market differs from the selected one.
2. The job compares the runner's installed compiler version with the pin and fails on a
   mismatch, before any model code runs.
3. It installs the Dataset Registry wheel, the registry tooling, and pytest into temporary
   folders outside the read-only compiler image.
4. It runs the runner script in the activated compiler environment:
   `scripts/run_model_compilation.sh`, or `scripts/run_llima_compilation.sh` for LLiMa.
5. `verify-artifact --write-manifest` checks each component and writes its
   `compile-result.json` with provenance: sources, calibration, compiler, fingerprint,
   commit, and runner.
6. `scripts/run_model_tests.sh compile` runs the model's compile tests.
7. The job uploads the output as `compiler-output-<model>-<profile>-<invocation>`.

A failing test stops the job before the upload, so the build stays unpublished.

In production, when the run's plan selected any conversion, every compile job of the run
gets a restricted registry session before step 4 and a fresh one before step 5. The
condition is the run's `conversion_sources`, not the build's own source
([private production reads](publication.md#private-production-reads)).

### `publish` (Publish)

Runs on a GitHub-hosted runner for an ONNX build. An LLiMa build uses a self-hosted ARM64
Ubuntu runner of the default class. The job is skipped for an on-demand build that is not
LLiMa, because `publish-candidates` publishes those.

1. `validate-compile-outputs` checks the complete component set.
2. The job assumes the publisher role.
3. `model-registry publish` records one compile run per component.
4. `assemble-package --all` and `publish-package` assemble and record the package.
5. `publish-catalog --compiled-dir --package-dir` records the build in the branch catalog.
   The step runs the command once more when another publisher keeps winning the snapshot
   conflict.
6. The job uploads `compiled-model-<model>-<profile>-<invocation>` for the benchmark.

Steps 4 to 6 run only when `publishable` is true. An on-demand build replaces steps 3 to 6
with `publish-candidate` ([on-demand builds](on-demand-builds.md)).

### `benchmark` (Benchmark)

Calls `vulcan-model-benchmarking.yml` with this run as `source_run_id`. It runs when the
build is `publishable`, is not on-demand, published a package, targets `modalix`, and
declares a benchmark stage.

## Benchmark workflow

`vulcan-model-benchmarking.yml` benchmarks one build profile. The orchestrator passes the
environment configuration and the DevKit runner. A manual dispatch resolves both in the
jobs `resolve-vulcan-config` and `resolve-benchmark-runner`.

### `stage` (Stage benchmark input)

Runs on a GitHub-hosted runner.

1. It validates the inputs. `source_branch` must be the branch the workflow runs on, and
   `source_sha` must be a full commit SHA.
2. With a `source_run_id`, it checks that run's commit, branch, and event, and requires
   exactly one unexpired `compiled-model-…` artifact.
3. It checks out the tooling at the workflow revision and the source revision in `_source`
   as data.
4. `model-registry benchmark-job` prints the backend (`core` or `llima`) and the DevKit
   job's timeout: 45 minutes, 90 for LLiMa performance, and 4320 for a speculative profile
   with a `benchmark_baseline`.
5. `scripts/require_source_board_tooling.sh` fails when the workflow revision and the
   source revision differ in a file the benchmark fingerprints hash. The newer push then
   benchmarks the build.
6. `scripts/stage_benchmark_inputs.sh` restores the build from the registry when no
   source run supplies it, and restores the baseline deployment for a performance stage
   ([restoring a published build](publication.md#restoring-a-published-build)).
7. `model-component-artifacts` verifies each component, and the job requires the package
   receipt. It uploads the result as `benchmark-input-<model>-<profile>-<invocation>`.

### `benchmark` (Benchmark on DevKit)

Runs on the resolved Modalix DevKit. Its scratch folder is
`/media/nvme/neat-tests/model-benchmark-<run>-<attempt>-<model>-<profile>`, and each runner
script removes it on exit.

1. It installs the registry tooling with the `tests` extra in a virtual environment.
2. It keeps each requested stage the profile declares and skips the others.
3. `resolve-benchmark-runtime` reads the pins, and the job fails when `DISTRO_VERSION` in
   `/etc/buildinfo` differs from `platform-version`.
4. For the `core` backend it installs the Dataset Registry wheel, assumes the dataset
   reader role, and runs `scripts/cache_accuracy_dataset.sh`. The script refuses to run
   when the NVMe is not mounted. It holds a board-wide lock, so concurrent jobs download an
   archive once.
5. `model-registry ensure-neat-core --repository _source --environment production`
   prepares Core.
6. For the `llima` backend, `scripts/install_mole_benchmark_runtime.sh` installs MoLE from
   the LLiMa pin and reuses a matching installation on the NVMe.
7. `scripts/run_model_performance.sh` measures performance. It needs step 5, and step 4 or
   step 6.
8. `scripts/run_model_accuracy.sh` measures accuracy. It needs steps 4 and 5.
9. After each measurement, `scripts/run_model_tests.sh` runs the model's test of that
   stage on `model-report.json` and writes `passed`, `failed`, or `skipped` to
   `test-status.txt`.
10. The job uploads `performance-…` and `accuracy-…` evidence, also after a failure.

`ensure-neat-core` reads `neat-core.install_target`. It resolves a floating target such as
`core@develop` to the published tag once, then compares `neat --json` with that artifact's
metadata. Installation is skipped only when the Core channel, tag, and artifact environment
match, and the installed Runner and Scheduler versions equal the Debian versions in the
artifact's checksum-verified `resolved-deps-manifest.json`. Any missing or unreadable value
installs with `sima-cli neat install --env production --type minimal`. An unresolved
floating target fails the job. Both paths print the PyNeat version and the installed
packages. Core artifacts come from production whatever the registry environment is. A
non-production Core index needs `--artifacts-base-url` or `NEAT_ARTIFACTS_BASE_URL`.

`run_model_performance.sh` does this for the `core` backend:

1. lists the MPKs with `model-component-artifacts`
2. resolves the first accuracy dataset from the NVMe cache with `resolve-devkit-accuracy`
   and unpacks it, and passes no `--dataset` for a profile that accuracy does not measure
3. fetches the model's pinned `resources` with `fetch-resources` and passes `--resources`
4. installs `scripts/requirements-performance.txt`, when present, beside PyNeat's
   environment
5. sets the MLA clock to 1 GHz and the CPU clock to 1.4 GHz, and stops when either does
   not hold
6. runs `performance.py` and copies its output to `performance.log`
7. runs `check-report`, which validates the report and writes `environment.json`

For the `llima` backend it skips steps 2 to 4 and passes `--deployment`, plus `--baseline`
when the staged input holds `benchmark-baseline.json`.

`run_model_accuracy.sh` resolves every declared dataset, lists the MPKs with
`model-component-mpks`, passes the build's `runtime_files` folder as `--resources` when the
compile declared any, installs `scripts/requirements-accuracy.txt`, runs `accuracy.py`, and
runs `check-report`.

### `publish` and `check`

`publish` runs on a GitHub-hosted runner when at least one stage recorded a test outcome.
It assumes the publisher role and runs `publish-report` for each such stage, with the
source branch and source SHA ([model reports](publication.md#model-reports)).

`check` fails the run when a published stage's tests failed. It is a separate job, so
re-running failed jobs repeats the check and publishes nothing twice. After a fully
successful DevKit job it deletes the run's `benchmark-input`, `performance`, and
`accuracy` artifacts. A failed stage keeps its evidence.

Each stage publishes on its own outcome. A failed measurement publishes nothing, and the
next push runs only the stage that has no run. A failed test still publishes its report,
so the stage is not measured again.

### Manual dispatch

Run the workflow from the branch named in `source_branch`.

| Input | Meaning |
| --- | --- |
| `environment` | `default` uses `vars.VULCAN_ENV`, or staging when that is unset. `staging` and `production` override it. |
| `source_run_id` | The orchestrator run that holds the `compiled-model-…` artifact. Empty restores the build from the registry. |
| `source_sha` | The revision whose model definition and scripts are measured. For a restore from the branch's own catalog it must be that catalog's current revision. |
| `deployment_branch` | Another branch whose catalog supplies the compiled build, for example `develop` when testing a feature branch. |
| `stages` | `performance`, `accuracy`, or both. A stage the profile does not declare is skipped. |
| `promoted_missing` | Queues the missing benchmarks of promoted builds and ignores the per-model inputs. |

With `promoted_missing`, the `promoted` job runs `scripts/benchmark_promoted_models.py` on
`develop` or `main`. `plan-promoted-benchmarks` selects each on-demand model with an
approved deployment whose tasks exclude `text_generation` and `visual_question_answering`.
It queues one separate dispatch per profile that lacks a completed run. The job summary
lists the queued and skipped profiles. Benchmark an LLM or VLM with a single-model dispatch
([LLiMa builds](llima-builds.md)).

## Conversion workflow

`vulcan-model-convert.yml` takes `environment`, `model`, and `conversion`, which must be
`default`.

| Job | Does |
| --- | --- |
| `config` | Resolves the selected environment. |
| `convert` | Runs on a self-hosted ARM64 Ubuntu runner of that environment, with a 90 minute timeout. Its GitHub environment is the input value. A workflow comment says the shared resolver returns the aliases `stg` and `prod` instead. |

The `convert` job runs `scripts/run_model_conversion.sh`, which builds a fresh Python 3.12
environment from the model's conversion requirements and runs `convert.py`. It then runs
the model's conversion tests and continues only when the outcome is `passed`. A skipped or
failed test publishes nothing. Last, it assumes the publisher role and runs
`publish-conversion` ([conversions](publication.md#conversions)).

Nothing runs this workflow automatically, and a conversion is not copied between
environments. Run it in each environment that needs the conversion, before a model's
source changes to `conversion` there.

## Registry package workflow

`vulcan-model-registry.yml` runs on a push that changes `deps/manifest.json`, `models/`,
`schemas/`, `taxonomies/`, `datasets/`, `artifact-browser/`, `.github/workflows/`,
`scripts/`, `tests/`, `conftest.py`, `packaging/model-registry/`, `tools/model_registry/`,
`tools/model_report/`, or `pyproject.toml`.

| Job | Does |
| --- | --- |
| `resolve-vulcan-config` | Resolves `vars.VULCAN_ENV`, or staging when unset. |
| `build` | Calls `model-registry-package.yml`. |
| `publish` | Publishes the package files under `models-utils/model-registry` with the shared artifact publisher. |
| `promote-latest` | On a branch, calls the shared `vulcan-update-latest-artifacts.yml`, which defines what the `latest` tag moves to. |

The package holds `model_registry` without the CI-only modules, plus `model_report`
([module map](architecture.md#module-map)).

## Model browser publication

| Path | Jobs | Destination |
| --- | --- | --- |
| Automatic | `portal`, `resolve-staging-browser-config`, `publish-model-browser` in the orchestrator | Staging, on every orchestrator run whose ref is `main` or `develop`, push or dispatch, whatever `VULCAN_ENV` is |
| Manual | `authorize`, `resolve-vulcan-config`, `build`, `publish` in `deploy-model-browser.yml` | Staging or production, for the `developer` or `customer` audience |

Both paths publish the checked `model-browser-site` artifact under the `models` prefix and
invalidate `/models` and `/models/*`. They share the concurrency group
`<repository>-model-browser-<environment>` without cancellation, so two uploads never
overlap. The group does not protect an upload from the run-level cancellation: a newer
push to `main` or `develop` cancels that ref's running push run, including
`publish-model-browser` during its upload.

No push publishes to production. Production is reached through the manual workflow, or
through `scripts/deploy-production-model-browser.sh` from an operator's machine.

The manual workflow runs from `main` or `develop`. `authorize` waits on the
`model-browser-deploy` environment and runs
`scripts/authorize-model-browser-deployment.cjs`. The script requires one reviewer team
with administrator bypass disabled, a first attempt, and an approval by the dispatching
actor. Every later job also requires attempt 1 and the original actor, so a rerun cannot
reuse an approval. The manual publish keeps files that the new site no longer has.
[The browser README](../artifact-browser/README.md) owns the release and rollback steps.

## Other workflows

### Branch lifecycle

`model-branch-lifecycle.yml` has two jobs. `cancel-branch-runs` runs
`scripts/cancel_branch_runs.sh` for a deleted branch and cancels its unfinished runs.
`update-registry` calls the registry's `purge` endpoint for a deleted branch and `restore`
for a created one, then waits up to five minutes for a purge to complete. It refuses
`main`, `develop`, and `release-*`. It does not refuse `integration/*`, although
`is_long_lived` treats those branches as long-lived. The job runs once for the staging
registry and once for the production registry, each in its own GitHub environment. A
failure in one registry does not stop the other.

### Registry release state

`state.json` holds `current_state` (`GA`, `Beta`, or `Alpha`) and `default_branch`
(`develop` or `main`). `vulcan-model-state.yml` publishes it to production as
`models/state.json`. Both jobs and `scripts/publish_registry_state.sh` require `main`, so a
merge to `develop` publishes nothing. The script validates both fields, uploads with
`Cache-Control: no-store`, and compares the uploaded file with the source. No fingerprint
includes this file. Readers outside this repository consume the published file, so keep
its two fields and its no-store policy when you change the script.

## Which environment a run targets

| Run | Registry and storage environment | Notes |
| --- | --- | --- |
| Orchestrator, push or dispatch | `vars.VULCAN_ENV`, `staging` or `production` | An unset or other value fails `select`, and with it every job that needs `select`, before a compiler runner is requested. `portal`, `resolve-benchmark-runner`, and the staging browser jobs do not need `select` and still run. No input overrides it. The value also selects the compiler runners and names the GitHub environment of each publishing job. |
| Benchmark called by the orchestrator | the orchestrator's environment | Reads, datasets, and report publication stay in the registry that holds the build. |
| Benchmark, manual | the `environment` input | See [manual dispatch](#manual-dispatch). |
| Conversion | the `environment` input | |
| Registry package | `vars.VULCAN_ENV`, or staging when unset | |
| Portal build and automatic publication | staging | |
| Registry release state, Main metadata | production | |
| Branch lifecycle | staging and production, one job each | A branch can hold runs in either registry. |

Every benchmark runs on a DevKit with the `vulcan-staging` label, for either registry
environment. Routing does not enter any fingerprint.

Compile and benchmark jobs set `SIMA_DATASET_S3_URI` to the selected environment's dataset
storage. How `dataset-registry` uses it, and what fills that storage, is defined in the
Dataset Registry repository.

Registry reads use `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` and fall back to
`VULCAN_ARTIFACT_REGISTRY_API_URL`. In production the jobs also set
`MODEL_REGISTRY_IAM_READ_API_URL` to the API URL. `read_registry_json_authenticated` in
`registry_http.py` then signs only requests under that URL's `/v1/` path. Other hosts and
artifact downloads stay unsigned. Production can therefore plan against a private registry
with no public read URL.

## Concurrency and cancellation

| Workflow | Group | A newer run |
| --- | --- | --- |
| Orchestrator, push | `model-compile-<ref>-<VULCAN_ENV>` | cancels the running one, with its compiles and benchmarks |
| Orchestrator, dispatch | the same, plus the run ID | runs beside it, and each dispatch runs to completion |
| Benchmarking | environment, ref, source SHA, source run, stages, model, profile, invocation | waits, never cancels |
| Target Branch Checks | `target-branch-checks-<ref>` | cancels |
| Registry package | `model-registry-<workflow>-<ref>` | cancels on a branch |
| Browser publication | `<repository>-model-browser-<environment>` | waits |
| Branch lifecycle (per branch), registry state, Main metadata | one group each | waits |
| Promote model | the run ID | runs beside it |

A cancelled orchestrator leaves published builds and benchmark runs in place. The next run
plans the builds left pending and the benchmarks without a completed run
([concurrent pushes](branch-catalogs.md#concurrent-pushes)). Deleting a branch cancels its
unfinished runs.

## Trust boundary

The orchestrator has no `pull_request` trigger. A self-hosted job therefore starts only
from a push to this repository or a manual dispatch.
`tests/test_target_branch_checks.py` asserts that only `Target Branch Checks` and
`PR Sanitization` have a pull request trigger.

| Job | Runner | Model code it runs | Credentials |
| --- | --- | --- | --- |
| `package` and `portal` on a pull request | GitHub-hosted | the host tests of each model | none, with `contents: read` |
| `select` | GitHub-hosted | none | none |
| `plan`, `initialize-catalog`, `publish-candidates`, `publish-catalog` | GitHub-hosted | none | the publisher role |
| `authorize-runner-classes` | GitHub-hosted | none | the config reader role |
| Compile | self-hosted Model Compiler runner | `compile.py` or `compile.sh`, and the compile tests | no publication step. In a production run with conversions, a session limited to `GET` on ONNX artifact metadata. No step configures dataset credentials. |
| Publish (compile workflow) | GitHub-hosted, or self-hosted for LLiMa | none | the publisher role |
| `stage` and `promoted` | GitHub-hosted | none | the publisher role, for production registry reads only |
| Benchmark on DevKit | DevKit | `performance.py`, `accuracy.py`, and the benchmark tests | the dataset reader role, passed only to the caching step, which runs no model code |
| `publish` (benchmark workflow) | GitHub-hosted | none | the publisher role |
| `convert` | self-hosted | `convert.py` and the conversion tests | the publisher role, assumed after conversion and its tests |
| Publish jobs of the release state, branch lifecycle, and Main metadata workflows | GitHub-hosted | none | the publisher role of their target |
| Package and browser publish jobs | set by the shared workflows in `sima-neat/.github` | none in this repository | the role this repository passes as `role_to_assume` |

The three self-hosted jobs that run model code (Compile, Benchmark on DevKit, and `convert`)
each request `id-token: write` and name a GitHub environment. What model code could obtain
there is limited by the roles' trust policies, which live outside this repository.

Four mechanisms inside this repository hold the boundary:

- The compile job has read-only repository access and never publishes. A separate job
  verifies its output again and publishes it.
- Model builds carry no compiler pin of their own. `select` derives the runner label from
  `deps/manifest.json`, and the compile job checks the runner's version against it.
- Benchmark jobs take the tooling from the workflow revision and the model definition from
  the source revision, checked out as data. Publication always uses the workflow
  revision's tooling.
- Each credentialed job names the GitHub environment it targets, so that environment's
  variables and protection rules apply.
