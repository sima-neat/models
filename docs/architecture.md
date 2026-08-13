# Model Registry architecture

The Model Registry repository owns small, reviewable lifecycle inputs:

- versioned JSON Schemas and policy;
- `models/<model-id>/model.yaml` definitions and model-local documentation;
- optional model-local deterministic surgery hooks;
- shared lifecycle tooling and runtime test code;
- CI orchestration, test definitions, baselines, and publication manifests.

It does not store ONNX files, calibration datasets, compiled model packs, logs,
or benchmark output. Those remain in immutable object storage and are referenced
by URI, byte size, and SHA-256.

## Component boundaries

| Component | Responsibility |
| --- | --- |
| GitHub Actions | Selects changed models, invokes lifecycle commands, and reports status. Untrusted pull requests use validation only. |
| Vulcan | Creates ephemeral, pinned Model Compiler runners and provides staging artifact storage and registry APIs. |
| Model Compiler runner | Verifies inputs, prepares and compiles one declared variant, and emits checksummed result manifests. |
| Kerrigan/Fleet Manager | Selects eligible Modalix DevKits, owns leases, schedules tests, and propagates cancellation. |
| DevKit agent | Downloads verified runtime artifacts, runs correctness/performance tests, and reports structured results. |
| S3/object storage | Stores immutable source cache entries, compiled artifacts, logs, and result manifests. |
| Artifact-registry API | Ingests provenance and results using authenticated, idempotent requests. Workflows never receive PostgreSQL credentials. |

## Directory contract

```text
.github/workflows/       Validation, compilation, DevKit, and publication workflows
deps/                    Repository-wide toolchain dependency pins
docs/                    Architecture, contribution, benchmark, and release policy
schemas/                 Versioned model and result JSON Schemas
tools/model_registry/    Shared host-side CLI and lifecycle implementation
tools/runtime/           Shared ARM64/Neat device-side runners
models/<model-id>/       Declarative metadata and genuinely model-specific assets
tests/                   Schema fixtures, unit tests, and contract tests
```

Common download, checksum, cache, discovery, MPK validation, publication, and
result logic must stay under `tools/`. A model directory owns a compile recipe
and may contain surgery only when the transformation is specific to that model
and cannot be expressed as shared parameters. The recipe's product is one
directly consumable `*_mpk.tar.gz`, not an ONNX or compiler intermediate.

The canonical product name is **Model Registry**. Code and paths use
`model-registry` or `model_registry`. The word `modelzoo` is used only when
documenting compatibility with an existing external CLI or legacy service.

## Lifecycle

```text
model.yaml
  -> schema and policy validation
  -> immutable fetch and SHA-256 verification
  -> deterministic preparation/surgery
  -> pinned Model Compiler build on Vulcan
  -> immutable staging artifacts and compile result
  -> Kerrigan correctness/performance sessions
  -> release gates
  -> artifact-registry API publication
```

## Compile workflow trust boundary

Pull requests run catalog discovery and schema validation on GitHub-hosted
runners only. The Vulcan job is restricted to trusted pushes and explicit
maintainer dispatches. Production is manual and uses the protected GitHub
`production` environment. Before requesting any self-hosted runner, the select
job validates `deps/manifest.json`, resolves its exact Model Compiler version,
and derives the `model-compiler-<version>` runner label. The selected runner's
`/etc/vulcan/model-compiler-artifact.json` version must also match before any
model code executes. Model variants do not carry independent compiler pins.

## Artifact publication contract

Compilation matrix jobs exchange outputs through short-lived GitHub artifacts;
they do not write S3 directly. A GitHub-hosted packaging job revalidates every
MPK and its provenance, then emits root and per-model `sima-cli` metadata. The
standard organization publisher writes the result below
`models/<encoded-branch>/<short-commit>/`, including `manifest.json` and the
repository `branches.json` index. Only after publication succeeds does the
model-aware `models/<encoded-branch>/latest.json` index publish and the standard
promotion workflow update `models/<encoded-branch>/latest.tag`. The CLI rejects
the pointers when `latest.json.catalog_tag` does not equal `latest.tag`, making
a partial promotion visible instead of silently mixing catalog revisions.

The root `metadata.json` exposes compiled model variants as selectable
resources so users can choose one or more. Each `<model-id>/metadata.json`
provides a stable direct-install target for that model. `catalog.json` carries
the same inventory for automation without interactive `sima-cli` selection.

To preserve incremental compilation without copying large MPKs, the selection
job pins the current branch's versioned `latest.json`, falling back to the
resolved base branch on a branch's first build. The base is explicit for pull
requests and manual dispatches and otherwise resolved from the closest remote
branch by Git graph distance, with `develop` winning ties. The comparison starts
at the merge base rather than treating GitHub's all-zero new-branch `before` SHA
as a request to rebuild the entire catalog. Each model entry owns an immutable
`latest_tag`, `artifact_branch`, build timestamp, per-variant provenance, and a
deterministic build fingerprint over model-local inputs, shared build inputs,
and the pinned Model Compiler version. Changed model outputs are stored under
the current commit; unchanged entries continue to reference their prior
immutable build. When no fingerprint requires compilation, the workflow still
publishes complete metadata and a branch index without leasing a Vulcan compiler
runner. Root metadata and `catalog.json` remain complete and use absolute
artifact URLs. Direct model installs use the model-aware index.

Cleanup collects references from all current model indexes before applying age
retention, so an older model build remains protected while any live index uses
it. A malformed index causes cleanup to fail closed for the models repository.
