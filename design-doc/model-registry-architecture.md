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
| Model-owned `compile.py` | Developer-facing local compile entry point. Reads one declared YAML variant and emits one MPK. |
| DevKit `benchmarker` | Developer-facing local performance entry point for the exact compiled MPK. |
| `model-registry` | Author-safe `sima-cli` package for validation, verified input handling, matrix inspection, and MPK checks. CI uses the full internal command surface for packaging, provenance, and publication. |
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
design-doc/              Internal architecture and lifecycle design documents
docs/                    Developer contribution, metadata, and benchmark guides
schemas/                 Versioned model and result JSON Schemas
taxonomies/              Versioned sanctioned model categories and display labels
tools/model_registry/    Shared host-side CLI and lifecycle implementation
tools/runtime/           Shared ARM64/Neat device-side runners
models/<model-id>/       Declarative metadata and genuinely model-specific assets
tests/                   Schema fixtures, unit tests, and contract tests
```

Common download, checksum, cache, discovery, accuracy evaluation, MPK
validation, publication, and result logic must stay under `tools/`. A model directory owns a compile recipe
and may contain surgery only when the transformation is specific to that model
and cannot be expressed as shared parameters. The recipe's product is one
directly consumable `*_mpk.tar.gz`, not an ONNX or compiler intermediate.
Accuracy sampling, preprocessing interpretation, host executors, metrics, and
report generation are owned by `model-registry evaluate`; model directories
provide only schema-validated correctness YAML.

The canonical product name is **Model Registry**. Code and paths use
`model-registry` or `model_registry`. The word `modelzoo` is used only when
documenting compatibility with an existing external CLI or legacy service.

Model authors install the non-publishing `model-registry` command through
`sima-cli`; its installer resolves the Model Compiler environment and installs
alongside it without replacing compiler-owned dependencies. Local model
execution remains `compile.py` followed by `benchmarker`. Registry packaging
and publication remain automation-only operations and are absent from the
contributor command surface.

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

Each successful compilation matrix item verifies its MPK and provenance,
uploads the archive to a collision-resistant immutable S3 key with native
SHA-256, and upserts one Model Registry run through IAM-authenticated,
idempotent API calls. The run records the immutable source, canonical
model/variant/compiler parameters, taxonomy, exact S3 object version and
checksum, and MPK validation result before finalization. A sibling matrix
failure cannot roll back or hide a completed run.

The registry database is the discovery and lifecycle source of truth. Public
read APIs select current builds and return presigned download URLs. The model
browser and `sima-cli models` commands consume those APIs directly. Compilation
workflows do not assemble or promote Models-specific `metadata.json`,
`catalog.json`, `latest.json`, or `latest.tag` objects.

Changed-model selection comes from the Git diff against an explicit or resolved
base branch. The comparison starts at the merge base rather than treating an
all-zero new-branch SHA as a full rebuild. Build fingerprints remain provenance
on registry runs; workflow selection no longer reads or carries forward an
object-storage index.

Branch deletion and recreation call registry purge and restore endpoints.
Retention operates on registered runs and exact S3 object versions. Previously
published JSON objects may remain as historical artifacts but are not consulted
or destructively bulk-deleted.
