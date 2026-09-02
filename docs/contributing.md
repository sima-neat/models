# Contributing with Git and GitHub

This guide covers the contribution boundary after a model works through the
local checks in [Local development and testing](local-development.md).

## Keep the change reviewable

Create a focused branch from the current integration branch and keep model
lifecycle inputs separate from unrelated changes:

```bash
git fetch origin
git switch -c feat/add-<model-id> origin/develop
```

A typical new-model change contains:

```text
models/<model-id>/README.md
models/<model-id>/model.yaml
models/<model-id>/compile.py
models/<model-id>/surgery.py              # only when declared
models/<model-id>/tests/correctness.yaml
models/<model-id>/tests/performance.yaml
```

A taxonomy change may also be necessary, but it changes a repository-wide
published contract. Explain it separately and follow the immutable versioning
rules in [Model categorization taxonomy](model-taxonomy.md).

Put common lifecycle behavior in `tools/`; do not copy download, cache,
checksum, MPK validation, or publication code into a model directory. Changes
to shared tooling, schemas, dependency pins, and workflows affect more than one
model and should include corresponding unit or contract tests.

## Never commit generated or sensitive artifacts

Do not add:

- ONNX files or other source model binaries;
- calibration, training, or correctness datasets;
- MPKs, compiler output, build directories, benchmark output, or logs;
- virtual environments, caches, or Python bytecode;
- credentials, tokens, cookies, signed URLs, or private keys;
- hand-authored generated catalog, package, checksum, or publication metadata.

Store approved large assets through an immutable location supported by the
registry source policy and declare its URI, exact byte size, SHA-256, license,
and provenance.
The repository `.gitignore` is a guardrail, not permission to omit a review of
staged files.

## Pre-commit checklist

From the repository root:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id>
model-registry fetch-calibration models/<model-id>
model-registry validate-catalog --repository .
model-registry matrix --repository . --changed-path models/<model-id>/model.yaml
model-registry resolve-compiler --repository .
git status --short
git diff --check
git diff --cached --stat
```

Before committing, confirm:

- the directory name, model ID, variant IDs, and test IDs follow naming rules;
- the source and ready calibration data fetch and verify from a clean cache;
- ownership, license, redistribution, attribution, and upstream revision are
  explicit;
- tensor names, dtypes, shapes, layouts, preprocessing, and postprocessing
  match the pinned source and intended Neat route;
- recipe and surgery behavior is deterministic and all inputs are declared;
- every intended supported variant appears in the matrix;
- each intended variant was compiled with the pinned Model Compiler when that
  environment was available, and its output passed `model-registry verify-mpk`;
- the exact locally compiled MPK was exercised by the declared performance test
  on a compatible Modalix DevKit when one was available;
- local compiler and DevKit result summaries are ready for the pull request,
  while MPKs, JSON results, and logs remain unstaged;
- unsupported variants include durable qualification evidence;
- pending calibration or test evidence says exactly what remains;
- no downloaded, generated, or secret material is staged.

Commit messages should describe the model or lifecycle contract being added,
not just the filenames changed.

## Commit and push the reviewed inputs

Stage only the source-controlled model definition and recipe files you reviewed.
Do not use the generated MPK or local benchmark output as Git inputs.

```bash
git add models/<model-id>
git diff --cached --check
git diff --cached
git commit -m "feat(models): add <model-id>"
git push --set-upstream origin HEAD
```

Open a pull request against `develop`. Repository automation reruns validation
and matrix selection from the committed revision. It does not compile or
benchmark pull-request code. Trusted post-merge workflows perform those later
stages; authors do not invoke those publication steps themselves.

## Pull request description

Give reviewers enough evidence to reproduce and assess the contribution:

- model purpose, upstream project, immutable revision, and conversion history;
- license and redistribution assessment;
- source and calibration SHA-256 values;
- input/output and pre/postprocessing decisions;
- compile variants, precision, target, and why each variant is needed;
- model-specific surgery or compiler workarounds;
- correctness datasets, thresholds, measured results, and remaining gaps;
- performance configuration and any existing Modalix measurements;
- exact local commands run and their results;
- compiler or DevKit checks not run locally, with the environment limitation
  that prevented them;
- known limitations, unsupported combinations, and follow-up work.

Link the model's tracking issue and any upstream model card or conversion PR.
Screenshots are not a substitute for machine-readable logs or result files.

## Contributor tool boundary

The `model-registry` package installed from `models-utils` exposes the
validation, fetch, matrix, dependency, and MPK verification commands used in
this guide. Its contributor command surface is intentionally non-publishing.
Model authors compile through `models/<model-id>/compile.py` and benchmark
through the DevKit `benchmarker` command.

Do not run or document direct use of `model-registry publish`,
`publish-benchmark`, or `publish-taxonomy` as part of a model
contribution. Those commands require controlled credentials and provenance and
are reserved for CI.

## CI and trust boundaries

Pull requests run discovery, schema/taxonomy validation, dependency resolution,
and deterministic matrix construction on GitHub-hosted infrastructure. They do
not schedule contributor-controlled recipe or surgery code on a self-hosted
Vulcan runner.

For a model change, interpret a successful pull-request workflow as confirmation
that:

- the complete catalog and taxonomy are structurally valid;
- the repository-wide Model Compiler dependency resolves;
- the comparison base and changed models were identified; and
- the resulting supported-variant matrix is deterministic.

It is **not** evidence that the recipe compiled, the MPK passed validation,
accuracy met a threshold, or the model ran on a DevKit. Those claims must come
from stated local evidence or the trusted staging workflow after merge.

## After merge: trusted staging automation

When the change is merged to `develop`, the trusted push workflow selects the
affected variants and then:

1. authorizes each requested runner class against the protected Vulcan
   environment specification;
2. verifies the pinned Model Compiler runner and fetches immutable source,
   calibration, and applicable accuracy data;
3. runs each selected supported recipe in an isolated output directory;
4. evaluates declared dataset-backed accuracy when it applies;
5. requires and validates exactly one MPK;
6. publishes the compilation and provenance to the staging registry; and
7. dispatches each newly compiled Modalix variant to the dedicated DevKit
   benchmark workflow, which publishes the successful result.

The compile and benchmark are separate GitHub Actions runs. A successful
compile does not by itself mean the dispatched benchmark succeeded; inspect the
per-variant `Vulcan Model Benchmarking` run and the model's staging registry
record. If trusted compilation or benchmarking fails, follow up with a focused
manifest or recipe change and reproducible evidence.

Production is an explicit protected workflow choice. Targeted manual variants
publish independently without updating shared mutable catalog pointers.

## Review and merge expectations

A model is ready to merge when its declarative contract is accurate, local and
PR checks pass, and reviewers accept its licensing, ownership, test plan, and
remaining maturity gaps. Merge does not automatically mean production
approval. Compilation, device evidence, and promotion remain separate trusted
stages.

If trusted compilation exposes a model-specific failure, update the recipe or
manifest with reproducible evidence and tests. Use `status: unsupported` only
after a combination is proven unusable, not as a shortcut around an unresolved
failure.
