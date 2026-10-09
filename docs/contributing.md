# Submit for review

This guide covers the contribution boundary after a model works through the
required checks in [Qualify locally](local-development.md). It
owns Git hygiene, pull-request evidence, review, and what happens after merge;
it does not repeat the local qualification procedure.

## Entry requirement

Do not start submission until every supported component/profile pair in the change has
completed the local qualification loop. Keep the resulting MPKs, accuracy
reports, benchmark JSON, and logs outside Git. You will summarize that evidence
in the pull request so reviewers can see exactly what was tested.

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
models/<model-id>/model.yaml              # identity, builds, validation and benchmark settings
models/<model-id>/scripts/compile.py
models/<model-id>/scripts/performance.py
models/<model-id>/scripts/export.py       # only for an upstream checkpoint source
models/<model-id>/scripts/surgery.py      # only when declared
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

## Review, commit, and push the inputs

Stage only the source-controlled model definition and recipe files you reviewed.
Do not use the generated MPK or local benchmark output as Git inputs.

```bash
git add models/<model-id>
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "feat(models): add <model-id>"
git push --set-upstream origin HEAD
```

If the change includes a separately reviewed taxonomy or shared-tool update,
stage that explicit path too. Before committing, confirm that no downloaded
input, generated output, result, log, cache, or credential appears in the
staged diff. Write a commit message that describes the model or lifecycle
contract, not just the filenames.

Open a pull request against `develop`. Repository automation reruns validation
and matrix selection from the committed revision. It does not compile or
benchmark pull-request code. Trusted post-merge workflows perform those later
stages; authors do not invoke those publication steps themselves.

## Pull request description

Give reviewers enough evidence to reproduce and assess the contribution:

- model purpose, upstream project, immutable revision, and conversion history;
- license and redistribution assessment;
- source, calibration, and correctness dataset identities and hashes;
- components, build profiles, packages, and any model-specific surgery or compiler workarounds;
- exact local commands and pass/fail summaries for compilation, applicable
  accuracy, MPK verification, and Modalix benchmarking;
- pinned Model Compiler, DevKit, dataset, and test-profile versions used;
- known limitations, unsupported combinations, and follow-up work.

Link the model's tracking issue, upstream model card or conversion pull
request, and Dataset Registry change when new data was introduced. Keep the
machine-readable local results available for review; screenshots alone are not
reproducible evidence.

## Publication boundary

Model authors submit the reviewed contract; they do not publish registry state.
The `model-registry` publication commands require controlled credentials and
provenance and are reserved for CI.

## CI and trust boundaries

Pull requests run discovery, schema/taxonomy validation, dependency resolution,
and deterministic matrix construction on GitHub-hosted infrastructure. They do
not schedule contributor-controlled recipe or surgery code on a self-hosted
Vulcan runner.

For a model change, interpret a successful pull-request workflow as confirmation
that:

- the complete catalog, taxonomy, conversion sources, validation and benchmark
  entries, and declarative compositions are structurally valid;
- the repository-wide Model Compiler dependency resolves;
- the comparison base and changed models were identified; and
- the resulting supported component/profile matrix is deterministic.

It is **not** evidence that the recipe compiled, the MPK passed validation,
accuracy met a threshold, or the model ran on a DevKit. Those claims must come
from stated local evidence or the trusted staging workflow after merge.

## After merge: trusted staging automation

When the change is merged to `develop`, the trusted push workflow selects the
affected component/profile pairs or compile invocations and then:

1. authorizes each requested runner class against the protected Vulcan
   environment specification;
2. verifies the pinned Model Compiler runner and fetches immutable source,
   calibration, and applicable accuracy data;
3. runs each selected supported recipe in an isolated output directory, once
   for every matrix entry;
4. evaluates declared dataset-backed accuracy when it applies;
5. validates the output manifest and each emitted MPK, then uploads one complete
   compiler output;
6. downloads and verifies that output on a GitHub-hosted runner before publishing
   each component and its provenance to the staging registry;
7. assembles every complete declared package; and
8. dispatches each newly compiled Modalix component to the dedicated DevKit
   benchmark workflow, which benchmarks grouped components sequentially on one
   board and publishes each successful result.

The compile and benchmark are separate GitHub Actions runs. A successful
compile does not by itself mean the dispatched benchmark succeeded; inspect the
per-component/profile `Vulcan Model Benchmarking` run and the model's staging registry
record. If trusted compilation or benchmarking fails, follow up with a focused
manifest or recipe change and reproducible evidence.

Production is an explicit protected workflow choice. Targeted manual profiles
publish independently without updating shared mutable catalog pointers.

## Review and merge expectations

A model is ready to merge when the local qualification evidence is complete,
PR checks pass, and reviewers accept the contract, licensing, ownership, test
plan, and remaining maturity gaps. Merge does not mean production approval;
trusted staging still verifies the committed revision, and promotion remains a
separate protected stage.

If trusted compilation exposes a model-specific failure, update the recipe or
manifest with reproducible evidence and tests. Use `status: unsupported` only
after a combination is proven unusable, not as a shortcut around an unresolved
failure.
