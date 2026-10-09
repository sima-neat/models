# Submit for review

This guide covers the contribution boundary after a model works through the
required checks in [Qualify locally](local-development.md). It
owns Git hygiene, pull-request evidence, review, and what happens after merge;
it does not repeat the local qualification procedure.

## Entry requirement

Do not start submission until every supported component/profile pair in the change has
completed the local qualification loop: each build compiled, its MPKs
validated, and its declared performance and accuracy stages run on a DevKit.
Keep the resulting MPKs, fidelity and accuracy reports, performance reports,
and logs outside Git. You will summarize that evidence
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
models/<model-id>/model.yaml                        # identity, builds, fidelity, accuracy, benchmark
models/<model-id>/scripts/compile.py
models/<model-id>/scripts/performance.py
models/<model-id>/scripts/accuracy.py               # only with a devkit_script accuracy entry
models/<model-id>/scripts/requirements-accuracy.txt # optional, with accuracy.py
models/<model-id>/scripts/convert.py                # only with a declared conversion
models/<model-id>/scripts/requirements-convert.txt  # required with convert.py
models/<model-id>/scripts/surgery.py                # only when declared
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
- calibration, training, fidelity, or accuracy datasets;
- MPKs, compiler output, build directories, performance or accuracy output, or logs;
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
Do not use the generated MPK or local stage output as Git inputs.

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
and lists the declared builds from the committed revision. It does not compile
or measure pull-request code. Trusted post-merge workflows perform those later
stages; authors do not invoke those publication steps themselves.

## Pull request description

Give reviewers enough evidence to reproduce and assess the contribution:

- model purpose, upstream project, immutable revision, and conversion history;
- license and redistribution assessment;
- source, calibration, fidelity, and accuracy dataset identities and hashes;
- components, build profiles, packages, and any model-specific surgery or compiler workarounds;
- exact local commands and pass/fail summaries for each build's compilation,
  MPK verification, fidelity report, performance stage, and accuracy stage;
- the pinned Model Compiler, DevKit platform, Core, and dataset versions used;
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

Pull requests run catalog and taxonomy validation, compiler-pin and base-branch
resolution, and an offline stage plan on GitHub-hosted infrastructure. They do
not schedule contributor-controlled recipe, surgery, or measurement code on a
self-hosted runner. The offline plan reads no registry state, so it lists every
declared build rather than the ones the change will rerun.

For a model change, interpret a successful pull-request workflow as confirmation
that:

- the complete catalog, taxonomy, conversion sources, fidelity, accuracy, and
  benchmark entries, and declarative compositions are structurally valid;
- the repository-wide Model Compiler dependency resolves;
- the base branch was identified; and
- the declared builds form a deterministic plan.

It is **not** evidence that the recipe compiled, the MPK passed validation,
fidelity or accuracy met a threshold, or the model ran on a DevKit. Those claims must come
from stated local evidence or the trusted staging workflow after merge.

## After merge: trusted staging automation

When the change is merged to `develop`, the push runs **Vulcan Model Compile**
on the reviewed revision:

1. **Plan.** `plan-stages` compares each build's compile key with the branch
   catalog, or the base branch's catalog for a new branch. A build compiles
   when no published run carries its key. A build that is reused but lacks a
   completed performance or accuracy run for its current stage key is queued
   for measurement only.
2. **Authorize.** Each requested runner class is checked against the protected
   environment specification.
3. **Compile.** On a self-hosted compiler runner with read-only repository
   access, `scripts/run_model_compilation.sh` runs the recipe, and every MPK is
   verified with its provenance.
4. **Publish.** A GitHub-hosted job verifies the complete component set,
   publishes each component and its fidelity report as a compile run, starts
   the DevKit stages for a new Modalix build, assembles and publishes the
   packages the build completes, and records the build in the catalog.
5. **Publish the catalog.** The catalog snapshot for the revision is published
   and promoted.
6. **Measure.** For reused builds, the workflow starts only the DevKit stages
   they still lack.

**Vulcan Model Benchmarking** runs the requested `performance` and `accuracy`
stages of one build on a DevKit, then publishes each result as its own
registry run under the compile run.

The compile and the DevKit stages are separate GitHub Actions runs. A
successful compile does not by itself mean the dispatched benchmark succeeded;
inspect the `Vulcan Model Benchmarking` run of each build and the model's
staging registry record. If trusted compilation or measurement fails, follow
up with a focused manifest, recipe, or script change and reproducible evidence.

Production is an explicit protected workflow choice, and the DevKit stages run
only in staging. A manual dispatch can name models, build profiles, or
components and can force every selected build to recompile. A dispatch that
names build profiles or components compiles and publishes those builds but
does not publish packages or update the catalog.

## Review and merge expectations

A model is ready to merge when the local qualification evidence is complete,
PR checks pass, and reviewers accept the contract, licensing, ownership, test
plan, and remaining maturity gaps. Merge does not mean production approval;
trusted staging still verifies the committed revision, and promotion remains a
separate protected stage.

Reviewers also check that the change keeps the repository structure
consistent:

- every companion change listed in [keep the structure
  consistent](https://github.com/sima-neat/models/blob/develop/AGENTS.md#keep-the-structure-consistent)
  is in the same pull request;
- the description states which stages the change reruns, following [what a
  change reruns](adding-a-model.md#9-know-what-a-change-reruns);
- model-specific code is under `models/<model-id>/scripts/`, and shared code is
  in the module the [module
  map](https://github.com/sima-neat/models/blob/develop/design-doc/model-registry-architecture.md#shared-tooling)
  assigns to its job.

If trusted compilation exposes a model-specific failure, update the recipe or
manifest with reproducible evidence and tests. Use `status: unsupported` only
after a combination is proven unusable, not as a shortcut around an unresolved
failure.
