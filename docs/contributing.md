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
models/<model-id>/model.yaml                        # identity, builds, fidelity, accuracy, performance
models/<model-id>/scripts/compile.py
models/<model-id>/scripts/performance.py
models/<model-id>/scripts/accuracy.py               # only with an accuracy declaration
models/<model-id>/scripts/requirements-accuracy.txt # optional, with accuracy.py
models/<model-id>/scripts/convert.py                # only with a declared conversion
models/<model-id>/scripts/requirements-convert.txt  # required with convert.py
models/<model-id>/scripts/surgery.py                # only when declared
```

A taxonomy change may also be necessary, but it changes a repository-wide
published contract. Explain it separately and follow the publication rules in
[Model categorization taxonomy](model-taxonomy.md).

Put common lifecycle behavior in `tools/`; do not copy download, cache,
checksum, MPK validation, or publication code into a model directory. Changes
to shared tooling, schemas, dependency pins, and workflows affect more than one
model and should include corresponding unit or contract tests. A model change
updates that model's own tests under `models/<model-id>/tests/` ([write the
model's tests](adding-a-model.md#10-write-the-models-tests)).

## Request additional compile resources

For expensive runners or on-demand purchasing, follow the [runner request
guide](runner-class-requests.md) before enabling the model's request. Link the
reviewed Vulcan policy PR, exact model IDs, owner, duration/memory or interruption
evidence, expected build frequency, and cost tradeoff in the Models PR. Update
the model README with the request. Larger size and on-demand are separate
permissions; an existing expensive-size grant does not authorize on-demand.

A passing Models PR validates the manifest, not the protected policy. Confirm
the grant and compatible workflows/scaler are deployed as described in the
[runner request guide](runner-class-requests.md#obtain-access-through-a-vulcan-policy-pr) before expecting a trusted compile to acquire a runner. A runner-only
change can reuse an existing build; use a narrowly selected forced staging build
only when deliberately collecting new runner evidence.

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

Open a pull request against `develop`. `Target Branch Checks` reruns
validation on the merge result. It does not compile or measure pull-request
code. Trusted post-merge workflows perform those later
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

Pull requests run `Target Branch Checks` on the merge result, on
GitHub-hosted infrastructure: the repository tests and every model's tests
that need no stage output, which also validate the complete catalog and
taxonomy and resolve the compiler pin, the `model-registry` package
build, and the model browser checks. They do not schedule
contributor-controlled recipe, surgery, or measurement code on a self-hosted
runner, and they read no registry state.

For a model change, interpret a successful pull-request workflow as confirmation
that:

- the complete catalog, taxonomy, conversion sources, fidelity, performance,
  and accuracy declarations, and declarative compositions are structurally
  valid;
- the repository-wide Model Compiler dependency resolves;
- the `model-registry` package builds; and
- the model browser builds and loads.

It is **not** evidence that the recipe compiled, the MPK passed validation,
fidelity or accuracy met a threshold, or the model ran on a DevKit. Those claims must come
from stated local evidence or the trusted staging workflow after merge.

## After merge: trusted staging automation

On-demand candidates require a separate manual **Promote model** dispatch to become approved
on Develop. Enter the model folder ID and a build profile ID from its `model.yaml`;
the workflow resolves the newest matching candidate receipt and pins it for review.
Pending requests mention `@sima-neat/owners-models` in the issue and send a Slack
reminder to the configured `SLACK_VULCAN_EVENT_CHANNEL_ID`. After the decision,
the same channel receives the backend outcome, including unsuccessful promotion attempts.
A successful build or metadata publication is not approval. See
[promoting a model](model-promotion.md) for the protected review and recovery procedure.

When the change is merged to `develop`, the push runs **Vulcan Model Orchestrator**
on the reviewed revision. Pushes still target staging; merging production-read
support does not switch `develop` to production. Production requires an explicit
manual dispatch with `environment=production`, not a change to `VULCAN_ENV`:

1. **Check the portal.** The model browser is built and smoke-tested in
   Chromium. If it fails, nothing is planned, compiled, benchmarked, or published
   for the revision. Every push runs this check, not only merges, and a
   push to `main` or `develop` publishes the checked browser to staging.
2. **Resolve configuration and plan.** `plan` waits for `select`, the portal
   check and `resolve-vulcan-config`. It requests OIDC permission; only a production
   dispatch assumes the selected GitHub environment's
   `VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN` before reading registry state.
   That publisher-capable role must trust the workflow and permit this repository's
   catalog/resolve reads, or planning fails before compilation. Staging skips this
   assumption and retains public reads. Read URLs prefer
   `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` and fall back to
   `VULCAN_ARTIFACT_REGISTRY_API_URL`; only production reads under the configured
   API base's `/v1/` path are SigV4-signed. Public-reader URLs, other hosts and
   artifact downloads retain their existing behavior. This enables private
   production reads without opening public access and does not grant credentials
   to pull-request validation. `plan-stages` compares each build's compile fingerprint with the
   branch's catalog and, on a feature branch, its base branches' catalogs. A
   build compiles when no published run carries its fingerprint. A build that
   is reused but lacks a completed performance or accuracy run for its current
   benchmark fingerprint is queued for that benchmark only.
3. **Authorize.** Each requested runner class and market is checked against the protected
   environment specification before a self-hosted compile is queued.
4. **Compile.** On a self-hosted compiler runner with read-only repository
   access, the backend's runner (`run_model_compilation.sh` for ONNX or
   `run_llima_compilation.sh` for LLiMa) runs the recipe. Each artifact is
   verified with its provenance, and the model's compile tests check the
   output. A failing test leaves the build unpublished.
5. **Publish.** A GitHub-hosted job verifies the complete component set,
   publishes each component and its fidelity report as a compile run, starts
   the benchmarks for a new Modalix build, assembles and publishes the
   packages the build completes, and records the build in the catalog.
6. **Benchmark.** For reused builds, the workflow starts only the benchmarks
   they still lack, once the catalog marks the selected builds pending. They
   run while the changed builds compile.
7. **Publish the catalog.** The catalog snapshot for the revision is published
   and promoted.

**Vulcan Model Benchmarking** runs the requested `performance` and `accuracy`
stages of one build on a DevKit, runs the model's benchmark tests on each result,
then publishes each result as its own registry run under the build profile's
package run, with the test outcome. A failed test fails the run after
publication.

The compile and the benchmarks are separate GitHub Actions runs. A
successful compile does not by itself mean the dispatched benchmark succeeded;
inspect the `Vulcan Model Benchmarking` run of each build and the model's
staging registry record. If trusted compilation or measurement fails, follow
up with a focused manifest, recipe, or script change and reproducible evidence.

Production is an explicit protected workflow choice, and the benchmarks run
only in staging. A manual dispatch can name models, build profiles, or
components and can force every selected build to recompile, which also
applies a changed model test at once; otherwise a changed test applies the next
time its stage runs. A dispatch that
names build profiles or components compiles and publishes those builds but
skips ordinary package and catalog updates. On-demand builds publish their
complete candidate, including packages; a feature-branch run also selects it
in that branch's catalog after finalization. LLiMa does this in the existing
Publish job; models with multiple invocations join them in the orchestrator's
`publish-candidates` job before publication.
Each complete candidate publishes independently of failures in other models
or profiles.
Develop and Main still require separate approval and promotion.

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
