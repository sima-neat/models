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
models/<model-id>/scripts/requirements-performance.txt # optional, with performance.py
models/<model-id>/scripts/accuracy.py               # only with an accuracy declaration
models/<model-id>/scripts/requirements-accuracy.txt # optional, with accuracy.py
models/<model-id>/scripts/convert.py                # only with a declared conversion
models/<model-id>/scripts/requirements-convert.txt  # required with convert.py
models/<model-id>/scripts/surgery.py                # only when the recipe rewrites the graph
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
guide](runner-class-requests.md) before enabling the model's request. In the
Models pull request, link the reviewed Vulcan policy pull request and give the
exact model IDs, the owner, the duration, memory, or interruption evidence, the
expected build frequency, and the cost tradeoff. Larger size and on-demand are
separate permissions. An existing expensive-size grant does not authorize
on-demand.

A passing Models pull request validates the request in `model.yaml`, not the
approval. Confirm that the grant is deployed, as the
[runner request guide](runner-class-requests.md#obtain-access-through-a-vulcan-policy-pr)
describes, before you expect CI to compile on the requested runner. A change to
the runner request alone reuses the existing build. Force a new build of that
one model only to collect new runner evidence.

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
staged diff. Pull requests reject any committed file that `.gitignore`
excludes, even when added with `git add -f`, and any file over 2 MiB; a test
fixture that must be committed anyway goes in `ALLOWED_FILES` in
`tests/test_target_branch_checks.py`. Write a commit message that describes
the model or lifecycle contract, not just the filenames.

Open a pull request against `develop`. `Target Branch Checks` reruns
validation on the merge result. It does not compile or measure the model.
CI compiles, measures, and publishes after the merge, and
authors do not invoke those publication steps themselves.

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

A pull request runs `Target Branch Checks` on the merge result:

- `ruff check .`
- the repository tests and each model's tests that need no stage output, which
  also validate the complete catalog and taxonomy, resolve the compiler pin,
  and reject the committed files described above
- the `model-registry` package build
- the model browser build and its checks
- for a pull request into `main`, a check that `develop` has built and measured
  everything `main` will show. It reads the registry and prints each missing build
  or result. A maintainer re-runs it once those exist.

A pull request never compiles a model, measures it on a DevKit, or publishes a
result. Those steps run only after a push to this repository.

For a model change, a passing pull request confirms that:

- every model's `model.yaml`, its tests, and the taxonomy are valid
- the repository-wide Model Compiler pin resolves
- the `model-registry` package builds
- the model browser builds and loads

It is **not** evidence that the recipe compiled, the MPK passed validation,
fidelity or accuracy met a threshold, or the model ran on a DevKit. Those
claims must come from your stated local results, or from CI after the merge.

## After merge: trusted staging automation

The merge to `develop` starts the **Vulcan Model Orchestrator** workflow on the
merged revision. You start nothing yourself. The workflow:

1. validates every model
2. compiles each build whose compile fingerprint has no published result, and
   runs the model's compile tests on the output
3. runs the performance and accuracy stages of each new build on a DevKit, and
   the model's benchmark tests on each report
4. publishes the builds and reports, which the model browser then shows

CI reuses a build whose fingerprints did not change, as [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns) explains. CI validates a
model with `build-on-demand: true` but compiles it only on request. See
[on-demand builds](adding-a-model.md#on-demand-builds).

The workflow runs on every push to this repository, not only on a merge. When
you push your branch here, CI compiles and benchmarks its changed builds too.

Follow the run in the repository's Actions tab. Each build has its own compile
job and benchmark jobs, grouped under the build's name. A successful compile
does not by itself mean the benchmark succeeded, so check each build's
benchmark jobs, then the model's page in the model browser.

When a stage fails:

- A failed compile, or a failed compile test, leaves the build unpublished.
  Read the job log, reproduce the failure with [Qualify
  locally](local-development.md), and open a pull request with the fix.
- A failed benchmark test still publishes its report, with test status
  `failed`, and then fails the run. Fix the script, or update `expected` in
  `tests/tests.yaml` when a reviewed change moved the value ([set up
  `tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)).
- When the failure is not in your model's files, tell a maintainer.

Each fix is a focused change to `model.yaml`, the recipe, or a script, with
the local results that show it works.

## Review and merge expectations

A model is ready to merge when the local qualification evidence is complete,
the pull request checks pass, and reviewers accept the contract, licensing,
ownership, test plan, and remaining gaps. A merge is not a release. CI still
compiles and measures the merged revision, and
[promotion](model-promotion.md) of a build-on-demand model is a separate
reviewed step.

Reviewers also check that:

- a change to a model's scripts, inputs, or outputs updates that model's
  tests, `tests/tests.yaml`, README, and hero in the same pull request
- the description states which stages the change reruns, following [what a
  change reruns](adding-a-model.md#9-know-what-a-change-reruns)
- model-specific code is under `models/<model-id>/scripts/`, and shared code
  is under `tools/`

If CI exposes a model-specific failure after the merge, update the recipe or
`model.yaml` with reproducible evidence and tests. To withhold a build profile
that does not work, comment it out in `model.yaml`, so that CI neither
compiles nor benchmarks it. Do that only after the profile is proven unusable,
not as a shortcut around an unresolved failure.
