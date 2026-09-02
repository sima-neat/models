import ContributionPath from '@site/src/components/ContributionPath';

# Contribute a model

Take a model from a pinned source to a review-ready contribution: describe it
in `model.yaml`, validate its immutable inputs, compile and test it where the
required hardware is available, and submit only the source-controlled contract
through Git. Pull-request checks validate that contract without executing
contributor code on SiMa infrastructure. Trusted automation compiles,
publishes, and benchmarks the model after merge.

## Understand the three execution boundaries

| Phase | What runs | What it proves |
| --- | --- | --- |
| Local development | Schema and catalog validation, verified downloads, matrix inspection, model-owned compilation, declared accuracy evaluation, MPK validation, and DevKit benchmarking when those environments are available. | The author has reproducible evidence for the declared model and tested variants. |
| Pull request | Catalog and taxonomy validation, compiler dependency resolution, changed-model discovery, and deterministic matrix construction on GitHub-hosted infrastructure. | The committed contract is safe to review and selects the intended work. Pull requests do **not** run recipe or surgery code on Vulcan runners. |
| Trusted automation | On a trusted push, runner admission, compilation, optional dataset-backed accuracy, MPK verification, staging publication, and a dispatched Modalix benchmark. | Staging artifacts, device evidence, and provenance were produced from the trusted revision. Production remains an explicit protected maintainer action. |

Local compiler or DevKit access may be unavailable. In that case, run every
check your environment supports and list the omitted commands and limitation in
the pull request. The trusted workflow is not a substitute for claiming local
evidence that was never produced.

## Workflow

A successful contribution moves through six traceable stages, with each stage
producing the verified input or evidence required by the next. Select a stage
in the infographic to open its detailed guide.

<ContributionPath />

By the time you open the pull request, the model contract is complete and every
available local check has run. Each locally tested variant should produce one
valid MPK, and that exact MPK should have a machine-readable benchmark result.
If an environment was unavailable, record the missing check instead of
implying that it passed.

## Before you begin

You need a checkout of the Models repository, `sima-cli`, and the
repository-pinned Model Compiler installed through the NEAT SDK or as a
standalone tool. A compatible Modalix DevKit is required for the benchmarking
stage.

Install the author-safe contributor command into the Model Compiler
environment:

```bash
sima-cli neat install --stg models-utils/model-registry@develop:latest
model-registry --version
```

The installer finds the compiler environment and preserves its dependencies.
The command can validate, fetch, inspect variants, and verify MPKs; it cannot
publish registry artifacts.

Continue with [Local development and testing](local-development.md) if you need
the complete environment setup.

## Guides

- [Adding a model](adding-a-model.md) — model directory, source, recipe, and
  test declarations.
- [`model.yaml` reference](metadata-schema.md) — field-level contracts and
  validation requirements.
- [Compile runner class requests and approvals](runner-class-requests.md) —
  per-variant resource requests and Vulcan admission policy.
- [Model taxonomy](model-taxonomy.md) — sanctioned category IDs.
- [Local development and testing](local-development.md) — validation, input
  fetching, compilation, MPK verification, and common failures.
- [Local accuracy development](local-accuracy-development.md) — declarative
  dataset-backed reference and candidate evaluation.
- [Synthetic model benchmarking](benchmarking.md) — DevKit setup, manifests,
  execution, and results.
- [Contributing with Git and GitHub](contributing.md) — pre-commit review, pull
  request evidence, and CI behavior.

The JSON Schemas under
[`schemas/`](https://github.com/sima-neat/models/tree/develop/schemas) and the
taxonomy files under
[`taxonomies/`](https://github.com/sima-neat/models/tree/develop/taxonomies)
are authoritative. These guides explain how to work with those contracts; they
do not replace them.
