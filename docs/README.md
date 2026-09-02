import ContributionPath from '@site/src/components/ContributionPath';

# Contribute a model

Bringing a model into the registry starts with proving that it works in your
hands. You pin the exact source, describe the model in `model.yaml`, compile
every supported variant with the pinned Model Compiler, and test each resulting
MPK on a compatible Modalix DevKit before opening a pull request. Git carries
the reproducible model contract—not downloaded inputs or generated artifacts.
Pull-request checks make sure that contract is coherent and safe to review.
After merge, trusted automation rebuilds the trusted revision, publishes it to
staging, and records its benchmark evidence.

## Understand the three execution boundaries

| Phase | What runs | What it proves |
| --- | --- | --- |
| Local development | Schema and catalog validation, verified downloads, matrix inspection, model-owned compilation, applicable declared accuracy evaluation, MPK validation, and DevKit benchmarking. | The author has reproducible evidence for every submitted supported variant. |
| Pull request | Catalog and taxonomy validation, compiler dependency resolution, changed-model discovery, and deterministic matrix construction on GitHub-hosted infrastructure. | The committed contract is safe to review and selects the intended work. Pull requests do **not** run recipe or surgery code on Vulcan runners. |
| Trusted automation | On a trusted push, runner admission, compilation, optional dataset-backed accuracy, MPK verification, staging publication, and a dispatched Modalix benchmark. | Staging artifacts, device evidence, and provenance were produced from the trusted revision. Production remains an explicit protected maintainer action. |

The pinned Model Compiler and a compatible Modalix DevKit are submission
prerequisites, not optional CI substitutes. Do not open a model pull request
until every submitted supported variant has compiled successfully, produced one
valid MPK, completed each applicable accuracy evaluation, and benchmarked that
exact MPK on the DevKit. Trusted automation verifies the committed revision; it
does not replace the developer's local qualification responsibility.

## Workflow

A successful contribution moves through six traceable stages, with each stage
producing the verified input or evidence required by the next. Select a stage
in the infographic to open its detailed guide.

<ContributionPath />

By the time you open the pull request, the model contract is complete, every
submitted supported variant produces one valid MPK, each applicable accuracy
evaluation completes successfully, and that exact MPK has a successful
machine-readable DevKit benchmark result.

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
- [Datasets for local model development](datasets.md) — Artifactory access,
  registered dataset resolution, and introducing new datasets or variants.
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
