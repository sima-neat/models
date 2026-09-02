import ContributionPath from '@site/src/components/ContributionPath';

# Contribute a model

Take a model from a pinned source to a review-ready contribution: describe it
in `model.yaml`, compile it locally, benchmark the resulting MPK on a Modalix
DevKit, and submit the files through Git. CI repeats the qualification flow
from the committed revision.

## Workflow

A successful contribution moves through five traceable stages, with each stage
producing the verified input or evidence required by the next. Select a stage
in the infographic to open its detailed guide.

<ContributionPath />

By the time you open the pull request, the model contract is complete, each
tested variant produces one valid MPK, and the exact MPK has a machine-readable
benchmark result. If an environment was unavailable, record the missing check
instead of implying that it passed.

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
  planned per-variant resource requests and Vulcan admission policy.
- [Model taxonomy](model-taxonomy.md) — sanctioned category IDs.
- [Local development and testing](local-development.md) — validation, input
  fetching, compilation, MPK verification, and common failures.
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
