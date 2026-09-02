import ContributionPath from '@site/src/components/ContributionPath';

# Contribute a model

A model contribution is eligible for review only after completing the required
local qualification. Pin the model source and required datasets to immutable
revisions, define the reproducible contract in `model.yaml`, compile every
supported variant with the repository-pinned Model Compiler, complete each
applicable accuracy evaluation, verify the resulting MPK, and benchmark that
exact MPK on a compatible Modalix DevKit. Commit only the source-controlled
contract; downloaded inputs and generated artifacts must remain outside Git.
Pull-request checks validate the structure, consistency, and safety of the
submitted contract. After merge, trusted automation rebuilds the reviewed
revision, publishes qualified artifacts to staging, and records benchmark
results and provenance.

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
stage. Models that use calibration or accuracy data also require private
Artifactory access.

Start with [Define the model contract](adding-a-model.md) to describe the
source-controlled contract. The [local qualification
guide](local-development.md) owns the tool installation and complete test
sequence; follow it before you submit anything.

## Guides

Follow these guides in order:

1. [Define the model contract](adding-a-model.md) — define the model directory,
   immutable source, recipe, variants, and test declarations.
2. [Qualify locally](local-development.md) — install the
   contributor tools, compile and test every supported variant, and retain the
   required local evidence.
3. [Submit for review](contributing.md) — commit only reviewed
   inputs and submit the local evidence for review.

Use these supporting references when the model needs them:

- [Datasets for local model development](datasets.md) — Artifactory access,
  registered dataset resolution, and introducing new datasets or variants.
- [Local accuracy development](local-accuracy-development.md) — declarative
  dataset-backed reference and candidate evaluation.
- [Synthetic model benchmarking](benchmarking.md) — DevKit setup, manifests,
  execution, and result semantics.
- [`model.yaml` reference](metadata-schema.md) — field-level contracts and
  validation requirements.
- [Compile runner class requests and approvals](runner-class-requests.md) —
  per-variant resource requests and Vulcan admission policy.
- [Model taxonomy](model-taxonomy.md) — sanctioned category IDs.

The JSON Schemas under
[`schemas/`](https://github.com/sima-neat/models/tree/develop/schemas) and the
taxonomy files under
[`taxonomies/`](https://github.com/sima-neat/models/tree/develop/taxonomies)
are authoritative. These guides explain how to work with those contracts; they
do not replace them.
