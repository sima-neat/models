import ContributionPath from '@site/src/components/ContributionPath';

# Reproduce or contribute a model

The Model Browser lists models that run on SiMa.ai platforms, each with its
measured performance and accuracy. This repository holds what produces a model:
its pinned source, its compile recipe, and the scripts that measure it. You can
run those scripts yourself and reproduce every step on your own machine.

## Start here

Read [Getting started](getting-started.md) first. It shows the path every model
takes, from the official model to the measured results, and walks through it once with
real commands. A contributor follows the same path, so read it before the contribution
guides below. [Concepts](concepts.md) defines the terms both use, such as build, build
profile, stage, and fingerprint.

## Contribute a model

A contribution adds or updates one model folder. Its `model.yaml` records the
exact source, components, build profiles, and datasets, and its scripts compile
and measure each build. These guides explain how to define a model, qualify it
locally, and submit it for review, and what CI does after the merge.

### Where each step runs

| Phase | What runs | What it shows |
| --- | --- | --- |
| Local development | Validation, verified downloads, the compile of each build, MPK checks, and the performance and accuracy stages on a DevKit. | You have reproducible results for every build you submit. |
| Pull request | `Target Branch Checks`: lint, the repository tests and model tests, the `model-registry` package build, and the model browser checks. `PR Sanitization`: the target branch, and a GitHub issue reference in the description. | The committed files are valid on the merge result. Pull requests do **not** run recipe or surgery code to compile or measure a model. |
| After the merge | CI compiles and benchmarks each changed build and publishes the results. | The published model and its measurements come from the reviewed revision. |

The pinned Model Compiler and a compatible Modalix DevKit are submission
prerequisites, not optional CI substitutes. Do not open a model pull request
until every submitted build has compiled, produced valid MPKs, and completed
its declared performance and accuracy stages on a DevKit. CI repeats those
stages after the merge. It does not replace local qualification.

### Workflow

A contribution moves through six steps. Each produces what the next one needs.
Select a step to open its guide.

<ContributionPath />

### Before you begin

You need a checkout of this repository, `sima-cli`, and the repository-pinned
Model Compiler from the NEAT SDK or a standalone installation. The DevKit
stages need a compatible Modalix DevKit. Models that use calibration,
fidelity, or accuracy data also need private Artifactory access.

Start with [Define the model contract](adding-a-model.md). The [local
qualification guide](local-development.md) owns the tool installation and
complete test sequence. Follow it before you submit anything.

### Guides

Follow these guides in order:

1. [Define the model contract](adding-a-model.md): the model directory,
   scripts, components, sources, build profiles, packages, and stage entries.
2. [Qualify locally](local-development.md): install the contributor tool,
   then compile and measure every submitted build.
3. [Submit for review](contributing.md): commit only reviewed inputs and
   report the stages you ran.
4. [Promote a model](model-promotion.md): make a tested build of a
   build-on-demand model available for download.

Use these references when the model needs them:

- [Concepts](concepts.md): build, build profile, stage, benchmark,
  fingerprint, model component, model package, and MPK.
- [Model README](model-readme.md): the README and hero rules, with the README
  template.
- [Datasets for local model development](datasets.md): Artifactory access,
  registered dataset resolution, and adding a dataset or variant.
- [Accuracy and fidelity](local-accuracy-development.md): compile-time
  numerical fidelity and DevKit task accuracy.
- [DevKit performance](benchmarking.md): the performance script, its
  workload, and how results are published.
- [`model.yaml` reference](metadata-schema.md): the meaning of each field.
- [Expensive and on-demand compile runners](runner-class-requests.md):
  decide whether a model qualifies for an exception, prepare approval evidence,
  configure the request, and verify a build.
- [Model taxonomy](model-taxonomy.md): sanctioned category IDs.

The JSON Schemas under
[`schemas/`](https://github.com/sima-neat/models/tree/develop/schemas) and the
taxonomy files under
[`taxonomies/`](https://github.com/sima-neat/models/tree/develop/taxonomies)
are authoritative. These guides explain how to work with them.
