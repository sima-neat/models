import ContributionPath from '@site/src/components/ContributionPath';

# Contribute a model

The Model Browser makes models discoverable, reproducible, and deployable on
SiMa.ai platforms. A contribution is a versioned contract, not a model binary:
it records the exact source, ownership, component interfaces, compile recipe,
build profiles, packages, datasets, and the evidence each build must produce.

This documentation is for model developers who create or update that contract.
It explains how to define a model, qualify it locally, submit it for review,
and what trusted automation does after merge.

## How a build is benchmarked

A **build** is one compile invocation of one build profile of one model. Each
build passes through up to three **stages**, each driven by one script in the
model's `scripts/` directory:

| Stage | Runs on | Model script | Produces |
| --- | --- | --- | --- |
| compile | pinned Model Compiler | `compile.py` | one MPK per model component, plus a numerical fidelity report when `model.yaml` declares one |
| performance | Modalix DevKit | `performance.py` | one whole-model performance report for the build profile |
| accuracy | Modalix DevKit | `accuracy.py` | one whole-model accuracy report for the build profile |

The performance and accuracy stages are the build's **benchmarks**. Each stage
has a **fingerprint**, a SHA-256 of every input that decides its result. CI
reruns a stage only when no published result carries its current fingerprint,
so editing `performance.py` reruns the performance benchmark without
recompiling, and editing a compile input recompiles and then reruns the
benchmarks. See [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns), and the
[vocabulary](https://github.com/sima-neat/models/blob/develop/AGENTS.md#vocabulary)
for these terms.

## Understand the three execution boundaries

| Phase | What runs | What it proves |
| --- | --- | --- |
| Local development | Validation, verified downloads, each declared compile invocation, MPK checks, and the performance and accuracy stages on a DevKit. | The author has reproducible evidence for every submitted build. |
| Pull request | `Target Branch Checks`: ruff, unit tests with catalog, taxonomy, compiler-pin, and committed-file validation, the package build, and the model browser checks, on GitHub-hosted runners. | The committed contract is valid on the merge result. Pull requests do **not** run recipe or surgery code on self-hosted runners. |
| Trusted automation | On a push, each stage whose fingerprint has no completed registry run: compilation, publication, then the benchmarks. | Staging artifacts, measurements, and provenance come from the reviewed revision. Production is an explicit protected maintainer action. |

The pinned Model Compiler and a compatible Modalix DevKit are submission
prerequisites, not optional CI substitutes. Do not open a model pull request
until every submitted build has compiled, produced valid MPKs, and completed
its declared performance and accuracy stages on a DevKit. Trusted automation
repeats those stages for the committed revision; it does not replace local
qualification.

## Workflow

A contribution moves through six stages. Each produces the input or evidence
the next one needs. Select a stage to open its guide.

<ContributionPath />

## Before you begin

You need a checkout of this repository, `sima-cli`, and the repository-pinned
Model Compiler from the NEAT SDK or a standalone installation. The DevKit
stages need a compatible Modalix DevKit. Models that use calibration,
fidelity, or accuracy data also need private Artifactory access.

Start with [Define the model contract](adding-a-model.md). The [local
qualification guide](local-development.md) owns the tool installation and
complete test sequence; follow it before you submit anything.

## Guides

Follow these guides in order:

1. [Define the model contract](adding-a-model.md): the model directory,
   scripts, components, sources, build profiles, packages, and stage entries.
2. [Qualify locally](local-development.md): install the contributor tool,
   then compile and measure every submitted build.
3. [Submit for review](contributing.md): commit only reviewed inputs and
   report the stages you ran.

Use these references when the model needs them:

- [Datasets for local model development](datasets.md): Artifactory access,
  registered dataset resolution, and adding a dataset or variant.
- [Accuracy and fidelity](local-accuracy-development.md): compile-time
  numerical fidelity and DevKit task accuracy.
- [DevKit performance](benchmarking.md): the performance script, its
  workload, and how results are published.
- [`model.yaml` reference](metadata-schema.md): field-level contracts.
- [Expensive and on-demand compile runners](runner-class-requests.md):
  decide whether a model qualifies for an exception, prepare approval evidence,
  configure the request, and verify a build.
- [Model taxonomy](model-taxonomy.md): sanctioned category IDs.

The JSON Schemas under
[`schemas/`](https://github.com/sima-neat/models/tree/develop/schemas) and the
taxonomy files under
[`taxonomies/`](https://github.com/sima-neat/models/tree/develop/taxonomies)
are authoritative. These guides explain how to work with them.
