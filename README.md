# SiMa Neat Model Registry

This repository defines the models in the SiMa Neat Model Registry and how
each one is compiled, benchmarked, and published. `model.yaml` declares a model,
the model's `scripts/` directory holds code specific to that model, and
`tools/model_registry` holds the shared lifecycle.

Git holds no model binaries, datasets, compiled model packs (MPKs), logs, or
results. A model declaration pins every input it downloads by URI, byte size,
and SHA-256.

## Registry release state

[`state.json`](state.json) defines the release experience consumed by the SDK's
Palette SDK component: `current_state` accepts `GA`, `Beta`, or `Alpha`, and
`default_branch` accepts `develop` or `main`. Initial values are `Alpha` and
`develop`, allowing the consumer to display `Current Release: Alpha`.

The [state workflow](.github/workflows/vulcan-model-state.yml) publishes
`models/state.json` in the production artifact bucket only from `main`. Changes
to the file, workflow, or publishing script on `main` deploy automatically;
manual runs must also select `main`. Other branches cannot publish this state.
Publication validates both fields, sets JSON content type and `Cache-Control:
no-store`, and verifies the uploaded bytes. CDN configuration must honor that
cache policy. Palette SDK integration must fetch this file; it is not implemented
here. This file does not participate in model compilation or fingerprints.

## How a model is built

A **build** is one compile invocation of one build profile of one model. Each
build passes through up to three **stages**:

| Stage | Runs on | Model entry point | Runner script |
| --- | --- | --- | --- |
| compile | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| performance | Modalix DevKit | `scripts/performance.py` | `scripts/run_model_performance.sh` |
| accuracy | Modalix DevKit | `scripts/accuracy.py` | `scripts/run_model_accuracy.sh` |

Each stage has a **fingerprint**: a SHA-256 of every input that decides its
result. CI compiles a build only when the branch catalog records no build with
its current compile fingerprint, and runs a **benchmark** (the performance or
accuracy stage) only when no completed registry run carries its current
benchmark fingerprint. A change therefore reruns exactly the stages whose
inputs it touches. [What a change
reruns](docs/adding-a-model.md#9-know-what-a-change-reruns) lists the inputs of
each fingerprint, and the [vocabulary](AGENTS.md#vocabulary) defines these
terms.

Conversion publishes a validated ONNX output as a Vulcan registry run. Models
can declare `compilation.source: conversion`; compile planning selects the latest
matching conversion and records its exact output. See [conversion artifacts](docs/conversion-artifacts.md).

## Set up the contributor tool

Install the repository-pinned Model Compiler first. Then install
`model-registry` into it and check a model:

```bash
sima-cli neat install --stg models-utils/model-registry@develop:latest
model-registry --version
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
```

The installer finds the Model Compiler environment from the NEAT SDK or a
standalone installation and adds `model-registry` without replacing
compiler-owned packages. This package contains only author commands; it
cannot publish.

## Download a published model

```bash
sima-cli models list --branch develop
sima-cli models download --id resnet_50 --variant w8a8 --branch develop
```

## CI

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `Target Branch Checks` | pull request | Runs `python -m pytest`, builds the `model-registry` package, and builds, browser-tests, and smoke-tests the model browser on the merge result. Contributor code never reaches a self-hosted runner. |
| `Vulcan Model Orchestrator` | every push, manual dispatch | Plans each stage against the registry, runs `Vulcan Model Compile` for each changed build and `Vulcan Model Benchmarking` for each reused build that lacks a benchmark, and publishes the branch catalog. A newer push cancels the running orchestrator with its compiles and benchmarks. A push that changes no fingerprint only publishes the catalog. On `main` and `develop`, it also publishes the model browser it checked to staging, independently of `VULCAN_ENV`. |
| `Vulcan Model Compile` | called by the orchestrator | Compiles one build, runs the model's compile tests on it, publishes it, and benchmarks it. |
| `Vulcan Model Benchmarking` | called by `Vulcan Model Compile` or the orchestrator, manual dispatch | Runs the requested `performance` and `accuracy` stages of one build on a Modalix DevKit, runs the model's benchmark tests on each result, and publishes each result with its test outcome. |
| `Vulcan Model Convert` | manual dispatch | Runs `scripts/run_model_conversion.sh` and the model's conversion tests, publishes a validated ONNX bundle and its provenance to Vulcan. |
| `Build and Publish Model Registry` | push, manual dispatch | Runs `python -m pytest`, builds the `model-registry` package with `packaging/model-registry/build.sh`, publishes it, and moves the branch's `latest` tag. |
| `Model branch lifecycle` | branch create or delete | Cancels a deleted branch's unfinished runs and purges or restores its registry state. |

## Documentation

- [Developer documentation](docs/README.md): the contribution path from
  `model.yaml` to review.
- [Model Registry architecture](design-doc/model-registry-architecture.md):
  components, repository layout, and the CI flow.
- [Model taxonomy](docs/model-taxonomy.md): sanctioned category IDs.
