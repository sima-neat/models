# SiMa Neat Model Registry

This repository defines the models in the SiMa Neat Model Registry and how
each one is compiled, measured, and published. `model.yaml` declares a model,
the model's `scripts/` directory holds code specific to that model, and
`tools/model_registry` holds the shared lifecycle.

Git holds no model binaries, datasets, compiled model packs (MPKs), logs, or
results. A model declaration pins every input it downloads by URI, byte size,
and SHA-256.

## How a model is built

A **build** is one compile invocation of one build profile of one model. Each
build passes through up to three **stages**:

| Stage | Runs on | Model entry point | Runner script |
| --- | --- | --- | --- |
| compile | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| performance | Modalix DevKit | `scripts/performance.py` | `scripts/run_model_performance.sh` |
| accuracy | Modalix DevKit | `scripts/accuracy.py` | `scripts/run_model_accuracy.sh` |

Each stage has a **stage key**: a SHA-256 of everything its result depends on.
CI compiles a build only when the branch catalog records no build with its
current compile key, and runs a DevKit stage only when no completed registry
run carries its current key. A change therefore reruns exactly the
stages whose inputs it touches. [Stages and stage
keys](docs/adding-a-model.md#9-know-what-a-change-reruns) lists the inputs of
each key.

Conversion is not a stage. `scripts/convert.py`, run by
`scripts/run_model_conversion.sh`, exports an upstream checkpoint to the ONNX
file that `model.yaml` pins as the component source.

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
sima-cli models download --id resnet_50 --variant modalix_int8 --branch develop
```

## CI

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `Target Branch Checks` | pull request | Runs `pytest tests/unit`, builds the `model-registry` package, and builds, browser-tests, and smoke-tests the model browser on the merge result. Contributor code never reaches a self-hosted runner. |
| `Vulcan Model Orchestrator` | every push, manual dispatch | Plans each stage against the registry, starts `Vulcan Model Compile` for each changed build, starts the DevKit stages reused builds lack, and publishes the branch catalog. A push that changes no stage key only publishes the catalog. |
| `Vulcan Model Compile` | started by the orchestrator | Compiles one build, publishes it, and starts its DevKit stages. |
| `Vulcan Model Benchmarking` | started by `Vulcan Model Compile` or the orchestrator | Runs the requested `performance` and `accuracy` stages of one build on a Modalix DevKit and publishes each result. |
| `Vulcan Model Convert` | manual dispatch | Runs `scripts/run_model_conversion.sh`, publishes the ONNX file, and prints its source pin. |
| `Build and Publish Model Registry` | push, manual dispatch | Runs `pytest tests/unit`, builds the `model-registry` package with `packaging/model-registry/build.sh`, publishes it, and moves the branch's `latest` tag. |
| `Vulcan Model Browser` | push to `main` or `develop`, manual dispatch | Publishes the model browser that `Developer Docs Validation` built and checked to staging. |
| `Model branch lifecycle` | branch create or delete | Cancels a deleted branch's unfinished runs and purges or restores its registry state. |

## Documentation

- [Developer documentation](docs/README.md): the contribution path from
  `model.yaml` to review.
- [Model Registry architecture](design-doc/model-registry-architecture.md):
  components, repository layout, and the CI flow.
- [Model taxonomy](docs/model-taxonomy.md): sanctioned category IDs.
- [SSD-MobileNet precision matrix plan](design-doc/ssd-mobilenet-precision-matrix-plan.md).
