<p align="center">
  <img src="docs/assets/readme/hero.png" width="100%" alt="The four steps every model takes: declare it in model.yaml, compile it with the Model Compiler, benchmark it on a Modalix DevKit, and publish it for sima-cli models download">
</p>

<h1 align="center">SiMa Neat Model Browser</h1>

<p align="center">
  The models that run on SiMa.ai Modalix, each one declared in a <code>model.yaml</code>,<br>
  compiled with the Model Compiler, and benchmarked on a Modalix DevKit.
</p>

<p align="center">
  <a href="https://developer.sima.ai/models/">Model Browser</a> ·
  <a href="docs/adding-a-model.md">Add a Model</a> ·
  <a href="docs/README.md">Developer Documentation</a> ·
  <a href="design-doc/model-registry-architecture.md">Architecture</a> ·
  <a href="https://github.com/sima-neat/models/issues">Issues</a>
</p>

Git holds the declaration and the code of each model, not its files. A
declaration pins every input it downloads by URI, byte size, and SHA-256.
Model binaries, datasets, compiled model packages (MPKs), logs, and results
live in object storage.

## Use a Model

List the published models and download one:

```bash
sima-cli models list --branch develop
sima-cli models download --id resnet_50 --variant w8a8 --branch develop
```

The [Model Browser](https://developer.sima.ai/models/) shows each model with
its measured performance and accuracy. It needs your SiMa Developer Portal
login.

## How a Model Is Built

A **build** is one compile invocation of one build profile of one model. Each
build passes through up to three **stages**:

| Stage | Runs on | Model entry point | Runner script |
| --- | --- | --- | --- |
| compile | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| performance | Modalix DevKit | `scripts/performance.py` | `scripts/run_model_performance.sh` |
| accuracy | Modalix DevKit | `scripts/accuracy.py` | `scripts/run_model_accuracy.sh` |

Each stage has a **fingerprint**: a SHA-256 of every input that decides its
result. CI runs a stage only when no published result carries its current
fingerprint, so a change reruns exactly the stages whose inputs it touches.
[What a change reruns](docs/adding-a-model.md#9-know-what-a-change-reruns)
lists the inputs of each fingerprint, and the
[vocabulary](AGENTS.md#vocabulary) defines these terms.

A model can also compile from a converted ONNX file instead of a pinned
download. See [conversion artifacts](docs/conversion-artifacts.md).

## Add a Model

Install the repository-pinned Model Compiler first. Then install
`model-registry` into it and check a model:

```bash
sima-cli neat install --stg models-utils/model-registry@develop:latest
model-registry --version
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
```

The installer adds `model-registry` to the Model Compiler environment of the
Neat SDK or of a standalone installation, without replacing the compiler's
packages. The package holds only author commands. It cannot publish.

[Add a model](docs/adding-a-model.md) covers the path from `model.yaml` to
review.

## Repository Layout

```text
models/<model-id>/      model.yaml, README.md, hero.png, scripts/, tests/
tools/model_registry/   shared CLI and lifecycle code
scripts/                stage runner scripts and CI helpers
schemas/                JSON Schemas for model.yaml, tests, and reports
artifact-browser/       the Model Browser
docs/                   contribution guides
design-doc/             architecture
```

The [architecture](design-doc/model-registry-architecture.md#directory-contract)
describes every directory.

## Documentation

- [Developer documentation](docs/README.md): the contribution path from
  `model.yaml` to review.
- [Model Browser architecture](design-doc/model-registry-architecture.md):
  components, the CI workflows, and the registry release state.
- [Model taxonomy](docs/model-taxonomy.md): sanctioned category IDs.
