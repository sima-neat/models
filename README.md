# SiMa Neat Model Registry

This repository is the source of truth for declarative model definitions,
reproducible preparation and compilation, device validation, and publication to
the Vulcan artifact registry.

Large model binaries, calibration datasets, compiled model packs, and benchmark
artifacts do not belong in Git. Model directories contain immutable provenance
and checksums; shared lifecycle implementation lives in `tools/model_registry`.

Model directories do own model-specific surgery and compile recipes. Those
recipes consume verified sources and must emit exactly one directly consumable
`*_mpk.tar.gz`; common discovery, downloading, runner selection, MPK validation,
checksumming, and publication remain shared.

`model-registry` is an author-safe `sima-cli` package for local validation,
verified input handling, variant inspection, and MPK checks. It is not a direct
registry publishing client. Model authors compile with the model-owned
`compile.py` and run performance tests with the DevKit `benchmarker`; CI keeps
the separate packaging and publication command surface.

## Local setup

```bash
sima-cli neat install --stg models-utils/model-registry@develop:latest
model-registry --version
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
```

Install the repository-pinned Model Compiler first. The package installer finds
its NEAT SDK or standalone environment and installs `model-registry` there
without replacing compiler-owned dependencies.

Start with the [developer documentation](docs/README.md) for the end-to-end
model contribution path, the field-level `model.yaml` contract, local testing,
and the GitHub review checklist.

## Vulcan compilation

[`Vulcan-Model-Compile.yml`](.github/workflows/Vulcan-Model-Compile.yml) validates
pull requests without granting them a self-hosted runner. Trusted pushes build
only changed models on staging. The first build of a new branch resolves its
closest base branch and compares from the merge base, so GitHub's all-zero
previous SHA does not force a full rebuild. A maintainer can
run the workflow manually, choose `staging` or protected `production`, and
optionally provide a comma-separated model list and variant list or an explicit
base branch; empty model and variant lists build the full catalog.

The Python-only `benchmarker` is published separately when its source,
schemas, dependency pins, or packaging change. Install its branch package with:

```bash
sima-cli neat install --stg models-utils/benchmarker@develop:latest
```

The package installs its benchmarker-specific declared `core@v0.4.0` dependency when needed, then
installs the benchmarker wheel into the Core-managed pyneat environment.

The workflow first resolves the repository-wide Model Compiler pin from
[`deps/manifest.json`](deps/manifest.json), then requests the environment label,
`model-compiler`, and the derived `model-compiler-<version>` label. Successful matrix outputs are
published individually. Each successful matrix item validates its MPK, uploads
it under an immutable object key, records its checksum, provenance, exact S3
object version, and taxonomy through the IAM-authenticated Model Registry API,
and then finalizes the registry run.

Model discovery and checksum-verified downloads use registry database records
and presigned artifact URLs:

```bash
sima-cli models list --branch develop
sima-cli models download --id resnet_50 --variant modalix_bf16 --branch develop
```

The former Models-specific `metadata.json`, `latest.json`, `latest.tag`, and
`catalog.json` publication path is retired. Existing objects remain historical
artifacts, but workflows no longer read, carry forward, validate, generate, or
promote those pointers. Branch purge and restore use registry API operations
over registered runs and immutable S3 object versions.

Each record links the model source checksum, variant parameters, Model Compiler
version, build fingerprint, and exact versioned S3 MPK. For staging feature
branches, the workflow discovers the Modalix compile artifacts actually produced
by the current run and benchmarks each one on the resolved Modalix GitHub runner.
The runner verifies the manifest-pinned platform, installs the pinned benchmarker
and Neat runtime, and returns a schema-v3 result and log for each model variant. A
trusted hosted job publishes every completed result under its model's S3 hierarchy
and records an immutable `model_benchmark` child of the matching compile run.

See [the developer documentation](docs/README.md),
[the Model Registry design](design-doc/model-registry-architecture.md), the
[local accuracy development guide](docs/local-accuracy-development.md), the
[synthetic benchmarking guide](docs/benchmarking.md), the
[model taxonomy](docs/model-taxonomy.md), and the
[SSD-MobileNet matrix qualification plan](design-doc/ssd-mobilenet-precision-matrix-plan.md)
for repository conventions.
