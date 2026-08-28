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

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
python -m pytest
```

Start with the [developer documentation](docs/README.md) for the end-to-end
model contribution path, the field-level `model.yaml` contract, local testing,
and the GitHub review checklist.

## Vulcan compilation

[`Vulcan-Model-Compile.yml`](.github/workflows/Vulcan-Model-Compile.yml) validates
pull requests without granting them a self-hosted runner. Trusted pushes build
only changed models on staging. The first build of a new branch resolves its
closest base branch, compares from the merge base, and inherits unchanged model
artifacts from that branch's published `latest.json`; it does not rebuild the
catalog merely because GitHub reports an all-zero previous SHA. A maintainer can
run the workflow manually, choose `staging` or protected `production`, and
optionally provide a comma-separated model list and variant list or an explicit
base branch; empty model and variant lists build the full catalog.
A targeted variant run is compile-only and never advances the published catalog.

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
assembled and published with the standard Vulcan branch layout:

```text
models/
  <url-encoded-branch>/
    latest.tag                         # 12-character catalog commit
    latest.json                        # per-model latest build and category index
    <12-character-commit>/
      metadata.json                    # selectable model catalog
      catalog.json                     # machine-readable catalog
      taxonomy.json                    # canonical searchable category vocabulary
      latest.json                      # immutable copy of the index
      manifest.json                    # Vulcan checksums and S3 keys
      <changed-model-id>/
        metadata.json                  # directly installable model package
        <variant-id>/
          *_mpk.tar.gz
          SHA256SUMS
          compile-result.json
  branches.json
```

The shared Vulcan publisher uploads the immutable commit folder first. The
branch's `latest.tag` is advanced only after the commit manifest exists. For
example:

```bash
# Select one or more models interactively from the branch catalog.
sima-cli neat install --stg models@develop:latest --install-dir ./compiled-models

# Install every compiled variant for one model directly.
sima-cli neat install --stg models/resnet_50@develop:latest --install-dir ./compiled-models
```

Only affected models are recompiled. `latest.tag` remains a plain-text pointer
to the newest complete catalog for existing Vulcan consumers. The versioned
[`latest.json`](schemas/latest-index-v2.schema.json) maps each model to its own
build tag, build time, source/compiler provenance, and artifact branch. Root
catalog metadata uses immutable URLs from that index, while a direct
`models/<model-id>@<branch>:latest` install resolves the model's build tag.
Unchanged MPKs are not copied into every catalog commit. A new branch inherits
unchanged pointers from its resolved base branch until those models are rebuilt
on the branch. Each compiled model records a deterministic fingerprint covering
its model-local inputs, shared build tooling, and Model Compiler version. A
matching fingerprint is reused; a missing or changed fingerprint rebuilds every
supported variant of that model. A branch with no compilation work still
publishes a lightweight catalog whose entries reference inherited immutable
artifacts. Model README and test-description edits do not invalidate compiled
MPKs.

Successful staging compilations are also written through the IAM-authenticated
Vulcan Artifact Registry API as one immutable run per newly compiled variant.
Each record links the model source checksum, variant parameters, Model Compiler
version, build fingerprint, and exact versioned S3 MPK. For staging feature
branches, the workflow discovers the Modalix compile artifacts actually produced
by the current run and benchmarks each one on the resolved Modalix GitHub runner.
The runner verifies the manifest-pinned platform, installs the pinned benchmarker
and Neat runtime, and returns a schema-v2 result and log for each model variant. A
trusted hosted job publishes every completed result under its model's S3 hierarchy
and records an immutable `model_benchmark` child of the matching compile run.

See [the developer documentation](docs/README.md),
[the architecture](docs/architecture.md), the
[synthetic benchmarking guide](docs/benchmarking.md), the
[model taxonomy](docs/model-taxonomy.md), and the
[SSD-MobileNet matrix qualification plan](docs/ssd-mobilenet-precision-matrix-plan.md)
for repository conventions.
