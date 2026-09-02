# Adding a model

This guide takes a model from an upstream ONNX file to a reviewable Model
Registry contribution. Read the [`model.yaml` reference](metadata-schema.md)
for field-level details and [local development](local-development.md) for the
complete command reference.

## 1. Confirm that the model is suitable

Before creating files, establish:

- an ONNX source that can be pinned to immutable content;
- the source byte size and SHA-256;
- an SPDX license expression or an internal `LicenseRef-*`, redistribution
  policy, and required attribution;
- a responsible GitHub user or team;
- the exact input/output tensor contract and preprocessing/postprocessing;
- the Modalix compile variants to attempt;
- the calibration and correctness data needed for those variants.

Do not add a model whose source or license is still ambiguous. `experimental`
maturity permits incomplete qualification evidence; it does not permit mutable
provenance or unknown ownership.

## 2. Create the model directory

Model IDs use lowercase snake case. The directory name and `model.yaml` `id`
must match.

```text
models/<model-id>/
  README.md
  model.yaml
  compile.py
  surgery.py                 # only when model-specific surgery is required
  tests/
    correctness.yaml
    performance.yaml
```

Use an existing model with a similar task as a reference:

- [`resnet_50`](https://github.com/sima-neat/models/tree/develop/models/resnet_50) for a simple classification model with
  no surgery and pending qualification data;
- [`ssd_mobilenet_v1`](https://github.com/sima-neat/models/tree/develop/models/ssd_mobilenet_v1) for pinned calibration,
  model-specific surgery, and several compile variants;
- [`yolo26_tiny_drone`](https://github.com/sima-neat/models/tree/develop/models/yolo26_tiny_drone) for a restricted source,
  qualified reference results, and route-specific benchmarking.

The model README should record upstream context, unusual preparation or
compiler decisions, known limitations, and the meaning of each variant. It
must not be the only place where a runtime requirement is declared.

## 3. Pin the source

Hugging Face sources use a full 40-character repository commit:

```yaml
source:
  uri: hf://models/owner/repository@0123456789abcdef0123456789abcdef01234567/path/model.onnx
  sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  size_bytes: 123456
  media_type: application/onnx
```

`models`, `datasets`, and `spaces` are valid Hugging Face repository types.
Mutable revisions such as `main`, credentials, query parameters, fragments,
and unapproved hosts are rejected. Approved HTTPS sources are pinned Hugging
Face resolve URLs and GitHub release assets. See `SourceSpec.resolve_url()` in
[`source.py`](https://github.com/sima-neat/models/blob/develop/tools/model_registry/source.py)
for the enforced source policy.

Record checksums from the exact bytes the registry will download. On macOS use
`shasum`; on Linux, `sha256sum` is usually available:

```bash
shasum -a 256 /path/to/model.onnx
wc -c < /path/to/model.onnx
```

Do not add this temporary download to the repository. After entering its digest
and byte count in `model.yaml`, verify the declaration:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id>
```

The fetch is streamed, checked for exact size and SHA-256, and atomically stored
in a checksum-addressed cache. It never makes the downloaded binary a Git input.

## 4. Describe the model contract

Create a schema-version 2 `model.yaml`. It must describe:

- identity, ownership, license, upstream provenance, and maturity;
- sanctioned taxonomy references from
  [`model-taxonomy-v2.yaml`](https://github.com/sima-neat/models/blob/develop/taxonomies/model-taxonomy-v2.yaml);
- every ONNX input and output, including name, dtype, shape, and useful layout;
- ordered preprocessing and postprocessing behavior;
- calibration status and immutable calibration source when ready;
- surgery behavior;
- one or more compile variants;
- correctness and performance test declarations.

The manifest is a stable local contract. Do not rely on an upstream model card
remaining unchanged or on a recipe silently inferring decisions that belong in
metadata. Inspect the pinned ONNX with ONNX tooling and compare it with the
upstream conversion; do not guess tensor names, shapes, layouts, or opset.

If the correct category does not exist, follow the versioning rules in the
[taxonomy guide](model-taxonomy.md). Do not invent an authoritative category in
one model manifest.

## 5. Declare calibration and tests honestly

Use `calibration.status: ready` only when the declared source is immutable,
checksummed, representative, and legally usable. Use `pending` with a concrete
reason while evidence is being assembled, or `not_required` with a reason when
the compile path does not require calibration.

Add every referenced test file inside the model directory. A pending test file
should say what evidence is missing. A qualified reference should identify the
dataset checksum, thresholds, and measured values needed by reviewers. The
model schema verifies that test paths exist; model-specific test formats and
release gates may impose additional requirements.

For image-classification accuracy measurement, add an executable correctness
YAML with `engine: model_registry`. Keep sampling, workers, executors, metrics,
and reporting declarative; do not add a model-local evaluator. Follow
[Local accuracy development](local-accuracy-development.md) to compile and run
the regular and qualification profiles inside the SDK.

## 6. Add model-specific preparation and compilation

Set surgery to `none` when the pinned source can be compiled directly:

```yaml
surgery:
  type: none
```

For a genuinely model-specific transformation, keep the hook inside the model
directory and declare its callable and parameters:

```yaml
surgery:
  type: python
  entrypoint: surgery.py:prepare_model
  parameters:
    input_name: images
```

Generic download, checksum, cache, MPK validation, and publication behavior
belongs under `tools/`, not in model code.

The model recipe is the developer-facing compilation interface. The
author-safe `model-registry` package provides validation, verified input
handling, matrix inspection, and artifact checks; it is not a model-author
publishing client. CI retains separate registry publication commands.

Each model owns a Python compile recipe with this interface:

```text
compile.py --model <verified-source> --output-dir <new-directory> --variant <variant-id>
```

The recipe must:

- consume the already verified source passed by `--model`;
- read only the selected, declared variant parameters;
- fail on an unknown or incompatible variant;
- make preparation, quantization, and compilation deterministic;
- create the new output directory rather than reuse stale output;
- leave exactly one directly consumable `*_mpk.tar.gz` there;
- never download an unpinned input or select another compiler version.

The repository-wide compiler version comes from
[`deps/manifest.json`](https://github.com/sima-neat/models/blob/develop/deps/manifest.json).
Do not put a compiler pin in a
variant.

## 7. Declare variants

Give each meaningful target/configuration combination a unique snake-case ID
and explicit parameters:

```yaml
compile_variants:
  - id: modalix_int8
    target: modalix
    recipe: compile.py
    parameters:
      precision: int8
      input_shape: [1, 3, 224, 224]
```

Variants are supported by default. If qualification proves that a declared
combination cannot work with the repository compiler, retain it as
`status: unsupported` and provide the evidence in `reason`. Unsupported
variants remain documented but are excluded from compilation and packaging.
Do not mark an untested or temporarily failing variant unsupported.

Variants may request a logical runner class when the default runner is
insufficient. Read [Compile runner class requests and
approvals](runner-class-requests.md) before adding one. Elevated capacity must
be approved in the Vulcan scaler specification and is enforced before the build
is scheduled; a model manifest cannot authorize its own infrastructure cost.

## 8. Validate the contribution

Run the model-owned recipe and benchmark flow from
[Local development and testing](local-development.md). Use these internal,
non-publishing repository checks before and after that work:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id>
model-registry fetch-calibration models/<model-id>
model-registry matrix --repository . --changed-path models/<model-id>/model.yaml
model-registry resolve-compiler --repository .
model-registry validate-catalog --repository .
```

`fetch-calibration` reports that no calibration is required when status is not
`ready`. Inspect the matrix JSON and confirm that it contains every intended
supported variant and no unintended model.

When the exact pinned Model Compiler environment and a compatible Modalix
DevKit are available, continue through local compilation, MPK verification, and
the declared performance test in
[Local development and testing](local-development.md). If either environment is
unavailable, stop at the last completed stage and state exactly which compiler
or DevKit checks were not run in the pull request.

## 9. Submit the change

Follow [Contributing with Git and GitHub](contributing.md). A typical new-model
commit contains only the model directory and, when necessary, a separately
reviewable taxonomy update. Never commit downloaded models, calibration or test
datasets, MPKs, compiler intermediates, logs, credentials, or signed URLs.

See [Contributing with Git and GitHub](contributing.md) for the submission and
automation boundary. Model authors never publish registry state directly.
