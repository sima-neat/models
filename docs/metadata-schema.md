# `model.yaml` reference

[`schemas/model.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/model.schema.json)
is the authoritative
schema-version 2 contract. YAML is loaded safely and validated with JSON Schema
Draft 2020-12. Semantic validation then checks taxonomy references, source
policy, duplicate variants, directory identity, and model-local file paths.

Unknown fields are rejected. This makes misspellings fail instead of silently
changing a build.

## Identity and ownership

| Field | Requirement and intent |
| --- | --- |
| `schema_version` | Must be `2`; version 1 manifests are rejected. |
| `id` | Lowercase snake case and identical to the model directory name. |
| `display_name` | Human-readable product/model name. |
| `description` | Concise statement of task, dataset or specialization, and important conversion details. |
| `owners` | One or more GitHub users or teams such as `@sima-neat/models`. |
| `maturity` | One of `experimental`, `staging`, `approved`, or `deprecated`. It reports lifecycle state; it does not waive other requirements. |

Start new models as `experimental` unless the required qualification and
release review have already occurred.

## Taxonomy

Every model declares the taxonomy version, broad model type, tasks, semantic
input/output modalities, and architecture:

```yaml
taxonomy_version: 3
model_type: vision
tasks: [image_classification]
modalities:
  inputs: [image]
  outputs: [class_scores]
architecture:
  id: resnet_50
  variant: imagenet_1k
```

`tags` are optional normalized secondary descriptors. IDs must satisfy the
schema and sanctioned references must exist in
[`model-taxonomy-v3.yaml`](https://github.com/sima-neat/models/blob/develop/taxonomies/model-taxonomy-v3.yaml).
Task and
architecture compatibility with `model_type` is checked semantically. Composite
architectures may declare `components`, each with a `role` and
`architecture_id`.

See [Model categorization taxonomy](model-taxonomy.md) before proposing a new
category. Published taxonomy versions are immutable.

## License

```yaml
license:
  spdx: Apache-2.0
  redistribution: allowed
  attribution: Derived from the named upstream model and distributed under Apache-2.0.
```

`redistribution` is `allowed`, `restricted`, `prohibited`, or `unknown`.
Use a valid SPDX expression when one exists and an approved `LicenseRef-*` for
internal terms. Attribution must be meaningful. Restricted or internal assets
need an approved distribution path; do not work around policy by committing the
bytes or embedding a credentialed URL.

## Immutable source and upstream provenance

`source` identifies the exact downloaded bytes used by compilation:

```yaml
source:
  uri: hf://models/owner/repository@0123456789abcdef0123456789abcdef01234567/model.onnx
  sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  size_bytes: 123456
  media_type: application/onnx
```

The SHA-256 is exactly 64 lowercase hexadecimal characters and `size_bytes` is
positive. Supported sources are full-commit Hugging Face resources or approved
GitHub release assets. The enforced allowlist and URL shapes live in
[`source.py`](https://github.com/sima-neat/models/blob/develop/tools/model_registry/source.py).

`upstream` records human-reviewable provenance rather than downloadable build
input:

```yaml
upstream:
  repository: https://huggingface.co/owner/repository
  revision: 0123456789abcdef0123456789abcdef01234567
  model_card: https://huggingface.co/owner/repository
  derived_from: https://example.com/original-project
```

`repository` and `revision` are required. `model_card` and `derived_from` are
optional URLs. Keep them precise enough for a reviewer to trace the conversion.

## ONNX interface

`interface.format` is currently `onnx`. Declare at least one input and output:

```yaml
interface:
  format: onnx
  opset: 17
  inputs:
    - name: images
      dtype: float32
      shape: [1, 3, height, width]
      layout: NCHW
      description: Normalized RGB image tensor.
  outputs:
    - name: logits
      dtype: float32
      shape: [1, 1000]
      layout: NC
```

Tensor names must match the pinned ONNX. Dimensions are positive integers or
lowercase symbolic names. Supported dtypes are `float16`, `float32`, `float64`,
`int8`, `uint8`, `int32`, `int64`, and `bool`. `layout`, `description`, and
`opset` are optional in the schema but should be present when they prevent
ambiguity.

## Preprocessing and postprocessing

Both fields contain an ordered `pipeline`. Each stage has a snake-case `op` and
an explicit `parameters` map:

```yaml
preprocessing:
  pipeline:
    - op: resize
      parameters: {height: 224, width: 224, resample: bilinear}
    - op: transpose
      parameters: {from: HWC, to: CHW}
postprocessing:
  pipeline:
    - op: softmax
      parameters: {axis: 1}
```

These declarations preserve the semantic contract even when operations are
implemented by an application, Neat, or model-specific route. Record ownership
or ordering parameters when that distinction matters. Use `pipeline: []` only
when no operation is required.

## Calibration

Calibration is always explicit:

- `pending` requires a reason describing the missing evidence;
- `ready` requires an immutable source using the same URI, checksum, and size
  policy as the model source;
- `not_required` requires a reason explaining why.

```yaml
calibration:
  status: pending
  reason: A representative, redistribution-approved dataset is not yet pinned.
```

Synthetic data may exercise a compiler path, but it does not become
representative accuracy evidence merely because compilation succeeds.

## Surgery

Use `type: none` when no model-specific transformation is needed. Python
surgery requires an entry point in `path.py:function` form and an explicit
parameter map:

```yaml
surgery:
  type: python
  entrypoint: surgery.py:prepare_model
  parameters:
    input_name: images
```

The referenced Python file must exist within the model directory. Keep reusable
transformations in shared tooling rather than copying them between models.

## Compile variants

Every model has at least one compile variant:

```yaml
compile_variants:
  - id: modalix_int8
    target: modalix
    recipe: compile.py
    parameters:
      precision: int8
      calibration_samples: 128
```

Variant IDs are unique lowercase snake case. The recipe is a model-local Python
file and parameters are the canonical, reviewable inputs for that variant. The
repository compiler pin in
[`deps/manifest.json`](https://github.com/sima-neat/models/blob/develop/deps/manifest.json)
applies to every variant.

`status` defaults to `supported`. A proven incompatible combination may use:

```yaml
status: unsupported
reason: Model Compiler 2.x rejects the documented operator after the recorded qualification attempt.
```

Unsupported variants are visible in the manifest but omitted from matrices and
release packaging. A non-empty reason is mandatory.

## Tests

Tests reference YAML files inside the model directory:

```yaml
tests:
  correctness:
    - id: dataset_accuracy
      config: tests/correctness.yaml
  performance:
    - id: modalix_latency
      config: tests/performance.yaml
```

Test IDs are lowercase and may include underscores or hyphens. Empty arrays are
schema-valid, but a real contribution should make missing qualification work
visible with a pending declaration instead of implying that testing is
unnecessary. See [Synthetic model benchmarking](benchmarking.md) for supported
performance settings and result semantics.

## Validation behavior

Run validation from the repository root. The author-safe `model-registry`
package is read-only in these commands; validation does not compile or publish
the model.

```bash
model-registry validate models/<model-id>
model-registry validate-catalog --repository .
```

Errors include paths such as `source.sha256` and the model directory. Catalog
validation checks every manifest and taxonomy reference, which catches changes
that are individually valid but inconsistent with repository-wide policy.
