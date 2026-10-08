# `model.yaml` v2 reference

[`schemas/model.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/model.schema.json)
is the authoritative contract. JSON Schema rejects unknown fields; semantic
validation then checks references, dependency cycles, taxonomy compatibility,
source policy, and model-local paths.

Version 2 describes a deployable model product rather than assuming that one
catalog entry always produces one MPK. A product can contain several compiled
components, support several build profiles, include runtime resources, define
how the pieces compose, and publish one or more downloadable packages.

```text
model
├── conversion ───── optional named upstream sources for scripts/convert.py
├── components ───── independently compiled ONNX units
├── resources ────── packaged non-compiled files
├── calibrations ─── reusable compile-time data
├── build_profiles ─ target-wide settings plus component bindings
├── compositions ─── how component entrypoints form an application
├── packages ─────── what a user downloads
├── validation ───── compile-host checks scoped to a component or package
└── benchmark ────── DevKit performance measurements scoped to a component
```

`model.yaml` is the model's only configuration file. Every stage reads its own
section, and model-owned code lives in `scripts/` beside it:

```text
models/<model-id>/
  model.yaml
  README.md
  scripts/      compile.py, performance.py, and any model-specific helpers
```

## Product identity and provenance

The identity, taxonomy, ownership, license, `upstream`, and `maturity` fields
retain their existing meaning. `schema_version` must be `2`, and `id` must
match the model directory. See [Model taxonomy](model-taxonomy.md) for the
sanctioned taxonomy values.

`upstream` records human-reviewable provenance. Downloadable inputs belong in
a `conversion`, component, or resource `source`, where exact bytes are pinned.

## Conversion

`conversion` is optional. It names the upstream checkpoints that
`scripts/convert.py` exports to ONNX; each entry holds one `source`:

```yaml
conversion:
  default:
    source:
      type: file
      uri: https://github.com/owner/repo/releases/download/v1.0/checkpoint.pth
      sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
      size_bytes: 123456
```

Conversion sources follow the same source policy and file-set rules as
component sources. A model that declares `conversion` must also provide
`scripts/convert.py` and `scripts/requirements-convert.txt`, which the
**Vulcan Model Convert** workflow runs in an isolated Python environment for the
selected entry. The workflow publishes `model.onnx` and prints the source pin
to copy into the component `source` after surgery and compiler validation.

## Components and source sets

Each member of `components` is independently compiled and has its own source,
interface, preprocessing, postprocessing, surgery, and recipe:

```yaml
components:
  - id: encoder
    role: encoder
    source:
      type: file
      uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/encoder.onnx
      sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
      size_bytes: 123456
      media_type: application/onnx
    interface:
      format: onnx
      opset: 17
      entrypoints:
        - id: default
          inputs: [{name: input, dtype: float32, shape: [1, 80, frames]}]
          outputs: [{name: state, dtype: float32, shape: [1, 512]}]
      shape_profiles: []
    preprocessing: {pipeline: []}
    postprocessing: {pipeline: []}
    surgery: {type: none}
    recipe: scripts/compile.py
```

A source may be one `file` or a `file_set`. Use a file set when one recipe input
consists of an entrypoint plus related files:

```yaml
source:
  type: file_set
  entrypoint: encoder/model.onnx
  files:
    - path: encoder/model.onnx
      uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/encoder/model.onnx
      sha256: 1111111111111111111111111111111111111111111111111111111111111111
      size_bytes: 1000
      media_type: application/onnx
    - path: encoder/model.onnx.data
      uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/encoder/model.onnx.data
      sha256: 2222222222222222222222222222222222222222222222222222222222222222
      size_bytes: 2000
      media_type: application/octet-stream
```

Every member is verified separately. The tool materializes the declared
relative layout in a content-addressed cache and passes the entrypoint to the
recipe. `depends_on` may express ordering between components; cycles and
unknown component IDs are rejected.

The entrypoint may be an upstream checkpoint when the same file set also pins
the model-specific exporter package. In that case the recipe calls a local
`export.py` and keeps the generated ONNX temporary. Registry tooling only
fetches and verifies the declared files; it does not interpret checkpoint
formats or provide a universal converter.

An interface contains named `entrypoints`. Each entrypoint declares its input
and output tensors and may declare state and streaming behavior. Symbolic
dimensions are allowed. Put supported concrete bindings in `shape_profiles`,
then select one from a build-profile component binding.

## Resources and calibrations

`resources` are files shipped with a package but not compiled into an MPK—for
example tokenizers, vocabularies, labels, or configuration:

```yaml
resources:
  - id: tokenizer
    source:
      type: file
      uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/tokenizer.json
      sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      size_bytes: 4096
      media_type: application/json
    install_path: resources/tokenizer.json
```

A resource can carry its own license when it differs from the model license.
For a single-file source, `install_path` is the installed filename. For a file
set, it is a directory prefix and the declared member layout is preserved
below it. Packages list resources explicitly, so a resource is not shipped
merely because it is declared.

`calibrations` are named compile-time inputs. A ready calibration uses either
an immutable `source` or a Dataset Registry reference. A pending calibration
has a reason. A component binding references calibration by ID only when that
build needs it:

```yaml
calibrations:
  - id: representative_images
    status: ready
    dataset: {id: imagenet-2012, variant: calibration, split: train}
```

Calibration prepares quantization ranges; it is not an accuracy measurement.
Validation entries separately compare model outputs against declared evidence.
See [Datasets for local model development](datasets.md).

## Build profiles

A build profile names one product-wide target configuration. Its component
bindings say how each component is built under that profile:

```yaml
build_profiles:
  - id: modalix_int8
    target: modalix
    components:
      - id: encoder
        shape_profile: batch_1
        calibration: representative_audio
        parameters: {precision: int8}
      - id: decoder
        parameters: {precision: int8}
```

This avoids a Cartesian product of unrelated component configurations. The
compile matrix emits one job for every supported component binding unless the
profile declares `compile_invocations`:

```yaml
compile_invocations:
  - id: detector
    components: [backbone, transformer]
    python_packages:
      - path: exporter-1.2.3-py3-none-any.whl
        imports: [exporter]
```

One invocation runs the shared recipe once and must emit a schema-v2
`compile-outputs.json` with one MPK path for every listed component, derived
graph hashes, and one validation result. `python_packages` explicitly names
verified file-set members and imports needed by the recipe; CI does not infer
them from filenames. Grouped
components must share their source content, recipe, and effective runner class.
Selecting one grouped component selects the whole invocation. `runner_class` may
be set on the profile or overridden on a binding. A binding proven incompatible
may use `status: unsupported` with a reason; untested work is not unsupported.
If the invocation needs calibration data, its first component selects the one
shared calibration input that CI passes to the recipe.

A grouped recipe with an `engine: model_recipe` validation entry writes one
`fidelity.json` beside its owning component MPK. The report follows
`schemas/numerical-fidelity-v1.schema.json` and records numerical agreement for
the full composed pipeline. The validation entry names that owner with
`result_component` and declares the output metrics. CI validates the report
against that entry before publishing any component. Numerical metrics
are diagnostic and do not gate compilation or publication. Execution failures,
invalid reports, and incomplete component sets still fail validation.

The repository compiler version remains pinned centrally in
[`deps/manifest.json`](https://github.com/sima-neat/models/blob/develop/deps/manifest.json).

A component binding may also declare `artifact_suffix` outside `parameters`.
It uses the identifier syntax and controls the browser download filename:
`<model_id>_<artifact_suffix>_mpk.tar.gz`. Include the component name in the suffix
when useful. For example, RF-DETR uses `backbone_int8_mse` and `transformer_bf16` under
one `modalix_int8_bf16` profile. The suffix is explicit and is not inferred from
precision. The complete download basename must fit within 255 characters.
Omitting or removing the suffix preserves the original download filename.

Suffix changes update catalog metadata while retaining existing compiled artifacts.
They do not change build identities, trigger compilation or benchmarking, or rename
stored objects and MPK contents. Registry support must deploy before catalogs
publish this optional field.

Use [`naming-conventions.yaml`](../naming-conventions.yaml) to compose suffixes
from the shared vocabulary, for example `int8_mse_tmio` or
`int8_mse_h1024_bci_l2e`. Prefix the component ID for multi-component models.
The model ID and `_mpk.tar.gz` are added by the download path. Keep the target,
versions, calibration sample counts, dataset identities, and seeds out of the
suffix. Add a token to the vocabulary when a new option needs a name.

This vocabulary is an author reference, with no token parser or automatic name
generation. Write `artifact_suffix` explicitly and check its meaning against the
recipe. It does not select compiler settings or replace existing parameters;
`compile.py` continues to own the AFE calls.

## Compositions and packages

A composition explains how the compiled pieces are used together. A
single-component model uses the built-in form:

```yaml
compositions:
  - id: default
    type: single
    component: model
    entrypoint: default
```

A multi-component product uses `type: declarative` and declares its graph
inline. Packaging writes the graph into the package as
`composition/composition.yaml`, validated against
`schemas/composition-v1.schema.json`:

```yaml
compositions:
  - id: pipeline
    type: declarative
    schema_version: 1
    components:
      - {id: encoder, component: encoder, entrypoint: default}
      - {id: decoder, component: decoder, entrypoint: default}
    connections:
      - from: {component: encoder, output: state}
        to: {component: decoder, input: state}
    inputs:
      - {name: audio, to: {component: encoder, input: input}}
    outputs:
      - {name: text, from: {component: decoder, output: tokens}}
```

A package is the retrieval unit visible to users:

```yaml
packages:
  - id: default
    default: true
    components: [encoder, decoder]
    resources: [tokenizer]
    build_profiles: [modalix_int8]
    composition: pipeline
```

Exactly one package is the default. Validation ensures each package includes
the components required by its composition and declares only profiles that can
build those components. Trusted assembly preserves each MPK intact inside one
deterministic bundle with a manifest, the selected resources, and the
declarative composition document when one is used.
The embedded manifest is validated against
[`package-manifest-v1.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/package-manifest-v1.schema.json).

## Validation and benchmark entries

`validation` entries are checks the compile workflow runs on the compile host,
except `engine: devkit_script` accuracy, which runs on the DevKit after the
benchmark. `benchmark` entries are measurements taken on a Modalix DevKit. Each entry
declares its scope and applicable build profiles, followed by its settings:

```yaml
validation:
  - id: encoder_accuracy
    scope: {type: component, component: encoder}
    build_profiles: [modalix_int8]
    engine: model_registry
    # accuracy settings; see Local accuracy development
benchmark:
  - id: encoder_latency
    scope: {type: component, component: encoder}
    build_profiles: [modalix_int8]
    status: provisional
    metrics: [latency_avg_ms, throughput_fps]
```

Entry IDs are unique within `validation` and within `benchmark`; model
scripts look up their settings by ID. Component entries qualify one MPK.
Package entries qualify the composed product. A validation entry with `engine: model_registry` runs the shared
accuracy evaluator, and one with `engine: model_recipe` declares the numerical
fidelity report a grouped recipe writes. One with `engine: devkit_script` runs
the model's `scripts/accuracy.py` on the benchmarked MPKs; see
[Accuracy on the DevKit](local-accuracy-development.md#accuracy-on-the-devkit).
An entry without `engine` holds
settings that only the model's own scripts read. `model-registry validate`
checks every entry's settings, including unknown fields and that
`required_metrics` are declared in `metrics`.

## Validation

Run both checks from the repository root:

```bash
model-registry validate models/<model-id>
model-registry validate-catalog --repository .
```

Validation proves that the source-controlled contract is coherent. Developers
must still compile and test every supported component/profile combination
locally before submitting it.
