# `model.yaml` reference

[`schemas/model.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/model.schema.json)
is the authoritative contract. JSON Schema rejects unknown fields; semantic
validation then checks references, dependency cycles, taxonomy compatibility,
source policy, and model-local paths.

`model.yaml` describes a deployable model product rather than assuming that
one catalog entry always produces one MPK. A product can contain several compiled
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
├── fidelity ─────── numerical comparisons the compile stage computes
├── accuracy ─────── DevKit task accuracy, or recorded results
└── benchmark ────── DevKit performance measurements scoped to a component
```

`model.yaml` is the model's only configuration file. Every stage reads its own
section, and model-owned code lives in `scripts/` beside it:

```text
models/<model-id>/
  model.yaml
  README.md
  hero.png      1200x480 catalog card image
  scripts/      compile.py, performance.py, accuracy.py, convert.py, and helpers
```

Validation rejects any other entry in the model directory.

## Product identity and provenance

The identity, taxonomy, ownership, license, `upstream`, and `maturity` fields
are product metadata. `id` must match the model directory. See [Model taxonomy](model-taxonomy.md) for the
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
**Vulcan Model Convert** workflow runs through
`scripts/run_model_conversion.sh` in a fresh Python environment for the
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
    dataset:
      id: imagenet-2012
      variant: mini
      sha256: d769e759abba1fe6e03db704020dcff3f1e8309a8cac8885ab60f5a0e512c239
      size_bytes: 356052889
```

The recipe receives the verified calibration as `--dataset` and chooses its
samples. Calibration prepares quantization ranges; it is not a quality
measurement. `fidelity` and `accuracy` entries measure quality separately.
See [Datasets for local model development](datasets.md).

## Build profiles

A build profile names one product-wide target configuration. Its component
bindings say how each component is built under that profile:

```yaml
build_profiles:
  - id: w8a8
    target: modalix
    components:
      - id: encoder
        shape_profile: batch_1
        calibration: representative_audio
        parameters: {precision: w8a8}
      - id: decoder
        parameters: {precision: w8a8}
```

The profile ID is the variant ID users select. Compose it from the precision
tokens in [`naming-conventions.yaml`](../naming-conventions.yaml): one token
per component in component order, with a repeated token written once. Both
components above are W8A8, so the ID is `w8a8`; an INT8 backbone followed by a
BF16 transformer is `w8a8_w16a16`. Leave the target out of the ID. If two
profiles of one model would share an ID, append the first vocabulary tokens
that tell them apart. The Model Browser shows variant IDs in uppercase,
while `sima-cli models download --variant` takes the lowercase ID.

Each binding's `parameters.precision` is that component's precision token,
such as `w8a8`, and the recipe maps it to compiler settings.

This avoids a Cartesian product of unrelated component configurations. Each
supported binding is its own compile invocation, named after its component,
unless the profile declares `compile_invocations`:

```yaml
compile_invocations:
  - id: detector
    components: [backbone, transformer]
```

An invocation has only an `id` and its `components`. Every supported binding
belongs to exactly one invocation. Grouped components must share their source
content and effective runner class. They use at most one calibration, which
the first component must select; CI passes it to the recipe. A package or a `devkit_script`
accuracy entry must compile all its components in one invocation.

A grouped recipe runs once and writes a `compile-outputs.json` with
`model_id`, `build_profile_id`, `compile_invocation_id`, an `outputs` list
naming each component's MPK and SHA-256, a `derived_graphs` list with the
SHA-256 of each component's derived graph, and a `correctness` entry pointing
at the result component's `fidelity.json`. For a single-component invocation,
`scripts/run_model_compilation.sh` writes the file with `outputs` only when the
recipe does not.

`runner_class` may be set on the profile or overridden on a binding. A binding
proven incompatible may use `status: unsupported` with a reason; untested work
is not unsupported.

The repository compiler version remains pinned centrally in
[`deps/manifest.json`](https://github.com/sima-neat/models/blob/develop/deps/manifest.json).

Each compiled MPK is named after its model and variant:
`<model_id>_<profile_id>_mpk.tar.gz`, or
`<model_id>_<profile_id>_<component_id>_mpk.tar.gz` when the profile has
several components. `scripts/compile.py` passes the name without
`_mpk.tar.gz` to the compiler as the model name, and publication rejects an
MPK with any other name. For example, RF-DETR's `w8a8_w16a16` profile
produces `rf_detr_nano_w8a8_w16a16_backbone_mpk.tar.gz` and
`rf_detr_nano_w8a8_w16a16_transformer_mpk.tar.gz`. The stored MPK, the
package, and the browser and CLI downloads all use this name.

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
inline. Validation checks the graph against
[`composition.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/composition.schema.json):

```yaml
compositions:
  - id: pipeline
    type: declarative
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
    build_profiles: [w8a8]
    composition: pipeline
```

Exactly one package is the default. Validation ensures each package includes
the components required by its composition and declares only profiles that
build all of them in one compile invocation. Trusted assembly keeps each MPK
intact. A package with one component and no resources is that component's
MPK, `<model_id>_<profile_id>_mpk.tar.gz`. Any other package is a
deterministic `<model_id>_<profile_id>_<package_id>.zip` holding the MPKs at
its top level and each resource at its `install_path`. Assembly records the
package manifest in `package-result.json`, validated against
[`package-manifest.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/package-manifest.schema.json);
the manifest is not shipped inside the package.

A multi-component model can add `pipeline` so the model browser shows the
order its components run in. List each component once, in run order, and
name any CPU step between them:

```yaml
pipeline:
  - component: encoder
  - cpu: Beam search
  - component: decoder
```

The browser reads `pipeline` only for display; it does not change what is
compiled or packaged.

## Fidelity, accuracy, and benchmark entries

These three sections declare the quality and performance evidence of each
build. Every entry has an `id`, a `scope` (`{type: component, component: <id>}`
or `{type: package, package: <id>}`), and the `build_profiles` it covers. IDs
are unique within each section, and an entry may cover only components and
packages that are available in each listed build profile.

### `fidelity`

Computed by `scripts/compile.py` during the compile stage and written as
`fidelity.json` beside the result component's MPK.

| Field | Meaning |
| --- | --- |
| `scope` | A package. |
| `status` | `diagnostic`, `provisional`, or `ready`. |
| `result_component` | The package component whose outputs are compared and whose directory holds the report. |
| `dataset` | `id`, `variant`, `sha256`, and `size_bytes` of a registered dataset. |
| `selection` | `method: seeded_random_without_replacement`, with `seed`, `offset`, and `samples`. |
| `outputs` | Output names of the result component's entrypoints to compare. |

The report follows
[`numerical-fidelity.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/numerical-fidelity.schema.json).
Fidelity entries are compile inputs.

### `accuracy`

An entry with `engine: devkit_script` is measured by `scripts/accuracy.py` in
the accuracy stage:

| Field | Meaning |
| --- | --- |
| `engine` | `devkit_script`. |
| `status` | `ready` or `provisional`. |
| `result_component` | A scoped component; its compile run is the parent of the published result. |
| `dataset` | `id`, `variant`, `sha256`, and `size_bytes` of a registered dataset. |
| `metrics` | Metric names the script may report. |
| `required_metrics` | Optional subset of `metrics` that every result must contain. |

A build profile has at most one `devkit_script` entry, and its scoped
components must compile in one invocation.

An entry without `engine` records results measured elsewhere. It needs only
`id`, `scope`, and `build_profiles` and may carry any other fields. No stage
runs it.

### `benchmark`

Measured by `scripts/performance.py` in the performance stage, once for each
covered component:

| Field | Meaning |
| --- | --- |
| `status` | A lowercase identifier such as `provisional`; a status starting with `pending` needs a `reason` or `note`. |
| `metrics` | Metric names, as strings or `{name: ...}` objects. |
| `required_metrics` | Optional subset of `metrics` that must be measured for publication. |
| `target`, `reason`, `note` | Optional descriptive fields. |

```yaml
benchmark:
  - id: encoder_latency
    scope: {type: component, component: encoder}
    build_profiles: [w8a8]
    status: provisional
    metrics: [latency_avg_ms, throughput_fps]
```

See [Accuracy and fidelity](local-accuracy-development.md) and [DevKit
performance](benchmarking.md) for the scripts and results, and [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns) for how each section
feeds the stage keys.

## Published metrics

`published_metrics` lists the scores the source model's authors published for
these weights, copied by hand from a pinned page. The model page shows them
beside the measured scores, and on their own when no build has an accuracy
result yet. Each entry has:

| Field | Meaning |
| --- | --- |
| `source` | HTTPS URL of the page that prints the scores, pinned to a commit or revision. |
| `dataset` | `id` and `split` of the evaluation set, such as `{id: coco-2017, split: val}`. |
| `conditions` | How the authors measured, such as input size, preprocessing, and precision. |
| `note` | Optional caveat a reader needs, such as a split the source does not name. |
| `metrics` | A map from metric names to values. |

```yaml
published_metrics:
- source: https://huggingface.co/Ultralytics/YOLO26/blob/<revision>/README.md
  dataset: {id: coco-2017, split: val}
  conditions: 640x640, NMS-free one-to-one head
  metrics: {coco_keypoint_map: 0.572, coco_keypoint_ap50: 0.833}
```

Metric names come from
[`taxonomies/accuracy-metrics.yaml`](https://github.com/sima-neat/models/blob/develop/taxonomies/accuracy-metrics.yaml),
the names `accuracy.py` reports, and every value is a ratio from 0 to 1. Record
only numbers the source prints, not values derived from them. Two entries may
not give the same metric for the same dataset. A published score appears beside
a measured one only when both name the same dataset `id`. The field feeds no
stage key, so editing it reruns nothing.

## Validation

Run both checks from the repository root:

```bash
model-registry validate models/<model-id>
model-registry validate-models --repository .
```

Validation proves that the source-controlled contract is coherent. Developers
must still compile and measure every supported build locally before submitting
it.
