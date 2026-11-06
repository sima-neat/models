# `model.yaml` reference

[`schemas/model.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/model.schema.json)
is the authoritative contract. JSON Schema rejects unknown fields; semantic
validation then checks references, taxonomy compatibility, source policy, and
model-local paths.

`model.yaml` describes a deployable model: the source it compiles, the build
profiles users choose between, and what each profile's whole model must
measure on a DevKit.

```text
model
├── conversion ───────────── optional upstream source for scripts/convert.py
├── compilation ──────────── the source the recipe compiles, and its components
├── datasets ─────────────── immutable dataset pins shared across stages
├── calibrations ─────────── Dataset Registry archives build profiles calibrate on
├── build_profiles ───────── each variant's per-component build parameters
├── pipeline ─────────────── optional run order, for the model browser
├── numerical_fidelity ───── compiled outputs compared during compilation
├── accuracy ─────────────── whole-model DevKit accuracy
├── performance ──────────── whole-model DevKit performance
└── reference_metrics ────── scores the source model's authors published
```

`model.yaml` is the model's only configuration file. Every stage reads its own
section, and model-owned code lives in `scripts/` beside it:

```text
models/<model-id>/
  model.yaml
  README.md
  hero.png      1200x480 catalog card image
  scripts/      compile.py (or compile.sh for LLiMa), performance.py, accuracy.py, convert.py, and helpers
  tests/        tests.yaml and the model's stage tests
```

Validation rejects any other entry in the model directory.

## Product identity and provenance

Keep the display name, description, taxonomy, license and provenance. The model
directory supplies `id`, and `architecture` is a plain ID such as `qwen`.
Ownership lives in `CODEOWNERS`.
See [Model taxonomy](model-taxonomy.md) for sanctioned taxonomy values.

`upstream` records human-reviewable provenance. It defaults to the checkpoint's
Hugging Face repository and revision; `model_card` defaults to its repository
URL. Keep an explicit `upstream` when it identifies a different original model.
Downloadable inputs belong in a `conversion` or `compilation` `source`, where
exact bytes are pinned.

Optional top-level `build-on-demand` is a boolean, default `false`, for any
model requiring explicit build selection. When `true`, pushes validate the
model and publish its metadata but never compile or measure it. Manual
selection requires model and profile IDs. This field changes scheduling and
catalog metadata only; it is excluded from compilation fingerprints. See
[on-demand builds](adding-a-model.md#on-demand-builds).

## Conversion

`conversion` is optional. It names the upstream checkpoint that
`scripts/convert.py` exports to ONNX:

```yaml
conversion:
  source:
    uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/checkpoint.pth
    sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    size_bytes: 123456
```

A model that declares `conversion` must also provide `scripts/convert.py` and
`scripts/requirements-convert.txt`, which the **Vulcan Model Convert** workflow
runs through `scripts/run_model_conversion.sh` in a fresh Python environment.
The workflow publishes a validated ONNX bundle to Vulcan. Set
`compilation.source: conversion` to resolve its latest compatible output during
compile planning. The declaration remains environment independent; see
[conversion artifacts](conversion-artifacts.md) for rollout and reproduction.

## Compilation and sources

`compilation` names the one source the recipe compiles and its format:

```yaml
compilation:
  format: onnx
  source:
    uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/model.onnx
    sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    size_bytes: 123456
```

The recipe is `scripts/compile.py` for `onnx` and `scripts/compile.sh` for
`huggingface`. A single-component model produces one component, `model`. A
multi-component model lists its components in run order:

```yaml
compilation:
  format: onnx
  source: {uri: ..., sha256: ..., size_bytes: ...}
  components: [backbone, decoder]
```

`entrypoint` names the package's public entrypoint when it is not `default`:
`generate` for Qwen and Gemma, `transcribe` for Whisper. Preprocessing,
postprocessing and surgery live in executable code, with application
instructions in the README. Tensor details come from Core execution plans
rather than a copied inventory.

A source is one file, as above, or a set of files with the `main_file` the
recipe opens. Files in one Hugging Face repository share its `repository` and
`revision`:

```yaml
source:
  repository: owner/repo
  revision: 0123456789abcdef0123456789abcdef01234567
  main_file: config.json
  files:
    config.json: {sha256: 1111111111111111111111111111111111111111111111111111111111111111, size_bytes: 1000}
    model.safetensors: {sha256: 2222222222222222222222222222222222222222222222222222222222222222, size_bytes: 2000}
```

A member's own `uri` overrides the shared repository and revision. Every
member is verified separately; the tool materializes the declared layout in a
content-addressed cache and passes the main file to the recipe.

## Datasets and calibration

Each stage pins the Dataset Registry archive it uses. A pin gives the
dataset `id`, `variant`, `sha256`, and `size_bytes`, and the archive goes by
its Registry source ID, `<id>-<variant>`. Calibration archives are
`calibrations` entries:

```yaml
calibrations:
- id: coco-2017-cal
  dataset: {id: coco-2017, variant: cal, sha256: 54c64d84…, size_bytes: 187510866}
```

A build profile that needs calibration names its entry with
`calibration_dataset`. The recipe receives the verified archive as
`--dataset`, calibrates the first component, and calibrates later components
on its outputs. Calibration prepares quantization ranges; it is not a
quality measurement. See [Datasets for local model
development](datasets.md).

## Build profiles

A build profile is one variant users can select. It gives each component's
build parameters:

```yaml
build_profiles:
  w8a8_w16a16:
    calibration_dataset: coco-2017-cal
    components:
      backbone:
        parameters: {precision: w8a8, calibration_samples: 50}
      decoder:
        parameters: {precision: w16a16, calibration_samples: 50}
```

Compose the profile ID from the precision tokens in
[`naming-conventions.yaml`](../naming-conventions.yaml): one token per
component in component order, with a repeated token written once. Both
components W8A8 is `w8a8`; an INT8 backbone followed by a BF16 decoder is
`w8a8_w16a16`. Leave the target out of the ID. If two profiles of one model
would share an ID, append the first vocabulary tokens that tell them apart.
The Model Browser shows variant IDs in uppercase, while
`sima-cli models download --variant` takes the lowercase ID.

Each component's `parameters.precision` is its precision token, and the
recipe maps it to compiler settings. The target is `modalix`.

A profile may declare `benchmark_baseline: {model_id: gemma_4_e2b_it_textonly,
build_profile: w4a16}` to link to a normal-decoding build. Both references are
validated. This changes only its performance fingerprint; it neither compiles
nor benchmarks the baseline. Future comparison runners must check matching target
weights and context/image settings.

| Runner field | Values | Default | Access requirement |
| --- | --- | --- | --- |
| `runner_class` | `default`, `moderate`, `expensive`, `most-expensive` | `default` | Elevated classes require a sufficient Vulcan maximum-class grant. |
| `runner_market` | `spot`, `on-demand` | `spot` | On-demand requires an independent v3 `allow_on_demand: true` grant, including at default size. |

Both fields belong on the profile or a component binding, outside `parameters`.
Bindings inherit from the profile; prefer profile-level settings because all its
components compile on one runner. Effective markets must agree across that grouped
invocation. Neither field changes a compile fingerprint. `build-on-demand` is a
separate top-level scheduling flag, not a purchasing market. See [runner access
and approval](runner-class-requests.md) for eligibility, evidence and usage.

The recipe compiles a profile's components together in one compile
invocation, named after the profile, and writes a `compile-outputs.json` with
`model_id`, `build_profile_id`, `compile_invocation_id`, an `outputs` list
naming each component's MPK and SHA-256, and a `derived_graphs` list with the
SHA-256 of each component's derived graph. For a single-component model,
`scripts/run_model_compilation.sh` writes the file with `outputs` only when
the recipe does not.

The repository compiler version remains pinned centrally in
[`deps/manifest.json`](https://github.com/sima-neat/models/blob/develop/deps/manifest.json).

Each compiled MPK is named after its model and variant:
`<model_id>_<profile_id>_mpk.tar.gz`, or
`<model_id>_<profile_id>_<component_id>_mpk.tar.gz` when the model has
several components. `scripts/compile.py` passes the name without
`_mpk.tar.gz` to the compiler as the model name, and publication rejects an
MPK with any other name. For example, RF-DETR's `w8a8_w16a16` profile
produces `rf_detr_nano_w8a8_w16a16_backbone_mpk.tar.gz` and
`rf_detr_nano_w8a8_w16a16_decoder_mpk.tar.gz`. The stored MPK, the
package, and the browser and CLI downloads all use this name.

## Packages and pipeline

Every build profile publishes one package, `default`, holding all its
components. A single-component package is that component's MPK,
`<model_id>_<profile_id>_mpk.tar.gz`. A multi-component package is a
deterministic `<model_id>_<profile_id>_default.zip` holding the MPKs at its
top level. Trusted assembly keeps each MPK intact and records the package
manifest in `package-result.json`, validated against
[`package-manifest.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/package-manifest.schema.json);
the manifest is not shipped inside the package.

An ONNX model whose application also runs files on the CPU, such as CPU-side
ONNX graphs, tokenizer tables or voice data, lists them in `resources`, pinned
like a source. Every package of the model installs each resource's files
under its `install_path`, and the package is then a ZIP even with one
component:

```yaml
resources:
  supertone_runtime:
    source:
      repository: Supertone/supertonic-3
      revision: 724fb5abbf5502583fb520898d45929e62f02c0b
      main_file: onnx/tts.json
      files:
        onnx/text_encoder.onnx: {sha256: c7befd5e..., size_bytes: 36416150}
    install_path: supertonic-3
```

`model-registry validate` and `validate-models` hold each resource's source to
the rules of a compile source, before any compile runs.

A file the compile recipe derives for the CPU, such as tables from graph
surgery, goes in `compile-outputs.json` as `runtime_files`: a list of `path`,
relative to the profile's output directory, and `sha256`. The package carries
it at that path, beside the MPKs it was derived with. A resource is a compile
input, so changing one recompiles the model; LLiMa packages ship their
deployment folder and take neither.

A multi-component model can add `pipeline` so the model browser shows the
order its components run in. List each component once, in run order, and
name any CPU step between them:

```yaml
pipeline:
- {component: backbone}
- {cpu: Top 300 proposals}
- {component: decoder}
```

The browser reads `pipeline` only for display; the model's scripts run the
components.

## Quality and performance

Omit a section when its stage is not implemented. Each covers the whole
model, every component of a build profile together.

### `numerical_fidelity`

Computed by the compile recipe and written as `fidelity.json` beside the
profile's MPKs. The package run publishes it.

| Field | Meaning |
| --- | --- |
| `status` | `diagnostic`, `provisional`, or `ready`. |
| `dataset` | The Dataset Registry pin it compares on. |
| `selection` | `method: seeded_random_without_replacement`, with `seed`, `offset`, and `samples`. |
| `outputs` | Output names of the model to compare. |

The report follows
[`numerical-fidelity.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/numerical-fidelity.schema.json).
The section is a compile input.

### `accuracy`

Measured on a DevKit by `scripts/accuracy.py` in the accuracy stage:

```yaml
accuracy:
  datasets:
  - {id: imagenet-2012, variant: mini, sha256: d769e759…, size_bytes: 356052889}
  build_profiles: [w8a8]
```

| Field | Meaning |
| --- | --- |
| `datasets` | Dataset Registry pins; the stage resolves each on the DevKit and evaluates the model on all of them in one run. Each goes by its source ID, such as `imagenet-2012-mini`. |
| `build_profiles` | Optional; the profiles measured. Omitted, every profile is measured. |

The metrics each dataset's result must hold come from
`datasets/<registry id>.yaml` for the model's `tasks`; see [Required accuracy
metrics](datasets.md#required-accuracy-metrics). Metric names come from
[`tools/model_report/metrics.json`](../tools/model_report/metrics.json).
`tests/tests.yaml` supplies the expected values per build profile and dataset;
a missing expected measurement fails.

### `performance`

Measured on a DevKit by `scripts/performance.py` in the performance stage:

```yaml
performance: {}
```

It takes only the optional `build_profiles`, as `accuracy` does. Every report
must hold the metrics
[`metrics.json`](../tools/model_report/metrics.json) marks `required`. Decoder selection and measurement settings stay in the script, which
customers can copy and run with PyNeat; YAML does not duplicate them.

See [Accuracy and fidelity](local-accuracy-development.md) and [DevKit
performance](benchmarking.md) for the scripts and reports, and [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns) for how each section
feeds the fingerprints.

## Reference metrics

`reference_metrics` lists the scores the source model's authors published for
these weights, copied by hand from a pinned page. The model page shows them
beside the measured scores, and on their own when no build has an accuracy
report yet. Each entry has:

| Field | Meaning |
| --- | --- |
| `source` | HTTPS URL of the page that prints the scores, pinned to a commit or revision. |
| `dataset` | `id` and `split` of the evaluation set, such as `{id: coco-2017, split: val}`. |
| `conditions` | How the authors measured, such as input size, preprocessing, and precision. |
| `note` | Optional caveat a reader needs, such as a split the source does not name. |
| `metrics` | A map from accuracy metric names to values, in each metric's unit from `tools/model_report/metrics.json`. |

```yaml
reference_metrics:
- source: https://huggingface.co/Ultralytics/YOLO26/blob/<revision>/README.md
  dataset: {id: coco-2017, split: val}
  conditions: 640x640, NMS-free one-to-one head
  metrics: {coco_keypoint_map: 0.572, coco_keypoint_ap50: 0.833}
```

Metric names are accuracy metrics from `tools/model_report/metrics.json`, the
names `accuracy.py` reports, and every value is in its metric's unit. Record
only numbers the source prints, not values derived from them. Two entries may
not give the same metric for the same dataset. A reference score appears
beside a measured one only when both name the same dataset `id`. The field
feeds no fingerprint, so editing it reruns nothing.

## Validation

Run both checks from the repository root:

```bash
model-registry validate models/<model-id>
model-registry validate-models --repository .
```

Validation proves that the source-controlled contract is coherent. Developers
must still compile and measure every supported build locally before submitting
it.


## LLiMa checkpoint and deployment formats

A Hugging Face model uses `compilation.format: huggingface` and the
conventional `scripts/compile.sh`. Its source is a file set with `config.json`
as `main_file`. Declare the public `entrypoint: generate` or `transcribe`
separately; do not copy prompt/image/text tensor inventories into YAML. A
complete LLiMa deployment is one component, `model`, including its vision
encoder when present.
For speculative decoding, the source file set includes target and draft files
under their existing checkpoint folder names. `main_file` names the target's
nested `config.json`, and `parameters.draft_model_path` selects the draft folder.
Each deployed checkpoint retains its own `devkit/` and `elf_files/` folders.

The compiled component contains `deployment/` and a sibling
`deployment-manifest.json`. `compile-outputs.json` carries `artifact_path`
(`model/deployment-manifest.json` for the `model` component) and
`artifact_sha256`, the SHA-256 of that JSON index. `compile-result.json`
marks `format: llima` and `layout: directory`; its `size_bytes` is the sum
of deployment file sizes.

A native full package is a deployment folder with `manifest.json` at its
root. Its component uses `format: llima`, `path: .`, and the compile index's
checksum. Both manifests list deployment files with exactly these fields:

```text
files: [{path, artifact, sha256, size_bytes}]
```

`path` is relative to the deployment root. `artifact` is the stable logical
name `file-<sha256(relative POSIX path)>`; it does not encode a storage URL.
In a published package, every reference names a `model-file` artifact on
that same package run. The root artifact is `package`, type `model-package`,
media type `application/json`. Native compile runs use root artifact `llima`,
type `model-pack`, also a JSON index. Their artifact metadata includes
`layout: directory`, `storage_backend: s3`, `deployment_size_bytes`, and
`file_count`.

Files use content-addressed S3 storage, reused between compile and package
publication. Each file must be at most 5 GiB until multipart uploads
([#292](https://github.com/sima-neat/models/issues/292)); aggregate deployment
size has no limit. Retrieval uses public resolved artifact URLs to preserve
the directory and verify each file's size and SHA-256. Existing ONNX/MPK
metadata and packages retain their format.
