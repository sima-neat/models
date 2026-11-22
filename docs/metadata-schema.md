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
├── display_name ─────────── name on the model card
├── description ──────────── one-line description on the model card
├── example_app ──────────── optional: Apps portal page of an application that runs the model
├── parameter_count_billion  optional: parameter count in billions
├── model_type ───────────── taxonomy ID
├── tasks ────────────────── taxonomy IDs
├── modalities ───────────── input and output taxonomy IDs
├── architecture ─────────── model family ID
├── license ──────────────── license of the source model
├── upstream ─────────────── where the source model comes from
├── build-on-demand ──────── optional: build only when selected
├── conversion ───────────── optional: upstream source for scripts/convert.py
├── compilation ──────────── the source the recipe compiles, and its components
├── resources ────────────── optional: pinned files every package installs
├── calibrations ─────────── optional: Dataset Registry archives build profiles calibrate on
├── build_profiles ───────── each variant's per-component build parameters
├── pipeline ─────────────── optional: run order, for the model browser
├── numerical_fidelity ───── optional: compiled outputs compared during compilation
├── accuracy ─────────────── optional: whole-model DevKit accuracy
├── performance ──────────── optional: whole-model or independent-component DevKit performance
└── reference_metrics ────── optional: reference scores shown beside the measured accuracy
```

The tree lists every top-level field. A field without "optional" is required, except
`upstream`, which has a [default](#product-identity-and-provenance) for some sources.
Each stage pins its own datasets under `calibrations`, `numerical_fidelity`, or
`accuracy`.

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

Optional `parameter_count_billion` gives the unquantized total parameter count,
including embeddings and all experts, in billions (for example `0.8` or `5.1`).
Speculative models use the target's count.

Optional `example_app` is the Apps portal page of an example application that runs
this model:

```yaml
example_app: https://developer.sima.ai/examples/app/object-detection%2Fyolo26-object-detector
```

Copy the address from the application's page on the
[Apps portal](https://developer.sima.ai/examples). The model browser shows an
application icon beside the download action that opens it. Changing it reruns no
stage.

`upstream` records where the source model comes from, for people to review:

```yaml
upstream:
  repository: https://github.com/owner/repo
  revision: 0123456789abcdef0123456789abcdef01234567
  model_card: https://github.com/owner/repo/releases/tag/v1.0.0
  derived_from: https://github.com/original-owner/original-repo
```

| Field | Meaning |
| --- | --- |
| `repository` | Required. URL of the repository that holds the source model. |
| `revision` | Required. The revision of that repository, such as a commit. |
| `model_card` | Optional HTTPS URL of the page the model browser links as the original model. Defaults to `repository`. |
| `derived_from` | Optional URL of the original model or code that `repository` derives from, such as the base model of a requantized checkpoint. The model browser does not show it. |

`upstream` has a default in one case only. When `compilation.source` is a file set
that names a `repository` and `revision`, `upstream` defaults to that Hugging Face
repository and revision. Keep an explicit `upstream` there when it identifies a
different original model. Every other model declares `upstream` itself, including a
model whose source is one file or `conversion`. Without it, validation fails with
`upstream is required when the source has no Hugging Face repository`.

Downloadable inputs belong in a `conversion` or `compilation` `source`, where
exact bytes are pinned.

### License

`license` records the license of the source model:

```yaml
license:
  spdx: Apache-2.0
  redistribution: allowed
  attribution: Derived from microsoft/resnet-50 and distributed under Apache-2.0.
```

- `spdx` is an ID listed in [`artifact-browser/licenses.json`](../artifact-browser/licenses.json): an SPDX
  identifier, or a `LicenseRef-` ID for a license SPDX does not list.
- `redistribution` is `allowed`, `restricted`, `prohibited`, or `unknown`.
  Publication refuses a package marked `prohibited` or `unknown`.
- `attribution` is the credit line the model page shows.

Each entry there holds the license's `name`, a `summary` of what it
asks from someone who redistributes the model, and `downloadable`. The model
page shows the name and summary in its License section and offers the model's
downloads only when `downloadable` is `true`. A model whose license forbids
redistribution still compiles and benchmarks in CI, and its page shows **Not
available for download**.

Read the license at the source model's repository before declaring it. When it
is not listed, add an entry, validated by
`schemas/licenses.schema.json`. `python -m pytest tests/test_licenses.py` fails
when a model declares an unlisted license. No fingerprint includes the license
or the license list, so a correction reruns nothing.

### License files in the package

A license that allows redistribution asks that its text, and for MIT and BSD
the authors' copyright line, travel with every copy. A downloadable ONNX model
therefore ships its upstream license inside every package, as a `license`
[resource](#packages-and-pipeline) installed under `licenses/`:

```yaml
resources:
  license:
    source:
      uri: https://raw.githubusercontent.com/ultralytics/ultralytics/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/LICENSE
      sha256: 0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0
      size_bytes: 34523
    install_path: licenses/LICENSE
```

- Pin the license file of the source model itself, at the commit `upstream`
  names: the checkpoint's Hugging Face repository when it holds the file, or
  the authors' code repository as a GitHub raw file.
- When neither holds the file and the license is standard text with no
  authors' copyright line, such as Apache-2.0, pin that text from another
  repository and say so in a comment, as `resnet_50` does. Never borrow
  another project's MIT or BSD file: it names that project's authors.
- When the weights carry a second license, add one resource per license. The
  torchvision weights that timm republishes under Apache-2.0 ship
  `licenses/LICENSE` and `licenses/LICENSE-torchvision`.
- When the upstream has a `NOTICE` file, pin it the same way as
  `licenses/NOTICE`.
- A model whose license is `downloadable: false` offers no package and
  declares no license resource.
- The model's `performance.py` takes `--resources`, because the performance
  runner passes every model's resources, and reads nothing from it.

A resource is a compile input, so adding or changing a license pin recompiles
the model and remeasures it. LLiMa packages ship their deployment folder and
take no resources.

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

The checkpoint is a pinned Hugging Face file or a release file of a public
GitHub repository. [Pin every input](adding-a-model.md#3-pin-every-input) lists
the approved forms.

A model that declares `conversion` must also provide `scripts/convert.py` and
`scripts/requirements-convert.txt`, which the **Vulcan Model Convert** workflow
runs through `scripts/run_model_conversion.sh` in a fresh Python environment.
The workflow publishes the converted files. Set `compilation.source: conversion`
to compile its latest matching output. See
[conversion artifacts](conversion-artifacts.md) for how to publish a conversion and
when to run it again.

`conversion.outputs` is optional and lists the files `convert.py` writes, such as
several ONNX models or a `.pth` checkpoint. It defaults to `[model.onnx]`, and
`compile.py` receives the first as `--model`. See
[publish other files](conversion-artifacts.md#publish-other-files).

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

A single file, and each member of a file set, may add `media_type`, the file's content
type. It defaults to `application/octet-stream`. In a file set it is part of the
source's identity, so changing it recompiles the model:

```yaml
files:
  config.json: {sha256: 1111111111111111111111111111111111111111111111111111111111111111, size_bytes: 1000, media_type: application/json}
```

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
on its outputs. Calibration prepares quantization ranges. It is not a
quality measurement. See [Datasets for local model
development](datasets.md).

A component's `calibration_samples` parameter sets how many samples it
calibrates on. Keep it at 20 or fewer, because every sample adds compile time
to each build. Validation does not check this value.

## Build profiles

A build profile is one variant users can select. It gives each component's
build parameters:

```yaml
build_profiles:
  w8a16_w16a16:
    calibration_dataset: coco-2017-cal
    components:
      backbone:
        parameters: {precision: w8a16, per_token: true}
      decoder:
        parameters: {precision: w16a16, calibration_bins: 2048, calibration_samples: 20}
```

The model's `compile.py` decides what each parameter does. In this RF-DETR
Nano profile, `per_token: true` gives each token's activations their own INT8
scale in the backbone's INT8-weight operations.

Compose the profile ID from the precision tokens in
[`naming-conventions.yaml`](../naming-conventions.yaml): one token per
component in component order, with a repeated token written once. Both
components W8A8 is `w8a8`. A W8A16 backbone followed by a BF16 decoder is
`w8a16_w16a16`, as above. Leave the target out of the ID. If two profiles of one model
would share an ID, append the first vocabulary tokens that tell them apart.
The Model Browser shows variant IDs in uppercase, while
`sima-cli models download --variant` takes the lowercase ID.

Each component's `parameters.precision` is its precision token, and the
recipe maps it to compiler settings. The target is `modalix`.

A profile may declare `benchmark_baseline: {model_id: gemma_4_e2b_it_textonly,
build_profile: w4a16}` to link to a normal-decoding build. Both references are
validated. This changes only its performance fingerprint; it neither compiles
nor benchmarks the baseline.

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
invocation, named after the profile.
[Define build profiles](adding-a-model.md#6-define-build-profiles) lists what the
recipe must write.

`model.yaml` names no compiler version.
[`deps/manifest.json`](https://github.com/sima-neat/models/blob/develop/deps/manifest.json)
pins the Model Compiler for every model.

Each compiled MPK is named after its model and variant:
`<model_id>_<profile_id>_mpk.tar.gz`, or
`<model_id>_<profile_id>_<component_id>_mpk.tar.gz` when the model has
several components. `scripts/compile.py` passes the name without
`_mpk.tar.gz` to the compiler as the model name, and CI rejects an
MPK with any other name. For example, RF-DETR's `w8a16_w16a16` profile
produces `rf_detr_nano_w8a16_w16a16_backbone_mpk.tar.gz` and
`rf_detr_nano_w8a16_w16a16_decoder_mpk.tar.gz`. The package and the browser and
CLI downloads all use this name.

## Packages and pipeline

Every build profile publishes one package, `default`, holding all its
components. An ONNX model's package is a deterministic
`<model_id>_<profile_id>_default.zip` holding the MPKs at its top level, one
for a single-component model. A single-component build published before
packages became ZIPs stays a bare `<model_id>_<profile_id>_mpk.tar.gz` until
its next compile.

An ONNX model whose application also runs files on the CPU, such as CPU-side
ONNX graphs, tokenizer tables or voice data, lists them in `resources`. Pin
each resource like a source, but without `main_file`, because no recipe opens
a resource file. Give each resource its own `repository` and `revision`, even
when they match the compile source, so a resource can come from another
repository. Every package of the model installs each resource's files under
its `install_path`, and the package is then a ZIP even with one component:

```yaml
resources:
  supertone_runtime:
    source:
      repository: Supertone/supertonic-3
      revision: 724fb5abbf5502583fb520898d45929e62f02c0b
      files:
        onnx/text_encoder.onnx: {sha256: c7befd5e..., size_bytes: 36416150}
    install_path: supertonic-3
```

`model-registry validate` and `validate-models` hold each resource's source to
the rules of a compile source, before any compile runs. When a model's CPU code
must match its compiled graphs, as Supertonic's text encoder must match its
vector estimator, the model's build test checks that the two pins name the
same revision.

A file the compile recipe derives for the CPU, such as tables from graph
surgery, goes in `compile-outputs.json` as `runtime_files`: a list of `path`,
relative to the profile's output directory, and `sha256`. The package carries
it at that path, beside the MPKs it was derived with. A resource is a compile
input, so changing one recompiles the model; LLiMa packages ship their
deployment folder and take neither.

CLIP and SigLIP ship `text_encoder/tokenizer.json` and
`text_encoder/token_embeddings.npy` as `runtime_files`. Surgery extracts both from
the converted model, and the package carries them beside the encoder MPKs. The
accuracy runner passes the profile's output directory as `--resources`. Accuracy
evaluation generates the class embeddings, so the package does not hold them.

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
profile's MPKs. CI publishes it with the package.

| Field | Meaning |
| --- | --- |
| `status` | `diagnostic`, `provisional`, or `ready`. |
| `dataset` | The Dataset Registry pin it compares on. |
| `selection` | `method: seeded_random_without_replacement`, with `seed`, `offset`, and `samples`. `samples` is 1 to 5, because the compile runs the source and the compiled model on each one. Validation rejects more. |
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
| `samples` | Optional. A positive whole number. CI passes it to `accuracy.py` as `--limit`, so the script evaluates at most that many samples of each dataset. Keep it no larger than the smallest dataset the model evaluates, because most `accuracy.py` scripts draw that many samples at random and stop with an error when a dataset holds fewer. Omitted, the script evaluates every sample. Changing it measures accuracy again. |
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

The optional `build_profiles` selects the measured profiles, as for `accuracy`.
The optional `scope` is `model` by default: one measurement of the whole model,
including composed pipelines such as RF-DETR. Independent encoders declare:

```yaml
performance:
  scope: components
```

Each component then has separate metrics, workload, telemetry and execution
statistics in the report's `components` map, with no combined metrics. Both
scopes write one report per build profile. Core reports must hold the metrics
[`metrics.json`](../tools/model_report/metrics.json) marks `required`. Decoder selection and measurement settings stay in the script, which
customers can copy and run with PyNeat; YAML does not duplicate them.

LLMs and VLMs use [MoLE workloads](benchmarking.md#llima-text-and-image-performance)
with TTFT and token rates.

See [Accuracy and fidelity](local-accuracy-development.md) and [DevKit
performance](benchmarking.md) for the scripts and reports, and [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns) for how each section
feeds the fingerprints.

## Reference metrics

`reference_metrics` lists reference scores for these weights. Each entry has an
`origin`:

- `published` (the default): scores the source model's authors published,
  copied by hand from a pinned page.
- `fp32_baseline`: what the FP32 source model scores on the package's own
  inputs and protocol, measured on a host by the model's
  `scripts/fp32_baseline_accuracy.py`. Use it when no published score matches
  the package's setup. No stage runs the script, and it is not a compile input.

The model page shows each origin beside the measured scores under its own label,
**Published** or **FP32 baseline**, and on their own when no build has an
accuracy report yet. Each entry has:

| Field | Meaning |
| --- | --- |
| `origin` | `published` or `fp32_baseline`; `published` when left out. |
| `source` | For a published entry: HTTPS URL of the page that prints the scores, pinned to a commit or revision. |
| `script` | For an FP32 baseline: `scripts/fp32_baseline_accuracy.py`, which measured it. |
| `dataset` | `id` and `split` of the evaluation set, such as `{id: coco-2017, split: val}`. |
| `conditions` | How the scores were measured, such as input size, preprocessing, and precision. |
| `note` | Optional caveat a reader needs, such as a split the source does not name. |
| `metrics` | A map from accuracy metric names to values, in each metric's unit from `tools/model_report/metrics.json`. |

```yaml
reference_metrics:
- source: https://huggingface.co/Ultralytics/YOLO26/blob/<revision>/README.md
  dataset: {id: coco-2017, split: val}
  conditions: 640x640, NMS-free one-to-one head
  metrics: {coco_keypoint_map: 0.572, coco_keypoint_ap50: 0.833}
- origin: fp32_baseline
  script: scripts/fp32_baseline_accuracy.py
  dataset: {id: yfcc100m-keypoints, split: val}
  conditions: FP32 ONNX on ONNX Runtime CPU, 512 keypoints from a 640x480 image
  metrics: {pose_auc_5: 0.2865}
```

Metric names are accuracy metrics from `tools/model_report/metrics.json`, the
names `accuracy.py` reports, and every value is in its metric's unit. Record
only numbers the source or the script prints, not values derived from them. Two
entries of one origin may not give the same metric for the same dataset. An FP32
baseline names one of the model's `accuracy` datasets. A reference score appears
beside a measured one only when both name the same dataset `id`. The field
feeds no fingerprint, so editing it reruns nothing. `model-registry validate` and
the DevKit accuracy test compare each build profile's accuracy with it, as
[comment out a variant far below its reference score](adding-a-model.md#comment-out-a-variant-far-below-its-reference-score)
describes.

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

The compiled component is a `deployment/` folder with a sibling
`deployment-manifest.json`. The manifest lists each deployment file's `path`,
`artifact` name, `sha256`, and `size_bytes`. The package is that folder, delivered
as individual files with `manifest.json` at its root. No LLiMa build produces a ZIP.
CI rejects a deployment file larger than 5 GiB.
[LLiMa language and vision-language models](adding-a-model.md#llima-language-and-vision-language-models)
describes the recipe and how to check a deployment folder.
