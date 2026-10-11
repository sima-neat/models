# ResNet-50

![ResNet-50 model hero](hero.png)

## Overview

ResNet-50 is an ImageNet-1k image classifier based on the pinned ONNX Community
conversion of `microsoft/resnet-50`. It accepts a normalized 224×224 RGB image
and returns logits for 1,000 ImageNet classes. The INT8 build is measured with a
deterministic ImageNet mini validation proxy, while the BF16 build serves as a
compiler canary rather than an accuracy-qualified variant. The current proxy is
provisional because its calibration and validation images overlap.

## Usage

Download either published build from the `develop` catalog:

```bash
sima-cli models download --id resnet_50 --variant modalix_bf16 --branch develop
sima-cli models download --id resnet_50 --variant modalix_int8 --branch develop
```

Validate and fetch the pinned source for local development:

```bash
model-registry validate models/resnet_50
model-registry fetch models/resnet_50 --component model
dataset-registry resolve imagenet-2012 --variant mini
```

## Model interface

The model consumes one NCHW FP32 RGB tensor with shape `1×3×224×224`. Images
are resized on their shortest edge to 256 pixels, center-cropped to 224×224,
rescaled to `[0, 1]`, and normalized with the ImageNet mean and standard
deviation declared in `model.yaml`. The output is one FP32 tensor containing
1,000 ImageNet class logits. Applications own class-label lookup and ranking.

## Available builds

| Build profile | Precision | Purpose |
| --- | --- | --- |
| `modalix_bf16` | BF16 | Compiler canary; no accuracy measurement |
| `modalix_int8` | INT8 | Quantized build with DevKit accuracy measurement |

Both profiles use the repository-wide Model Compiler version pinned in
`deps/manifest.json`, apply direct-MLA tessellation, and produce one directly
consumable MPK archive for the `model` component.

## Quality and performance

Both builds report numerical fidelity: during compilation, the recipe compares
the compiled model's logits with the FP32 source on 50 seeded images from the
validation split of the registered `imagenet-2012/mini` archive. Fidelity is a
diagnostic measurement, not task accuracy.

The INT8 build also measures ImageNet top-1 and top-5 accuracy on a Modalix
DevKit with its MPK, over the validation split of the same archive. That split
holds five images for each of the 1,000 classes. The result is provisional.
INT8 calibration uses the archive's separate 500-image calibration split. BF16
uses one zero-valued sample only to satisfy the compiler quantization
entrypoint and has no accuracy measurement.

Both builds are benchmarked on a Modalix DevKit with Neat Core's default
synthetic workload.

## Source and license

The source is the ONNX Community conversion of
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50), pinned to a
full Hugging Face repository revision and verified by size and SHA-256 in
`model.yaml`. The preprocessing values come from the pinned upstream
`preprocessor_config.json`. The model is distributed under Apache-2.0.

## Reproducing the artifact

### Compilation

In the pinned Model Compiler environment, compile one build with the shared
runner script. It resolves the source, the calibration archive, and the
fidelity archive, runs `scripts/compile.py`, and validates the MPK:

```bash
scripts/run_model_compilation.sh models/resnet_50 modalix_int8 model /tmp/resnet50
```

The MPK and `fidelity.json` land in `/tmp/resnet50/resnet_50/modalix_int8/model/`.
Use `modalix_bf16` for the BF16 build. The recipe must run after
`activate-model-compiler` in the same interactive shell so executables such as
`mla-masm` and their runtime libraries remain available.

### Accuracy

On a Modalix DevKit, measure the INT8 build's accuracy from that compile
output:

```bash
REGISTRY_PYTHON=<python with model-registry and dataset-registry> \
  scripts/run_model_accuracy.sh models/resnet_50 modalix_int8 /tmp/resnet50 /tmp/resnet50-accuracy
```

The script resolves the dataset, runs `scripts/accuracy.py` on all 5,000
validation images, and writes `accuracy-result.json`. See
[Accuracy on the DevKit](../../docs/local-accuracy-development.md#accuracy-on-the-devkit).
Generated models and reports must remain outside Git.
