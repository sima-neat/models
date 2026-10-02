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
| `modalix_int8` | INT8 | Quantized build with deterministic accuracy measurement |

Both profiles use the repository-wide Model Compiler version pinned in
`deps/manifest.json`, apply direct-MLA tessellation, and produce one directly
consumable MPK archive for the `model` component.

## Quality and performance

INT8 correctness uses the registered `imagenet-2012/mini` archive. It contains
500 calibration images and a balanced 5,000-image validation proxy with five
images for every ImageNet-1k class. The regular CI profile evaluates 500 images;
the qualification profile evaluates all 5,000.

Reports record the selected profile, status, evaluated and available sample
counts, deterministic selection digest, worker count, artifact and dataset
hashes, top-1 and top-5 accuracy, agreement, and FP32 deltas. These results are
provisional because calibration and validation currently overlap. BF16 uses one
zero-valued sample only to satisfy the compiler quantization entrypoint and does
not run an accuracy measurement.

## Source and license

The source is the ONNX Community conversion of
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50), pinned to a
full Hugging Face repository revision and verified by size and SHA-256 in
`model.yaml`. The preprocessing values come from the pinned upstream
`preprocessor_config.json`. The model is distributed under Apache-2.0.

## Reproducing the artifact

### Compilation

Resolve the registered dataset, then compile the verified ONNX in the pinned
Model Compiler environment:

```bash
dataset="$(dataset-registry resolve imagenet-2012 --variant mini | python -c 'import json,sys; print(json.load(sys.stdin)["path"])')"
python models/resnet_50/compile.py \
  --model /path/to/model.onnx \
  --dataset "$dataset" \
  --output-dir /tmp/resnet50-int8 \
  --build-profile modalix_int8 \
  --component model
```

The Vulcan recipe must run after `activate-model-compiler` in the same
interactive shell so executables such as `mla-masm` and their runtime libraries
remain available.

### Validation

Run the regular correctness profile after compilation:

```bash
model-registry evaluate models/resnet_50 \
  --build-profile modalix_int8 \
  --component model \
  --reference-model /path/to/model.onnx \
  --candidate-model /tmp/resnet50-int8/resnet_50_modalix_int8.sima \
  --dataset "$dataset" \
  --profile regular \
  --workers 4 \
  --output /tmp/resnet50-int8/accuracy.json
```

Use `--profile qualification` for the full 5,000-image proxy. Vulcan uses 500
samples by default; a manual dispatch may set `accuracy_samples` to `5000` for
an on-demand qualification run. Generated models and reports must remain
outside Git.
