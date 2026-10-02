# YOLO26n-P2 Tiny Drone INT8 QAT

![YOLO26n-P2 Tiny Drone INT8 QAT model hero](hero.png)

## Overview

YOLO26n-P2 Tiny Drone is a one-class object detector optimized for small drones
in 1600×1600 imagery. It accepts a letterboxed RGB image and returns eight P2–P5
box and class prediction tensors for application-side decoding and tracking.
The INT8 QAT source achieved 0.9649 held-out AP50 and 0.9810 tiny-object
recall50. Source and calibration images are restricted internal assets and must
not be redistributed.

## Usage

Download the published Modalix build from the `develop` catalog:

```bash
sima-cli models download \
  --id yolo26_tiny_drone \
  --variant modalix_int8_tessellation_mla \
  --branch develop
```

To install the model package from a feature branch while it is under review:

```bash
sima-cli neat install --stg \
  models/yolo26_tiny_drone@<branch>:latest \
  --install-dir models/yolo26_tiny_drone
```

After merge, use `models/yolo26_tiny_drone@main:latest` for the production
package.

## Model interface

The model consumes one NCHW FP32 RGB tensor with shape `1×3×1600×1600`. Images
are letterboxed to 1600×1600 and normalized to `[0, 1]`. The graph emits four
box tensors and four one-class logit tensors over the P2–P5 feature levels. The
application owns tensor unpacking, box decoding, thresholding, suppression, and
tracker integration.

The compiler may pack the eight outputs into one MLA buffer, but the public
interface retains all eight named bindings declared in `model.yaml`.

## Available builds

The registry declares one `modalix_int8_tessellation_mla` build profile with a
single `model` component. It compiles the immutable ModelSDK 3.0.0 quantized
snapshot as one MLA segment, with the INT8 quantizer feeding MLA directly and
tensor-layout conversion placed on MLA.

## Quality and performance

The epoch-7 QAT source measured 0.9649 held-out AP50 and 0.9810 tiny-object
recall50. These are source-model qualification measurements. Device benchmark
results are published separately for the compiled Modalix artifact.

The compile audit rejects A65 fallback, host shared objects, external
tessellation or detessellation transforms, non-INT8 convolution or attention
compute, and changes to the eight named P2–P5 output bindings. It also validates
INT8 unpacking when the compiler packs outputs into one MLA buffer.

## Source and license

The source is an internal one-class YOLO26n-P2 detector derived from
Ultralytics and the pinned Anti-UAV training lineage declared in `model.yaml`.
The epoch-7 QAT export, source digest, and upstream revision are immutable
registry inputs.

The model uses `LicenseRef-SiMa-Internal`. Redistribution is restricted: do not
publish or copy the source model, calibration images, or derived internal
training assets outside approved SiMa systems.

## Reproducing the artifact

### Calibration and source reconstruction

The build uses exactly 128 representative MultiUAV training images. The
calibration archive pins every JPEG digest and the reconstructed NHWC FP32
tensor digest. CI verifies those bytes before compilation.

The immutable ModelSDK 3.0.0 quantized snapshot is rebuilt from the same ONNX
and calibration inputs with AFE defaults. CI does not recalibrate the model.

### Compilation and validation

Validate the declaration and fetch the pinned source from the repository root:

```bash
model-registry validate models/yolo26_tiny_drone
model-registry fetch models/yolo26_tiny_drone --component model
```

Dispatch **Vulcan Model Compile** in staging with model
`yolo26_tiny_drone`. The recipe must produce one MLA segment and preserve the
eight-output public contract. Compilation, structural graph inspection, MPK
inspection, accuracy evidence checks, and artifact publication must all succeed
before the build is usable. Generated sources, calibration data, MPKs, and
reports must remain outside Git.
