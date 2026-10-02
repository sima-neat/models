# SSD-MobileNet V1 300x300

![SSD-MobileNet V1 300x300 model hero](hero.png)

## Overview

SSD-MobileNet V1 is a COCO object detector that accepts a normalized 300×300
RGB image and returns 12 raw box and class predictor heads. Neat owns anchor
decoding, scoring, and non-maximum suppression, so applications must complete
postprocessing outside the compiled graph. Modalix builds are available in INT8
and BF16 with the same FP32 public interface. The published quality evidence
measures raw-head agreement against the floating model rather than end-to-end
COCO detection accuracy.

## Usage

Download either published build from the `develop` catalog:

```bash
sima-cli models download --id ssd_mobilenet_v1 --variant modalix_int8 --branch develop
sima-cli models download --id ssd_mobilenet_v1 --variant modalix_bf16 --branch develop
```

Use the 12 outputs in strict `loc[0..5], confidence[0..5]` order. The
application is responsible for generating anchors, decoding boxes, computing
class scores, and applying NMS.

## Model interface

The model consumes one NCHW FP32 RGB tensor with shape `1×3×300×300`. Images
are stretched to 300×300 with bilinear resampling and normalized to `[-1, 1]`.
It emits six localization heads followed by six confidence heads as FP32 public
buffers. The graph deliberately ends before the source model's dynamic
postprocessing.

## Available builds

| Build profile | Internal precision | Public interface |
| --- | --- | --- |
| `modalix_int8` | W8A8 INT8 | FP32 input and 12 FP32 heads |
| `modalix_bf16` | BF16 weights and activations | FP32 input and 12 FP32 heads |

AFE 3.0 owns tensor-layout defaults. These profiles replace the former
`_tess_mla` and `_tess_off_mla` pairs without changing calibration,
quantization, or the public tensor contract.

## Quality and performance

Every compiler run executes the floating and quantized raw-head models on 32
held-out, category-balanced COCO samples. The audit records per-head cosine
similarity, normalized RMSE, mean absolute error, and maximum absolute error;
the calibration and validation image IDs are disjoint. Precision-specific
acceptance thresholds in `tests/correctness.yaml` gate compilation.

Independent structural checks verify all 47 MLA convolutions, the expected
INT8 or BF16 compute representation, one MLA ELF, no shared objects or A65
stages, all 12 output transport types, FP32 public buffer sizes, and no external
tensor-layout transforms. A profile name alone is not accepted as evidence of
the compiled precision.

## Source and license

The immutable source is the ONNX Model Zoo SSD-MobileNet V1 model published as
`onnxmodelzoo/ssd_mobilenet_v1_12`. Its repository revision, file size, and
SHA-256 are pinned in `model.yaml`. The model is derived from the TensorFlow
Object Detection implementation and distributed under Apache-2.0.

## Reproducing the artifact

### Graph surgery

`surgery.py` cuts the graph at the normalized `Preprocessor/sub:0` tensor,
publishes it as `images`, and extracts predictor outputs in strict
`loc[0..5],confidence[0..5]` order. It renames existing producer tensors rather
than appending Identity nodes, which would add 12 unnecessary A65 output stages.

### Calibration

The immutable calibration prerelease contains 128 calibration samples and 32
disjoint validation samples selected from COCO val2017. The selection greedily
balances categories with the pinned annotations and covers all 80 annotated
COCO categories. Original images are decoded as RGB, stretched once to 300×300,
and normalized to FP32 `[-1, 1]`.

The recipe verifies the archive, calibration tensor, validation tensor, and
annotation checksums declared in `model.yaml` and the calibration code. It also
checks exact shapes, dtypes, IDs, finite ranges, and split disjointness.
Regenerate the asset from the repository root with:

```bash
python models/ssd_mobilenet_v1/calibration.py \
  --annotations instances_val2017.json \
  --images val2017 \
  --output ssd_mobilenet_v1_coco_val2017_calibration128_validation32_v1.npz
```

### Compilation and validation

Dispatch **Vulcan Model Compile** in staging with model
`ssd_mobilenet_v1`. Leaving the profile selection empty builds both declared
profiles; select one profile only for focused compiler diagnosis. Compilation,
raw-head quality validation, structural graph inspection, MPK inspection, and
artifact publication must all succeed before a build is usable.
