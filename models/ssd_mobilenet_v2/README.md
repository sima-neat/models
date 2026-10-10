# SSD-MobileNet V2 300x300

## Overview

SSD-MobileNet V2 is a COCO object detector that accepts a normalized 300×300
RGB image and returns 12 raw box and class predictor heads. Neat owns anchor
decoding, scoring, and non-maximum suppression, so applications must complete
postprocessing outside the compiled graph. Modalix builds are available in INT8
and BF16 with the same FP32 public interface. The published quality evidence
measures raw-head agreement against the floating model rather than end-to-end
COCO detection accuracy.

## Usage

Download either published build from the `develop` catalog:

```bash
sima-cli models download --id ssd_mobilenet_v2 --variant modalix_int8 --branch develop
sima-cli models download --id ssd_mobilenet_v2 --variant modalix_bf16 --branch develop
```

Use the 12 outputs in strict `loc[0..5], confidence[0..5]` order. The
application is responsible for generating anchors, decoding boxes, computing
class scores, and applying NMS.

## Model interface

The model consumes one NCHW FP32 RGB tensor with shape `1×3×300×300`. Images
are stretched to 300×300 with bilinear resampling and normalized to `[-1, 1]`.
It emits six localization heads followed by six confidence heads as FP32 public
buffers. The immutable source is already cut at these raw predictor heads.

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
the calibration and validation image IDs are disjoint. INT8 requires at least
`0.98` minimum cosine and at most `0.20` normalized RMSE. BF16 requires `0.999`
and `0.05`. Malformed, wrong-shaped, non-finite, zero-signal, or regressed
results fail immediately.

Independent structural checks verify all 76 MLA convolutions, the expected
INT8 or BF16 compute representation, one MLA ELF, no shared objects or A65
stages, all 12 output transport types, FP32 public buffer sizes, and no external
tensor-layout transforms. A profile name alone is not accepted as evidence of
the compiled precision.

## Source and license

The immutable source is
[`Kalray/ssd-mobilenet-v2`](https://huggingface.co/Kalray/ssd-mobilenet-v2),
pinned by repository revision, file size, and SHA-256 in `model.yaml`. It is a
static NCHW FP32 ONNX model with 12 existing raw predictor heads and is
distributed under Apache-2.0.

## Reproducing the artifact

### Graph surgery

`surgery.py` validates the complete input and output contract, ONNX opset 13,
and the 76 source Conv nodes. It renames the existing input and producer tensors
and reorders only graph-output metadata into
`loc[0..5],confidence[0..5]`. It inserts, removes, or changes no computational
node and adds no Identity publication stages.

The expected serialized `ConvAddActivation` count is derived from the source's
76 Conv nodes. The audit deliberately stops qualification if a compiler version
represents that topology differently.

### Calibration

V2 reuses the immutable 300×300 archive qualified for V1 because both models
have the same input shape and preprocessing contract. It contains 128
calibration samples and 32 disjoint validation samples selected from COCO
val2017. The selection covers all 80 annotated categories. Images are decoded
as RGB, stretched once to 300×300, and normalized to FP32 `[-1, 1]`.

The recipe verifies the archive, calibration tensor, validation tensor, and
annotation checksums declared in `model.yaml` and the calibration code. It also
checks exact shapes, dtypes, IDs, finite ranges, and split disjointness.
Regenerate the shared asset from the repository root with:

```bash
python models/ssd_mobilenet_v2/scripts/calibration.py \
  --annotations instances_val2017.json \
  --images val2017 \
  --output ssd_mobilenet_v1_coco_val2017_calibration128_validation32_v1.npz
```

### Compilation and validation

Dispatch **Vulcan Model Orchestrator** in staging with model
`ssd_mobilenet_v2`. Leaving the profile selection empty builds both declared
profiles; select one profile only for focused compiler diagnosis. Compilation,
raw-head quality validation, structural graph inspection, MPK inspection, and
artifact publication must all succeed before a build is usable.

The workflow compiles a build only when the catalog has no build with its
current compile key; set `force_rebuild_all` to recompile anyway. A dispatch
that selects profiles or components publishes those compile runs but does not
assemble packages or update the catalog.
