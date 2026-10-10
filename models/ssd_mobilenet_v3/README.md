# SSD-MobileNet V3 Large 320x320

## Overview

SSD-MobileNet V3 Large is a TorchVision COCO object detector that accepts a
normalized 320×320 RGB image and returns 12 raw box and class predictor heads.
Neat owns anchor decoding, scoring, and non-maximum suppression, so applications
must complete postprocessing outside the compiled graph. The INT8 build uses a
qualified quantization-aware-trained source, while BF16 uses the corresponding
FP32 source. Published compiler checks measure raw-head agreement; the source
qualification provides the end-to-end COCO accuracy evidence.

## Usage

Download either published build from the `develop` catalog:

```bash
sima-cli models download --id ssd_mobilenet_v3 --variant modalix_int8 --branch develop
sima-cli models download --id ssd_mobilenet_v3 --variant modalix_bf16 --branch develop
```

Use the 12 outputs as six localization heads followed by six confidence heads.
The application is responsible for generating anchors, decoding boxes,
computing class scores, and applying NMS.

## Model interface

The model consumes one NCHW FP32 RGB tensor with shape `1×3×320×320`. Images
are stretched to 320×320, rescaled to `[0, 1]`, normalized with the ImageNet
mean and standard deviation, and transposed to NCHW. It emits 12 FP32 public
buffers over six predictor levels sized 20, 10, 5, 3, 2, and 1. Each level uses
six anchors, for 3,234 anchors in total.

## Available builds

| Build profile | Source | Internal precision | Public interface |
| --- | --- | --- | --- |
| `modalix_int8` | Qualified QAT QDQ ONNX | INT8 | FP32 input and 12 FP32 heads |
| `modalix_bf16` | FP32 ONNX | BF16 | FP32 input and 12 FP32 heads |

AFE 3.0 owns tensor-layout defaults. The immutable source bundle contains both
graphs with the same input, outputs, and 98-convolution topology. `surgery.py`
validates the bundle and extracts the correct member without editing weights or
adding runtime nodes.

## Quality and performance

The qualified source measured `0.21268746` FP32 bbox AP and `0.20692255` QAT
bbox AP over full COCO validation, a `0.57649`-point loss. The QAT exporter
rejects incomplete or smoke training, unfrozen observers, missing COCO evidence,
or a default loss above 1.5 AP points.

Each compiler run reloads the saved graph in a clean process and evaluates all
32 held-out, category-balanced COCO samples. INT8 is compared with the QDQ
source and requires at least `0.98` minimum cosine and at most `0.20` normalized
RMSE. BF16 is compared with the FP32 source and requires `0.999` and `0.05`.

Structural checks require exactly 98 `SIMA_QUANTIZED` MLA convolutions, the
declared precision for tensors and weights, one MLA ELF, no shared libraries or
A65 stages, all 12 output transport types, correct FP32 public buffer sizes, and
no external tessellation stage. The QAT audit also records activation QDQ count,
minimum learned weight scale, and all-zero quantized weight channels.

## Source and license

The source is derived from TorchVision 0.18.1
`ssdlite320_mobilenet_v3_large` COCO_V1 weights. The immutable bundle pins the
upstream weight URL, source and checkpoint digests, tool versions, training
schedule, both ONNX digests, and full-COCO accuracy evidence. Exact bundle
provenance is declared in `model.yaml`.

TorchVision and the derived model are distributed under BSD-3-Clause.

## Reproducing the artifact

### QAT training and export

The INT8 source uses SiMa fake quantization after all 70 mature BatchNorm layers
are folded. Activations use per-tensor INT8 fake quantization and convolution
weights use symmetric axis-0 per-output-channel INT8 fake quantization. No
post-hoc weight floor, SmoothQuant, bias correction, BF16 fallback, INT16, or
mixed-precision escape is applied.

Follow [`scripts/qat/README.md`](scripts/qat/README.md) for the reproducible Python 3.10
environment, full `train2017` training command, frozen-observer requirements,
export contract, and provenance checks. Diagnostic `--smoke` checkpoints are
explicitly ineligible for publication.

### Calibration

The compiler reuses the immutable category-balanced archive containing 128
calibration tensors and 32 disjoint validation tensors. The archive stores the
older `[-1, 1]` representation; the recipe converts it back to `[0, 1]` and
applies ImageNet normalization before execution. This preserves the pinned
selection and checksums without republishing the source images.

### Compilation and validation

Dispatch **Vulcan Model Orchestrator** in staging with model
`ssd_mobilenet_v3`. Leaving the profile selection empty builds both declared
profiles; select one profile only for focused diagnosis. Compilation,
fresh-process numerical validation, structural graph inspection, MPK
inspection, and artifact publication must all succeed before a build is usable.

The workflow compiles a build only when the catalog has no build with its
current compile key; set `force_rebuild_all` to recompile anyway. A dispatch
that selects profiles or components publishes those compile runs but does not
assemble packages or update the catalog.
