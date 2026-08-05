# SSD-MobileNet V3 Large 320x320

This entry deploys the 12 raw predictor heads of TorchVision's pretrained
`ssdlite320_mobilenet_v3_large` COCO model. Neat owns anchor decode, scoring,
and NMS, so postprocessing is deliberately outside the compiled graph.

## Four supported variants

| Variant | Precision source | Tessellation | Public ABI |
| --- | --- | --- | --- |
| `modalix_int8_tess_mla` | learned W8A8 QAT QDQ | MLA | FP32 input and 12 FP32 heads |
| `modalix_int8_tess_off_mla` | learned W8A8 QAT QDQ | EV74 outside MLA | FP32 input and 12 FP32 heads |
| `modalix_bf16_tess_mla` | FP32 source, BF16 compiler quantization | MLA | FP32 input and 12 FP32 heads |
| `modalix_bf16_tess_off_mla` | FP32 source, BF16 compiler quantization | EV74 outside MLA | FP32 input and 12 FP32 heads |

The immutable source bundle contains one FP32 ONNX and one learned INT8 QDQ
ONNX with the same input, outputs, and 98-convolution topology. `surgery.py`
only validates the bundle and extracts the appropriate member; it never edits
weights or adds runtime nodes. Both tessellation placements therefore consume
the same precision source and can differ only in compiler placement.

Model Compiler 2.1.3 lacks `quantization_mla` and public BF16/FP32 boundary
APIs, so this recipe does not use them. The public buffers remain FP32. The
serialized graph and MPK audits prove internal precision and placement instead
of inferring them from a variant name.

## Why INT8 uses QAT

Pure post-training W8A8 quantization of the former TensorFlow V3 source failed
the shared raw-head gate even after 128 representative samples, SmoothQuant,
iterative bias correction, and lossless graph preparation. The failure was
numerical, not a tessellation problem, so changing placement could not repair
it.

The replacement uses TorchVision's stock COCO_V1 SSDLite checkpoint and SiMa
fake quantization. All 70 mature BatchNorm layers are folded before observers
are inserted, making training see the exact small folded weights that will be
deployed. Activations use per-tensor INT8 fake quantization and convolution
weights use symmetric axis-0 per-output-channel INT8 fake quantization. No
post-hoc weight floor, SmoothQuant, bias correction, BF16 fallback, INT16, or
mixed-precision escape is applied.

The QAT exporter will publish only a full `train2017` run evaluated against all
of `val2017`, with at least 128 representative observer images, frozen
observers, and no more than 1.5 bbox AP points of loss from the independently
measured FP32 baseline. Diagnostic `--smoke` checkpoints are explicitly marked
ineligible. See [`qat/README.md`](qat/README.md) for the reproducible Python
3.10 environment, training command, export contract, and provenance checks.
The qualified v1 source measured `0.21268746` FP32 AP and `0.20692255` QAT AP,
a `0.57649`-point drop.

## Independent compiler gates

Each compiler run reloads the saved graph in a clean process and evaluates all
32 held-out, category-balanced COCO samples. INT8 is compared with its qualified
QDQ source; BF16 is compared with the FP32 source. The audit records every raw
head's cosine similarity, normalized RMSE, mean absolute error, and maximum
absolute error. INT8 requires at least `0.98` minimum cosine and at most `0.20`
normalized RMSE; BF16 requires `0.999` and `0.05`.

Structural checks independently require:

- exactly 98 `SIMA_QUANTIZED` MLA convolutions;
- per-channel integer convolution weights for INT8 or BF16 tensors and weights
  for BF16;
- one MLA ELF, no shared libraries, and no A65 stages;
- all 12 internal output transport types and the FP32 public buffer sizes;
- an external EV74 tessellation stage exactly when tessellation is not fused
  into MLA.

The QAT ONNX audit also reports activation QDQ count, minimum learned weight
scale, and all-zero quantized weight channels. Such channels are visible to and
optimized by QAT, so a compiler warning about them is evidence about the learned
source rather than silent PTQ corruption. The full-COCO AP gate and compiled
numerical gate remain authoritative.

## Source, input, and calibration

Release `ssd-mobilenet-v3-qat-source-v1` contains the deterministic bundle
(`5655500a6f5046b60c3e90b24d18d9c67977a732abeeccdbacadf9dfbce3ac5a`,
15,784,877 bytes). It pins TorchVision 0.18.1, the COCO_V1 weight URL, its full SHA-256,
the QAT checkpoint digest, tool versions, training schedule, both ONNX digests,
and full-COCO accuracy evidence. `model.yaml` pins the deterministic bundle by
release asset ID, size, and SHA-256. The six predictor maps are 20, 10, 5, 3,
2, and 1; every level uses six anchors, for 3,234 anchors total.

The public tensor is RGB stretched to 320x320, rescaled to `[0, 1]`, normalized
by ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`, then transposed to NCHW. The existing immutable
calibration asset contains 128 calibration and 32 disjoint validation tensors
in the old `[-1, 1]` representation. The compiler recipe converts those values
back to `[0, 1]` and applies ImageNet normalization before execution, preserving
the pinned category-balanced sample selection and hashes without republishing
the images.

## Run on Vulcan

Dispatch `Vulcan Model Compile` in staging with model `ssd_mobilenet_v3`.
Leaving `variants` empty builds all four supported combinations; one variant ID
can be supplied for focused diagnosis. Compilation, fresh-process numerical
validation, structural graph audit, MPK audit, and artifact publication must
all succeed before a variant is usable.
