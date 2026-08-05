# SSD-MobileNet V3 Large 320x320

This entry compiles the raw box and class predictor heads of the pinned
TensorFlow Object Detection Model Zoo SSD-MobileNet V3 Large model. Neat owns
anchor decode, scoring, and NMS, so the graph deliberately stops before the
source model's postprocessing.

## Four-variant matrix

| Variant | Internal precision | Tessellation placement | Qualification | Public ABI |
| --- | --- | --- | --- | --- |
| `modalix_int8_tess_mla` | W8A8 INT8 | MLA | Unsupported on compiler 2.1.3 | FP32 input and 12 FP32 heads |
| `modalix_int8_tess_off_mla` | W8A8 INT8 | outside MLA | Unsupported on compiler 2.1.3 | FP32 input and 12 FP32 heads |
| `modalix_bf16_tess_mla` | BF16 weights/activations | MLA | Supported | FP32 input and 12 FP32 heads |
| `modalix_bf16_tess_off_mla` | BF16 weights/activations | outside MLA | Supported | FP32 input and 12 FP32 heads |

The public boundary is intentionally left at the Model Compiler 2.1.3 default.
The recipe does not use `quantization_mla`, `public_input_types`, or
`public_output_types`, because those APIs are not available in the qualified
compiler. Only internal precision and tessellation placement vary.

The strongest pure-W8A8 recipe found for INT8 uses asymmetric per-tensor
activations, symmetric per-channel weights, min/max calibration over all 128
representative images, SmoothQuant, iterative bias correction, and SiMa
requantization. Before calibration, the INT8-only preparation raises one
already-nonzero coefficient in each of 865 negligible output channels to a
signed `1e-4` floor. It leaves truly zero channels untouched, adds no operators,
and produced bit-identical FP32 raw heads on all 160 calibration and held-out
samples. This prevents Model Compiler 2.1.3 from zeroing trained channels
during PTQ.

Removing that structural failure is not enough to qualify the result. Across
all 32 held-out samples, the recipe measured `0.94209` minimum raw-head cosine
similarity and `0.33537` maximum normalized RMSE. The same small diagnostic
split measured COCO AP `0.18420`, versus `0.23034` for FP32. It therefore fails
the same INT8 gate used by V1 and V2 (`0.98` minimum cosine and `0.20` maximum
normalized RMSE). Both INT8 placements remain explicit unsupported contracts
and are excluded from compilation and publication. BF16 uses the compiler's
MSE calibration over the same 128-image set and SiMa requantization.

The audit proves the resulting behavior from two independent compiler outputs.
The serialized quantized topology checks all 98 MLA convolutions and verifies
per-channel integer weights for INT8 or BF16 compute tensors/weights for BF16.
The recipe also rejects compiler warnings that report zeroed convolution
weights; serialized INT8 type metadata alone is not accepted as proof of a
usable quantization.
The MPK checks one MLA ELF, no shared objects or A65 stages, all 12 output
transport tensor types, FP32 public buffer sizes, and whether the external EV74
`tessellate` or fused INT8 `quanttess` stage is present. A variant name alone is
never accepted as evidence of precision.

Every compiler run also executes the floating and quantized raw-head models on
all 32 held-out, category-balanced COCO samples. The audit records per-head
cosine similarity, normalized RMSE, mean absolute error, and maximum absolute
error; calibration and validation image IDs are disjoint. The compiler gate
rejects malformed, non-finite, and zero-energy results. INT8 requires minimum
raw-head cosine similarity `0.98` and maximum normalized RMSE `0.20`; BF16
requires `0.999` and `0.05`. The first pinned 2.1.3 BF16 qualification measured
0.99965453 minimum cosine similarity and 0.02665781 maximum normalized RMSE.
The recipe replaces its quantize/compile process and
reloads the saved model in a clean process before executing held-out samples.
The compiler's TVM floating executor segfaults on this V3 graph, so validation
uses the prepared ONNX directly as an independent FP32 reference and the
compiler's MLA-matching JAX executor for the serialized quantized graph. Output
layouts are normalized explicitly before comparison. Numeric qualification and
the independent serialized-graph/MPK audits must both pass.
The independent zero-weight warning audit runs before model serialization, so
changing executors cannot mask a structurally invalid INT8 quantization.

## Source and precision-specific preparation

The immutable ONNX source is private release asset `500803387`, with SHA-256
`3b8a49dae52f4053e9197755aaa11fc227af8e234c7df19c7fe42ac847eed1c5`.
It was finalized from TensorFlow's
`ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz`, whose SHA-256 is
`8dade970d957ae5cb8b47e9fcefbe124e40b38c8641e484d2ab15b81d9b5126f`.
The source preserves the six predictor levels at 20, 10, 5, 3, 2, and 1.

The offline finalization folds four resized BatchNorm paths into their
convolutions and selects the NCHW predictor tensors without adding naming
Identity nodes. The registry `surgery.py` then proves the exact source hash,
FP32 signature, 98-convolution topology, upstream archive provenance, and
absence of Identity/BatchNorm runtime stages. BF16 receives a byte-for-byte
copy. INT8 receives the deterministic negligible-channel stabilization
described above, with the floor and exact changed-channel count embedded in
ONNX metadata and checked again after serialization. Both paths preserve the
public graph contract and prevent twelve A65 output copies.

## Representative calibration

The immutable calibration prerelease contains 128 calibration samples and 32
disjoint validation samples selected from COCO val2017. Selection greedily
balances categories using the pinned `instances_val2017.json`; the calibration
set covers all 80 annotated COCO categories. Both splits use original images
from `merve/coco` revision
`9e50abcdc1361852f34841af4939cbcd2d37c92f`, not the previously used
letterboxed 640x640 copies.

Each image is decoded as RGB, stretched once to 320x320 with bilinear resampling,
and normalized to float32 `[-1, 1]`. The recipe checks:

- archive SHA-256 `e412f8ed3d7ff8238eface1337c46bdf874180d85b7d68dcbe2508222e7ba98f`;
- calibration tensor SHA-256 `0e6d77ef341f68d80f6b22238c27df8e05f6c0128aac151789a17d257375b1a5`;
- validation tensor SHA-256 `3ba4fbecf2adc03c7f18578cfbe727e29b1ef6b034e60b4fccc7fafe045434c2`;
- exact sample shapes, dtypes, IDs, finite range, and split disjointness.

Regenerate the asset with:

```bash
python models/ssd_mobilenet_v3/calibration.py \
  --annotations instances_val2017.json \
  --images val2017 \
  --output ssd_mobilenet_v3_coco_val2017_calibration128_validation32_v1.npz
```

The annotations file must have SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

## Run on Vulcan

Dispatch `Vulcan Model Compile` in staging with model
`ssd_mobilenet_v3`. Leaving `variants` empty builds every supported combination
and skips cells with recorded compiler evidence. An explicit request for an
unsupported cell fails with that evidence. A single supported variant can be
selected for compiler diagnosis without changing the recipe or metadata.
