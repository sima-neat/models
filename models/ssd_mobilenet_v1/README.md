# SSD-MobileNet V1 300x300

This entry compiles the raw box and class predictor heads of the pinned ONNX
Model Zoo SSD-MobileNet V1 model. Neat owns anchor decode, scoring, and NMS, so
the graph deliberately stops before the source model's dynamic postprocessing.

## Four-variant matrix

| Variant | Internal precision | Tessellation placement | Public ABI |
| --- | --- | --- | --- |
| `modalix_int8_tess_mla` | W8A8 INT8 | MLA | FP32 input and 12 FP32 heads |
| `modalix_int8_tess_off_mla` | W8A8 INT8 | outside MLA | FP32 input and 12 FP32 heads |
| `modalix_bf16_tess_mla` | BF16 weights/activations | MLA | FP32 input and 12 FP32 heads |
| `modalix_bf16_tess_off_mla` | BF16 weights/activations | outside MLA | FP32 input and 12 FP32 heads |

The public boundary is intentionally left at the Model Compiler 2.1.3 default.
The recipe does not use `quantization_mla`, `public_input_types`, or
`public_output_types`, because those APIs are not available in the qualified
compiler. Only internal precision and tessellation placement vary.

The audit proves the resulting behavior from two independent compiler outputs.
The serialized quantized topology checks all 47 MLA convolutions and verifies
per-channel integer weights for INT8 or BF16 compute tensors/weights for BF16.
The MPK checks one MLA ELF, no shared objects or A65 stages, all 12 output
transport tensor types, FP32 public buffer sizes, and whether the external EV74
`tessellate` or fused INT8 `quanttess` stage is present. A variant name alone is
never accepted as evidence of precision.

Every compiler run also executes the floating and quantized raw-head models on
all 32 held-out, category-balanced COCO samples. The audit records per-head
cosine similarity, normalized RMSE, mean absolute error, and maximum absolute
error; calibration and validation image IDs are disjoint. Precision-specific
acceptance thresholds live in `tests/correctness.yaml` and fail compilation on
a quality regression.

## Source and zero-cost surgery

The source is immutable:

- Hugging Face repository: `onnxmodelzoo/ssd_mobilenet_v1_12`
- revision: `019281f3fcb151a90e491f3b2f0273f9f31bd6be`
- SHA-256: `b8fba5e404077d4048d27fcd1667e85e27e192eb9bf51e696c46a3acd7d21058`

`surgery.py` cuts the graph at the normalized tensor
`Preprocessor/sub:0`, publishes it as `images`, and extracts predictor outputs in
the strict order `loc[0..5],confidence[0..5]`. It renames the existing producer
tensors instead of appending Identity nodes; the latter would introduce twelve
unnecessary A65 output stages.

## Representative calibration

The immutable calibration prerelease contains 128 calibration samples and 32
disjoint validation samples selected from COCO val2017. Selection greedily
balances categories using the pinned `instances_val2017.json`; the calibration
set covers all 80 annotated COCO categories. Both splits use original images
from `merve/coco` revision
`9e50abcdc1361852f34841af4939cbcd2d37c92f`, not the previously used
letterboxed 640x640 copies.

Each image is decoded as RGB, stretched once to 300x300 with bilinear resampling,
and normalized to float32 `[-1, 1]`. The recipe checks:

- archive SHA-256 `ae4d46084839c7659c679a170366e8e2d143b1c42c911697bb59c5639e3bce20`;
- calibration tensor SHA-256 `ac4447c0b7a1b41ecd5c7d59528226dbc7b8f2e0d1340dd14dc285b0c316e25f`;
- validation tensor SHA-256 `c285dbc293b9cf24a25140bafbaad33768f7acef9014db21606d594a5bf2a6a1`;
- exact sample shapes, dtypes, IDs, finite range, and split disjointness.

Regenerate the asset with:

```bash
python models/ssd_mobilenet_v1/calibration.py \
  --annotations instances_val2017.json \
  --images val2017 \
  --output ssd_mobilenet_v1_coco_val2017_calibration128_validation32_v1.npz
```

The annotations file must have SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

## Run on Vulcan

Dispatch `Vulcan Model Compile` in staging with model
`ssd_mobilenet_v1`. Leaving `variants` empty builds all four combinations. A
single variant can be selected for compiler diagnosis without changing the
recipe or metadata.
