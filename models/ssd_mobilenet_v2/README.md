# SSD-MobileNet V2 300x300

This entry compiles the 12 raw box and class predictor heads of the pinned
Kalray SSD-MobileNet V2 ONNX model. Neat owns anchor decode, scoring, and NMS;
the immutable source is already cut at the raw predictor heads.

## Compiler-default build profiles

| Profile | Internal precision | Public ABI |
| --- | --- | --- |
| `modalix_int8` | W8A8 INT8 | FP32 input and 12 FP32 heads |
| `modalix_bf16` | BF16 weights/activations | FP32 input and 12 FP32 heads |

AFE 3.0 owns tensor-layout defaults. These two precision profiles replace
both `_tess_mla` and `_tess_off_mla` profile pairs. Calibration, quantization
and the public tensor contract are unchanged.

The audit proves the resulting behavior from two independent compiler outputs.
The serialized quantized topology checks all 76 MLA convolutions and verifies
per-channel integer weights for INT8 or BF16 compute tensors/weights for BF16.
The MPK checks one MLA ELF, no shared objects or A65 stages, all 12 output
transport tensor types, FP32 public buffer sizes, and absence of external
tensor-layout transforms in the MPK manifest. A profile name alone is
never accepted as evidence of precision.

Every compiler run also executes the floating and quantized raw-head models on
all 32 held-out, category-balanced COCO samples. The audit records per-head
cosine similarity, normalized RMSE, mean absolute error, and maximum absolute
error; calibration and validation image IDs are disjoint. Vulcan measurements
qualify INT8 at minimum cosine `0.98` and maximum normalized RMSE `0.20`, and
BF16 at `0.999` and `0.05`. Malformed, wrong-shaped, non-finite, zero-signal,
or regressed results fail immediately.

## Source and zero-cost surgery

The source is immutable:

- Hugging Face repository: `Kalray/ssd-mobilenet-v2`
- revision: `25a67532e30418eb68d1f49888374c16e8abfa73`
- file: `ssd-mobilenet-v2.onnx` (67,387,440 bytes)
- SHA-256: `f0d9458cd8ae5c0cde8eadf22b75c6451059f4d9dd46df266e996dcbec417a8f`

The source contract is static NCHW FP32 `1x3x300x300`, normalized to `[-1, 1]`,
with 12 existing raw heads in interleaved level order. `surgery.py` validates
that complete contract, ONNX opset 13, and the 76 source Conv nodes. It then
renames the existing input and producer tensors and reorders only graph output
metadata into `loc[0..5],confidence[0..5]`. No computational node is inserted,
removed, or changed: only value names and graph-output metadata change. In
particular, no Identity publication stages are added.

The expected serialized `ConvAddActivation` count is therefore derived from
the source's 76 Conv nodes. This mapping is compiler-dependent until the first
Model Compiler 2.1.3 Vulcan artifact proves it, so the audit is deliberately
strict and will stop qualification if the compiler represents that topology
differently.

## Representative calibration

V2 deliberately reuses the exact immutable 300x300 archive qualified for V1;
the models have the same input shape and preprocessing contract. The prerelease
contains 128 calibration samples and 32 disjoint validation samples selected
from COCO val2017. Selection greedily
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
python models/ssd_mobilenet_v2/calibration.py \
  --annotations instances_val2017.json \
  --images val2017 \
  --output ssd_mobilenet_v1_coco_val2017_calibration128_validation32_v1.npz
```

The annotations file must have SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

## Run on Vulcan

Dispatch `Vulcan Model Compile` in staging with model
`ssd_mobilenet_v2`. Leaving `profiles` empty builds all four combinations. A
single build profile can be selected for compiler diagnosis without changing the
recipe or metadata.
