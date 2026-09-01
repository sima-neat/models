# DepthART Base dynamic-p64 strict INT8

This experimental entry packages the 448x576 DepthART Base QAT candidate. The
PyTorch surgery is mathematically equivalent; accuracy recovery comes from
source QAT plus an exact integer `q8 + p64` recurrence rather than a different
float model.

## Runtime tensor contract

| Boundary | Contract |
| --- | --- |
| Input | `image`, INT8 `[1,448,576,3]` NHWC |
| MLA compute | INT8 activations, per-output-channel INT8 weights, integer recurrence auxiliaries |
| Output | `depth`, INT8 `[1,448,576,1]` NHWC |

Quantize an ImageNet-normalized FP32 input with float64 intermediates:

```python
codes = np.clip(np.rint(normalized.astype(np.float64) * 53.62121217871723 - 14), -128, 127).astype(np.int8)
```

NumPy `rint` is ties-to-even. Decode relative inverse depth with:

```python
relative_inverse_depth = (codes.astype(np.float32) + 128) * 0.11401007324457169
```

The output is relative inverse depth, not meters. Metric depth needs an external
scale-and-shift alignment.

## What is different

The source patch under `training/depthart_base/` expresses the original affine
selective scan as an associative tree, splits independent state into physical
C128 blocks, and removes only `a * 0 + b = b` for the first chunk. The source
tests compare the old and new real-valued paths. No approximation is introduced
by architecture surgery.

The QAT API then observes and freezes power-of-two `p64` bases and executes the
following boundaries with target-exact integer math and STE gradients:

```text
static INT8 --static-to-p64--> q8+p64
q8+p64 x q8+p64 --integer product--> q8+p64
q8+p64 + q8+p64 --aligned integer add--> q8+p64
q8+p64 --p64-to-static--> static INT8
```

Every tree composition is published back to its declared static Q/DQ grid
before structural Slice/Concat/Reshape operators. The companion profile binds
918 products and 918 adds to exact ONNX source nodes and fails closed on a hash,
qparam, name, or carrier-ABI mismatch.

## Build

```bash
model-registry validate models/depthart_base
model-registry fetch models/depthart_base
python models/depthart_base/compile.py \
  --model <verified-source-onnx> \
  --output-dir <new-output-directory> \
  --variant modalix_strict_int8_dynamic_p64
python models/depthart_base/audit_mpk.py \
  --mpk <new-output-directory>/*_mpk.tar.gz \
  --output <new-output-directory>/precision-audit.json
```

The recipe verifies the ONNX, content-bound profile, strict-INT8 serialized
snapshot, public ABI, and single-MLA topology before compilation. The MPK audit
requires one stage, one ELF, zero shared objects, zero BF16/FP16/FP32 tensors,
and exact counts for all four recurrence primitives.

This variant requires the Model Compiler features listed in `model.yaml`. The
current registry pin does not yet imply those features in every stock 2.1.3
installation; keep the entry experimental until the corresponding compiler
integration is released. It must not silently fall back to BF16 or a runtime
CPU implementation.

## Accuracy evidence

The 100-image host recurrence isolation result is delta1 `0.919875`, AbsRel
`0.088424`, and `98.84%` delta1 retention versus the matched local FP32 model.
That result keeps non-recurrence layers in FP32 and is not a silicon claim. The
entry is promoted only after one strict-INT8 ELF passes the precision audit and
all 654 checksum-pinned NYUv2 test images pass on Modalix.

## Training and licensing

See `training/depthart_base/README.md` and `qat.yaml` for the exact QAT revision,
source patch, data policy, losses, export hashes, and acceptance gates. The
checkpoint model card declares Apache-2.0, but the upstream source repository
has no license file, so redistribution remains restricted pending approval.
