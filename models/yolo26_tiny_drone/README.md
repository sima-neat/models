# YOLO26n-P2 Tiny Drone INT8 QAT

This registry entry publishes the qualified one-class, 1600-pixel YOLO26n-P2
detector used by the Apps tiny-drone tracker. The source is the epoch-7 QAT
export; its held-out AP50 is 0.9649 and tiny-object recall50 is 0.9810.

The `modalix_int8_tessellation_mla` build profile uses exactly 128 representative
MultiUAV training images. The calibration archive pins every JPEG digest and
the reconstructed NHWC FP32 tensor digest. CI revalidates those bytes and then
compiles the immutable ModelSDK 3.0.0 quantized snapshot, rebuilt from the same
ONNX and calibration inputs with AFE defaults. CI does not recalibrate the
model. ModelSDK must produce one MLA segment in which the INT8 quantizer feeds
MLA directly and tensor-layout conversion on MLA. The compiler may pack the
eight outputs into one MLA buffer; the audit validates the INT8 unpacking and
eight public output bindings. The compile audit rejects A65 fallback, host
shared objects, external tessellation or detessellation transforms, non-INT8
convolution/attention compute, and any
change to the eight named P2-P5 output bindings.

Install the branch package while the model PR is under review:

```bash
sima-cli neat install --stg \
  models/yolo26_tiny_drone@feat/yolo26-p2-tiny-drone-qat:latest \
  --install-dir models/yolo26_tiny_drone
```

After merge, use `models/yolo26_tiny_drone@main:latest` in production.
