# YOLO26n-P2 Tiny Drone INT8 QAT

This registry entry publishes the qualified one-class, 1600-pixel YOLO26n-P2
detector used by the Apps tiny-drone tracker. The source is the epoch-7 QAT
export; its held-out AP50 is 0.9649 and tiny-object recall50 is 0.9810.

The `modalix_int8_tessellation_mla` variant uses exactly 128 representative
MultiUAV training images. The calibration archive pins every JPEG digest and
the reconstructed NHWC FP32 tensor digest. CI revalidates those bytes and then
compiles the immutable, qualified ModelSDK 2.1.3 quantized snapshot; it does not
silently recalibrate an already-qualified QAT model. ModelSDK must produce one
MLA segment in which the INT8 quantizer feeds MLA directly, with HWC input
tessellation and HWC16 output detessellation implemented on MLA. The compile
audit rejects A65 fallback, host shared objects, external tessellation or
detessellation transforms, non-INT8 convolution/attention compute, and any
change to the eight named P2-P5 output bindings.

Install the branch package while the model PR is under review:

```bash
sima-cli neat install --stg \
  models/yolo26_tiny_drone@feat/yolo26-p2-tiny-drone-qat:latest \
  --install-dir models/yolo26_tiny_drone
```

After merge, use `models/yolo26_tiny_drone@main:latest` in production.
