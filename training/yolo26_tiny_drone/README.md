# YOLO26-P2 tiny-drone INT8 QAT

This staging recipe produces a one-class, batch-1 detector for the Neat
multi-stream tracker. It is deliberately outside the published model catalog
until a real drone dataset, a qualified checkpoint, and a separately licensed
representative calibration asset have immutable digests. A placeholder
`model.yaml` would make CI look complete while publishing no reproducible
model, so promotion is a separate, evidence-backed step.

## Why this YOLO26 variant

- P2 adds a stride-4 head; a 4–12 pixel drone retains substantially more
  spatial evidence than it does at stride 8.
- The end-to-end YOLO26 training head and its small-target-aware assignment are
  retained.
- The two PSA sites are replaced with C3k2 convolutional blocks before FP32
  training. SDK 2.0 QAT leaves their MatMul/Softmax path without QDQ. Keeping
  them would silently create a mixed-precision model; the convolutional
  replacement is trainable and exports as signed INT8 throughout the compute
  graph.
- Deployment exports the one-to-one raw heads in grouped-by-role order:
  `bbox_0..3, class_logit_0..3`. Neat `BoxDecodeType::YoloV26` derives strides
  4, 8, 16, and 32 from their spatial dimensions.

## Dataset contract

Use YOLO labels (`class cx cy width height`, normalized, class `0`) and a JSON
manifest. Every frame has an explicit source sequence, and a sequence may
appear in only one split. This prevents adjacent frames from the same video
inflating validation accuracy.

```json
{
  "schema_version": 1,
  "frames": [
    {
      "image": "images/sequence-01/frame-000001.jpg",
      "label": "labels/sequence-01/frame-000001.txt",
      "sequence": "sequence-01",
      "split": "train"
    }
  ]
}
```

Image and label paths are relative to the manifest directory and may not
escape it. Labels must follow Ultralytics' `images/...` to `labels/...` path
mapping, and every normalized box must lie fully inside its source image.

Production QAT refuses fewer than 128 distinct representative training
images. Use hundreds or thousands where available; 128 is a hard floor, not a
quality target. Put the exact evaluation video and near-miss negatives in the
held-out test sequence, never the training split.

## Environment

The SiMa QAT API requires Python 3.10 and PyTorch 2.3.1. Bootstrap from an SDK
2.0 QAT wheel or installation:

```bash
SIMA_QAT_WHEEL=/path/to/swml_qat.whl \
  training/yolo26_tiny_drone/bootstrap.sh
```

or:

```bash
SIMA_QAT_SITE_PACKAGES=/path/to/sdk/site-packages \
  training/yolo26_tiny_drone/bootstrap.sh
```

The public YOLO26n base weights are pinned to SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Supply the local file explicitly; the scripts do not silently change it.

## FP32 training

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/train_fp32.py \
  --manifest /data/drone/manifest.json \
  --base-weights /data/checkpoints/yolo26n.pt \
  --output-dir build/yolo26-drone-fp32 \
  --teacher-epochs 100 --recovery-epochs 30 --batch-size 16 --device 0
```

The first stage fine-tunes the official attention-based P2 model as an FP32
teacher. The second stage transfers its compatible backbone, neck, and head
weights into the INT8-safe graph and recovers the two surgically replaced
blocks. Only the recovered student proceeds to QAT. Both stages use
Ultralytics MuSGD, and the lineage records both checkpoint digests.

## Target-exact QAT

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/train_qat.py \
  --manifest /data/drone/manifest.json \
  --fp32-checkpoint build/yolo26-drone-fp32/student/weights/best.pt \
  --output-dir build/yolo26-drone-qat \
  --observer-warmup-images 128 \
  --epochs 10 --batch-size 1 --accumulation-steps 32 --device cuda
```

Batch 1 is mandatory. PT2E specializes YOLO26 reshapes to the example batch,
and the Modalix app is batch 1. Gradient accumulation supplies the effective
optimization batch without training a graph that cannot be deployed.

All mature BatchNorm layers are folded while retaining both training heads.
Activations use signed per-tensor INT8 fake quantization; convolution weights
use signed symmetric axis-0 per-channel INT8. Observers see at least 128 unique
unaugmented images selected reproducibly without replacement across the full
training split, and the checkpoint records every selected path and digest.
Observer updates are temporarily disabled for every held-out validation pass,
so validation images cannot influence quantization ranges, and are permanently
frozen after the first 20% of epochs by default, before the remaining low-rate
refinement. Set `--freeze-observers-fraction 0` to retain only the ranges from
the representative warm-up. Each fresh run records its fake-quantized accuracy
before the first optimizer step, separating initialization loss from training
regressions.

A frozen production checkpoint can be refined at a lower learning rate without
recalibrating or exposing observers to validation data. `--epochs` is the
number of additional epochs for a resumed invocation, and the new checkpoint
records the immutable resume digest and starting metrics. The exact resume
checkpoint remains the best candidate unless a new frozen epoch improves the
AP50/tiny-recall ordering:

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/train_qat.py \
  --manifest /data/drone/manifest.json \
  --fp32-checkpoint build/yolo26-drone-fp32/student/weights/best.pt \
  --resume-checkpoint build/yolo26-drone-qat/checkpoint-best-frozen.pt \
  --output-dir build/yolo26-drone-qat-refined \
  --epochs 10 --learning-rate 2e-5 \
  --observer-warmup-images 128 --batch-size 1 \
  --accumulation-steps 32 --device cuda
```

## Export gates

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/export.py \
  --checkpoint build/yolo26-drone-qat/checkpoint-best-frozen.pt \
  --fp32-checkpoint build/yolo26-drone-fp32/student/weights/best.pt \
  --output-dir build/yolo26-drone-source
```

Training writes `checkpoint-latest.pt` after every epoch for recovery and
retains `checkpoint-best-frozen.pt` using frozen-observer AP50 first and tiny
recall as the tie-breaker. Export the latter so a final-epoch regression cannot
silently replace a better publishable candidate.

Export fails when:

- observer evidence is below 128 images or observers were not frozen;
- validation has fewer than 100 tiny instances, QAT AP50 is below 0.30, or
  QAT tiny-object recall is below 0.50;
- AP50 drops more than 2 points or tiny-object recall drops more than 5 points;
- the input is not `1x3x640x640` or the eight P2–P5 outputs differ;
- BatchNorm, MatMul, Einsum, or Softmax remains;
- any convolution weight lacks signed symmetric per-channel INT8 QDQ.

## ModelSDK compile

Run the compiler in a ModelSDK 2.1 environment with `mla-masm`, `mla-msim`,
and CUDA `libdevice.10.bc` installed. Compilation independently selects a
seeded sample of at least 128 training images and reuses the
exact 640-square RGB letterbox preprocessing from training:

```bash
python training/yolo26_tiny_drone/compile.py \
  --export-dir build/yolo26-drone-source \
  --manifest /data/drone/manifest.json \
  --output-dir build/yolo26-drone-modalix \
  --calibration-images 128
```

The compiler refuses a manifest outside the checkpoint lineage or an
unqualified export. Its report pins every selected image, the assembled
calibration tensor, source ONNX, serialized INT8 graph, and MPK by SHA-256. It
also verifies 131 per-channel INT8 MLA convolutions, one MLA segment, no A65
fallback or host shared object, INT8 transport for all eight raw heads, and
float32 public boundaries expected by Neat.

`--allow-smoke` exists only for structural development checks. Its report is
explicitly marked ineligible for qualification and must never be promoted.

After compilation, validate the qualified MPK on Modalix with the held-out
drone sequences and the Apps tracking evaluator. Promotion into `models/`
still requires immutable model/calibration sources plus those target accuracy
results; this staging recipe does not publish synthetic or unmeasured assets.
