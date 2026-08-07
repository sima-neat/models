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
- The canonical graph retains both PSA attention sites from upstream
  YOLO26-P2. Before QAT capture, its fixed two-head attention blocks
  are unrolled per head so every MatMul keeps deployment batch axis zero at
  one, as required by ModelSDK. This exact lowering preserves the original
  parameters, state-dict paths, and attention math. The attention-capable SiMa
  QAT quantizer then adds signed INT8 QDQ to both inputs and the output of every
  MatMul, and to the input and output of every Softmax. Export rewrites each
  rank-4 MatMul to the exactly equivalent two-input Einsum spelling documented
  by ModelSDK for batch matrix multiplication; tensor shapes, QDQ boundaries,
  parameters, and arithmetic do not change.
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
FP32 training records a deterministic dataset digest over the manifest
metadata, relative paths, sizes, and bytes of every referenced image and
label. QAT rehashes the manifest and all referenced files before every
eligible frozen checkpoint, and compilation rehashes them again. Any content
change—including mutation through the hard links created by dataset
preparation—fails closed.

The published Anti-UAV Track 3 archive assigns frames rather than source
sequences to train and validation; all 200 source sequences consequently
cross that boundary. Prepare it with whole-sequence 80/10/10 splits before
training. The preparation verifies the published archive digest, validates
every label, retains negative frames with explicit empty labels, balances
sequence/frame/object totals, uses hard links rather than duplicating the
corpus, and records the CC-BY-4.0 source lineage in the manifest:

```bash
python training/yolo26_tiny_drone/prepare_multiuav.py \
  --source-archive /data/AntiUAV_train_val.zip \
  --source-root /data/extracted/AntiUAV_train_val \
  --output-dir /data/AntiUAV-sequence-safe
```

`MultiUAV-003`, the source of the reference tracker clip, is absent from this
training/validation archive and remains a fully external acceptance sequence.
Use repeatable `--test-sequence` arguments to force any included acceptance
sequences into the internal test partition.

Production QAT refuses fewer than 128 distinct representative training
images. Use hundreds or thousands where available; 128 is a hard floor, not a
quality target. Put the exact evaluation video and near-miss negatives in the
held-out test sequence, never the training split.

## Environment

The SiMa QAT API requires Python 3.10 and PyTorch 2.3.1. It must include the
SiMa MatMul and Softmax PT2E annotators; the original `swml-qat` develop.20
package does not. The dependency is implemented by internal `swml-qat` commit
`dee0915` on `feat/int8-matmul-softmax-qat`; this Models change must not merge
before that dependency. Bootstrap from an attention-capable QAT wheel or
source installation:

```bash
SIMA_QAT_WHEEL=/path/to/swml_qat.whl \
  training/yolo26_tiny_drone/bootstrap.sh
```

or:

```bash
SIMA_QAT_SITE_PACKAGES=/path/to/sdk/site-packages \
  training/yolo26_tiny_drone/bootstrap.sh
```

Bootstrap prepares a real two-head Ultralytics PSA block on CPU and fails
unless all four per-head MatMul and two Softmax operators receive complete
fake quantization. An import-only check is insufficient because older packages
silently leave attention outside QAT.

The public YOLO26n base weights are pinned to SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Supply the local file explicitly; the scripts do not silently change it.

## FP32 training

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/train_fp32.py \
  --manifest /data/drone/manifest.json \
  --base-weights /data/checkpoints/yolo26n.pt \
  --initial-weights /data/checkpoints/rgb-drone-p2.pt \
  --output-dir build/yolo26-drone-fp32 \
  --image-size 1600 \
  --epochs 100 --batch-size 16 --device 0
```

`--image-size` is a static deployment contract, not a training-only resize.
It must be divisible by 32 and is recorded in FP32 lineage, QAT checkpoints,
the ONNX tensor ABI, compiler calibration evidence, and the MPK audit. Larger
inputs such as 1600 are useful for single-digit-pixel targets, but they must
pass compiler and board memory/performance qualification before promotion.

FP32 and QAT use the same canonical, attention-based deployable graph. QAT
later folds BatchNorm and adds fake-quantization scaffolding in place without
changing the detector architecture. Compatible tensors are loaded by name and
shape from the initial checkpoint. Lineage records the canonical YAML digest,
initialization digest, and exact compatible/target tensor counts so partial
transfer is explicit.

`--initial-weights` optionally starts from an existing one-class, four-head
YOLO26-P2 checkpoint, such as the sequence-safe single-drone model. Omit it to
initialize from the pinned official YOLO26n base. The latter necessarily has
lower transfer coverage because it has three heads and a different class
count; either way, there is only one FP32 training stage and its output is
`build/yolo26-drone-fp32/fp32/weights/best.pt`.

## Target-exact QAT

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/train_qat.py \
  --manifest /data/drone/manifest.json \
  --fp32-checkpoint build/yolo26-drone-fp32/fp32/weights/best.pt \
  --fp32-lineage build/yolo26-drone-fp32/lineage.json \
  --output-dir build/yolo26-drone-qat \
  --image-size 1600 \
  --observer-warmup-images 1024 \
  --max-train-images 32768 \
  --epochs 10 --batch-size 1 --accumulation-steps 32 --device cuda
```

Batch 1 is mandatory. PT2E specializes YOLO26 reshapes to the example batch,
and the Modalix app is batch 1. Gradient accumulation supplies the effective
optimization batch without training a graph that cannot be deployed.
The FP32 lineage is mandatory and cryptographically binds the checkpoint,
manifest, every referenced image and label, architecture, and deployment
shape to QAT. QAT verifies the recorded canonical YAML digest before loading
the checkpoint. A smoke/ineligible FP32 source remains ineligible through
QAT, export, and compilation even if its numeric accuracy happens to pass
later gates.

All mature BatchNorm layers are folded while retaining both training heads.
Activations—including both operands and results of attention MatMul and both
Softmax boundaries—use signed per-tensor INT8 fake quantization. Convolution
weights use signed symmetric axis-0 per-channel INT8. The two two-head sites
lower to eight MatMul and four Softmax operators; QAT preparation refuses any
other count or incomplete fake-quant coverage.
Observers see at least 128 unique unaugmented images selected reproducibly
without replacement across the full training split, and the checkpoint
records every selected path and digest.
Observer updates are temporarily disabled for every held-out validation pass,
so validation images cannot influence quantization ranges, and are permanently
frozen after the first 20% of epochs by default, before the remaining low-rate
refinement. Set `--freeze-observers-fraction 0` to retain only the ranges from
the representative warm-up. Each fresh run records its fake-quantized accuracy
before the first optimizer step, separating initialization loss from training
regressions.

Accuracy evaluation retains candidates down to score `0.001`, bounds NMS to
the top 30,000 candidates, and reports at most 300 detections per image. This
keeps AP50 independent of the application's normal confidence cutoff while
bounding worst-case NMS cost.

For very large video corpora, `--max-train-images` bounds each QAT epoch while
keeping at least one temporally distributed sample from every training
sequence. Quotas are proportional to sequence length, sampled from uniform
temporal bins, and the exact selection digest and per-sequence counts are
stored in the checkpoint. Observer warm-up remains an independent sample from
the full training split; validation still runs on the complete held-out split.

A frozen production checkpoint can be refined at a lower learning rate without
recalibrating or exposing observers to validation data. `--epochs` is the
number of additional epochs for a resumed invocation, and the new checkpoint
records the immutable resume digest and starting metrics. The exact resume
checkpoint remains the best candidate unless a new frozen epoch improves the
AP50/tiny-recall ordering:

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/train_qat.py \
  --manifest /data/drone/manifest.json \
  --fp32-checkpoint build/yolo26-drone-fp32/fp32/weights/best.pt \
  --fp32-lineage build/yolo26-drone-fp32/lineage.json \
  --resume-checkpoint build/yolo26-drone-qat/checkpoint-best-frozen.pt \
  --output-dir build/yolo26-drone-qat-refined \
  --epochs 10 --learning-rate 2e-5 --trainable-scope head \
  --observer-warmup-images 128 --batch-size 1 \
  --accumulation-steps 32 --device cuda
```

`--trainable-scope head` freezes the quantized backbone and neck and updates
only the one-to-many and one-to-one YOLO26 detection-head parameters. This is
useful when full-network recovery leaves a sharp frozen-observer optimum; the
checkpoint records both the scope and exact number of trainable parameters.
Use `--trainable-scope classification` to freeze the box branches as well and
recover only one-to-many and one-to-one class-score heads when localization
metrics are already acceptable but INT8 confidence ranking remains degraded.
Add `--class-distillation-weight 1.0` to preserve those raw class logits with a
frozen copy of the same fused FP32 detector. This optional refinement teacher
does not change the architecture or the normal FP32-to-QAT path. Smooth-L1
distillation covers both the one-to-many loss branch and the one-to-one
deployment branch, while the normal ground-truth detection loss remains
active. The weight and measured distillation loss are retained in every
checkpoint history entry.

## Export gates

```bash
.venv-yolo26-qat/bin/python training/yolo26_tiny_drone/export.py \
  --checkpoint build/yolo26-drone-qat/checkpoint-best-frozen.pt \
  --fp32-checkpoint build/yolo26-drone-fp32/fp32/weights/best.pt \
  --output-dir build/yolo26-drone-source
```

Training writes `checkpoint-latest.pt` after every epoch for recovery and
retains `checkpoint-best-frozen.pt` using frozen-observer AP50 first and tiny
recall as the tie-breaker. Export the latter so a final-epoch regression cannot
silently replace a better publishable candidate.

Export fails when:

- the FP32 lineage or QAT checkpoint is ineligible for qualification (unless
  explicitly exporting a diagnostic smoke artifact, which remains ineligible);
- observer evidence is below 128 images or observers were not frozen;
- validation has fewer than 100 tiny instances, QAT AP50 is below 0.30, or
  QAT tiny-object recall is below 0.50;
- AP50 drops more than 2 points or tiny-object recall drops more than 5 points;
- the input or eight P2–P5 outputs differ from the checkpoint's recorded
  static deployment contract;
- BatchNorm or raw MatMul remains, the expected eight canonicalized Einsum/four
  Softmax operators are missing, or any attention input/output lacks signed
  INT8 QDQ;
- any convolution weight lacks signed symmetric per-channel INT8 QDQ.

## ModelSDK compile

Run the compiler in a ModelSDK 2.1 environment with `mla-masm`, `mla-msim`,
and CUDA `libdevice.10.bc` installed. Compilation independently selects a
seeded sample of at least 128 training images and reuses the
exact checkpoint-recorded square RGB letterbox preprocessing from training:

```bash
python training/yolo26_tiny_drone/compile.py \
  --export-dir build/yolo26-drone-source \
  --manifest /data/drone/manifest.json \
  --output-dir build/yolo26-drone-modalix \
  --calibration-images 128
```

The compiler refuses a manifest or referenced dataset content outside the
checkpoint lineage, or an unqualified export. `--allow-smoke` can inspect an
ineligible source but can never turn it into an eligible MPK. The report pins
every selected image, the assembled calibration tensor, source ONNX,
serialized INT8 graph, and MPK by SHA-256. Dataset contents are rehashed after
the calibration tensor is assembled and before compilation starts. It
also enables non-4D MLA tensors while preserving deployment batch one and
verifies that every source convolution becomes a per-channel INT8 MLA
convolution, all eight per-head Einsum operators become `SIMA_QUANTIZED` INT8
MLA BatchMatmul layers, all four Softmax layers are `SIMA_QUANTIZED` INT8 MLA
layers, there is one MLA segment with no A65 fallback or host shared object,
all eight raw heads use INT8 transport, and Neat's public boundaries remain
float32.

`--allow-smoke` exists only for structural development checks. Its report is
explicitly marked ineligible for qualification and must never be promoted.

After compilation, validate the qualified MPK on Modalix with the held-out
drone sequences and the Apps tracking evaluator. Promotion into `models/`
still requires immutable model/calibration sources plus those target accuracy
results; this staging recipe does not publish synthetic or unmeasured assets.
