# TorchVision SSDLite-MobileNetV3 QAT

This directory produces the INT8 source for the TorchVision
`ssdlite320_mobilenet_v3_large` replacement. It intentionally separates QAT
from ModelSDK compilation:

1. Python 3.10, PyTorch 2.3.1, TorchVision 0.18.1, and the SDK 2.0 SiMa QAT
   package train and export an opset-17 QDQ model.
2. The Model Compiler imports the immutable source bundle and compiles one
   build per precision.

The `modalix_bf16` build uses the FP32 source in the bundle, and `modalix_int8`
uses the QDQ source. PTQ-only SmoothQuant, bias correction, channel
equalization, and negligible-weight surgery must not be applied to the QAT
source.

## Environment

The bootstrap downloads a managed CPython 3.10.18 interpreter. Creating a
venv from the ModelSDK 2.1 Python 3.12 interpreter is not sufficient because
PT2E QAT is broken on that interpreter.

Use either an SDK 2.0 QAT wheel:

```bash
SIMA_QAT_WHEEL=/path/to/swml_qat.whl \
  models/ssd_mobilenet_v3/scripts/qat/bootstrap.sh
```

or an existing SDK installation without copying its proprietary source into
the repository:

```bash
SIMA_QAT_SITE_PACKAGES=/path/to/sdk/site-packages \
  models/ssd_mobilenet_v3/scripts/qat/bootstrap.sh
```

The bootstrap also requires `swml-qat==1.0.0.dev0+develop.20`. Direct and
transitive public dependencies are pinned in `requirements.in` and
`requirements.lock`. Regenerate the lock deliberately with the same Python
minor version:

```bash
uv pip compile models/ssd_mobilenet_v3/scripts/qat/requirements.in \
  --python-version 3.10 \
  --output-file models/ssd_mobilenet_v3/scripts/qat/requirements.lock
```

## Training

Place COCO in the conventional layout:

```text
COCO_ROOT/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  train2017/*.jpg
  val2017/*.jpg
```

Run the qualification schedule in full FP32 computation. Fake quantization
simulates per-tensor INT8 activations and per-output-channel INT8 weights:

```bash
.venv-qat/bin/python models/ssd_mobilenet_v3/scripts/qat/train.py \
  --coco-root "$COCO_ROOT" \
  --output-dir build/ssd-mobilenet-v3-qat \
  --epochs 1 \
  --batch-size 16 \
  --accumulation-steps 2 \
  --observer-warmup-images 128
```

Before observers are inserted, all 70 mature pretrained BatchNorm layers are
folded into their adjacent convolutions. Fake quantization therefore sees and
optimizes the exact deployed weights, including channels whose folded weights
are much smaller than their pre-fold values. Multi-epoch schedules freeze
observers for their final portion; the qualified v1 schedule froze them after
the initial full-data pass and kept them frozen throughout the low-rate
refinement stage.

If a completed stage needs low-rate refinement, continue from its frozen
observers with an explicit optimizer restart. The original checkpoint digest
and new schedule are recorded in the exported lineage:

```bash
.venv-qat/bin/python models/ssd_mobilenet_v3/scripts/qat/train.py \
  --coco-root "$COCO_ROOT" \
  --output-dir build/ssd-mobilenet-v3-qat-refined \
  --resume build/ssd-mobilenet-v3-qat/checkpoint-latest.pt \
  --restart-optimizer \
  --skip-baseline-eval \
  --epochs 2 \
  --learning-rate 1e-4 \
  --observer-warmup-images 128
```

Ordinary `--resume` preserves the original cosine schedule exactly;
`--restart-optimizer` must extend the completed epoch count and never
re-enables frozen observers.

`--smoke` unlocks partial-data and short observer runs for diagnostics. Smoke
checkpoints are marked ineligible and the exporter refuses to publish them
unless `--allow-smoke` is explicitly supplied.

## Export and qualification

```bash
.venv-qat/bin/python models/ssd_mobilenet_v3/scripts/qat/export.py \
  --checkpoint build/ssd-mobilenet-v3-qat/checkpoint-latest.pt \
  --output-dir build/ssd-mobilenet-v3-source
```

The exporter rejects incomplete runs, missing frozen observers, absent COCO
evidence, or a default accuracy loss above 1.5 AP points. It then proves:

- the exact `1x3x320x320` input and 12-output ABI;
- all six levels use six anchors, for 3,234 anchors total;
- exactly 98 convolutions and no runtime BatchNorm;
- every convolution weight has symmetric axis-0 per-channel QDQ;
- activation QDQ is per-tensor;
- FP32 and QAT graphs use opset 17 and the pinned TorchVision weights.

The deterministic bundle contains `manifest.json`, the FP32 ONNX used for
BF16, and the QAT QDQ ONNX used for INT8. Its SHA-256 and size are printed for
promotion into `model.yaml` only after full COCO qualification.

The public tensor is RGB after resize, rescale to `[0, 1]`, and ImageNet
normalization with mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`. Normalization stays at the app boundary so the
deployable graph remains exactly the upstream 98-convolution detector.
