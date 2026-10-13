# RF-DETR Nano

![RF-DETR Nano model hero](hero.png)

## Overview

RF-DETR Nano is a COCO object detector that accepts a normalized 384×384 RGB
image and produces 300 boxes with 91 class logits. The Modalix package splits
inference into independently deployable backbone and transformer components.
Applications must select the 300 highest-scoring backbone proposals and gather
their boxes on the A65 before invoking the transformer. Detection accuracy is
measured as COCO mAP on a Modalix DevKit over the full COCO 2017 validation
split.

## Usage

Download the complete published variant from the `develop` catalog:

```bash
sima-cli models download \
  --id rf_detr_nano \
  --variant modalix_int8_bf16 \
  --branch develop
```

The package contains both `backbone` and `transformer`. An application must run
both components with the CPU proposal-selection bridge described below; either
MPK alone is not a complete detector.

## Model interface

The backbone consumes a normalized NCHW FP32 RGB tensor with shape
`1×3×384×384`. It emits the final feature map, 576 proposal scores, and 576
decoded proposal boxes. On the A65, select the 300 highest scores and gather the
matching boxes. The transformer consumes those proposals and the backbone
feature map, then emits 300 boxes and 91 class logits for each box.

```text
image -> backbone.mpk -> scores + boxes -> A65 TopK/Gather -> transformer.mpk
                         feature map -----------------------> transformer.mpk
```

To decode detections on the A65, apply a sigmoid to `labels` and rank all
300×91 query-class scores. Each `dets` row is a normalized (cx, cy, w, h) box;
multiply it by the original image width and height, because preprocessing
resizes without preserving the aspect ratio. The class index is the COCO
category ID; 11 of the 91 slots are unused.

The package exposes `proposals` as an input because package composition connects
only the two MPKs. `calibration.py` contains the deterministic reference bridge
used during compilation and fidelity validation.

## Available builds

The registry declares one `modalix_int8_bf16` build profile. It compiles the
backbone with INT8 quantization and regular bias correction, and the
transformer with BF16 quantization. Both use histogram MSE calibration, set by
`calibration_bins` in `model.yaml`: 1,024 bins for the backbone and 2,048 for
the transformer. In one build of each, 1,024 backbone bins measured 0.7 COCO
val2017 mAP above 2,048. The grouped `rf_detr` invocation emits separate
`backbone` and `transformer` MPK archives plus their shared compile manifest.

## Quality and performance

Compilation produces a fidelity report comparing the quantized AFE backbone,
the CPU proposal bridge, and the quantized AFE transformer with the original
exported ONNX model. These metrics are diagnostic rather than detection
accuracy: query reordering can reduce numerical similarity for equivalent
detections, so fidelity does not gate publication.

The benchmark script measures each MPK separately with three trials of 1,000
samples. CI runs the backbone and transformer sequentially on one Modalix board
and publishes one benchmark result for each component.

On the same board and MPKs, `scripts/accuracy.py` then measures the 12 COCO
bounding-box AP and AR metrics on all 5,000 val2017 images. The `coco_accuracy`
result is published as an accuracy run under the transformer's compile run, and
the model browser shows it with the build beside Roboflow's
[published scores](https://github.com/roboflow/rf-detr/blob/56cdbbf57fc51c090f3aca398802d2cbc6703a05/README.md) for the source model: mAP 50–95 0.484 and AP50 0.676.

The script resizes each image to 384×384 with bilinear sampling and applies
ImageNet normalization, runs the backbone, the CPU proposal bridge, and the
transformer, and keeps the 300 highest-scoring box and class pairs with a score
of at least 0.001. Boxes are clipped to the image, as in RF-DETR's current
postprocessor.

## Source and license

The source is the RF-DETR Nano checkpoint on Hugging Face, exported to ONNX
with `transformers` 5.16.1. Normal compilation consumes the pinned ONNX and does
not install the exporter dependencies.

- [RF-DETR source repository](https://github.com/roboflow/rf-detr/tree/9b009fa928d6218320439803d1da01869a85c072), for background; conversion uses the Hugging Face checkpoint
- [RF-DETR Nano model card](https://huggingface.co/Roboflow/rf-detr-nano)
- [Pinned checkpoint revision](https://huggingface.co/Roboflow/rf-detr-nano/tree/9f201577d9415f0378084748d27e68586e6b3600)

RF-DETR is distributed under Apache-2.0.

## Reproducing the artifact

### Conversion

The `conversion.default` entry in `model.yaml` pins the three Hugging Face
checkpoint files, and `scripts/requirements-convert.txt` pins the export
dependencies. `convert.py` exports through `transformers` and folds constants
with onnxsim, so surgery starts from a static graph. With Python 3.12 on Linux ARM64, run:

```bash
python3.12 -E -s -m venv /tmp/rfdetr-export-venv
/tmp/rfdetr-export-venv/bin/python -E -s -m pip --isolated install . -r models/rf_detr_nano/scripts/requirements-convert.txt
/tmp/rfdetr-export-venv/bin/python -E -s models/rf_detr_nano/scripts/convert.py --output-dir /tmp/rfdetr-export
```

To publish a conversion, manually run **Vulcan Model Convert** with model
`rf_detr_nano`. The workflow validates ONNX, compares a synthetic input with
PyTorch, and uploads `model.onnx`. The comparison first checks the 576
proposal scores; when two tied scores swap order, it tries the next seed,
because a swap pairs decoder queries with different boxes. Copy the printed source pin into `model.yaml`
only after surgery and compiler validation. The smoke comparison does not
measure model accuracy.

### Calibration and compilation

`surgery.py` removes export no-ops before calibration: constant all-false
masks, a scatter that overwrites every proposal, and decoder outputs kept only
for training losses. It then rewrites the attention layout: the nine backbone
window blocks run in head/window order, and the two decoder self-attention
blocks run in NCHW with a 1x1 Conv projection. In all twelve backbone blocks,
it folds the 1/8 attention scale into the Q projection weights instead of
multiplying every attention score; a power-of-two scale keeps the float graph
exact.

Two further rewrites remove work that the compiler would otherwise schedule:

- Each GELU folds its 1/√2 and 1/2 constants into the fc1 and fc2 weights,
  removing a Div and a Mul on every MLP hidden activation.
- The decoder's interleaved sine/cosine query embedding becomes one MatMul, a
  phase Add and one Sin, using cos(a) = sin(a + π/2).

These rewrites change only float32 rounding. The rewrites leave the interfaces
unchanged and fail on an unexpected export.

The recipe sorts the image names of the COCO 2017 `cal` dataset, 1,000
train2017 images that no accuracy run scores, and shuffles them with the
declared seed. It uses the first 50 images for calibration and the next 50 for
fidelity; labels are not needed for the tensor comparison. Resolve the dataset,
then run the grouped recipe in the pinned Model Compiler environment:

```bash
output_dir=/tmp/rf-detr-nano

source_json="$(model-registry fetch models/rf_detr_nano \
  --component backbone --json)"
verified_model="$(jq -er .path <<<"$source_json")"

calibration_json="$(model-registry fetch-calibration models/rf_detr_nano \
  --build-profile modalix_int8_bf16 --component backbone --json)"
resolved_coco_archive="$(jq -er 'select(.kind == "dataset") | .path' \
  <<<"$calibration_json")"
test -f "$verified_model"
test -e "$resolved_coco_archive"

python models/rf_detr_nano/scripts/compile.py \
  --model "$verified_model" \
  --dataset "$resolved_coco_archive" \
  --fidelity-dataset "$resolved_coco_archive" \
  --output-dir "$output_dir" \
  --build-profile modalix_int8_bf16 \
  --compile-invocation rf_detr
```

### Outputs and validation

The recipe partitions the downloaded ONNX, calibrates the backbone, executes
the quantized backbone and CPU bridge to create correlated transformer inputs,
calibrates the transformer, and writes:

```text
<output>/
  backbone/*_mpk.tar.gz
  transformer/*_mpk.tar.gz
  transformer/fidelity.json
  compile-outputs.json
```

The published ONNX is retained for reuse. Partition graphs, intermediate
tensors, generated models, and reports must remain outside Git.

### DevKit accuracy

The `coco_accuracy` accuracy entry declares the COCO 2017 `val` dataset
archive, the published metrics, and Roboflow's reference scores. On a Modalix
DevKit, run the accuracy stage on the output of
`scripts/run_model_compilation.sh` above, setting `REGISTRY_PYTHON` to a Python
that has the full Model Registry tooling and Dataset Registry:

```bash
scripts/run_model_accuracy.sh models/rf_detr_nano modalix_int8_bf16 \
  /tmp/rf-detr /media/nvme/neat-tests/rf-detr-accuracy
```

The script installs `scripts/requirements-accuracy.txt` beside PyNeat's
environment, extracts the archive under `/media/nvme/neat-tests/`, and removes
both when it exits. See
[Accuracy on the DevKit](../../docs/local-accuracy-development.md#accuracy-on-the-devkit)
for the result format and publication.
