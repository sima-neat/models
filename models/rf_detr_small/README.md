# RF-DETR Small 512x512

## Overview

RF-DETR Small is a COCO object detector that accepts a normalized 512×512 RGB
image and produces 300 boxes with 91 class scores. The Modalix package splits
inference into independently deployable backbone and transformer components.
Applications must select the 300 highest-scoring backbone proposals and gather
their boxes on the A65 before invoking the transformer. Published fidelity
measurements diagnose the compiled tensor pipeline but do not constitute
detection-accuracy qualification.

## Usage

Download the complete published variant from the `develop` catalog:

```bash
sima-cli models download \
  --id rf_detr_small \
  --variant modalix_int8_bf16 \
  --branch develop
```

The package contains both `backbone` and `transformer`. An application must run
both components with the CPU proposal-selection bridge described below; either
MPK alone is not a complete detector.

## Model interface

The backbone consumes a normalized NCHW FP32 RGB tensor with shape
`1×3×512×512`. It emits the final feature map, 1,024 proposal scores, and 1,024
decoded proposal boxes. On the A65, select the 300 highest scores and gather the
matching boxes. The transformer consumes those proposals and the backbone
feature map, then emits 300 boxes and 91 class scores for each box.

```text
image -> backbone.mpk -> scores + boxes -> A65 TopK/Gather -> transformer.mpk
                         feature map -----------------------> transformer.mpk
```

The package exposes `proposals` as an input because package composition connects
only the two MPKs. `calibration.py` contains the deterministic reference bridge
used during compilation and fidelity validation.

## Available builds

The registry declares one `modalix_int8_bf16` build profile. It compiles the
backbone with INT8 quantization and the transformer with BF16 quantization. The
grouped `rf_detr` invocation emits separate `backbone` and `transformer` MPK
archives plus their shared compile manifest.

## Quality and performance

Compilation produces a fidelity report comparing the quantized AFE backbone,
the CPU proposal bridge, and the quantized AFE transformer with the original
exported ONNX model. These metrics are diagnostic rather than detection
accuracy: query reordering can reduce numerical similarity for equivalent
detections, so fidelity does not gate publication.

The benchmark script measures each MPK separately with three trials of 1,000
samples. CI runs the backbone and transformer sequentially on one Modalix board
and publishes one benchmark result for each component.

## Source and license

The source is derived from the official RF-DETR Small checkpoint and the 1.9.4
exporter. Normal compilation consumes the pinned ONNX and does not install the
exporter dependencies.

- [RF-DETR repository](https://github.com/roboflow/rf-detr/tree/9b009fa928d6218320439803d1da01869a85c072)
- [RF-DETR Small model card](https://huggingface.co/Roboflow/rf-detr-small)
- [RF-DETR 1.9.4 exporter](https://pypi.org/project/rfdetr/1.9.4/)

RF-DETR is distributed under Apache-2.0.

## Reproducing the artifact

### Conversion

`convert.yaml` pins the checkpoint, and `requirements-convert.txt` pins the
export dependencies. With Python 3.12 on Linux ARM64, run:

```bash
python3.12 -E -s -m venv /tmp/rfdetr-export-venv
/tmp/rfdetr-export-venv/bin/python -E -s -m pip --isolated install . -r models/rf_detr_small/requirements-convert.txt
/tmp/rfdetr-export-venv/bin/python -E -s models/rf_detr_small/convert.py --output-dir /tmp/rfdetr-export
```

To publish a conversion, manually run **Vulcan Model Convert** with model
`rf_detr_small`. The workflow validates ONNX, compares one synthetic input with
PyTorch, and uploads `model.onnx`. Copy the printed source pin into `model.yaml`
only after surgery and compiler validation. The smoke comparison does not
measure model accuracy.

### Calibration and compilation

The recipe sorts the COCO 2017 mini image names and shuffles them with the
declared seed. It uses the first 50 images for calibration and the next 50 for
fidelity; labels are not needed for the tensor comparison. Resolve the dataset,
then run the grouped recipe in the pinned Model Compiler environment:

```bash
output_dir=/tmp/rf-detr-small

source_json="$(model-registry fetch models/rf_detr_small \
  --component backbone --json)"
verified_model="$(jq -er .path <<<"$source_json")"

calibration_json="$(model-registry fetch-calibration models/rf_detr_small \
  --build-profile modalix_int8_bf16 --component backbone --json)"
resolved_coco_archive="$(jq -er 'select(.kind == "dataset") | .path' \
  <<<"$calibration_json")"
test -f "$verified_model"
test -e "$resolved_coco_archive"

python models/rf_detr_small/compile.py \
  --model "$verified_model" \
  --dataset "$resolved_coco_archive" \
  --correctness-dataset "$resolved_coco_archive" \
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
