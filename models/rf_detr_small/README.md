# RF-DETR Small

This recipe compiles the 512×512 RF-DETR Small detector from a pinned ONNX
file on Vulcan. A separate manual conversion uses the official Small checkpoint
and the 1.9.4 exporter. Normal compilation downloads the ONNX without installing
exporter dependencies.

## Components

The backbone consumes a normalized RGB image and emits the final feature map,
1024 proposal scores, and 1024 decoded proposal boxes. On A65, select the 300
highest scores and gather the matching boxes. The transformer consumes those
300 proposals and the backbone feature map, then emits 300 boxes and 91 class
scores for each box.

```text
image -> backbone.mpk -> scores + boxes -> A65 TopK/Gather -> transformer.mpk
                         feature map -----------------------> transformer.mpk
```

The package exposes `proposals` as an input because its composition connects
only the two MPKs. The application must run TopK/Gather on the backbone outputs
before invoking the transformer. `calibration.py` contains the deterministic
reference operation used during compilation and fidelity validation.

`surgery.py` also rewrites the attention layout before calibration: the nine
backbone window blocks run in head/window order, and the three decoder
self-attention blocks run in NCHW with a 1x1 Conv projection. Both rewrites
leave the interfaces unchanged and fail on an unexpected export.

## Manual conversion

`convert.yaml` pins the checkpoint; `requirements-convert.txt` pins export
dependencies. With Python 3.12 on Linux ARM64, run locally:

```bash
python3.12 -E -s -m venv /tmp/rfdetr-export-venv
/tmp/rfdetr-export-venv/bin/python -E -s -m pip --isolated install . -r models/rf_detr_small/requirements-convert.txt
/tmp/rfdetr-export-venv/bin/python -E -s models/rf_detr_small/convert.py --output-dir /tmp/rfdetr-export
```

To publish, manually run **Vulcan Model Convert** with model `rf_detr_small`.
The workflow creates its export venv under `/mnt/nvme`, validates ONNX and checks
one synthetic input against PyTorch, then uploads one `model.onnx`. Copy the
printed source pin into `model.yaml` after surgery and compiler validation.
This smoke check does not measure model accuracy.

## Build and validation

`compile.py` partitions the downloaded ONNX, calibrates the INT8 backbone with
COCO images, executes that
quantized backbone and TopK/Gather to create correlated transformer inputs,
calibrates the BF16 transformer, and writes:

```text
<output>/
  backbone/*_mpk.tar.gz
  transformer/*_mpk.tar.gz
  transformer/fidelity.json
  compile-outputs.json
```

The fidelity report compares the quantized AFE backbone, CPU bridge, and
quantized AFE transformer with the original exported ONNX model. These metrics
are diagnostic, not detection accuracy: query reordering can reduce numerical
similarity even for equivalent detections. They do not gate publication.
The shared benchmark script calls `model.benchmark(...)` separately for each
MPK, using three trials of 1,000 samples. CI runs the backbone and transformer
sequentially on one Modalix board and publishes each result.

The recipe sorts the COCO 2017 mini image names and shuffles them with the
declared seed. It uses the first 50 images for calibration and the next 50 for
fidelity. Labels are not needed for this tensor comparison. Resolve the dataset
with Dataset Registry, then run the grouped recipe in the pinned Model Compiler
environment:

```bash
dataset-registry resolve --repository <dataset-repository> \
  --variant mini coco-2017

python models/rf_detr_small/compile.py \
  --model <verified-model.onnx> \
  --dataset <resolved-coco-archive> \
  --correctness-dataset <resolved-coco-archive> \
  --output-dir <new-output-directory> \
  --build-profile modalix_int8_bf16 \
  --compile-invocation rf_detr
```

The published ONNX is retained for reuse. Partition graphs and intermediate
tensors remain temporary. Generated models and reports must remain outside Git.

## Upstream

- [RF-DETR repository](https://github.com/roboflow/rf-detr/tree/9b009fa928d6218320439803d1da01869a85c072)
- [RF-DETR Small model card](https://huggingface.co/Roboflow/rf-detr-small)
- [RF-DETR 1.9.4 exporter](https://pypi.org/project/rfdetr/1.9.4/)

RF-DETR is distributed under Apache-2.0.
