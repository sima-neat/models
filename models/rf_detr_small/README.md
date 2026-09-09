# RF-DETR Small

This recipe builds the 512×512 RF-DETR Small detector from the official Small
checkpoint with the 1.9.4 exporter. Model Registry verifies both files before invoking
`compile.py`. The recipe creates the full ONNX model and both partitions in a
temporary directory.

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

## Build and validation

`compile.py` is the single orchestration entry point. It exports and partitions
the model, calibrates the INT8 backbone with COCO images, executes that
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
  --model <verified-checkpoint> \
  --dataset <resolved-coco-archive> \
  --correctness-dataset <resolved-coco-archive> \
  --output-dir <new-output-directory> \
  --build-profile modalix_int8_bf16 \
  --compile-invocation rf_detr
```

Generated ONNX files and intermediate tensors remain temporary. MPKs and
reports are build output and must remain outside Git.

## Upstream

- [RF-DETR repository](https://github.com/roboflow/rf-detr/tree/9b009fa928d6218320439803d1da01869a85c072)
- [RF-DETR Small model card](https://huggingface.co/Roboflow/rf-detr-small)
- [RF-DETR 1.9.4 exporter](https://pypi.org/project/rfdetr/1.9.4/)

RF-DETR is distributed under Apache-2.0.
