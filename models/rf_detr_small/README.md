# RF-DETR Small

This recipe compiles the 512×512 RF-DETR Small detector from a verified ONNX
artifact. A separate manual conversion uses the official Roboflow Small
checkpoint and the 1.9.4 exporter. Model Registry downloads the pinned ONNX
before invoking `compile.py`; compilation reuses that artifact.

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

## Conversion

`convert.yaml` pins the Roboflow checkpoint, validation image and isolated
Python dependency lock. `convert.py` exports the full graph and checks its
opset-17 interface and PyTorch/ONNX agreement on two seeded synthetic images
and a real image. Failed validation prevents publication.

Validate the declaration locally, then run conversion with a compatible
Python 3.12 installation. The checked-in dependency lock targets Linux ARM64,
matching the conversion runner.

```bash
model-registry validate-conversion models/rf_detr_small --json
model-registry convert models/rf_detr_small \
  --python python3.12 \
  --work-dir <new-conversion-directory> \
  --cache-dir <conversion-cache-directory> --json
```

To publish, manually run the **Vulcan Model Convert** workflow with model ID
`rf_detr_small` and configuration `convert.yaml`. It runs the same conversion,
publishes an immutable ONNX bundle and returns a source declaration. Pin that
source in `model.yaml` after validating surgery and compiler compatibility.
Normal compilation does not install exporter dependencies or rerun conversion.

`convert-hf.yaml` selects the experimental Hugging Face SafeTensors route in
the same script. It uses Transformers and fixed-shape positional encoding on
an export copy, preserving the original model for numerical comparison. This
route needs its own numerical and surgery qualification before its output can
replace the Roboflow graph. Use `--config convert-hf.yaml` for local testing.

## Build and validation

`compile.py` partitions the downloaded ONNX, calibrates the INT8 backbone with
COCO images, executes the quantized backbone and TopK/Gather to create
correlated transformer inputs,
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

The published full ONNX is retained for reuse. Surgery creates temporary
partition graphs and intermediate tensors. Generated models, MPKs and reports
must remain outside Git.

## Upstream

- [RF-DETR repository](https://github.com/roboflow/rf-detr/tree/9b009fa928d6218320439803d1da01869a85c072)
- [RF-DETR Small model card](https://huggingface.co/Roboflow/rf-detr-small)
- [RF-DETR 1.9.4 exporter](https://pypi.org/project/rfdetr/1.9.4/)

RF-DETR is distributed under Apache-2.0.
