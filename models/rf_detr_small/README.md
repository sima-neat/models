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

## Attention layout optimization

Graph preparation applies two guarded ONNX rewrites in `attention.py`, before
calibration:

- The nine backbone window blocks evaluate independent groups in head/window
  order instead of window/head order, restoring the original output order.
  Only four Transpose attributes change per block. The frontend can then remove
  redundant layout conversions. Global attention and its projections are unchanged.
- The three decoder self-attention blocks use NCHW Einsum, Split/Concat head
  merge, and a 1x1 Conv carrying the original output-projection weights/bias.
  Deformable attention is unchanged. This layout also benefits legacy attention
  and does not force FlashAttention.

The passes validate the pinned export's shapes, permutations, reduction axes,
scaling, private intermediates, and block counts. Unexpected patterns fail with
an error rather than silently changing semantics. Public partition interfaces,
TopK/Gather, and calibration selection remain unchanged; both partitions are
calibrated after surgery. No compiler selector or kernel changes are required.

On the pinned Small export, both optimized partitions passed ONNX checking and
float comparisons on 50 held-out COCO images: backbone outputs were bit-exact;
the decoder's maximum absolute difference was 1.25e-4. Three complete split
pipeline comparisons also passed. The float surgery check uses `rtol=2e-4`,
`atol=2e-4` to allow changed float32 accumulation in the equivalent decoder
projection, and rejects non-finite outputs or changed shapes.

Prior Nano experiments measured 10.62% fewer full-backbone simulator cycles for
the window rewrite and 1.625x full-decoder speedup for the NCHW rewrite. Those
measurements used the development compiler's MLA simulator at 13.63 DMA bytes
per cycle, excluding host orchestration and boundary EV casts. They are not
Small performance claims. Recompile and benchmark this recipe with its pinned compiler to obtain
Small results; the ONNX checks do not establish quantized accuracy or latency.

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
