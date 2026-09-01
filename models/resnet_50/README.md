# ResNet-50

This entry pins the ONNX Community conversion of
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) at a full
Hugging Face repository commit and verifies the 102 MB ONNX file by SHA-256.

```bash
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
dataset-registry resolve imagenet-2012 --variant mini
```

The `modalix_bf16` and `modalix_int8` variants use the repository-wide Model
Compiler version pinned in `deps/manifest.json`. `compile.py` loads the verified ONNX, applies the
variant's quantization scheme and direct-MLA tessellation, and emits one
directly consumable MPK archive per variant.

BF16 uses one zero-valued sample because the quantization entrypoint requires
an iterable. INT8 uses all 500 calibration images in the registered
`imagenet-2012/mini` Artifactory archive. The same archive contains a balanced
5,000-image validation proxy (five images for every ImageNet-1k class). The
model-local recipe only compiles; shared `model-registry evaluate` tooling reads
`tests/correctness.yaml` to run deterministic accuracy measurement for INT8.
BF16 remains a compiler canary and does not run accuracy measurement.

```bash
dataset="$(dataset-registry resolve imagenet-2012 --variant mini | python -c 'import json,sys; print(json.load(sys.stdin)["path"])')"
python models/resnet_50/compile.py \
  --model /path/to/model.onnx \
  --dataset "$dataset" \
  --output-dir /tmp/resnet50-int8 \
  --variant modalix_int8
```

Run the regular 500-image profile after compilation:

```bash
model-registry evaluate models/resnet_50 \
  --variant modalix_int8 \
  --reference-model /path/to/model.onnx \
  --candidate-model /tmp/resnet50-int8/resnet_50_modalix_int8.sima \
  --dataset "$dataset" \
  --profile regular \
  --workers 4 \
  --output /tmp/resnet50-int8/accuracy.json
```

Use `--profile qualification` for all 5,000 mini validation images. Reports
include the profile/status, evaluated and available sample counts, deterministic
selection digest, worker count, artifact and dataset hashes, top-1/top-5
accuracy, agreement, and FP32 deltas. The current dataset is explicitly marked
provisional because of its known calibration/validation overlap.

Vulcan builds use 500 samples by default. A manual workflow dispatch can set
`accuracy_samples` to `5000` for an on-demand full-mini qualification run.

The Vulcan runner recipe must execute after `activate-model-compiler`; the
workflow does this in the same interactive shell so compiler executables such
as `mla-masm` and their runtime libraries are available.

The preprocessing values are pinned from the upstream conversion's
`preprocessor_config.json`. No model-specific download script is needed.
