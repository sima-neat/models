# ResNet-50

This entry pins the ONNX Community conversion of
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) at a full
Hugging Face repository commit and verifies the 102 MB ONNX file by SHA-256.

```bash
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
```

The `modalix_bf16` and `modalix_int8` variants use the repository-wide Model
Compiler version pinned in `deps/manifest.json`. `compile.py` loads the verified ONNX, applies the
variant's quantization scheme and direct-MLA tessellation, and emits one
directly consumable MPK archive per variant.

BF16 uses one zero-valued sample because the quantization entrypoint requires
an iterable. INT8 currently uses eight deterministic synthetic samples to
exercise compilation. Those samples are not representative calibration data,
so the INT8 MPK is a compile smoke artifact and is not accuracy-qualified. An
immutable, redistribution-approved ImageNet calibration set must be pinned
before the model can be promoted beyond experimental maturity.

The Vulcan runner recipe must execute after `activate-model-compiler`; the
workflow does this in the same interactive shell so compiler executables such
as `mla-masm` and their runtime libraries are available.

The preprocessing values are pinned from the upstream conversion's
`preprocessor_config.json`. No model-specific download script is needed.
