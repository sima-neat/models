# Quantize and Compile Models for SiMa Modalix

`scripts/quantize_compile.py` runs the standard local workflow. AFE selects the
Gen2 Modalix target and current layout, tessellation, any-shape-on-MLA, and
verification defaults. Gen1 is not supported.

The helper supports FP32 4D image inputs. For non-image or mixed-dtype models,
use the import API with matching input names, shapes, dtypes, and calibration
layout instead of this command.

## Commands

ONNX input names are inferred from the model. Do not pass `--input_names` for
ONNX. Runtime inputs use the order stored in the graph, excluding initializers.
Static shapes are inferred in that same order:

```bash
python3 scripts/quantize_compile.py \
  --model_path /abs/path/model.onnx \
  --model_format onnx \
  --build_dir /abs/path/build
```

For dynamic ONNX, pass one `--input_shapes` value per runtime input in graph
order. Without them, the script stops and names the input with a dynamic shape.

PyTorch requires explicit names, shapes, and NCHW layout:

```bash
python3 scripts/quantize_compile.py \
  --model_path /abs/path/model.pt \
  --model_format pytorch \
  --model_layout NCHW \
  --input_names input \
  --input_shapes 1,3,224,224 \
  --build_dir /abs/path/build
```

Pass an absolute `--build_dir`. Its default, `./build`, resolves inside this
skill's folder when you run the command from there.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `--model_layout` | `NCHW` | Input tensor layout, `NCHW` or `NHWC`. |
| `--no-simplify` | simplification on | Skips ONNX preparation. Preparation runs `onnxsim` with the fixed input shapes, runs shape inference, sets IR version 8, and saves `<model>_prepared.onnx` beside the source. If preparation fails, the script logs the error and continues with the original model. |
| `--real_data` | off | Calibrates on images. Without it, the script calibrates on random placeholders. |
| `--dataset_images` | `./calib_images` | Folder of `.jpg`, `.jpeg`, `.png`, and `.bmp` calibration images. Every input receives the same images, resized to its own shape with bilinear resampling, which matches the deployed image preprocessor. |
| `--num_calib_samples` | `50` | The most images calibration reads, taken in sorted file-name order. |
| `--calib_method` | `mse` | Calibration method, such as `mse` or `entropy`. |
| `--mean`, `--std` | `0 0 0`, `1 1 1` | RGB normalization applied after pixels are scaled to [0, 1]. |
| `--requant_mode` | `sima` | INT8 requantization mode, `sima` or `tflite`. |
| `--bf16-weights` | off | BF16 weights. BF16 weights also select BF16 activations. |
| `--bf16-activations` | off | BF16 activations with INT8 weights. |
| `--analyse-error` | off | Per-layer quantization error analysis (mean absolute error on the calibration data). |
| `--verify` | off | Runs the quantized and the floating-point model on one random input and logs each output's maximum and mean absolute difference. |
| `--no-compile` | compilation on | Quantizes and saves the model without compiling it. |
| `--batch_size` | `1` | Compilation batch size. |

INT8 quantization defaults to `--requant_mode sima` for faster quantization.
If the resulting accuracy is insufficient, retry with `--requant_mode tflite`.
It is slower but may improve accuracy.

## Outputs

The script writes into `<build_dir>/<model_basename>/`, named after the source
file. It saves the quantized model there, and a compile adds the compiled
archive and `quantization_manifest.json`, which records the effective
`activation_precision`, `weight_precision`, and `device`.

## QAT QDQ exports

For FP32 4D image QAT exports, use the default ONNX command with INT8 settings.
For non-image or mixed-dtype exports, use the import API as noted above.
Use the normal `load_model()` path and preserve QDQ hints. Do not set internal
`is_quantized=True`.
Omit `--real_data` to supply random calibration placeholders: AFE retains
QDQ scales and zero points instead of re-estimating those ranges. Inputs must
match the model's interface and valid domain. Regions quantized without QDQ
hints need representative calibration data. Accuracy checks need real data.

The [Model Compiler quantization guide](https://github.com/sima-neat/model-compiler/blob/661044476aafd629f3aa7cacb53ba2d4031e29b1/docs/guides/quantization.md)
describes QAT ONNX import and compilation in its QAT exports section.

## Memory errors

If compilation is killed or runs out of host memory, lower MLA simulator
parallelism and retry the same command:

```bash
export SIMA_MLA_SIM_PARALLEL=<lower-thread-count>
```

Lower values reduce peak memory but can increase compilation time. Do not
prescribe `1` by default.
