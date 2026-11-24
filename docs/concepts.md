# Concepts

This page defines the terms the other pages use, starting with the ones you meet first
when you reproduce or add a model. Look here when a page, a script, or a CI message uses a name
you do not know. Each example comes from
[`models/resnet_50`](../models/resnet_50/model.yaml).

Two tools appear in every definition. The Model Compiler turns an ONNX model into the
file a board runs. A Modalix DevKit is the board that runs the compiled model.

## Model

A model is one folder under `models/`, and the folder name is the model ID. Its
`model.yaml` declares the model, and its `scripts/` folder holds the code that converts,
compiles, and measures it.

Example: `models/resnet_50` is the model with the ID `resnet_50`. The Model Browser shows
it under its `display_name`, ResNet-50.

## Build profile

A build profile is one named way to compile a model. `model.yaml` lists the build
profiles under `build_profiles`, and each sets the parameters its components compile
with, such as the precision.

Example: `resnet_50` has three build profiles. `w8a8` stores weights and activations as
8-bit integers (INT8). `w8a16` keeps INT8 weights and stores activations in BF16, a 16-bit
float format. `w16a16` uses BF16 for both. The model page shows one tab per build
profile, such as **W8A8**.

## Build

A build is one build profile of one model. It compiles in one compile invocation, which
has the same name as the build profile.

Example: `resnet_50` has three builds, one per build profile. The scripts name the `w8a8`
build with `--build-profile w8a8 --compile-invocation w8a8`.

## Stage

A stage is one step a build passes through. There are three: `compile`, `performance`,
and `accuracy`. Every build compiles. The performance and accuracy stages run when
`model.yaml` declares them.

| Stage | Where it runs | Model script | What it produces |
| --- | --- | --- | --- |
| compile | Model Compiler | `scripts/compile.py` | one [MPK](#mpk) per [model component](#model-component), plus `fidelity.json` when `model.yaml` declares `numerical_fidelity`, a comparison of the quantized graph's outputs, run in the Model Compiler, with the original's |
| performance | Modalix DevKit | `scripts/performance.py` | a model report with the performance of the whole model |
| accuracy | Modalix DevKit | `scripts/accuracy.py` | a model report with the accuracy on each declared dataset |

A language model compiles with `scripts/compile.sh` instead, and its compile stage
produces a deployment folder instead of MPKs.

Example: `resnet_50` declares `performance` and `accuracy` in `model.yaml`, so each of its
builds passes through all three stages.

## Benchmark

The benchmarks are the performance and accuracy stages. Both run on a DevKit and measure
a compiled build.

Example: the benchmarks of the `resnet_50` `w8a8` build are a performance run and an
accuracy run on the dataset `imagenet-2012-val`.

## Model component

A model component is a separately compiled part of a model. A model with one component
names it `model`. A model with several components lists them in `model.yaml` in the order
they run.

Example: `resnet_50` has the one component `model`. By contrast, `rf_detr_nano` has two,
`backbone` and `decoder`.

## MPK

MPK is the file format of a compiled model component: a `.tar.gz` archive that a DevKit
loads and runs. The compile stage writes one MPK per model component.

Example: the `w8a8` build of `resnet_50` compiles to `resnet_50_w8a8_mpk.tar.gz`.

## Model package

A model package is what a build profile ships. It is one ZIP file with the MPK of each
model component and the supporting files an application needs, such as the license file
that `model.yaml` pins under `resources`. A language model's package is its deployment
folder instead. The download on a model page is the model package of the selected build
profile.

Example: the package of the `w8a8` build is `resnet_50_w8a8_default.zip`. It holds
`resnet_50_w8a8_mpk.tar.gz` and the license file `licenses/LICENSE`.

## Conversion

Conversion exports the official checkpoint of a model to ONNX, the format the compile
stage reads. The model's `scripts/convert.py` does it, started by
`scripts/run_model_conversion.sh`. A model needs no conversion when its `model.yaml` pins
an ONNX file.

Example: `resnet_50` has no conversion. Its `compilation.source` pins `onnx/model.onnx`
from a Hugging Face repository, with the file's SHA-256 and size.

## Calibration

Calibration runs sample inputs through the model so that the compiler can choose the
value ranges of an INT8 build. `model.yaml` pins each calibration dataset under
`calibrations`, and a build profile names the one it uses.

Example: the `w8a8` build profile of `resnet_50` sets
`calibration_dataset: imagenet-2012-cal` and `calibration_samples: 20`. The `w8a16` and
`w16a16` build profiles use no calibration dataset.

## Fingerprint

A fingerprint is a SHA-256 of every input that decides the result of a stage, such as the
source, the scripts, and the compiler version. CI reruns a stage only when one of its
inputs changed, which it detects from a changed fingerprint. A `sha256` field is
different: it is the SHA-256 of one file.

Example: a change to `models/resnet_50/scripts/performance.py` changes only the
performance fingerprint, so CI measures performance again without compiling. A change to
`compile.py` changes the compile fingerprint, so CI compiles again and then reruns both
benchmarks. [What a change reruns](adding-a-model.md#9-know-what-a-change-reruns) lists
the inputs of each fingerprint.

## Model report

A model report is the file `model-report.json` that a benchmark writes. A performance
report holds one number per metric under `metrics`. An accuracy report holds, under
`datasets`, the metrics measured on each dataset.
[`model-report.schema.json`](../schemas/model-report.schema.json) defines the format.

Example: the performance report of `resnet_50` holds `metrics.throughput_fps`. Its
accuracy report holds `top1_accuracy` and `top5_accuracy` for `imagenet-2012-val`.

## Metric

A metric is what a benchmark measures, with its unit and the direction that is better.
[`metrics.json`](../tools/model_report/metrics.json) names every metric a model report
may hold.

Example: `throughput_fps` counts inferences per second, and higher is better.
`top1_accuracy` is a ratio from 0 to 1, and higher is better.

## Measurement

A measurement is one measured value of a metric, for one build.

Example: `models/resnet_50/tests/tests.yaml` records 1508.0 as the expected
`throughput_fps` of the `w8a8` build. That number is a measurement of the metric
`throughput_fps`.

## Reference score

A reference score is an accuracy of the source model before compilation, which
`reference_metrics` in `model.yaml` records. The source model's authors published it, or
the model's `scripts/fp32_baseline_accuracy.py` measured it as an FP32 baseline.
Validation and the DevKit accuracy test compare each build profile with one reference
score per dataset: the FP32 baseline, else the published score for the evaluated split.
They fail a profile more than 8% below it
([comment out a variant far below its reference score](adding-a-model.md#comment-out-a-variant-far-below-its-reference-score)).

Example: `resnet_50` records a published top-1 accuracy of 0.80382 on the ImageNet
validation split. Its `w8a8` build expects 0.7754, which is 3.5% below that reference
score.

## On-demand build

An on-demand build belongs to a model that sets `build-on-demand: true` in `model.yaml`.
CI checks such a model on every push, but compiles and benchmarks it only when a
maintainer starts the build for chosen build profiles. Models that are expensive to
compile use this setting. [Promote a model](model-promotion.md) describes how such a
build becomes available for download.

Example: `resnet_50` does not set the field, and its default is `false`. CI therefore
compiles and benchmarks its builds after each change that affects them.
