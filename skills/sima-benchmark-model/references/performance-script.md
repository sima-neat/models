# The performance script

A performance script measures a build's whole model on the DevKit: every
compiled MPK and the CPU work between them. It loads each MPK with
`pyneat.Model`, and a single-component script calls Neat Core's benchmark API.
Core owns warm-up, latency and throughput measurement, board telemetry, and
runner lifetime. The script hands Core's results to
`model_report.from_performance()`, which writes the report.

The script needs PyNeat, and `tools/model_report` and `tools/model_inputs` on
`PYTHONPATH`. It needs no registry installation or YAML configuration.
`model_report` writes the report, and `model_inputs.frame` reads an image and
letterboxes it into the 720p benchmark frame. Every
published model's script is in its folder in the Models repository, such as
[ResNet-50's](https://github.com/sima-neat/models/blob/main/models/resnet_50/scripts/performance.py),
and [DevKit performance](https://github.com/sima-neat/models/blob/main/docs/benchmarking.md)
holds the full rules.

## Arguments

| Argument | Value |
| --- | --- |
| `--mpk` | the MPK, or `<component>=<path>` once per component for a multi-component model, in `model.yaml` order |
| `--dataset <source id>=<directory>` | the model's accuracy dataset, extracted, as `accuracy.py` takes it. A model without an accuracy dataset takes none. |
| `--resources <directory>` | the files a model declares as `resources`, in the layout its package installs. A model whose only resource is its license takes the argument and reads nothing from it. |
| `--output <directory>` | where the report goes |

## Single-component shape

This is ResNet-50's script, shortened. It measures one ImageNet validation
image in a 720p frame:

```python
import argparse, random
from pathlib import Path

import model_report
import pyneat
from model_inputs.frame import FRAME_HEIGHT, FRAME_WIDTH, letterbox_to_720p, read_rgb


parser = argparse.ArgumentParser()
parser.add_argument("--mpk", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--resources")
args = parser.parse_args()
dataset_id, dataset_dir = args.dataset.split("=", 1)
# One image, chosen with a fixed seed so every run measures the same input.
image = random.Random(0).choice(sorted((Path(dataset_dir) / "val2012").glob("*.JPEG")))

options = pyneat.ModelOptions()
options.preprocess.kind = pyneat.InputKind.Image
options.preprocess.input_max_width = FRAME_WIDTH
options.preprocess.input_max_height = FRAME_HEIGHT
options.preprocess.enable = pyneat.AutoFlag.On
options.preprocess.resize.enable = pyneat.AutoFlag.On
options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
options.preprocess.normalize.enable = pyneat.AutoFlag.On
options.preprocess.preset = pyneat.NormalizePreset.ImageNet
model = pyneat.Model(args.mpk, options)

benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_mla_layer_stats = True
benchmark_options.include_mla_stage_stats = True
# The image as RGB, scaled to fit a 1280x720 frame, centered, and padded with grey (114).
frame = letterbox_to_720p(read_rgb(image))
benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]

report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=f"{dataset_id} {image.parent.name}/{image.name} in a 1280x720 RGB frame, "
    "Neat preprocessing stretched to the model input",
)
report.save(args.output)
```

The settings follow the published scripts:

- `num_samples = 1000` sizes the latency pass, in place of Core's default of 100.
- `min_duration_ms = 60000` keeps the throughput pass running for at least 60
  seconds, in place of Core's default of 30. Sentinel samples every 2 seconds,
  so it records about 30 readings of the loaded board instead of about 15.
- `include_mla_layer_stats` and `include_mla_stage_stats` request MLA
  profiling, as [MLA profiling](#mla-profiling) describes.
- The script contains the exact model selection and decoder settings. YOLO26
  boxes select `BoxDecodeType.YoloV26` and keep the top 10 detections that
  score above 0.25, as an application would.

## Benchmark input

An image model repeats one real camera-sized frame, so its throughput is what an
application gets from a 720p frame to decoded results, not the MPK alone:

1. Pick one image from the dataset's validation split with `random.Random(0)`.
   Pose models pick among the images with labelled people.
2. Read it as RGB and letterbox it into a 1280x720 frame: scaled to fit,
   centered, and padded with grey (114).
3. Pass the frame to Core in `BenchmarkOptions.inputs` as an RGB `pyneat.Tensor`
   on the image route (`preprocess.kind = InputKind.Image`). Neat's
   preprocessing then resizes, normalizes, quantizes, and tessellates the frame
   in every timed inference.
4. Set `preprocess.input_max_width` and `input_max_height` to the frame size.
   Otherwise Neat sizes its input buffers for 1920x1080, and the execution plan
   shows that size as the model's input.

Set the model's own resize and normalization:

| Model | Resize | Normalization |
| --- | --- | --- |
| YOLO26 | `ResizeMode.Letterbox` | Neat's default scaling of pixels to [0, 1] |
| ResNet-50, RF-DETR's backbone | `ResizeMode.Stretch` | `preprocess.preset = NormalizePreset.ImageNet` |
| ViT-B/16, CLIP ViT-B/16, SigLIP 2 ViT-B/16 | `ResizeMode.Stretch` | their own `preprocess.normalize.mean` and `stddev` (0.5 for ViT-B/16), with `normalize.has_explicit_stats = True` |

Without a preset or explicit statistics, Neat applies no mean or standard
deviation. Give a Model one frame size only, because Core builds the benchmark
runner for the frame it receives.

Other inputs:

- A stereo model passes one pair from the dataset's test list, picked with
  `random.Random(0)`, as one side-by-side frame with the left view beside the
  right one. It skips the letterbox, stretches, and sets the ImageNet preset.
- A model whose inputs are not images, such as a keypoint matcher, measures one
  sample of its accuracy dataset, picked with `random.Random(0)`. It passes the
  arrays as `pyneat.Tensor` inputs on the tensor route
  (`preprocess.kind = InputKind.Tensor`, with resizing, color conversion, and
  normalization off). Each timed sample also runs the CPU steps that turn the
  MPK's outputs into the model's result.
- A model without an accuracy dataset builds a fixed, realistic input in its
  own script and takes no `--dataset`. An image model passes one 1280x720 RGB
  frame on the image route.

The `workload` line names the dataset, the image or sample, the frame, and the
decoder settings. Preprocessing is part of the measurement and can bound
throughput.

## What `from_performance()` takes and writes

```python
model_report.from_performance(
    benchmark,              # Core's BenchmarkReport, or a pyneat.BenchmarkReport the script filled
    models,                 # {component: pyneat.Model}, in execution order
    workload="...",         # one line that says what ran
    telemetry=None,         # a TelemetryMonitor.stop() result, when the script timed the model itself
    mla_layer_stats=None,   # {component: stats}, required for more than one component
    mla_stage_stats=None,   # {component: stats}, required for more than one component
)
```

Without `telemetry`, `from_performance()` reads the telemetry in `benchmark`.
Without the MLA stats, it reads them from `benchmark` too, which works for one
component only. It records each Model's
execution plan, from the same Model the script benchmarks, so the plan reflects
the script's options. It rejects a Core result that holds a non-finite number.

`report.save(output)` writes two files:

| File | Contents |
| --- | --- |
| `model-report.json` | `schema` (`sima.model-report`), `schema_version` (1), `stage` (`performance`), `runtime` (the PyNeat version and build), `workload`, `metrics`, Core's whole `telemetry`, and `components`, which holds each component's `execution_plan`, `mla_layer_stats`, and `mla_stage_stats` |
| `benchmark-report.json` | Core's own report, kept as evidence |

[Metrics](metrics.md) explains how `metrics` is filled.

## Multi-component models

A multi-component script measures every MPK and the CPU work between them as one
pipeline. RF-DETR's
[script](https://github.com/sima-neat/models/blob/main/models/rf_detr_nano/scripts/performance.py)
does it in this order:

1. It runs Core's benchmark on each MPK only for its MLA layer and stage stats.
2. It feeds the benchmark frame to the backbone on the image route, and runs
   each Model through its runner (`Model.build`, `push`, `pull`).
3. A latency pass sends one frame at a time. A throughput pass runs one thread
   per stage, so the backbone, the CPU step, and the decoder each work on a
   different frame. Each pass measures at least 1000 frames over at least 60
   seconds after 100 warm-up frames.
4. One `pyneat.TelemetryMonitor` samples the board over the throughput window.
5. It fills a `pyneat.BenchmarkReport` with what it timed and passes it to
   `from_performance()` with the monitor's telemetry, each component's Model,
   and each component's MLA stats:

```python
benchmark = pyneat.BenchmarkReport()
benchmark.latency_ms = float(np.mean(latencies) * 1000)
benchmark.latency_p50_ms = percentile_ms(latencies, 50)
benchmark.latency_p95_ms = percentile_ms(latencies, 95)
benchmark.latency_p99_ms = percentile_ms(latencies, 99)
benchmark.fps = throughput.fps
benchmark.throughput_duration_ms = throughput.seconds * 1000
benchmark.throughput_samples = throughput.frames
report = model_report.from_performance(
    benchmark,
    {"backbone": backbone_model, "decoder": decoder_model},
    workload=workload,
    telemetry=throughput.telemetry,
    mla_layer_stats=mla_layer_stats,
    mla_stage_stats=mla_stage_stats,
)
report.save(output)
```

The report's scores are the whole model's. Component numbers are never added
together. Throughput cannot exceed `1 / (1 / backbone_fps + 1 / decoder_fps)`
for component rates measured on the same board, because the MLA runs one MPK at
a time. A result well below that rate points at the pipeline, not the model.

A model that carries state from one input to the next cannot give every stage a
different input. EdgeTAM's throughput pass lets only the image encoder work
ahead, while one thread runs the stateful stages in order.

## Independent components

Components that run independently, such as CLIP's image and text encoders, are
measured one at a time. `model.yaml` declares `performance: {scope: components}`,
and the script builds one result per component, then saves one report:

```python
reports[component] = model_report.from_performance(
    model.benchmark(options), {component: model}, workload=workload,
)
model_report.from_component_performance(reports).save(output)
```

Each entry must be a performance report of exactly that component, and every
component must use the same runtime. The report has `scope: components` and a
`components` map whose entries hold their own `metrics`, `workload`,
`telemetry`, execution plan, and MLA stats. It has no top-level combined
metrics, and timings, power, memory, and energy are never added or averaged.
Core's evidence files are named `<component>-benchmark-report.json`.

## MLA profiling

With `include_mla_layer_stats` and `include_mla_stage_stats`, Core profiles each
MLA stage with MLA-RT after latency and throughput measurement:

- `mla_layer_stats` gives each compiled layer's `stage_index`, `name`, and
  `latency_us`. Names describe compiled layers, which may fuse or split source
  ONNX operators.
- `mla_stage_stats` gives each MLA stage's `stage_index` and mean `latency_us`.

Each has a `status`: `ok`, `not requested`, or why Core could not profile. A
stage index counts all executable stages, including A65 stages, so MLA indices
can have gaps. Profile on an idle DevKit with `mla-rt` in `PATH`. The profile is
a separate pass with MLA-RT's default inputs, so its summed time is not the
model's end-to-end latency.
