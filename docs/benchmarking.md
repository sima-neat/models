# DevKit performance

For LLMs and VLMs, see [LLiMa performance](#llima-text-and-image-performance).

The Core performance stage measures each build profile's whole model on a Modalix
DevKit: every compiled MPK and the CPU work between them. Each model's
standalone `scripts/performance.py` contains its benchmark setup, including
its input: one image from the model's accuracy dataset in a 1280x720 frame,
which Neat preprocesses inside the measurement, or one sample as tensors for a
model whose inputs are not images (see [Benchmark input](#benchmark-input)).
Scripts of single-component models load the MPK
with `pyneat.Model` and call Neat Core's benchmark API; RF-DETR, a
multi-component model, times its pipeline itself in its script. Core owns
warmup, latency and throughput measurement, board telemetry, and runner
lifetime. The scripts hand Core's results to `tools/model_report`, which
writes the one report the registry publishes.

## Set up a DevKit

The DevKit must run the platform version that
[`deps/manifest.json`](../deps/manifest.json) pins as `platform-version`. It must
also hold the Core package that `neat-core.install_target` names in that file.

CI installs Core from production with
`sima-cli neat install --env production --type minimal core@<branch>:<tag>`. For a
target that names only a branch, such as `core@develop`, CI uses the newest published
tag of that branch. Install the same package on your DevKit, so that your numbers
compare with the published ones.

Core's board installer also installs Sentinel, the DevKit telemetry daemon.
Core's benchmark reads power, temperature, CPU usage, and memory from Sentinel
over the throughput pass. Without Sentinel the benchmark still measures speed,
and the report names why the board telemetry is missing.

Results are measured with the MLA at 1 GHz, the highest clock its devfreq
governor offers. Throughput and power depend on the clock, while model outputs
do not. Before measuring, `scripts/run_model_performance.sh` reads
`/sys/class/devfreq/5000000.mla/cur_freq`; when the clock is lower, it writes
1 GHz to `userspace/set_freq` with `sudo -n`. When the clock is still not 1 GHz,
the stage fails without measuring.

The CPU feeds the MLA and runs postprocessing, so its clock also bounds
throughput: YOLO26n INT8 with the box decoder measures about 1060 FPS at
1.4 GHz and about 930 FPS at 1.2 GHz. One cpufreq policy covers every Modalix
core. When `/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq` is not
1.4 GHz, the script writes 1.4 GHz to `scaling_setspeed`, which requires the
`userspace` governor, and fails without measuring when the clock still differs.

The runner user therefore needs passwordless `sudo` for both writes.

## The performance script

Every model that declares `performance` includes `scripts/performance.py` in
its own GitHub folder. CI runs that same script. Copy the model folder,
`tools/model_report` and `tools/model_inputs` to the DevKit, extract the model's accuracy dataset, and
run the script with PyNeat; no registry installation or YAML configuration is
needed:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /workspace/resnet_50_w8a8_mpk.tar.gz \
  --dataset imagenet-2012-val=/media/nvme/datasets/imagenet-2012-val \
  --output /workspace/performance
```

The script takes:

- `--mpk`: the MPK, or `<component>=<path>` once per component for a
  [multi-component model](#multi-component-models);
- `--dataset <source id>=<directory>`: the model's accuracy dataset,
  extracted, as `accuracy.py` takes it (see [the accuracy
  script](local-accuracy-development.md#the-accuracy-script)); a model that
  declares no accuracy dataset takes no `--dataset`; and
- `--output <directory>`.

The script contains the exact model selection and measurement settings. For
example, YOLO26 boxes explicitly selects `BoxDecodeType.YoloV26` and keeps the
top 10 detections that score above 0.25, as an application would; pose and
segmentation use their corresponding decoders with the same limits. The
current scripts of single-component models request 1000 samples, a 60000 ms
minimum duration and MLA layer and stage stats. Customers can read or adapt
these settings directly in the model's script.

The script hands what Core returned to `model_report.from_performance()`:

```python
report = model_report.from_performance(
    model.benchmark(benchmark_options),
    {"model": model},
    workload=f"{dataset_id} {image.name} in a 1280x720 RGB frame, Neat preprocessing, ...",
)
report.save(args.output)
```

`report.save()` writes `model-report.json`, the whole-model report the
registry publishes unchanged and the model browser reads, and Core's own
`benchmark-report.json` beside it as evidence.
[`model-report.schema.json`](../schemas/model-report.schema.json) defines the
report. Its `metrics` hold one number per measured metric, named as
[`tools/model_report/metrics.json`](../tools/model_report/metrics.json)
defines them. It also keeps Core's whole `telemetry` and, for each component,
its execution plan and MLA layer and stage stats. `workload` is one line that says what
ran, such as the selected decoder and postprocessing.

`model.yaml` declares that the model is measured and, optionally, which build
profiles:

```yaml
performance: {}
```

Without `build_profiles`, every build profile is measured. Every Core report must
hold the metrics [`metrics.json`](../tools/model_report/metrics.json) marks
`required`: average latency, throughput, average SoM power, and energy per
inference. An LLM or VLM report is checked per workload instead. Each workload must
hold `ttft_ms` and `decode_tps`, a text workload also `prefill_tps`, and each
workload's telemetry must give `avg_power_watts`. Decoder and
measurement settings have one source of truth in `performance.py`; they are
not repeated in YAML.

Scripts measure 1000 samples rather than Core's default of 100, which sizes
the latency pass. Scripts also keep the throughput pass running for at least
60 seconds (`BenchmarkOptions.min_duration_ms`) rather than Core's default of
30, so Sentinel, which samples every 2 seconds, records about 30 readings of
the loaded board instead of about 15.

The report records each component's execution plan: each step, the processor
it runs on, and the tensors between steps. The plan comes from the same Model
the script benchmarks, so it reflects that script's options, such as a box
decoder. The model page draws it as the component's execution graph.

A failed call exits unsuccessfully. Keep the log: a missing report is not a
successful benchmark. Keep archives and results outside Git.

The script runs once per benchmarked build profile, with every component's MPK.

### Benchmark input

CI passes `--dataset` with the model's first accuracy dataset, and only for a build
profile that accuracy measures.

`scripts/unpack_benchmark_datasets.sh` unpacks each dataset into a dataset folder. A
marker beside each unpack lets a later step of the same job reuse it without reading the
archive again. The archive comes from the DevKit pool's shared dataset archive share when
the share's index lists the dataset's SHA-256 and size and the share answers a listing
within 20 seconds. The script reads that archive once, in place, and checks its SHA-256
and size against `model.yaml` as it unpacks it. When the share cannot give the archive or
the check fails, the script removes what it unpacked, and the Dataset Registry downloads
the archive into a cache on the NVMe. The script checks that archive before it unpacks it.

In CI the dataset step unpacks into the job's dataset folder, both stages reuse it, and
the job removes it when it ends. On your own DevKit, `scripts/run_model_performance.sh`
and `scripts/run_model_accuracy.sh` each unpack the datasets into their own work folder
and remove it when they end. To unpack once for both, set `DATASET_DIR` to a folder
outside that work folder, and delete it when you are done.

These environment variables set the locations:

| Variable | Default | Holds |
|---|---|---|
| `DATASET_DIR` | `$WORK_DIR/datasets` | the unpacked datasets |
| `DATASET_CACHE` | `/media/nvme/datasets/artifacts` | the archives the Dataset Registry downloads |
| `DATASET_LOCK` | `/media/nvme/datasets/locks/prepare.lock` | the lock that lets one run at a time download an archive |
| `SHARED_DATASET_ARCHIVES` | `/mnt/dataset-private/current` | the shared dataset archive share. Without it, each archive comes from the Dataset Registry |

`NVME_MOUNT`, by default `/media/nvme`, moves `DATASET_CACHE` and `DATASET_LOCK` together.

For an image model, the benchmark repeats one real camera-sized frame, so its
throughput is what an application gets from a 720p frame to decoded results,
not the MPK alone:

1. The script picks one image from the dataset's validation split with
   `random.Random(0)`, so every run measures the same image. Pose models pick
   among the images with labelled people, which pose accuracy scores.
2. It reads the image as RGB with `read_rgb` and letterboxes it into a
   1280x720 frame with `letterbox_to_720p`: scaled to fit, centered, and padded
   with grey (114). Both come from `model_inputs.frame`, a shared module of
   OpenCV and NumPy code under `tools/`.
3. It passes the frame to Core in `BenchmarkOptions.inputs` as an RGB
   `pyneat.Tensor`. The Model uses the image route
   (`preprocess.kind = InputKind.Image`), so Neat's preprocessing resizes,
   normalizes, quantizes and tessellates the frame in every timed inference.
   It sets `preprocess.input_max_width` and `input_max_height` to the frame
   size. Otherwise Neat sizes its input buffers for 1920x1080, and the
   execution plan shows that size as the model's input.
   YOLO26 resizes with `ResizeMode.Letterbox` and keeps Neat's default
   scaling of pixels to [0, 1]. ResNet-50 and RF-DETR's backbone resize with
   `ResizeMode.Stretch` and set `preprocess.preset =
   NormalizePreset.ImageNet`. ViT-B/16, CLIP ViT-B/16 and SigLIP 2 ViT-B/16
   also stretch, and set their own mean and standard deviation (0.5 for
   ViT-B/16) in `preprocess.normalize.mean` and `stddev` with
   `normalize.has_explicit_stats = True`. Without a preset or explicit
   statistics, Neat applies no mean or standard deviation.

The report's `workload` names the dataset, the image, the frame, and the
decoder settings. Preprocessing is part of the measurement and can bound
throughput. Give a Model one frame size only: Core builds the benchmark runner
for the frame it receives.

A stereo model takes its pair as one side-by-side frame, so it skips the
letterbox: it picks one pair from the dataset's test list with
`random.Random(0)` (for Scene Flow, `sceneflow_finalpass_test.txt`),
reads both views as RGB, and passes them as one frame, the left view beside
the right one (1920x540 for Scene Flow). Its Model resizes with
`ResizeMode.Stretch` and sets the ImageNet preset, and the workload names the
pair's left image.

A model whose inputs are not images, such as a keypoint matcher, measures one
sample of its accuracy dataset instead of a frame. The script picks the sample
from the dataset's validation list with `random.Random(0)` (for SuperGlue, one
pair of `pairs_val.txt` whose images both hold 512 keypoints), passes its arrays
to Core as `pyneat.Tensor` inputs on the tensor route (`preprocess.kind =
InputKind.Tensor`, with resizing, color conversion and normalization off), and
names the dataset and the sample in `workload`. Each timed sample also runs the
CPU steps that turn the MPK's outputs into the model's result, such as
SuperGlue's Sinkhorn iterations and match extraction, so throughput is what an
application gets from inputs to matches, as for image models. The shared
dataset-image test in `tests/test_model_report.py` skips a model without an
image input, so its own `tests/benchmark/test_performance.py` checks the sample,
the tensors and the CPU steps.

A model without an accuracy dataset builds a fixed, realistic input in its
own script and takes no `--dataset`. An image model passes one 1280x720 RGB
frame on the image route, as above, and the shared fixed-frame test in
`tests/test_model_report.py` checks that every run benchmarks the same
non-blank frame.

A model whose application also reads files beyond its MPKs, such as Supertonic's
text models and voices, declares them as `resources` in `model.yaml`, and its
script takes `--resources <directory>`. The runner passes the directory to
every model that declares resources, so a model whose only resource is its
[license](metadata-schema.md#license-files-in-the-package) takes the argument
and reads nothing from it. For a standalone run, fetch them in the
layout the package installs, or extract the package ZIP to the same effect, and
install the model's `scripts/requirements-performance.txt` beside PyNeat's
environment, as the stage does:

```bash
resources="$(model-registry fetch-resources models/supertonic_3 --output-dir /workspace/resources)"
~/pyneat/bin/python -m pip install --no-deps --only-binary=:all: \
  --target /workspace/site -r models/supertonic_3/scripts/requirements-performance.txt
PYTHONPATH=tools:/workspace/site ~/pyneat/bin/python models/supertonic_3/scripts/performance.py \
  --mpk vector_estimator=/workspace/supertonic_3_w8a16_w16a16_vector_estimator_mpk.tar.gz \
  --mpk vocoder=/workspace/supertonic_3_w8a16_w16a16_vocoder_mpk.tar.gz \
  --resources "$resources" --output /workspace/performance
```

## Multi-component models

A multi-component model's script measures every MPK and the CPU work between
them, running as one pipeline. It gets one `--mpk component=<archive>` per
component, in `model.yaml` order:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/rf_detr_nano/scripts/performance.py \
  --mpk backbone=/workspace/rf_detr_nano_w8a16_w16a16_backbone_mpk.tar.gz \
  --mpk decoder=/workspace/rf_detr_nano_w8a16_w16a16_decoder_mpk.tar.gz \
  --dataset coco-2017-val=/media/nvme/datasets/coco-2017-val \
  --output /workspace/performance
```

RF-DETR's script first runs Core's benchmark on each MPK, with Core's
synthetic inputs, only for its MLA layer and stage stats. It then feeds the
[benchmark input](#benchmark-input) frame to the backbone, which stretches it
to the model input on the image route, and runs each Model through its async
runner (`Model.build`, `push`, `pull`) in two passes, each measuring at least
1000 frames over at least 60 seconds after 100 warm-up frames. The latency pass sends one frame at
a time. The throughput pass runs one thread per stage, so the backbone, the
CPU step, and the decoder each work on a different frame, and each stage ends
the next one's input when it finishes. One `pyneat.TelemetryMonitor` samples
the board over the throughput window.

The script fills a `pyneat.BenchmarkReport` with what it timed and passes it
to `model_report.from_performance()` with the monitor's telemetry, each component's
Model, and each component's MLA layer and stage stats. The report's scores are the whole
model's; component numbers are never added together.

Throughput cannot exceed `1 / (1 / backbone_fps + 1 / decoder_fps)` for
component rates measured on the same board, because the MLA runs one MPK at a
time. A result well below that rate points at the pipeline, not the model.

A model that carries state from one input to the next cannot give every stage
a different input. EdgeTAM tracks an object through a video, and a frame's
memory attention reads the memory the previous frame wrote. Its script clicks
an annotated object on the first frame of one DAVIS 2017 val sequence, then
measures tracked frames on repeats of that frame. In the
throughput pass only the image encoder works ahead, on the next frames, while
one thread runs the memory attention, the mask decoder, the CPU steps and the
memory encoder in order.

### Independent components

Components that can run independently, such as CLIP's image and text encoders,
use this declaration instead of a composed pipeline benchmark:

```yaml
performance:
  scope: components
```

Omitting `scope` keeps the whole-model behavior (`scope: model`), including for
RF-DETR. Both modes publish one performance run per build profile and require
all declared components. Independent components are measured one at a time;
their timings, power, memory and energy are not added or averaged.

The script builds each component's result with the existing helper, then saves
one report:

```python
reports[component] = model_report.from_performance(
    model.benchmark(options), {component: model}, workload=workload,
)
model_report.from_component_performance(reports).save(output)
```

A component report has `scope: components` and a `components` map. Every entry
contains its own `metrics`, `workload`, `telemetry`, execution plan and MLA stats.
There are no top-level combined metrics. The required metrics and regression
limits apply to every component; `tests/tests.yaml` declares expectations under
`performance.expected.<profile>.components.<component>.throughput_fps`.
Core's evidence files are named `<component>-benchmark-report.json`.

The model page shows a component selector and the selected component's
metric cards. The catalog table lists each component separately. CSV downloads
identify the selected component and contain its readings.

For CLIP, the image benchmark includes Neat preprocessing of a 720p RGB frame.
The text benchmark uses prepared 77-token embeddings. Tokenization, CPU token
lookup and image/text similarity scoring happen outside these independent
encoder measurements; each workload states that boundary.

## MLA profiling

The model scripts enable `BenchmarkOptions.include_mla_layer_stats` and
`include_mla_stage_stats` explicitly. RF-DETR requests them separately for each
component. After latency and throughput measurement, Core profiles each MLA
stage with MLA-RT. Its BenchmarkReport holds two results, which the model report
keeps under each component:

- `mla_layer_stats`: each compiled layer's `stage_index`, `name` and
  `latency_us`. Zero-time entries are kept in compiler order. Names describe
  compiled layers, which may fuse or split source ONNX operators.
- `mla_stage_stats`: each MLA stage's `stage_index` and mean `latency_us`.

Each also has a `status`: `ok`, `not requested`, or why Core could not profile.
A stage index counts all executable stages, including A65 stages, so MLA
indices can have gaps.

Use an idle DevKit with `mla-rt` in `PATH` and the MIP-to-kernel handoff fix
(verified with MLA-RT `603a8c6` on eLxr B2121). Core runs profiling after its
benchmark runners close. The profile is a separate pass with MLA-RT's default
inputs; its summed time is not the model's end-to-end latency or a breakdown of
throughput-pass telemetry. Ordinary inference can resume when it returns.

`model_report` keeps each component's stats unchanged, and each model's
performance stage test requires both statuses to be `ok`, nonempty stage stats,
and at least one nonzero layer. Changing a performance script remeasures
existing builds without invalidating their compile fingerprints.

### MLA utilization

After an MPK compile, `scripts/run_model_compilation.sh` runs
`model-registry simulate-mla-stages`. It runs the Model Compiler's `mla-msim
--dry-run` on each MLA stage's ELF and writes `mla-stage-simulation.json` beside
`compile-outputs.json`: for each component, each MLA stage's `stage_index`
(numbered as Core numbers it), `effective_ops`, `total_ops`, and
`simulated_latency_us`. A missing or failing `mla-msim` is recorded in the
component's `status` rather than failing the build.
[`mla-stage-simulation.schema.json`](../schemas/mla-stage-simulation.schema.json)
defines the file. The package carries it, and CI publishes it with the package.

The model browser's developer mode compares it with the measured
`mla_stage_stats` as **MLA Utilization**. Effective TOPS divides a stage's
effective operations by its measured latency; packing is
`effective_ops / total_ops`; and utilization is its effective TOPS as a share of
the MLA's peak: 50 TOPS at INT8, and 12.5 TOPS for a stage with BF16
activations (w16a16, w8a16, w4a16), because the MLA multiplies only INT8 or
BF16 operands and a BF16 multiply takes four times the cycles.

The model browser's **Runtime Layer Analysis** shows the profiled MLA time, layer
count, and share in the ten slowest layers. **Duration** displays a ranked chart
of measured execution times in microseconds, with colors inferred from compiled
layer names and a fixed time axis. **Table** shows names, execution time and
share in compiler order, without bars. Only the selected view is displayed.
Both views share name search and a minimum-execution-time slider. Filtering
preserves the component totals, percentage denominator and time axis; a zero
threshold includes zero-time layers. Stage indices remain visible per layer.
**Download CSV** includes every layer in compiler order, regardless of filters.

## Workload and measurements

The workload is the script's [benchmark input](#benchmark-input) on the image
route, or on the tensor route for a model whose inputs are not images, with the
script's model and benchmark options; the report's `workload`
states both. Results measured before scripts took a dataset image used Core's
synthetic inputs without image preprocessing, and YOLO26 decoded up to 100
detections at any score. Do not compare those results with current ones.

Core reports mean latency and its p50, p95, and p99 percentiles, throughput in
logical inferences per second, and average SoM power. Throughput counts every
inference in a batch, so `model_report` derives energy per inference by
dividing average power by throughput alone. Core writes 0 for a headline value
it did not measure, and Sentinel provides 0 samples for a reading it did not
take; those metrics are absent from the report and the browser shows them as
unavailable. A missing required metric fails the stage before
publication.

`model_report` reads these Sentinel metrics from the telemetry, each over the
throughput pass:

| Metric | Reading |
| --- | --- |
| `peak_power_watts` | Peak SoM power; the browser shows each power rail from the report's telemetry |
| `avg_mla_power_watts`, `peak_mla_power_watts` | MLA rail power |
| `avg_die_temperature_c`, `peak_die_temperature_c` | SoC temperature: the hottest on-die sensor at each sample |
| `avg_board_temperature_c`, `peak_board_temperature_c` | SoM temperature: the hotter of the top-side and bottom-side board sensors |
| `avg_cpu_usage_pct`, `peak_cpu_usage_pct` | CPU usage averaged over all cores |
| `avg_cpu_core_usage_pct`, `peak_cpu_core_usage_pct` | The busiest CPU core: the highest per-core average, and the highest per-core peak, which can be different cores |
| `mla_memory_avg_mib`, `mla_memory_peak_mib` | MLA memory: buffers from the MLA memory pool, plus shared buffers only the MLA uses |
| `ev74_memory_avg_mib`, `ev74_memory_peak_mib` | EV74 memory: buffers EV74 holds, such as the frames it preprocesses for the MLA |
| `avg_linux_memory_mib`, `peak_linux_memory_mib` | Memory the whole board uses (Linux `MemTotal - MemAvailable`) |

Sentinel gives each buffer one owner, so MLA and EV74 memory add up to the
model's accelerator memory: a shared buffer EV74 also uses counts as EV74, and
frames a video decoder or camera ISP produced count as neither. Linux memory
covers every process on the board and includes EV74's buffers, so it is board
context, not the model's memory, and does not add to the other two. EV74
memory needs Sentinel with `ev74_mem_allocated_mb`; with an older daemon the
report has no EV74 memory. Reports measured before that change carried the
whole contiguous memory (CMA) pool in use instead; the browser no longer shows
it, and their MLA memory also counted the buffers EV74 hands to the MLA. The
individual temperature sensors and CPU cores are not registry metrics; they
stay in the report's `telemetry`, where the browser reads the per-core load.

`tools/model_report/metrics.json` is the one metric vocabulary: each metric's
label, unit, direction, Vulcan measurement name (`registry_name`), and the Core report field or
telemetry statistic it comes from. Add a reading by adding a row there and its
display entry in the browser's `metrics.js`; a test keeps the labels, units,
and directions of the two in step. It appears under **Other measurements**
until `DETAIL_CARDS` in `performance-section.js` gives it a row in one of the
Performance Details cards.

Initialization timings and MLA memory before and after the run are not exported
by this Core API, so reports do not carry them. Adding them belongs in Core
before Models consumes them.

## CI and registry publication

Model authors do not publish benchmark results directly. After a build profile's
package publishes, CI measures each declared stage on a DevKit with the model's own
`performance.py` and `accuracy.py`. It then runs the model's benchmark tests on each
report and publishes the report.

What you see after a push:

- CI measures a stage only when no published result matches its benchmark
  [fingerprint](concepts.md#fingerprint), a hash of every input that decides the
  result.
- CI does not measure an on-demand build on a push. A maintainer starts that
  benchmark by hand.
- A failed benchmark test still publishes its report, marked `failed`, and then
  fails the CI run. CI does not measure that build again until its fingerprint
  changes. [Set up `tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)
  describes the tests.
- A measurement that fails publishes nothing, and the next push measures it again.
- Performance and accuracy publish separately. When one stage fails, CI still
  publishes the other's report.
- The model browser shows "Not measured at this revision." for a build profile with
  no published result.

The performance stage has its own fingerprint. Editing the model's `performance.py`,
its `performance` declaration, the shared report and publication tooling, or the
DevKit runtime pin therefore measures performance again from the published MPKs
without recompiling. See
[what a change reruns](adding-a-model.md#9-know-what-a-change-reruns).

## LLiMa text and image performance

LLMs and VLMs declare `performance: {}`. The stage uses Core's LLiMa runtime and
MoLE from the LLiMa revision that `deps/manifest.json` pins.

| Deployment | Workloads |
| --- | --- |
| Ordinary LLM | Text input buckets |
| Ordinary VLM | Text buckets, plus one encoder-sized image with “Describe what you see in the image.” |
| Speculative LLM | Full GSM8K, BFCL v3 simple/multiple and AlpacaEval datasets |
| Speculative VLM | Those text datasets, plus full RealWorldQA, DocVQA validation and ChartQA at dataset image resolution |

Text buckets double from 128 through `max_context - 1024`, including that
boundary: 128, 256, 512, 1024, 2048 and 3072 for 4K context. Ordinary workloads
use one warmup and five measurements; datasets use one warmup and one measurement
per prompt. Warmup is excluded, and caches reset before every request. Image and
dataset requests allow up to 256 generated tokens; ordinary text measures the
first token plus 256 decode steps.

Speculative profiles declare an ordinary model/profile in `benchmark_baseline`.
Both deployments use the same dataset sources, prompts, context and output
budgets. BFCL is skipped for both deployments if either lacks a native tool
template; the reason is recorded in the log and JSON report.
Reports include TTFT (time to first token), text prefill TPS, decode TPS excluding the first
token, draft contribution (accepted draft/generated tokens) and generation speedup (aggregate
speculative/baseline decode TPS, excluding prompt processing and the first token). The browser
labels decode TPS as generation speed and the non-speculative comparison as Baseline.
Power, temperature, CPU and memory are recorded
per workload and deployment. Failed requests prevent publication.
The benchmark assumes exclusive use of the DevKit and stops any existing
Sentinel recording once at startup.

For local runs, use MoLE with `PYTHONPATH=tools` and the model's `performance.py`
with `--deployment`, `--output` and, for speculative models, `--baseline` pointing
to the staged baseline receipt. Results use `model-report.json`; raw reports
retain request and token timings.
