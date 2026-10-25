# DevKit performance

The performance stage measures each build profile's whole model on a Modalix
DevKit: every compiled MPK and the CPU work between them. Each model's
standalone `scripts/performance.py` contains its benchmark setup. Scripts of
single-component models load the MPK with `pyneat.Model` and call Neat Core's
benchmark API; RF-DETR, a multi-component model, times its pipeline itself in
its script. Core owns
ordinary synthetic inputs, warmup, latency and throughput measurement, board
telemetry, and runner lifetime. The scripts hand Core's results to
`tools/model_report`, which writes the one report the registry publishes.

## Set up a DevKit

The DevKit must run the platform version that `deps/manifest.json` pins, and
the stage uses the Core package named by `neat-core.install_target` in that
file. CI checks the platform version, then installs Core with:

```bash
sima-cli neat install --env production --type minimal '<neat-core.install_target>'
~/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__, pyneat.__file__)'
```

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
its own GitHub folder. CI runs that same script. Copy the model folder and
`tools/model_report` to the DevKit and run the script with PyNeat; no registry
installation or YAML configuration is needed:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /workspace/resnet_50_mpk.tar.gz --output /workspace/performance
```

The script contains the exact model selection and measurement settings. For
example, YOLO26 boxes explicitly selects `BoxDecodeType.YoloV26` and top 100
results; pose and segmentation use their corresponding decoders. The current
scripts of single-component models request 1000 samples, a 60000 ms minimum
duration and MLA layer and stage stats. Customers can read or adapt these settings
directly in the model's script.

The script hands what Core returned to `model_report.from_performance()`:

```python
report = model_report.from_performance(
    model.benchmark(benchmark_options), {"model": model}, workload="Model.benchmark with default options"
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

Without `build_profiles`, every build profile is measured. Every report must
hold the metrics [`metrics.json`](../tools/model_report/metrics.json) marks
`required`: average latency, throughput, average SoM power, and energy per
inference. Decoder and
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

## Multi-component models

A multi-component model's script measures every MPK and the CPU work between
them, running as one pipeline. It gets one `--mpk component=<archive>` per
component, in `model.yaml` order:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/rf_detr_nano/scripts/performance.py \
  --mpk backbone=/workspace/backbone_mpk.tar.gz \
  --mpk decoder=/workspace/decoder_mpk.tar.gz --output /workspace/performance
```

RF-DETR's script first runs Core's benchmark on each MPK only for its MLA
layer and stage stats. It then runs each Model through its async runner (`Model.build`,
`push`, `pull`) in two passes, each measuring at least 1000 frames over at
least 60 seconds after 100 warm-up frames. The latency pass sends one frame at
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
defines the file. The package carries it, and a staging package run publishes it
as the `performance/mla-stage-simulation` test result.

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

The workload is the default route Core selects from the MPK, with Core's
synthetic inputs and the script's benchmark options. A script that configures
the model, such as YOLO26's box decoder, says so in the report's `workload`.

Default-MPK results are a baseline of their own. Do not compare them with
historical YOLO measurements that configured image preprocessing and
BoxDecode explicitly.

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
until `DETAIL_CARDS` in `model-view.js` gives it a row in one of the
Performance Details cards.

Initialization timings and MLA memory before and after the run are not exported
by this Core API, so reports do not carry them. Adding them belongs in Core
before Models consumes them.

## CI and registry publication

Model authors do not publish benchmark results directly. After a build profile's
package publishes, **Vulcan Model Compile** starts **Vulcan Model
Benchmarking** on the same branch for each build profile whose performance or
accuracy benchmark fingerprint has no completed run. On-demand builds are not
benchmarked. The DevKit has no publication credentials.

On the DevKit, the workflow first checks that the profile's components are
registered and its package receipt is present. `scripts/run_model_performance.sh`
then:

1. lists each component's MPK with `model-registry model-component-mpks`;
2. sets the MLA and CPU clocks;
3. runs the model's `performance.py`, which writes `model-report.json` and
   `benchmark-report.json`, and copies its output to `performance.log`;
4. runs `model-registry check-report`, which validates the report against
   [`model-report.schema.json`](../schemas/model-report.schema.json) and the
   `model.yaml` declaration and writes the board's installed packages and
   clocks to `environment.json`.

`scripts/run_model_tests.sh` then runs the model's
`tests/benchmark/test_performance.py` on the report, which compares its
throughput with the expected value for that build profile in
`tests/tests.yaml`, checks the execution plans and layer timings, and records
`passed`, `failed`, or `skipped` ([set up
`tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)).

A hosted job then runs `model-registry publish-report`, which publishes the
report as one `model_performance` run under the build profile's `model_package`
run. Its `run_fingerprint` is the performance benchmark fingerprint. The run
has one `compiled-model-<component>` input per component, the `model-report`,
Core report, and log artifacts, a `performance/model` test result whose summary
is the whole report and whose status is the test outcome, and one measurement
per metric. The board's clocks and installation go in the run environment. A
failed test still publishes the report and completes its run, then fails the
workflow run. A failed measurement keeps its workflow evidence and is not
published. Performance and accuracy publish on their own outcomes: when one
stage fails, the other's report is still published, and the next push reruns
only the stage that has no run. A dispatched run stops before measuring when
its branch has since changed the tooling the benchmark fingerprints hash; the
newer push benchmarks that build instead.

The catalog lists each build profile's benchmark fingerprints under
`benchmark_fingerprints`, and the model browser resolves the runs by those
fingerprints. A build profile with no completed run shows "Not measured at this
revision."

Because the performance stage has its own fingerprint, editing the model's
`performance.py`, its `performance` declaration, the shared report and
publication tooling, or the DevKit runtime pin reruns the performance benchmark
from the published MPKs without recompiling. See [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns).
