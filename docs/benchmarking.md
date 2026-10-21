# DevKit performance

The performance stage measures each compiled component on a Modalix DevKit, or
a model's whole package when its `benchmark` entry has package scope.
Each model's standalone `scripts/performance.py` contains its benchmark setup.
Single-component scripts load an MPK with `pyneat.Model` and call Neat Core's
benchmark API; RF-DETR's script measures its complete package pipeline. Core owns ordinary synthetic inputs, warmup, latency and throughput
measurement, board telemetry, and runner lifetime.

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

Every model that declares a benchmark includes `scripts/performance.py` in its
own GitHub folder. CI runs that same script. Copy it to the DevKit and run it
with PyNeat; no registry installation or YAML configuration is needed:

```bash
~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /workspace/resnet_50_mpk.tar.gz --output /workspace/benchmark
```

The script contains the exact model selection and measurement settings. For
example, YOLO26 boxes explicitly selects `BoxDecodeType.YoloV26` and top 100
results; pose and segmentation use their corresponding decoders. The current
scripts request 1000 samples, a 60000 ms minimum duration and layer statistics.
Customers can read or adapt these settings directly in the model's script.

`model.yaml` declares which workload runs, its scope/profiles and publication
gates. `benchmark: true` is shorthand for the single-component
`modalix_latency` entry over all supported profiles. Use an explicit entry for
a different workload ID, scope or required telemetry:

```yaml
benchmark:
- id: modalix_latency
  required_metrics: [avg_power_watts, energy_per_inference_joules]
```

Decoder and measurement settings have one source of truth in `performance.py`;
they are not repeated as YAML options or passed through a separate JSON flag.

`BenchmarkReport.save()` writes `benchmark-report.json`, the whole Core report
including its `telemetry` and `layer_stats`, so a field Core adds later reaches
the registry without a script change. `benchmark-context.json` records what the
report does not: the Core version, the requested sample count and minimum
duration, and the execution plan the model browser draws.

Scripts measure 1000 samples rather than Core's default of 100. The samples
size the latency pass and the energy the registry divides per inference, so the
script reports its sample count. Scripts also keep the throughput pass running
for at least 60 seconds (`BenchmarkOptions.min_duration_ms`) rather than Core's
default of 30, so Sentinel, which samples every 2 seconds, records about 30
readings of the loaded board instead of about 15.

The script also records the Model's execution plan: each step, the processor
it runs on, and the tensors between steps. The plan comes from the same Model
the script benchmarks, so it reflects that script's options, such as a box
decoder. The model page draws it as the component's execution graph.

A failed call exits unsuccessfully. Keep the log: a missing report is not a
successful benchmark. Keep archives and results outside Git.

The script runs once for each component that a `benchmark` entry covers in the
build profile.

## Package benchmark

A multi-component model can measure its package instead: every MPK and the
CPU work between them, running as one pipeline. RF-DETR declares one
package-scope entry:

```yaml
- id: orchestration
  scope: {type: package, package: default}
  required_metrics: [avg_power_watts, energy_per_inference_joules]
```

The script gets one `--mpk component=<archive>` per component:

```bash
~/pyneat/bin/python models/rf_detr_nano/scripts/performance.py \
  --mpk backbone=/workspace/backbone_mpk.tar.gz \
  --mpk decoder=/workspace/decoder_mpk.tar.gz --output /workspace/result
```

It first runs Core's benchmark on each MPK only for its layer timings, and
records each Model's execution plan. It then runs each Model through its async
runner (`Model.build`, `push`, `pull`) in two passes, each measuring at least
1000 frames over at least 60 seconds after 100 warm-up frames. The latency pass
sends one frame at a time. The throughput pass runs one thread per stage, so
the backbone, the CPU step, and the decoder each work on a different frame,
and each stage ends the next one's input when it finishes. One
`pyneat.TelemetryMonitor` samples the board over the throughput window.
`benchmark-report.json` uses Core's report fields; `benchmark-context.json`
adds each component's layer timings and execution plan.

The result is a schema 6 `performance-result.json` with `benchmark_type`
`model.orchestration`, `model.package_id`, a `components` list (each
component's build, layer timings, and execution plan). Its scores are the
whole model's. Component numbers are never added together.

Throughput cannot exceed `1 / (1 / backbone_fps + 1 / decoder_fps)` for
component rates measured on the same board, because the MLA runs one MPK at a
time. A result well below that rate points at the pipeline, not the model.

## Per-layer profiling

The model scripts enable `BenchmarkOptions.include_layer_stats` explicitly.
RF-DETR requests layer timings separately for each component.
After latency and throughput measurement, Core profiles each MLA stage using
MLA-RT and includes the compiled layer's `name`, zero-based `stage_index`, and
`total_execution_time_us` in `benchmark_report.layer_stats`. A stage index counts
all executable stages, including A65 stages, so MLA indices can have gaps.
Zero-time entries are kept in compiler order. Names describe compiled layers,
which may fuse or split source ONNX operators.

Use an idle DevKit with `mla-rt` in `PATH` and the MIP-to-kernel handoff fix
(verified with MLA-RT `603a8c6` on eLxr B2121). Core runs profiling after its
benchmark runners close. The profile is a separate pass with MLA-RT's default
inputs; its summed time is not the model's end-to-end latency or a breakdown of
throughput-pass telemetry. Ordinary inference can resume when it returns.

The importer preserves the saved report unchanged, validates every layer timing,
and fails when a requested profile is absent. Each model's performance stage
test also requires nonempty timings with at least one nonzero layer. Changing
these performance scripts reruns measurement for existing builds without
invalidating their compile keys. Historical results without layer stats stay
readable; new measurements add the analysis view.

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
synthetic inputs and the script's benchmark options. A script that configures the
model, such as YOLO26's box decoder, adds its selection and postprocessing to
its report, and CI records them in place of the defaults. CI records the
script's sample count, and records the input shapes, batch size,
preprocessing, and postprocessing separately using Core's model-inspection API;
that step loads the MPK but runs no inference.

Default-MPK results are a baseline of their own. Do not compare them with
historical YOLO measurements that configured image preprocessing and
BoxDecode explicitly.

Core reports mean latency and its p50, p95, and p99 percentiles, throughput in
logical inferences per second, average SoM power, and total energy. The
registry derives energy per inference by dividing total energy by the sample
count times the compiled batch size.
Missing power stays unavailable. A `benchmark` entry's `required_metrics` must
be present for publication; RF-DETR requires average power and energy per
inference.

`import-core-benchmark` maps the Sentinel telemetry to these metrics, each over
the throughput pass:

| Metric | Reading |
| --- | --- |
| `peak_power_watts` | Peak SoM power, with the average and peak of each power rail in its details |
| `avg_mla_power_watts`, `peak_mla_power_watts` | MLA rail power |
| `avg_die_temperature_c`, `peak_die_temperature_c` | SoC temperature: the hottest on-die sensor at each sample |
| `avg_board_temperature_c`, `peak_board_temperature_c` | SoM temperature: the hotter of the top-side and bottom-side board sensors |
| `avg_cpu_usage_pct`, `peak_cpu_usage_pct` | CPU usage averaged over all cores |
| `avg_cpu_core_usage_pct`, `peak_cpu_core_usage_pct` | The busiest CPU core: the highest per-core average, and the highest per-core peak, which can be different cores |
| `mla_memory_avg_mib`, `mla_memory_peak_mib` | MLA memory |
| `avg_cma_memory_mib`, `peak_cma_memory_mib` | The contiguous memory (CMA) pool in use |
| `avg_linux_memory_mib`, `peak_linux_memory_mib` | Linux memory in use |

The CMA reading covers the whole pool, which EV74 and other drivers share and
Linux lends to other memory, so it is not the model's memory alone. The
individual temperature sensors and CPU cores are not registry metrics; they
stay in the result's `benchmark_report`, where the browser reads the per-core
load.

The registry and the model browser read every metric from the saved report
with the same table: `REPORT_METRICS` and `TELEMETRY_METRICS` in
`tools/model_registry/benchmark_metrics.py`, mirrored in the browser's
`benchmark-adapters.js`. Core writes 0 for a headline value it did not measure
and 0 samples for a reading Sentinel did not provide; those metrics are not
published and the browser shows them as unavailable. Add a reading by adding a
row to both tables, with its Vulcan measurement name, and its label in the
browser's `metrics.js`. It
appears under **Other measurements** until `DETAIL_CARDS` in
`model-view.js` gives it a row in one of the Benchmark details cards.

Initialization timings and MLA memory before and after the run are not exported
by this Core API, so results do not carry them. Adding them belongs in Core before
Models consumes them.

## CI and registry publication

Model authors do not publish benchmark results directly. After a build
compiles and publishes, **Vulcan Model Compile** starts **Vulcan Model
Benchmarking** on the same branch for each build whose performance or accuracy
stage key has no completed run. The DevKit has no publication credentials.

On the DevKit, `scripts/run_model_performance.sh` sets the MLA and CPU
clocks, then handles each component:

1. `make-benchmark-manifest` writes `performance-test.json` with the
   compilation's identity.
2. The model's `performance.py` writes `benchmark-report.json`, `benchmark-context.json`,
   and `benchmark.log`.
3. The workload is recorded in the manifest.
4. `import-core-benchmark` stores the saved report unchanged as
   `benchmark_report` and the plan as `execution_plan` in
   `performance-result.json`, following
   [`performance-result.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/performance-result.schema.json)
   (`schema_version` 6). It records the compilation identity, the script
   digest, the source revision, the actual Core and platform versions, and
   the MLA and CPU clocks. It never runs or times a model.

A package benchmark runs once, in its last component's folder, with every
component's MPK.

`scripts/run_model_tests.sh` then runs the model's
`tests/measure/test_performance.py` on each component's result, which
compares its throughput with the expected value in `tests/tests.yaml`, checks
that the result records the execution plan the model browser draws, and
records `passed`, `failed`, or `skipped` ([set up
`tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)).

A separate hosted job validates the result set and publishes each result as a
`model_benchmark` run under the component's compile run. The run records the
performance stage key and carries a `performance-result` artifact, the
benchmark log, a `performance/model-synthetic` test result whose status is the
test outcome, and the measurements. A package result publishes under its last
component's compile run, with one `compiled-model-<component>` input per
component, and its stage key covers every component's build. A failed test still publishes the result
and completes its run, then fails the workflow run. The model browser reads
performance results with `schema_version` 1 through 6, shows measured values
with their units, and marks absent metrics unavailable. Its Speed card shows the
MLA and CPU clocks when the result records them. A failed measurement keeps its
workflow evidence and is not published as a successful measurement. Performance
and accuracy publish on their own outcomes: when one stage fails, the other's
result is still published, and the next push reruns only the stage that has no
run. A dispatched run stops before measuring when its branch has since changed
the tooling the stage keys hash; the newer push measures that build instead.

Because the performance stage has its own key, editing the selected benchmark implementation, a
`benchmark` entry, or the DevKit runtime pin remeasures performance from the
published MPKs without recompiling. See [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns).
