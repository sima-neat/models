# DevKit performance

The performance stage measures each compiled component on a Modalix DevKit.
Each model owns `scripts/performance.py`, which loads one MPK with
`pyneat.Model` and calls Neat Core's benchmark API. Core owns the synthetic
inputs, warmup, latency and throughput measurement, power telemetry, and runner
lifetime. There is no shared benchmark executor and no model-specific timing
loop.

## Set up a DevKit

The DevKit must run the platform version that `deps/manifest.json` pins, and
the stage uses the Core package named by `neat-core.install_target` in that
file. CI checks the platform version, then installs Core with:

```bash
sima-cli neat install --env production --type minimal '<neat-core.install_target>'
~/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__, pyneat.__file__)'
```

The scripts use the existing `BenchmarkReport` fields; no Core extension is
required.

## The performance script

`scripts/performance.py` takes `--mpk <archive>` and `--output <report>`:

```bash
~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /workspace/resnet_50_mpk.tar.gz --output /workspace/core-benchmark.json
```

Every model's script follows the same pattern:

```python
import argparse
import json

import pyneat

parser = argparse.ArgumentParser()
parser.add_argument("--mpk", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
report = pyneat.Model(args.mpk).benchmark(benchmark_options)
with open(args.output, "w", encoding="utf-8") as output:
    json.dump({
        "core_version": pyneat.__version__,
        "latency_ms": report.latency_ms,
        "fps": report.fps,
        "avg_power_watts": report.avg_power_watts,
        "energy_joules": report.energy_joules,
        "samples": benchmark_options.num_samples,
    }, output, allow_nan=False)
```

Scripts measure 1000 samples rather than Core's default of 100. At about
1000 FPS, 100 samples last about 0.1 s, so startup and ramp-up dominate the
throughput window and the 100 ms power sampling catches the board before it
reaches full load: YOLO26n INT8 measured about 945 FPS and 8.7 W with 100
samples, and about 1040 FPS and 11.9 W with 1000. The script reports its
sample count because the registry divides total energy by it.

A failed call exits unsuccessfully. Keep the log: a missing report is not a
successful benchmark. Keep archives and results outside Git.

The script runs once for each component that a `benchmark` entry covers in the
build profile. RF-DETR's backbone and transformer therefore produce separate
results; their numbers are not a measured combined pipeline.

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

Core reports mean latency, throughput in logical inferences per second, average
power, and total energy. The registry derives energy per inference by dividing
total energy by the sample count times the compiled batch size. Missing power
stays unavailable. A `benchmark` entry's `required_metrics` must be present for
publication; RF-DETR requires average power and energy per inference.

Percentiles, initialization timings, peak MLA memory, and temperature traces
are not exported by this Core API. Results mark those metrics unavailable
rather than estimating them or copying them from old results. Adding them
belongs in Core before Models consumes them.

## CI and registry publication

Model authors do not publish benchmark results directly. After a build
compiles and publishes, **Vulcan Model Compile** starts **Vulcan Model
Benchmarking** on the same branch for each build whose performance or accuracy
stage key has no completed run. The DevKit has no publication credentials.

On the DevKit, `scripts/run_model_performance.sh` handles each component:

1. `make-benchmark-manifest` writes `performance-test.json` with the
   compilation's identity.
2. `performance.py` writes `core-benchmark.json` and `benchmark.log`.
3. The workload is recorded in the manifest.
4. `import-core-benchmark` checks the report and writes
   `performance-result.json`, following
   [`performance-result.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/performance-result.schema.json)
   (`schema_version` 5). It records the compilation identity, the script
   digest, the source revision, and the actual Core and platform versions. It
   never runs or times a model.

`scripts/run_model_tests.sh` then runs the model's
`tests/measure/test_performance.py` on each component's result and records
`passed`, `failed`, or `skipped` ([write the model's
tests](adding-a-model.md#10-write-the-models-tests)).

A separate hosted job validates the result set and publishes each result as a
`model_benchmark` run under the component's compile run. The run records the
performance stage key and carries a `performance-result` artifact, the
benchmark log, a `performance/model-synthetic` test result whose status is the
test outcome, and the measurements. A failed test still publishes the result
and completes its run, then fails the workflow run. The model browser reads
performance results with `schema_version` 1 through 5, shows measured values
with their units, and marks absent metrics unavailable. A failed measurement
keeps its workflow evidence and is not published as a successful measurement.
Performance and accuracy publish on their own outcomes: when one stage fails,
the other's result is still published, and the next push reruns only the stage
that has no run. A dispatched run stops before measuring when its branch has
since changed the tooling the stage keys hash; the newer push measures that
build instead.

Because the performance stage has its own key, editing `performance.py`, a
`benchmark` entry, or the DevKit runtime pin remeasures performance from the
published MPKs without recompiling. See [what a change
reruns](adding-a-model.md#9-know-what-a-change-reruns).
