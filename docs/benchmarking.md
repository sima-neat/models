# Synthetic model benchmarking

Each model owns `scripts/performance.py`. The script loads the compiled MPK through
`pyneat.Model` and calls Core's benchmark API. Core owns synthetic inputs,
warmup, latency and throughput measurement, power telemetry, and runner lifetime.
There is no shared benchmark executor or model-specific timing loop.

## Install on a DevKit

Use the Core install target in `deps/manifest.json`. CI verifies Platform 3.0.0
before installing the `release-3.0.0-prep` package. The scripts use
the existing `BenchmarkReport` fields; no Core extension is required.

```bash
sima-cli neat install --env production '<neat-core.install_target>'
source ~/pyneat/bin/activate
python -c 'import pyneat; print(pyneat.__version__, pyneat.__file__)'
```

## Run a model

Run the model's own script on the DevKit, supplying its MPK and output report:

```bash
python models/resnet_50/scripts/performance.py \
  /workspace/resnet_50_mpk.tar.gz /workspace/core-benchmark.json
```

Every script follows the same pattern:

```python
import json
import sys

import pyneat

model = pyneat.Model(sys.argv[1])
report = model.benchmark()
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump({
        "core_version": pyneat.__version__,
        "latency_ms": report.latency_ms,
        "fps": report.fps,
        "avg_power_watts": report.avg_power_watts,
        "energy_joules": report.energy_joules,
    }, output, allow_nan=False)
```

A failed call exits unsuccessfully. Preserve the log; a missing report is not a
successful benchmark. Keep model archives and results outside Git.

## Workload and measurements

The workload is the default route selected by Core from the MPK, using Core's
synthetic input and benchmark defaults. The raw report contains measurements.
CI records the input shapes, batch size, preprocessing, postprocessing, and default
sample configuration separately in its manifest using Core's model-inspection API.
Model scripts must use these default model and benchmark options.
RF-DETR's backbone and transformer each produce a separate result. Their numbers
are not a measured combined pipeline.

This changes the previous YOLO workload, which explicitly configured image
preprocessing and BoxDecode. Default-MPK results are a new baseline and must not
be treated as equivalent to those historical fused-pipeline measurements.

Core reports mean latency, throughput in logical inferences per second, average
power, and total energy. The registry derives energy per logical inference by
dividing total energy by the default sample count times the inspected compiled
batch size. Missing power remains unavailable.
RF-DETR still requires measured power and energy per inference for publication.

Percentiles, initialization timings, peak MLA memory, and temperature traces are
not exported by this Core API. New registry results mark those metrics unavailable;
they are not copied from old results or estimated. Historical reports retain their
original fields. Adding these measurements belongs in Core before Models consumes
them.

## CI and registry publication

Each published compilation dispatches its benchmark on the branch it was built
from, so the benchmark workflow and `model-registry` come from the same branch
as the build. A benchmark run only accepts builds from its own branch.

CI verifies the dispatched component set and compiled archive digests, installs
the pinned Core package, then invokes each model's script once per component.
`model-registry import-core-benchmark` validates the raw Core report and packages
it with compilation identity, script digest, source revision, runner build, and
actual Core/Internals package versions. The CI metadata step loads the same MPK
with default options to inspect its workload; it does not run inference. The
importer never executes or times a model. A model with a `devkit_script`
validation entry then measures its task accuracy with the same MPKs; see
[Accuracy on the DevKit](local-accuracy-development.md#accuracy-on-the-devkit).

Model authors do not publish benchmark results directly.
The board has no registry publication credentials. A separate hosted job validates
the result set and publishes each schema-v5 report and log as a benchmark child
of its immutable compilation. The script's JSON export is embedded unchanged in that
result. Failures retain the available workflow evidence and are not published as
successful measurements.

The registry UI reads schemas 1–5, displays measured values with their units,
marks absent metrics unavailable, and labels workloads in details and comparisons.
A script or runtime dependency change invalidates benchmark fingerprints without
forcing model recompilation. CI can restore the existing immutable MPKs for the
new benchmark run.
