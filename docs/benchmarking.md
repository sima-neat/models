# Synthetic model benchmarking

The Models benchmark runner measures one compiled MPK on a Modalix DevKit using
deterministic synthetic inputs and public pyneat APIs. It measures the logical
model route represented by the MPK; it does not
measure camera or network ingest, application decode/NMS, overlays, Insight, or
model accuracy.

## Compatibility

The first runner release is pinned to:

| Component | Version |
| --- | --- |
| Platform | 2.1.3 |
| Neat and pyneat | 0.4.0 |
| MPK Model SDK | 2.1.3 |

The installer rejects a different DevKit `DISTRO_VERSION` or pyneat base
version. A local build such as `0.4.0+develop.<sha>` is accepted because its
public API base version is 0.4.0.

An MPK may contain more than one ELF. The runner verifies every MLA stage's ELF
reference and passes the complete archive to `pyneat.Model`; Neat therefore
benchmarks the resolved graph end to end instead of choosing one ELF. The JSON
records the ELF and MLA-stage counts and aggregate model metrics. With
`include_plugin_latency`, Neat also collects per-plugin timing detail in the
DevKit log. An MPK with no MLA stage, a missing referenced ELF, or inconsistent
compiled batch sizes fails validation.

## Install on a DevKit

Install the benchmarker package on the DevKit:

```bash
sima-cli neat install --stg models-utils/benchmarker@develop:latest
source ~/pyneat/bin/activate
benchmarker --version
```

The package declares `core@v0.4.0` in its own package metadata. Its installer
checks the DevKit platform, installs that Core package through `sima-cli` when
pyneat 0.4.0 is not already satisfied, and installs the benchmarker wheel into
the resulting `~/pyneat` environment. Verify the installed benchmarker and
pyneat versions before execution.

For a downloaded package payload, the equivalent manual installation is:

```bash
sima-cli neat install core@v0.4.0
~/pyneat/bin/python -m pip install --no-deps --force-reinstall \
  sima_neat_benchmarker-*.whl
source ~/pyneat/bin/activate
```

The package installation finishes only after importing both benchmarker and
pyneat and verifying their versions.

Neat 0.4.0 reads Modalix PMBus rails through `/dev/i2c-3` or `/dev/i2c-4`.
The runtime user must have read/write access, normally through the `i2c` group.
The benchmark still measures latency and throughput without that access, but it
prints an actionable warning and records power metrics as `unavailable` with a
machine-readable reason. Group changes require a fresh login session.

## Create a benchmark manifest

The input follows
[`performance-test-v1.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/performance-test-v1.schema.json):

```json
{
  "schema_version": 1,
  "model": {
    "id": "resnet_50",
    "build_profile_id": "modalix_bf16",
    "component_id": "model",
    "path": "/workspace/models/resnet_50_modalix_bf16_mpk.tar.gz",
    "sha256": "<64 lowercase hexadecimal characters>"
  },
  "compatibility": {
    "platform_version": "2.1.3",
    "neat_version": "0.4.0"
  },
  "benchmark": {
    "samples": 1000,
    "trials": 3,
    "include_plugin_latency": false
  },
  "provenance": {
    "models_commit": "<git SHA>"
  }
}
```

For a local benchmark, keep the local Git SHA when available and record any
uncommitted state separately. Automation may add registry provenance later; a
model author does not need or create that publication identity.

Model options are declarative. Enum values use an explicit public pyneat
reference; ordinary strings remain ordinary strings:

```json
{
  "model_options": {
    "decode_type": "pyneat:BoxDecodeType.YoloV26",
    "top_k": 100
  }
}
```

Raw/default MPK routes should omit `model_options`. Route-specific measurements
must use distinct benchmark manifests and result identities.

When `trials` is omitted, the runner uses three trials.

Models whose inference time makes `Model.benchmark()`'s fixed internal warmup
budget impractical can opt into a bounded public `ModelRunner` measurement in
their `tests/performance.yaml`:

```yaml
benchmark:
  mode: bounded_runner
  warmup_samples: 1
  samples: 5
  trials: 3
  original_width: 1280
  original_height: 720
```

Bounded mode executes exactly the configured warmups, latency samples, and
throughput samples per trial. It uses synthetic tensors matching the MPK input
specifications and records measured latency percentiles, throughput, board
power, and energy. For image inputs with dynamic spatial dimensions,
`original_width` and `original_height` materialize representative source
geometry before preprocessing. Inputs with static spatial dimensions continue
to use the shape declared by the MPK.

## Run

```bash
benchmarker \
  --manifest /workspace/benchmark-test.json \
  --output /workspace/results/performance-result.json
```

Run this against the exact MPK that passed local compiler validation. Preserve
the JSON result and benchmark log outside Git. Local results are developer
evidence; they are not registry publications.

The output follows
[`performance-result-v4.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/performance-result-v4.schema.json)
and is written atomically. Average latency and throughput are available from
Neat 0.4.0. Percentiles are marked `unsupported` for the default
`Model.benchmark()` path; bounded runner measurements include them. Missing
power telemetry is `unavailable`, never a numeric zero. Every metric includes
its aggregation, minimum, maximum, population standard deviation, sample count,
and an availability reason. The v3 metric map is additive: new measurements
that follow the common metric contract do not require a result-schema bump.

Startup is measured once before the steady-state trials with a public
`ModelRunner`. `model_load_ms` covers `pyneat.Model(...)` construction.
`time_to_first_inference_ms` uses the same monotonic start and ends when the
first synchronous `runner.run(...)` returns, so it also includes synthetic input
preparation and graph build. The result records the source MPK path and its
mount/device when Linux exposes them. Filesystem and extracted-package caches
are not flushed, and checksum validation reads the MPK before the timer starts,
so the cache state is explicitly `uncontrolled_after_sha256_validation`.
Compare runs only when their storage placement and cache preparation are
equivalent. The startup runner explicitly enables Neat startup preflight and
records that choice in the metric details. A model that cannot use this generic
startup probe reports time to first inference as `unavailable` without
invalidating its steady-state benchmark.

Power is total board PMBus rail power during Neat's throughput phase. It is not
isolated MLA power and no idle baseline is subtracted. When telemetry is
available, the report also derives energy per logical inference using the
compiled batch size.

MLA memory is sampled every 100 ms from before model construction until all
trials finish. The benchmarker uses the same allocator summary as Sentinel:
the hexadecimal `Total allocated size` value from `/dev/simaai-mem`. It reports
the baseline, absolute peak, final value, and peak increase over the baseline
in MiB. `mla_memory_peak_delta_mib` is the primary comparison measurement.
This is board-wide allocator usage, not process-attributed model memory or MLA
compute utilization. Keep automated runners workload-isolated and inspect the
recorded baseline when comparing runs. Device, permission, and parsing failures
make only the MLA measurements unavailable; they do not invalidate latency or
throughput results.

Each result records a deterministic identity, resolved model route and output
topology, board/platform/kernel/Core information, CPU governor, and start/end
load and temperature observations. Schema-v3 results also record a timestamped
temperature sample at benchmark start, after every completed trial, and at
benchmark end. Temperature readings include the hwmon sensor ID and source so
duplicate kernel labels such as `temp1` remain distinguishable. Invalid
manifests and runtime failures also produce schema-valid error reports at
`--output`.

## Publication boundary

Model authors do not publish benchmark results directly. The DevKit benchmarker
has no registry credentials, and `model-registry publish-benchmark` is reserved
for CI. Keep local results outside Git and include only a concise summary with
the contribution.

## Develop the benchmarker itself

This section applies only when changing the benchmarker implementation, not
when adding or testing a model.

Local tests use a fake public pyneat module and synthetic MPK fixtures, so they
run without Modalix hardware or a Linux ARM64 wheel:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/unit/test_benchmarker.py
```

Build its package locally with:

```bash
scripts/build_benchmarker_vulcan_package.sh
```

Packaging and publication workflow details belong to repository automation and
are outside this contributor guide.
