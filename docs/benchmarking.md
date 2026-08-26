# Synthetic model benchmarking

The Models benchmark runner measures one compiled MPK on a Modalix DevKit using
deterministic synthetic inputs and the installed public `pyneat.Model.benchmark()`
API. It measures the logical model route represented by the MPK; it does not
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

Install the published package from Vulcan:

```bash
sima-cli neat install --stg models-utils/benchmarker@develop:latest
source ~/pyneat/bin/activate
benchmarker --version
```

The package declares `core@v0.4.0` in its own package metadata. Its installer
checks the DevKit platform, installs that Core package through `sima-cli` when
pyneat 0.4.0 is not already satisfied, and installs the benchmarker wheel into
the resulting `~/pyneat` environment.

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

The input follows [`performance-test-v1.schema.json`](../schemas/performance-test-v1.schema.json):

```json
{
  "schema_version": 1,
  "model": {
    "id": "resnet_50",
    "variant_id": "modalix_bf16",
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
    "models_commit": "<git SHA>",
    "compile_run_id": "<Vulcan run UUID>"
  }
}
```

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

## Run

```bash
benchmarker \
  --manifest /workspace/benchmark-test.json \
  --output /workspace/results/performance-result.json
```

The output follows
[`performance-result-v1.schema.json`](../schemas/performance-result-v1.schema.json)
and is written atomically. Average latency and throughput are available from
Neat 0.4.0. Percentiles are marked `unsupported` until the installed public
BenchmarkReport exposes them. Missing power telemetry is `unavailable`, never a
numeric zero. Every metric includes its mean, minimum, maximum, population
standard deviation, sample count, and an availability reason.

Power is total board PMBus rail power during Neat's throughput phase. It is not
isolated MLA power and no idle baseline is subtracted. When telemetry is
available, the report also derives energy per logical inference using the
compiled batch size.

Each result records a deterministic identity, resolved model route and output
topology, board/platform/kernel/Core information, CPU governor, and start/end
load and temperature observations. Invalid manifests and runtime failures also
produce schema-valid error reports at `--output`.

## Publish a result to Vulcan

The DevKit benchmarker never receives registry credentials. From a trusted host
with repository tooling and AWS session credentials, publish the validated
result as a benchmark run linked to its originating compile run:

```bash
model-registry publish-benchmark \
  --result /workspace/results/performance-result.json \
  --api-url "$VULCAN_ARTIFACT_REGISTRY_API_URL" \
  --aws-region us-west-2 \
  --bucket "$VULCAN_ARTIFACT_BUCKET" \
  --environment staging \
  --branch "$(git branch --show-current)" \
  --git-sha "$(git rev-parse HEAD)" \
  --compile-run-id '<Vulcan compile run UUID>' \
  --json
```

`--compile-run-id` may be omitted when the manifest supplied
`provenance.compile_run_id`. Publication revalidates the result schema and its
canonical fingerprint, creates an idempotent `model_benchmark` child run,
uploads the complete JSON report, records the measured values in Vulcan's
measurement index, and finalizes the child run. A separate run is required
because the originating `model_compile` run is immutable after finalization.

## Local development

Local tests use a fake public pyneat module and synthetic MPK fixtures, so they
run without Modalix hardware or a Linux ARM64 wheel:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/unit/test_benchmarker.py
```

Build the wheel and sima-cli/Vulcan package locally with:

```bash
scripts/build_benchmarker_vulcan_package.sh
```

Changes to the benchmarker source, schemas, dependency declaration, packaging,
or workflow trigger `.github/workflows/Vulcan-Benchmarker.yml`. Pull requests
build and verify the package without publishing. Trusted branch pushes publish
under the isolated `models-utils/benchmarker` Vulcan namespace and advance that branch's
benchmarker-specific `latest.tag`.

The future automated hardware acceptance step can run the same manifest and CLI
through the Kerrigan/Fleet Manager runner bridge. Until then, the DevKit returns
the JSON report and log for manual publication from a trusted host.
