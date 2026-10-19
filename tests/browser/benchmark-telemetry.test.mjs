import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {adaptBenchmarkResult, generativeWorkloads} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const fixture = () => JSON.parse(readFileSync(
  new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url), 'utf8'));

test('names the Sentinel source and window of a saved Core report', () => {
  const benchmark = adaptBenchmarkResult(fixture()).value;
  assert.equal(benchmark.telemetry.status, 'ok');
  assert.equal(benchmark.telemetry.samples, 15);
  assert.equal(benchmark.telemetry.intervalMs, 1999);
  assert.ok(benchmark.telemetry.windowMs >= 30000);
  assert.match(benchmark.telemetry.sentinelVersion, /\S/);
});

test('keeps the rail breakdown on peak SoM power', () => {
  const {measurements} = adaptBenchmarkResult(fixture()).value;
  const rails = measurements.peak_power_watts.details.breakdown;
  assert.ok(rails.length > 1);
  assert.ok(rails.every((rail) => rail.unit === 'W' && rail.mean <= rail.maximum));
  assert.equal(measurements.peak_die_temperature_c.details.breakdown, undefined);
  assert.equal(measurements.peak_die_temperature_c.unit, '°C');
});

test('reports no telemetry source when an older Core report has no telemetry', () => {
  const result = {...fixture(), schema_version: 5, metrics: {}, core_report: {}};
  assert.equal(adaptBenchmarkResult(result).value.telemetry, null);
});

test('leaves out what the saved Core report did not measure', () => {
  const result = fixture();
  result.benchmark_report.latency_p99_ms = 0;
  result.benchmark_report.telemetry.cpu_usage_pct.samples = 0;
  const {measurements} = adaptBenchmarkResult(result).value;
  assert.equal(measurements.latency_p99_ms, undefined);
  assert.equal(measurements.avg_cpu_usage_pct, undefined);
  assert.equal(measurements.throughput_fps.value, result.benchmark_report.fps);
});

test('names the model, build, and component a result measured', () => {
  const benchmark = adaptBenchmarkResult(fixture()).value;
  assert.equal(benchmark.modelId, 'resnet_50');
  assert.equal(benchmark.buildProfileId, 'w8a8');
  assert.equal(benchmark.componentId, 'model');
});

test('reads every published format and no unknown one', () => {
  const kinds = [1, 2, 3, 4, 5, 6, 7].map((version) => adaptBenchmarkResult({...fixture(), schema_version: version}).kind);
  assert.deepEqual(kinds, ['supported', 'supported', 'supported', 'supported', 'supported', 'supported', 'unsupported']);
});

test('GenAI workloads retain separate buckets and do not fabricate FPS or board telemetry', () => {
  const result = JSON.parse(readFileSync(
    new URL('../../artifact-browser/fixtures/genai-performance-result.json', import.meta.url), 'utf8'));
  const adapted = adaptBenchmarkResult(result);
  assert.equal(adapted.kind, 'supported');
  assert.equal(adapted.value.schemaVersion, 6);
  assert.deepEqual(adapted.value.workloads, result.benchmark_report.workloads);
  assert.deepEqual(adapted.value.measurements, {});
  assert.equal(adapted.value.telemetry, null);
  assert.equal(adapted.value.synthetic, false);
  result.provenance = {synthetic: true};
  assert.equal(adaptBenchmarkResult(result).value.synthetic, true);
});

test('comparison workloads sort text buckets numerically and retain their own metrics and telemetry', () => {
  const result = JSON.parse(readFileSync(
    new URL('../../artifact-browser/fixtures/genai-performance-result.json', import.meta.url), 'utf8'));
  result.benchmark_report.workloads.reverse();
  const workloads = generativeWorkloads(result);
  assert.deepEqual(workloads.slice(0, 6).map(workload => workload.inputTokens), [128, 256, 512, 1024, 2048, 3072]);
  const first = workloads[0];
  assert.equal(first.key, 'text_128');
  assert.equal(first.label, '128 input tokens');
  assert.ok(first.measurements.ttft_ms.value > 43 && first.measurements.ttft_ms.value < 44);
  assert.ok(first.measurements.avg_cpu_usage_pct.value > 0);
  assert.equal(first.cpuCores.length, 16);
  assert.equal(first.measurements.throughput_fps, undefined);
  const image = workloads.at(-1);
  assert.equal(image.key, 'image_448x448');
  assert.equal(image.kind, 'image_text');
  assert.equal(image.measurements.prefill_tps, undefined);
  assert.ok(image.measurements.ttft_ms.value > 132 && image.measurements.ttft_ms.value < 133);
});

test('comparison keeps image resolutions separate and ignores absent or unsupported reports', () => {
  assert.deepEqual(generativeWorkloads(null), []);
  assert.deepEqual(generativeWorkloads({schema_version: 99}), []);
  assert.deepEqual(generativeWorkloads(fixture()), []);
  const result = JSON.parse(readFileSync(
    new URL('../../artifact-browser/fixtures/genai-performance-result.json', import.meta.url), 'utf8'));
  const image = result.benchmark_report.workloads.find(workload => workload.kind === 'image_text');
  result.benchmark_report.workloads = [image, {...image, image_size: [224, 224]}];
  assert.deepEqual(generativeWorkloads(result).map(workload => workload.key), ['image_224x224', 'image_448x448']);
});

test('a selected GenAI workload uses the same board metrics, rails and CPU cores as Core', () => {
  const core = fixture();
  const workload = {telemetry: core.benchmark_report.telemetry,
    throughput_duration_ms: 16000, avg_power_watts: core.benchmark_report.avg_power_watts,
    metrics: {ttft_ms: {value: 43.0, unit: 'ms', sample_count: 5}}};
  const {value} = adaptBenchmarkResult({...core, benchmark_type: 'model.generative', benchmark_report: workload});
  assert.equal(value.measurements.ttft_ms.value, 43.0);
  assert.equal(value.measurements.throughput_fps, undefined);
  assert.equal(value.measurements.avg_cpu_usage_pct.value, core.benchmark_report.telemetry.cpu_usage_pct.mean);
  assert.equal(value.cpuCores.length, core.benchmark_report.telemetry.cpu_cores_usage_pct.length);
  assert.equal(value.telemetry.windowMs, 16000);
  assert.equal(value.measurements.peak_power_watts.details.breakdown.length,
    core.benchmark_report.telemetry.power_rails_watts.length);
});
