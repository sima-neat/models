import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {adaptBenchmarkResult} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const fixture = () => JSON.parse(readFileSync(
  new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url), 'utf8'));

test('names the Sentinel source and window of a schema-5 result', () => {
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

test('reports no telemetry source when the Core report has no telemetry', () => {
  const result = fixture();
  delete result.core_report.telemetry;
  assert.equal(adaptBenchmarkResult(result).value.telemetry, null);
});

test('names the model, build, and component a result measured', () => {
  const benchmark = adaptBenchmarkResult(fixture()).value;
  assert.equal(benchmark.modelId, 'resnet_50');
  assert.equal(benchmark.buildProfileId, 'w8a8');
  assert.equal(benchmark.componentId, 'model');
});

test('reads every published format and no unknown one', () => {
  const kinds = [1, 2, 3, 4, 5, 6].map((version) => adaptBenchmarkResult({...fixture(), schema_version: version}).kind);
  assert.deepEqual(kinds, ['supported', 'supported', 'supported', 'supported', 'supported', 'unsupported']);
});
