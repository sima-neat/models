import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {adaptBenchmarkResult} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const fixture = (version) => JSON.parse(readFileSync(
  new URL(`../../artifact-browser/fixtures/performance-result-v${version}.json`, import.meta.url), 'utf8'));

test('names the Sentinel source and window of a schema-5 result', () => {
  const benchmark = adaptBenchmarkResult(fixture(5)).value;
  assert.equal(benchmark.telemetry.status, 'ok');
  assert.equal(benchmark.telemetry.samples, 15);
  assert.equal(benchmark.telemetry.intervalMs, 1999);
  assert.ok(benchmark.telemetry.windowMs >= 30000);
  assert.match(benchmark.telemetry.sentinelVersion, /\S/);
});

test('keeps the rail breakdown on peak SoM power', () => {
  const {measurements} = adaptBenchmarkResult(fixture(5)).value;
  const rails = measurements.peak_power_watts.details.breakdown;
  assert.ok(rails.length > 1);
  assert.ok(rails.every((rail) => rail.unit === 'W' && rail.mean <= rail.maximum));
  assert.equal(measurements.peak_die_temperature_c.details.breakdown, undefined);
  assert.equal(measurements.peak_die_temperature_c.unit, '°C');
});

test('reports no telemetry source for a result from before Sentinel', () => {
  assert.equal(adaptBenchmarkResult(fixture(4)).value.telemetry, null);
});
