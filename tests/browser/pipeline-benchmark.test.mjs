import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {adaptPipelineResult} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const fixture = () => JSON.parse(readFileSync(
  new URL('../../artifact-browser/fixtures/pipeline-result.json', import.meta.url), 'utf8'));

test('reads the pipeline report as the saved Core report it mirrors', () => {
  const result = fixture();
  const {measurements, telemetry, samples, cpuStage} = adaptPipelineResult(result).value;
  const report = result.benchmark_report;
  assert.equal(measurements.throughput_fps.value, report.fps);
  assert.equal(measurements.latency_avg_ms.value, report.latency_ms);
  // Core's convention: energy over the configured samples at the measured rate, so per inference is power / rate.
  assert.ok(Math.abs(measurements.energy_per_inference_joules.value - report.avg_power_watts / report.fps) < 1e-9);
  assert.equal(telemetry.status, 'ok');
  assert.equal(samples, report.throughput_samples);
  assert.deepEqual(cpuStage, result.cpu_stage);
});

test('does not render a pipeline format it does not know', () => {
  assert.deepEqual(adaptPipelineResult({...fixture(), schema_version: 2}), {kind: 'unsupported', schemaVersion: 2});
  assert.deepEqual(adaptPipelineResult(undefined), {kind: 'missing'});
});
