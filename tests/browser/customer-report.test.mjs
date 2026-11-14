import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {CUSTOMER_HIDDEN_METRICS, customerPerformanceReport, readPerformance} = await import(
  new URL('../../artifact-browser/model-report.js', import.meta.url));
const fixture = () => JSON.parse(readFileSync(
  new URL('../../artifact-browser/fixtures/model-report-performance.json', import.meta.url), 'utf8'));
const run = {id: 'run-1', metadata: {model_id: 'resnet_50', build_profile_id: 'w8a8', stage: 'performance'}};
const read = (report) => readPerformance({run, parameters: {}, report}).value;

test('the customer report leaves out Linux memory', () => {
  const developer = read(fixture()).measurements;
  for (const key of CUSTOMER_HIDDEN_METRICS) assert.ok(developer[key], `the fixture measures ${key}`);

  const report = customerPerformanceReport(fixture());
  const customer = read(report).measurements;
  for (const key of CUSTOMER_HIDDEN_METRICS) assert.equal(customer[key], undefined, key);
  assert.equal(report.telemetry.linux_memory_used_mib, undefined);
});

test('the customer report keeps SoM power, its rails, and the energy computed from it', () => {
  const developer = read(fixture()).measurements;
  const customer = read(customerPerformanceReport(fixture())).measurements;
  assert.ok(customer.peak_power_watts.details.breakdown.length > 1);
  for (const key of ['avg_power_watts', 'peak_power_watts', 'energy_per_inference_joules']) {
    assert.deepEqual(customer[key], developer[key], key);
  }
});

test('the customer report keeps every other reading as the developer view shows it', () => {
  const developer = read(fixture()).measurements;
  const customer = read(customerPerformanceReport(fixture())).measurements;
  const kept = Object.keys(developer).filter((key) => !CUSTOMER_HIDDEN_METRICS.includes(key));
  assert.deepEqual(Object.keys(customer).sort(), kept.sort());
  for (const key of ['throughput_fps', 'avg_mla_power_watts', 'peak_mla_power_watts', 'mla_memory_peak_mib']) {
    assert.deepEqual(customer[key], developer[key], key);
  }
});

test('the customer report filters each independently measured component', () => {
  const {metrics, telemetry, workload} = fixture();
  const component = {metrics, telemetry, workload, execution_plan: null,
    mla_layer_stats: {status: 'not requested', layers: []}, mla_stage_stats: {status: 'not requested', stages: []}};
  const report = customerPerformanceReport({schema: 'sima.model-report', schema_version: 1, stage: 'performance',
    runtime: {pyneat: 'test'}, scope: 'components', components: {image_encoder: component, text_encoder: component}});
  for (const {measurements} of read(report).components) {
    assert.ok(measurements.avg_power_watts);
    assert.equal(measurements.peak_linux_memory_mib, undefined);
    assert.equal(measurements.throughput_fps.value, metrics.throughput_fps);
  }
});

test('the customer report filters generation workloads and their baselines', () => {
  const telemetry = fixture().telemetry;
  const workload = {id: 'text_128', kind: 'text', input_tokens: 128, telemetry,
    metrics: {decode_tps: {value: 12}, avg_power_watts: {value: 9}, energy_per_inference_joules: {value: 1}}};
  const report = customerPerformanceReport({...fixture(), generation: {workloads: [{...workload, baseline: workload}]}});
  const [measured] = read(report).workloads;
  for (const {measurements} of [measured, measured.baseline]) {
    assert.equal(measurements.decode_tps.value, 12);
    assert.ok(measurements.avg_power_watts);
    for (const key of CUSTOMER_HIDDEN_METRICS) assert.equal(measurements[key], undefined, key);
  }
});
