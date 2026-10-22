import assert from 'node:assert/strict';
import {test} from 'node:test';
import {matchMlaStages, mlaStageSimulation, mlaUtilizationMetrics} from '../../artifact-browser/mla-utilization.js';

// resnet_50 w8a8: mla-msim's counts for its one MLA stage, and a board latency for it.
const simulated = {status: 'ok', stages: [{stage_index: 0, effective_ops: 8213719016, total_ops: 10445905152, simulated_latency_us: 678.432}]};
const measured = {status: 'ok', stages: [{stage_index: 0, latency_us: 821.4}]};

test('finds the simulation among the package run test results', () => {
  const summary = {schema_version: 1, components: {model: simulated}};
  const tests = [{suite: 'correctness', name: 'numerical-fidelity', summary: {schema_version: 1}},
    {suite: 'performance', name: 'mla-stage-simulation', summary}];
  assert.equal(mlaStageSimulation(tests), summary);
  assert.equal(mlaStageSimulation([]), null);
  assert.equal(mlaStageSimulation(undefined), null);
});

test('pairs measured and simulated stages by stage index', () => {
  assert.deepEqual(matchMlaStages(measured, simulated).stages, [
    {index: 0, measuredUs: 821.4, simulatedUs: 678.432, effectiveOps: 8213719016, totalOps: 10445905152},
  ]);
});

test('says why stages cannot be compared', () => {
  assert.equal(matchMlaStages({status: 'mla-rt not found', stages: []}, simulated).reason, 'Not measured: mla-rt not found');
  assert.equal(matchMlaStages(undefined, simulated).reason, 'Not measured: no MLA stage stats');
  assert.equal(matchMlaStages(measured, {status: 'mla-msim not found', stages: []}).reason, 'Not simulated: mla-msim not found');
  assert.equal(matchMlaStages(measured, undefined).reason, 'Not simulated: no MLA stage simulation');
  const shifted = {status: 'ok', stages: [{stage_index: 2, latency_us: 821.4}]};
  assert.equal(matchMlaStages(shifted, simulated).reason, 'The board and the simulator report different MLA stages.');
  assert.equal(matchMlaStages(shifted, simulated).stages, undefined);
});

test('rates each stage from its measured latency, with utilization at INT8 only', () => {
  const {stages} = matchMlaStages(measured, simulated);
  const [stage] = mlaUtilizationMetrics(stages, {int8: true}).stages;
  assert.ok(Math.abs(stage.effectiveTops - 10.0) < 0.01);
  assert.ok(Math.abs(stage.packing - 0.7863) < 0.0001);
  assert.ok(Math.abs(stage.utilization - 0.2) < 0.001);
  assert.equal(mlaUtilizationMetrics(stages).stages[0].utilization, null);
});

test('totals the stages of a component', () => {
  const stages = [
    {index: 0, measuredUs: 100, simulatedUs: 90, effectiveOps: 1e9, totalOps: 2e9},
    {index: 2, measuredUs: 300, simulatedUs: 250, effectiveOps: 3e9, totalOps: 4e9},
    {index: 3, measuredUs: 0, simulatedUs: 0, effectiveOps: 0, totalOps: 0},
  ];
  const {stages: rated, total} = mlaUtilizationMetrics(stages, {int8: true});
  assert.deepEqual(rated.map(stage => stage.effectiveTops), [10, 10, null]);
  assert.equal(rated[2].packing, null);
  assert.deepEqual([total.measuredUs, total.simulatedUs, total.effectiveOps, total.totalOps], [400, 340, 4e9, 6e9]);
  assert.equal(total.effectiveTops, 10);
  assert.equal(total.utilization, 0.2);
});
