import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {analyzeLayers, selectLayers, layerTimeScale} from '../../artifact-browser/layer-stats.js';
import {adaptBenchmarkResult} from '../../artifact-browser/benchmark-adapters.js';

const row = (name, time, stage = 0) => ({name, total_execution_time_us: time, stage_index: stage});
const rows = [row('conv', 20), row('noop', 0), row('conv', 30, 2), row('squeeze', 0, 2), row('tail', 20)];

test('preserves compiler order, duplicate names, zero times and gaps between MLA stages', () => {
  const analysis = analyzeLayers(rows);
  assert.equal(analysis.totalUs, 70);
  assert.deepEqual(analysis.stages, [{index: 0, timeUs: 40, count: 3}, {index: 2, timeUs: 30, count: 2}]);
  assert.deepEqual(analysis.layers.map(layer => layer.order), [1, 2, 3, 4, 5]);
  assert.deepEqual(selectLayers(analysis).layers.map(layer => layer.order), [3, 1, 5, 2, 4]);
  assert.deepEqual(selectLayers(analysis, {order: 'compiler'}).layers.map(layer => layer.order), [1, 2, 3, 4, 5]);
  assert.deepEqual(analysis.layers.map(layer => layer.order), [1, 2, 3, 4, 5]);
});

test('search never changes the component percentage denominator', () => {
  const analysis = analyzeLayers(rows);
  const selected = selectLayers(analysis, {query: ' CoNV '});
  assert.deepEqual(selected.layers.map(layer => layer.name), ['conv', 'conv']);
  assert.equal(selected.totalUs, 70);
  assert.equal(selected.count, 5);
  assert.equal(selected.zeroCount, 2);
  assert.equal(selected.top10Share, 100);
  assert.equal(selectLayers(analysis, {query: 'missing'}).layers.length, 0);
});

test('top ten share includes ten layers only', () => {
  const analysis = analyzeLayers(Array.from({length: 20}, (_, i) => row(`layer${i}`, 1)));
  assert.equal(selectLayers(analysis).top10Share, 50);
});

test('all-zero profiles produce finite summaries', () => {
  const analysis = analyzeLayers([row('noop', 0)]);
  assert.equal(selectLayers(analysis).top10Share, 0);
  assert.equal(selectLayers(analysis).zeroCount, 1);
});

test('missing or malformed data never produces a partially rendered misleading profile', () => {
  for (const value of [undefined, null, [], {}, [null], [row('', 1)], [row('x', -1)], [row('x', NaN)],
    [row('x', Infinity)], [row('x', '1')], [row('x', 1, -1)], [row('x', 1, 0.5)],
    [row('x', 1, '0')], [row('x', 1), row('bad', null)], [row('x', Number.MAX_VALUE), row('y', Number.MAX_VALUE)]]) {
    assert.equal(analyzeLayers(value), null);
  }
});

test('adapters expose layer analysis for saved reports and historical core reports', () => {
  const result = JSON.parse(readFileSync(new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url)));
  result.benchmark_report.layer_stats = rows;
  const current = adaptBenchmarkResult(result);
  assert.equal(current.value.layerAnalysis.totalUs, 70);
  const old = {...result, schema_version: 5, core_report: result.benchmark_report, metrics: {}};
  assert.equal(adaptBenchmarkResult(old).value.layerAnalysis.totalUs, 70);
  delete old.core_report.layer_stats;
  assert.equal(adaptBenchmarkResult(old).value.layerAnalysis, null);
});

test('timing bars have a fixed, finite microsecond axis independent of the search', () => {
  const analysis = analyzeLayers([row('slow', 578.71), row('fast', 20)]);
  assert.equal(layerTimeScale(analysis), 600);
  selectLayers(analysis, {query: 'fast'});
  assert.equal(layerTimeScale(analysis), 600);
  assert.equal(layerTimeScale(analyzeLayers([row('noop', 0)])), 1);
  assert.equal(layerTimeScale(analyzeLayers([row('small', 0.04)])), 0.04);
  assert.equal(layerTimeScale(analyzeLayers([row('large', Number.MAX_VALUE)])), Number.MAX_VALUE);
});
