import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {analyzeLayers, selectLayers, layerTimeScale, layerFamily} from '../../artifact-browser/mla-layer-stats.js';
import {readPerformance} from '../../artifact-browser/model-report.js';

const row = (name, time, stage = 0) => ({stage_index: stage, name, latency_us: time});
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

test('a model report exposes the layer analysis of each component', () => {
  const report = JSON.parse(readFileSync(new URL('../../artifact-browser/fixtures/model-report-performance.json', import.meta.url)));
  report.components.model.mla_layer_stats = {status: 'ok', layers: rows};
  assert.equal(readPerformance({report}).value.components[0].layerAnalysis.totalUs, 70);
  report.components.model.mla_layer_stats = {status: 'not requested', layers: []};
  assert.equal(readPerformance({report}).value.components[0].layerAnalysis, null);
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

test('minimum execution time combines with search without changing totals or hiding zeros by default', () => {
  const analysis = analyzeLayers(rows);
  const selection = selectLayers(analysis, {query: "conv", minUs: 21});
  assert.deepEqual(selection.layers.map(layer => layer.order), [3]);
  assert.equal(selection.totalUs, 70);
  assert.equal(selection.count, 5);
  assert.equal(selectLayers(analysis, {minUs: 0}).zeroCount, 2);
  assert.equal(selectLayers(analysis, {minUs: 31}).layers.length, 0);
});

test('operation families follow AFE composite names and tolerate unknown names', () => {
  assert.equal(layerFamily('MLA_0/conv2d_add_relu'), 'conv');
  assert.equal(layerFamily('MLA_0/tuple_einsum_307'), 'matmul');
  assert.equal(layerFamily('pool'), 'pool');
  assert.equal(layerFamily('swish'), 'activation');
  assert.equal(layerFamily('concat'), 'layout');
  assert.equal(layerFamily('placeholder'), 'io');
  assert.equal(layerFamily('reshape'), 'layout');
  assert.equal(layerFamily('MLA_0/add_relu_5'), 'elementwise');
  assert.equal(layerFamily('MLA_0/multiply_100'), 'elementwise');
  assert.equal(layerFamily('MLA_0/softmax_110'), 'normalization');
  assert.equal(layerFamily('MLA_0/image.resize2d_57/0'), 'resize');
  assert.equal(layerFamily('MLA_0/sum_10'), 'reduction');
  assert.equal(layerFamily('MLA_0/qnn_conv2d_bias_10'), 'conv');
  assert.equal(layerFamily('MLA_0/qnn_add_12'), 'elementwise');
  assert.equal(layerFamily('MLA_0/block_0/conv2d'), 'conv');
  assert.equal(layerFamily('MLA_0/block_0/activation'), 'activation');
  assert.equal(layerFamily('unrecognized_conv_name'), 'other');
});

test('N2A helpers are classified by their operation, not producer or consumer names', () => {
  for (const name of [
    'slice_MLA_0/transpose_301_0_MLA_0/tuple_einsum_304_0',
    'slice_MLA_0/swish_1_0_MLA_0/conv2d_add_2_1',
    'concat_MLA_0/conv2d_add_0_MLA_0/swish_1_0',
    'batch_concat_MLA_0/conv2d_add_2_off0_sz4',
    'MLA_0/tuple_einsum_304_batch_slice_from_MLA_0/transpose_301_0',
    'concat_MLA_0/softmax_110_MLA_0/softmax_110_output',
  ]) assert.equal(layerFamily(name), 'layout', name);
  assert.equal(layerFamily('MLA_0/placeholder_0_0_MLA_0/conv2d_add_relu_0_0'), 'io');
  assert.equal(layerFamily('MLA_0/conv2d_add_374_output'), 'io');
  assert.equal(layerFamily('MLA_0/softmax_110_output'), 'io');
  assert.equal(layerFamily('MLA_0/conv2d_add_374_output_segment_0'), 'io');
});

test('segments and nested reshape transformations retain the parent operation family', () => {
  assert.equal(layerFamily('MLA_0/tuple_einsum_104_segment_0'), 'matmul');
  assert.equal(layerFamily('MLA_0/softmax_105_batch_segment_1'), 'normalization');
  assert.equal(layerFamily('MLA_0/reshape_94/reshape_channel/split_and_concat'), 'layout');
  assert.equal(layerFamily('MLA_0/reshape_94/reshape_spatial/squeeze_idim2'), 'layout');
  assert.equal(layerFamily('MLA_0/conv2d_add_10_output_reshape'), 'layout');
});
