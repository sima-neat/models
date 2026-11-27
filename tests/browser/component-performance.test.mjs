import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {readPerformance} from '../../artifact-browser/model-report.js';

const source = readFileSync(new URL('../../artifact-browser/model-view.js', import.meta.url), 'utf8');
const performanceSource = readFileSync(new URL('../../artifact-browser/performance-section.js', import.meta.url), 'utf8');
const start = source.indexOf('function modelPerformance(');
const end = source.indexOf('\nfunction componentPrecisions(', start);
const render = new Function('readPerformance', 'pipelineOrder', 'humanize', 'componentPanels', 'performanceSummary',
  'layerAnalysisSection', 'renderMlaUtilization', 'mlaStageSimulation', 'componentPrecisions', 'executionGraphSection',
  `return (${source.slice(start, end)}\n);`);
const stats = {execution_plan: null, mla_layer_stats: {status: 'not requested', layers: []},
  mla_stage_stats: {status: 'not requested', stages: []}};
const base = {schema: 'sima.model-report', schema_version: 1, stage: 'performance', runtime: {pyneat: 'test'}};
const humanize = id => id.replaceAll('_', ' ');

function page(report) {
  const summaries = [], panels = [];
  const modelPerformance = render(readPerformance, () => ['image_encoder', 'text_encoder'], humanize,
    (labels, markup) => {panels.push(labels); return markup.join('');},
    (result, run, modelId, componentId = '') => {
      summaries.push({rate: result.measurements.throughput_fps.value, componentId,
        componentCount: result.components.length, workload: result.workload});
      return String(result.measurements.throughput_fps.value);
    }, () => '', () => '', () => null, () => ({}), () => '');
  const html = modelPerformance({performance: {run: {id: 'run'}, report}}, 'clip');
  return {html, summaries, panels};
}

test('the page presents each independent component with its own readings and workload', () => {
  const result = page({...base, scope: 'components', components: {
    text_encoder: {...stats, metrics: {throughput_fps: 640}, workload: '77 token embeddings', telemetry: {}},
    image_encoder: {...stats, metrics: {throughput_fps: 230}, workload: '720p input', telemetry: {}},
  }});
  assert.deepEqual(result.panels, [['image encoder', 'text encoder']]);
  assert.deepEqual(result.summaries, [
    {rate: 230, componentId: 'image_encoder', componentCount: 1, workload: '720p input'},
    {rate: 640, componentId: 'text_encoder', componentCount: 1, workload: '77 token embeddings'},
  ]);
});

test('a composed two-component model keeps one combined performance summary', () => {
  const result = page({...base, metrics: {throughput_fps: 160}, workload: 'composed pipeline', telemetry: {},
    components: {backbone: stats, decoder: stats}});
  assert.deepEqual(result.panels, []);
  assert.deepEqual(result.summaries, [{rate: 160, componentId: '', componentCount: 2, workload: 'composed pipeline'}]);
});

test('comparison does not turn independent component rates into whole-model throughput', () => {
  const compare = readFileSync(new URL('../../artifact-browser/compare-view.js', import.meta.url), 'utf8');
  const begin = compare.indexOf('function variantMeasurements(');
  const finish = compare.indexOf('\nfunction bestValue(', begin);
  const measurements = new Function('readPerformance', `return (${compare.slice(begin, finish)});`)(readPerformance);
  const independent = {...base, scope: 'components', components: {
    text_encoder: {...stats, metrics: {throughput_fps: 440}, workload: 'one prompt', telemetry: {}},
    detector: {...stats, metrics: {throughput_fps: 100}, workload: 'one image', telemetry: {}},
  }};
  assert.deepEqual(measurements({performance: {report: independent}}), {});
  const whole = {...base, metrics: {throughput_fps: 100}, workload: 'one image', telemetry: {}, components: {model: stats}};
  assert.equal(measurements({performance: {report: whole}}).throughput_fps.value, 100);
});

test('generation exports keep the deployment identity, workload and decoding role', () => {
  const fixture = JSON.parse(readFileSync(new URL('../../artifact-browser/fixtures/genai-model-report.json', import.meta.url), 'utf8'));
  const run = {id: 'run', metadata: {model_id: 'gemma_4_e2b_it_textonly_mtp', build_profile_id: 'w4a16'}};
  const baseline = {model_id: 'gemma_4_e2b_it_textonly', build_profile_id: 'w8a16'};
  const report = {...fixture, generation: {...fixture.generation,
    settings: {...fixture.generation.settings, decoding: 'speculative'}, baseline,
    workloads: fixture.preview_speculative_workloads}};
  const performance = readPerformance({run, report}).value;
  const exports = [];
  // generativeBenchmark is the last function of the Performance section.
  const begin = performanceSource.indexOf('function generativeBenchmark(');
  const renderGeneration = new Function('performanceDetails', 'infoMark', 'escapeHtml', 'disclosure',
    `return (${performanceSource.slice(begin)}\n);`)(
    reading => {exports.push(reading); return {markup: ''};}, () => '', String, () => '');
  const fileBegin = performanceSource.indexOf('function csvFileName(');
  const fileEnd = performanceSource.indexOf('\nfunction performanceContext(', fileBegin);
  const filename = new Function(`return (${performanceSource.slice(fileBegin, fileEnd)});`)();
  renderGeneration(performance, run, 'gemma4_mtp');
  for (const [index, workload] of performance.workloads.entries()) {
    const [target, ordinary] = exports.slice(index * 2, index * 2 + 2);
    assert.equal(target.modelId, run.metadata.model_id);
    assert.equal(target.buildProfileId, 'w4a16');
    assert.equal(ordinary.modelId, baseline.model_id);
    assert.equal(ordinary.buildProfileId, 'w8a16');
    assert.equal(target.decodingRole, 'mtp');
    assert.equal(ordinary.decodingRole, 'baseline');
    assert.equal(target.workloadId, workload.id);
    assert.equal(ordinary.workloadId, workload.id);
    assert.deepEqual(target.measurements, workload.measurements);
    assert.deepEqual(ordinary.measurements, workload.baseline.measurements);
    assert.equal(filename(target), `${run.metadata.model_id}-w4a16-${workload.id}-mtp-performance.csv`);
    assert.equal(filename(ordinary), `${baseline.model_id}-w8a16-${workload.id}-baseline-performance.csv`);
  }
  assert.equal(new Set(exports.map(filename)).size, exports.length);
  assert.equal(filename({modelId: 'resnet_50', buildProfileId: 'w8a8'}), 'resnet_50-w8a8-performance.csv');
});

test('the Performance info mark names the runtime that ran the model', () => {
  const methodStart = source.indexOf('function performanceMethod(');
  const performanceMethod = new Function('readPerformance', 'targetName',
    `return (${source.slice(methodStart, source.indexOf('\nfunction modelPerformance(', methodStart))}\n);`)(
    readPerformance, () => 'Modalix');
  const method = report => performanceMethod({performance: {run: {id: 'run'}, report}});
  assert.equal(method({...base, scope: 'model', metrics: {throughput_fps: 30}, telemetry: {}, components: {model: stats}}),
    'Measured on the Modalix DevKit, running the compiled model with the Neat Model API.');
  assert.equal(method({...base, runtime: {llima: 'test', mole: 'test'}, metrics: {}, generation: {settings: {}, workloads: []}}),
    'Measured on the Modalix DevKit, running the compiled model with LLiMa.');
  assert.equal(performanceMethod({performance: null}), '');
});

test('a speculative subset shows each dataset as measured of available prompts', () => {
  const fixture = JSON.parse(readFileSync(new URL('../../artifact-browser/fixtures/genai-model-report.json', import.meta.url), 'utf8'));
  const subset = workload => ({...workload, dataset: {...workload.dataset, available_samples: 1319}});
  const report = {...fixture, generation: {...fixture.generation,
    settings: {...fixture.generation.settings, decoding: 'speculative'}, baseline: {model_id: 'baseline'},
    workloads: fixture.preview_speculative_workloads.map(subset)}};
  const performance = readPerformance({run: {id: 'run', metadata: {}}, report}).value;
  const begin = performanceSource.indexOf('function generativeBenchmark(');
  const render = new Function('performanceDetails', 'infoMark', 'escapeHtml', 'disclosure',
    `return (${performanceSource.slice(begin)}\n);`)(() => ({markup: ''}), () => '', String, () => '');
  const markup = render(performance, {id: 'run'}, 'gemma4_mtp');
  assert.match(markup, /<td>5 of 1,319<\/td>/);
  assert.match(markup, /Both use identical prompts and output budgets/);
  assert.match(markup, /Seeded random subset of each dataset/);
  const full = render(readPerformance({run: {id: 'run', metadata: {}}, report: {...report, generation: {
    ...report.generation, workloads: fixture.preview_speculative_workloads}}}).value, {id: 'run'}, 'gemma4_mtp');
  assert.match(full, /Full datasets/);
  assert.doesNotMatch(full, /<td>[0-9,]+ of [0-9,]+<\/td>/);
});
