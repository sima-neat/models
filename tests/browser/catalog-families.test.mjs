import assert from 'node:assert/strict';
import {beforeEach, test} from 'node:test';

const elements = new Map();
globalThis.window = {};
globalThis.location = {search: ''};
globalThis.document = {
  documentElement: {dataset: {}},
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      value: '', checked: false, disabled: false, hidden: false, dataset: {},
      innerHTML: '', textContent: '', setAttribute() {}, querySelectorAll: () => [],
    });
    return elements.get(id);
  },
};
const {state} = await import('../../artifact-browser/state.js');
const {groupModelFamilies, familyOverview, renderRuns} = await import('../../artifact-browser/catalog-view.js');

const hero = (id) => ({run_id: id, hero: {artifact_name: 'hero', artifact_sha256: 'a'.repeat(64)}});
const model = (id, family, name, tasks = ['detection'], compiler = '1') => ({
  model_id: id, architecture_id: family, display_name: name, tasks, description: 'Size-specific description.',
  documentation: hero(id),
  builds: [{run_id: id, build_profile_id: 'w8a8', toolchain_version: compiler,
    run: {id, metadata: {model_id: id, build_profile_id: 'w8a8'}}}],
});

beforeEach(() => {
  for (const el of elements.values()) { el.value = ''; el.disabled = false; el.checked = false; }
  Object.assign(state, {catalogMode: 'models', catalogLayout: 'cards', hasMore: false, models: [], runs: [], compileRuns: [],
    taxonomy: {architectures: {
      z_family: {display_name: 'A readable title', description: 'Rotated boxes in aerial imagery.'},
      a_family: {display_name: 'Z readable title'},
    }}});
});

test('family IDs order groups; the existing size order picks the first available hero', () => {
  const nano = model('nano', 'z_family', 'YOLOv8n-det');
  delete nano.documentation;
  const small = model('small', 'z_family', 'YOLOv8s-det');
  const large = model('large', 'z_family', 'YOLOv8l-det', ['segmentation', 'detection']);
  const other = model('other', 'a_family', 'Other');
  const families = groupModelFamilies([large, other, small, nano]);
  assert.deepEqual(families.map(f => f.id), ['a_family', 'z_family']);
  assert.deepEqual(families[1].models.map(m => m.model_id), ['nano', 'small', 'large']);
  assert.equal(families[1].hero, small);
  assert.deepEqual(families[1].tasks, ['detection', 'segmentation']);
  assert.deepEqual(groupModelFamilies([nano])[0].hero, nano);
});

test('overview uses taxonomy text and singleton cards open their model directly', () => {
  state.models = [model('one', 'a_family', 'One'), model('two', 'z_family', 'Two'), model('three', 'z_family', 'Three')];
  renderRuns();
  const html = elements.get('result-list').innerHTML;
  assert.equal(elements.get('result-count').textContent, '2 model families');
  assert.match(html, /data-family-card="a_family" data-run-id="one"/);
  assert.match(html, /data-family-card="z_family" data-family-id="z_family"/);
  assert.match(html, /<strong>A readable title<\/strong>/);
  assert.match(html, /Rotated boxes in aerial imagery\./);
  assert.doesNotMatch(html, /Size-specific description/);
  // Historical taxonomy entries may lack descriptions; never substitute model text.
  assert.equal((html.match(/model-card-description/g) || []).length, 1);
});

test('task and compiler filtering precede family membership, tasks and hero selection', () => {
  state.models = [model('old', 'a_family', 'Old', ['detection'], '0'),
    model('one', 'z_family', 'One', ['detection']), model('two', 'z_family', 'Two', ['segmentation']),
    {...model('pending', 'a_family', 'Pending'), builds: []}];
  elements.get('compiler-filter').value = '1';
  elements.get('task-filter').value = 'segmentation';
  renderRuns();
  const html = elements.get('result-list').innerHTML;
  assert.equal(elements.get('result-count').textContent, '1 model family');
  assert.match(html, /data-family-card="z_family" data-run-id="two"/);
  assert.match(html, /\/runs\/two\/artifacts\/hero\/content/);
  assert.doesNotMatch(html, /a_family|detection|\/runs\/one\//);
});

test('search finds family descriptions and shows individual models', () => {
  state.models = [model('one', 'z_family', 'One'), model('two', 'z_family', 'Two'), model('three', 'a_family', 'Three')];
  elements.get('search').value = 'aerial imagery';
  renderRuns();
  assert.equal(elements.get('result-count').textContent, '2 models');
  assert.equal((elements.get('result-list').innerHTML.match(/data-model-card=/g) || []).length, 2);
  assert.doesNotMatch(elements.get('result-list').innerHTML, /data-family-card/);
  elements.get('search').value = 'one';
  renderRuns();
  assert.equal(elements.get('result-count').textContent, '1 model');
});

test('selected family shows model cards; Table and history never use family overview', () => {
  state.models = [model('one', 'z_family', 'One'), model('two', 'a_family', 'Two')];
  elements.get('architecture-filter').value = 'z_family';
  renderRuns();
  assert.match(elements.get('result-list').innerHTML, /data-model-card="one"/);
  assert.doesNotMatch(elements.get('result-list').innerHTML, /data-model-card="two"/);
  assert.equal(elements.get('clear-family').hidden, false);
  elements.get('architecture-filter').value = '';
  state.catalogLayout = 'table';
  assert.equal(familyOverview(), false);
  state.catalogLayout = 'cards';
  state.catalogMode = 'runs';
  assert.equal(familyOverview(), false);
  state.catalogMode = 'models';
  assert.equal(familyOverview(), true);
});

test('empty filtered results never produce a family card', () => {
  state.models = [model('one', 'z_family', 'One')];
  elements.get('task-filter').value = 'missing';
  renderRuns();
  assert.equal(elements.get('result-count').textContent, '0 model families');
  assert.doesNotMatch(elements.get('result-list').innerHTML, /data-family-card/);
  assert.match(elements.get('result-list').innerHTML, /No matching models/);
});

test('a failed taxonomy request keeps models usable without a nonfunctional family selection', () => {
  state.models = [model('one', 'z_family', 'One'), model('two', 'z_family', 'Two')];
  elements.get('architecture-filter').disabled = true;
  renderRuns();
  assert.equal((elements.get('result-list').innerHTML.match(/data-model-card=/g) || []).length, 2);
  elements.get('architecture-filter').disabled = false;
  state.taxonomy = null;
  assert.equal(familyOverview(), false);
});
