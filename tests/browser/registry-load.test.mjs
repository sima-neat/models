import assert from 'node:assert/strict';
import {test} from 'node:test';

// Enough of a page for the browser modules to load and render the catalog table.
function fakeElement() {
  return {
    dataset: {}, hidden: false, innerHTML: '', textContent: '', value: '', checked: false,
    setAttribute() {}, removeAttribute() {}, querySelector: () => null, querySelectorAll: () => [],
    addEventListener() {}, focus() {},
  };
}
const pageElements = new Map();
globalThis.window = {};
globalThis.location = {search: ''};
globalThis.CSS = {escape: (value) => value};
globalThis.document = {
  documentElement: {dataset: {}},
  getElementById: (id) => pageElements.get(id) ?? pageElements.set(id, fakeElement()).get(id),
};
const requests = [];
const responses = [];
globalThis.fetch = async (url) => {
  requests.push(String(url));
  const status = responses.shift() ?? 404;
  return {status, ok: status < 400, json: async () => ({})};
};
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (callback) => realSetTimeout(callback, 0);

const {state} = await import(new URL('../../artifact-browser/state.js', import.meta.url));
const {api} = await import(new URL('../../artifact-browser/api.js', import.meta.url));
const {renderRuns} = await import(new URL('../../artifact-browser/catalog-view.js', import.meta.url));

const settle = () => new Promise((resolve) => realSetTimeout(resolve, 20));

function showTable(selectedId) {
  requests.length = 0;
  responses.length = 0;
  state.catalogMode = 'models';
  state.catalogLoaded = true;
  state.catalogLayout = 'table';
  state.selectedId = selectedId;
  state.models = [{
    model_id: 'resnet_50',
    benchmark_fingerprints: {w8a8: {performance: 'measured-fingerprint'}},
    builds: [{run_id: 'build', build_profile_id: 'w8a8', run: {id: 'build', metadata: {model_id: 'resnet_50'}}}],
  }];
  renderRuns();
}

test('the catalog table loads no results while a model page is open', async () => {
  showTable('build');
  await settle();
  assert.deepEqual(requests, []);
});

test('the catalog table loads its results when the list is shown', async () => {
  showTable(null);
  await settle();
  assert.ok(requests.some(url => url.includes('/resolve?') && url.includes('measured-fingerprint')));
  assert.ok(requests.every(url => !url.includes('/v1/runs/build')));
});

test('a registry request answered 429 is retried until it succeeds', async () => {
  requests.length = 0;
  responses.splice(0, Infinity, 429, 429, 200);
  assert.deepEqual(await api('/v1/runs/build'), {});
  assert.equal(requests.length, 3);
});

test('a registry that stays busy fails with its 429 after the retries', async () => {
  requests.length = 0;
  responses.splice(0, Infinity, 429, 429, 429, 429, 429);
  await assert.rejects(api('/v1/runs/build'), {status: 429});
  assert.equal(requests.length, 4);
});
