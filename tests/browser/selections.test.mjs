import assert from 'node:assert/strict';
import {test} from 'node:test';

// Enough of a page for the browser modules to load.
function fakeElement() {
  return {
    dataset: {}, hidden: false, innerHTML: '', textContent: '', value: '', checked: false,
    setAttribute() {}, removeAttribute() {}, querySelector: () => null, querySelectorAll: () => [],
    addEventListener() {}, focus() {},
  };
}
const pageElements = new Map();
globalThis.window = {MODEL_REGISTRY_CONFIG: {repository: {organization: 'sima-neat', name: 'models'}}};
globalThis.location = {search: ''};
globalThis.CSS = {escape: (value) => value};
globalThis.document = {
  documentElement: {dataset: {}},
  getElementById: (id) => pageElements.get(id) ?? pageElements.set(id, fakeElement()).get(id),
};
const requests = [];
let respond = () => ({status: 404, body: {}});
globalThis.fetch = async (url) => {
  requests.push(String(url));
  const {status, body} = respond(String(url));
  return {status, ok: status < 400, json: async () => body};
};

const {state} = await import(new URL('../../artifact-browser/state.js', import.meta.url));
const {chooseSelection, fetchSelections, fetchCatalogDocument} = await import(
  new URL('../../artifact-browser/catalog-view.js', import.meta.url));

const selection = (platform, core, isDefault, revision = core.replaceAll('.', '').padEnd(40, 'a')) => ({
  selection_id: `${platform}-${core}`, platform_version: platform, neat_core_version: core,
  neat_core_revision: revision, is_default: isDefault, retired: false,
});
const selections = [selection('3.0.1', '0.6.2', false), selection('3.0.1', '0.6.3', true), selection('3.1.0', '0.7.0', true)];

test('a page opens on the newest platform and its default Core selection', () => {
  assert.equal(chooseSelection(selections, null, null).selection_id, '3.1.0-0.7.0');
});

test('a platform compares as a version, not as text', () => {
  const later = selection('3.10.0', '0.8.0', true);
  assert.equal(chooseSelection([...selections, later], null, null), later);
});

test('a link keeps the platform and Core selection it names', () => {
  assert.equal(chooseSelection(selections, '3.0.1', '0.6.2').selection_id, '3.0.1-0.6.2');
  assert.equal(chooseSelection(selections, '3.0.1', null).selection_id, '3.0.1-0.6.3');
  assert.equal(chooseSelection(selections, '2.9.0', '0.6.2').selection_id, '3.1.0-0.7.0');
});

test('a platform whose default was retired keeps the newest platform that has one', () => {
  const withoutDefault = [selection('3.2.0', '0.8.0', false), ...selections];
  assert.equal(chooseSelection(withoutDefault, null, null).selection_id, '3.1.0-0.7.0');
});

test('no published selection leaves the page on a branch catalog', async () => {
  assert.equal(chooseSelection([], null, null), null);
  respond = () => ({status: 404, body: {}});
  assert.deepEqual(await fetchSelections(), []);
  respond = () => ({status: 200, body: {selections: []}});
  assert.deepEqual(await fetchSelections(), []);
});

test('the selected catalog is read by platform and Core commit, and a branch by its ref', async () => {
  respond = () => ({status: 200, body: {catalog: {document: {models: []}}}});
  requests.length = 0;
  state.selection = selections[2];
  await fetchCatalogDocument('main');
  state.selection = null;
  await fetchCatalogDocument('main');
  assert.match(requests[0], /catalogs\/current\?platform_version=3\.1\.0&neat_core_revision=070a+$/);
  assert.match(requests[1], /catalogs\/current\?git_ref=refs%2Fheads%2Fmain$/);
});
