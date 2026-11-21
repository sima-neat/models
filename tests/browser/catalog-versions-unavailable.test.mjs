import assert from 'node:assert/strict';
import {test} from 'node:test';

import {element, installPage, openLink, requests, settle, toasts} from './page-harness.mjs';

const catalog = {models: [{
  id: 'yolo', display_name: 'YOLO', categories: {tasks: ['detection'], architecture: {id: 'family'}},
  builds: [{run_id: 'yolo-w8a8', build_profile_id: 'w8a8', component_id: 'model', toolchain_version: '2.3.0'}],
}], taxonomy: {tasks: {}, architectures: {}}};

function startPage(uiMode) {
  installPage({uiMode, routes: (request) => (request.includes('/catalogs/current?git_ref=') ? {catalog: {document: catalog}} : undefined)});
}

// The registry has no filter-discovery route: it answers 404.
startPage('customer');
const versions = await import('../../artifact-browser/catalog-versions.js');
const {state} = await import('../../artifact-browser/state.js');
await import('../../artifact-browser/app.js');
await settle();

test('without the published versions the customer view shows no new filter', () => {
  assert.equal(versions.versionFiltersAvailable(), false);
  assert.equal(element('platform-filter').innerHTML, '');
  assert.equal(element('core-filter').innerHTML, '');
  assert.notEqual(document.documentElement.dataset.versionFilters, '1');
  assert.deepEqual(state.models.map((model) => model.model_id), ['yolo']);
  assert.ok(requests.some((request) => request.endsWith('/catalogs/current?git_ref=refs%2Fheads%2Fmain')));
  assert.ok(requests.every((request) => !request.includes('selection_id')));
});

test('without the published versions a version link opens the Main catalog unchanged', async () => {
  const link = '?platform_version=3.0.0&core_version=1.1.0&compiler=2.3.0';
  await openLink(link);
  assert.equal(location.search, link);
  assert.equal(versions.selectionId(), null);
  assert.equal(element('compiler-filter').value, '');
  assert.deepEqual(toasts(), []);
  assert.deepEqual(state.models.map((model) => model.model_id), ['yolo']);
  assert.ok(requests.every((request) => !request.includes('selection_id') && !request.includes('/models/filters')));
});
