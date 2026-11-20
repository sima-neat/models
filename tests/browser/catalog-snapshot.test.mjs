import assert from 'node:assert/strict';
import {test} from 'node:test';
import {resolveCatalogSnapshot} from '../../website/scripts/pin-catalog-snapshot.mjs';

const revision = 'a'.repeat(40);
const canonicalSha256 = 'b'.repeat(64);
const catalogs = 'https://registry.test/v1/repositories/sima-neat/models/catalogs';
const currentUrl = `${catalogs}/current?git_ref=refs%2Fheads%2Fmain`;
const pinnedUrl = `${catalogs}/${revision}?git_ref=refs%2Fheads%2Fmain&snapshot=${canonicalSha256}`;
const snapshot = {revision, canonicalSha256};

// A registry that answers each URL from a table and records what was asked.
function registry(answers) {
  const requests = [];
  const fetch = async (url) => {
    requests.push(url);
    const answer = answers[url];
    if (!answer) return {ok: false, status: 404, json: async () => ({})};
    return {ok: true, status: 200, json: async () => ({catalog: answer})};
  };
  return {fetch, requests};
}

function deployConfig(catalogSnapshot = null) {
  return {
    apiBase: 'https://registry.test', repository: {organization: 'sima-neat', name: 'models'},
    defaultBranch: 'main', catalogSnapshot,
  };
}

const served = {catalog_revision: revision, canonical_sha256: canonicalSha256, document: {models: []}};

// The page modules read window.MODEL_REGISTRY_CONFIG and the DOM when they load.
function fakeElement() {
  return {
    dataset: {}, hidden: false, innerHTML: '', textContent: '', value: '', checked: false,
    setAttribute() {}, removeAttribute() {}, querySelector: () => null, querySelectorAll: () => [],
    addEventListener() {}, focus() {},
  };
}
const pageElements = new Map();
const storage = new Map();
globalThis.window = {
  MODEL_REGISTRY_CONFIG: {
    uiMode: 'customer', apiBase: 'https://registry.test', repository: {organization: 'sima-neat', name: 'models'},
    catalogSnapshot: snapshot,
  },
};
globalThis.location = {search: ''};
globalThis.CSS = {escape: (value) => value};
globalThis.document = {
  documentElement: {dataset: {}},
  getElementById: (id) => pageElements.get(id) ?? pageElements.set(id, fakeElement()).get(id),
};
globalThis.localStorage = {
  getItem: (key) => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value),
};
const pageRequests = [];
globalThis.fetch = async (url) => {
  pageRequests.push(String(url));
  return {status: 200, ok: true, json: async () => ({catalog: served})};
};

const {state} = await import(new URL('../../artifact-browser/state.js', import.meta.url));
const {readCachedCatalog, writeCachedCatalog} = await import(new URL('../../artifact-browser/api.js', import.meta.url));
const {fetchCatalogDocument} = await import(new URL('../../artifact-browser/catalog-view.js', import.meta.url));

test('a deploy pins the current catalog after the registry serves that exact snapshot', async () => {
  const {fetch, requests} = registry({[currentUrl]: served, [pinnedUrl]: served});
  assert.deepEqual(await resolveCatalogSnapshot(deployConfig(), fetch), snapshot);
  assert.deepEqual(requests, [currentUrl, pinnedUrl]);
});

test('a requested snapshot is pinned without reading the current catalog', async () => {
  const {fetch, requests} = registry({[pinnedUrl]: served});
  assert.deepEqual(await resolveCatalogSnapshot(deployConfig(snapshot), fetch), snapshot);
  assert.deepEqual(requests, [pinnedUrl]);
});

test('a snapshot the registry does not serve fails the deploy', async () => {
  const {fetch} = registry({});
  await assert.rejects(resolveCatalogSnapshot(deployConfig(snapshot), fetch), /HTTP 404/);
  const other = registry({[pinnedUrl]: {...served, canonical_sha256: 'c'.repeat(64)}});
  await assert.rejects(resolveCatalogSnapshot(deployConfig(snapshot), other.fetch), /served snapshot a+:c+ for a+:b+/);
  const otherRevision = registry({[pinnedUrl]: {...served, catalog_revision: 'd'.repeat(40)}});
  await assert.rejects(resolveCatalogSnapshot(deployConfig(snapshot), otherRevision.fetch), /served snapshot d+:b+ for a+:b+/);
  const noModels = registry({[pinnedUrl]: {...served, document: {}}});
  await assert.rejects(resolveCatalogSnapshot(deployConfig(snapshot), noModels.fetch), /no model catalog/);
});

test('a pinned page requests its snapshot instead of the current catalog', async () => {
  pageRequests.length = 0;
  assert.deepEqual(await fetchCatalogDocument('main'), served.document);
  assert.deepEqual(pageRequests, [pinnedUrl]);
});

test('a pinned page reuses only a saved catalog of its own snapshot', () => {
  state.catalogMode = 'models';
  state.catalogDefinitions = [];
  writeCachedCatalog();
  assert.equal(storage.size, 1);
  assert.equal(readCachedCatalog()?.snapshot, canonicalSha256);

  const [key, value] = [...storage][0];
  storage.set(key, JSON.stringify({...JSON.parse(value), snapshot: 'c'.repeat(64)}));
  assert.equal(readCachedCatalog(), null);
  const {snapshot: _, ...unpinned} = JSON.parse(value);
  storage.set(key, JSON.stringify(unpinned));
  assert.equal(readCachedCatalog(), null);
});
