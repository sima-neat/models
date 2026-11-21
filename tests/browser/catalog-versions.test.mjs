import assert from 'node:assert/strict';
import {test} from 'node:test';

import {element, installPage, openLink, requests, settle, toasts} from './page-harness.mjs';

const selection = (id, platform, core, compilers, isDefault = false) => ({
  selection_id: id, catalog_id: `catalog-${id}`, platform_version: platform, neat_core_version: core,
  neat_core_revision: 'a'.repeat(40), retired: false, is_default: isDefault, toolchain_versions: compilers,
});
// Platform 3.0.0 has a default Neat Core; 3.1.0 has none, and one of its versions is retired.
const selections = [
  selection('sel-300-100', '3.0.0', '1.0.0', ['2.0.0']),
  selection('sel-300-110', '3.0.0', '1.1.0', ['2.0.0', '2.1.0'], true),
  selection('sel-300-120', '3.0.0', '1.2.0', ['2.1.0']),
  selection('sel-310-120', '3.1.0', '1.2.0', ['2.1.0']),
  selection('sel-310-130', '3.1.0', '1.3.0', ['2.2.0']),
  {...selection('sel-310-140', '3.1.0', '1.4.0', ['2.2.0']), retired: true},
];
const published = {
  scope: 'published', platform_versions: ['3.0.0', '3.1.0'], neat_core_versions: ['1.0.0', '1.1.0', '1.2.0', '1.3.0'],
  toolchain_versions: ['2.0.0', '2.1.0', '2.2.0'], selections,
};
const branch = {scope: 'branch', platform_versions: [], neat_core_versions: [], toolchain_versions: ['2.2.0', '2.3.0'], selections: []};

// A catalog document's build names no compiler. The compiler each fake build was made with is kept beside
// it, for the /models route to answer from.
const buildCompilers = new Map();
const build = (model, compiler, profile = 'w8a8') => {
  const run_id = `${model}-${profile}-${compiler}`;
  buildCompilers.set(run_id, compiler);
  return {
    run_id, build_profile_id: profile, component_id: 'model', artifact_sha256: 'b'.repeat(64),
    compile_fingerprint: 'c'.repeat(64), compile_invocation_id: profile,
  };
};
const model = (id, builds, deployments = {}) => ({
  id, display_name: id, categories: {tasks: ['detection'], architecture: {id: 'family'}}, builds,
  benchmark_fingerprints: {w8a8: {performance: `${id}-performance`}}, ...deployments,
});
// Each version's catalog holds different models, so a listed model shows which catalog was read.
const catalogs = {
  latest: [
    model('newest', [build('newest', '2.3.0')]),
    model('shared', [build('shared', '2.2.0')]),
    // An approved on-demand deployment: the registry matches it by its compiler requirement.
    model('on_demand', [], {approved_deployments: {w8a8: {
      requirements: {compiler_version: '2.3.0'},
      artifacts: [{component_id: 'model', run_id: 'on-demand-run', artifact_name: 'mpk'}],
    }}}),
  ],
  'sel-300-110': [
    model('shared', [build('shared', '2.0.0'), build('shared', '2.1.0', 'w16a16')]),
    model('older', [build('older', '2.1.0')]),
  ],
  'sel-310-130': [model('shared', [build('shared', '2.2.0')])],
};

// The /models route: the models with a build by the compiler or a deployment that requires it, each with
// all of its builds and their compilers. It serves one model per page here, to show paging.
let compilerRouteDown = false;
function compilerModels(query) {
  if (compilerRouteDown) return undefined;
  const compiler = query.get('toolchain_version');
  const matching = catalogs[query.get('selection_id') || 'latest'].filter((item) => (
    item.builds.some((entry) => buildCompilers.get(entry.run_id) === compiler)
    || Object.values(item.approved_deployments || {}).some((entry) => entry.requirements.compiler_version === compiler)
  )).sort((left, right) => left.id.localeCompare(right.id));
  const start = Number(query.get('cursor') || 0);
  const {id, builds, approved_deployments} = matching[start] || {};
  return {
    models: id ? [{model_id: id, approved_deployments,
      builds: builds.map((entry) => ({...entry, toolchain_version: buildCompilers.get(entry.run_id)}))}] : [],
    has_more: start + 1 < matching.length, next_cursor: start + 1 < matching.length ? String(start + 1) : null,
  };
}

installPage({uiMode: 'customer', routes(request) {
  if (request.endsWith('/models/filters')) return published;
  if (request.includes('/models/filters?git_ref=refs%2Fheads%2Fmain')) return branch;
  const catalog = request.match(/\/catalogs\/current\?(git_ref|selection_id)=([^&]+)$/);
  if (catalog) {
    const models = catalogs[catalog[1] === 'git_ref' ? 'latest' : catalog[2]];
    return models && {catalog: {document: {models, taxonomy: {tasks: {}, architectures: {}}}}};
  }
  if (request.includes('/models?')) return compilerModels(new URLSearchParams(request.split('?')[1]));
  if (request.includes('/resolve?')) return {run: {id: 'benchmark-run'}};
  if (request.startsWith('/v1/runs/benchmark-run')) return {run: {id: 'benchmark-run'}, test_results: []};
  return undefined;
}});

const versions = await import('../../artifact-browser/catalog-versions.js');
const {state} = await import('../../artifact-browser/state.js');
const {loadBenchmarkResults, readCachedCatalog, variantCacheKey} = await import('../../artifact-browser/api.js');
const {audienceFacts} = await import('../../artifact-browser/audience.js');
const {modelDownloadCommand} = await import('../../artifact-browser/downloads.js');
await import('../../artifact-browser/app.js');
await settle();

const optionValues = (id) => [...element(id).innerHTML.matchAll(/<option value="([^"]*)"/g)].map((match) => match[1]);
const listedModels = () => state.models.map((item) => item.model_id);
const catalogReads = () => requests.filter((request) => request.includes('/catalogs/current'));
const address = () => new URLSearchParams(location.search);

async function choose(id, value) {
  requests.length = 0;
  element(id).value = value;
  await element(id).fire('change');
  await settle();
}

test('only the customer view has the version filters, and it never loads run history for them', () => {
  assert.equal(audienceFacts('customer').versionFilters, true);
  assert.equal(audienceFacts('developer').versionFilters, false);
  assert.equal(audienceFacts('customer').catalogRunMetadata, false);
});

test('the page asks for the published versions once and opens on Latest', async () => {
  assert.equal(requests.filter((request) => request.includes('/models/filters')).length, 2);
  assert.deepEqual(optionValues('platform-filter'), ['', '3.0.0', '3.1.0']);
  assert.match(element('platform-filter').innerHTML, /<option value="">Latest<\/option>/);
  assert.equal(element('platform-filter').value, '');
  assert.equal(element('platform-filter').closest().hidden, false);
  assert.equal(element('core-filter').disabled, true);
  assert.deepEqual(optionValues('compiler-filter'), ['', '2.3.0', '2.2.0']);
  assert.equal(document.documentElement.dataset.versionFilters, '1');
  assert.deepEqual(listedModels(), ['newest', 'on_demand', 'shared']);
  assert.ok(requests.some((request) => request.endsWith('/catalogs/current?git_ref=refs%2Fheads%2Fmain')));
  assert.ok(requests.every((request) => !request.includes('selection_id') && !request.includes('/runs')));

  await openLink('');
  assert.ok(requests.every((request) => !request.includes('/models/filters')));
});

test('Neat Core lists the versions published for the platform, not every published Neat Core', () => {
  assert.deepEqual(versions.platformVersions(selections), ['3.0.0', '3.1.0']);
  assert.deepEqual(versions.coreVersions(selections, '3.0.0'), ['1.0.0', '1.1.0', '1.2.0']);
  assert.deepEqual(versions.coreVersions(versions.publishedSelections(), '3.1.0'), ['1.2.0', '1.3.0']);
});

test('a platform opens with its default Neat Core, else with the last one', () => {
  assert.equal(versions.defaultSelection(selections, '3.0.0').neat_core_version, '1.1.0');
  assert.equal(versions.defaultSelection(versions.publishedSelections(), '3.1.0').neat_core_version, '1.3.0');
  assert.equal(versions.defaultSelection(selections, '9.9.9'), null);
});

test('choosing a platform reads its catalog by selection and puts the version in the address', async () => {
  await choose('platform-filter', '3.0.0');
  assert.equal(element('core-filter').value, '1.1.0');
  assert.equal(element('core-filter').disabled, false);
  assert.deepEqual(optionValues('core-filter'), ['1.0.0', '1.1.0', '1.2.0']);
  assert.deepEqual(optionValues('compiler-filter'), ['', '2.1.0', '2.0.0']);
  assert.deepEqual(catalogReads(), ['/v1/repositories/sima-neat/models/catalogs/current?selection_id=sel-300-110']);
  assert.ok(requests.every((request) => !request.includes('git_ref') && !request.includes('/runs')));
  assert.deepEqual(listedModels(), ['older', 'shared']);
  assert.equal(address().get('platform_version'), '3.0.0');
  assert.equal(address().get('core_version'), '1.1.0');
  assert.equal(address().has('compiler'), false);
});

const modelReads = () => requests.filter((request) => request.includes('/models?'));

test('a compiler keeps only the models and variants the registry says it built', async () => {
  await choose('compiler-filter', '2.1.0');
  assert.deepEqual(modelReads(), [
    '/v1/repositories/sima-neat/models/models?selection_id=sel-300-110&toolchain_version=2.1.0&limit=200',
    '/v1/repositories/sima-neat/models/models?selection_id=sel-300-110&toolchain_version=2.1.0&limit=200&cursor=1',
  ]);
  assert.deepEqual(listedModels(), ['older', 'shared']);
  assert.deepEqual(state.models[1].builds.map((item) => item.build_profile_id), ['w16a16']);

  await choose('compiler-filter', '2.0.0');
  assert.deepEqual(modelReads(), ['/v1/repositories/sima-neat/models/models?selection_id=sel-300-110&toolchain_version=2.0.0&limit=200']);
  assert.deepEqual(listedModels(), ['shared']);
  assert.deepEqual(state.models[0].builds.map((item) => item.build_profile_id), ['w8a8']);
  assert.equal(address().get('compiler'), '2.0.0');
  assert.ok(requests.every((request) => !request.includes('/runs')));
});

test('the registry is asked once for each version and compiler', async () => {
  await choose('compiler-filter', '2.1.0');
  await choose('compiler-filter', '2.0.0');
  assert.deepEqual(modelReads(), []);
  assert.deepEqual(listedModels(), ['shared']);
});

test('a model link and Compare keep the version in the address', async () => {
  await element('result-list').fire('click', {target: {closest: (selector) => (selector === '[data-run-id]' ? {dataset: {runId: 'shared-w8a8-2.0.0'}} : null)}});
  assert.equal(address().get('model'), 'shared');
  assert.equal(address().get('platform_version'), '3.0.0');
  assert.equal(address().get('core_version'), '1.1.0');
  assert.equal(address().get('compiler'), '2.0.0');

  await element('nav-compare').fire('click');
  await settle();
  assert.equal(address().get('view'), 'compare');
  assert.equal(address().get('platform_version'), '3.0.0');
  assert.equal(address().get('compiler'), '2.0.0');
});

test('a link restores its version and compiler', async () => {
  await openLink('?platform_version=3.1.0&core_version=1.3.0&compiler=2.2.0');
  assert.equal(element('platform-filter').value, '3.1.0');
  assert.equal(element('core-filter').value, '1.3.0');
  assert.equal(element('compiler-filter').value, '2.2.0');
  assert.deepEqual(catalogReads(), ['/v1/repositories/sima-neat/models/catalogs/current?selection_id=sel-310-130']);
  assert.deepEqual(modelReads(), ['/v1/repositories/sima-neat/models/models?selection_id=sel-310-130&toolchain_version=2.2.0&limit=200']);
  assert.deepEqual(listedModels(), ['shared']);

  const url = new URL('https://models.test/?q=yolo');
  versions.writeVersionParams(url, element('compiler-filter').value);
  assert.equal(url.search, '?q=yolo&platform_version=3.1.0&core_version=1.3.0&compiler=2.2.0');
  assert.equal(versions.selectLinkedVersion(url.searchParams), true);
  assert.equal(versions.selectionId(), 'sel-310-130');
  assert.equal(versions.linkedCompiler(url.searchParams), '2.2.0');
});

test('Latest and any compiler leave the address without version parameters', async () => {
  await choose('platform-filter', '');
  assert.equal(location.search, '?compiler=2.2.0');
  assert.equal(element('core-filter').disabled, true);
  assert.deepEqual(catalogReads(), ['/v1/repositories/sima-neat/models/catalogs/current?git_ref=refs%2Fheads%2Fmain']);
  assert.deepEqual(modelReads(), ['/v1/repositories/sima-neat/models/models?git_ref=refs%2Fheads%2Fmain&toolchain_version=2.2.0&limit=200']);
  assert.deepEqual(listedModels(), ['shared']);

  await choose('compiler-filter', '');
  assert.equal(location.search, '');
  assert.deepEqual(modelReads(), []);
  assert.deepEqual(listedModels(), ['newest', 'on_demand', 'shared']);
});

test('a failed compiler read shows a toast and clears the compiler', async () => {
  compilerRouteDown = true;
  element('toast-region').children.length = 0;
  await choose('compiler-filter', '2.3.0');
  compilerRouteDown = false;
  assert.equal(toasts().length, 1);
  assert.match(toasts()[0], /^The compiler filter is temporarily unavailable\./);
  assert.equal(element('compiler-filter').value, '');
  assert.equal(location.search, '');
  assert.deepEqual(listedModels(), ['newest', 'on_demand', 'shared']);

  await choose('compiler-filter', '2.3.0');
  assert.equal(modelReads().length, 2);
  assert.deepEqual(listedModels(), ['newest', 'on_demand']);
  await choose('compiler-filter', '');
});

test('a deployment is kept by the compiler its requirements name', async () => {
  await choose('compiler-filter', '2.3.0');
  assert.deepEqual(listedModels(), ['newest', 'on_demand']);
  await choose('compiler-filter', '');
});

test('a link to a version that is no longer published opens Latest with a toast', async () => {
  for (const link of ['?platform_version=9.9.9&core_version=1.0.0', '?platform_version=3.1.0&core_version=1.4.0']) {
    element('toast-region').children.length = 0;
    await openLink(link);
    assert.deepEqual(toasts(), ['That version is no longer published. Showing Latest.']);
    assert.equal(element('platform-filter').value, '');
    assert.equal(location.search, '');
    assert.deepEqual(catalogReads(), ['/v1/repositories/sima-neat/models/catalogs/current?git_ref=refs%2Fheads%2Fmain']);
  }
});

test('a model the version does not hold gets one plain sentence, and the filters stay', async () => {
  await openLink('?model=newest&variant=w8a8&platform_version=3.0.0&core_version=1.1.0');
  assert.equal(element('catalog-notice').textContent, 'newest is not published for Platform 3.0.0 with Neat Core 1.1.0.');
  assert.equal(element('catalog-notice').hidden, false);
  assert.equal(element('workspace').dataset.view, 'catalog');
  assert.equal(element('platform-filter').value, '3.0.0');
  assert.deepEqual(listedModels(), ['older', 'shared']);

  await choose('platform-filter', '');
  assert.equal(element('catalog-notice').hidden, true);
});

test('benchmark results are read and kept per version', async () => {
  const shared = {model_id: 'shared', benchmark_fingerprints: {w8a8: {performance: 'shared-performance'}}};
  const resolves = () => requests.filter((request) => request.includes('/resolve?'));

  requests.length = 0;
  versions.selectVersion('');
  await loadBenchmarkResults(shared, 'w8a8');
  await loadBenchmarkResults(shared, 'w8a8');
  assert.equal(resolves().length, 1);
  assert.ok(!resolves()[0].includes('selection_id'));

  versions.selectVersion('3.0.0');
  await loadBenchmarkResults(shared, 'w8a8');
  await loadBenchmarkResults(shared, 'w8a8');
  assert.equal(resolves().length, 2);
  assert.match(resolves()[1], /fingerprint=shared-performance&selection_id=sel-300-110$/);

  versions.selectVersion('3.1.0', '1.2.0');
  await loadBenchmarkResults(shared, 'w8a8');
  assert.match(resolves()[2], /selection_id=sel-310-120$/);
});

test('the saved catalog and the loaded variants are kept per version', () => {
  const run = {git_ref: 'refs/heads/main', commit_sha: 'commit', metadata: {model_id: 'shared', build_profile_id: 'w8a8'}};

  versions.selectVersion('3.0.0');
  const pinnedKey = variantCacheKey(run);
  assert.deepEqual(readCachedCatalog().catalogDefinitions.map((item) => item.model_id), ['shared', 'older']);

  versions.selectVersion('3.0.0', '1.0.0');
  assert.equal(readCachedCatalog(), null);
  assert.notEqual(variantCacheKey(run), pinnedKey);

  versions.selectVersion('');
  assert.deepEqual(readCachedCatalog().catalogDefinitions.map((item) => item.model_id), ['newest', 'shared', 'on_demand']);
  assert.notEqual(variantCacheKey(run), pinnedKey);
});

test('the copied command names the chosen version, and nothing more for Latest', () => {
  const version = versions.selectVersion('3.0.0');
  assert.equal(modelDownloadCommand('yolo', 'w8a8', null, false, version),
    'sima-cli models download --id yolo --variant w8a8 --platform-version 3.0.0 --core-version 1.1.0');
  assert.equal(modelDownloadCommand('yolo', 'w8a8', null, false, versions.selectVersion('')),
    'sima-cli models download --id yolo --variant w8a8');
});
