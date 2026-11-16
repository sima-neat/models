import assert from 'node:assert/strict';
import {test} from 'node:test';

const {deploymentCatalog, branchModels, catalogBuildRuns, withoutProfiles} = await import(new URL('../../artifact-browser/branch-builds.js', import.meta.url));

const run = (id, model, ref, time, extra = {}) => ({
  id, repository: 'sima-neat/models', run_type: 'model_compile', status: 'completed', git_ref: ref,
  commit_sha: `${ref}-commit`, toolchain_name: 'compiler', toolchain_version: '3.0.0', target_platform: 'modalix',
  finished_at: time, metadata: {model_id: model, build_profile_id: 'int8', component_id: 'model', github_run_id: `${id}-workflow`},
  ...extra,
});
const develop = 'refs/heads/develop';
const feature = 'refs/heads/feat/ui';
const resnet = run('resnet-develop', 'resnet_50', develop, '2026-10-01T00:00:00Z');
const yolo = run('yolo-develop', 'yolo', develop, '2026-10-01T00:00:00Z');
const catalogBuild = (item) => ({run_id: item.id, commit_sha: item.commit_sha, build_profile_id: 'int8', component_id: 'model'});
const catalog = [
  {model_id: 'resnet_50', display_name: 'ResNet-50', builds: [catalogBuild(resnet)],
   packages: [{run_id: 'resnet-package', build_profile_id: 'int8', component_run_ids: {model: 'resnet-develop'}}]},
  {model_id: 'yolo', display_name: 'YOLO', builds: [catalogBuild(yolo)]},
];

function batchIds(path) {
  const url = new URL(path, 'https://registry.example');
  assert.match(url.pathname, /\/runs$/);
  assert.equal(url.searchParams.get('limit'), '1');
  return url.searchParams.get('summary_ids').split(',');
}

test('approved native roots select original runs and packages while newer candidates stay hidden', async () => {
  const newer = run('newer', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const definition = {...catalog[0], builds: [], packages: [], build_on_demand: true,
    build_requirements: {int8: {packages: [{id: 'default', component_ids: ['model']}]}},
    approved_deployments: {
      int8: {deployment_id: 'approved', manifest_digest: 'digest', artifacts: [
        {component_id: 'model', run_id: resnet.id, artifact_name: 'llima', layout: 'directory', format: 'llima'},
        {package_id: 'default', run_id: 'approved-package', artifact_name: 'package', layout: 'directory', format: 'llima'},
      ]},
  }};
  const inherited = await catalogBuildRuns([definition], [newer], async path => {
    assert.deepEqual(batchIds(path), [resnet.id]);
    return {summaries: [{run: resnet}]};
  }, '/v1/repositories/sima-neat/models');
  assert.deepEqual(inherited, [resnet]);
  const [model] = branchModels([newer, ...inherited], [], [definition]);
  assert.deepEqual(model.builds.map(build => build.run_id), [resnet.id]);
  assert.equal(model.builds[0].artifact_name, 'llima');
  assert.equal(model.packages[0].run_id, 'approved-package');
  assert.deepEqual(model.packages[0].component_run_ids, {model: resnet.id});
});

test('approved subset packages cannot become full-model downloads', () => {
  const components = ['backbone', 'head'].map(id => run(id, 'example', feature, '2026-10-01T00:00:00Z', {
    metadata: {model_id: 'example', build_profile_id: 'int8', component_id: id},
  }));
  const definition = {model_id: 'example', build_on_demand: true,
    build_requirements: {int8: {packages: [
      {id: 'backbone-only', component_ids: ['backbone']},
      {id: 'complete', component_ids: ['backbone', 'head']},
    ]}},
    approved_deployments: {int8: {artifacts: [
      ...components.map(item => ({component_id: item.id, run_id: item.id})),
      {package_id: 'backbone-only', run_id: 'partial-package'},
      {package_id: 'complete', run_id: 'complete-package'},
    ]}},
  };
  const projected = deploymentCatalog(definition);
  assert.deepEqual(projected.packages.map(item => item.component_run_ids), [
    {backbone: 'backbone'}, {backbone: 'backbone', head: 'head'},
  ]);
  for (const runs of [components, []]) {
    const [model] = branchModels(runs, [], [definition]);
    assert.deepEqual(model.builds.map(item => item.run_id), ['backbone', 'head']);
    assert.deepEqual(model.packages.map(item => item.run_id), ['complete-package']);
  }
  const [unknown] = branchModels(components, [], [{...definition, build_requirements: undefined}], '', 'develop');
  assert.deepEqual(unknown.packages, []);
});

test('a branch that compiled nothing lists every catalog model with its real provenance', () => {
  const models = branchModels([resnet, yolo], [], catalog);

  assert.deepEqual(models.map((model) => model.model_id), ['resnet_50', 'yolo']);
  assert.equal(models[0].builds[0].run.git_ref, develop);
  assert.deepEqual(models[0].packages.map((item) => item.run_id), ['resnet-package']);
});

test('a newer build compiled on the branch replaces the catalog build of its model and profile only', () => {
  const branchResnet = run('resnet-feature', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const models = branchModels([branchResnet, resnet, yolo], [], catalog);

  assert.deepEqual(models.map((model) => model.builds[0].run_id), ['resnet-feature', 'yolo-develop']);
  // The catalog package bundles the replaced build, so it no longer applies.
  assert.deepEqual(models[0].packages, []);
});

test('an older build compiled on the branch does not replace the catalog build', () => {
  const stale = run('resnet-stale', 'resnet_50', feature, '2026-09-01T00:00:00Z');

  assert.equal(branchModels([stale, resnet, yolo], [], catalog)[0].builds[0].run_id, 'resnet-develop');
});

test('a profile the catalog no longer declares shows no variant, even with a newer run', () => {
  const renamed = run('resnet-renamed', 'resnet_50', develop, '2026-10-02T00:00:00Z',
    {metadata: {...resnet.metadata, build_profile_id: 'modalix_int8'}});
  const [model] = branchModels([renamed, resnet, yolo], [], catalog);

  assert.deepEqual([...new Set(model.builds.map((build) => build.build_profile_id))], ['int8']);
});

test('a model the catalog no longer lists shows no variant, even with runs that predate component ids', () => {
  const removed = (id, profile) => run(id, 'ssd_mobilenet_v1', develop, '2026-09-02T00:00:00Z',
    {metadata: {model_id: 'ssd_mobilenet_v1', build_profile_id: profile, github_run_id: `${id}-workflow`}});
  const models = branchModels([removed('ssd-int8', 'int8'), removed('ssd-bf16', 'bf16'), resnet, yolo], [], catalog);

  assert.deepEqual(models.map((model) => model.model_id), ['resnet_50', 'yolo']);
});

test('a branch without a catalog shows every completed run', () => {
  assert.deepEqual(branchModels([resnet, yolo], [], []).map((model) => model.model_id), ['resnet_50', 'yolo']);
});

test('a profile still pending in the catalog shows its completed run', () => {
  const pending = run('resnet-pending', 'resnet_50', develop, '2026-10-02T00:00:00Z',
    {metadata: {...resnet.metadata, build_profile_id: 'bf16'}});
  const declared = [{...catalog[0], pending_builds: [{model_id: 'resnet_50', build_profile_id: 'bf16'}]}, catalog[1]];
  const [model] = branchModels([pending, resnet, yolo], [], declared);

  assert.deepEqual(model.builds.map((build) => build.build_profile_id).sort(), ['bf16', 'int8']);
});

test('catalog runs are requested once per exact ID in batches of up to 40', async () => {
  const superseded = Array.from({length: 42}, (_, index) =>
    run(`superseded-${String(index).padStart(2, '0')}`, 'yolo', develop, '2026-09-01T00:00:00Z'));
  const models = [...catalog, {model_id: 'old', builds: [...superseded.map(catalogBuild), catalogBuild(resnet)]}];
  const expected = new Map([resnet, ...superseded].map((item) => [item.id, item]));
  const batches = [];
  const runs = await catalogBuildRuns(models, [yolo], async (path) => {
    const ids = batchIds(path);
    batches.push(ids);
    return {summaries: ids.map((id) => ({run: expected.get(id)}))};
  }, '/v1/repositories/sima-neat/models');

  assert.deepEqual(batches.map((batch) => batch.length), [40, 3]);
  assert.deepEqual(batches.flat(), [...expected.keys()].sort());
  assert.deepEqual(runs, [...expected.values()]);
});

test('an older reader falls back to the published builds without individual run requests', async () => {
  let requests = 0;
  const runs = await catalogBuildRuns(catalog, [], async (path) => {
    requests += 1;
    assert.deepEqual(batchIds(path), [resnet.id, yolo.id]);
    return {runs: [resnet]};
  }, '/repo');
  assert.equal(requests, 1);
  assert.deepEqual(runs, []);
  assert.deepEqual(branchModels(runs, [], catalog).map((model) => model.builds[0].run_id), [resnet.id, yolo.id]);
  assert.deepEqual(branchModels(runs, [], catalog, resnet.toolchain_version), []);
});

test('completed candidates never replace an on-demand catalog approved build or package', () => {
  const candidate = run('candidate', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const approved = {...catalog[0], build_on_demand: true};
  const models = branchModels([candidate, resnet], [], [approved]);
  assert.equal(models[0].builds[0].run_id, resnet.id);
  assert.equal(models[0].packages[0].run_id, 'resnet-package');
});

test('feature catalogs show finalized selections rather than arbitrary completed runs', async () => {
  const candidate = run('candidate', 'qwen', feature, '2026-10-02T00:00:00Z');
  const newer = run('newer', 'qwen', feature, '2026-10-03T00:00:00Z');
  const definition = {model_id: 'qwen', display_name: 'Qwen', build_on_demand: true, builds: [], packages: []};
  assert.deepEqual(branchModels([candidate], [], [definition], '', 'feat/ui'), []);
  const selected = {...definition, branch_deployments: {int8: {artifacts: [
    {component_id: 'model', run_id: candidate.id, artifact_name: 'llima', layout: 'directory', format: 'llima'},
  ]}}};
  const resolved = await catalogBuildRuns([selected], [newer], async path => {
    assert.deepEqual(batchIds(path), [candidate.id]);
    return {summaries: [{run: candidate}]};
  }, '/v1/repositories/sima-neat/models');
  const [model] = branchModels([newer, ...resolved], [], [selected]);
  assert.equal(model.builds[0].run_id, candidate.id);
  assert.equal(model.builds[0].artifact_name, 'llima');
  assert.deepEqual(definition.builds, []);
  assert.deepEqual(branchModels([candidate], [], [selected], '2.1.3'), []);
});

test('branch selections replace inherited approvals and pin their matching package', () => {
  const candidate = run('candidate', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const definition = {...catalog[0], build_on_demand: true,
    build_requirements: {int8: {packages: [{id: 'default', component_ids: ['model']}]}},
    approved_deployments: {int8: {artifacts: [
      {component_id: 'model', run_id: resnet.id, artifact_name: 'llima'},
    ]}},
    branch_deployments: {int8: {artifacts: [
      {component_id: 'model', run_id: candidate.id, artifact_name: 'llima'},
      {package_id: 'default', run_id: 'candidate-package', artifact_name: 'package'},
    ]}},
  };
  const packageRun = {id: 'unselected-package', git_ref: feature, status: 'completed', run_type: 'model_package',
    metadata: {model_id: 'resnet_50', build_profile_id: 'int8', component_run_ids: {model: candidate.id}}};
  const [model] = branchModels([candidate, resnet], [packageRun], [definition]);
  assert.equal(model.builds[0].run_id, candidate.id);
  assert.deepEqual(model.packages.map(item => item.run_id), ['candidate-package']);
  assert.equal(deploymentCatalog({...definition, branch_deployments: undefined}).builds[0].run_id, resnet.id);
});

test('catalog deployment metadata survives branch selection', () => {
  const definition = {...catalog[0], build_on_demand: true,
    builds: [{...catalog[0].builds[0], layout: 'directory', storage_backend: 's3'}],
    packages: [{...catalog[0].packages[0], layout: 'directory', storage_backend: 's3'}]};
  const [model] = branchModels([resnet], [], [definition]);
  assert.equal(model.builds[0].layout, 'directory');
  assert.equal(model.packages[0].storage_backend, 's3');
});


test('unapproved packages cannot replace the catalog package of an on-demand build', () => {
  const approved = {...catalog[0], build_on_demand: true};
  const candidate = {id: 'candidate-package', status: 'completed', run_type: 'model_package',
    metadata: {model_id: 'resnet_50', build_profile_id: 'int8', component_run_ids: {model: resnet.id}}};
  const [model] = branchModels([resnet], [candidate], [approved]);
  assert.deepEqual(model.packages.map((item) => item.run_id), ['resnet-package']);
});

test('deployment-only catalogs resolve inherited builds and keep the selected package despite newer runs', async () => {
  const selection = {artifacts: [
    {run_id: resnet.id, component_id: 'model', artifact_name: 'llima', layout: 'directory'},
    {run_id: 'folder-package', package_id: 'default', artifact_name: 'package', layout: 'directory'},
  ]};
  const definitions = [{model_id: 'resnet_50', builds: [], pending_builds: [],
    build_requirements: {int8: {packages: [{id: 'default', component_ids: ['model']}]}},
    branch_deployments: {int8: selection}}];
  const inherited = await catalogBuildRuns(definitions, [], async path => {
    assert.deepEqual(batchIds(path), [resnet.id]);
    return {summaries: [{run: resnet}]};
  }, '/repo');
  const newer = {...resnet, id: 'newer', finished_at: '2026-10-17T00:00:00Z'};
  const [model] = branchModels([newer, ...inherited], [], definitions);
  assert.equal(model.builds[0].run_id, resnet.id);
  assert.equal(model.packages[0].run_id, 'folder-package');
  assert.deepEqual(model.packages[0].component_run_ids, {model: resnet.id});
  assert.equal(branchModels([resnet], [], definitions, 'other').length, 0);
});


test('only a profile showing the catalog build keeps its benchmark fingerprints', () => {
  const benchmark_fingerprints = {int8: {performance: 'performance-fingerprint', accuracy: 'accuracy-fingerprint'}};
  const [model] = branchModels([resnet], [], [{...catalog[0], benchmark_fingerprints}]);
  assert.deepEqual(model.benchmark_fingerprints, benchmark_fingerprints);
  // The fingerprints name the catalog build's MPKs, so a newer build compiled on the branch has none.
  const branchResnet = run('resnet-feature', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const [rebuilt] = branchModels([branchResnet, resnet], [], [{...catalog[0], benchmark_fingerprints}]);
  assert.equal(rebuilt.builds[0].run_id, branchResnet.id);
  assert.deepEqual(rebuilt.benchmark_fingerprints, {});
});

for (const field of ['branch_deployments', 'approved_deployments']) {
  test(`${field} keep the benchmark fingerprints of the selected build's profile`, () => {
    const selection = {artifacts: [{run_id: resnet.id, component_id: 'model', artifact_name: 'mpk'}]};
    const benchmark_fingerprints = {int8: {performance: 'performance-fingerprint'}};
    const definition = {model_id: 'resnet_50', builds: [], benchmark_fingerprints, [field]: {int8: selection}};
    const [model] = branchModels([resnet], [], [definition]);
    assert.equal(model.builds[0].run_id, resnet.id);
    assert.deepEqual(model.benchmark_fingerprints, benchmark_fingerprints);
  });
}

test('a hidden profile leaves the catalog, and a model without a profile leaves with it', () => {
  const wide = {...resnet, id: 'resnet-wide', metadata: {...resnet.metadata, build_profile_id: 'int16'}};
  const twoProfiles = {...catalog[0], builds: [catalogBuild(resnet), {...catalogBuild(wide), build_profile_id: 'int16'}]};
  const shown = withoutProfiles([twoProfiles, catalog[1]], {resnet_50: ['int8'], yolo: ['int8']});
  assert.deepEqual(shown.map((model) => model.model_id), ['resnet_50']);
  assert.deepEqual(shown[0].packages, []);
  const [model] = branchModels([resnet, wide, yolo], [], shown);
  assert.deepEqual(model.builds.map((build) => build.build_profile_id), ['int16']);
  assert.deepEqual(withoutProfiles(catalog, {}), catalog);
});

test('a hidden profile takes its pending build and its deployment selection with it', () => {
  const selection = {artifacts: [{run_id: resnet.id, component_id: 'model', artifact_name: 'mpk'}]};
  const pending = {model_id: 'resnet_50', builds: [], pending_builds: [{build_profile_id: 'int8'}]};
  const selected = {model_id: 'yolo', builds: [], approved_deployments: {int8: selection, int16: selection}};
  const shown = withoutProfiles([pending, selected], {resnet_50: ['int8'], yolo: ['int8']});
  assert.deepEqual(shown.map((model) => model.model_id), ['yolo']);
  assert.deepEqual(Object.keys(shown[0].approved_deployments), ['int16']);
});
