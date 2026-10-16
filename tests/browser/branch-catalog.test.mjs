import assert from 'node:assert/strict';
import {test} from 'node:test';

const {branchModels, catalogBuildRuns} = await import(new URL('../../artifact-browser/branch-builds.js', import.meta.url));

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

test('approved native roots select original runs and packages while newer candidates stay hidden', async () => {
  const newer = run('newer', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const definition = {...catalog[0], builds: [], packages: [], build_on_demand: true, approved_deployments: {
    int8: {deployment_id: 'approved', manifest_digest: 'digest', artifacts: [
      {component_id: 'model', run_id: resnet.id, artifact_name: 'llima', layout: 'directory', format: 'llima'},
      {package_id: 'default', run_id: 'approved-package', artifact_name: 'package', layout: 'directory', format: 'llima'},
    ]},
  }};
  const inherited = await catalogBuildRuns([definition], [newer], async path => {
    assert.equal(path, `/v1/runs/${resnet.id}`);
    return {run: resnet};
  }, '/v1/repositories/sima-neat/models');
  assert.deepEqual(inherited, [resnet]);
  const [model] = branchModels([newer, ...inherited], [], [definition]);
  assert.deepEqual(model.builds.map(build => build.run_id), [resnet.id]);
  assert.equal(model.builds[0].artifact_name, 'llima');
  assert.equal(model.packages[0].run_id, 'approved-package');
  assert.deepEqual(model.packages[0].component_run_ids, {model: resnet.id});
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

test('catalog runs are resolved by commit, then by id for runs the list omits', async () => {
  const superseded = run('superseded', 'yolo', develop, '2026-09-01T00:00:00Z');
  const models = [...catalog, {model_id: 'old', builds: [catalogBuild(superseded)]}];
  const paths = [];
  const runs = await catalogBuildRuns(models, [yolo], async (path) => {
    paths.push(path);
    if (path.startsWith('/v1/runs/')) return {run: superseded};
    assert.match(path, /^\/v1\/repositories\/sima-neat\/models\/runs\?/);
    assert.match(path, /run_type=model_compile/);
    return {runs: [resnet, run('unrelated', 'other', develop, '2026-10-01T00:00:00Z')], has_more: false};
  }, '/v1/repositories/sima-neat/models');

  assert.deepEqual(runs.map((item) => item.id), ['resnet-develop', 'superseded']);
  // One query for the shared commit; the known yolo run is not fetched again.
  assert.equal(paths.filter((path) => path.includes('commit_sha=')).length, 1);
  assert.deepEqual(paths.filter((path) => path.startsWith('/v1/runs/')), ['/v1/runs/superseded']);
});


test('completed candidates never replace an on-demand catalog approved build or package', () => {
  const candidate = run('candidate', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const approved = {...catalog[0], build_on_demand: true};
  const models = branchModels([candidate, resnet], [], [approved]);
  assert.equal(models[0].builds[0].run_id, resnet.id);
  assert.equal(models[0].packages[0].run_id, 'resnet-package');
});

test('a selected feature branch shows its successful on-demand build without promotion', () => {
  const definition = {model_id: 'qwen', display_name: 'Qwen', build_on_demand: true, builds: [], packages: []};
  const candidate = run('candidate', 'qwen', feature, '2026-10-02T00:00:00Z');
  const branch = 'feat/ui';
  assert.deepEqual(branchModels([], [], [definition], '', branch), []);
  assert.deepEqual(branchModels([{...candidate, status: 'failed'}], [], [definition], '', branch), []);
  assert.deepEqual(branchModels([candidate], [], [definition]), []);
  for (const other of ['develop', 'main', 'feat/other']) {
    assert.deepEqual(branchModels([candidate], [], [definition], '', other), []);
  }
  const [model] = branchModels([candidate], [], [definition], '', branch);
  assert.equal(model.builds[0].run_id, candidate.id);
  assert.deepEqual(definition.builds, []);
  assert.deepEqual(branchModels([candidate], [], [definition], '2.1.3', branch), []);
  assert.equal(branchModels([candidate], [], [definition], '3.0.0', branch)[0].builds[0].run_id, candidate.id);
});

test('feature branch candidates replace inherited builds and keep their matching package', () => {
  const candidate = run('candidate', 'resnet_50', feature, '2026-10-02T00:00:00Z');
  const definition = {...catalog[0], build_on_demand: true};
  const packageRun = {id: 'candidate-package', git_ref: feature, status: 'completed', run_type: 'model_package',
    metadata: {model_id: 'resnet_50', build_profile_id: 'int8', component_run_ids: {model: candidate.id}}};
  const [model] = branchModels([candidate, resnet], [packageRun], [definition], '', 'feat/ui');
  assert.equal(model.builds[0].run_id, candidate.id);
  assert.deepEqual(model.packages.map(item => item.run_id), [packageRun.id]);
  const [developModel] = branchModels([candidate, resnet], [packageRun], [definition], '', 'develop');
  assert.equal(developModel.builds[0].run_id, resnet.id);
  assert.deepEqual(developModel.packages.map(item => item.run_id), ['resnet-package']);
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
