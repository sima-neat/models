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
