import assert from 'node:assert/strict';
import {test} from 'node:test';

const {variantBuildKey, variantBuildRuns, variantBuilds, workflowIdentity} = await import(new URL('../../artifact-browser/branch-builds.js', import.meta.url));
const backbone = {
  id: 'backbone-run', repository: 'sima-neat/models', git_ref: 'refs/heads/develop', commit_sha: 'commit',
  toolchain_name: 'compiler', toolchain_version: '2.1.3', target_platform: 'modalix', created_at: '2026-01-01T00:00:00Z',
  metadata: {model_id: 'rf_detr_small', build_profile_id: 'mixed', component_id: 'backbone', github_run_id: 'workflow'},
};
const transformer = {...backbone, id: 'transformer-run', metadata: {...backbone.metadata, component_id: 'transformer'}};
const resnet = {
  ...backbone, id: 'resnet-run',
  metadata: {model_id: 'resnet_50', build_profile_id: 'w8a8', component_id: 'model', github_run_id: 'workflow'},
};

test('workflow identity falls back from GitHub run to CI URL, compile invocation and run id', () => {
  assert.equal(workflowIdentity({id: 'r', ci_run_url: 'u', metadata: {github_run_id: 'g', compile_invocation_id: 'c'}}), 'g');
  assert.equal(workflowIdentity({id: 'r', ci_run_url: 'u', metadata: {compile_invocation_id: 'c'}}), 'u');
  assert.equal(workflowIdentity({id: 'r', metadata: {compile_invocation_id: 'c'}}), 'c');
  assert.equal(workflowIdentity({id: 'r'}), 'r');
});

test('groups the components of one build and keeps single-component builds separate', () => {
  const groups = variantBuilds([transformer, resnet, backbone]);
  assert.deepEqual(groups.map(({modelId, profile, runs}) => ({modelId, profile, runs: runs.map(run => run.id)})), [
    {modelId: 'rf_detr_small', profile: 'mixed', runs: ['backbone-run', 'transformer-run']},
    {modelId: 'resnet_50', profile: 'w8a8', runs: ['resnet-run']},
  ]);
  assert.equal(groups[0].key, variantBuildKey(backbone));
});

test('a different workflow, commit or compiler is a different build', () => {
  const otherWorkflow = {...transformer, id: 'w', metadata: {...transformer.metadata, github_run_id: 'other'}};
  const otherCommit = {...transformer, id: 'c', commit_sha: 'old'};
  const otherCompiler = {...transformer, id: 't', toolchain_version: '2.1.2'};
  assert.equal(variantBuilds([backbone, otherWorkflow, otherCommit, otherCompiler]).length, 4);
});

test('the model component anchors a build, else the first component; a rerun replaces its component', () => {
  assert.equal(variantBuilds([transformer, backbone])[0].anchor.id, 'backbone-run');
  const model = {...transformer, id: 'model-run', metadata: {...transformer.metadata, component_id: 'model'}};
  assert.equal(variantBuilds([backbone, model])[0].anchor.id, 'model-run');
  const rerun = {...backbone, id: 'backbone-rerun', created_at: '2026-01-02T00:00:00Z'};
  assert.deepEqual(variantBuilds([backbone, transformer, rerun])[0].runs.map(run => run.id), ['backbone-rerun', 'transformer-run']);
});

test('either component run resolves to the same build', async () => {
  const other = {...backbone, id: 'other-workflow', metadata: {...backbone.metadata, github_run_id: 'other'}};
  const fetchPage = async (path) => {
    assert.match(path, /^\/v1\/repositories\/sima-neat\/models\/runs\?/);
    assert.match(path, /q=rf_detr_small/);
    return {runs: [transformer, backbone, other], has_more: false};
  };
  const fromBackbone = await variantBuildRuns(backbone, fetchPage);
  const fromTransformer = await variantBuildRuns(transformer, fetchPage);
  assert.deepEqual(fromBackbone.map(run => run.id), ['backbone-run', 'transformer-run']);
  assert.deepEqual(fromTransformer.map(run => run.id), ['backbone-run', 'transformer-run']);
});

test('the requested run stands for its component even when a rerun is newer', async () => {
  const rerun = {...backbone, id: 'backbone-rerun', created_at: '2026-01-02T00:00:00Z'};
  const runs = await variantBuildRuns(backbone, async () => ({runs: [rerun, transformer], has_more: false}));
  assert.deepEqual(runs.map(run => run.id), ['backbone-run', 'transformer-run']);
});
