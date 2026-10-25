import assert from 'node:assert/strict';
import {test} from 'node:test';

const {variantBuildKey, variantBuildRuns, variantBuilds, workflowIdentity} = await import(new URL('../../artifact-browser/branch-builds.js', import.meta.url));
const backbone = {
  id: 'backbone-run', repository: 'sima-neat/models', git_ref: 'refs/heads/develop', commit_sha: 'commit',
  toolchain_name: 'compiler', toolchain_version: '2.1.3', target_platform: 'modalix', created_at: '2026-01-01T00:00:00Z',
  metadata: {model_id: 'rf_detr_small', build_profile_id: 'mixed', component_id: 'backbone', github_run_id: 'workflow'},
};
const decoder = {...backbone, id: 'decoder-run', metadata: {...backbone.metadata, component_id: 'decoder'}};
const resnet = {
  ...backbone, id: 'resnet-run',
  metadata: {model_id: 'resnet_50', build_profile_id: 'w8a8', component_id: 'model', github_run_id: 'workflow'},
};

test('workflow identity is the GitHub run, else the run itself', () => {
  assert.equal(workflowIdentity({id: 'r', metadata: {github_run_id: 'g'}}), 'g');
  assert.equal(workflowIdentity({id: 'r', metadata: {}}), 'r');
});

test('groups the components of one build and keeps single-component builds separate', () => {
  const groups = variantBuilds([decoder, resnet, backbone]);
  assert.deepEqual(groups.map(({modelId, profile, runs}) => ({modelId, profile, runs: runs.map(run => run.id)})), [
    {modelId: 'rf_detr_small', profile: 'mixed', runs: ['backbone-run', 'decoder-run']},
    {modelId: 'resnet_50', profile: 'w8a8', runs: ['resnet-run']},
  ]);
  assert.equal(groups[0].key, variantBuildKey(backbone));
});

test('a different workflow, commit or compiler is a different build', () => {
  const otherWorkflow = {...decoder, id: 'w', metadata: {...decoder.metadata, github_run_id: 'other'}};
  const otherCommit = {...decoder, id: 'c', commit_sha: 'old'};
  const otherCompiler = {...decoder, id: 't', toolchain_version: '2.1.2'};
  assert.equal(variantBuilds([backbone, otherWorkflow, otherCommit, otherCompiler]).length, 4);
});

test('the model component anchors a build, else the first component; a rerun replaces its component', () => {
  assert.equal(variantBuilds([decoder, backbone])[0].anchor.id, 'backbone-run');
  const model = {...decoder, id: 'model-run', metadata: {...decoder.metadata, component_id: 'model'}};
  assert.equal(variantBuilds([backbone, model])[0].anchor.id, 'model-run');
  const rerun = {...backbone, id: 'backbone-rerun', created_at: '2026-01-02T00:00:00Z'};
  assert.deepEqual(variantBuilds([backbone, decoder, rerun])[0].runs.map(run => run.id), ['backbone-rerun', 'decoder-run']);
});

test('either component run resolves to the same build', async () => {
  const other = {...backbone, id: 'other-workflow', metadata: {...backbone.metadata, github_run_id: 'other'}};
  const fetchPage = async (path) => {
    assert.match(path, /^\/v1\/repositories\/sima-neat\/models\/runs\?/);
    assert.match(path, /q=rf_detr_small/);
    return {runs: [decoder, backbone, other], has_more: false};
  };
  const fromBackbone = await variantBuildRuns(backbone, fetchPage);
  const fromDecoder = await variantBuildRuns(decoder, fetchPage);
  assert.deepEqual(fromBackbone.map(run => run.id), ['backbone-run', 'decoder-run']);
  assert.deepEqual(fromDecoder.map(run => run.id), ['backbone-run', 'decoder-run']);
});

test('the requested run stands for its component even when a rerun is newer', async () => {
  const rerun = {...backbone, id: 'backbone-rerun', created_at: '2026-01-02T00:00:00Z'};
  const runs = await variantBuildRuns(backbone, async () => ({runs: [rerun, decoder], has_more: false}));
  assert.deepEqual(runs.map(run => run.id), ['backbone-run', 'decoder-run']);
});
