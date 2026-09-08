import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

const source = await readFile(new URL('../../artifact-browser/downloads.js', import.meta.url), 'utf8');
const {variantDownloads, prepareDownloads, startDownloads} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const run = {
  id: 'transformer-run', repository: 'sima-neat/models', git_ref: 'refs/heads/develop',
  commit_sha: 'commit', toolchain_name: 'compiler', toolchain_version: '2.1.3', target_platform: 'modalix',
  metadata: {model_id: 'rf_detr_small', build_profile_id: 'mixed', component_id: 'transformer', github_run_id: 'workflow'},
};
const backbone = {...run, id: 'backbone-run', metadata: {...run.metadata, component_id: 'backbone'}};
const summary = (item) => ({run: item, artifacts: [
  {name: 'mpk', artifact_type: 'model-pack', size_bytes: 1234},
  {name: 'numerical-fidelity', artifact_type: 'numerical-fidelity'},
]});

test('collects every component across pages, retaining selected run and excluding other build contexts', async () => {
  const other = {...backbone, id: 'wrong-compiler', toolchain_version: '9'};
  const anotherWorkflow = {...backbone, id: 'wrong-workflow', metadata: {...backbone.metadata, github_run_id: 'other'}};
  const old = {...backbone, id: 'wrong-commit', commit_sha: 'old'};
  const requested = [];
  const result = await variantDownloads(run, async (path) => {
    requested.push(path);
    if (path.includes('/repositories/')) return path.includes('cursor=next')
      ? {runs: [backbone, other, anotherWorkflow, old], has_more: false}
      : {runs: [run], has_more: true, next_cursor: 'next'};
    assert.ok(['/v1/runs/backbone-run', '/v1/runs/transformer-run'].includes(path));
    return summary(run);
  });
  assert.deepEqual(result.map(({runId, component, name}) => ({runId, component, name})), [
    {runId: 'backbone-run', component: 'backbone', name: 'mpk'},
    {runId: 'transformer-run', component: 'transformer', name: 'mpk'},
  ]);
  assert.equal(requested.filter(path => path.includes('/repositories/')).length, 2);
});

test('single-component models retain their one file', async () => {
  const result = await variantDownloads(run, async path => path.includes('/repositories/') ? {runs: [run]} : summary(run));
  assert.equal(result.length, 1);
  assert.equal(result[0].runId, run.id);
});

test('missing component artifact fails instead of silently downloading a partial variant', async () => {
  await assert.rejects(variantDownloads(run, async path => {
    if (path.includes('/repositories/')) return {runs: [run, backbone]};
    return path.endsWith('backbone-run') ? {artifacts: []} : summary(run);
  }), /No model file is available for backbone/);
});

test('requests all URLs concurrently and does not finish until every URL is ready', async () => {
  const requests = [];
  const releases = [];
  const pending = prepareDownloads([
    {runId: 'one', name: 'model pack', component: 'backbone'},
    {runId: 'two', name: 'mpk', component: 'transformer'},
  ], path => {
    requests.push(path);
    return new Promise(resolve => releases.push(resolve));
  });
  assert.equal(requests.length, 2);
  assert.match(requests[0], /model%20pack$/);
  releases.forEach((resolve, index) => resolve({artifact: {download_url: `https://example.com/${index}`}}));
  assert.equal((await pending).length, 2);
});

test('a failed URL prevents starting the batch', async () => {
  let started = false;
  await assert.rejects(async () => {
    const ready = await prepareDownloads([{runId: 'one', name: 'mpk', component: 'backbone'}], async () => ({artifact: {}}));
    started = true;
    return ready;
  }, /No download URL for backbone/);
  assert.equal(started, false);
});

test('each file gets an independent download context without navigating the page', () => {
  const frames = [];
  startDownloads([{component: 'backbone', url: '/one'}, {component: 'transformer', url: '/two'}], {
    createElement: tag => { assert.equal(tag, 'iframe'); return {}; },
    body: {append: frame => frames.push(frame)},
  });
  assert.deepEqual(frames.map(frame => frame.src), ['/one', '/two']);
  assert.ok(frames.every(frame => frame.hidden));
});


test('shows the actual archive filename instead of the logical artifact name', async () => {
  const items = [{runId: 'one', name: 'mpk', component: 'backbone'}];
  const [archive] = await prepareDownloads(items, async () => ({artifact: {
    s3_key: 'models/develop/backbone/rf_detr_small_backbone_mpk.tar.gz',
    download_url: 'https://example.com/download?signature=private',
  }}));
  assert.equal(archive.filename, 'rf_detr_small_backbone_mpk.tar.gz');
  assert.equal(archive.name, 'mpk');
  const [fromUrl] = await prepareDownloads(items, async () => ({artifact: {
    download_url: 'https://example.com/models/model%20pack.tar.gz?signature=private',
  }}));
  assert.equal(fromUrl.filename, 'model pack.tar.gz');
});
