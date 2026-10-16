import assert from 'node:assert/strict';
import {test} from 'node:test';

const {variantDownloads, prepareDownloads, startDownloads, packageFormat, artifactDelivery, deploymentSize, huggingFaceLocation, cliDownloadCommand} = await import(new URL('../../artifact-browser/downloads.js', import.meta.url));
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

test('approved downloads use exact selected roots without rediscovering newer component runs', async () => {
  const catalog = {model_id: 'rf_detr_small', builds: [], approved_deployments: {
    mixed: {artifacts: [
      {component_id: 'transformer', run_id: run.id, artifact_name: 'llima', layout: 'directory', format: 'llima'},
      {component_id: 'backbone', run_id: backbone.id, artifact_name: 'mpk', layout: 'archive', format: 'mpk'},
    ]},
  }};
  const downloads = await variantDownloads(run, async path => {
    assert.ok(!path.includes('/repositories/'));
    const item = path.endsWith(backbone.id) ? backbone : run;
    return {run: item, artifacts: [
      {name: item === backbone ? 'mpk' : 'llima', artifact_type: 'model-pack'},
      {name: 'other-root', artifact_type: 'model-pack'},
    ]};
  }, catalog);
  assert.deepEqual(downloads.map(item => item.name), ['mpk', 'llima']);
});

test('a historical download keeps its requested run when the current catalog approves a different candidate', async () => {
  const catalog = {model_id: 'rf_detr_small', approved_deployments: {mixed: {artifacts: [
    {component_id: 'transformer', run_id: 'current-approved', artifact_name: 'llima'},
  ]}}};
  const downloads = await variantDownloads(run, async path => {
    assert.ok(!path.includes('current-approved'));
    return path.includes('/repositories/') ? {runs: [run]} : summary(run);
  }, catalog);
  assert.equal(downloads[0].runId, run.id);
});

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


test('a single-component package is an MPK and any other package is a ZIP', () => {
  assert.equal(packageFormat({media_type: 'application/gzip'}), 'MPK');
  assert.equal(packageFormat({}, 'yolo26n_w8a8_mpk.tar.gz'), 'MPK');
  assert.equal(packageFormat({media_type: 'application/zip'}, 'rf_detr_nano_w8a8_w16a16_default.zip'), 'ZIP');
});


test('previously published LLiMa ZIPs keep their stored filename', async () => {
  const native = {...run, id: 'llima-run', metadata: {...run.metadata, model_id: 'qwen3_5_0_8b',
    build_profile_id: 'w4a16', component_id: 'model', artifact_kind: 'llima'}};
  const filename = 'qwen3_5_0_8b_w4a16_llima.zip';
  const items = await variantDownloads(native, async path => path.includes('/repositories/')
    ? {runs: [native]} : {artifacts: [{name: 'llima', artifact_type: 'model-pack', size_bytes: 4096}]});
  assert.equal(items[0].delivery, 'archive');
  const ready = await prepareDownloads(items, async path => {
    assert.equal(path, '/v1/runs/llima-run/artifacts/llima');
    return {artifact: {download_url: `https://example.com/${filename}`}};
  });
  assert.equal(ready[0].filename, filename);
});


const folder = {
  name: 'llima', artifact_type: 'model-pack', media_type: 'application/json', size_bytes: 2048,
  metadata: {layout: 'directory', storage_backend: 's3', deployment_size_bytes: 4_000_000_000, file_count: 301},
};

test('native folder uses aggregate deployment size and never requests an archive name', async () => {
  const native = {...run, id: 'llima-run', metadata: {...run.metadata, component_id: 'model', artifact_kind: 'llima'}};
  const catalog = {model_id: 'rf_detr_small', builds: [{run_id: native.id, component_id: 'model',
    build_profile_id: 'mixed'}]};
  const items = await variantDownloads(native, async path => path.includes('/repositories/')
    ? {runs: [native]} : {artifacts: [folder, {name: 'file-layer', artifact_type: 'model-file'}]}, catalog);
  assert.equal(items.length, 1);
  assert.equal(items[0].size, 4_000_000_000);
  assert.equal(items[0].delivery, 'directory');
  const ready = await prepareDownloads(items, async path => {
    assert.ok(!path.includes('download_filename'));
    return {artifact: {download_url: 'https://example.com/deployment-index.json'}};
  });
  assert.equal(ready[0].delivery, 'directory');
  assert.equal(ready[0].url, null);
  let frames = 0;
  assert.throws(() => startDownloads(ready, {createElement: () => { frames += 1; }}), /Folder deployments/);
  assert.equal(frames, 0);
});

test('resolved storage location determines Hugging Face delivery independently of source and S3', async () => {
  const revision = 'a'.repeat(40);
  const [ready] = await prepareDownloads([{runId: run.id, name: 'llima', component: 'model', artifact: folder}], async () => ({
    artifact: {metadata: {storage_backend: 'huggingface', huggingface: {repo_id: 'sima-ai/compiled', revision, subdirectory: 'a16w4/context 4096'}}},
  }));
  assert.equal(ready.delivery, 'huggingface');
  assert.equal(ready.url, null);
  assert.equal(huggingFaceLocation(ready.artifact).url,
    `https://huggingface.co/sima-ai/compiled/tree/${revision}/a16w4/context%204096`);
  assert.equal(ready.size, 4_000_000_000);
  assert.throws(() => startDownloads([ready], {}), /Folder deployments/);
});

test('Hugging Face deployment links require immutable revisions and safe repository paths', () => {
  for (const location of [
    {repo_id: 'sima-ai/compiled', revision: 'main'},
    {repo_id: 'https://example.com/model', revision: 'a'.repeat(40)},
    {repo_id: 'sima-ai/compiled', revision: 'a'.repeat(40), subdirectory: '../source'},
    {repo_id: 'sima-ai/compiled', revision: 'a'.repeat(40), subdirectory: '/source'},
    {repo_id: 'sima-ai/compiled', revision: 'a'.repeat(40), subdirectory: 'x\\y'},
  ]) assert.equal(huggingFaceLocation({metadata: {huggingface: location}}), null);
});

test('JSON indexes are folders even when older metadata omitted the explicit layout', () => {
  assert.equal(artifactDelivery({media_type: 'application/json; charset=utf-8', size_bytes: 40}), 'directory');
  assert.equal(deploymentSize({media_type: 'application/json', size_bytes: 40}), undefined);
  assert.equal(artifactDelivery({media_type: 'application/zip'}), 'archive');
  assert.equal(cliDownloadCommand(run), 'sima-cli models download --id rf_detr_small --variant mixed --branch develop');
});


test('an approved deployment command targets the selected catalog branch and keeps source provenance', () => {
  const source = {...run, git_ref: 'refs/heads/feature/native-candidate'};
  assert.equal(cliDownloadCommand(source, 'develop'),
    'sima-cli models download --id rf_detr_small --variant mixed --branch develop');
  assert.equal(source.git_ref, 'refs/heads/feature/native-candidate');
});
