import assert from 'node:assert/strict';
import {test} from 'node:test';

const {deploymentCatalog, branchModels, catalogBuildRuns} = await import(new URL('../../artifact-browser/branch-builds.js', import.meta.url));

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
  const definition = {...catalog[0], builds: [], packages: [], build_on_demand: true,
    build_requirements: {int8: {packages: [{id: 'default', component_ids: ['model']}]}},
    approved_deployments: {
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
  const [model] = branchModels(components, [], [definition], '', 'develop');
  assert.deepEqual(model.packages.map(item => item.run_id), ['complete-package']);
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

test('a profile still pending in the catalog shows its completed run', () => {
  const pending = run('resnet-pending', 'resnet_50', develop, '2026-10-02T00:00:00Z',
    {metadata: {...resnet.metadata, build_profile_id: 'bf16'}});
  const declared = [{...catalog[0], pending_builds: [{model_id: 'resnet_50', build_profile_id: 'bf16'}]}, catalog[1]];
  const [model] = branchModels([pending, resnet, yolo], [], declared);

  assert.deepEqual(model.builds.map((build) => build.build_profile_id).sort(), ['bf16', 'int8']);
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

test('feature catalogs show finalized selections rather than arbitrary completed runs', async () => {
  const candidate = run('candidate', 'qwen', feature, '2026-10-02T00:00:00Z');
  const newer = run('newer', 'qwen', feature, '2026-10-03T00:00:00Z');
  const definition = {model_id: 'qwen', display_name: 'Qwen', build_on_demand: true, builds: [], packages: []};
  assert.deepEqual(branchModels([candidate], [], [definition], '', 'feat/ui'), []);
  const selected = {...definition, branch_deployments: {int8: {artifacts: [
    {component_id: 'model', run_id: candidate.id, artifact_name: 'llima', layout: 'directory', format: 'llima'},
  ]}}};
  const resolved = await catalogBuildRuns([selected], [newer], async path => {
    assert.equal(path, `/v1/runs/${candidate.id}`);
    return {run: candidate};
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
    assert.equal(path, '/v1/runs/resnet-develop');
    return {run: resnet};
  }, '/repo');
  const newer = {...resnet, id: 'newer', finished_at: '2026-10-17T00:00:00Z'};
  const [model] = branchModels([newer, ...inherited], [], definitions);
  assert.equal(model.builds[0].run_id, resnet.id);
  assert.equal(model.packages[0].run_id, 'folder-package');
  assert.deepEqual(model.packages[0].component_run_ids, {model: resnet.id});
  assert.equal(branchModels([resnet], [], definitions, 'other').length, 0);
});


for (const field of ['branch_deployments', 'approved_deployments']) {
  test(`${field} retains accuracy only from the exact selected component build`, () => {
    const selection = {artifacts: [{run_id: resnet.id, component_id: 'model', artifact_name: 'mpk'}]};
    const exact = {...catalogBuild(resnet), validation_fingerprint: 'measured-accuracy'};
    const mismatches = [
      {...exact, run_id: 'other-run', validation_fingerprint: 'wrong-run'},
      {...exact, build_profile_id: 'bf16', validation_fingerprint: 'wrong-profile'},
      {...exact, component_id: 'backbone', validation_fingerprint: 'wrong-component'},
    ];
    const definition = {model_id: 'resnet_50', builds: [...mismatches, exact], [field]: {int8: selection}};
    const [model] = branchModels([resnet], [], [definition]);
    assert.equal(model.builds[0].run_id, resnet.id);
    assert.equal(model.builds[0].validation_fingerprint, 'measured-accuracy');
    const [unmeasured] = branchModels([resnet], [], [{...definition, builds: mismatches}]);
    assert.equal(unmeasured.builds[0].validation_fingerprint, undefined);
  });
}
