import assert from 'node:assert/strict';
import test from 'node:test';
import {resolveEnvironment, branchPolicy} from '../../artifact-browser/environment.js';
import {portalConfig} from '../../website/scripts/portal-config.mjs';

test('staging is unchanged by default or unrecognized environment values', () => {
  const config = portalConfig({});
  for (const query of ['', '?env=staging', '?env=evil', '?env=https://evil.test']) {
    assert.equal(resolveEnvironment(config, query), config);
  }
});

test('production preview changes the whole backend without changing audience or repository', () => {
  for (const uiMode of ['customer', 'developer']) {
    const base = portalConfig({MODEL_BROWSER_UI_MODE: uiMode});
    assert.equal(base.uiMode, uiMode);
    const config = resolveEnvironment(base, '?env=prod&branch=develop');
    assert.equal(config.apiBase, 'https://models-registry.neat.sima.ai');
    assert.equal(config.environment, 'Production');
    assert.equal(config.heroCdnBaseUrl, '');
    assert.equal(config.uiMode, base.uiMode);
    assert.deepEqual(config.repository, base.repository);
    assert.equal(base.environment, 'Staging');
  }
});

test('production and custom/local configurations do not opt into query switching', () => {
  for (const config of [
    portalConfig({MODEL_REGISTRY_API_URL: 'http://localhost:8767'}),
    {environment: 'Production', apiBase: 'https://production.test', allowProductionPreview: true},
    {environment: 'Staging', apiBase: 'https://custom.test'},
  ]) assert.equal(resolveEnvironment(config, '?env=prod'), config);
});


test('branches=0 selects Main without changing audience or environment', () => {
  const config = portalConfig({});
  assert.deepEqual(branchPolicy(config, '?branches=0&branch=develop'), {enabled: false, defaultBranch: 'main'});
  assert.deepEqual(branchPolicy(config, '?env=prod&branches=0'), {enabled: false, defaultBranch: 'main'});
  for (const query of ['', '?branches=1', '?branches=no']) {
    assert.deepEqual(branchPolicy(config, query), {enabled: true, defaultBranch: 'develop'});
  }
  assert.deepEqual(branchPolicy({uiMode: 'customer'}, '?branches=1'), {enabled: false, defaultBranch: 'main'});
});
