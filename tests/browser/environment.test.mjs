import assert from 'node:assert/strict';
import test from 'node:test';
import {resolveEnvironment} from '../../artifact-browser/environment.js';
import {portalConfig} from '../../website/scripts/portal-config.mjs';

test('staging is unchanged by default or unrecognized environment values', () => {
  const config = portalConfig({});
  for (const query of ['', '?env=staging', '?env=evil', '?env=https://evil.test']) {
    assert.equal(resolveEnvironment(config, query), config);
  }
});

test('production preview changes the whole backend without changing audience or repository', () => {
  for (const ref of ['refs/heads/main', 'refs/heads/develop']) {
    const base = portalConfig({GITHUB_REF: ref});
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
