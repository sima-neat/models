import assert from 'node:assert/strict';
import test from 'node:test';
import {portalConfig} from '../../website/scripts/portal-config.mjs';

test('Main builds the customer UI against the shared staging registry', () => {
  const config = portalConfig({GITHUB_REF: 'refs/heads/main'});
  assert.equal(config.uiMode, 'customer');
  assert.equal(config.apiBase, 'https://models-registry.stg.neat.sima.ai');
  assert.equal(config.defaultBranch, 'main');
  assert.equal(config.siteUrl, 'https://developer-stg.sima.ai');
  assert.equal(config.environment, 'Staging');
});

test('Develop, feature branches and pull requests keep the developer staging UI', () => {
  for (const ref of ['refs/heads/develop', 'refs/heads/feature/ui', 'refs/pull/1/merge']) {
    const config = portalConfig({GITHUB_REF: ref});
    assert.equal(config.uiMode, 'developer');
    assert.equal(config.apiBase, 'https://models-registry.stg.neat.sima.ai');
    assert.equal(config.defaultBranch, 'develop');
    assert.equal(config.siteUrl, 'https://developer-stg.sima.ai');
  }
});

test('Changing the build branch changes the UI without changing the backend or publication site', () => {
  const developer = portalConfig({GITHUB_REF: 'refs/heads/develop'});
  const customer = portalConfig({GITHUB_REF: 'refs/heads/main'});
  assert.notEqual(developer.uiMode, customer.uiMode);
  assert.equal(developer.apiBase, customer.apiBase);
  assert.equal(developer.siteUrl, customer.siteUrl);
  assert.equal(developer.environment, customer.environment);
});

test('Local previews can build either UI against a local API', () => {
  const config = portalConfig({MODEL_BROWSER_UI_MODE: 'customer', MODEL_REGISTRY_API_URL: 'http://localhost:8767'});
  assert.equal(config.uiMode, 'customer');
  assert.equal(config.apiBase, 'http://localhost:8767');
  assert.throws(() => portalConfig({MODEL_BROWSER_UI_MODE: 'typo'}), /Unsupported/);
});


test('Production selects its public reader and site independently of audience', () => {
  for (const uiMode of ['customer', 'developer']) {
    const config = portalConfig({MODEL_BROWSER_ENVIRONMENT: 'production', MODEL_BROWSER_UI_MODE: uiMode});
    assert.equal(config.apiBase, 'https://models-registry.neat.sima.ai');
    assert.equal(config.siteUrl, 'https://developer.sima.ai');
    assert.equal(config.environment, 'Production');
    assert.equal(config.allowProductionPreview, false);
    assert.equal(config.heroCdnBaseUrl, '');
    assert.equal(config.defaultBranch, uiMode === 'customer' ? 'main' : 'develop');
  }
  assert.throws(() => portalConfig({MODEL_BROWSER_ENVIRONMENT: 'prodution'}), /Unsupported/);
});
