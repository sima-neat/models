import assert from 'node:assert/strict';
import test from 'node:test';
import {portalConfig} from '../../website/scripts/portal-config.mjs';

test('Every branch and pull request builds the developer staging UI', () => {
  for (const ref of ['refs/heads/main', 'refs/heads/develop', 'refs/heads/feature/ui', 'refs/pull/1/merge']) {
    const config = portalConfig({GITHUB_REF: ref});
    assert.equal(config.uiMode, 'developer');
    assert.equal(config.apiBase, 'https://models-registry.stg.neat.sima.ai');
    assert.equal(config.defaultBranch, 'develop');
    assert.equal(config.siteUrl, 'https://developer-stg.sima.ai');
    assert.equal(config.environment, 'Staging');
  }
});

test('Selecting the customer audience changes the UI without changing the backend or publication site', () => {
  const developer = portalConfig({});
  const customer = portalConfig({MODEL_BROWSER_UI_MODE: 'customer'});
  assert.equal(customer.uiMode, 'customer');
  assert.equal(customer.defaultBranch, 'main');
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
    assert.equal(config.heroCdnBaseUrl, uiMode === 'developer' ? 'https://models-registry.neat.sima.ai/images/sha256' : '');
    assert.equal(config.defaultBranch, uiMode === 'customer' ? 'main' : 'develop');
  }
  assert.throws(() => portalConfig({MODEL_BROWSER_ENVIRONMENT: 'prodution'}), /Unsupported/);
});


test('Deployment identity survives a staging preview of production data', async () => {
  const {resolveEnvironment} = await import('../../artifact-browser/environment.js');
  const staged = {...portalConfig({}), build: {branch: 'feature/browser', commit: 'abc123'}};
  const preview = resolveEnvironment(staged, '?env=prod&branch=main');
  assert.equal(preview.deploymentEnvironment, 'staging');
  assert.equal(preview.environment, 'Production');
  assert.equal(preview.build.branch, 'feature/browser');
  assert.equal(portalConfig({MODEL_BROWSER_ENVIRONMENT: 'production'}).deploymentEnvironment, 'production');
});

test('A customer build carries the catalog snapshot it pins, and every other build reads the current catalog', () => {
  const revision = 'a'.repeat(40);
  const canonicalSha256 = 'b'.repeat(64);
  const pinned = portalConfig({MODEL_BROWSER_UI_MODE: 'customer', MODEL_BROWSER_CATALOG_SNAPSHOT: `${revision}:${canonicalSha256}`});
  assert.deepEqual(pinned.catalogSnapshot, {revision, canonicalSha256});
  assert.equal(portalConfig({MODEL_BROWSER_UI_MODE: 'customer', MODEL_BROWSER_CATALOG_SNAPSHOT: ''}).catalogSnapshot, null);
  assert.equal(portalConfig({}).catalogSnapshot, null);
});

test('A catalog snapshot names a full revision and hash, and pins only the customer view', () => {
  const snapshot = `${'a'.repeat(40)}:${'b'.repeat(64)}`;
  for (const malformed of ['a'.repeat(40), `${'a'.repeat(12)}:${'b'.repeat(64)}`, `${'A'.repeat(40)}:${'b'.repeat(64)}`, `${snapshot}\nX=1`]) {
    assert.throws(() => portalConfig({MODEL_BROWSER_UI_MODE: 'customer', MODEL_BROWSER_CATALOG_SNAPSHOT: malformed}), /catalog snapshot must be/);
  }
  assert.throws(() => portalConfig({MODEL_BROWSER_CATALOG_SNAPSHOT: snapshot}), /only the customer view/);
});

test('Each environment reports to its own Developer Center GA4 stream on its own host', () => {
  const staging = portalConfig({});
  assert.deepEqual(staging.analytics, {measurementId: 'G-T1D371DW1K', host: 'developer-stg.sima.ai'});
  const production = portalConfig({MODEL_BROWSER_ENVIRONMENT: 'production'});
  assert.deepEqual(production.analytics, {measurementId: 'G-Y6YREB4EG5', host: 'developer.sima.ai'});
});

test('Analytics can be disabled, pointed at a local host, and rejects a malformed ID', () => {
  assert.equal(portalConfig({MODEL_BROWSER_GA_MEASUREMENT_ID: ''}).analytics.measurementId, '');
  assert.equal(portalConfig({MODEL_BROWSER_ANALYTICS_HOST: '127.0.0.1'}).analytics.host, '127.0.0.1');
  assert.throws(() => portalConfig({MODEL_BROWSER_GA_MEASUREMENT_ID: 'UA-123'}), /GA4 measurement ID/);
});
