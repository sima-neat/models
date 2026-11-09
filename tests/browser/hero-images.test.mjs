import assert from 'node:assert/strict';
import {test} from 'node:test';
import {heroImageSources, installHeroImageRecovery} from '../../artifact-browser/hero-images.js';
import {portalConfig} from '../../website/scripts/portal-config.mjs';
const documentation = {run_id: 'run-id', hero: {artifact_name: 'hero', artifact_sha256: 'a'.repeat(64)}};
test('CDN keys use the validated digest and preserve the exact API fallback', () => {
  const sources = heroImageSources(documentation, 'https://api', 'https://cdn/images/sha256/');
  assert.equal(sources.src, `https://cdn/images/sha256/${'a'.repeat(64)}.png`);
  assert.equal(sources.fallback, 'https://api/v1/runs/run-id/artifacts/hero/content');
  assert.equal(heroImageSources(documentation, 'https://api').src, sources.fallback);
  assert.equal(heroImageSources({...documentation, hero: {...documentation.hero, artifact_sha256: '../secret'}}, 'https://api'), null);
  assert.equal(heroImageSources(null, 'https://api'), null);
});
function fixture(src = 'cdn') {
  let handler;
  const queue = [], delays = [], requests = [];
  const attributes = {src, 'data-hero-fallback': 'api'};
  const image = {tagName: 'IMG', isConnected: true, getAttribute: key => attributes[key],
    setAttribute(key, value) { attributes[key] = value; requests.push(value); }};
  installHeroImageRecovery({addEventListener(type, callback, capture) {
    assert.equal(type, 'error'); assert.equal(capture, true); handler = callback;
  }}, (callback, delay) => { queue.push(callback); delays.push(delay); });
  return {image, queue, delays, requests, fail() {handler({target: image}); queue.shift()?.();}};
}
test('CDN retries stop and fall back without an infinite API error loop', () => {
  const f = fixture();
  for (let i = 0; i < 8; i++) f.fail();
  assert.deepEqual(f.requests, ['cdn', 'cdn', 'api', 'api']);
  assert.deepEqual(f.delays, [500, 1500, 3000, 5000]);
});
test('API-only mode retries and removed images do not start new requests', () => {
  const f = fixture('api');
  for (let i = 0; i < 6; i++) f.fail();
  assert.equal(f.requests.length, 4);
  const removed = fixture(); removed.image.isConnected = false; removed.fail();
  assert.equal(removed.requests.length, 0);
});
test('standard staging and production developer builds opt in by default, with an explicit rollback', () => {
  assert.match(portalConfig({GITHUB_REF: 'refs/heads/develop'}).heroCdnBaseUrl, /stg.*images\/sha256$/);
  assert.equal(portalConfig({MODEL_BROWSER_UI_MODE: 'customer'}).heroCdnBaseUrl, '');
  assert.equal(portalConfig({MODEL_REGISTRY_API_URL: 'https://example.com'}).heroCdnBaseUrl, '');
  assert.equal(portalConfig({MODEL_HERO_CDN_BASE_URL: ''}).heroCdnBaseUrl, '');
  assert.equal(portalConfig({MODEL_BROWSER_ENVIRONMENT: 'production'}).heroCdnBaseUrl, 'https://models-registry.neat.sima.ai/images/sha256');
  assert.equal(portalConfig({MODEL_BROWSER_ENVIRONMENT: 'production', MODEL_HERO_CDN_BASE_URL: ''}).heroCdnBaseUrl, '');
  assert.equal(portalConfig({MODEL_BROWSER_ENVIRONMENT: 'production', MODEL_REGISTRY_API_URL: 'https://example.com'}).heroCdnBaseUrl, '');
  assert.throws(() => portalConfig({MODEL_HERO_CDN_BASE_URL: 'http://example.com'}), /HTTPS/);
});
