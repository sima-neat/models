import assert from 'node:assert/strict';
import test from 'node:test';
import {analyticsEnabled, parseConsent, CONSENT_KEY} from '../../artifact-browser/analytics.js';

test('Analytics runs only with a measurement ID on the configured host', () => {
  const analytics = {measurementId: 'G-T1D371DW1K', host: 'developer-stg.sima.ai'};
  assert.equal(analyticsEnabled(analytics, 'developer-stg.sima.ai'), true);
  assert.equal(analyticsEnabled(analytics, '127.0.0.1'), false);
  assert.equal(analyticsEnabled({...analytics, measurementId: ''}, 'developer-stg.sima.ai'), false);
  assert.equal(analyticsEnabled(undefined, 'developer-stg.sima.ai'), false);
});

test('The consent record matches the one the core documentation writes', () => {
  assert.equal(CONSENT_KEY, 'sima-developer-center-cookie-consent');
  const stored = JSON.stringify({version: 1, analytics: true, marketing: false, updatedAt: '2026-11-17T00:00:00Z'});
  assert.equal(parseConsent(stored).analytics, true);
  assert.equal(parseConsent(JSON.stringify({version: 2, analytics: true})), null);
  assert.equal(parseConsent('not json'), null);
  assert.equal(parseConsent(null), null);
});
