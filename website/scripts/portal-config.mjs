import {readFileSync} from 'node:fs';

// The public registry endpoint of each Vulcan environment, the same file the Python tooling reads.
const registry = JSON.parse(readFileSync(new URL('../../config/registry.json', import.meta.url), 'utf8'));
const publicUrl = (environment) => registry.environments[environment].public_url;

// Every branch builds the developer view. A deployment or local preview names the customer audience explicitly.
export function portalConfig(env = process.env) {
  const uiMode = env.MODEL_BROWSER_UI_MODE || 'developer';
  if (!['customer', 'developer'].includes(uiMode)) throw new Error(`Unsupported model browser UI mode: ${uiMode}`);
  const customer = uiMode === 'customer';
  const environment = env.MODEL_BROWSER_ENVIRONMENT || 'staging';
  if (!['staging', 'production'].includes(environment)) throw new Error(`Unsupported model browser environment: ${environment}`);
  const production = environment === 'production';
  const apiBase = env.MODEL_REGISTRY_API_URL || publicUrl(environment);
  if (!/^https:\/\/[^/]+\/?$/.test(apiBase) && !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(apiBase)) {
    throw new Error('The model registry API URL must be an HTTPS origin (or localhost for previews).');
  }
  // Standard developer deployments use their own validated image CDN.
  // Empty override rolls back to API URLs; customer/custom-origin builds stay opt-in.
  const standardHeroOrigins = [publicUrl('staging'), publicUrl('production')];
  const heroCdnBaseUrl = env.MODEL_HERO_CDN_BASE_URL ?? (!customer && standardHeroOrigins.includes(apiBase.replace(/\/$/, ''))
    ? `${apiBase.replace(/\/$/, '')}/images/sha256` : '');
  if (heroCdnBaseUrl && !/^https:\/\/[^/?#]+(?:\/[^?#]*)?$/.test(heroCdnBaseUrl)) {
    throw new Error('Hero CDN base URL must be HTTPS without a query or fragment.');
  }
  // GA4 web streams of the Developer Center property, one per environment. Empty disables analytics.
  const siteUrl = production ? 'https://developer.sima.ai' : 'https://developer-stg.sima.ai';
  const gaMeasurementId = (env.MODEL_BROWSER_GA_MEASUREMENT_ID ?? (production ? 'G-Y6YREB4EG5' : 'G-T1D371DW1K')).trim();
  if (gaMeasurementId && !/^G-[A-Z0-9]+$/.test(gaMeasurementId)) {
    throw new Error('The GA4 measurement ID must look like G-XXXXXXXXXX, or be empty to disable analytics.');
  }
  // Analytics runs only on the site's own host, so local previews and smoke tests never report.
  const analyticsHost = (env.MODEL_BROWSER_ANALYTICS_HOST || new URL(siteUrl).hostname).trim();
  return {
    uiMode,
    deploymentEnvironment: environment,
    analytics: {measurementId: gaMeasurementId, host: analyticsHost},
    heroCdnBaseUrl,
    apiBase: apiBase.replace(/\/$/, ''),
    environment: production ? 'Production' : 'Staging',
    allowProductionPreview: !production && apiBase.replace(/\/$/, '') === publicUrl('staging'),
    productionApiBase: publicUrl('production'),
    defaultBranch: customer ? 'main' : 'develop',
    siteUrl,
    repository: {organization: 'sima-neat', name: 'models'},
  };
}
