// CI supplies GITHUB_REF. Local previews can select either audience explicitly.
export function portalConfig(env = process.env) {
  const uiMode = env.MODEL_BROWSER_UI_MODE || (env.GITHUB_REF === 'refs/heads/main' ? 'customer' : 'developer');
  if (!['customer', 'developer'].includes(uiMode)) throw new Error(`Unsupported model browser UI mode: ${uiMode}`);
  const customer = uiMode === 'customer';
  const environment = env.MODEL_BROWSER_ENVIRONMENT || 'staging';
  if (!['staging', 'production'].includes(environment)) throw new Error(`Unsupported model browser environment: ${environment}`);
  const production = environment === 'production';
  const apiBase = env.MODEL_REGISTRY_API_URL || (production ? 'https://models-registry.neat.sima.ai' : 'https://models-registry.stg.neat.sima.ai');
  if (!/^https:\/\/[^/]+\/?$/.test(apiBase) && !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(apiBase)) {
    throw new Error('The model registry API URL must be an HTTPS origin (or localhost for previews).');
  }
  // Standard developer deployments use their own validated image CDN.
  // Empty override rolls back to API URLs; customer/custom-origin builds stay opt-in.
  const standardHeroOrigins = ['https://models-registry.stg.neat.sima.ai', 'https://models-registry.neat.sima.ai'];
  const heroCdnBaseUrl = env.MODEL_HERO_CDN_BASE_URL ?? (!customer && standardHeroOrigins.includes(apiBase.replace(/\/$/, ''))
    ? `${apiBase.replace(/\/$/, '')}/images/sha256` : '');
  if (heroCdnBaseUrl && !/^https:\/\/[^/?#]+(?:\/[^?#]*)?$/.test(heroCdnBaseUrl)) {
    throw new Error('Hero CDN base URL must be HTTPS without a query or fragment.');
  }
  return {
    uiMode,
    deploymentEnvironment: environment,
    heroCdnBaseUrl,
    apiBase: apiBase.replace(/\/$/, ''),
    environment: production ? 'Production' : 'Staging',
    allowProductionPreview: !production && apiBase.replace(/\/$/, '') === 'https://models-registry.stg.neat.sima.ai',
    defaultBranch: customer ? 'main' : 'develop',
    siteUrl: production ? 'https://developer.sima.ai' : 'https://developer-stg.sima.ai',
    repository: {organization: 'sima-neat', name: 'models'},
  };
}
