// CI supplies GITHUB_REF. Local previews can select either audience explicitly.
export function portalConfig(env = process.env) {
  const uiMode = env.MODEL_BROWSER_UI_MODE || (env.GITHUB_REF === 'refs/heads/main' ? 'customer' : 'developer');
  if (!['customer', 'developer'].includes(uiMode)) throw new Error(`Unsupported model browser UI mode: ${uiMode}`);
  const customer = uiMode === 'customer';
  const apiBase = env.MODEL_REGISTRY_API_URL || 'https://models-registry.stg.neat.sima.ai';
  if (!/^https:\/\/[^/]+\/?$/.test(apiBase) && !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(apiBase)) {
    throw new Error('The model registry API URL must be an HTTPS origin (or localhost for previews).');
  }
  return {
    uiMode,
    apiBase: apiBase.replace(/\/$/, ''),
    environment: 'Staging',
    defaultBranch: customer ? 'main' : 'develop',
    siteUrl: 'https://developer-stg.sima.ai',
    repository: {organization: 'sima-neat', name: 'models'},
  };
}
