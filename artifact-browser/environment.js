// Only the configured staging site may opt into the known production reader.
// Query parameters never supply an arbitrary API origin or AWS credentials.
export function resolveEnvironment(config, search = '') {
  if (!config.allowProductionPreview || config.environment?.toLowerCase() !== 'staging') return config;
  if (new URLSearchParams(search).get('env') !== 'prod') return config;
  return {
    ...config,
    apiBase: 'https://models-registry.neat.sima.ai',
    environment: 'Production',
    // Use production API image reads until a production derivative CDN is enabled.
    heroCdnBaseUrl: '',
  };
}

// Branchless previews keep the existing UI audience, but always browse Main.
export function branchPolicy(config, search = '') {
  const enabled = config.uiMode !== 'customer' && new URLSearchParams(search).get('branches') !== '0';
  return {enabled, defaultBranch: enabled ? 'develop' : 'main'};
}
