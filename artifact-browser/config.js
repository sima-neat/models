window.MODEL_REGISTRY_CONFIG = Object.freeze({
  uiMode: "developer",
  apiBase: "https://models-registry.stg.neat.sima.ai",
  heroCdnBaseUrl: "https://models-registry.stg.neat.sima.ai/images/sha256",
  environment: "Staging",
  allowProductionPreview: true,
  // A deployed build takes both endpoints from config/registry.json. This file serves only a page opened as is.
  productionApiBase: "https://models-registry.neat.sima.ai",
  repository: Object.freeze({ organization: "sima-neat", name: "models" }),
});
