// Usage: node website/scripts/pin-catalog-snapshot.mjs >> "$GITHUB_ENV"
//
// Prints MODEL_BROWSER_CATALOG_SNAPSHOT=<catalog_revision>:<canonical_sha256> for a production customer build,
// and nothing for any other build. The pinned snapshot is the one MODEL_BROWSER_CATALOG_SNAPSHOT requests, or
// else Main's current catalog. The registry must serve that exact snapshot, so a wrong pin fails the deploy
// instead of the page.
import {fileURLToPath} from 'node:url';
import {parseCatalogSnapshot, portalConfig} from './portal-config.mjs';

async function readCatalogSnapshot(url, fetch) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`The registry answered HTTP ${response.status} for ${url}`);
  const {catalog} = await response.json();
  if (!Array.isArray(catalog?.document?.models)) throw new Error(`The registry returned no model catalog for ${url}`);
  return parseCatalogSnapshot(`${catalog?.catalog_revision}:${catalog?.canonical_sha256}`);
}

export async function resolveCatalogSnapshot(config, fetch = globalThis.fetch) {
  const {organization, name} = config.repository;
  const catalogs = `${config.apiBase}/v1/repositories/${encodeURIComponent(organization)}/${encodeURIComponent(name)}/catalogs`;
  const gitRef = encodeURIComponent(`refs/heads/${config.defaultBranch}`);
  const snapshot = config.catalogSnapshot ?? await readCatalogSnapshot(`${catalogs}/current?git_ref=${gitRef}`, fetch);

  const served = await readCatalogSnapshot(
    `${catalogs}/${snapshot.revision}?git_ref=${gitRef}&snapshot=${snapshot.canonicalSha256}`, fetch,
  );
  if (served.revision !== snapshot.revision || served.canonicalSha256 !== snapshot.canonicalSha256) {
    throw new Error(`The registry served snapshot ${served.revision}:${served.canonicalSha256} for ${snapshot.revision}:${snapshot.canonicalSha256}`);
  }
  return snapshot;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const config = portalConfig();
    if (config.deploymentEnvironment === 'production' && config.uiMode === 'customer') {
      const {revision, canonicalSha256} = await resolveCatalogSnapshot(config);
      console.log(`MODEL_BROWSER_CATALOG_SNAPSHOT=${revision}:${canonicalSha256}`);
    } else if (config.catalogSnapshot) {
      throw new Error('A catalog snapshot pins only a production customer build.');
    }
  } catch (error) {
    console.error(`Cannot pin the catalog snapshot: ${error.message}`);
    process.exitCode = 1;
  }
}
