import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {cpSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {portalConfig} from './portal-config.mjs';

const siteDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
const repository = join(siteDirectory, '..');
const docsBuild = join(siteDirectory, 'build');
const output = join(siteDirectory, 'portal-build');
const browser = join(repository, 'artifact-browser');
const deploymentPath = process.env.DOCS_BASE_URL?.trim().replace(/^\/+|\/+$/g, '') || '';
const routeBase = deploymentPath ? `/${deploymentPath}` : '';
const revision = (
  process.env.GITHUB_SHA
  || execFileSync('git', ['rev-parse', 'HEAD'], {cwd: repository, encoding: 'utf8'}).trim()
).slice(0, 12);

const fingerprintedAssets = [
  'app', 'state', 'audience', 'ui', 'metrics', 'api', 'catalog-versions', 'catalog-view', 'compare-view', 'model-view', 'page-parts',
  'performance-section', 'execution-section', 'accuracy-section', 'download-dialog',
  'model-report', 'mla-layer-stats', 'mla-utilization', 'branch-builds', 'downloads', 'markdown', 'accuracy-metrics',
  'brand', 'styles', 'config', 'portal-shell', 'hero-images', 'environment', 'analytics',
];
const stylesheets = ['brand', 'styles'];
const modules = fingerprintedAssets.filter((asset) => !stylesheets.includes(asset) && asset !== 'config');
function gitMetadata(args) {
  try { return execFileSync('git', args, {cwd: repository, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore']}).trim(); }
  catch { return null; } // Source archives may have CI metadata but no Git directory.
}
const config = {
  ...portalConfig(),
  build: {
    commit: process.env.GITHUB_SHA || execFileSync('git', ['rev-parse', 'HEAD'], {cwd: repository, encoding: 'utf8'}).trim(),
    branch: process.env.GITHUB_HEAD_REF || process.env.GITHUB_REF_NAME || gitMetadata(['branch', '--show-current']) || 'Branch unavailable',
    modified: Boolean(gitMetadata(['status', '--porcelain'])),
  },
};
const configSource = `window.MODEL_REGISTRY_CONFIG = Object.freeze(${JSON.stringify(config, null, 2)});\n`;
// Audience and backend can change without a source commit. Preserve each configuration
// so an earlier entry point still loads its original settings after rollback.
const configFile = `config.${createHash('sha256').update(configSource).digest('hex')}.js`;

rmSync(output, {recursive: true, force: true});
mkdirSync(output, {recursive: true});
cpSync(docsBuild, output, {recursive: true});
cpSync(join(browser, 'favicon.svg'), join(output, 'favicon.svg'));

for (const asset of fingerprintedAssets.filter((asset) => asset !== 'config')) {
  const extension = stylesheets.includes(asset) ? 'css' : 'js';
  cpSync(join(browser, `${asset}.${extension}`), join(output, `${asset}.${revision}.${extension}`));
}
writeFileSync(join(output, configFile), configSource);

let browserHtml = readFileSync(join(browser, 'index.html'), 'utf8');
browserHtml = browserHtml
  .replace('href="brand.css"', `href="brand.${revision}.css"`)
  .replace('href="styles.css"', `href="styles.${revision}.css"`)
  .replace('src="config.js"', `src="${configFile}"`)
  .replace('src="app.js"', `src="app.${revision}.js"`)
  .replace('</head>', `${modules.map((asset) => `  <link rel="modulepreload" href="${asset}.${revision}.js">`).join('\n')}\n</head>`);
writeFileSync(join(output, 'index.html'), browserHtml);

// Repository data the page reads beside the registry API. api.js fetches each file by its unfingerprinted
// name: the order each multi-component model runs in, the scores each source model's authors published, and
// the variants the customer view hides for low accuracy, and the name of each dataset, all built from the
// repository by the registry tooling, and the browser's licenses.json, shipped as authored.
const apiModule = join(output, `api.${revision}.js`);
let apiSource = readFileSync(apiModule, 'utf8');
function shipModelData(name, write) {
  const url = `./${name}.json`;
  const file = `${name}.${revision}.json`;
  write(join(output, file));
  if (!apiSource.includes(`"${url}"`)) {
    throw new Error(`api.js no longer fetches ${url}; update the model data step in assemble-portal.mjs`);
  }
  apiSource = apiSource.replace(`"${url}"`, `"./${file}"`);
}
for (const name of ['component-pipelines', 'reference-metrics', 'low-accuracy-variants', 'dataset-names', 'example-apps']) {
  shipModelData(name, (file) => execFileSync(process.env.PYTHON || 'python3', [
    '-m', 'model_registry', name, '--repository', repository, '--output', file,
  ], {env: {...process.env, PYTHONPATH: join(repository, 'tools')}, stdio: 'inherit'}));
}
shipModelData('licenses', (file) => cpSync(join(browser, 'licenses.json'), file));
writeFileSync(apiModule, apiSource);

const relativeImport = /(["'])(\.\.?\/[^"'\s]*\.js)\1/g;
for (const asset of modules) {
  const path = join(output, `${asset}.${revision}.js`);
  const source = readFileSync(path, 'utf8').replace(relativeImport, (_, quote, specifier) => {
    const name = specifier.match(/^\.\/([\w-]+)\.js$/)?.[1];
    if (!modules.includes(name)) {
      throw new Error(`${asset}.js imports ${specifier}, which is not a fingerprinted asset; browser modules sit in artifact-browser/ and are listed in fingerprintedAssets`);
    }
    return `${quote}./${name}.${revision}.js${quote}`;
  });
  writeFileSync(path, source);
}

console.log(`Assembled Model Browser portal at ${output}`);
console.log(`Catalog: ${routeBase}/`);
console.log(`Developer docs: ${routeBase}/docs`);
console.log(`Asset revision: ${revision}`);
console.log(`Audience: ${config.uiMode}; registry: ${config.environment}; catalog: ${config.defaultBranch}`);
