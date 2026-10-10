import {execFileSync} from 'node:child_process';
import {cpSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

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

const fingerprintedAssets = ['app', 'benchmark-adapters', 'branch-builds', 'downloads', 'markdown', 'styles', 'config'];

rmSync(output, {recursive: true, force: true});
mkdirSync(output, {recursive: true});
cpSync(docsBuild, output, {recursive: true});
cpSync(join(browser, 'favicon.svg'), join(output, 'favicon.svg'));

for (const asset of fingerprintedAssets) {
  const extension = asset === 'styles' ? 'css' : 'js';
  cpSync(join(browser, `${asset}.${extension}`), join(output, `${asset}.${revision}.${extension}`));
}

let browserHtml = readFileSync(join(browser, 'index.html'), 'utf8');
browserHtml = browserHtml
  .replace('href="styles.css"', `href="styles.${revision}.css"`)
  .replace('src="config.js"', `src="config.${revision}.js"`)
  .replace('src="app.js"', `src="app.${revision}.js"`);
writeFileSync(join(output, 'index.html'), browserHtml);

const modules = fingerprintedAssets.filter((asset) => asset !== 'styles');
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

console.log(`Assembled Model Registry portal at ${output}`);
console.log(`Catalog: ${routeBase}/`);
console.log(`Developer docs: ${routeBase}/docs`);
console.log(`Asset revision: ${revision}`);
