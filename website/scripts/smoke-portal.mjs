// Load the model browser in Chromium and fail when a page does not start: a
// module or stylesheet that does not load, an uncaught error, a console error,
// or a page that never renders its content.
//
// By default it serves the assembled portal against the preview API fixture.
// With SMOKE_URL it checks a deployed portal instead; GITHUB_SHA, when set,
// makes it first wait until that URL serves the assets of this revision.
import {spawn} from 'node:child_process';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const siteDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
const preview = join(siteDirectory, '..', 'artifact-browser', 'preview.py');
const portal = join(siteDirectory, 'portal-build');
const port = Number(process.env.SMOKE_PORT || 8766);
const pageAssets = new Set(['document', 'script', 'stylesheet']);
const deployTimeoutMs = 10 * 60_000;

function startPreview() {
  const server = spawn('python3', [preview, '--root', portal, '--port', String(port)], {
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  const ready = new Promise((resolve, reject) => {
    server.once('exit', (code) => reject(new Error(`preview server exited with code ${code}`)));
    server.stdout.on('data', (chunk) => {
      const url = String(chunk).match(/Model browser preview: (\S+)/)?.[1];
      if (url) resolve(new URL(url));
    });
  });
  return {server, ready};
}

function previewChecks(variantUrl) {
  const branchUrl = new URL('/', variantUrl);
  branchUrl.search = new URLSearchParams({branch: 'feature/yolo-input', model: 'resnet_50', variant: 'modalix_int8'});
  const uncatalogedUrl = new URL('/?branch=feature/no-catalog', variantUrl);
  return [
    ['model list', new URL('/', variantUrl), ['#result-list .model-row']],
    ['variant page', variantUrl, ['#components-section .component-row']],
    // The feature branch's catalog includes main's builds it started from.
    ['inherited build on a branch', branchUrl, ['#components-section .component-row']],
    ['branch without a catalog', uncatalogedUrl, ['#result-list h3:text("No completed builds on this branch")']],
  ];
}

// A CDN can serve the previous index until its invalidation completes.
async function waitForRevision(url, revision) {
  const asset = `app.${revision.slice(0, 12)}.js`;
  const deadline = Date.now() + deployTimeoutMs;
  while (Date.now() < deadline) {
    const index = await fetch(url, {cache: 'no-store'}).then((response) => response.text()).catch(() => '');
    if (index.includes(asset)) return;
    await new Promise((resolve) => setTimeout(resolve, 15_000));
  }
  throw new Error(`${url} did not serve ${asset} within ${deployTimeoutMs / 60_000} minutes`);
}

// Waits for each selector in turn, clicking the first match of each but the last to reach the next.
async function checkPage(browser, url, selectors) {
  const page = await browser.newPage();
  const problems = [];
  page.on('pageerror', (error) => problems.push(`uncaught error: ${error.message}`));
  page.on('console', (message) => {
    // Handled API 404s also log "Failed to load resource"; asset failures are caught below.
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource')) {
      problems.push(`console error: ${message.text()}`);
    }
  });
  page.on('response', (response) => {
    if (pageAssets.has(response.request().resourceType()) && response.status() >= 400) {
      problems.push(`${response.status()} ${response.url()}`);
    }
  });
  page.on('requestfailed', (request) => {
    if (pageAssets.has(request.resourceType())) {
      problems.push(`${request.failure()?.errorText} ${request.url()}`);
    }
  });
  await page.goto(url.href);
  for (const [position, selector] of selectors.entries()) {
    const element = await page.waitForSelector(selector, {timeout: 15_000}).catch(() => null);
    if (!element) {
      problems.push(`${selector} did not render`);
      break;
    }
    if (position < selectors.length - 1) await element.click();
  }
  await page.close();
  return problems;
}

async function runChecks(checks) {
  const browser = await chromium.launch();
  let failed = false;
  try {
    for (const [name, url, selectors] of checks) {
      const problems = await checkPage(browser, url, selectors);
      console.log(`${problems.length ? 'FAIL' : 'ok'}  ${name}  ${url.href}`);
      problems.forEach((problem) => console.log(`      ${problem}`));
      failed ||= problems.length > 0;
    }
  } finally {
    await browser.close();
  }
  return !failed;
}

let passed;
if (process.env.SMOKE_URL) {
  const url = new URL(process.env.SMOKE_URL);
  if (process.env.GITHUB_SHA) await waitForRevision(url, process.env.GITHUB_SHA);
  passed = await runChecks([
    ['model list and its first model', url, ['#result-list .model-row', '#components-section .component-row']],
  ]);
} else {
  const {server, ready} = startPreview();
  try {
    passed = await runChecks(previewChecks(await ready));
  } finally {
    server.kill();
  }
}
process.exitCode = passed ? 0 : 1;
