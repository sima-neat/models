// Load the assembled portal in Chromium against the preview API fixture and fail
// when a page does not start: a module or stylesheet that does not load, an
// uncaught error, a console error, or a page that never renders its content.
import {spawn} from 'node:child_process';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const siteDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
const preview = join(siteDirectory, '..', 'artifact-browser', 'preview.py');
const portal = join(siteDirectory, 'portal-build');
const port = Number(process.env.SMOKE_PORT || 8766);
const pageAssets = new Set(['document', 'script', 'stylesheet']);

function startPreview() {
  const server = spawn(process.env.PYTHON || 'python3', [preview, '--root', portal, '--port', String(port)], {
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

async function checkPage(browser, url, readySelector) {
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
  await page.waitForSelector(readySelector, {timeout: 15_000})
    .catch(() => problems.push(`${readySelector} did not render`));
  await page.close();
  return problems;
}

async function checkLayerAnalysis(browser, url) {
  for (const [name, viewport, theme] of [
    ['desktop light', {width: 1440, height: 1000}, 'light'],
    ['mobile dark', {width: 390, height: 844}, 'dark'],
  ]) {
    const page = await browser.newPage({viewport});
    try {
      await page.goto(url.href);
      await page.locator('[data-disclosure^="layers:"] > summary').first().click();
      const panel = page.locator('[data-layer-analysis]').first();
      await panel.waitFor({state: 'visible'});
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      if (await panel.locator('.layer-chart li').count() !== 15) throw new Error('layer chart must show 15 rows');
      await panel.locator('[data-layer-filter="stage"]').selectOption('2');
      await panel.locator('[data-layer-filter="query"]').fill('RESHAPE');
      const summary = await panel.locator('.layer-summary').innerText();
      if (await panel.locator('.layer-chart li').count() !== 8) throw new Error('stage/name filtering failed');
      await panel.locator('[data-layer-filter="order"]').selectOption('compiler');
      await panel.locator('[data-disclosure^="layer-table:"] > summary').click();
      if (await panel.locator('.layer-table tbody tr').count() !== 8) throw new Error('full layer table did not filter');
      if (await panel.locator('.layer-table tbody tr').first().locator('td').first().innerText() !== '27') {
        throw new Error('compiler order lost after sorting');
      }
      await panel.locator('[data-layer-filter="query"]').fill('no such layer');
      if (!await panel.getByText('No layers match this search.').isVisible()) throw new Error('missing empty search state');
      if (await panel.locator('.layer-summary').innerText() !== summary) throw new Error('search changed percentage denominator');
      await panel.locator('[data-layer-filter="query"]').fill('');
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('page overflows viewport');
      console.log(`ok  layer analysis · ${name}`);
    } finally {
      await page.close();
    }
  }
}

const {server, ready} = startPreview();
let failed = false;
try {
  const variantUrl = await ready;
  const browser = await chromium.launch();
  try {
    const checks = [
      ['model list', new URL('/', variantUrl), '#result-list .model-card'],
      ['variant page', variantUrl, '#components-section .component-performance'],
      ['component pipeline', new URL('/?model=rf_detr_nano&variant=w8a8_w16a16', variantUrl), '#components-section .component-pipeline'],
    ];
    await checkLayerAnalysis(browser, variantUrl);
    for (const [name, url, readySelector] of checks) {
      const problems = await checkPage(browser, url, readySelector);
      console.log(`${problems.length ? 'FAIL' : 'ok'}  ${name}  ${url.pathname}${url.search}`);
      problems.forEach((problem) => console.log(`      ${problem}`));
      failed ||= problems.length > 0;
    }
  } finally {
    await browser.close();
  }
} finally {
  server.kill();
}
process.exitCode = failed ? 1 : 0;
