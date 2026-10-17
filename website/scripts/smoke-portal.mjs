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
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      if (await panel.locator('.layer-duration-table tbody tr').count() !== 48) throw new Error('Duration must show every layer');
      if (await panel.locator('[data-layer-filter="stage"]').count()) throw new Error('stage dropdown should be absent');
      const axis = await panel.locator('.layer-axis').innerText();
      const summary = await panel.locator('.layer-summary').innerText();
      await panel.locator('[data-layer-filter="query"]').fill('RESHAPE');
      if (await panel.locator('.layer-axis').innerText() !== axis) throw new Error('search changed the time axis');
      if (await panel.locator('.layer-duration-table tbody tr').count() !== 16) throw new Error('name filtering failed');
      const slider = await panel.locator('[data-layer-filter="minUs"]').elementHandle();
      await slider.evaluate(el => { el.value = Number(el.max) / 2; el.dispatchEvent(new Event('input', {bubbles: true})); });
      if (!await slider.evaluate(el => el.isConnected)) throw new Error('filtering replaced the slider during dragging');
      await slider.evaluate(el => { el.value = 0; el.dispatchEvent(new Event('input', {bubbles: true})); });
      await panel.getByRole('button', {name: 'Table', exact: true}).click();
      if (await panel.locator('.layer-numeric-table tbody tr').count() !== 16) throw new Error('view switch lost filters');
      if (await panel.locator('.layer-bar-track').count()) throw new Error('Table should show numbers without bars');
      if (await panel.locator('.layer-name small').first().innerText() !== 'Stage 0 · #3') throw new Error('Table lost compiler order');
      // A classic scrollbar and the widest percentage must leave the Share column readable.
      await page.addStyleTag({content: '.layer-table-scroll::-webkit-scrollbar { width: 18px; height: 18px; }'});
      const layout = await panel.evaluate(panel => {
        const region = panel.querySelector('.layer-table-scroll');
        const share = panel.querySelector('.layer-share');
        share.textContent = '100.00%';
        const text = document.createRange();
        text.selectNodeContents(share);
        const cell = share.getBoundingClientRect();
        const bounds = region.getBoundingClientRect();
        const textBounds = text.getBoundingClientRect();
        const time = panel.querySelector('.layer-timing-number');
        const timeRange = document.createRange();
        timeRange.selectNodeContents(time);
        const timeBounds = timeRange.getBoundingClientRect();
        return {
          headerFits: [...panel.querySelectorAll('thead th')].every(header => {
            const range = document.createRange();
            range.selectNodeContents(header);
            return range.getBoundingClientRect().right <= header.getBoundingClientRect().right + 1;
          }),
          rightPadding: bounds.left + region.clientWidth - cell.right,
          fits: textBounds.left >= cell.left && textBounds.right <= cell.right,
          numericAlignment: Math.abs(textBounds.top - timeBounds.top),
          fontSize: Number.parseFloat(getComputedStyle(share).fontSize),
        };
      });
      if (layout.rightPadding < 10 || !layout.fits) throw new Error('Share column is clipped by the scrollbar');
      if (!layout.headerFits) throw new Error('Layer table headers overlap');
      if (layout.numericAlignment > 1) throw new Error('Time and share values are not aligned');
      if (layout.fontSize < 13) throw new Error('Layer text is too small');
      await panel.locator('[data-layer-filter="query"]').fill('no such layer');
      if (!await panel.getByText('No layers match these filters.').isVisible()) throw new Error('missing empty filter state');
      if (await panel.locator('.layer-summary').innerText() !== summary) throw new Error('filters changed summary totals');
      await panel.locator('[data-layer-filter="query"]').fill('');
      await panel.getByRole('button', {name: 'Duration', exact: true}).click();
      if (await panel.locator('.layer-bar-track').count() !== 48) throw new Error('Duration lost layers');
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('page overflows viewport');
      console.log(`ok  Runtime Layer Analysis · ${name}`);
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
