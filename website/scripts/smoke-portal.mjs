// Load the assembled portal in Chromium against the preview API fixture and fail
// when a page does not start: a module or stylesheet that does not load, an
// uncaught error, a console error, or a page that never renders its content.
import {spawn} from 'node:child_process';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {portalConfig} from './portal-config.mjs';

const siteDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
const preview = join(siteDirectory, '..', 'artifact-browser', 'preview.py');
const portal = join(siteDirectory, 'portal-build');
const port = Number(process.env.SMOKE_PORT || 8766);
const pageAssets = new Set(['document', 'script', 'stylesheet']);
const config = portalConfig();
const customer = config.uiMode === 'customer';

function startPreview() {
  const server = spawn(process.env.PYTHON || 'python3', [preview, '--root', portal, '--port', String(port)], {
    env: {...process.env, MODEL_BROWSER_UI_MODE: config.uiMode},
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
  if (await page.locator('[data-model-card-body]').count()) {
    await page.waitForSelector('[data-model-card-body] .model-documentation-body', {timeout: 15_000})
      .catch(() => problems.push('Inline model instructions did not load'));
    if (await page.locator('[data-disclosure="model-card"]').count()) problems.push('Model instructions require an extra disclosure');
  }
  if (customer && await page.locator('.detail-title').count()) {
    if (await page.locator('#details-section, [data-scroll-to="details-section"]').count()) problems.push('Customer UI exposes Details');
    if (await page.locator('[data-disclosure^="accuracy"], [data-disclosure="published-scores"]').count()) problems.push('Customer UI exposes evaluation detail rows');
  }
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
      if (await panel.locator('.layer-table tbody tr').count() !== 48) throw new Error('Layer timings must show every layer');
      if (await panel.locator('.layer-legend i').count() !== 10) throw new Error('Preview must demonstrate all ten recognized operation families');
      if (await panel.locator('tbody .layer-family-other').count()) throw new Error('Preview must use recognized compiler operations');
      await page.evaluate(() => document.fonts.ready);
      const legendPositions = () => panel.locator('.layer-legend').evaluate(legend => {
        const bounds = legend.getBoundingClientRect();
        return [...legend.querySelectorAll('[data-layer-family]')].map(item => {
          const box = item.getBoundingClientRect();
          return [box.left - bounds.left, box.top - bounds.top, box.width, box.height];
        });
      });
      const initialLegendPositions = await legendPositions();
      for (const family of ['conv', 'activation']) {
        const row = panel.locator(`tbody .layer-family-${family}`).first();
        for (const cell of ['.layer-name', '.layer-timing-number', '.layer-bar-track', '.layer-share']) {
          await row.locator(cell).hover();
          const highlighted = panel.locator('.layer-legend .is-highlighted');
          if (await highlighted.count() !== 1 || await highlighted.getAttribute('data-layer-family') !== family) throw new Error('Row hover must emphasize only its own family');
          if (await highlighted.locator('.layer-legend-label > span').evaluate(label => getComputedStyle(label).fontWeight) !== '600') throw new Error('Hovered family label should be bold');
          if (await row.locator('.layer-timing-number').evaluate(value => getComputedStyle(value).fontWeight) !== '400') throw new Error('Execution times must remain regular weight');
          const hoveredPositions = await legendPositions();
          if (hoveredPositions.some((box, index) => box.some((value, axis) => Math.abs(value - initialLegendPositions[index][axis]) > .5))) throw new Error(`Row hover shifts the legend: ${JSON.stringify({initialLegendPositions, hoveredPositions})}`);
        }
      }
      await panel.locator('.layer-results').hover();
      if (await panel.locator('.layer-legend .is-highlighted').count()) throw new Error('Legend emphasis must clear after leaving the row');
      if (!(await panel.locator('.layer-results').innerText()).includes('Slowest layers first')) throw new Error('Layer ordering caption is missing');
      if (await panel.locator('[data-layer-filter="stage"]').count()) throw new Error('stage dropdown should be absent');
      const axis = await panel.locator('.layer-axis').innerText();
      if (await panel.locator('.layer-axis > span').first().innerText() !== '0') throw new Error('Time axis must start at zero');
      const maximum = Number((await panel.locator('.layer-axis > span').last().innerText()).replaceAll(',', ''));
      if (maximum !== Number(await panel.locator('[data-layer-filter="minUs"]').getAttribute('max'))) throw new Error('Time axis and filters use different scales');
      const barWidths = await panel.locator('tbody tr').evaluateAll(rows => Object.fromEntries(rows.map(row => [
        row.querySelector('.layer-name small').textContent,
        row.querySelector('.layer-bar-track > span').style.width,
      ])));
      const scaleUnchanged = () => panel.locator('tbody tr').evaluateAll((rows, widths) => rows.every(row =>
        row.querySelector('.layer-bar-track > span').style.width === widths[row.querySelector('.layer-name small').textContent]
      ), barWidths);
      const summary = await panel.locator('.layer-summary').innerText();
      await panel.locator('[data-layer-filter="query"]').fill('RESHAPE');
      if (!await scaleUnchanged()) throw new Error('Search changed the bar scale');
      if (await panel.locator('.layer-axis').innerText() !== axis) throw new Error('Search changed the time axis');
      if (await panel.locator('.layer-table tbody tr').count() !== 16) throw new Error('name filtering failed');
      const slider = await panel.locator('[data-layer-filter="minUs"]').elementHandle();
      await slider.evaluate(el => { el.value = Number(el.max) / 2; el.dispatchEvent(new Event('input', {bubbles: true})); });
      if (!await slider.evaluate(el => el.isConnected)) throw new Error('filtering replaced the slider during dragging');
      await slider.evaluate(el => { el.value = 0; el.dispatchEvent(new Event('input', {bubbles: true})); });
      await panel.getByRole('button', {name: 'Runtime order', exact: true}).click();
      if (!(await panel.locator('.layer-results').innerText()).includes('Runtime order')) throw new Error('Runtime ordering caption is missing');
      if (await panel.locator('.layer-table tbody tr').count() !== 16) throw new Error('Reordering lost filters');
      if (await panel.locator('.layer-bar-track').count() !== 16 || await panel.locator('.layer-share').count() !== 16) throw new Error('Reordering lost bars or percentages');
      if (await panel.locator('.layer-name small').first().innerText() !== 'Stage 0 · #3') throw new Error('Runtime order lost the reported layer sequence');
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
      await panel.getByRole('button', {name: 'Slowest layers first', exact: true}).click();
      if (await panel.locator('.layer-bar-track').count() !== 48) throw new Error('Reordering lost layers');
      const times = await panel.locator('.layer-timing-number').allTextContents();
      const durations = times.map(value => Number(value.replaceAll(',', '')));
      if (durations.some((value, index) => index && value > durations[index - 1])) throw new Error('Slowest layers are not first');
      if (!await scaleUnchanged()) throw new Error('Reordering changed the bar scale');
      if (await panel.locator('.layer-axis').innerText() !== axis) throw new Error('Reordering changed the time axis');
      if (!await panel.getByRole('button', {name: 'Slowest layers first', exact: true}).getAttribute('aria-pressed').then(value => value === 'true')) throw new Error('Selected order is not marked');
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('page overflows viewport');
      console.log(`ok  Runtime Layer Analysis · ${name}`);
    } finally {
      await page.close();
    }
  }
}

async function checkCatalogLayouts(browser, url) {
  for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
    for (const theme of ['light', 'dark']) {
      const page = await browser.newPage({viewport});
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      try {
        await page.goto(url.href);
        await page.locator('.model-card').first().waitFor();
        if (await page.locator('#more-filters, #model-type-filter, #paradigm-filter, #input-modality-filter').count()) throw new Error('removed filters remain in the catalog');
        if (viewport.width === 390) await page.locator('#filter-toggle').click();
        for (const id of ['branch-filter', 'compiler-filter', 'history-filter']) {
          if (await page.locator(`#${id}`).isVisible() !== !customer) throw new Error(`${id} is not visible for the correct audience`);
        }
        const taskFilter = page.locator('#task-filter');
        await taskFilter.click();
        await page.locator('#result-title').click();
        if (await taskFilter.evaluate(el => getComputedStyle(el).outlineStyle !== 'none')) throw new Error('mouse use left a focus outline on the filter');
        await page.locator('#search').focus();
        await page.keyboard.press('Tab');
        if (viewport.width === 390) await page.keyboard.press('Tab');
        if (!await taskFilter.evaluate(el => document.activeElement === el && getComputedStyle(el).outlineStyle === 'solid')) throw new Error('keyboard navigation lost its filter focus outline');
        if (viewport.width === 390) await page.locator('#filter-toggle').click();
        const names = await page.locator('.model-card strong').allTextContents();
        const tasksByModel = await page.locator('.model-card').evaluateAll(cards => Object.fromEntries(cards.map(card => [
          card.querySelector('strong').textContent,
          [...card.querySelectorAll('.model-card-tag')].map(tag => tag.textContent),
        ])));
        const tableButton = page.getByRole('button', {name: 'Table', exact: true});
        await tableButton.focus();
        await tableButton.press('Enter');
        await page.locator('.catalog-table').first().waitFor();
        const settled = () => page.waitForFunction(() => !document.querySelector('.catalog-pending'));
        await settled();
        const tableNames = await page.locator('.catalog-model-link').allTextContents();
        if (new Set(tableNames).size !== names.length || names.some(name => tableNames.filter(value => value === name).length !== tasksByModel[name].length)) throw new Error('model is missing from a task group or duplicated within one');
        if (await page.locator('#result-count').innerText() !== `${names.length} models`) throw new Error('task groups inflated the model count');
        const headings = await page.locator('.catalog-task-group h3').allTextContents();
        if (JSON.stringify(headings) !== JSON.stringify([...headings].sort((a, b) => a.localeCompare(b)))) throw new Error('task groups are not alphabetical');
        for (const group of await page.locator('.catalog-task-group').all()) {
          const groupNames = await group.locator('.catalog-model-link').allTextContents();
          if (JSON.stringify(groupNames) !== JSON.stringify(names.filter(name => groupNames.includes(name)))) throw new Error('task grouping changed family ordering');
          const heading = await group.locator('h3').innerText();
          for (const name of groupNames) {
            if (!tasksByModel[name].includes(heading)) throw new Error('model appears under an unrelated task');
          }
        }
        const resnet = page.locator('[data-catalog-entry="resnet_50"]');
        if (!/%/.test(await resnet.locator('[data-table-accuracy]').innerText())) throw new Error('missing measured accuracy');
        const original = await resnet.locator('[data-table-performance]').innerText();
        if (!original.includes('inferences/s') || !original.includes('ms')) throw new Error('missing throughput or latency');
        await resnet.getByRole('combobox').selectOption('w16a16');
        await settled();
        if (await resnet.locator('[data-table-performance]').innerText() === original) throw new Error('precision changed without refreshing measurements');
        await resnet.getByRole('button', {name: 'ResNet-50', exact: true}).click();
        await page.locator('.detail-title').waitFor();
        if (new URL(page.url()).searchParams.get('variant') !== 'w16a16') throw new Error('model opened on the wrong precision');
        await page.locator('#nav-models').click();
        await page.locator('.catalog-table').first().waitFor();
        if (await tableButton.getAttribute('aria-pressed') !== 'true') throw new Error('returning lost table selection');
        await page.locator('#search').fill('rf detr');
        await page.waitForFunction(() => document.querySelectorAll('.catalog-model-link').length === 4);
        if (await page.locator('.catalog-task-group').count() !== 1 || await page.locator('.catalog-task-group h3').innerText() !== 'Object Detection') throw new Error('filtered task grouping is incorrect');
        await settled();
        // A package result is the model's one reading; its components are not listed or added up.
        const packageText = await page.locator('[data-catalog-entry="rf_detr_nano"] [data-table-performance]').innerText();
        if (!packageText.includes('inferences/s') || /backbone|decoder/.test(packageText)) throw new Error('package measurement is not the model\'s one reading');
        await page.getByRole('button', {name: 'Cards', exact: true}).click();
        if (await page.locator('.model-card').count() !== 4) throw new Error('switching lost the search filter');
        await tableButton.click();
        await page.locator('#search').fill('');
        if (viewport.width === 390) await page.locator('#filter-toggle').click();
        await page.locator('#task-filter').selectOption('visual_question_answering');
        await page.getByRole('heading', {name: 'Visual Question Answering', exact: true}).waitFor();
        if (await page.locator('.catalog-model-link').count() !== 1 || await page.locator('.catalog-model-link').innerText() !== 'Qwen3.5-0.8B') throw new Error('multi-task model did not follow the selected task');
        await page.locator('#task-filter').selectOption('');
        await page.locator('#search').fill('no such model xyz');
        await page.getByRole('heading', {name: 'No matching models', exact: true}).waitFor();
        await page.goto(url.href);
        await page.locator('.catalog-table').first().waitFor();
        await settled();
        await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
        if (await tableButton.getAttribute('aria-pressed') !== 'true') throw new Error('table preference did not survive reload');
        if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('catalog table overflows the page');
        await page.screenshot({path: `/tmp/models-ui-table-${viewport.width}-${theme}.png`});
        if (viewport.width === 390) {
          await page.locator('.catalog-table-scroll').evaluateAll(elements => elements.forEach(el => { el.scrollLeft = el.scrollWidth; }));
          await page.screenshot({path: `/tmp/models-ui-table-metrics-${theme}.png`});
        }
        await page.getByRole('button', {name: 'Cards', exact: true}).click();
        await page.screenshot({path: `/tmp/models-ui-cards-${viewport.width}-${theme}.png`});
        await tableButton.click();
        await page.goto(new URL('/?history=1&branch=develop', url).href);
        if (customer) {
          await page.locator('.model-card, .catalog-table').first().waitFor();
          if (await page.locator('#history-filter').isVisible() || await page.locator('#branch-filter').isVisible()) throw new Error('customer UI exposes developer controls');
          if (await page.locator('#branch-filter').inputValue() !== 'main') throw new Error('customer URL changed the catalog branch');
        } else {
          await page.locator('.run-row').first().waitFor();
          if (await page.locator('#catalog-layout-toggle').isVisible()) throw new Error('model layout switch leaked into build history');
          // preview.py OLD_LAYOUT_RUN_IDS: compile runs without a build profile or recipe timing.
          for (const id of ['00000000-0000-4000-8000-0000000000a1', '00000000-0000-4000-8000-0000000000a2']) {
            if (await page.locator(`.run-row[data-run-id="${id}"]`).count()) throw new Error(`build history shows old-layout run ${id}`);
          }
        }
        if (errors.length) throw new Error(errors.join('; '));
        console.log(`ok  Catalog cards/table · ${viewport.width}px ${theme}`);
      } finally {
        await page.close();
      }
    }
  }
}

async function checkExecutionTooltips(browser, url) {
  for (const [width, theme] of [[1440, 'light'], [390, 'dark']]) {
    const page = await browser.newPage({viewport: {width, height: 1000}, hasTouch: width === 390});
    try {
      await page.goto(url.href);
      await page.locator('[data-disclosure^="graphs:"] > summary').first().click();
      const stage = page.locator('.execution-step[data-runs-on="MLA"]').first();
      await stage.scrollIntoViewIfNeeded();
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      if (width === 390) await stage.tap();
      else await stage.hover();
      const tooltip = page.locator('#execution-tooltip');
      await tooltip.waitFor();
      if (await tooltip.locator('strong').first().innerText() !== 'Model Inference') throw new Error('stage tooltip names the wrong stage');
      if (await tooltip.locator('p').innerText() !== 'Runs the compiled model.') throw new Error('stage tooltip has wrong explanation');
      if (await tooltip.locator('.execution-tooltip-processor').innerText() !== 'MLA · Machine Learning Accelerator') throw new Error('stage tooltip has wrong processor');
      const colors = await stage.evaluate(stage => {
        const graph = getComputedStyle(stage.querySelector('small'));
        const badge = getComputedStyle(document.querySelector('.execution-tooltip-processor'));
        return graph.color === badge.color && graph.backgroundColor === badge.backgroundColor;
      });
      if (!colors) throw new Error('tooltip processor color differs from the graph');
      if (await stage.getAttribute('aria-describedby') !== 'execution-tooltip') throw new Error('tooltip is not associated with its stage');
      const bounds = await tooltip.boundingBox();
      if (bounds.x < 0 || bounds.x + bounds.width > width || bounds.y < 0 || bounds.y + bounds.height > 1000) throw new Error('stage tooltip is clipped by viewport');
      if (width === 1440) {
        await tooltip.hover();
        await page.waitForTimeout(200);
        if (!await tooltip.isVisible()) throw new Error('stage tooltip cannot be hovered');
      } else {
        await page.waitForTimeout(200);
        if (!await tooltip.isVisible()) throw new Error('tap did not keep the stage tooltip open');
      }
      await page.keyboard.press('Escape');
      if (await tooltip.isVisible()) throw new Error('Escape did not dismiss stage tooltip');
      await stage.blur();
      await stage.focus();
      await tooltip.waitFor();
      await page.locator('.execution-details').first().focus();
      if (await tooltip.isVisible()) throw new Error('stage tooltip survived focus moving away');
      console.log(`ok  Execution stage tooltip · ${width}px ${theme}`);
    } finally { await page.close(); }
  }
}

async function checkTaskComparison(browser, base) {
  for (const viewport of [{width: 1440, height: 1100}, {width: 390, height: 844}]) {
    for (const theme of ['light', 'dark']) {
      const page = await browser.newPage({viewport});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      try {
        const url = new URL('/?view=compare&compare_task=object_detection', base);
        await page.goto(url.href);
        const add = page.locator('[data-compare-add]');
        await add.waitFor();
        await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
        const count = await add.locator('option').count() - 1;
        if (await page.locator('.compare-model-list').count()) throw new Error('full task list remains visible');
        if (count < 4 || await page.locator('.compare-point').count() < 2) throw new Error('comparison is missing task models or points');
        const profiles = await add.locator('option').allTextContents();
        if (!profiles.some(text => text.endsWith('W8A8')) || !profiles.some(text => text.endsWith('W16A16'))) throw new Error('task list lost precision variants');
        if (!await page.locator('.compare-point[data-compare-toggle$=".w8a8"]').count() || !await page.locator('.compare-point[data-compare-toggle$=".w16a16"]').count()) throw new Error('chart did not include every precision');
        // Neighbouring points' hit areas overlap on a phone; the last point is painted on top of the others.
        const topmost = page.locator('.compare-point').last();
        await topmost.hover();
        const tooltip = page.getByRole('tooltip');
        await tooltip.waitFor();
        if (!await tooltip.locator('strong').innerText()) throw new Error('tooltip is missing model name');
        if (JSON.stringify(await tooltip.locator('dt').allTextContents()) !== JSON.stringify(['Variant', 'Throughput', 'Accuracy'])) throw new Error('tooltip fields are incomplete');
        const topmostKey = await topmost.getAttribute('data-compare-toggle');
        if (await tooltip.locator('dd').first().innerText() !== topmostKey.split('.').at(-1).toUpperCase()) throw new Error('tooltip names the wrong variant');
        if (!/inferences\/s/.test(await tooltip.innerText()) || !/%/.test(await tooltip.innerText())) throw new Error('tooltip is missing measured throughput or accuracy');
        const bounds = await tooltip.boundingBox();
        if (bounds.x < 0 || bounds.x + bounds.width > viewport.width || bounds.y < 0 || bounds.y + bounds.height > viewport.height) throw new Error('tooltip is clipped by viewport');
        await page.screenshot({path: `/tmp/models-ui-compare-tooltip-${viewport.width}-${theme}.png`});
        await page.mouse.move(0, 0);
        if (await tooltip.isVisible()) throw new Error('tooltip remains visible after pointer leaves');
        await topmost.focus();
        await topmost.press('Enter');
        if (await page.locator(`.compare-legend [data-compare-toggle="${topmostKey}"]`).count() !== 1) throw new Error('plotted selection is missing from the legend');
        await add.selectOption(await add.locator('option').nth(1).getAttribute('value'));
        await add.selectOption(await add.locator('option').nth(1).getAttribute('value'));
        if (await page.locator('.compare-table thead th').count() !== 3) throw new Error('selection did not populate three comparison columns');
        if (!await add.isDisabled()) throw new Error('selection limit was lost');
        if (new URL(page.url()).searchParams.getAll('build').length !== 3) throw new Error('selected builds were not saved in URL');
        await page.reload();
        await page.locator('.compare-table').waitFor();
        if (await page.locator('.compare-table thead th').count() !== 3) throw new Error('reload lost selected models');
        await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
        if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('comparison overflows the page');
        await page.screenshot({path: `/tmp/models-ui-compare-${viewport.width}-${theme}.png`});
        await page.locator('.compare-table-wrap').first().screenshot({path: `/tmp/models-ui-compare-table-${viewport.width}-${theme}.png`});
        await page.locator('.compare-legend button').first().click();
        if (await page.locator('.compare-table thead th').count() !== 2) throw new Error('model could not be deselected');
        const remove = page.locator('.compare-table .compare-remove').first();
        const removedKey = await remove.getAttribute('data-compare-toggle');
        await remove.focus();
        await remove.press('Enter');
        if (await page.locator('.compare-table thead th').count() !== 1) throw new Error('table removal did not update selected columns');
        if (await page.locator(`.compare-point[data-compare-toggle="${removedKey}"][aria-pressed="true"]`).count()) throw new Error('table removal did not update the chart');
        if (new URL(page.url()).searchParams.getAll('build').includes(removedKey)) throw new Error('table removal left the variant in the URL');
        if (await add.isDisabled() || !await add.locator(`option[value="${removedKey}"]`).count()) throw new Error('table removal did not restore the variant to Add model');
        if (await page.locator('[data-compare-control="component"]').count()) throw new Error('component control remains visible');
        // A multi-component model competes with its whole-model result as its one value.
        const pipelinePoints = page.locator('.compare-point[data-compare-toggle^="rf_detr_"]');
        if (!await pipelinePoints.count()) throw new Error('multi-component model is missing from the chart');
        const pipelineKey = await pipelinePoints.first().getAttribute('data-compare-toggle');
        if (await add.locator(`option[value="${pipelineKey}"]`).count()) await add.selectOption(pipelineKey);
        const columnKeys = await page.locator('.compare-table thead .compare-remove').evaluateAll(buttons => buttons.map(button => button.dataset.compareToggle));
        const pipelineThroughput = await page.locator('.compare-table tbody tr', {hasText: 'Throughput'}).locator('td').nth(columnKeys.indexOf(pipelineKey)).innerText();
        if (!/inferences\/s/.test(pipelineThroughput)) throw new Error('multi-component model shows no throughput');
        if (await page.locator('[data-compare-control="axis"] option[value="latency"]').count()) throw new Error('latency remains selectable as an axis');
        const chartLabels = await page.locator('.compare-axis-title').allTextContents();
        if (chartLabels[0] !== 'Throughput (inferences/s)' || !chartLabels[1].includes('mAP')) throw new Error('accuracy/throughput axes changed');
        if (!(await page.locator('.compare-chart-dataset').innerText()).includes('coco-2017')) throw new Error('chart does not identify its accuracy dataset');
        await page.locator('[data-compare-control="task"]').selectOption('image_classification');
        await add.waitFor();
        const choices = await add.locator('option').allTextContents();
        if (choices.length !== 3 || !choices.slice(1).every(text => text.includes('ResNet-50'))) throw new Error('comparison mixed tasks or omitted a variant');
        const dataset = await page.locator('.compare-chart-dataset').innerText();
        if (!dataset.includes('imagenet-2012') || !dataset.includes('mini')) throw new Error('chart dataset did not follow the selected task');
        // The FP16 classification variant has no accuracy result; it stays available in Add model.
        if (await page.locator('.compare-point').count() !== 1) throw new Error('chart included a variant without measured accuracy');
        await add.selectOption('resnet_50.w16a16');
        if (!(await page.locator('.compare-table').innerText()).includes('W16A16')) throw new Error('variant without accuracy is no longer selectable');
        if (await page.locator('.compare-legend button').count()) throw new Error('selection without measured accuracy appears in the legend');
        await page.locator('.compare-table .compare-remove').click();
        if (await page.locator('.compare-table, .compare-selected').count()) throw new Error('removing the final variant left an empty table');
        if (new URL(page.url()).searchParams.getAll('build').length) throw new Error('removing the final variant left a selection in the URL');
        await page.locator('[data-compare-control="task"]').selectOption('text_generation');
        await page.locator('.compare-tbd').waitFor();
        if (await page.locator('.compare-chart, .compare-table, [data-compare-add]').count()) throw new Error('LLM task rendered vision measurements');
        if (errors.length) throw new Error(errors.join('; '));
        console.log(`ok  Task comparison · ${viewport.width}px ${theme}`);
      } finally { await page.close(); }
    }
  }
}

const {server, ready} = startPreview();
let failed = false;
try {
  // The first catalog model may have no benchmark data.
  const variantUrl = new URL('/?model=resnet_50&variant=w8a8', await ready);
  const browser = await chromium.launch();
  try {
    const checks = [
      ['model list', new URL('/', variantUrl), '#result-list .model-card'],
      ['variant page', variantUrl, '#components-section .component-performance'],
      ['component pipeline', new URL('/?model=rf_detr_nano&variant=w8a8_w16a16', variantUrl), '#components-section .component-pipeline'],
    ];
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
      const page = await browser.newPage({viewport});
      await page.goto(new URL('/?branches=0&branch=develop', variantUrl).href);
      // The developer fixture has no Main catalog: selection must stay Main even then.
      await page.waitForFunction(() => document.querySelector('#branch-filter').value === 'main');
      if (await page.locator('#branch-filter').isVisible()) throw new Error('branches=0 exposes branch filter');
      if (await page.locator('#branch-filter').inputValue() !== 'main') throw new Error('branches=0 did not select main');
      await page.locator('#nav-compare').click();
      await page.locator('#nav-models').click();
      if (new URL(page.url()).searchParams.get('branches') !== '0') throw new Error('Navigation lost branches=0');
      if (!await page.locator('#clear-filters').isVisible()) await page.locator('#filter-toggle').click();
      await page.locator('#clear-filters').click();
      if (await page.locator('#branch-filter').inputValue() !== 'main') throw new Error('Clear filters left main');
      await page.screenshot({path: `/tmp/model-browser-branches-${viewport.width}.png`});
      await page.close();
      console.log(`ok  Main-only branches=0 · ${viewport.width}px`);
    }
    await checkLayerAnalysis(browser, variantUrl);
    await checkExecutionTooltips(browser, variantUrl);
    await checkCatalogLayouts(browser, new URL('/', variantUrl));
    await checkTaskComparison(browser, variantUrl);
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
