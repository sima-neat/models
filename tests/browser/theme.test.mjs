import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import vm from 'node:vm';

// The first-paint theme script in index.html (sima-neat/models#1246), run as written.
const html = await readFile(new URL('../../artifact-browser/index.html', import.meta.url), 'utf8');
const themeScript = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function firstPaint({search = '', cookie = '', stored = {}, systemDark = false} = {}) {
  const dataset = {};
  const writes = [];
  vm.runInNewContext(themeScript, {
    URLSearchParams,
    location: {pathname: '/models/', search, hash: '', replace() {}},
    localStorage: {
      getItem: (key) => stored[key] ?? null,
      setItem: (key, value) => writes.push([key, value]),
    },
    document: {
      get cookie() { return cookie; },
      set cookie(value) { writes.push(['cookie', value]); },
      documentElement: {dataset},
    },
    matchMedia: () => ({matches: systemDark}),
  });
  return {theme: dataset.theme, source: dataset.themeSource, writes};
}

const STORED_LIGHT = {cookie: 'sima-neat-theme=light', stored: {'portal-theme': 'light', theme: 'light'}};

test('?theme wins over the cookie, localStorage and the system, and is marked as from the URL', () => {
  assert.deepEqual(firstPaint({...STORED_LIGHT, search: '?theme=dark'}), {theme: 'dark', source: 'url', writes: []});
  assert.deepEqual(firstPaint({cookie: 'sima-neat-theme=dark', systemDark: true, search: '?theme=light'}),
    {theme: 'light', source: 'url', writes: []});
});

test('?theme works next to the model page parameters', () => {
  assert.equal(firstPaint({...STORED_LIGHT, search: '?branch=main&model=yolov8n&variant=w8a8&theme=dark'}).theme, 'dark');
});

test('invalid ?theme values are ignored', () => {
  for (const search of ['?theme=blue', '?theme=Dark', '?theme=']) {
    assert.deepEqual(firstPaint({...STORED_LIGHT, search, systemDark: true}), {theme: 'light', source: undefined, writes: []}, search);
  }
});

test('without ?theme the order is unchanged: cookie, then localStorage, then the system', () => {
  assert.equal(firstPaint({cookie: 'sima-neat-theme=dark', stored: {theme: 'light'}}).theme, 'dark');
  assert.equal(firstPaint({stored: {'sima-neat-artifact-theme': 'dark'}}).theme, 'dark');
  assert.equal(firstPaint({systemDark: true}).theme, 'dark');
  assert.equal(firstPaint().theme, 'light');
  assert.equal(firstPaint({systemDark: true}).source, undefined);
});
