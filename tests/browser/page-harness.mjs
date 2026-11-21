// A fake page and registry for tests that load app.js: enough DOM for the catalog to start, render, and
// handle filter events. Import it before any browser module.
const realSetTimeout = globalThis.setTimeout;

class FakeElement {
  constructor() {
    Object.assign(this, {
      dataset: {}, hidden: false, innerHTML: '', textContent: '', checked: false, disabled: false,
      listeners: {}, attributes: {}, children: [], classList: {add() {}, remove() {}, contains: () => false},
    });
    this.selectedValue = '';
  }
  // A select keeps only a value one of its options has, as a browser does.
  get value() { return this.selectedValue; }
  set value(value) {
    const listed = !this.innerHTML.includes('<option') || this.innerHTML.includes(`value="${value}"`);
    this.selectedValue = listed ? String(value) : '';
  }
  get options() { return this.innerHTML.match(/<option /g) || []; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  fire(type, event = {}) {
    return Promise.all((this.listeners[type] || []).map((listener) => listener({target: this, preventDefault() {}, ...event})));
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) { delete this.attributes[name]; }
  closest() { return this.parent ||= new FakeElement(); }
  querySelector() { return new FakeElement(); }
  querySelectorAll() { return []; }
  contains() { return false; }
  matches() { return false; }
  append(child) { this.children.push(child); }
  remove() {}
  focus() {}
}

export const elements = new Map();
export const element = (id) => elements.get(id) ?? elements.set(id, new FakeElement()).get(id);
export const requests = [];
export const settle = () => new Promise((resolve) => realSetTimeout(resolve, 30));

const windowListeners = {};
const stored = new Map();

function setAddress(url) {
  const next = new URL(url, 'https://models.test/');
  globalThis.location = {href: next.href, search: next.search, protocol: next.protocol, hostname: next.hostname};
}

// `routes` maps a request to its JSON answer, or to undefined for a 404.
export function installPage({uiMode, routes}) {
  setAddress('https://models.test/');
  // index.html opens the workspace on the catalog.
  element('workspace').dataset.view = 'catalog';
  globalThis.window = {
    MODEL_REGISTRY_CONFIG: {uiMode, apiBase: 'https://registry.test', environment: 'Production', repository: {organization: 'sima-neat', name: 'models'}},
    addEventListener(type, listener) { (windowListeners[type] ||= []).push(listener); },
    scrollTo() {}, scrollY: 0, location: globalThis.location,
  };
  globalThis.document = {
    documentElement: {dataset: {}, classList: {add() {}}},
    body: new FakeElement(), head: new FakeElement(), activeElement: null, cookie: '',
    getElementById: element,
    querySelector: () => new FakeElement(),
    createElement: () => new FakeElement(),
    addEventListener() {},
  };
  globalThis.history = {
    scrollRestoration: 'auto',
    pushState(state, title, url) { setAddress(url); },
    replaceState(state, title, url) { setAddress(url); },
  };
  globalThis.localStorage = {
    getItem: (key) => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, String(value)),
  };
  globalThis.CSS = {escape: (value) => value};
  globalThis.matchMedia = () => ({matches: false});
  globalThis.setTimeout = (callback) => realSetTimeout(callback, 0);
  globalThis.fetch = async (url) => {
    const request = String(url).replace('https://registry.test', '');
    requests.push(request);
    const body = routes(request);
    return {
      status: body ? 200 : 404, ok: Boolean(body), headers: {get: () => 'application/json'},
      json: async () => body ?? {}, text: async () => JSON.stringify(body ?? {}),
    };
  };
}

// Opens a link as Back and Forward do: the address changes and the page reads its state from it.
export async function openLink(search) {
  requests.length = 0;
  setAddress(`https://models.test/${search}`);
  (windowListeners.popstate || []).forEach((listener) => listener());
  await settle();
}

export const toasts = () => element('toast-region').children.map((toast) => toast.textContent);
