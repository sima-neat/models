import { adaptBenchmarkResult } from "./benchmark-adapters.js";

const config = window.MODEL_REGISTRY_CONFIG || {};
const configuredApiBase = String(config.apiBase || "").replace(/\/$/, "");
const configuredRepository = config.repository || {};
const DEFAULT_BRANCH = "main";
const CATEGORY_QUERY_FIELDS = {
  "model-type-filter": "model_type",
  "task-filter": "task",
  "architecture-filter": "architecture_id",
  "paradigm-filter": "architecture_paradigm",
  "input-modality-filter": "input_modality",
};
const initialQuery = new URLSearchParams(location.search);
const initialSearch = initialQuery.get("q") || "";
const initialIncludeHistory = initialQuery.get("history") === "1";
const DETAIL_LAYOUT_KEY = "sima-neat-model-detail-layout";
const TRIAL_METRICS = {
  latency_avg_ms: { label: "Average latency", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  latency_p50_ms: { label: "Latency p50", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  latency_p95_ms: { label: "Latency p95", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  throughput_fps: { label: "Throughput", unit: "fps", digits: 1, scale: 1, direction: "higher" },
  avg_power_watts: { label: "Board power", unit: "W", digits: 2, scale: 1, direction: "lower" },
  energy_joules: { label: "Trial energy", unit: "J", digits: 2, scale: 1, direction: "lower" },
  energy_per_inference_joules: {
    label: "Energy / inference", unit: "mJ/inference", digits: 2, scale: 1000, direction: "lower",
  },
};

const state = {
  apiBase: configuredApiBase,
  repository: configuredRepository,
  runs: [],
  cursor: null,
  hasMore: false,
  availableBranches: [],
  runsRequestId: 0,
  selectedId: new URLSearchParams(location.search).get("run"),
  preferredBranch: null,
  detailCache: new Map(),
  comparisonCache: new Map(),
  compareSelection: new Map(),
  taxonomy: null,
  detailLayout: readDetailLayout(),
  trialMetric: "latency_avg_ms",
  categorySelection: Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, initialQuery.get(queryName) || ""]),
  ),
};

const elements = Object.fromEntries(
  [
    "theme-toggle", "environment-name", "search", "filter-toggle",
    "filter-count", "filter-drawer", "branch-filter", "compiler-filter", "history-filter", "model-type-filter", "task-filter",
    "architecture-filter", "paradigm-filter", "input-modality-filter", "clear-filters", "result-title",
    "result-count", "result-list", "load-more", "detail-pane", "toast-region", "workspace",
    "compare-button", "compare-label", "compare-count", "compare-dialog", "compare-close", "compare-content",
    "compare-title", "compare-description",
  ].map((id) => [id, document.getElementById(id)]),
);

const icons = {
  sun: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.3 15.2A8.5 8.5 0 0 1 8.8 3.7 8.5 8.5 0 1 0 20.3 15.2Z"/></svg>`,
  arrow: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5"/></svg>`,
  download: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 20h14"/></svg>`,
  copy: `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="1.5"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>`,
  external: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5m0-5-8 8M19 14v5H5V5h5"/></svg>`,
  github: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.75a9.25 9.25 0 0 0-2.92 18.03c.46.08.63-.2.63-.45v-1.8c-2.57.56-3.11-1.09-3.11-1.09-.42-1.07-1.03-1.35-1.03-1.35-.84-.58.06-.57.06-.57.93.07 1.42.96 1.42.96.83 1.42 2.17 1.01 2.7.77.08-.6.32-1.01.59-1.24-2.05-.23-4.21-1.03-4.21-4.57 0-1.01.36-1.83.95-2.48-.1-.23-.41-1.17.09-2.44 0 0 .78-.25 2.54.95A8.8 8.8 0 0 1 12 7.1a8.7 8.7 0 0 1 2.31.31c1.76-1.2 2.53-.95 2.53-.95.51 1.27.19 2.21.1 2.44.59.65.95 1.47.95 2.48 0 3.55-2.17 4.33-4.23 4.56.33.29.63.85.63 1.72v2.67c0 .25.16.54.63.45A9.25 9.25 0 0 0 12 2.75Z"/></svg>`,
  // Official no-border logo from https://huggingface.co/brand.
  huggingFace: `<svg viewBox="0 0 95 88" aria-hidden="true"><path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z"/><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"/><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"/><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"/><path fill="#3A3B45" fill-rule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clip-rule="evenodd"/><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"/><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"/><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"/><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"/></svg>`,
  benchmark: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 19V13M12 19V5M19 19V9"/></svg>`,
  branch: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="5" r="2.25"/><circle cx="18" cy="7" r="2.25"/><circle cx="6" cy="19" r="2.25"/><path d="M6 7.25v9.5M8.25 17c5.25-.5 3.5-8.5 7.5-9.6"/></svg>`,
  precision: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5v14M4.5 7.5 7 5l2.5 2.5M14 5h4.5v6H14zM14 14h4.5v5H14z"/></svg>`,
  compiler: `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m12 0h4M2 15h4m12 0h4M10 10h4v4h-4z"/></svg>`,
  modelType: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z"/><path d="m4 12 8 4.5 8-4.5M4 16.5 12 21l8-4.5"/></svg>`,
  architecture: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 12h4m0 0 6-5m-6 5 6 5"/></svg>`,
  task: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6h11M8 12h11M8 18h11"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>`,
};

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function branchName(ref = "") {
  return ref.replace(/^refs\/heads\//, "") || "detached";
}

function shortSha(sha = "") {
  return sha.slice(0, 12);
}

function githubModelUrl(run, modelId) {
  const organization = String(state.repository.organization || "");
  const repository = String(state.repository.name || "");
  const configuredRepository = `${organization}/${repository}`.toLowerCase();
  const runRepository = String(run.repository || "").toLowerCase();
  const commit = String(run.commit_sha || "");
  if (!/^[A-Za-z0-9_.-]+$/.test(organization)
    || !/^[A-Za-z0-9_.-]+$/.test(repository)
    || runRepository !== configuredRepository
    || !/^[a-f0-9]{40}$/i.test(commit)
    || !/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test(modelId)) return null;
  return `https://github.com/${encodeURIComponent(organization)}/${encodeURIComponent(repository)}/tree/${commit}/models/${encodeURIComponent(modelId)}`;
}

function mpkValidationMark(tests = []) {
  const passed = tests.some((test) => test.name === "mpk-validation" && test.status === "passed");
  if (!passed) return "";
  const message = "MPK package integrity verified: archive structure, checksum, and build provenance match.";
  return `<span class="mpk-validation-mark" tabindex="0" role="img" aria-label="${message}" data-tooltip="${message}">✓</span>`;
}

function trustedHuggingFaceUrl(value = "") {
  try {
    const url = new URL(String(value));
    if (url.protocol !== "https:" || url.hostname !== "huggingface.co" || url.username || url.password) return null;
    return url.href;
  } catch {
    return null;
  }
}

function huggingFaceSourceLink(input = {}) {
  const sourceReference = String(input.source_reference || "");
  if (!sourceReference.startsWith("hf://") && !trustedHuggingFaceUrl(sourceReference)) return null;
  const declaredModelCard = trustedHuggingFaceUrl(input.metadata?.model_card);
  if (declaredModelCard) {
    return { href: declaredModelCard, label: declaredModelCard.replace(/^https:\/\//, "").replace(/\/$/, "") };
  }

  const match = /^hf:\/\/(models|datasets|spaces)\/([^/@?#]+)\/([^/@?#]+)@[a-f0-9]{40}\/.+$/i.exec(
    String(input.source_reference || ""),
  );
  if (!match) return null;
  const [, repositoryType, owner, repository] = match;
  const typePrefix = repositoryType === "models" ? "" : `${repositoryType}/`;
  const href = `https://huggingface.co/${typePrefix}${encodeURIComponent(owner)}/${encodeURIComponent(repository)}`;
  return { href, label: href.replace(/^https:\/\//, "") };
}

function sourceInputPanel(input) {
  return `<section><div class="detail-section-title"><h3>Source input</h3></div><div class="source-block">
    <strong>${escapeHtml(input.name)}</strong>
    <p>${escapeHtml(input.source_reference || input.input_type)}</p>
    <div><span>${escapeHtml(formatBytes(input.size_bytes))}</span><code>${escapeHtml(input.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(input.sha256)}" aria-label="Copy source checksum">${icons.copy}</button></div>
  </div></section>`;
}

function formatDate(value, includeTime = true) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", year: date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    ...(includeTime ? { hour: "numeric", minute: "2-digit" } : {}),
  }).format(date);
}

function formatDuration(start, end) {
  if (!start || !end) return "—";
  const seconds = Math.max(0, Math.round((new Date(end) - new Date(start)) / 1000));
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function readDetailLayout() {
  try { return localStorage.getItem(DETAIL_LAYOUT_KEY) === "focused" ? "focused" : "balanced"; }
  catch { return "balanced"; }
}

function applyDetailLayout() {
  elements.workspace.dataset.detailLayout = state.detailLayout;
}

async function api(path) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { Accept: "application/json" },
    mode: "cors",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error?.message || payload.message || `Registry request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function catalogCacheKey() {
  const repo = state.repository;
  const categories = Object.entries(categoryFilters()).filter(([, value]) => value).sort();
  const search = elements.search.value.trim();
  const history = elements["history-filter"].checked ? "history" : "latest";
  return `sima-neat-model-catalog:${state.apiBase}:${repo.organization}/${repo.name}:${history}:${search}:${JSON.stringify(categories)}`;
}

function branchPreferenceKey() {
  const repo = state.repository;
  return `sima-neat-model-branch:${state.apiBase}:${repo.organization}/${repo.name}`;
}

function readBranchPreference() {
  try {
    const stored = localStorage.getItem(branchPreferenceKey());
    return stored === null ? DEFAULT_BRANCH : stored;
  } catch {
    return DEFAULT_BRANCH;
  }
}

function writeBranchPreference(branch) {
  try {
    localStorage.setItem(branchPreferenceKey(), branch);
  } catch {
    // Storage may be disabled; branch selection still works for this page load.
  }
}

function readCachedCatalog() {
  try {
    const cached = JSON.parse(localStorage.getItem(catalogCacheKey()));
    if (!cached || !Array.isArray(cached.runs)) return null;
    return cached;
  } catch {
    return null;
  }
}

function writeCachedCatalog() {
  try {
    localStorage.setItem(catalogCacheKey(), JSON.stringify({
      savedAt: new Date().toISOString(),
      runs: state.runs,
      branches: state.availableBranches,
      cursor: state.cursor,
      hasMore: state.hasMore,
      taxonomy: state.taxonomy,
    }));
  } catch {
    // Storage may be disabled; the live registry remains fully functional.
  }
}

function taxonomyOptions(section, emptyLabel) {
  const entries = state.taxonomy?.[section] || {};
  return `<option value="">${escapeHtml(emptyLabel)}</option>${Object.entries(entries)
    .sort((left, right) => left[1].display_name.localeCompare(right[1].display_name))
    .map(([id, entry]) => `<option value="${escapeHtml(id)}">${escapeHtml(entry.display_name)}</option>`)
    .join("")}`;
}

function populateCategoryFilters() {
  const definitions = [
    ["model-type-filter", "model_types", "Any type"],
    ["task-filter", "tasks", "Any task"],
    ["architecture-filter", "architectures", "Any architecture"],
    ["paradigm-filter", "architecture_paradigms", "Any paradigm"],
    ["input-modality-filter", "modalities", "Any input"],
  ];
  definitions.forEach(([elementId, section, emptyLabel]) => {
    const element = elements[elementId];
    element.innerHTML = taxonomyOptions(section, emptyLabel);
    const selected = state.categorySelection[elementId] || "";
    element.value = [...element.options].some((option) => option.value === selected) ? selected : "";
    state.categorySelection[elementId] = element.value;
  });
}

function setCategoryFiltersDisabled(disabled) {
  Object.keys(CATEGORY_QUERY_FIELDS).forEach((elementId) => {
    elements[elementId].disabled = disabled;
  });
}

async function loadTaxonomy() {
  const repo = state.repository;
  const payload = await api(
    `/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/taxonomies/latest`,
  );
  state.taxonomy = payload.taxonomy?.document || null;
  if (!state.taxonomy) throw new Error("The registry did not return a model taxonomy.");
  populateCategoryFilters();
  setCategoryFiltersDisabled(false);
}

function categoryFilters() {
  return Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [
      queryName,
      state.taxonomy ? elements[elementId].value : state.categorySelection[elementId],
    ]),
  );
}

function writeCategoryFilterUrl() {
  const url = new URL(location.href);
  Object.entries(categoryFilters()).forEach(([queryName, value]) => {
    if (value) url.searchParams.set(queryName, value);
    else url.searchParams.delete(queryName);
  });
  history.replaceState({}, "", url);
}

function writeSearchUrl() {
  const url = new URL(location.href);
  const query = elements.search.value.trim();
  if (query) url.searchParams.set("q", query);
  else url.searchParams.delete("q");
  history.replaceState({}, "", url);
}

function writeHistoryUrl() {
  const url = new URL(location.href);
  if (elements["history-filter"].checked) url.searchParams.set("history", "1");
  else url.searchParams.delete("history");
  history.replaceState({}, "", url);
}

function activeFilterCount() {
  return [
    elements["branch-filter"].value && elements["branch-filter"].value !== DEFAULT_BRANCH,
    elements["history-filter"].checked,
    elements["compiler-filter"].value,
    ...Object.values(categoryFilters()),
  ].filter(Boolean).length;
}

function setFilterDrawer(open) {
  elements["filter-drawer"].hidden = !open;
  elements["filter-toggle"].setAttribute("aria-expanded", String(open));
}

function updateFilterDisclosure() {
  const count = activeFilterCount();
  elements["filter-count"].textContent = String(count);
  elements["filter-count"].hidden = count === 0;
}

function applyCatalog(payload, { append = false, cache = true } = {}) {
  state.runs = append ? [...state.runs, ...payload.runs] : payload.runs;
  state.availableBranches = [...new Set([
    ...state.availableBranches,
    ...(payload.branches || []),
    ...payload.runs.map((run) => branchName(run.git_ref)),
  ])].sort();
  state.cursor = payload.next_cursor ?? payload.cursor ?? null;
  state.hasMore = Boolean(payload.has_more ?? payload.hasMore);
  populateFilters({ persistBranchFallback: cache });
  renderRuns();
  if (cache) writeCachedCatalog();
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("sima-neat-artifact-theme", theme);
    localStorage.setItem("theme", theme);
  } catch {
    // Storage may be disabled; theme selection still works for this page load.
  }
  const dark = theme === "dark";
  elements["theme-toggle"].innerHTML = dark ? icons.sun : icons.moon;
  elements["theme-toggle"].setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} mode`);
  elements["theme-toggle"].title = `Switch to ${dark ? "light" : "dark"} mode`;
}

function filters() {
  return {
    query: elements.search.value.trim(),
    branch: elements["branch-filter"].value,
    compiler: elements["compiler-filter"].value,
    ...categoryFilters(),
  };
}

function normalizedSearchTerms(value) {
  return value.normalize("NFKD").toLowerCase().replace(/[._/-]+/g, " ")
    .replace(/[^a-z0-9\s]+/g, " ").split(/\s+/).filter(Boolean);
}

function taxonomyLabel(section, id) {
  return state.taxonomy?.[section]?.[id]?.display_name || id || "";
}

function runSearchDocument(run) {
  return [
    run.metadata?.model_id, run.metadata?.model_name, run.metadata?.display_name,
    run.metadata?.variant_id, run.git_ref, run.commit_sha, run.run_type,
    run.toolchain_version, run.target_platform,
    taxonomyLabel("model_types", run.model_type),
    taxonomyLabel("architectures", run.architecture_id),
    taxonomyLabel("architecture_paradigms", run.architecture_paradigm),
    ...(run.tasks || []).flatMap((id) => [id, taxonomyLabel("tasks", id)]),
    ...(run.input_modalities || []).flatMap((id) => [id, taxonomyLabel("modalities", id)]),
    ...(run.output_modalities || []).flatMap((id) => [id, taxonomyLabel("modalities", id)]),
    ...(run.model_tags || []),
  ].filter(Boolean).join(" ");
}

function filteredRuns() {
  const current = filters();
  const terms = normalizedSearchTerms(current.query);
  return state.runs.filter((run) => {
    const searchableTerms = new Set(normalizedSearchTerms(runSearchDocument(run)));
    const searchableText = [...searchableTerms].join(" ");
    return run.status === "completed"
      && terms.every((term) => searchableText.includes(term))
      && (!current.branch || branchName(run.git_ref) === current.branch)
      && (!current.compiler || run.toolchain_version === current.compiler)
      && (!current.model_type || run.model_type === current.model_type)
      && (!current.task || (run.tasks || []).includes(current.task))
      && (!current.architecture_id || run.architecture_id === current.architecture_id)
      && (!current.architecture_paradigm || run.architecture_paradigm === current.architecture_paradigm)
      && (!current.input_modality || (run.input_modalities || []).includes(current.input_modality));
  });
}

function populateFilters({ persistBranchFallback = false } = {}) {
  const selectedCompiler = elements["compiler-filter"].value;
  const branches = state.availableBranches;
  const compilers = [...new Set(state.runs.map((run) => run.toolchain_version).filter(Boolean))].sort().reverse();
  elements["branch-filter"].innerHTML = `<option value="">All branches</option>${branches.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  elements["compiler-filter"].innerHTML = `<option value="">Any version</option>${compilers.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  const preferredBranch = state.preferredBranch ?? DEFAULT_BRANCH;
  const selectedBranch = preferredBranch === "" || branches.includes(preferredBranch)
    ? preferredBranch
    : branches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH : "";
  elements["branch-filter"].value = selectedBranch;
  if (persistBranchFallback && selectedBranch !== preferredBranch) {
    state.preferredBranch = selectedBranch;
    writeBranchPreference(selectedBranch);
  }
  elements["compiler-filter"].value = compilers.includes(selectedCompiler) ? selectedCompiler : "";
  updateFilterDisclosure();
}

function comparisonRuns() {
  return state.compareSelection.size ? [...state.compareSelection.values()] : filteredRuns();
}

function updateCompareAction() {
  const selectedCount = state.compareSelection.size;
  const visibleCount = filteredRuns().length;
  const count = selectedCount || visibleCount;
  elements["compare-button"].disabled = count === 0;
  elements["compare-label"].textContent = selectedCount ? "Compare selected" : "Compare visible";
  elements["compare-count"].textContent = count;
  elements["compare-count"].hidden = count === 0;
  elements["compare-button"].setAttribute(
    "aria-label",
    selectedCount
      ? `Compare ${selectedCount} selected build${selectedCount === 1 ? "" : "s"}`
      : `Compare ${visibleCount} visible build${visibleCount === 1 ? "" : "s"}`,
  );
}

function renderRuns() {
  const runs = filteredRuns();
  elements["result-count"].textContent = `${runs.length}${state.hasMore ? "+" : ""} build${runs.length === 1 ? "" : "s"}`;
  elements["result-list"].setAttribute("aria-busy", "false");
  elements["load-more"].hidden = !state.hasMore;
  updateCompareAction();

  if (!runs.length) {
    elements["result-list"].innerHTML = `<div class="no-results"><span>0</span><h3>No matching artifacts</h3><p>Broaden the search or clear the current filters.</p><button type="button" data-action="clear">Reset filters</button></div>`;
    return;
  }

  elements["result-list"].innerHTML = runs.map((run) => {
    const model = run.metadata?.model_id || run.run_type.replaceAll("_", " ");
    const variant = run.metadata?.variant_id || "default artifact";
    const selected = run.id === state.selectedId;
    const compared = state.compareSelection.has(run.id);
    if (compared) state.compareSelection.set(run.id, run);
    const compareLabel = `${model} ${variant}`;
    return `<div class="run-entry${compared ? " compared" : ""}">
      <label class="compare-select" title="${compared ? "Remove from" : "Add to"} comparison">
        <input type="checkbox" data-compare-run-id="${run.id}" aria-label="${compared ? "Remove" : "Add"} ${escapeHtml(compareLabel)} ${compared ? "from" : "to"} comparison"${compared ? " checked" : ""} />
        <span aria-hidden="true">✓</span>
      </label>
      <button class="run-row${selected ? " selected" : ""}" type="button" data-run-id="${run.id}" aria-pressed="${selected}">
      <span class="run-identity"><strong>${escapeHtml(model)}</strong><small>${escapeHtml(variant)}</small></span>
      <span class="run-branch"><small>Branch</small>${escapeHtml(branchName(run.git_ref))}</span>
      <span class="run-compiler"><small>Compiler</small>${escapeHtml(run.toolchain_version || "—")}</span>
      <span class="run-date"><small>${escapeHtml(shortSha(run.commit_sha))}</small>${escapeHtml(formatDate(run.finished_at || run.created_at))}</span>
      <span class="row-arrow">${icons.arrow}</span>
      </button>
    </div>`;
  }).join("");
}

function comparisonMetric(metric, digits = 2) {
  if (!metric || !Number.isFinite(metric.value)) {
    return `<span class="comparison-unavailable" title="Not measured">—</span>`;
  }
  return `<span class="comparison-value">${escapeHtml(Number(metric.value).toFixed(digits))}<small>${escapeHtml(metric.unit || "")}</small></span>`;
}

function comparisonEntryFromDetail(run, detail) {
  const summary = detail?.summary || null;
  const adapted = detail?.benchmark ? adaptBenchmarkResult(detail.benchmark.result) : { kind: "missing" };
  return {
    run: summary?.run || run,
    artifacts: summary?.artifacts || [],
    benchmark: adapted.kind === "supported" ? adapted.value : null,
  };
}

async function loadComparisonEntry(run) {
  const detail = state.detailCache.get(run.id);
  if (detail) return comparisonEntryFromDetail(run, detail);
  if (state.comparisonCache.has(run.id)) return state.comparisonCache.get(run.id);

  const request = Promise.all([
    api(`/v1/runs/${run.id}`).catch(() => null),
    api(`/v1/runs/${run.id}/benchmarks/latest`).catch((error) => {
      if (error.status === 404) return null;
      return null;
    }),
  ]).then(([summary, benchmark]) => comparisonEntryFromDetail(run, { summary, benchmark }));
  state.comparisonCache.set(run.id, request);
  return request;
}

async function mapWithConcurrency(items, limit, mapper) {
  const results = new Array(items.length);
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await mapper(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

function renderComparisonTable(entries) {
  const rows = entries.map(({ run, artifacts, benchmark }) => {
    const model = run.metadata?.model_id || run.run_type?.replaceAll("_", " ") || "Unknown model";
    const variant = run.metadata?.variant_id || "default artifact";
    const artifactBytes = artifacts.reduce((total, artifact) => (
      total + (Number.isFinite(artifact.size_bytes) ? artifact.size_bytes : 0)
    ), 0);
    return `<tr>
      <th scope="row"><strong>${escapeHtml(model)}</strong><small>${escapeHtml(variant)}</small></th>
      <td>${escapeHtml(branchName(run.git_ref))}</td>
      <td>${escapeHtml(run.toolchain_version || "—")}</td>
      <td><span class="comparison-value">${escapeHtml(artifactBytes ? formatBytes(artifactBytes) : "—")}</span></td>
      <td>${comparisonMetric(benchmark?.metrics.latency)}</td>
      <td>${comparisonMetric(benchmark?.metrics.latencyP50)}</td>
      <td>${comparisonMetric(benchmark?.metrics.latencyP95)}</td>
      <td>${comparisonMetric(benchmark?.metrics.throughput, 1)}</td>
      <td>${comparisonMetric(benchmark?.metrics.power)}</td>
      <td>${comparisonMetric(benchmark?.metrics.energy, 4)}</td>
    </tr>`;
  }).join("");
  return `<div class="comparison-table-wrap"><table class="comparison-table">
    <caption class="sr-only">Artifact sizes and benchmark measurements for ${entries.length} builds</caption>
    <thead><tr>
      <th scope="col">Model / variant</th><th scope="col">Branch</th><th scope="col">Compiler</th>
      <th scope="col">Artifact size</th><th scope="col">Avg latency</th><th scope="col">p50</th>
      <th scope="col">p95</th><th scope="col">Throughput</th><th scope="col">Board power</th><th scope="col">Energy / inference</th>
    </tr></thead><tbody>${rows}</tbody>
  </table></div>`;
}

async function openComparison() {
  const runs = comparisonRuns();
  if (!runs.length) return;
  const selected = state.compareSelection.size > 0;
  elements["compare-title"].textContent = `Compare ${runs.length} build${runs.length === 1 ? "" : "s"}`;
  elements["compare-description"].textContent = selected
    ? "Showing your selected builds. Empty benchmark cells mean that measurement is unavailable."
    : "Showing all builds currently visible in the list. Empty benchmark cells mean that measurement is unavailable.";
  elements["compare-content"].innerHTML = `<div class="comparison-loading" role="status"><span class="spinner"></span> Loading artifact and benchmark data…</div>`;
  elements["compare-dialog"].showModal();
  const entries = await mapWithConcurrency(runs, 6, loadComparisonEntry);
  if (elements["compare-dialog"].open) elements["compare-content"].innerHTML = renderComparisonTable(entries);
}

function detailSkeleton() {
  elements["detail-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span><span></span></div>`;
}

async function selectRun(runId, { updateUrl = true } = {}) {
  state.selectedId = runId;
  renderRuns();
  detailSkeleton();
  if (updateUrl) {
    const url = new URL(location.href);
    url.searchParams.set("run", runId);
    history.replaceState({}, "", url);
  }
  try {
    let detail = state.detailCache.get(runId);
    if (!detail) {
      const [summary, provenance, tests, metrics, benchmark] = await Promise.all([
        api(`/v1/runs/${runId}`), api(`/v1/runs/${runId}/provenance`),
        api(`/v1/runs/${runId}/test-results`), api(`/v1/runs/${runId}/metrics`),
        api(`/v1/runs/${runId}/benchmarks/latest`).catch((error) => {
          if (error.status === 404) return null;
          throw error;
        }),
      ]);
      detail = { summary, provenance, tests, metrics, benchmark };
      state.detailCache.set(runId, detail);
    }
    if (state.selectedId === runId) renderDetail(detail);
  } catch (error) {
    if (state.selectedId === runId) {
      elements["detail-pane"].innerHTML = errorPanel(error.message, "Could not load this build", { closable: true });
    }
  }
}

function keyValue(label, value, { copy = false, mono = false } = {}) {
  const safeValue = escapeHtml(value ?? "—");
  return `<div class="key-value"><dt>${escapeHtml(label)}</dt><dd class="${mono ? "mono" : ""}"><span title="${safeValue}">${safeValue}</span>${copy && value ? `<button type="button" class="copy-button" data-copy="${safeValue}" aria-label="Copy ${escapeHtml(label)}">${icons.copy}</button>` : ""}</dd></div>`;
}

function detailTag(icon, value, kind) {
  return `<span class="detail-tag detail-tag-${kind}">${icon}<span>${escapeHtml(value)}</span></span>`;
}

function metricCard(label, metric, digits = 2) {
  const value = metric ? `${Number(metric.value).toFixed(digits)} ${metric.unit}` : "Not available";
  return `<div class="benchmark-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function availableTrialMetrics(trials) {
  return Object.keys(TRIAL_METRICS).filter((field) => (
    trials.filter((trial) => Number.isFinite(trial.metrics[field])).length >= 2
  ));
}

function trialChart(trials) {
  const available = availableTrialMetrics(trials);
  if (!available.length) return "";
  const field = available.includes(state.trialMetric) ? state.trialMetric : available[0];
  const definition = TRIAL_METRICS[field];
  const measuredTrials = trials
    .filter((trial) => Number.isFinite(trial.metrics[field]))
    .map((trial) => ({ ...trial, value: trial.metrics[field] * definition.scale }));
  const values = measuredTrials.map((trial) => trial.value);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const spread = maximum - minimum || Math.max(maximum * 0.05, 0.1);
  const low = Math.max(0, minimum - spread * 0.2);
  const high = maximum + spread * 0.2;
  const width = 640;
  const left = 70;
  const right = 28;
  const top = 28;
  const row = 34;
  const height = top + trials.length * row + 30;
  const x = (value) => left + ((value - low) / (high - low)) * (width - left - right);
  const grid = [0, .25, .5, .75, 1].map((fraction) => {
    const position = left + fraction * (width - left - right);
    const label = low + fraction * (high - low);
    return `<line x1="${position}" y1="${top - 8}" x2="${position}" y2="${height - 25}"/><text x="${position}" y="${height - 7}">${label.toFixed(2)}</text>`;
  }).join("");
  const points = measuredTrials.map((trial, index) => {
    const y = top + index * row;
    return `<text class="trial-label" x="0" y="${y + 4}">Trial ${trial.number}</text><circle cx="${x(trial.value)}" cy="${y}" r="6"/><text class="trial-value" x="${Math.min(x(trial.value) + 11, width - 55)}" y="${y + 4}">${trial.value.toFixed(definition.digits)}</text>`;
  }).join("");
  const options = available.map((metric) => (
    `<option value="${metric}"${metric === field ? " selected" : ""}>${escapeHtml(TRIAL_METRICS[metric].label)}</option>`
  )).join("");
  return `<figure class="trial-chart benchmark-chart"><figcaption><span class="chart-title">Trial comparison</span><label class="chart-control"><span>Metric</span><select data-trial-metric>${options}</select></label></figcaption><p class="chart-note">${escapeHtml(definition.unit)} · ${definition.direction} is better</p><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(definition.label)} in ${escapeHtml(definition.unit)} for ${measuredTrials.length} benchmark trials"><g class="chart-grid">${grid}</g><g class="chart-points">${points}</g></svg><table class="sr-only"><caption>${escapeHtml(definition.label)} by benchmark trial</caption><thead><tr><th>Trial</th><th>${escapeHtml(definition.label)} in ${escapeHtml(definition.unit)}</th></tr></thead><tbody>${measuredTrials.map((trial) => `<tr><td>${trial.number}</td><td>${trial.value}</td></tr>`).join("")}</tbody></table></figure>`;
}

function temperatureChart(series) {
  if (!series.length) return "";
  const hottest = (candidates) => [...candidates].sort((left, right) => (
    Math.max(...right.samples.map((sample) => sample.celsius))
    - Math.max(...left.samples.map((sample) => sample.celsius))
  ))[0];
  const temp1 = series.filter((item) => item.sensor.toLowerCase() === "temp1");
  const selected = hottest(temp1.length ? temp1 : series);
  const samples = selected.samples;
  const temperatures = samples.map((sample) => sample.celsius);
  const startTime = Math.min(...samples.map((sample) => sample.timestamp));
  const endTime = Math.max(...samples.map((sample) => sample.timestamp));
  const elapsed = Math.max(endTime - startTime, 1000);
  const minimum = Math.min(...temperatures);
  const maximum = Math.max(...temperatures);
  const spread = maximum - minimum || 1;
  const low = minimum - Math.max(spread * .25, .5);
  const high = maximum + Math.max(spread * .25, .5);
  const width = 640;
  const height = 190;
  const left = 48;
  const right = 24;
  const top = 18;
  const bottom = 35;
  const x = (timestamp) => left + ((timestamp - startTime) / elapsed) * (width - left - right);
  const y = (celsius) => top + ((high - celsius) / (high - low)) * (height - top - bottom);
  const grid = [0, .5, 1].map((fraction) => {
    const celsius = low + (1 - fraction) * (high - low);
    const position = top + fraction * (height - top - bottom);
    return `<line x1="${left}" y1="${position}" x2="${width - right}" y2="${position}"/><text class="temperature-axis-label" x="${left - 8}" y="${position + 4}">${celsius.toFixed(1)}°</text>`;
  }).join("");
  const path = samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.timestamp)} ${y(sample.celsius)}`).join(" ");
  const points = samples.map((sample) => {
    const label = sample.trial ? `Trial ${sample.trial}` : sample.phase;
    return `<circle cx="${x(sample.timestamp)}" cy="${y(sample.celsius)}" r="5"><title>${escapeHtml(label)}: ${sample.celsius.toFixed(2)} °C</title></circle>`;
  }).join("");
  const ticks = [0, .5, 1].map((fraction) => {
    const seconds = (elapsed * fraction) / 1000;
    const position = left + fraction * (width - left - right);
    return `<text x="${position}" y="${height - 8}">${seconds.toFixed(0)}s</text>`;
  }).join("");
  const legacy = selected.legacy && samples.length === 2;
  return `<figure class="temperature-chart benchmark-chart"><figcaption><span class="chart-title">Temperature ${legacy ? "change" : "over benchmark"}</span><span class="chart-series-label">${escapeHtml(selected.label)}</span></figcaption><p class="chart-note">${legacy ? "Start/end snapshots only" : `${samples.length} samples`} · ${minimum.toFixed(1)}–${maximum.toFixed(1)} °C</p><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Temperature in degrees Celsius for ${escapeHtml(selected.label)}"><g class="chart-grid">${grid}${ticks}</g><path class="temperature-line" d="${path}"/><g class="temperature-points">${points}</g></svg><table class="sr-only"><caption>Temperature samples for ${escapeHtml(selected.label)}</caption><thead><tr><th>Elapsed time</th><th>Temperature in degrees Celsius</th></tr></thead><tbody>${samples.map((sample) => `<tr><td>${((sample.timestamp - startTime) / 1000).toFixed(1)} seconds</td><td>${sample.celsius}</td></tr>`).join("")}</tbody></table></figure>`;
}

function benchmarkPanel(payload) {
  if (!payload) return `<section><div class="detail-section-title"><h3>Benchmark</h3></div><p class="muted-message">No completed benchmark is linked to this compilation yet.</p></section>`;
  const adapted = adaptBenchmarkResult(payload.result);
  if (adapted.kind === "unsupported") {
    return `<section id="benchmark-section" tabindex="-1"><div class="detail-section-title"><h3>Benchmark</h3><span>Schema v${escapeHtml(adapted.schemaVersion)}</span></div><p class="muted-message">This benchmark schema is newer than this browser can render.</p>${payload.result_artifact_url ? `<a class="external-link" href="${escapeHtml(payload.result_artifact_url)}">Open raw result ${icons.external}</a>` : ""}</section>`;
  }
  if (adapted.kind !== "supported") return "";
  const benchmark = adapted.value;
  return `<section id="benchmark-section" class="benchmark-section" tabindex="-1"><div class="detail-section-title"><h3>Benchmark</h3><span>Schema v${benchmark.schemaVersion}</span></div>
    <div class="benchmark-summary">${metricCard("Average latency", benchmark.metrics.latency)}${metricCard("Throughput", benchmark.metrics.throughput, 1)}${metricCard("Board power", benchmark.metrics.power)}${metricCard("Energy / inference", benchmark.metrics.energy, 4)}</div>
    ${trialChart(benchmark.trials)}
    ${temperatureChart(benchmark.temperatureSeries)}
    <dl class="benchmark-context">${keyValue("Samples / trial", benchmark.samples)}${keyValue("Platform", benchmark.platformVersion)}${keyValue("Neat", benchmark.neatVersion)}${keyValue("Measured", formatDate(benchmark.timestamp))}</dl>
  </section>`;
}

function renderDetail(detail) {
  const run = detail.summary.run;
  const artifacts = detail.summary.artifacts || [];
  const provenance = detail.provenance;
  const parameters = provenance.parameters?.parameters || {};
  const input = provenance.inputs?.[0];
  const tests = detail.tests.test_results || [];
  const model = run.metadata?.model_id || run.run_type;
  const variant = run.metadata?.variant_id || "default";
  const precision = parameters.compile_parameters?.precision;
  const categories = run.model_categories || {};
  const sourceUrl = githubModelUrl(run, model);
  const sourceLabel = `View ${model} source on GitHub at commit ${shortSha(run.commit_sha)}`;
  const benchmarkMarkup = benchmarkPanel(detail.benchmark);
  const hasBenchmark = Boolean(detail.benchmark && benchmarkMarkup);
  const primaryArtifact = artifacts[0];
  const huggingFaceSource = input ? huggingFaceSourceLink(input) : null;

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <div class="detail-topline"><button class="detail-layout-toggle" type="button" aria-pressed="${state.detailLayout === "focused"}" title="Give the selected model more space">${state.detailLayout === "focused" ? "Show wider list" : "Focus model"}</button><button class="close-detail" type="button" aria-label="Close details">×</button></div>
    <div class="detail-title"><div class="detail-title-copy"><p class="kicker">${escapeHtml(run.target_platform || "Model artifact")}</p><h2>${escapeHtml(model)}</h2><p>${escapeHtml(variant)}</p></div><div class="detail-title-actions">${primaryArtifact ? `<button class="artifact-download-shortcut" type="button" data-download="${escapeHtml(primaryArtifact.name)}" data-run-id="${run.id}" aria-label="Download model artifact" data-tooltip="Download model artifact">${icons.download}</button>` : ""}${mpkValidationMark(tests)}${hasBenchmark ? `<button class="benchmark-shortcut" type="button" data-scroll-benchmark aria-label="Jump to benchmark results" data-tooltip="Jump to benchmark results">${icons.benchmark}</button>` : ""}${huggingFaceSource ? `<a class="huggingface-source-link" href="${escapeHtml(huggingFaceSource.href)}" target="_blank" rel="noreferrer" aria-label="Open source ONNX model card on Hugging Face" data-tooltip="Open source ONNX model card on Hugging Face">${icons.huggingFace}</a>` : ""}${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" data-tooltip="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}</div></div>
    <div class="detail-tags">${precision ? detailTag(icons.precision, precision, "precision") : ""}${detailTag(icons.compiler, `${run.toolchain_name || "compiler"} ${run.toolchain_version || ""}`.trim(), "compiler")}${detailTag(icons.branch, branchName(run.git_ref), "branch")}${categories.model_type ? detailTag(icons.modelType, categories.model_type, "model-type") : ""}${categories.architecture?.id ? detailTag(icons.architecture, categories.architecture.id, "architecture") : ""}${(categories.tasks || []).map((task) => detailTag(icons.task, task, "task")).join("")}</div>

    <section class="artifact-section"><div class="detail-section-title"><h3>Artifacts</h3><span>${artifacts.length}</span></div>
      ${artifacts.length ? artifacts.map((artifact) => `<dl class="artifact-metadata">
        <div><dt>File size</dt><dd>${escapeHtml(formatBytes(artifact.size_bytes))}</dd></div>
        <div><dt>SHA-256</dt><dd class="artifact-hash"><code>${escapeHtml(artifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(artifact.sha256)}" aria-label="Copy artifact checksum">${icons.copy}</button></dd></div>
      </dl>`).join("") : `<p class="muted-message">No artifact metadata was registered for this run.</p>`}
    </section>

    <section><div class="detail-section-title"><h3>Build provenance</h3></div><dl class="key-grid">
      ${keyValue("Commit", run.commit_sha, { copy: true, mono: true })}
      ${keyValue("Branch", branchName(run.git_ref))}
      ${keyValue("Built", formatDate(run.finished_at || run.created_at))}
      ${keyValue("Duration", formatDuration(run.started_at, run.finished_at))}
      ${keyValue("Target", run.target_platform)}
      ${keyValue("Compiler", run.toolchain_version)}
      ${keyValue("Run ID", run.id, { copy: true, mono: true })}
      ${keyValue("Fingerprint", run.run_fingerprint, { copy: true, mono: true })}
    </dl>${run.ci_run_url ? `<a class="external-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">Open GitHub workflow ${icons.external}</a>` : ""}</section>

    ${input ? sourceInputPanel(input) : ""}

    ${benchmarkMarkup}

    ${Object.keys(parameters.compile_parameters || {}).length ? `<details><summary>Compile parameters <span>JSON</span></summary><pre>${escapeHtml(JSON.stringify(parameters.compile_parameters, null, 2))}</pre></details>` : ""}
  </div>`;
}

function errorPanel(message, title = "Registry unavailable", { closable = false } = {}) {
  return `<div class="error-state">${closable ? `<button class="close-detail" type="button" aria-label="Close details">×</button>` : ""}<span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}

async function downloadArtifact(runId, name, button) {
  const prior = button.innerHTML;
  const priorLabel = button.getAttribute("aria-label");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.setAttribute("aria-label", "Preparing model artifact download");
  button.innerHTML = `<span class="spinner"></span>`;
  try {
    const payload = await api(`/v1/runs/${runId}/artifacts/${encodeURIComponent(name)}`);
    if (!payload.artifact?.download_url) throw new Error("The registry did not return a download URL.");
    showToast(`Download ready for ${name}. The link is valid for ${Math.round((payload.artifact.download_url_expires_in || 300) / 60)} minutes.`);
    location.href = payload.artifact.download_url;
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    if (priorLabel) button.setAttribute("aria-label", priorLabel);
    button.innerHTML = prior;
  }
}

function showToast(message, isError = false) {
  const toast = document.createElement("div");
  toast.className = `toast${isError ? " error" : ""}`;
  toast.textContent = message;
  elements["toast-region"].append(toast);
  setTimeout(() => toast.remove(), 5000);
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    showToast("Copied to clipboard.");
  } catch {
    showToast("Clipboard access is unavailable.", true);
  }
}

async function loadRuns({ append = false, branch = elements["branch-filter"].value, limit = 50 } = {}) {
  const repo = state.repository;
  if (!repo) return;
  const requestId = ++state.runsRequestId;
  elements["load-more"].disabled = true;
  const query = new URLSearchParams({ limit: String(limit) });
  query.set("status", "completed");
  query.set("run_type", "model_compile");
  if (!elements["history-filter"].checked) query.set("latest_per_model_compiler", "true");
  if (branch) query.set("git_ref", `refs/heads/${branch}`);
  if (elements.search.value.trim()) query.set("q", elements.search.value.trim());
  Object.entries(categoryFilters()).forEach(([name, value]) => {
    if (value) query.set(name, value);
  });
  if (append && state.cursor) query.set("cursor", state.cursor);

  try {
    const payload = await api(`/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/runs?${query}`);
    if (requestId !== state.runsRequestId) return;
    if (!append && branch && !payload.runs.length && !elements.search.value.trim()
      && !Object.values(categoryFilters()).some(Boolean)) {
      const knownBranches = new Set([
        ...(payload.branches || []),
        ...state.availableBranches,
      ]);
      knownBranches.delete(branch);
      state.availableBranches = [...knownBranches].sort();
      const fallbackBranch = branch !== DEFAULT_BRANCH && knownBranches.has(DEFAULT_BRANCH) ? DEFAULT_BRANCH : "";
      state.preferredBranch = fallbackBranch;
      writeBranchPreference(fallbackBranch);
      await loadRuns({ branch: fallbackBranch, limit: fallbackBranch ? 50 : 200 });
      return;
    }
    applyCatalog(payload, { append });
  } catch (error) {
    if (requestId === state.runsRequestId) throw error;
  } finally {
    if (requestId === state.runsRequestId) elements["load-more"].disabled = false;
  }

}

async function initialize() {
  setTheme(document.documentElement.dataset.theme || "light");
  elements.search.value = initialSearch;
  elements["history-filter"].checked = initialIncludeHistory;
  state.preferredBranch = readBranchPreference();
  elements["environment-name"].textContent = `${config.environment || "Public"} registry`;
  elements["result-list"].innerHTML = Array.from({ length: 5 }, () => `<div class="row-skeleton"><span></span><span></span><span></span></div>`).join("");

  const validApiOrigin = /^https:\/\/[^/]+$/.test(state.apiBase)
    || /^http:\/\/(?:127\.0\.0\.1|localhost)(?::[0-9]+)?$/.test(state.apiBase);
  if (!validApiOrigin || !state.repository.organization || !state.repository.name) {
    elements["result-list"].innerHTML = errorPanel("Set a valid HTTPS API origin and repository in config.js.", "Portal configuration is invalid");
    elements["result-list"].setAttribute("aria-busy", "false");
    return;
  }

  const cached = readCachedCatalog();
  if (cached) {
    state.availableBranches = Array.isArray(cached.branches) ? cached.branches : [];
    state.taxonomy = cached.taxonomy || null;
    if (state.taxonomy) populateCategoryFilters();
    applyCatalog({
      runs: cached.runs,
      next_cursor: cached.cursor,
      has_more: cached.hasMore,
    }, { cache: false });
  }

  try {
    try {
      await loadTaxonomy();
    } catch (error) {
      setCategoryFiltersDisabled(true);
      showToast(`Category filters are temporarily unavailable. ${error.message}`, true);
    }
    if (activeFilterCount()) setFilterDrawer(true);
    if (cached) {
      await loadRuns({ branch: state.preferredBranch });
    } else {
      // The first broad page discovers branch choices. If it is incomplete,
      // switch to branch-aware pagination so the cursor matches the visible list.
      await loadRuns({ branch: "", limit: 200 });
      const selectedBranch = elements["branch-filter"].value;
      if (selectedBranch && state.hasMore) await loadRuns({ branch: selectedBranch });
    }
    elements["result-title"].textContent = state.repository.name === "models" ? "Models" : state.repository.name;
  } catch (error) {
    if (cached) showToast(error.message, true);
    else {
      elements["result-list"].innerHTML = errorPanel(error.message);
      elements["result-list"].setAttribute("aria-busy", "false");
    }
  }

  if (state.selectedId) await selectRun(state.selectedId, { updateUrl: false });
}

elements["theme-toggle"].addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
elements["filter-toggle"].addEventListener("click", () => {
  setFilterDrawer(elements["filter-toggle"].getAttribute("aria-expanded") !== "true");
});
let searchTimer;
elements.search.addEventListener("input", () => {
  renderRuns();
  writeSearchUrl();
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadRuns({ branch: state.preferredBranch }).catch((error) => showToast(error.message, true)), 250);
});
elements["branch-filter"].addEventListener("change", async () => {
  state.preferredBranch = elements["branch-filter"].value;
  writeBranchPreference(state.preferredBranch);
  updateFilterDisclosure();
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
});
elements["compiler-filter"].addEventListener("change", () => {
  updateFilterDisclosure();
  renderRuns();
});
elements["history-filter"].addEventListener("change", async () => {
  writeHistoryUrl();
  updateFilterDisclosure();
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
});
Object.keys(CATEGORY_QUERY_FIELDS).forEach((elementId) => elements[elementId].addEventListener("change", async () => {
  state.categorySelection[elementId] = elements[elementId].value;
  writeCategoryFilterUrl();
  updateFilterDisclosure();
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
}));
elements["clear-filters"].addEventListener("click", async () => {
  elements.search.value = "";
  elements["branch-filter"].value = "";
  state.preferredBranch = "";
  writeBranchPreference(state.preferredBranch);
  elements["compiler-filter"].value = "";
  elements["history-filter"].checked = false;
  Object.keys(CATEGORY_QUERY_FIELDS).forEach((elementId) => {
    elements[elementId].value = "";
    state.categorySelection[elementId] = "";
  });
  writeSearchUrl();
  writeCategoryFilterUrl();
  writeHistoryUrl();
  updateFilterDisclosure();
  try {
    await loadRuns({ branch: "", limit: 200 });
  } catch (error) {
    showToast(error.message, true);
  }
});
elements["result-list"].addEventListener("click", (event) => {
  const run = event.target.closest("[data-run-id]");
  if (run) selectRun(run.dataset.runId);
  if (event.target.closest('[data-action="clear"]')) elements["clear-filters"].click();
  if (event.target.closest('[data-action="retry"]')) initialize();
});
elements["result-list"].addEventListener("change", (event) => {
  const checkbox = event.target.closest("[data-compare-run-id]");
  if (!checkbox) return;
  const run = state.runs.find((item) => item.id === checkbox.dataset.compareRunId);
  if (!run) return;
  if (checkbox.checked) state.compareSelection.set(run.id, run);
  else state.compareSelection.delete(run.id);
  checkbox.closest(".run-entry")?.classList.toggle("compared", checkbox.checked);
  checkbox.setAttribute("aria-label", `${checkbox.checked ? "Remove" : "Add"} ${run.metadata?.model_id || run.run_type} ${run.metadata?.variant_id || "default artifact"} ${checkbox.checked ? "from" : "to"} comparison`);
  checkbox.closest("label")?.setAttribute("title", `${checkbox.checked ? "Remove from" : "Add to"} comparison`);
  updateCompareAction();
});
elements["compare-button"].addEventListener("click", () => {
  openComparison().catch((error) => {
    elements["compare-dialog"].close();
    showToast(error.message, true);
  });
});
elements["compare-close"].addEventListener("click", () => elements["compare-dialog"].close());
elements["compare-dialog"].addEventListener("click", (event) => {
  if (event.target === elements["compare-dialog"]) elements["compare-dialog"].close();
});
elements["detail-pane"].addEventListener("click", (event) => {
  if (event.target.closest('[data-action="retry"]') && state.selectedId) {
    selectRun(state.selectedId, { updateUrl: false });
    return;
  }
  const copy = event.target.closest("[data-copy]");
  if (copy) copyText(copy.dataset.copy);
  const download = event.target.closest("[data-download]");
  if (download) downloadArtifact(download.dataset.runId, download.dataset.download, download);
  if (event.target.closest("[data-scroll-benchmark]")) {
    const benchmark = elements["detail-pane"].querySelector("#benchmark-section");
    if (benchmark) {
      benchmark.focus({ preventScroll: true });
      benchmark.scrollIntoView({
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "start",
      });
    }
    return;
  }
  if (event.target.closest(".detail-layout-toggle")) {
    state.detailLayout = state.detailLayout === "focused" ? "balanced" : "focused";
    try { localStorage.setItem(DETAIL_LAYOUT_KEY, state.detailLayout); } catch { /* Storage is optional. */ }
    applyDetailLayout();
    if (state.selectedId) renderDetail(state.detailCache.get(state.selectedId));
    return;
  }
  if (event.target.closest(".close-detail")) {
    state.selectedId = null;
    const url = new URL(location.href);
    url.searchParams.delete("run");
    history.replaceState({}, "", url);
    renderRuns();
    elements["detail-pane"].innerHTML = `<div class="detail-empty"><h2>No build selected</h2><p>Select a build to inspect provenance, validation results, checksums, and downloads.</p></div>`;
  }
});
elements["detail-pane"].addEventListener("change", (event) => {
  const trialMetric = event.target.closest("[data-trial-metric]");
  if (trialMetric && TRIAL_METRICS[trialMetric.value]) {
    state.trialMetric = trialMetric.value;
    if (state.selectedId) renderDetail(state.detailCache.get(state.selectedId));
    return;
  }
});
elements["load-more"].addEventListener("click", () => loadRuns({
  append: true,
  branch: elements["branch-filter"].value,
}).catch((error) => showToast(error.message, true)));
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    elements.search.focus();
  }
  if (event.key === "Escape" && document.activeElement === elements.search) {
    elements.search.value = "";
    elements.search.blur();
    elements.search.dispatchEvent(new Event("input"));
  }
});

applyDetailLayout();
initialize();
