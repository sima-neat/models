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
  taxonomy: null,
  detailLayout: readDetailLayout(),
  categorySelection: Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, initialQuery.get(queryName) || ""]),
  ),
};

const elements = Object.fromEntries(
  [
    "theme-toggle", "environment-name", "registry-status", "registry-copy", "search", "filter-toggle",
    "filter-count", "filter-drawer", "branch-filter", "compiler-filter", "history-filter", "model-type-filter", "task-filter",
    "architecture-filter", "paradigm-filter", "input-modality-filter", "clear-filters", "result-title",
    "result-count", "result-list", "load-more", "detail-pane", "toast-region", "workspace",
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

function renderRuns() {
  const runs = filteredRuns();
  elements["result-count"].textContent = `${runs.length}${state.hasMore ? "+" : ""} build${runs.length === 1 ? "" : "s"}`;
  elements["result-list"].setAttribute("aria-busy", "false");
  elements["load-more"].hidden = !state.hasMore;

  if (!runs.length) {
    elements["result-list"].innerHTML = `<div class="no-results"><span>0</span><h3>No matching artifacts</h3><p>Broaden the search or clear the current filters.</p><button type="button" data-action="clear">Reset filters</button></div>`;
    return;
  }

  elements["result-list"].innerHTML = runs.map((run) => {
    const model = run.metadata?.model_id || run.run_type.replaceAll("_", " ");
    const variant = run.metadata?.variant_id || "default artifact";
    const selected = run.id === state.selectedId;
    return `<button class="run-row${selected ? " selected" : ""}" type="button" data-run-id="${run.id}" aria-pressed="${selected}">
      <span class="run-identity"><strong>${escapeHtml(model)}</strong><small>${escapeHtml(variant)}</small></span>
      <span class="run-branch"><small>Branch</small>${escapeHtml(branchName(run.git_ref))}</span>
      <span class="run-compiler"><small>Compiler</small>${escapeHtml(run.toolchain_version || "—")}</span>
      <span class="run-date"><small>${escapeHtml(shortSha(run.commit_sha))}</small>${escapeHtml(formatDate(run.finished_at || run.created_at))}</span>
      <span class="row-arrow">${icons.arrow}</span>
    </button>`;
  }).join("");
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

function metricCard(label, metric, digits = 2) {
  const value = metric ? `${Number(metric.value).toFixed(digits)} ${metric.unit}` : "Not available";
  return `<div class="benchmark-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function trialChart(trials) {
  if (trials.length < 2) return "";
  const values = trials.map((trial) => trial.latencyMs);
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
  const points = trials.map((trial, index) => {
    const y = top + index * row;
    return `<text class="trial-label" x="0" y="${y + 4}">Trial ${trial.number}</text><circle cx="${x(trial.latencyMs)}" cy="${y}" r="6"/><text class="trial-value" x="${Math.min(x(trial.latencyMs) + 11, width - 55)}" y="${y + 4}">${trial.latencyMs.toFixed(2)}</text>`;
  }).join("");
  return `<figure class="trial-chart"><figcaption>Latency consistency <span>milliseconds · lower is better</span></figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Average latency in milliseconds for ${trials.length} benchmark trials"><g class="chart-grid">${grid}</g><g class="chart-points">${points}</g></svg><table class="sr-only"><caption>Latency by benchmark trial</caption><thead><tr><th>Trial</th><th>Average latency in milliseconds</th></tr></thead><tbody>${trials.map((trial) => `<tr><td>${trial.number}</td><td>${trial.latencyMs}</td></tr>`).join("")}</tbody></table></figure>`;
}

function benchmarkPanel(payload) {
  if (!payload) return `<section><div class="detail-section-title"><h3>Benchmark</h3></div><p class="muted-message">No completed benchmark is linked to this compilation yet.</p></section>`;
  const adapted = adaptBenchmarkResult(payload.result);
  if (adapted.kind === "unsupported") {
    return `<section><div class="detail-section-title"><h3>Benchmark</h3><span>Schema v${escapeHtml(adapted.schemaVersion)}</span></div><p class="muted-message">This benchmark schema is newer than this browser can render.</p>${payload.result_artifact_url ? `<a class="external-link" href="${escapeHtml(payload.result_artifact_url)}">Open raw result ${icons.external}</a>` : ""}</section>`;
  }
  if (adapted.kind !== "supported") return "";
  const benchmark = adapted.value;
  return `<section class="benchmark-section"><div class="detail-section-title"><h3>Benchmark</h3><span>Schema v${benchmark.schemaVersion}</span></div>
    <div class="benchmark-summary">${metricCard("Average latency", benchmark.metrics.latency)}${metricCard("Throughput", benchmark.metrics.throughput, 1)}${metricCard("Board power", benchmark.metrics.power)}${metricCard("Energy / inference", benchmark.metrics.energy, 4)}</div>
    ${trialChart(benchmark.trials)}
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
  const measurements = detail.metrics.measurements || [];
  const model = run.metadata?.model_id || run.run_type;
  const variant = run.metadata?.variant_id || "default";
  const precision = parameters.compile_parameters?.precision;
  const categories = run.model_categories || {};
  const sourceUrl = githubModelUrl(run, model);
  const sourceLabel = `View ${model} source on GitHub at commit ${shortSha(run.commit_sha)}`;

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <div class="detail-topline"><button class="detail-layout-toggle" type="button" aria-pressed="${state.detailLayout === "focused"}" title="Give the selected model more space">${state.detailLayout === "focused" ? "Show wider list" : "Focus model"}</button><button class="close-detail" type="button" aria-label="Close details">×</button></div>
    <div class="detail-title"><div class="detail-title-copy"><p class="kicker">${escapeHtml(run.target_platform || "Model artifact")}</p><h2>${escapeHtml(model)}</h2><p>${escapeHtml(variant)}</p></div>${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" title="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}</div>
    <div class="detail-tags">${precision ? `<span>${escapeHtml(precision)}</span>` : ""}<span>${escapeHtml(run.toolchain_name || "compiler")} ${escapeHtml(run.toolchain_version || "")}</span><span>${escapeHtml(branchName(run.git_ref))}</span>${categories.model_type ? `<span>${escapeHtml(categories.model_type)}</span>` : ""}${categories.architecture?.id ? `<span>${escapeHtml(categories.architecture.id)}</span>` : ""}${(categories.tasks || []).map((task) => `<span>${escapeHtml(task)}</span>`).join("")}</div>

    <section class="artifact-section"><div class="detail-section-title"><h3>Artifacts</h3><span>${artifacts.length}</span></div>
      ${artifacts.length ? artifacts.map((artifact) => `<div class="artifact-item">
        <div class="artifact-icon" aria-hidden="true">MPK</div>
        <div class="artifact-main"><strong>${escapeHtml(artifact.name)}</strong><small>${escapeHtml(formatBytes(artifact.size_bytes))} · ${escapeHtml(artifact.media_type || artifact.artifact_type)}</small></div>
        <button class="download-button" type="button" data-download="${escapeHtml(artifact.name)}" data-run-id="${run.id}">${icons.download}<span>Download</span></button>
        <div class="artifact-checksum"><span>SHA-256</span><code>${escapeHtml(artifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(artifact.sha256)}" aria-label="Copy artifact checksum">${icons.copy}</button></div>
      </div>`).join("") : `<p class="muted-message">No downloadable artifacts were registered for this run.</p>`}
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

    ${input ? `<section><div class="detail-section-title"><h3>Source input</h3></div><div class="source-block"><strong>${escapeHtml(input.name)}</strong><p>${escapeHtml(input.source_reference || input.input_type)}</p><div><span>${escapeHtml(formatBytes(input.size_bytes))}</span><code>${escapeHtml(input.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(input.sha256)}" aria-label="Copy source checksum">${icons.copy}</button></div></div></section>` : ""}

    <section><div class="detail-section-title"><h3>Validation</h3><span>${tests.length}</span></div>
      <div class="test-list">${tests.length ? tests.map((test) => `<div><span class="test-mark ${escapeHtml(test.status)}">${test.status === "passed" ? "✓" : "!"}</span><p><strong>${escapeHtml(test.name)}</strong><small>${escapeHtml(test.suite)}${test.duration_ms ? ` · ${escapeHtml(test.duration_ms)} ms` : ""}</small></p><span>${escapeHtml(test.status)}</span></div>`).join("") : `<p class="muted-message">No validation records.</p>`}</div>
      ${measurements.length ? `<p class="measurement-note">${measurements.length} performance measurement${measurements.length === 1 ? "" : "s"} recorded.</p>` : ""}
    </section>

    ${benchmarkPanel(detail.benchmark)}

    ${Object.keys(parameters.compile_parameters || {}).length ? `<details><summary>Compile parameters <span>JSON</span></summary><pre>${escapeHtml(JSON.stringify(parameters.compile_parameters, null, 2))}</pre></details>` : ""}
  </div>`;
}

function errorPanel(message, title = "Registry unavailable", { closable = false } = {}) {
  return `<div class="error-state">${closable ? `<button class="close-detail" type="button" aria-label="Close details">×</button>` : ""}<span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}

async function downloadArtifact(runId, name, button) {
  const prior = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<span class="spinner"></span><span>Preparing…</span>`;
  try {
    const payload = await api(`/v1/runs/${runId}/artifacts/${encodeURIComponent(name)}`);
    if (!payload.artifact?.download_url) throw new Error("The registry did not return a download URL.");
    showToast(`Download ready for ${name}. The link is valid for ${Math.round((payload.artifact.download_url_expires_in || 300) / 60)} minutes.`);
    location.href = payload.artifact.download_url;
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
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

  if (!/^https:\/\/[^/]+$/.test(state.apiBase) || !state.repository.organization || !state.repository.name) {
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
    elements["registry-copy"].textContent = `${state.repository.organization}/${state.repository.name} · refreshing…`;
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
    elements["registry-status"].className = "status-dot online";
    elements["registry-copy"].textContent = `${state.repository.organization}/${state.repository.name} · live public data`;
    elements["result-title"].textContent = state.repository.name === "models" ? "Models" : state.repository.name;
  } catch (error) {
    elements["registry-status"].className = "status-dot offline";
    elements["registry-copy"].textContent = cached ? "Showing saved data · registry unavailable" : "Registry unavailable";
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
elements["detail-pane"].addEventListener("click", (event) => {
  if (event.target.closest('[data-action="retry"]') && state.selectedId) {
    selectRun(state.selectedId, { updateUrl: false });
    return;
  }
  const copy = event.target.closest("[data-copy]");
  if (copy) copyText(copy.dataset.copy);
  const download = event.target.closest("[data-download]");
  if (download) downloadArtifact(download.dataset.runId, download.dataset.download, download);
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
