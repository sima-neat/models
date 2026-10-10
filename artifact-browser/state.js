// Shared page state, configuration, DOM elements, and the readers of that state.
export const config = window.MODEL_REGISTRY_CONFIG || {};
const configuredApiBase = String(config.apiBase || "").replace(/\/$/, "");
const configuredRepository = config.repository || {};
export const DEFAULT_BRANCH = "develop";
export const CATEGORY_QUERY_FIELDS = {
  "model-type-filter": "model_type",
  "task-filter": "task",
  "architecture-filter": "architecture_id",
  "paradigm-filter": "architecture_paradigm",
  "input-modality-filter": "input_modality",
};
const initialQuery = new URLSearchParams(location.search);
export const DETAIL_LAYOUT_KEY = "sima-neat-model-detail-layout";

export const state = {
  apiBase: configuredApiBase,
  repository: configuredRepository,
  models: [],
  runs: [],
  compileRuns: [],
  packageRuns: [],
  catalogDefinitions: [],
  catalogMode: "runs",
  cursor: null,
  hasMore: false,
  availableBranches: [],
  runsRequestId: 0,
  navigationId: 0,
  taxonomyRequestId: 0,
  selectedId: new URLSearchParams(location.search).get("run"),
  preferredBranch: null,
  urlBranch: false,
  detailCache: new Map(),
  // Variant builds by build key, the selected one, and builds that can be selected for comparison.
  variantCache: new Map(),
  selectedVariantKey: null,
  variantGroups: new Map(),
  expandedRuns: new Set(),
  historicalCatalogModels: new Map(),
  documentationCache: new Map(),
  expandedDocumentationModelId: null,
  documentationRequestId: 0,
  comparisonCache: new Map(),
  compareSelection: new Map(),
  activeProfileByModel: new Map(),
  taxonomy: null,
  detailLayout: readDetailLayout(),
  trialMetric: "latency_avg_ms",
  comparisonMetric: "latency_avg_ms",
  comparisonEntries: [],
  categorySelection: Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, initialQuery.get(queryName) || ""]),
  ),
};

export const elements = Object.fromEntries(
  [
    "theme-toggle", "environment-name", "search", "filter-toggle",
    "filter-count", "filter-drawer", "branch-filter", "compiler-filter", "history-filter", "model-type-filter", "task-filter",
    "architecture-filter", "paradigm-filter", "input-modality-filter", "clear-filters", "result-title",
    "result-count", "result-list", "load-more", "detail-pane", "toast-region", "workspace",
    "compare-button", "compare-label", "compare-count", "compare-dialog", "compare-close", "compare-content",
    "compare-title", "compare-description",
    "download-dialog", "download-title", "download-description", "download-package-info",
    "download-list", "download-confirm", "download-cancel", "download-status",
  ].map((id) => [id, document.getElementById(id)]),
);

function readDetailLayout() {
  try { return localStorage.getItem(DETAIL_LAYOUT_KEY) === "focused" ? "focused" : "balanced"; }
  catch { return "balanced"; }
}

export function applyDetailLayout() {
  elements.workspace.dataset.detailLayout = state.detailLayout;
}

export function categoryFilters() {
  return Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [
      queryName,
      elements[elementId].disabled ? "" : (state.taxonomy ? elements[elementId].value : state.categorySelection[elementId]),
    ]),
  );
}

export function filters() {
  return {
    query: elements.search.value.trim(),
    branch: elements["branch-filter"].value,
    compiler: elements["compiler-filter"].value,
    ...categoryFilters(),
  };
}

export function catalogMode() {
  return state.catalogMode === "models";
}

export function branchSnapshotMode() {
  return Boolean(elements["branch-filter"].value) && !elements["history-filter"].checked;
}

export function currentVariant() {
  return state.variantCache.get(state.selectedVariantKey) || null;
}
