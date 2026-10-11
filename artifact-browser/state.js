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
  // Variant builds by build key, and the selected one.
  variantCache: new Map(),
  selectedVariantKey: null,
  // Keys of the disclosure rows the reader opened on the current model page.
  openDisclosures: new Set(),
  historicalCatalogModels: new Map(),
  documentationCache: new Map(),
  documentationRequestId: 0,
  // The Compare page's three pickers, each `<model id>.<build profile>` or empty.
  comparePicks: ["", "", ""],
  activeProfileByModel: new Map(),
  taxonomy: null,
  categorySelection: Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, initialQuery.get(queryName) || ""]),
  ),
};

export const elements = Object.fromEntries(
  [
    "theme-toggle", "environment-name", "search", "filter-toggle",
    "filter-count", "filter-drawer", "more-filters", "branch-filter", "compiler-filter", "history-filter", "model-type-filter", "task-filter",
    "architecture-filter", "paradigm-filter", "input-modality-filter", "clear-filters", "result-title",
    "result-count", "result-list", "load-more", "detail-pane", "compare-pane", "toast-region", "workspace",
    "nav-models", "nav-compare",
    "download-dialog", "download-title", "download-description", "download-package-info",
    "download-list", "download-confirm", "download-cancel", "download-status",
  ].map((id) => [id, document.getElementById(id)]),
);

export function taxonomyLabel(section, id) {
  return state.taxonomy?.[section]?.[id]?.display_name || id || "";
}

// The workspace shows the model catalog, one model page, or the Compare page; returning to the catalog
// restores where the reader left it.
// Each view's container and the element that takes focus when the view opens.
const VIEWS = { catalog: [".catalog-view", "result-title"], model: [".detail-pane", "detail-pane"], compare: [".compare-pane", "compare-pane"] };
let catalogScrollY = 0;

export function showView(view) {
  const current = elements.workspace.dataset.view;
  if (current === view) return;
  if (current === "catalog") catalogScrollY = window.scrollY;
  const hidden = elements.workspace.querySelector(VIEWS[current][0]);
  // A re-render may already have removed the focused element, leaving focus on the body.
  const focusWasHidden = document.activeElement === document.body || hidden.contains(document.activeElement);
  elements.workspace.dataset.view = view;
  const navigation = view === "compare" ? ["nav-compare", "nav-models"] : ["nav-models", "nav-compare"];
  elements[navigation[0]].setAttribute("aria-current", "page");
  elements[navigation[1]].removeAttribute("aria-current");
  window.scrollTo({ top: view === "catalog" ? catalogScrollY : 0 });
  // Focus left inside the hidden view would fall to the page body; land it at the start of the shown view.
  if (focusWasHidden) elements[VIEWS[view][1]].focus({ preventScroll: true });
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
