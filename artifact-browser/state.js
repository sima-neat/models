// Shared page state, configuration, DOM elements, and the readers of that state.
import {audienceFacts} from "./audience.js";
import {resolveEnvironment, branchPolicy} from "./environment.js";
export const config = resolveEnvironment(window.MODEL_REGISTRY_CONFIG || {}, location.search);
const configuredApiBase = String(config.apiBase || "").replace(/\/$/, "");
const configuredRepository = config.repository || {};
export const shows = audienceFacts(config.uiMode);
const branches = branchPolicy(config, location.search);
export const branchSelectionEnabled = branches.enabled;
export const DEFAULT_BRANCH = branches.defaultBranch;
document.documentElement.dataset.branches = branchSelectionEnabled ? "1" : "0";
document.documentElement.dataset.uiMode = config.uiMode === "customer" ? "customer" : "developer";
export const CATEGORY_QUERY_FIELDS = {
  "task-filter": "task",
  "architecture-filter": "architecture_id",
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
  // Whether the registry has answered for the selected branch. Until then an empty catalog is still loading.
  catalogLoaded: false,
  catalogLayout: "cards",
  cursor: null,
  hasMore: false,
  availableBranches: [],
  runsRequestId: 0,
  navigationId: 0,
  selectedId: new URLSearchParams(location.search).get("run"),
  preferredBranch: null,
  urlBranch: false,
  detailCache: new Map(),
  // Variant builds by build key, and the selected one.
  variantCache: new Map(),
  selectedVariantKey: null,
  // Keys of the disclosure rows the reader opened on the current model page.
  openDisclosures: new Set(),
  layerViews: new Map(),
  historicalCatalogModels: new Map(),
  documentationCache: new Map(),
  documentationRequestId: 0,
  // The Compare page's selected builds, each `<model id>.<build profile>` or empty.
  comparePicks: ["", "", ""],
  compareTask: initialQuery.get("compare_task") || "",
  compareAxis: "",
  compareWorkload: initialQuery.get("compare_workload") || "",
  activeProfileByModel: new Map(),
  taxonomy: null,
  taxonomyBranch: null,
  categorySelection: Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, initialQuery.get(queryName) || ""]),
  ),
};

export const elements = Object.fromEntries(
  [
    "theme-toggle", "environment-name", "search", "filter-toggle",
    "filter-count", "filter-drawer", "branch-filter", "compiler-filter", "history-filter", "task-filter",
    "architecture-filter", "clear-filters", "clear-family", "result-title",
    "result-count", "result-list", "catalog-layout-toggle", "load-more", "detail-pane", "compare-pane", "toast-region", "workspace",
    "nav-models", "nav-compare",
    "download-dialog", "download-title", "download-description", "download-package-info",
    "download-list", "download-confirm", "download-copy-cli", "download-cancel", "download-status", "execution-tooltip",
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
