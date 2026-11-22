import { heroImageSources } from "./hero-images.js";
// Registry API requests, local caches, and catalog build selection.
import { deploymentCatalog, runProfile, variantBuildKey, variantBuildRuns } from "./branch-builds.js";
import { BENCHMARK_RUN_TYPES, customerPerformanceReport } from "./model-report.js";
import { config, shows, DEFAULT_BRANCH, branchSelectionEnabled, state, elements, categoryFilters } from "./state.js";

// The registry answers 429 while all of its readers are busy, so those requests wait and try again.
const BUSY_RETRY_DELAYS_MS = [500, 1000, 2000];

export async function api(path) {
  let response;
  for (const delay of [...BUSY_RETRY_DELAYS_MS, null]) {
    response = await fetch(`${state.apiBase}${path}`, {
      headers: { Accept: "application/json" },
      mode: "cors",
    });
    if (response.status !== 429 || delay === null) break;
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error?.message || payload.message || `Registry request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

// Keep resolved reports by stage and fingerprint so Table, Compare, and model pages share their requests.
const benchmarkRuns = new Map();

function loadBenchmarkRun(stage, fingerprint) {
  if (!fingerprint) return null;
  const key = `${stage}:${fingerprint}`;
  if (!benchmarkRuns.has(key)) {
    const pending = fetchBenchmarkRun(stage, fingerprint).then((result) => {
      if (!result) benchmarkRuns.delete(key);
      return result;
    }, (error) => {
      benchmarkRuns.delete(key);
      throw error;
    });
    benchmarkRuns.set(key, pending);
  }
  return benchmarkRuns.get(key);
}

async function fetchBenchmarkRun(stage, benchmarkFingerprint) {
  const repo = state.repository;
  const query = new URLSearchParams({ run_type: BENCHMARK_RUN_TYPES[stage], fingerprint: benchmarkFingerprint });
  let resolved;
  try {
    ({ run: resolved } = await api(`/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/resolve?${query}`));
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
  // The resolved run omits its workflow; the full run gives the result both commit and workflow.
  const [summary, provenance, tests] = await Promise.all([
    api(`/v1/runs/${resolved.id}`).catch(() => null),
    api(`/v1/runs/${resolved.id}/provenance`),
    api(`/v1/runs/${resolved.id}/test-results`),
  ]);
  const report = (tests.test_results || []).find((item) => item.suite === stage && item.name === "model")?.summary || null;
  return {
    run: summary?.run || resolved,
    parameters: provenance.parameters?.parameters || {},
    report: report && !shows.boardMemory && stage === "performance" ? customerPerformanceReport(report) : report,
  };
}

// The catalog records each build profile's benchmark fingerprints on the model. A build missing from the
// current catalog, as in history mode, takes them from the catalog at the build's revision; a catalog
// that does not list the build has none for it.
async function benchmarkFingerprints(runs, profile) {
  const lists = (model) => runs.every((run) => model?.builds?.some((build) => build.run_id === run.id));
  const current = state.catalogMode === "runs" ? null : catalogModelForRun(runs[0]);
  const model = lists(current) ? current : await loadHistoricalCatalogModel(runs[0]);
  return lists(model) ? model.benchmark_fingerprints?.[profile] || {} : {};
}

export async function apiText(path) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { Accept: "text/markdown" },
    mode: "cors",
  });
  const body = await response.text();
  if (!response.ok) {
    let message = `Registry request failed (${response.status})`;
    try { message = JSON.parse(body).error?.message || message; } catch { /* The error body may be plain text. */ }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return body;
}

// The catalog card and the model card show the hero image published with the model README.
export function documentationHeroUrl(documentation) {
  return heroImageSources(documentation, state.apiBase, config.heroCdnBaseUrl)?.src || null;
}

export function documentationHeroFallbackUrl(documentation) {
  return heroImageSources(documentation, state.apiBase)?.fallback || "";
}

function catalogCacheKey() {
  const repo = state.repository;
  const history = elements["history-filter"].checked ? "history" : "latest";
  const branch = (state.preferredBranch ?? elements["branch-filter"].value) || "all";
  // Search, categories, and compiler filter a branch snapshot locally.
  const branchSnapshot = history === "latest" && branch !== "all";
  const categories = branchSnapshot ? [] : Object.entries(categoryFilters()).filter(([, value]) => value).sort();
  const search = branchSnapshot ? "" : elements.search.value.trim();
  const compiler = branchSnapshot ? "any" : elements["compiler-filter"].value || "any";
  return `sima-neat-model-catalog:v6:${state.apiBase}:${repo.organization}/${repo.name}:${history}:${branch}:${compiler}:${search}:${JSON.stringify(categories)}`;
}

function branchCatalogCacheKey() {
  const repo = state.repository;
  return `sima-neat-model-branches:${state.apiBase}:${repo.organization}/${repo.name}`;
}

function branchPreferenceKey() {
  const repo = state.repository;
  return `sima-neat-model-branch:${state.apiBase}:${repo.organization}/${repo.name}`;
}

export function readBranchPreference() {
  try {
    const stored = localStorage.getItem(branchPreferenceKey());
    return stored === null ? DEFAULT_BRANCH : stored;
  } catch {
    return DEFAULT_BRANCH;
  }
}

export function writeBranchPreference(branch) {
  if (!branchSelectionEnabled) return;
  try {
    localStorage.setItem(branchPreferenceKey(), branch);
  } catch {
    // Storage may be disabled; branch selection still works for this page load.
  }
}

export function readCachedBranches() {
  try {
    const cached = JSON.parse(localStorage.getItem(branchCatalogCacheKey()));
    return cached && Array.isArray(cached.branches) ? cached.branches : null;
  } catch {
    return null;
  }
}

export function writeCachedBranches() {
  try {
    localStorage.setItem(branchCatalogCacheKey(), JSON.stringify({
      savedAt: new Date().toISOString(),
      branches: state.availableBranches,
    }));
  } catch {
    // Storage may be disabled; branch discovery still works for this page load.
  }
}

export function readCachedCatalog() {
  try {
    const cached = JSON.parse(localStorage.getItem(catalogCacheKey()));
    return cached && ["models", "runs"].includes(cached.mode) ? cached : null;
  } catch {
    return null;
  }
}

export function writeCachedCatalog() {
  try {
    const payload = state.catalogMode === "models"
      ? { catalogDefinitions: state.catalogDefinitions }
      : { runs: state.runs, cursor: state.cursor, hasMore: state.hasMore };
    localStorage.setItem(catalogCacheKey(), JSON.stringify({
      mode: state.catalogMode,
      taxonomy: state.taxonomy,
      ...payload,
    }));
  } catch {
    // Storage may be disabled; the live registry remains fully functional.
  }
}

// The "model" component, else the first, opens a variant page.
export function anchorBuild(builds) {
  return builds.find((item) => item.component_id === "model") || builds[0];
}

function modelCategory(model, name, fallback) {
  return model[name] ?? fallback;
}

// Categories come from the current catalog: a reused build keeps the ones it was compiled with.
function catalogCategories(model) {
  const categories = model.categories || {};
  const architecture = categories.architecture || {};
  return {
    model_type: modelCategory(model, "model_type", categories.model_type),
    architecture_id: modelCategory(model, "architecture_id", architecture.id),
    architecture_paradigm: modelCategory(model, "architecture_paradigm", architecture.paradigm),
    tasks: modelCategory(model, "tasks", categories.tasks) || [],
    input_modalities: modelCategory(model, "input_modalities", categories.modalities?.inputs ?? categories.input_modalities) || [],
    output_modalities: modelCategory(model, "output_modalities", categories.modalities?.outputs ?? categories.output_modalities) || [],
    model_tags: modelCategory(model, "model_tags", categories.tags) || [],
    model_categories: categories,
  };
}

// The snapshot holds pending builds at the document level and categories under each model.
export function snapshotModels(document) {
  const pendingByModel = new Map();
  for (const build of document.pending_builds || []) {
    if (!pendingByModel.has(build.model_id)) pendingByModel.set(build.model_id, []);
    pendingByModel.get(build.model_id).push(build);
  }
  return document.models.map((model) => ({
    ...model,
    ...catalogCategories(model),
    model_id: model.id,
    pending_builds: pendingByModel.get(model.id) || [],
  }));
}

// branchModels gives every shown build the compile run that produced it.
export function runFromCatalogBuild(model, build) {
  return { ...build.run, ...catalogCategories(model) };
}

export function profileGroups(model) {
  const groups = new Map();
  model.builds.forEach((build) => {
    const profile = build.build_profile_id;
    if (!groups.has(profile)) groups.set(profile, []);
    groups.get(profile).push(build);
  });
  // W8A8 is the default variant, so a profile with a W8A8 component is listed first and opens first.
  const rank = (profile) => (profile.split("_").includes("w8a8") ? 0 : 1);
  return [...groups.entries()].sort(([left], [right]) => rank(left) - rank(right) || left.localeCompare(right));
}

export function activeModelBuild(model) {
  const profiles = profileGroups(model);
  if (!profiles.length) return { profile: null, builds: [], build: null };
  const requestedProfile = state.activeProfileByModel.get(model.model_id);
  const [profile, builds] = profiles.find(([id]) => id === requestedProfile) || profiles[0];
  state.activeProfileByModel.set(model.model_id, profile);
  return { profile, builds, build: anchorBuild(builds) };
}

export async function mapWithConcurrency(items, limit, mapper) {
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

export async function loadRunDetail(runId) {
  if (state.detailCache.has(runId)) return state.detailCache.get(runId);
  const [summary, provenance, tests] = await Promise.all([
    api(`/v1/runs/${runId}`), api(`/v1/runs/${runId}/provenance`), api(`/v1/runs/${runId}/test-results`),
  ]);
  const detail = { summary, provenance, tests };
  state.detailCache.set(runId, detail);
  return detail;
}

// The current catalog lists the latest build of each variant; any other build's components are found
// among the branch's compile runs.
async function variantRuns(run) {
  const model = state.models.find((item) => item.model_id === run.metadata?.model_id);
  const build = model?.builds.find((item) => item.run_id === run.id);
  if (!build) return variantBuildRuns(run, api);
  return model.builds.filter((item) => item.build_profile_id === build.build_profile_id)
    .map((item) => runFromCatalogBuild(model, item));
}

export function variantPackage(runs) {
  return runs.map(fullPackageForRun).find(Boolean) || null;
}

// The order each multi-component model runs in, generated from model.yaml when the portal is built.
// Without it the model page shows its components as before, so a failed fetch is not an error.
let componentPipelines = null;
function loadComponentPipeline(modelId) {
  componentPipelines ||= fetch(new URL("./component-pipelines.json", import.meta.url))
    .then((response) => (response.ok ? response.json() : {}))
    .catch(() => ({}));
  return componentPipelines.then((pipelines) => pipelines[modelId] || null);
}

// The scores each source model's authors published, generated from model.yaml when the portal is built.
// Without it the accuracy section shows measurements alone, so a failed fetch is not an error.
let referenceMetrics = null;
function loadReferenceMetrics(modelId) {
  referenceMetrics ||= fetch(new URL("./reference-metrics.json", import.meta.url))
    .then((response) => (response.ok ? response.json() : {}))
    .catch(() => ({}));
  return referenceMetrics.then((reference) => reference[modelId] || []);
}

// The name of each dataset by its Dataset Registry ID, from datasets/<id>.yaml when the portal is built.
// Without the list the page names a dataset by its ID, so a failed fetch is not an error.
let datasetNames = null;
function loadDatasetNames() {
  datasetNames ||= fetch(new URL("./dataset-names.json", import.meta.url))
    .then((response) => (response.ok ? response.json() : {}))
    .catch(() => ({}));
  return datasetNames;
}

// The Apps portal page of an example application that runs each model, from `example_app` in model.yaml when
// the portal is built. Without it the model page shows no application link, so a failed fetch is not an error.
let exampleApps = null;
function loadExampleApp(modelId) {
  exampleApps ||= fetch(new URL("./example-apps.json", import.meta.url))
    .then((response) => (response.ok ? response.json() : {}))
    .catch(() => ({}));
  return exampleApps.then((apps) => apps[modelId] || null);
}

// The licenses a model may declare, from licenses.json beside this module. Without the list the page
// cannot tell whether a license permits redistribution, so it offers no download.
let licenseList = null;
function loadLicenses() {
  licenseList ||= fetch(new URL("./licenses.json", import.meta.url))
    .then((response) => (response.ok ? response.json() : null))
    .catch(() => null);
  return licenseList;
}

// The package run's summary and test results: its artifacts and its numerical fidelity.
export function loadPackageRun(runId) {
  return Promise.all([
    api(`/v1/runs/${runId}`).catch(() => null),
    api(`/v1/runs/${runId}/test-results`).catch(() => null),
  ]);
}

// A variant's performance and accuracy each measure the whole model, once per build profile.
// What Compare reads of a build profile: its two benchmark results and the model's published scores. It skips
// the compile and package runs a model page loads, which Compare never shows.
export async function loadBenchmarkResults(model, profile) {
  const fingerprints = model.benchmark_fingerprints?.[profile] || {};
  const [performance, accuracy, reference] = await Promise.all([
    loadBenchmarkRun("performance", fingerprints.performance),
    loadBenchmarkRun("accuracy", fingerprints.accuracy),
    loadReferenceMetrics(model.model_id),
  ]);
  return { performance, accuracy, reference };
}

export async function loadVariant(anchor) {
  const key = variantBuildKey(anchor);
  if (state.variantCache.has(key)) return state.variantCache.get(key);
  const componentRuns = await variantRuns(anchor);
  const details = await mapWithConcurrency(componentRuns, 6, (run) => loadRunDetail(run.id));
  const runs = details.map((detail) => detail.summary.run);
  const packageRun = variantPackage(componentRuns);
  const fingerprints = await benchmarkFingerprints(runs, runProfile(anchor));
  const [performance, accuracy, [packageSummary, packageTests], pipeline, reference, licenses, names, exampleApp] = await Promise.all([
    loadBenchmarkRun("performance", fingerprints.performance),
    loadBenchmarkRun("accuracy", fingerprints.accuracy),
    packageRun ? loadPackageRun(packageRun.run_id) : [null, null],
    loadComponentPipeline(anchor.metadata?.model_id),
    loadReferenceMetrics(anchor.metadata?.model_id),
    loadLicenses(),
    loadDatasetNames(),
    loadExampleApp(anchor.metadata?.model_id),
  ]);
  // readAccuracy names each dataset of the report from this list.
  const variant = {
    key, modelId: anchor.metadata?.model_id, profile: runProfile(anchor), anchor, runs, details, performance,
    accuracy: accuracy && { ...accuracy, datasetNames: names },
    packageSummary, packageTests, requestedPackageId: packageRun?.run_id || null, pipeline, reference, licenses, exampleApp,
  };
  state.variantCache.set(key, variant);
  return variant;
}

export function modelPackArtifacts(artifacts) {
  return (artifacts || []).filter((artifact) => artifact.artifact_type === "model-pack");
}

// The catalog view describes its builds by the current catalog; history describes each build by the
// catalog at its own revision, when that has been loaded.
export function catalogModelForRun(run) {
  const modelId = run.metadata?.model_id;
  const current = state.models.find((model) => model.model_id === modelId) || null;
  if (state.catalogMode !== "runs" && current?.builds?.some((build) => build.run_id === run.id)) return current;
  const key = `${run.git_ref || ""}:${run.commit_sha || ""}:${modelId || ""}`;
  const historical = state.historicalCatalogModels.get(key);
  return historical?.model_id ? historical : current;
}

export async function loadHistoricalCatalogModel(run) {
  const modelId = run.metadata?.model_id;
  if (!modelId || !/^refs\/heads\/.+/.test(run.git_ref || "") || !/^[0-9a-f]{40}$/.test(run.commit_sha || "")) return null;
  const key = `${run.git_ref}:${run.commit_sha}:${modelId}`;
  if (state.historicalCatalogModels.has(key)) return state.historicalCatalogModels.get(key);
  const repo = state.repository;
  try {
    const query = new URLSearchParams({ git_ref: run.git_ref });
    const payload = await api(
      `/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/catalogs/${encodeURIComponent(run.commit_sha)}?${query}`,
    );
    const definition = (payload.catalog?.document?.models || []).find((model) => model.id === modelId);
    const model = definition ? deploymentCatalog({ ...definition, model_id: definition.id }) : null;
    state.historicalCatalogModels.set(key, model);
    return model;
  } catch {
    state.historicalCatalogModels.set(key, null);
    return null;
  }
}

function fullPackageForRun(run) {
  const model = catalogModelForRun(run);
  const profile = run.metadata?.build_profile_id;
  if (!model || !profile) return null;
  return (model.packages || []).find((item) => item.build_profile_id === profile
    && item.component_run_ids?.[run.metadata?.component_id] === run.id) || null;
}

export function compilerFilteredBuilds(model) {
  const compiler = elements["compiler-filter"].value;
  return compiler
    ? model.builds.filter((build) => build.toolchain_version === compiler)
    : model.builds;
}

export function selectedCatalogBuild(modelId, profileId) {
  const model = state.models.find((item) => item.model_id === modelId);
  if (!model) return null;
  const build = anchorBuild(compilerFilteredBuilds(model).filter((build) => build.build_profile_id === profileId));
  if (build) state.activeProfileByModel.set(modelId, profileId);
  return build || null;
}
