// Registry API requests, local caches, and catalog build selection.
import { runProfile, variantBuildKey, variantBuildRuns } from "./branch-builds.js";
import { DEFAULT_BRANCH, state, elements, categoryFilters } from "./state.js";

export async function api(path) {
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

// DevKit accuracy is its own registry run, keyed by the accuracy key the catalog records on the
// build that reports the result.
async function devkitAccuracy(accuracyKey) {
  if (!accuracyKey) return null;
  const repo = state.repository;
  const query = new URLSearchParams({ run_type: "model_accuracy", fingerprint: accuracyKey, fingerprint_version: "1" });
  try {
    const { run } = await api(`/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/resolve?${query}`);
    // The resolved run omits its workflow; the full run gives the Accuracy section both commit and workflow.
    const [payload, summary] = await Promise.all([
      api(`/v1/runs/${run.id}/accuracy/latest`), api(`/v1/runs/${run.id}`).catch(() => null),
    ]);
    return { ...payload, run: summary?.run || run };
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
}

// Only the build of the component that reports the result carries the key. A build missing from the
// current catalog, as in history mode, takes it from the catalog at the build's revision.
async function accuracyKeyForRuns(runs) {
  const runIds = new Set(runs.map((run) => run.id));
  const keyIn = (model) => model?.builds?.find((build) => runIds.has(build.run_id) && build.validation_fingerprint)
    ?.validation_fingerprint;
  const current = state.catalogMode === "runs" ? null : catalogModelForRun(runs[0]);
  if (current?.builds?.some((build) => runIds.has(build.run_id))) return keyIn(current);
  return keyIn(await loadHistoricalCatalogModel(runs[0]));
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
  const hero = documentation?.hero;
  if (!hero?.artifact_name || !hero?.artifact_sha256 || !documentation?.run_id) return null;
  return `${state.apiBase}/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(hero.artifact_name)}/content`;
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
  return `sima-neat-model-catalog:v4:${state.apiBase}:${repo.organization}/${repo.name}:${history}:${branch}:${compiler}:${search}:${JSON.stringify(categories)}`;
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
    if (!cached || (!Array.isArray(cached.runs) && !Array.isArray(cached.models))) return null;
    const mode = ["models", "runs"].includes(cached.mode)
      ? cached.mode
      : (cached.models?.length || !cached.runs?.length ? "models" : "runs");
    return { ...cached, mode };
  } catch {
    return null;
  }
}

export function writeCachedCatalog() {
  try {
    localStorage.setItem(catalogCacheKey(), JSON.stringify({
      savedAt: new Date().toISOString(),
      mode: state.catalogMode,
      models: state.models,
      runs: state.runs,
      compileRuns: state.compileRuns,
      packageRuns: state.packageRuns,
      catalogDefinitions: state.catalogDefinitions,
      cursor: state.cursor,
      hasMore: state.hasMore,
      taxonomy: state.taxonomy,
    }));
  } catch {
    // Storage may be disabled; the live registry remains fully functional.
  }
}

export function buildProfileId(build) {
  return build.build_profile_id || build.variant_id || "default";
}

export function componentId(build) {
  return build.component_id || "model";
}

// The "model" component, else the first, opens a variant page.
export function anchorBuild(builds) {
  return builds.find((item) => componentId(item) === "model") || builds[0];
}

function modelCategory(model, name, fallback) {
  return model[name] ?? fallback;
}

export function runFromCatalogBuild(model, build) {
  if (build.run) return build.run;
  const categories = model.categories || {};
  const architecture = categories.architecture || {};
  const branch = elements["branch-filter"].value || DEFAULT_BRANCH;
  return {
    id: build.run_id,
    repository: `${state.repository.organization}/${state.repository.name}`,
    run_type: "model_compile",
    status: build.status || "completed",
    git_ref: model.git_ref || `refs/heads/${branch}`,
    commit_sha: build.commit_sha || model.catalog_revision,
    created_at: build.created_at,
    finished_at: build.finished_at,
    toolchain_name: build.toolchain_name,
    toolchain_version: build.toolchain_version,
    target_platform: build.target_platform,
    metadata: {
      model_id: model.model_id,
      display_name: model.display_name,
      build_profile_id: buildProfileId(build),
      component_id: componentId(build),
      compile_invocation_id: build.compile_invocation_id,
    },
    model_type: modelCategory(model, "model_type", categories.model_type),
    architecture_id: modelCategory(model, "architecture_id", architecture.id),
    architecture_paradigm: modelCategory(model, "architecture_paradigm", architecture.paradigm),
    tasks: modelCategory(model, "tasks", categories.tasks) || [],
    input_modalities: modelCategory(model, "input_modalities", categories.input_modalities) || [],
    output_modalities: modelCategory(model, "output_modalities", categories.output_modalities) || [],
    model_tags: modelCategory(model, "model_tags", categories.tags) || [],
    model_categories: categories,
  };
}

export function profileGroups(model) {
  const groups = new Map();
  model.builds.forEach((build) => {
    const profile = buildProfileId(build);
    if (!groups.has(profile)) groups.set(profile, []);
    groups.get(profile).push(build);
  });
  // INT8 is the default variant, so it is listed first and opens first.
  const rank = (profile) => (profile.includes("int8") ? 0 : 1);
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
  const optional = (request) => request.catch((error) => {
    if (error.status === 404) return null;
    throw error;
  });
  const [summary, provenance, tests, benchmark, accuracy] = await Promise.all([
    api(`/v1/runs/${runId}`), api(`/v1/runs/${runId}/provenance`), api(`/v1/runs/${runId}/test-results`),
    optional(api(`/v1/runs/${runId}/benchmarks/latest`)), optional(api(`/v1/runs/${runId}/accuracy/latest`)),
  ]);
  // Builds measured before accuracy became its own run carry it on the benchmark run. A reused build is
  // measured at a later commit than it was built, so the benchmark run's own commit and workflow are kept.
  const benchmarkRunId = benchmark?.benchmark_run?.id;
  const [benchmarkTests, benchmarkRun] = benchmarkRunId
    ? await Promise.all([
      api(`/v1/runs/${benchmarkRunId}/test-results`).catch(() => null),
      api(`/v1/runs/${benchmarkRunId}`).then((benchmarkSummary) => benchmarkSummary.run).catch(() => null),
    ])
    : [null, null];
  const detail = { summary, provenance, tests, benchmark, benchmarkTests, benchmarkRun, accuracy };
  state.detailCache.set(runId, detail);
  return detail;
}

// The current catalog lists the latest build of each variant; any other build's components are found
// among the branch's compile runs.
async function variantRuns(run) {
  const model = state.models.find((item) => item.model_id === run.metadata?.model_id);
  const build = model?.builds.find((item) => item.run_id === run.id);
  if (!build) return variantBuildRuns(run, api);
  return model.builds.filter((item) => buildProfileId(item) === buildProfileId(build))
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

export async function loadVariant(anchor) {
  const key = variantBuildKey(anchor);
  if (state.variantCache.has(key)) return state.variantCache.get(key);
  const componentRuns = await variantRuns(anchor);
  const details = await mapWithConcurrency(componentRuns, 6, (run) => loadRunDetail(run.id));
  const runs = details.map((detail) => detail.summary.run);
  const packageRun = variantPackage(componentRuns);
  const [devkit, packageSummary, pipeline] = await Promise.all([
    accuracyKeyForRuns(runs).then(devkitAccuracy),
    packageRun ? api(`/v1/runs/${packageRun.run_id}`).catch(() => null) : null,
    loadComponentPipeline(anchor.metadata?.model_id),
  ]);
  const variant = {
    key, modelId: anchor.metadata?.model_id, profile: runProfile(anchor), anchor, runs, details, devkit, packageSummary,
    requestedPackageId: packageRun?.run_id || null, pipeline,
  };
  state.variantCache.set(key, variant);
  return variant;
}

export function modelPackArtifacts(artifacts) {
  return (artifacts || []).filter((artifact) => artifact.artifact_type === "model-pack");
}

export function catalogModelForRun(run) {
  const modelId = run.metadata?.model_id;
  const key = `${run.git_ref || ""}:${run.commit_sha || ""}:${modelId || ""}`;
  const historical = state.historicalCatalogModels.get(key);
  return historical?.model_id ? historical : state.models.find((model) => model.model_id === modelId) || null;
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
    const definition = (payload.catalog?.document?.models || []).find((model) => (model.model_id || model.id) === modelId);
    const model = definition ? { ...definition, model_id: definition.model_id || definition.id } : null;
    state.historicalCatalogModels.set(key, model);
    return model;
  } catch {
    state.historicalCatalogModels.set(key, null);
    return null;
  }
}

function fullPackageForRun(run) {
  const model = catalogModelForRun(run);
  const profile = run.metadata?.build_profile_id || run.metadata?.variant_id;
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
  const build = anchorBuild(compilerFilteredBuilds(model).filter((build) => buildProfileId(build) === profileId));
  if (build) state.activeProfileByModel.set(modelId, profileId);
  return build || null;
}
