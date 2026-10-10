// Model catalog and build history: filters, search, and the result list.
import { branchModels, variantBuilds } from "./branch-builds.js";
import {
  DEFAULT_BRANCH, CATEGORY_QUERY_FIELDS, state, elements, categoryFilters, filters, catalogMode,
} from "./state.js";
import { icons, escapeHtml, branchName, shortSha, profileLabel } from "./ui.js";
import { formatDate } from "./metrics.js";
import {
  api, writeBranchPreference, writeCachedBranches, writeCachedCatalog, buildProfileId, componentId,
  runFromCatalogBuild, profileGroups, activeModelBuild,
} from "./api.js";
import { updateCompareAction, compareCheckbox } from "./compare-view.js";

function taxonomyOptions(section, emptyLabel) {
  const entries = state.taxonomy?.[section] || {};
  return `<option value="">${escapeHtml(emptyLabel)}</option>${Object.entries(entries)
    .sort((left, right) => left[1].display_name.localeCompare(right[1].display_name))
    .map(([id, entry]) => `<option value="${escapeHtml(id)}">${escapeHtml(entry.display_name)}</option>`)
    .join("")}`;
}

export function populateCategoryFilters() {
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

export function setCategoryFiltersDisabled(disabled) {
  Object.keys(CATEGORY_QUERY_FIELDS).forEach((elementId) => {
    elements[elementId].disabled = disabled;
  });
}

export async function loadTaxonomy(branch = elements["branch-filter"].value || DEFAULT_BRANCH) {
  const repo = state.repository;
  const requestId = ++state.taxonomyRequestId;
  const payload = await api(
    `/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/catalogs/current?git_ref=${encodeURIComponent(`refs/heads/${branch}`)}`,
  );
  if (requestId !== state.taxonomyRequestId) return;
  state.taxonomy = payload.catalog?.document?.taxonomy || null;
  if (!state.taxonomy) throw new Error("The registry did not return a model taxonomy.");
  populateCategoryFilters();
  setCategoryFiltersDisabled(false);
}

export function activeFilterCount() {
  return [
    elements["branch-filter"].value && elements["branch-filter"].value !== DEFAULT_BRANCH,
    elements["history-filter"].checked,
    elements["compiler-filter"].value,
    ...Object.values(categoryFilters()),
  ].filter(Boolean).length;
}

export function setFilterDrawer(open) {
  elements["filter-drawer"].hidden = !open;
  elements["filter-toggle"].setAttribute("aria-expanded", String(open));
}

export function updateFilterDisclosure() {
  const count = activeFilterCount();
  elements["filter-count"].textContent = String(count);
  elements["filter-count"].hidden = count === 0;
}

export function applyCatalog(payload, { append = false, cache = true } = {}) {
  const models = Array.isArray(payload.models) ? payload.models : [];
  const runs = Array.isArray(payload.runs) ? payload.runs : [];
  const modelsMode = payload.mode === "models"
    || (payload.mode !== "runs" && Object.hasOwn(payload, "models"));
  state.catalogMode = modelsMode ? "models" : "runs";
  if (modelsMode) {
    if (Array.isArray(payload.compileRuns)) {
      state.compileRuns = payload.compileRuns;
      state.packageRuns = payload.packageRuns || [];
      state.catalogDefinitions = payload.catalogDefinitions || [];
      state.models = branchModels(state.compileRuns, state.packageRuns, state.catalogDefinitions, elements["compiler-filter"].value);
    } else {
      state.models = append ? [...state.models, ...models] : models;
    }
    state.runs = state.models.flatMap((model) => (model.builds || []).map((build) => runFromCatalogBuild(model, build)));
  } else {
    state.models = [];
    state.compileRuns = [];
    state.packageRuns = [];
    state.catalogDefinitions = [];
    state.runs = append ? [...state.runs, ...runs] : runs;
  }
  state.availableBranches = [...new Set([
    ...state.availableBranches,
    ...state.runs.map((run) => branchName(run.git_ref)),
  ])].sort();
  state.cursor = payload.next_cursor ?? payload.cursor ?? null;
  state.hasMore = Boolean(payload.has_more ?? payload.hasMore);
  populateFilters({ persistBranchFallback: cache });
  renderRuns();
  if (cache) {
    writeCachedCatalog();
    writeCachedBranches();
  }
}

export async function loadBranches() {
  const repo = state.repository;
  const payload = await api(
    `/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/branches`,
  );
  if (!Array.isArray(payload.branches)) throw new Error("The registry did not return a branch catalog.");
  const branches = [...new Set(payload.branches.map((branch) => {
    if (typeof branch === "string") return branch;
    const gitRef = String(branch?.git_ref || "");
    return gitRef.startsWith("refs/heads/") ? gitRef.slice("refs/heads/".length) : "";
  }).filter(Boolean))].sort();
  if (!branches.length && state.availableBranches.length) {
    throw new Error("The registry returned an empty branch catalog; keeping saved branches.");
  }
  state.availableBranches = branches;
  populateFilters({ persistBranchFallback: true });
  writeCachedBranches();
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
    run.metadata?.build_profile_id, run.metadata?.component_id,
    run.metadata?.variant_id, run.metadata?.package_id,
    run.git_ref, run.commit_sha, run.run_type,
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

function modelSearchDocument(model) {
  return [
    model.model_id, model.display_name, model.description, model.maturity, model.license,
    model.documentation?.overview_markdown,
    taxonomyLabel("model_types", model.model_type),
    taxonomyLabel("architectures", model.architecture_id),
    taxonomyLabel("architecture_paradigms", model.architecture_paradigm),
    ...(model.tasks || []).flatMap((id) => [id, taxonomyLabel("tasks", id)]),
    ...(model.input_modalities || []).flatMap((id) => [id, taxonomyLabel("modalities", id)]),
    ...(model.output_modalities || []).flatMap((id) => [id, taxonomyLabel("modalities", id)]),
    ...(model.model_tags || []),
    ...(model.builds || []).flatMap((build) => [
      buildProfileId(build), componentId(build), build.toolchain_name,
      build.toolchain_version, build.target_platform,
    ]),
  ].filter(Boolean).join(" ");
}

function filteredModels() {
  const current = filters();
  const terms = normalizedSearchTerms(current.query);
  return state.models.flatMap((model) => {
    const searchableText = normalizedSearchTerms(modelSearchDocument(model)).join(" ");
    const builds = (model.builds || []).filter((build) => (
      !current.compiler || build.toolchain_version === current.compiler
    ));
    const matches = terms.every((term) => searchableText.includes(term))
      && (!current.model_type || model.model_type === current.model_type)
      && (!current.task || (model.tasks || []).includes(current.task))
      && (!current.architecture_id || model.architecture_id === current.architecture_id)
      && (!current.architecture_paradigm || model.architecture_paradigm === current.architecture_paradigm)
      && (!current.input_modality || (model.input_modalities || []).includes(current.input_modality));
    return matches && builds.length ? [{ ...model, builds }] : [];
  });
}

function populateFilters({ persistBranchFallback = false } = {}) {
  const selectedCompiler = elements["compiler-filter"].value;
  const preferredBranch = state.preferredBranch ?? DEFAULT_BRANCH;
  const branches = state.urlBranch && preferredBranch && !state.availableBranches.includes(preferredBranch)
    ? [...state.availableBranches, preferredBranch].sort() : state.availableBranches;
  const compilers = [...new Set([
    selectedCompiler,
    ...state.compileRuns.map((run) => run.toolchain_version),
    ...state.runs.map((run) => run.toolchain_version),
  ].filter(Boolean))].sort().reverse();
  elements["branch-filter"].innerHTML = `<option value="">All branches</option>${branches.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  elements["compiler-filter"].innerHTML = `<option value="">Any version</option>${compilers.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
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

function componentList(runs) {
  const components = runs.map((run) => run.metadata?.component_id || "model");
  return components.length > 1 || components[0] !== "model" ? components.join(" · ") : "";
}

function renderHistory() {
  const builds = variantBuilds(filteredRuns());
  builds.forEach((group) => state.variantGroups.set(group.key, group));
  elements["result-count"].textContent = `${builds.length}${state.hasMore ? "+" : ""} build${builds.length === 1 ? "" : "s"}`;
  if (!builds.length) {
    elements["result-list"].innerHTML = `<div class="no-results"><span>0</span><h3>No matching builds</h3><p>Broaden the search or clear the current filters.</p><button type="button" data-action="clear">Reset filters</button></div>`;
    return;
  }

  elements["result-list"].innerHTML = builds.map((group) => {
    const run = group.anchor;
    const model = group.modelId || run.run_type.replaceAll("_", " ");
    const selected = group.key === state.selectedVariantKey || group.runs.some((item) => item.id === state.selectedId);
    const compared = state.compareSelection.has(group.key);
    if (compared) state.compareSelection.set(group.key, group);
    const components = componentList(group.runs);
    return `<div class="run-entry${compared ? " compared" : ""}">
      <label class="compare-select" title="${compared ? "Remove from" : "Add to"} comparison">
        ${compareCheckbox(group)}
        <span aria-hidden="true">✓</span>
      </label>
      <button class="run-row${selected ? " selected" : ""}" type="button" data-run-id="${run.id}" aria-pressed="${selected}">
      <span class="run-identity"><strong>${escapeHtml(model)}</strong><small>${escapeHtml(profileLabel(run))}</small>${components ? `<small class="run-components">${escapeHtml(components)}</small>` : ""}</span>
      <span class="run-branch"><small>Branch</small>${escapeHtml(branchName(run.git_ref))}</span>
      <span class="run-compiler"><small>Compiler</small>${escapeHtml(run.toolchain_version || "—")}</span>
      <span class="run-date"><small>${escapeHtml(shortSha(run.commit_sha))}</small>${escapeHtml(formatDate(group.time || run.created_at))}</span>
      <span class="row-arrow">${icons.arrow}</span>
      </button>
    </div>`;
  }).join("");
}

function renderModelRow(model) {
  const profiles = profileGroups(model);
  const active = activeModelBuild(model);
  const run = runFromCatalogBuild(model, active.build);
  const selected = model.builds.some((build) => build.run_id === state.selectedId);
  const displayName = model.display_name || model.model_id;
  const profileCount = profiles.length;
  return `<button class="model-row${selected ? " selected" : ""}" type="button" data-model-card="${escapeHtml(model.model_id)}" data-run-id="${run.id}" aria-current="${selected ? "true" : "false"}">
    <span class="model-row-identity"><strong>${escapeHtml(displayName)}</strong>${displayName !== model.model_id ? `<code>${escapeHtml(model.model_id)}</code>` : ""}</span>
    <span class="model-row-count">${profileCount}<small>variant${profileCount === 1 ? "" : "s"}</small></span>
    <span class="row-arrow">${icons.arrow}</span>
  </button>`;
}

function renderModels() {
  const models = filteredModels();
  elements["result-count"].textContent = `${models.length}${state.hasMore ? "+" : ""} model${models.length === 1 ? "" : "s"}`;
  if (!models.length) {
    const noBuilds = !state.compileRuns.length && elements["branch-filter"].value;
    elements["result-list"].innerHTML = noBuilds
      ? `<div class="no-results"><span>0</span><h3>No models on this branch yet</h3><p>Its catalog appears after its first push completes. Select another branch to browse its models.</p></div>`
      : `<div class="no-results"><span>0</span><h3>No matching models</h3><p>Broaden the search or clear the current filters.</p><button type="button" data-action="clear">Reset filters</button></div>`;
    return;
  }
  elements["result-list"].innerHTML = models.map(renderModelRow).join("");
}

export function renderRuns() {
  const modelsMode = catalogMode();
  const runTitle = elements["history-filter"].checked ? "Build history" : "Latest builds";
  elements["result-title"].textContent = modelsMode ? "Models" : runTitle;
  elements["load-more"].textContent = modelsMode ? "Load more models" : "Load more builds";
  elements["result-list"].setAttribute("aria-busy", "false");
  elements["load-more"].hidden = !state.hasMore;
  if (modelsMode) renderModels();
  else renderHistory();
  updateCompareAction();
}
