// Model catalog and build history: filters, search, and the result list.
import { approvedModel, branchModels, variantBuilds } from "./branch-builds.js";
import {
  DEFAULT_BRANCH, CATEGORY_QUERY_FIELDS, state, elements, categoryFilters, filters, catalogMode, taxonomyLabel,
} from "./state.js";
import { icons, escapeHtml, branchName, shortSha, profileLabel } from "./ui.js";
import { formatDate } from "./metrics.js";
import {
  api, writeBranchPreference, writeCachedBranches, writeCachedCatalog, buildProfileId, componentId,
  runFromCatalogBuild, profileGroups, activeModelBuild, documentationHeroUrl,
} from "./api.js";

// A filter lists only the values the loaded builds use, labeled from the taxonomy.
function filterOptions(section, values, emptyLabel) {
  return `<option value="">${escapeHtml(emptyLabel)}</option>${values
    .map((id) => [id, taxonomyLabel(section, id)])
    .sort((left, right) => left[1].localeCompare(right[1]))
    .map(([id, label]) => `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`)
    .join("")}`;
}

// A filter with fewer than two values to choose from is hidden unless it is set.
function hideFilterWithoutChoice(element) {
  element.closest("label").hidden = element.options.length < 3 && !element.value;
}

export function populateCategoryFilters() {
  if (!state.taxonomy) return;
  const definitions = [
    ["model-type-filter", "model_types", "model_type", "Any type"],
    ["task-filter", "tasks", "tasks", "Any task"],
    ["architecture-filter", "architectures", "architecture_id", "Any family"],
    ["paradigm-filter", "architecture_paradigms", "architecture_paradigm", "Any architecture"],
    ["input-modality-filter", "modalities", "input_modalities", "Any input"],
  ];
  definitions.forEach(([elementId, section, field, emptyLabel]) => {
    const element = elements[elementId];
    const selection = state.categorySelection[elementId] || "";
    const selected = Object.hasOwn(state.taxonomy[section] || {}, selection) ? selection : "";
    const values = [...new Set([selected, ...state.runs.flatMap((run) => run[field] ?? []), ...state.models.flatMap((model) => model[field] ?? [])])].filter(Boolean);
    element.innerHTML = filterOptions(section, values, emptyLabel);
    element.value = selected;
    state.categorySelection[elementId] = selected;
    hideFilterWithoutChoice(element);
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

// The filter rail is always shown on wide screens; on narrow screens the toggle opens it.
export function setFilterDrawer(open) {
  elements["filter-drawer"].dataset.open = String(open);
  elements["filter-toggle"].setAttribute("aria-expanded", String(open));
}

// Filters outside Task and Architecture sit under "More filters", which opens when one of them is set.
function moreFiltersActive() {
  const current = categoryFilters();
  return Boolean(
    current.model_type || current.architecture_paradigm || current.input_modality
    || (elements["branch-filter"].value && elements["branch-filter"].value !== DEFAULT_BRANCH)
    || elements["compiler-filter"].value || elements["history-filter"].checked,
  );
}

export function updateFilterDisclosure() {
  const count = activeFilterCount();
  elements["filter-count"].textContent = String(count);
  elements["filter-count"].hidden = count === 0;
  if (moreFiltersActive()) elements["more-filters"].open = true;
}

export function applyCatalog(payload, { append = false, cache = true } = {}) {
  const models = Array.isArray(payload.models) ? payload.models.map(approvedModel) : [];
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
    return matches && (builds.length || (model.build_on_demand && !model.builds?.length && !current.compiler)) ? [{ ...model, builds }] : [];
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
  hideFilterWithoutChoice(elements["branch-filter"]);
  hideFilterWithoutChoice(elements["compiler-filter"]);
  populateCategoryFilters();
  updateFilterDisclosure();
}

function componentList(runs) {
  const components = runs.map((run) => run.metadata?.component_id || "model");
  return components.length > 1 || components[0] !== "model" ? components.join(" · ") : "";
}

function renderHistory() {
  const builds = variantBuilds(filteredRuns());
  elements["result-count"].textContent = `${builds.length}${state.hasMore ? "+" : ""} build${builds.length === 1 ? "" : "s"}`;
  if (!builds.length) {
    elements["result-list"].innerHTML = `<div class="no-results"><span>0</span><h3>No matching builds</h3><p>Broaden the search or clear the current filters.</p><button type="button" data-action="clear">Reset filters</button></div>`;
    return;
  }

  elements["result-list"].innerHTML = builds.map((group) => {
    const run = group.anchor;
    const model = group.modelId || run.run_type.replaceAll("_", " ");
    const selected = group.key === state.selectedVariantKey || group.runs.some((item) => item.id === state.selectedId);
    const components = componentList(group.runs);
    return `<div class="run-entry">
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

// Size words order a family's models from smallest to largest; other names sort alphabetically.
const MODEL_SIZE_RANK = { nano: 1, tiny: 2, small: 3, medium: 4, large: 5, xlarge: 6 };
// YOLO-style names carry the size as one letter after the version, as in YOLO26n-det to YOLO26x-det.
const COMPACT_SIZE = /^([A-Za-z]+\d+)([nsmlx])(?=[-\s]|$)/;
const COMPACT_SIZE_WORDS = { n: "nano", s: "small", m: "medium", l: "large", x: "xlarge" };

function modelSortKey(model) {
  const name = model.display_name || model.model_id;
  const sizeWord = name.split(/\s+/).find((word) => MODEL_SIZE_RANK[word.toLowerCase()]);
  if (sizeWord) {
    return { family: name.replace(sizeWord, "").replace(/\s+/g, " ").trim(), rank: MODEL_SIZE_RANK[sizeWord.toLowerCase()], name };
  }
  const compact = name.match(COMPACT_SIZE);
  if (compact) return { family: name.replace(compact[0], compact[1]), rank: MODEL_SIZE_RANK[COMPACT_SIZE_WORDS[compact[2]]], name };
  return { family: name, rank: 0, name };
}

export function compareModels(left, right) {
  const a = modelSortKey(left);
  const b = modelSortKey(right);
  return a.family.localeCompare(b.family) || a.rank - b.rank || a.name.localeCompare(b.name);
}

function modelCardImage(model) {
  const heroUrl = documentationHeroUrl(model.documentation);
  return heroUrl
    ? `<span class="model-card-image"><img src="${escapeHtml(heroUrl)}" alt="" loading="lazy" decoding="async" /></span>`
    : `<span class="model-card-image model-card-image-empty" aria-hidden="true"><img src="favicon.svg" alt="" /></span>`;
}

function renderModelCard(model) {
  const profiles = profileGroups(model);
  const active = activeModelBuild(model);
  const run = active.build ? runFromCatalogBuild(model, active.build) : null;
  const displayName = model.display_name || model.model_id;
  const tasks = (model.tasks || []).map((id) => `<span class="model-card-tag">${escapeHtml(taxonomyLabel("tasks", id))}</span>`).join("");
  const variants = run ? `${profiles.length} variant${profiles.length === 1 ? "" : "s"}` : "Built on demand";
  return `<button class="model-card" type="button" data-model-card="${escapeHtml(model.model_id)}" ${run ? `data-run-id="${escapeHtml(run.id)}"` : `data-model-unbuilt="${escapeHtml(model.model_id)}"`}>
    ${modelCardImage(model)}
    <span class="model-card-body">
      <strong>${escapeHtml(displayName)}</strong>
      ${model.description ? `<span class="model-card-description">${escapeHtml(model.description)}</span>` : ""}
      <span class="model-card-tags">${tasks}<span class="model-card-variants">${escapeHtml(variants)}</span></span>
    </span>
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
  elements["result-list"].innerHTML = [...models].sort(compareModels).map(renderModelCard).join("");
}

export function renderRuns() {
  const modelsMode = catalogMode();
  const runTitle = elements["history-filter"].checked ? "Build history" : "Latest builds";
  elements["result-title"].textContent = modelsMode ? "Models" : runTitle;
  elements["load-more"].textContent = modelsMode ? "Load more models" : "Load more builds";
  elements["result-list"].setAttribute("aria-busy", "false");
  elements["result-list"].dataset.layout = modelsMode ? "cards" : "rows";
  elements["load-more"].hidden = !state.hasMore;
  if (modelsMode) renderModels();
  else renderHistory();
}
