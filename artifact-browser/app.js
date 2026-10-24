import { installHeroImageRecovery } from "./hero-images.js";
import { variantDownloads, prepareDownloads, startDownloads, packageFormat, artifactDelivery, deploymentSize, huggingFaceLocation, modelDownloadCommand } from "./downloads.js";
import { mountDeveloperPortal } from "./portal-shell.js";
import {
  allPages, branchModels, catalogBuildRuns, runProfile, variantBuildKey, variantBuilds,
} from "./branch-builds.js";
import {
  config, customerView, branchSelectionEnabled, DEFAULT_BRANCH, CATEGORY_QUERY_FIELDS, state, elements,
  categoryFilters, catalogMode, branchSnapshotMode, currentVariant, showView,
} from "./state.js";
import { icons, escapeHtml, branchName, showToast, profileLabel, variantLabel } from "./ui.js";
import { formatBytes } from "./metrics.js";
import {
  api, readBranchPreference, writeBranchPreference, readCachedBranches, readCachedCatalog,
  writeCachedCatalog, runFromCatalogBuild, loadRunDetail, variantPackage, loadVariant, loadPackageRun,
  loadHistoricalCatalogModel, selectedCatalogBuild,
} from "./api.js";
import {
  populateCategoryFilters, setCategoryFiltersDisabled, loadTaxonomy, activeFilterCount,
  setFilterDrawer, updateFilterIndicators, applyCatalog, loadBranches, renderRuns,
} from "./catalog-view.js";
import { renderComparison, picksFromUrl, writePicksToUrl, pickKey, toggleComparison, updateComparisonChart, showComparisonTooltip, hideComparisonTooltip } from "./compare-view.js";
import {
  detailSkeleton, showEmptyDetail, renderDetail, errorPanel, updateLayerAnalysis, selectExecutionGraph, highlightLayerLegend,
  showExecutionTooltip, hideExecutionTooltip, keepExecutionTooltip, deferHideExecutionTooltip,
} from "./model-view.js";

installHeroImageRecovery(document);

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

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("sima-neat-artifact-theme", theme);
    localStorage.setItem("theme", theme);
    localStorage.setItem("portal-theme", theme);
  } catch {
    // Storage may be disabled; theme selection still works for this page load.
  }
  document.cookie = `sima-neat-theme=${theme}; Path=/; Max-Age=31536000; SameSite=Lax${location.protocol === "https:" ? "; Secure" : ""}`;
  const dark = theme === "dark";
  elements["theme-toggle"].innerHTML = dark ? icons.sun : icons.moon;
  elements["theme-toggle"].setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} mode`);
  elements["theme-toggle"].title = `Switch to ${dark ? "light" : "dark"} mode`;
}

// The page a link opens: a model page (`run`, or `model` and `variant`), the Compare page (`view` and
// `build`), or the catalog when none is set.
function deletePageParams(url) {
  ["run", "model", "variant", "view", "build", "compare_task", "compare_precision"].forEach((name) => url.searchParams.delete(name));
}

function writeVariantUrl(run) {
  const url = new URL(location.href);
  const modelId = run.metadata?.model_id;
  const profile = run.metadata?.build_profile_id;
  deletePageParams(url);
  if (modelId && profile) {
    url.searchParams.set("model", modelId);
    url.searchParams.set("variant", profile);
  }
  // The latest build is addressed by its variant; any other build is pinned by one of its runs.
  if (!catalogMode() || !modelId || !profile) url.searchParams.set("run", run.id);
  // An explicit empty branch preserves the All branches view on refresh.
  url.searchParams.set("branch", elements["branch-filter"].value);
  history.pushState({}, "", url);
}

// Opens the variant build that `runId` belongs to; `expand` opens that run's component row. Rows the
// reader opened stay open across the variants of one model.
async function selectRun(runId, { updateUrl = true, expand = false } = {}) {
  const catalogRun = state.runs.find((run) => run.id === runId);
  const previousModelId = currentVariant()?.modelId;
  if (previousModelId !== catalogRun?.metadata?.model_id) {
    state.openDisclosures = new Set();
    state.layerViews.clear();
    state.documentationRequestId += 1;
  }
  state.selectedId = runId;
  if (expand) state.openDisclosures.add(`component:${runId}`);
  if (catalogRun?.metadata?.model_id) state.activeProfileByModel.set(catalogRun.metadata.model_id, runProfile(catalogRun));
  renderRuns();
  detailSkeleton();
  if (updateUrl && catalogRun) writeVariantUrl(catalogRun);
  try {
    const anchor = (await loadRunDetail(runId)).summary.run;
    const cached = state.variantCache.has(variantBuildKey(anchor));
    const variant = await loadVariant(anchor);
    const currentPackage = variantPackage(variant.runs);
    // A cached variant may gain package context later; avoid retrying a fetch from this selection.
    if (currentPackage && variant.packageSummary?.run?.id !== currentPackage.run_id
      && (cached || currentPackage.run_id !== variant.requestedPackageId)) {
      [variant.packageSummary, variant.packageTests] = await loadPackageRun(currentPackage.run_id);
    }
    if (state.catalogMode === "runs") await loadHistoricalCatalogModel(anchor);
    if (state.selectedId === runId) {
      state.selectedVariantKey = variant.key;
      renderRuns();
      const linkedBranch = new URLSearchParams(location.search).get("branch");
      // A branch lists the catalog builds it inherits, so only an unlisted run from another branch mismatches.
      const listed = state.runs.some((run) => run.id === anchor.id);
      if (linkedBranch && branchName(anchor.git_ref) !== linkedBranch && !listed) {
        elements["detail-pane"].innerHTML = errorPanel("This run belongs to a different branch than the link specifies.", "Branch and run do not match", { closable: true });
      } else renderDetail(variant);
    }
  } catch (error) {
    if (state.selectedId === runId) {
      elements["detail-pane"].innerHTML = errorPanel(error.message, "Could not load this build", { closable: true });
    }
  }
}

let pendingDownloads = null;
let downloadRequest = 0;
let pendingCommand = null;

function resetDownloadDialog({ title, description, confirmLabel }) {
  pendingDownloads = null;
  pendingCommand = null;
  elements["download-copy-cli"].hidden = true;
  elements["download-title"].textContent = title;
  elements["download-description"].textContent = description;
  elements["download-list"].innerHTML = "";
  elements["download-list"].hidden = false;
  elements["download-package-info"].hidden = true;
  elements["download-package-info"].classList.remove("download-package-summary");
  elements["download-package-info"].innerHTML = "";
  elements["download-dialog"].querySelector(".download-note").hidden = true;
  elements["download-status"].textContent = "";
  elements["download-confirm"].disabled = true;
  elements["download-confirm"].hidden = false;
  elements["download-confirm"].textContent = confirmLabel;
}

function catalogDownloadCommand(run) {
  const current = state.models.some((model) => model.model_id === run.metadata?.model_id
    && (model.builds?.some((build) => build.run_id === run.id)
      || model.packages?.some((item) => item.run_id === run.id)));
  return current ? modelDownloadCommand(run.metadata.model_id, runProfile(run),
    elements["branch-filter"].value || branchName(run.git_ref), config.environment?.toLowerCase() === "staging") : null;
}

function showFolderDownload(run, downloads) {
  const dialog = elements["download-dialog"];
  pendingCommand = catalogDownloadCommand(run);
  elements["download-title"].textContent = "Download model folder";
  elements["download-description"].textContent = "This model is provided as a folder of individual files, similar to a Hugging Face repository, rather than a ZIP archive. Use sima-cli 2.1.19 or later to download the files in parallel and assemble the model folder.";
  elements["download-list"].hidden = true;
  elements["download-package-info"].hidden = false;
  elements["download-package-info"].innerHTML = downloads.map((item) => {
    const location = item.delivery === "huggingface" ? huggingFaceLocation(item.artifact) : null;
    return `<dt>Deployment</dt><dd>${escapeHtml(item.component)}</dd>
      <dt>Files</dt><dd>${escapeHtml(item.artifact.metadata?.file_count ?? "Not recorded")}</dd>
      <dt>Download size</dt><dd>${escapeHtml(formatBytes(item.size))}</dd>
      <dt>Storage</dt><dd>${item.delivery === "huggingface" ? "Hugging Face" : item.delivery === "archive" ? "Archive" : "Registry folder"}</dd>
      ${location ? `<dt>Repository</dt><dd><a href="${escapeHtml(location.url)}" target="_blank" rel="noreferrer">${escapeHtml(location.repo_id)} ↗</a></dd><dt>Revision</dt><dd><code>${escapeHtml(location.revision)}</code></dd>${location.subdirectory ? `<dt>Folder</dt><dd>${escapeHtml(location.subdirectory)}</dd>` : ""}` : item.delivery === "huggingface" ? `<dt>Repository</dt><dd>Pinned repository location unavailable. Refresh the catalog to retry.</dd>` : ""}`;
  }).join("") + (pendingCommand ? `<dt>Command</dt><dd><code>${escapeHtml(pendingCommand)}</code></dd>`
    : `<dt>Command</dt><dd>Open the current catalog variant for its verified CLI download command.</dd>`);
  const note = dialog.querySelector(".download-note");
  note.hidden = false;
  note.textContent = downloads.some((item) => item.delivery === "huggingface")
    ? "The repository link opens the compiled deployment at its pinned revision. Use that revision and folder when downloading from Hugging Face."
    : "The CLI uses the manifest to preserve the folder structure and verify each file’s checksum.";
  elements["download-confirm"].innerHTML = `${icons.copy}<span>Copy CLI command</span>`;
  elements["download-confirm"].hidden = !pendingCommand;
  elements["download-confirm"].disabled = false;
}

async function downloadVariant(runId) {
  const request = ++downloadRequest;
  const catalogModels = elements["history-filter"].checked ? [] : state.models;
  const dialog = elements["download-dialog"];
  resetDownloadDialog({
    title: "Download components",
    description: "Checking the selected variant’s components…",
    confirmLabel: "Download all files",
  });
  dialog.showModal();
  try {
    const summary = await api(`/v1/runs/${encodeURIComponent(runId)}`);
    const model = catalogModels.find((model) => model.model_id === summary.run.metadata?.model_id);
    const downloads = await prepareDownloads(await variantDownloads(summary.run, api, model), api);
    if (request !== downloadRequest || !dialog.open) return;
    if (downloads.some((item) => item.delivery !== "archive" && item.delivery)) {
      showFolderDownload(summary.run, downloads);
      return;
    }
    pendingDownloads = downloads;
    pendingCommand = catalogDownloadCommand(summary.run);
    elements["download-copy-cli"].hidden = !pendingCommand;
    const profile = profileLabel(summary.run);
    const fileLabel = downloads.length === 1 ? "file" : "files";
    elements["download-description"].textContent = `${profile} contains ${downloads.length} model ${fileLabel}. ${downloads.length === 1 ? "Confirm to download this variant." : "Download all components to use this variant."}`;
    const note = elements["download-dialog"].querySelector(".download-note");
    note.hidden = downloads.length === 1;
    note.textContent = "Files download separately and in parallel. Your browser may ask you to allow multiple downloads.";
    elements["download-list"].innerHTML = downloads.map((item) => `<li><div><strong>${escapeHtml(item.component)}</strong><span>${escapeHtml(item.filename)}</span></div><span>${formatBytes(item.size)}</span></li>`).join("");
    elements["download-confirm"].textContent = `Download ${downloads.length} ${fileLabel}`;
    elements["download-confirm"].disabled = false;
  } catch (error) {
    if (request === downloadRequest && dialog.open) elements["download-status"].textContent = error.message;
  }
}

async function downloadPackage(runId) {
  const request = ++downloadRequest;
  const dialog = elements["download-dialog"];
  resetDownloadDialog({
    title: "Download model package",
    description: "Checking the package manifest and download metadata…",
    confirmLabel: "Download",
  });
  dialog.showModal();
  try {
    const summary = await api(`/v1/runs/${encodeURIComponent(runId)}`);
    const run = summary.run;
    const artifact = (summary.artifacts || []).find((item) => item.artifact_type === "model-package");
    if (!run || !artifact) throw new Error("No complete package is available for this build.");
    const metadata = run.metadata || {};
    const packageId = metadata.package_id;
    const profile = variantLabel(metadata.build_profile_id);
    const components = Object.keys(metadata.component_run_ids);
    // The registry serves the package under its published filename: the MPK itself, or a ZIP.
    const downloads = await prepareDownloads([{
      runId: run.id,
      component: "Full package",
      name: artifact.name,
      size: deploymentSize(artifact), artifact, delivery: artifactDelivery(artifact),
    }], api);
    const filename = downloads[0].filename;
    const format = packageFormat(artifact, filename);
    if (request !== downloadRequest || !dialog.open) return;
    if (downloads.some((item) => item.delivery !== "archive" && item.delivery)) {
      showFolderDownload(run, downloads);
      return;
    }
    pendingDownloads = downloads;
    pendingCommand = catalogDownloadCommand(run);
    elements["download-copy-cli"].hidden = !pendingCommand;
    const license = metadata.license || {};
    const redistribution = license.redistribution
      ? `${license.spdx || "License"} · redistribution ${license.redistribution}`
      : license.spdx || "Not recorded";
    const componentLabel = `${components.length} ${components.length === 1 ? "component" : "components"}`;
    const modelName = state.models.find((model) => model.model_id === metadata.model_id)?.display_name
      || metadata.model_id || "model";
    elements["download-title"].textContent = `Download ${modelName}`;
    elements["download-description"].textContent = `${profile} · ${formatBytes(deploymentSize(artifact))}`;
    elements["download-package-info"].hidden = false;
    elements["download-package-info"].classList.add("download-package-summary");
    elements["download-package-info"].innerHTML = `
      <div class="download-filename"><dt>File name</dt><dd><code>${escapeHtml(filename)}</code></dd></div>
      ${customerView ? "" : `<div class="download-technical-fact"><dt>Package</dt><dd>${escapeHtml(packageId)} · ${escapeHtml(componentLabel)} · ${escapeHtml(format)}</dd></div>
        <div class="download-technical-fact"><dt>SHA-256</dt><dd><code>${escapeHtml(artifact.sha256 || "Not recorded")}</code></dd></div>`}
      <div class="download-license"><dt>License</dt><dd>${escapeHtml(redistribution)}</dd></div>
      ${!customerView && license.attribution ? `<div class="download-technical-fact"><dt>Attribution</dt><dd>${escapeHtml(license.attribution)}</dd></div>` : ""}`;
    const note = elements["download-dialog"].querySelector(".download-note");
    note.hidden = true;
    note.textContent = "";
    elements["download-list"].hidden = true;
    elements["download-confirm"].textContent = "Download";
    elements["download-confirm"].disabled = false;
  } catch (error) {
    if (request === downloadRequest && dialog.open) elements["download-status"].textContent = error.message;
  }
}

elements["download-copy-cli"].innerHTML = `${icons.copy}<span>Copy CLI command</span>`;
elements["download-cancel"].addEventListener("click", () => elements["download-dialog"].close());
elements["download-copy-cli"].addEventListener("click", () => {
  if (pendingCommand) copyText(pendingCommand);
});
elements["download-dialog"].addEventListener("close", () => {
  downloadRequest += 1;
  pendingDownloads = null;
  pendingCommand = null;
});
elements["download-confirm"].addEventListener("click", async () => {
  if (!pendingDownloads && pendingCommand) { await copyText(pendingCommand); return; }
  if (!pendingDownloads) return;
  const request = downloadRequest;
  elements["download-confirm"].disabled = true;
  elements["download-status"].textContent = "Preparing all downloads…";
  try {
    const ready = await prepareDownloads(pendingDownloads, api);
    if (request !== downloadRequest || !elements["download-dialog"].open) return;
    startDownloads(ready, document);
    elements["download-dialog"].close();
    showToast(ready.length === 1 ? "Download requested." : `${ready.length} downloads requested. If prompted, allow multiple downloads in your browser.`);
  } catch (error) {
    if (request === downloadRequest && elements["download-dialog"].open) {
      elements["download-status"].textContent = error.message;
      elements["download-confirm"].disabled = false;
    }
  }
});

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
  const historical = elements["history-filter"].checked || !branch;
  query.set("status", "completed");
  query.set("run_type", "model_compile");
  if (historical && !elements["history-filter"].checked) query.set("latest_per_model_compiler", "true");
  if (branch) query.set("git_ref", `refs/heads/${branch}`);
  if (historical) {
    if (elements.search.value.trim()) query.set("q", elements.search.value.trim());
    Object.entries(categoryFilters()).forEach(([name, value]) => {
      if (value) query.set(name, value);
    });
  }
  if (append && state.cursor) query.set("cursor", state.cursor);

  try {
    const base = `/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}`;
    let payload;
    if (historical) {
      payload = await api(`${base}/runs?${query}`);
    } else {
      const compileQuery = new URLSearchParams(query);
      compileQuery.set("limit", "100");
      const packageQuery = new URLSearchParams({
        limit: "100", status: "completed", run_type: "model_package", git_ref: `refs/heads/${branch}`,
      });
      const modelQuery = new URLSearchParams({ limit: "100", git_ref: `refs/heads/${branch}` });
      const [branchRuns, packageRuns, catalogDefinitions] = await Promise.all([
        allPages(api, `${base}/runs`, compileQuery, "runs"),
        allPages(api, `${base}/runs`, packageQuery, "runs"),
        // A branch has no catalog until its first orchestrator run publishes one.
        allPages(api, `${base}/models`, modelQuery, "models").catch((error) => {
          if (error.status === 404) return [];
          throw error;
        }),
      ]);
      const gitRef = `refs/heads/${branch}`;
      const ownRuns = branchRuns.filter((run) => run.git_ref === gitRef);
      const inheritedRuns = await catalogBuildRuns(catalogDefinitions, ownRuns, api, base);
      payload = {
        mode: "models",
        compileRuns: [...ownRuns, ...inheritedRuns],
        packageRuns: packageRuns.filter((run) => run.git_ref === gitRef),
        catalogDefinitions,
      };
    }
    if (requestId !== state.runsRequestId) return;
    applyCatalog(payload, { append });
  } catch (error) {
    if (requestId === state.runsRequestId) throw error;
  } finally {
    if (requestId === state.runsRequestId) elements["load-more"].disabled = false;
  }

}

async function initialize() {
  const navigationId = ++state.navigationId;
  state.runsRequestId += 1;
  setTheme(document.documentElement.dataset.theme || "light");
  const link = new URLSearchParams(location.search);
  elements.search.value = link.get("q") || "";
  elements["history-filter"].checked = !customerView && link.get("history") === "1";
  state.selectedId = customerView ? null : link.get("run");
  if (customerView) elements["compiler-filter"].value = "";
  const compareLink = link.get("view") === "compare";
  // Other pages keep the picks, so returning to Compare finds them.
  if (compareLink) state.comparePicks = picksFromUrl(link);
  if (compareLink) {
    state.compareTask = link.get("compare_task") || "";
  }
  const linkedVariant = link.get("model") && link.get("variant")
    ? { modelId: link.get("model"), profile: link.get("variant") } : null;
  state.categorySelection = Object.fromEntries(
    Object.entries(CATEGORY_QUERY_FIELDS).map(([elementId, queryName]) => [elementId, link.get(queryName) || ""]),
  );
  state.taxonomy = null;
  if (compareLink) {
    showView("compare");
    elements["compare-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span></div>`;
  } else if (state.selectedId || linkedVariant) detailSkeleton();
  else showEmptyDetail();
  state.urlBranch = !branchSelectionEnabled || link.has("branch");
  state.preferredBranch = !branchSelectionEnabled ? DEFAULT_BRANCH : state.urlBranch ? link.get("branch") : readBranchPreference();
  const releaseNotice = document.getElementById("release-notice");
  if (releaseNotice) {
    releaseNotice.hidden = config.deploymentEnvironment !== "staging";
    document.getElementById("release-backend").textContent = config.environment || "Custom";
    document.getElementById("release-branch").textContent = config.build?.branch || "Source preview";
    document.getElementById("release-commit").textContent = config.build?.commit
      ? `${config.build.commit.slice(0, 12)}${config.build.modified ? " (local changes)" : ""}` : "Build revision unavailable";
  }
  elements["environment-name"].textContent = `${config.environment || "Public"} registry`;
  // Returning to the catalog keeps the cards on screen while they reload, so the restored scroll position holds.
  if (!elements["result-list"].querySelector(".model-card, .run-entry")) {
    elements["result-list"].innerHTML = Array.from({ length: 5 }, () => `<div class="row-skeleton"><span></span><span></span><span></span></div>`).join("");
  }

  const validApiOrigin = /^https:\/\/[^/]+$/.test(state.apiBase)
    || /^http:\/\/(?:127\.0\.0\.1|localhost)(?::[0-9]+)?$/.test(state.apiBase);
  if (!validApiOrigin || !state.repository.organization || !state.repository.name) {
    showEmptyDetail();
    elements["result-list"].innerHTML = errorPanel("Set a valid HTTPS API origin and repository in config.js.", "Portal configuration is invalid");
    elements["result-list"].setAttribute("aria-busy", "false");
    return;
  }

  if (!state.urlBranch && state.selectedId) {
    try {
      const summary = await api(`/v1/runs/${encodeURIComponent(state.selectedId)}`);
      const branch = branchName(summary.run?.git_ref);
      if (branch !== "detached") {
        state.preferredBranch = branch;
        state.urlBranch = true;
        const url = new URL(location.href);
        url.searchParams.set("branch", branch);
        history.replaceState({}, "", url);
      }
    } catch {
      // The detail request below will show the precise error for an invalid run.
    }
  }
  if (navigationId !== state.navigationId) return;

  const cached = readCachedCatalog();
  const cachedBranches = readCachedBranches();
  if (cachedBranches) state.availableBranches = cachedBranches;
  if (cached) {
    if (!cachedBranches && Array.isArray(cached.branches)) state.availableBranches = cached.branches;
    state.taxonomy = cached.taxonomy || null;
    if (state.taxonomy) populateCategoryFilters();
    applyCatalog(cached, { cache: false });
  }

  try {
    if (!branchSelectionEnabled) state.availableBranches = [DEFAULT_BRANCH];
    const branchResult = await Promise.resolve(!branchSelectionEnabled ? undefined : loadBranches()).then(
      () => ({ status: "fulfilled" }),
      (reason) => ({ status: "rejected", reason }),
    );
    if (!branchSelectionEnabled) {
      elements["branch-filter"].innerHTML = `<option value="${DEFAULT_BRANCH}">${DEFAULT_BRANCH}</option>`;
      elements["branch-filter"].value = DEFAULT_BRANCH;
    }
    const taxonomyResult = await Promise.resolve(loadTaxonomy(elements["branch-filter"].value || DEFAULT_BRANCH)).then(
      () => ({ status: "fulfilled" }),
      (reason) => ({ status: "rejected", reason }),
    );
    if (navigationId !== state.navigationId) return;
    if (taxonomyResult.status === "rejected") {
      setCategoryFiltersDisabled(true);
      showToast(`Category filters are temporarily unavailable. ${taxonomyResult.reason.message}`, true);
    }
    if (branchResult.status === "rejected") {
      showToast(`Branch catalog is temporarily unavailable. ${branchResult.reason.message}`, true);
    }
    if (activeFilterCount()) setFilterDrawer(true);
    if (cached || branchResult.status === "fulfilled" || state.availableBranches.length) {
      await loadRuns({ branch: elements["branch-filter"].value });
    } else {
      // Without a branch catalog, infer branches from a broad runs page
      // without making that page authoritative.
      await loadRuns({ branch: "", limit: 200 });
    }
  } catch (error) {
    if (cached) showToast(error.message, true);
    else {
      elements["result-list"].innerHTML = errorPanel(error.message);
      elements["result-list"].setAttribute("aria-busy", "false");
    }
  }

  if (navigationId !== state.navigationId) return;
  if (compareLink) {
    await renderComparison();
    return;
  }
  const linkedRun = Boolean(state.selectedId);
  if (!linkedRun && linkedVariant) {
    state.selectedId = latestVariantRun(linkedVariant)?.id || null;
    if (!state.selectedId) {
      const branch = elements["branch-filter"].value || "any branch";
      elements["detail-pane"].innerHTML = errorPanel(`No completed ${variantLabel(linkedVariant.profile)} build of ${linkedVariant.modelId} was found on ${branch}.`, "Variant not found", { closable: true });
      return;
    }
  }
  if (state.selectedId) await selectRun(state.selectedId, { updateUrl: false, expand: linkedRun });
}

// The latest build of a variant: the current catalog lists only latest builds, and history lists the
// newest runs first.
function latestVariantRun({ modelId, profile }) {
  const matches = state.runs.filter((run) => run.metadata?.model_id === modelId && runProfile(run) === profile);
  if (catalogMode()) {
    const model = state.models.find((item) => item.model_id === modelId);
    const build = model && selectedCatalogBuild(modelId, profile);
    return build ? matches.find((run) => run.id === build.run_id) : null;
  }
  return matches[0] ? variantBuilds(matches)[0].anchor : null;
}

elements["theme-toggle"].addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
window.addEventListener("developer-center-theme-change", (event) => {
  const theme = event.detail?.theme;
  if (theme === "light" || theme === "dark") setTheme(theme);
});
const portalMenu = document.querySelector(".portal-menu-toggle");
portalMenu.addEventListener("click", () => {
  portalMenu.setAttribute("aria-expanded", String(portalMenu.getAttribute("aria-expanded") !== "true"));
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && portalMenu.getAttribute("aria-expanded") === "true") {
    portalMenu.setAttribute("aria-expanded", "false");
    portalMenu.focus();
  }
});
mountDeveloperPortal();
elements["catalog-layout-toggle"].addEventListener("click", (event) => {
  const button = event.target.closest("[data-catalog-layout]");
  if (!button) return;
  state.catalogLayout = button.dataset.catalogLayout;
  try { localStorage.setItem("sima-neat-catalog-layout", state.catalogLayout); } catch { /* The view still works without storage. */ }
  renderRuns();
});
elements["filter-toggle"].addEventListener("click", () => {
  setFilterDrawer(elements["filter-toggle"].getAttribute("aria-expanded") !== "true");
});
// Native selects can retain :focus-visible after their popup is closed with the mouse.
elements["filter-drawer"].addEventListener("pointerdown", (event) => {
  if (event.target.matches("select")) event.target.dataset.pointerFocus = "true";
});
["keydown", "focusout"].forEach((type) => elements["filter-drawer"].addEventListener(type, (event) => {
  if (event.target.matches("select")) delete event.target.dataset.pointerFocus;
}));
let searchTimer;
elements.search.addEventListener("input", () => {
  renderRuns();
  writeSearchUrl();
  clearTimeout(searchTimer);
  if (branchSnapshotMode()) return;
  searchTimer = setTimeout(() => loadRuns({ branch: state.preferredBranch }).catch((error) => showToast(error.message, true)), 250);
});
elements["branch-filter"].addEventListener("change", async () => {
  const navigationId = ++state.navigationId;
  state.runsRequestId += 1;
  state.preferredBranch = elements["branch-filter"].value;
  state.urlBranch = true;
  writeBranchPreference(state.preferredBranch);
  state.selectedId = null;
  state.selectedVariantKey = null;
  state.models = [];
  state.runs = [];
  state.compileRuns = [];
  state.packageRuns = [];
  state.catalogDefinitions = [];
  elements["result-list"].innerHTML = Array.from({ length: 5 }, () => `<div class="row-skeleton"><span></span><span></span><span></span></div>`).join("");
  showEmptyDetail();
  const url = new URL(location.href);
  url.searchParams.set("branch", state.preferredBranch);
  deletePageParams(url);
  history.pushState({}, "", url);
  updateFilterIndicators();
  try {
    await loadTaxonomy(state.preferredBranch || DEFAULT_BRANCH);
  } catch (error) {
    if (navigationId !== state.navigationId) return;
    setCategoryFiltersDisabled(true);
    showToast(`Category filters are temporarily unavailable. ${error.message}`, true);
  }
  if (navigationId !== state.navigationId) return;
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
});
elements["compiler-filter"].addEventListener("change", async () => {
  updateFilterIndicators();
  if (branchSnapshotMode() && catalogMode()) {
    state.models = branchModels(state.compileRuns, state.packageRuns, state.catalogDefinitions,
      elements["compiler-filter"].value);
    state.runs = state.models.flatMap((model) => model.builds.map((build) => runFromCatalogBuild(model, build)));
    renderRuns();
    writeCachedCatalog();
    return;
  }
  renderRuns();
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
});
elements["history-filter"].addEventListener("change", async () => {
  writeHistoryUrl();
  updateFilterIndicators();
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
});
Object.keys(CATEGORY_QUERY_FIELDS).forEach((elementId) => elements[elementId].addEventListener("change", async () => {
  state.categorySelection[elementId] = elements[elementId].value;
  writeCategoryFilterUrl();
  updateFilterIndicators();
  if (branchSnapshotMode()) {
    renderRuns();
    return;
  }
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
}));
elements["clear-filters"].addEventListener("click", async () => {
  elements.search.value = "";
  const resetBranch = !branchSelectionEnabled || state.availableBranches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH : state.availableBranches[0] || "";
  elements["branch-filter"].value = resetBranch;
  state.preferredBranch = resetBranch;
  state.urlBranch = true;
  writeBranchPreference(state.preferredBranch);
  const resetUrl = new URL(location.href);
  resetUrl.searchParams.set("branch", resetBranch);
  deletePageParams(resetUrl);
  history.pushState({}, "", resetUrl);
  state.selectedId = null;
  state.selectedVariantKey = null;
  showEmptyDetail();
  elements["compiler-filter"].value = "";
  elements["history-filter"].checked = false;
  Object.keys(CATEGORY_QUERY_FIELDS).forEach((elementId) => {
    elements[elementId].value = "";
    state.categorySelection[elementId] = "";
  });
  writeSearchUrl();
  writeCategoryFilterUrl();
  writeHistoryUrl();
  updateFilterIndicators();
  try {
    await loadRuns({ branch: resetBranch });
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
  const picker = event.target.closest("[data-catalog-profile]");
  if (!picker) return;
  const modelId = picker.dataset.catalogProfile;
  const selector = `[data-catalog-profile="${CSS.escape(modelId)}"]`;
  const pickerIndex = [...elements["result-list"].querySelectorAll(selector)].indexOf(picker);
  state.activeProfileByModel.set(modelId, picker.value);
  renderRuns();
  elements["result-list"].querySelectorAll(selector)[pickerIndex]?.focus({preventScroll: true});
});
function returnToCatalog() {
  state.selectedId = null;
  state.selectedVariantKey = null;
  const url = new URL(location.href);
  deletePageParams(url);
  history.pushState({}, "", url);
  renderRuns();
  showEmptyDetail();
}
elements["nav-models"].addEventListener("click", (event) => {
  event.preventDefault();
  returnToCatalog();
});
// The Compare page adds the model page the reader came from to its first empty picker.
elements["nav-compare"].addEventListener("click", (event) => {
  event.preventDefault();
  if (elements.workspace.dataset.view === "compare") return;
  const variant = elements.workspace.dataset.view === "model" ? currentVariant() : null;
  const viewed = variant?.modelId && variant.profile ? pickKey(variant.modelId, variant.profile) : "";
  const empty = state.comparePicks.indexOf("");
  if (viewed && !state.comparePicks.includes(viewed) && empty >= 0) state.comparePicks[empty] = viewed;
  if (viewed) {
    state.compareTask = state.models.find(model => model.model_id === variant.modelId)?.tasks?.[0] || "";
  }
  const url = new URL(location.href);
  deletePageParams(url);
  url.searchParams.set("view", "compare");
  writePicksToUrl(url);
  history.pushState({}, "", url);
  renderComparison();
});
elements["compare-pane"].addEventListener("change", (event) => {
  const add = event.target.closest("[data-compare-add]");
  const control = event.target.closest("[data-compare-control]");
  if (!add && !control) return;
  const key = control?.dataset.compareControl;
  if (add) {
    if (!add.value) return;
    toggleComparison(add.value);
  } else {
    const fields = {task: "compareTask", axis: "compareAxis"};
    state[fields[key]] = control.value;
    if (key === "task") renderComparison();
    else updateComparisonChart();
  }
  const url = new URL(location.href);
  writePicksToUrl(url);
  history.replaceState({}, "", url);
  const next = elements["compare-pane"].querySelector(add ? "[data-compare-add]" : `[data-compare-control="${key}"]`);
  (next?.disabled ? elements["compare-pane"].querySelector(".compare-legend button:last-child") : next)?.focus({preventScroll: true});
});
elements["compare-pane"].addEventListener("click", (event) => {
  const open = event.target.closest("[data-compare-open]");
  if (open) selectRun(open.dataset.compareOpen);
  const point = event.target.closest("[data-compare-toggle]");
  if (!point) return;
  const key = point.dataset.compareToggle;
  const chart = point.classList.contains("compare-point");
  toggleComparison(key);
  const url = new URL(location.href);
  writePicksToUrl(url);
  history.replaceState({}, "", url);
  elements["compare-pane"].querySelector(chart ? `.compare-point[data-compare-toggle="${CSS.escape(key)}"]` : "[data-compare-add]")?.focus({preventScroll: true});
});
elements["compare-pane"].addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideComparisonTooltip();
  if ((event.key === "Enter" || event.key === " ") && event.target.matches(".compare-point")) {
    event.preventDefault();
    event.target.dispatchEvent(new MouseEvent("click", {bubbles: true}));
  }
});
elements["compare-pane"].addEventListener("pointerover", (event) => {
  const point = event.target.closest(".compare-point");
  if (point && !point.contains(event.relatedTarget)) showComparisonTooltip(point);
});
elements["compare-pane"].addEventListener("pointerout", (event) => {
  const point = event.target.closest(".compare-point");
  if (point && !point.contains(event.relatedTarget)) hideComparisonTooltip();
});
elements["compare-pane"].addEventListener("focusin", (event) => {
  const point = event.target.closest(".compare-point");
  if (point) showComparisonTooltip(point);
});
elements["compare-pane"].addEventListener("focusout", hideComparisonTooltip);
window.addEventListener("scroll", hideComparisonTooltip, true);
window.addEventListener("resize", hideComparisonTooltip);
elements["detail-pane"].addEventListener("pointerover", (event) => {
  highlightLayerLegend(event);
  if (event.pointerType === "touch") return;
  const stage = event.target.closest("[data-execution-stage]");
  if (stage && !stage.contains(event.relatedTarget)) showExecutionTooltip(stage);
});
elements["detail-pane"].addEventListener("pointerout", (event) => {
  highlightLayerLegend(event);
  if (event.pointerType === "touch") return;
  const stage = event.target.closest("[data-execution-stage]");
  if (stage && !stage.contains(event.relatedTarget)) deferHideExecutionTooltip();
});
elements["detail-pane"].addEventListener("focusin", (event) => {
  const stage = event.target.closest("[data-execution-stage]");
  if (stage) showExecutionTooltip(stage);
});
elements["detail-pane"].addEventListener("focusout", hideExecutionTooltip);
elements["execution-tooltip"].addEventListener("pointerenter", keepExecutionTooltip);
elements["execution-tooltip"].addEventListener("pointerleave", deferHideExecutionTooltip);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideExecutionTooltip();
});
document.addEventListener("click", (event) => {
  const stage = event.target.closest("[data-execution-stage]");
  if (stage) showExecutionTooltip(stage);
  else if (!elements["execution-tooltip"].contains(event.target)) hideExecutionTooltip();
});
window.addEventListener("scroll", (event) => {
  if (!elements["execution-tooltip"].contains(event.target)) hideExecutionTooltip();
}, true);
window.addEventListener("resize", hideExecutionTooltip);
window.addEventListener("popstate", hideExecutionTooltip);
elements["detail-pane"].addEventListener("input", updateLayerAnalysis);
elements["detail-pane"].addEventListener("click", updateLayerAnalysis);
elements["detail-pane"].addEventListener("click", selectExecutionGraph);
elements["detail-pane"].addEventListener("click", (event) => {
  const variantTab = event.target.closest("[data-detail-profile-id]");
  if (variantTab) {
    const build = selectedCatalogBuild(variantTab.dataset.detailModelId, variantTab.dataset.detailProfileId);
    if (build) selectRun(build.run_id);
    return;
  }
  if (event.target.closest('[data-action="retry"]') && state.selectedId) {
    selectRun(state.selectedId, { updateUrl: false });
    return;
  }
  const copy = event.target.closest("[data-copy]");
  if (copy) copyText(copy.dataset.copy);
  const variantDownload = event.target.closest("[data-download-variant]");
  if (variantDownload) downloadVariant(variantDownload.dataset.downloadVariant);
  const download = event.target.closest("[data-download]");
  if (download?.dataset.download === "package") downloadPackage(download.dataset.runId);
  else if (download) downloadArtifact(download.dataset.runId, download.dataset.download, download);
  // A Details button, such as the execution graph's tensors or the MLA stages, opens its table and, like a
  // disclosure row, stays open on re-render.
  const details = event.target.closest("[data-details]");
  if (details) {
    const key = details.dataset.details;
    const open = details.getAttribute("aria-expanded") !== "true";
    details.setAttribute("aria-expanded", String(open));
    document.getElementById(key).hidden = !open;
    if (open) state.openDisclosures.add(key);
    else state.openDisclosures.delete(key);
    return;
  }
  const sectionLink = event.target.closest("[data-scroll-to]");
  if (sectionLink) {
    const section = elements["detail-pane"].querySelector(`#${sectionLink.dataset.scrollTo}`);
    if (section) {
      section.focus({ preventScroll: true });
      section.scrollIntoView({
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "start",
      });
    }
    return;
  }
  if (event.target.closest(".close-detail")) returnToCatalog();
});
// Re-rendering the page keeps the rows the reader opened.
elements["detail-pane"].addEventListener("toggle", (event) => {
  const key = event.target.dataset?.disclosure;
  if (!key) return;
  if (event.target.open) state.openDisclosures.add(key);
  else state.openDisclosures.delete(key);
}, true);
elements["detail-pane"].addEventListener("keydown", (event) => {
  const tab = event.target.closest('[role="tab"]');
  if (!tab || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [...tab.closest('[role="tablist"]').querySelectorAll('[role="tab"]')];
  const current = tabs.indexOf(tab);
  const next = event.key === "Home" ? 0
    : event.key === "End" ? tabs.length - 1
      : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  tabs[next].click();
});
elements["load-more"].addEventListener("click", () => loadRuns({
  append: true,
  branch: elements["branch-filter"].value,
}).catch((error) => showToast(error.message, true)));
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && elements.workspace.dataset.view === "catalog"
    && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    elements.search.focus();
  }
  if (event.key === "Escape" && document.activeElement === elements.search) {
    elements.search.value = "";
    elements.search.blur();
    elements.search.dispatchEvent(new Event("input"));
  }
});

// The page restores the catalog position itself; browser restoration would reset it after Back.
history.scrollRestoration = "manual";
try { state.catalogLayout = localStorage.getItem("sima-neat-catalog-layout") === "table" ? "table" : "cards"; } catch { /* Use cards when storage is unavailable. */ }
initialize();
window.addEventListener("popstate", () => {
  // Rehydrate branch and selection from the URL, never from the last local choice.
  state.runsRequestId += 1;
  initialize();
});
