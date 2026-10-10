// Build selection and the comparison dialog.
import { adaptBenchmarkResult } from "./benchmark-adapters.js";
import { variantBuildRuns } from "./branch-builds.js";
import { state, elements } from "./state.js";
import { escapeHtml, branchName, shortSha, profileLabel, variantLabel } from "./ui.js";
import { METRIC_DEFINITIONS, metricDefinition, metricDisplay, formatBytes } from "./metrics.js";
import { api, mapWithConcurrency, modelPackArtifacts } from "./api.js";

function comparisonGroups() {
  return [...state.compareSelection.values()];
}

export function updateCompareAction() {
  const selectedCount = state.compareSelection.size;
  elements["compare-button"].disabled = selectedCount < 2;
  elements["compare-label"].textContent = "Compare selected";
  elements["compare-count"].textContent = selectedCount;
  elements["compare-count"].hidden = selectedCount === 0;
  elements["compare-button"].setAttribute(
    "aria-label",
    selectedCount < 2
      ? `Select ${2 - selectedCount} more variant${selectedCount === 1 ? "" : "s"} to compare`
      : `Compare ${selectedCount} selected variants`,
  );
}

export function setComparisonSelection(checkbox) {
  const key = checkbox.dataset.compareVariant;
  const group = state.variantGroups.get(key) || state.compareSelection.get(key);
  if (!group) return null;
  if (checkbox.checked) state.compareSelection.set(key, group);
  else state.compareSelection.delete(key);
  checkbox.closest(".run-entry")?.classList.toggle("compared", checkbox.checked);
  checkbox.setAttribute("aria-label", `${checkbox.checked ? "Remove" : "Add"} ${variantLabel(group)} ${checkbox.checked ? "from" : "to"} comparison`);
  checkbox.closest("label")?.setAttribute("title", `${checkbox.checked ? "Remove from" : "Add to"} comparison`);
  updateCompareAction();
  return group;
}

export function compareCheckbox(group) {
  const compared = state.compareSelection.has(group.key);
  return `<input type="checkbox" data-compare-variant="${escapeHtml(group.key)}" aria-label="${compared ? "Remove" : "Add"} ${escapeHtml(variantLabel(group))} ${compared ? "from" : "to"} comparison"${compared ? " checked" : ""} />`;
}

function comparisonMetric(metric, definition) {
  const display = metricDisplay(metric, definition);
  if (!display) {
    const reason = metric?.reason ? metric.reason.replaceAll("_", " ") : "Not measured";
    return `<span class="comparison-unavailable" title="${escapeHtml(reason)}">—</span>`;
  }
  return `<span class="comparison-value">${escapeHtml(display.value.toFixed(display.digits))}<small>${escapeHtml(display.unit)}</small></span>`;
}

function comparisonEntryFromDetail(run, detail) {
  const summary = detail?.summary || null;
  const adapted = detail?.benchmark ? adaptBenchmarkResult(detail.benchmark.result) : { kind: "missing" };
  return {
    run: summary?.run || run,
    artifacts: modelPackArtifacts(summary?.artifacts),
    benchmark: adapted.kind === "supported" ? adapted.value : null,
  };
}

async function loadComparisonEntry(run) {
  const detail = state.detailCache.get(run.id);
  if (detail) return comparisonEntryFromDetail(run, detail);
  if (state.comparisonCache.has(run.id)) return state.comparisonCache.get(run.id);

  const request = Promise.all([
    api(`/v1/runs/${run.id}`).catch(() => null),
    api(`/v1/runs/${run.id}/benchmarks/latest`).catch(() => null),
  ]).then(([summary, benchmark]) => comparisonEntryFromDetail(run, { summary, benchmark }));
  state.comparisonCache.set(run.id, request);
  return request;
}

// Each component is measured on its own; the CPU stage between components is not, so component
// measurements are listed per component and never added up.
export function renderComparisonTable(variants) {
  const entries = variants.flatMap((variant) => variant.entries);
  const available = [...new Set(entries.flatMap(({ benchmark }) => Object.keys(benchmark?.measurements || {})))]
    .filter((key) => entries.some(({ benchmark }) => benchmark?.measurements[key]?.status === "measured"))
    .sort((left, right) => {
      const knownLeft = Object.keys(METRIC_DEFINITIONS).indexOf(left);
      const knownRight = Object.keys(METRIC_DEFINITIONS).indexOf(right);
      if (knownLeft >= 0 || knownRight >= 0) return (knownLeft < 0 ? 1000 : knownLeft) - (knownRight < 0 ? 1000 : knownRight);
      return left.localeCompare(right);
    });
  const field = available.includes(state.comparisonMetric) ? state.comparisonMetric : available[0];
  if (field) state.comparisonMetric = field;
  const sampleMetric = field
    ? entries.map(({ benchmark }) => benchmark?.measurements[field]).find(Boolean)
    : null;
  const definition = field ? metricDefinition(field, sampleMetric) : null;
  const groups = variants.map(({ group, entries: variantEntries }) => {
    const run = variantEntries[0]?.run || group.anchor;
    const heading = `<tr class="comparison-variant"><th scope="rowgroup" colspan="4"><strong>${escapeHtml(group.modelId || "Unknown model")}</strong><small>${escapeHtml(profileLabel(run))}</small><small>${escapeHtml(branchName(run.git_ref))} · ${escapeHtml(run.toolchain_version || "—")} · ${escapeHtml(shortSha(run.commit_sha))}</small></th></tr>`;
    const rows = variantEntries.map(({ run: componentRun, artifacts, benchmark }) => {
      const artifactBytes = artifacts.reduce((total, artifact) => (
        total + (Number.isFinite(artifact.size_bytes) ? artifact.size_bytes : 0)
      ), 0);
      return `<tr>
        <th scope="row">${escapeHtml(componentRun.metadata?.component_id || "model")}</th>
        <td>${escapeHtml(benchmark?.workload || "No benchmark")}</td>
        <td><span class="comparison-value">${escapeHtml(artifactBytes ? formatBytes(artifactBytes) : "—")}</span></td>
        <td>${field ? comparisonMetric(benchmark?.measurements[field], definition) : '<span class="comparison-unavailable">—</span>'}</td>
      </tr>`;
    }).join("");
    return `<tbody>${heading}${rows}</tbody>`;
  }).join("");
  const options = available.map((key) => {
    const item = metricDefinition(key);
    return `<option value="${escapeHtml(key)}"${key === field ? " selected" : ""}>${escapeHtml(item.label)}</option>`;
  }).join("");
  const note = definition
    ? `${definition.unit}${definition.direction === "neutral" ? "" : ` · ${definition.direction} is better`} · per component`
    : "No shared measurements are available";
  return `<div class="comparison-measurement-bar"><div class="comparison-measurement-control"><label class="chart-control"><span>Compare by</span><select data-comparison-metric>${options}</select></label><p>${escapeHtml(note)}</p></div></div>
  <div class="comparison-table-wrap"><table class="comparison-table">
    <caption class="sr-only">Artifact sizes and benchmark measurements for the components of ${variants.length} variants</caption>
    <thead><tr>
      <th scope="col">Variant / component</th><th scope="col">Workload</th>
      <th scope="col">Artifact size</th><th scope="col">${escapeHtml(definition?.label || "Measurement")}</th>
    </tr></thead>${groups}
  </table></div>`;
}

// A history entry holds only the runs on screen, so a search or page boundary can split a build;
// compare every component of each selected build.
async function completeVariantGroup(group) {
  const cached = state.variantCache.get(group.key);
  return { ...group, runs: cached ? cached.runs : await variantBuildRuns(group.anchor, api) };
}

export async function openComparison() {
  const selected = comparisonGroups();
  if (selected.length < 2) return;
  elements["compare-title"].textContent = `Compare ${selected.length} variants`;
  elements["compare-description"].textContent = "Each component is listed under its variant. Empty benchmark cells mean that measurement is unavailable.";
  elements["compare-content"].innerHTML = `<div class="comparison-loading" role="status"><span class="spinner"></span> Loading artifact and benchmark data…</div>`;
  elements["compare-dialog"].showModal();
  let groups;
  try {
    groups = await Promise.all(selected.map(completeVariantGroup));
  } catch (error) {
    if (elements["compare-dialog"].open) {
      elements["compare-content"].innerHTML = `<p class="muted-message">Could not load every component of the selected variants. ${escapeHtml(error.message)}</p>`;
    }
    return;
  }
  const runs = groups.flatMap((group) => group.runs);
  const entries = await mapWithConcurrency(runs, 6, loadComparisonEntry);
  let offset = 0;
  const variants = groups.map((group) => {
    const variantEntries = entries.slice(offset, offset + group.runs.length);
    offset += group.runs.length;
    return { group, entries: variantEntries };
  });
  state.comparisonEntries = variants;
  if (elements["compare-dialog"].open) elements["compare-content"].innerHTML = renderComparisonTable(variants);
}
