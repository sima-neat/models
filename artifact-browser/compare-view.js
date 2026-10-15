// The Compare page: up to three variant builds of the branch catalog side by side.
import { adaptBenchmarkResult } from "./benchmark-adapters.js";
import { state, elements, catalogMode, showView, taxonomyLabel } from "./state.js";
import { escapeHtml } from "./ui.js";
import { metricDefinition, metricDisplay } from "./metrics.js";
import { anchorBuild, compilerFilteredBuilds, loadRunDetail, loadVariant, profileGroups } from "./api.js";
import { compareModels } from "./catalog-view.js";
import { accuracyHeadline } from "./model-view.js";

// Three builds fit side by side on a laptop screen and stay readable on a phone.
export const MAX_COMPARED_BUILDS = 3;

// A pick names a variant as `<model id>.<build profile>`, which needs no escaping in a link; the
// catalog resolves it to that variant's current build.
export function pickKey(modelId, profile) {
  return `${modelId}.${profile}`;
}

export function picksFromUrl(params) {
  const picks = params.getAll("build").filter(Boolean).slice(0, MAX_COMPARED_BUILDS);
  return Array.from({ length: MAX_COMPARED_BUILDS }, (_, index) => picks[index] || "");
}

export function writePicksToUrl(url) {
  url.searchParams.delete("build");
  state.comparePicks.filter(Boolean).forEach((pick) => url.searchParams.append("build", pick));
}

const PERFORMANCE_ROWS = [
  ["throughput_fps", "Throughput"],
  ["latency_avg_ms", "Latency"],
  ["avg_power_watts", "SoM power"],
  ["energy_per_inference_joules", "Energy per inference"],
];

function catalogVariants() {
  return [...state.models].sort(compareModels).flatMap((model) => (
    profileGroups({ ...model, builds: compilerFilteredBuilds(model) }).map(([profile, builds]) => ({
      key: pickKey(model.model_id, profile), model, profile, build: anchorBuild(builds),
    }))
  ));
}

function picker(index, options) {
  const pick = state.comparePicks[index];
  const choices = options.map((option) => (
    `<option value="${escapeHtml(option.key)}"${option.key === pick ? " selected" : ""}>${escapeHtml(`${option.model.display_name || option.model.model_id} · ${option.profile}`)}</option>`
  )).join("");
  return `<label><span>Build ${index + 1}</span><select data-pick="${index}"><option value="">Choose a build</option>${choices}</select></label>`;
}

function measuredText(metric, key) {
  const definition = metricDefinition(key, metric);
  const display = metricDisplay(metric, definition);
  if (!display) return null;
  return { number: Number(display.value.toFixed(display.digits)), text: `${display.value.toFixed(display.digits)} ${display.unit.replace("/inference", "")}`, direction: definition.direction };
}

// Each component is measured on its own and the CPU stage between components is not, so a build lists
// one value per component and never adds them up.
function componentMeasurements(variant) {
  return variant.details.map((detail) => {
    const adapted = detail.benchmark ? adaptBenchmarkResult(detail.benchmark.result) : null;
    return {
      component: variant.details.length > 1 ? detail.summary.run.metadata?.component_id || "model" : "model",
      measurements: adapted?.kind === "supported" ? adapted.value.measurements : {},
    };
  });
}

// A single-component build competes with other single-component builds; a component of a larger model
// competes only with the same component of another build.
function bestValues(columns, key) {
  const byComponent = new Map();
  let direction = "neutral";
  for (const column of columns) {
    for (const { component, measurements } of column.components) {
      const value = measuredText(measurements[key], key);
      if (!value) continue;
      direction = value.direction;
      byComponent.set(component, [...(byComponent.get(component) || []), value.number]);
    }
  }
  if (direction === "neutral") return new Map();
  const better = direction === "lower" ? Math.min : Math.max;
  return new Map([...byComponent].filter(([, values]) => values.length > 1).map(([component, values]) => [component, better(...values)]));
}

function performanceCell(column, key, best) {
  if (!column.components.length) return "<td>—</td>";
  const lines = column.components.map(({ component, measurements }) => {
    const value = measuredText(measurements[key], key);
    const isBest = value && best.get(component) === value.number;
    const text = value ? `<span class="${isBest ? "best" : ""}">${escapeHtml(value.text)}${isBest ? '<span class="sr-only"> (best)</span>' : ""}</span>` : "<span>—</span>";
    return column.components.length > 1
      ? `<span class="compare-line"><span class="compare-component">${escapeHtml(component)}</span>${text}</span>`
      : text;
  });
  return `<td>${lines.join("")}</td>`;
}

function comparisonTable(columns) {
  const group = (title) => `<tr class="compare-group"><th scope="rowgroup" colspan="${columns.length + 1}">${escapeHtml(title)}</th></tr>`;
  const row = (label, cells) => `<tr><th scope="row">${escapeHtml(label)}</th>${cells.join("")}</tr>`;
  const text = (value) => `<td>${escapeHtml(value || "—")}</td>`;
  const performance = PERFORMANCE_ROWS.map(([key, label]) => {
    const best = bestValues(columns, key);
    return row(label, columns.map((column) => performanceCell(column, key, best)));
  }).join("");
  return `<div class="compare-table-wrap"><table class="compare-table">
    <thead><tr><td></td>${columns.map(({ option, failed }) => `<th scope="col">${escapeHtml(option.model.display_name || option.model.model_id)}<small>${escapeHtml(failed ? "Could not load this build" : option.profile)}</small></th>`).join("")}</tr></thead>
    <tbody>
      ${group("Model")}
      ${row("Task", columns.map(({ option }) => text((option.model.tasks || []).map((id) => taxonomyLabel("tasks", id)).join(", "))))}
      ${row("Components", columns.map(({ variant }) => text(variant ? String(variant.details.length) : "")))}
      ${group("Performance on Modalix")}
      ${performance}
      ${group("Accuracy")}
      ${row("Headline metric", columns.map(({ accuracy }) => text(accuracy && `${accuracy.label} ${accuracy.value}`)))}
      ${row("Published", columns.map(({ accuracy }) => text(accuracy?.published)))}
    </tbody>
  </table></div>`;
}

async function loadPickedVariant(option) {
  const anchor = (await loadRunDetail(option.build.run_id)).summary.run;
  return loadVariant(anchor);
}

let compareRequestId = 0;

export async function renderComparison() {
  showView("compare");
  const requestId = ++compareRequestId;
  const intro = `<header class="catalog-intro"><h2 id="compare-title" tabindex="-1">Compare</h2>
    <p>Pick up to three builds. Bold marks the best value in each row. A model with several components lists each one on its own line, compared only with the same component of another build.</p></header>`;
  if (!catalogMode()) {
    elements["compare-pane"].innerHTML = `${intro}<p class="muted-message">Comparison lists the builds of one branch. Choose a branch under Models and turn off build history.</p>`;
    return;
  }
  const options = catalogVariants();
  const picked = state.comparePicks.map((key) => options.find((option) => option.key === key)).filter(Boolean);
  elements["compare-pane"].innerHTML = `${intro}
    <div class="compare-picks">${state.comparePicks.map((_, index) => picker(index, options)).join("")}</div>
    <div class="compare-results" aria-live="polite">${picked.length ? '<p class="muted-message compare-loading" role="status"><span class="spinner"></span> Loading builds…</p>' : '<p class="muted-message">Choose a build to start.</p>'}</div>`;
  if (!picked.length) return;
  const variants = await Promise.all(picked.map((option) => loadPickedVariant(option).catch(() => null)));
  if (requestId !== compareRequestId) return;
  const columns = picked.map((option, index) => {
    const variant = variants[index];
    return {
      option, variant, failed: !variant,
      components: variant ? componentMeasurements(variant) : [],
      accuracy: variant ? accuracyHeadline(variant) : null,
    };
  });
  elements["compare-pane"].querySelector(".compare-results").innerHTML = comparisonTable(columns);
}
