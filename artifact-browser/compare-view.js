// Explore all variants of one task; inspect up to three measured builds side by side.
import { CUSTOMER_HIDDEN_METRICS, readPerformance } from "./model-report.js";
import { customerView, state, elements, catalogMode, showView, taxonomyLabel } from "./state.js";
import { escapeHtml, variantLabel } from "./ui.js";
import { datasetName } from "./accuracy-metrics.js";
import { metricDefinition, metricDisplay } from "./metrics.js";
import { anchorBuild, compilerFilteredBuilds, loadRunDetail, loadVariant, profileGroups, mapWithConcurrency } from "./api.js";
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
  const picks = [...new Set(params.getAll("build").filter(Boolean))].slice(0, MAX_COMPARED_BUILDS);
  return Array.from({ length: MAX_COMPARED_BUILDS }, (_, index) => picks[index] || "");
}

export function writePicksToUrl(url) {
  url.searchParams.delete("build");
  state.comparePicks.filter(Boolean).forEach((pick) => url.searchParams.append("build", pick));
  if (state.compareWorkload) url.searchParams.set("compare_workload", state.compareWorkload);
  else url.searchParams.delete("compare_workload");
  if (state.compareTask) url.searchParams.set("compare_task", state.compareTask);
  else url.searchParams.delete("compare_task");
  url.searchParams.delete("compare_precision");
}

const PERFORMANCE_ROWS = [
  ["throughput_fps", "Throughput"],
  ["latency_avg_ms", "Latency"],
  ["avg_power_watts", "SoM power"],
  ["energy_per_inference_joules", "Energy per inference"],
  ["mla_memory_peak_mib", "Peak MLA memory"],
  ["ev74_memory_peak_mib", "Peak EV74 memory"],
].filter(([key]) => !customerView || !CUSTOMER_HIDDEN_METRICS.includes(key));

function catalogVariants() {
  return [...state.models].sort(compareModels).flatMap((model) => (
    profileGroups({ ...model, builds: compilerFilteredBuilds(model) }).map(([profile, builds]) => ({
      key: pickKey(model.model_id, profile), model, profile, build: anchorBuild(builds),
    }))
  ));
}

function measuredText(metric, key) {
  const definition = metricDefinition(key);
  const display = metricDisplay(metric, definition);
  if (!display) return null;
  return { number: Number(display.value.toFixed(display.digits)), text: `${display.value.toFixed(display.digits)} ${display.unit.replace("/inference", "")}`, direction: definition.direction };
}

// Independent components have no whole-model performance to plot or compare.
function variantMeasurements(variant) {
  const result = readPerformance(variant.performance);
  return result.kind === "supported" ? result.value.measurements || {} : {};
}

function bestValue(columns, key) {
  const values = columns.map((column) => measuredText(column.measurements[key], key)).filter(Boolean);
  if (values.length < 2 || values[0].direction === "neutral") return null;
  return (values[0].direction === "lower" ? Math.min : Math.max)(...values.map((value) => value.number));
}

function performanceCell(column, key, best) {
  const value = measuredText(column.measurements[key], key);
  if (!value) return "<td>—</td>";
  const isBest = best === value.number;
  return `<td><span class="${isBest ? "best" : ""}">${escapeHtml(value.text)}${isBest ? '<span class="sr-only"> (best)</span>' : ""}</span></td>`;
}

function comparisonTable(columns) {
  const group = (title) => `<tr class="compare-group"><th scope="rowgroup" colspan="${columns.length + 1}">${escapeHtml(title)}</th></tr>`;
  const row = (label, cells) => `<tr><th scope="row">${escapeHtml(label)}</th>${cells.join("")}</tr>`;
  const text = (value) => `<td>${escapeHtml(value || "—")}</td>`;
  const rows = generative ? [
    ["ttft_ms", "Time to first token"],
    ...(state.compareTask === "visual_question_answering" ? [] : [["prefill_tps", "Prefill"]]),
    ["decode_tps", "Generation"],
    ["draft_contribution_pct", "Draft contribution"], ["decode_speedup", "Generation speedup"],
    ...PERFORMANCE_ROWS.filter(([key]) => !["throughput_fps", "latency_avg_ms", "energy_per_inference_joules"].includes(key)),
  ] : PERFORMANCE_ROWS;
  const performance = rows.map(([key, label]) => {
    const best = bestValue(columns, key);
    return row(label, columns.map((column) => performanceCell(column, key, best)));
  }).join("");
  return `<div class="compare-table-wrap" role="region" aria-label="Selected model measurements" tabindex="0"><table class="compare-table">
    <thead><tr><td>Measurements</td>${columns.map(({ option, failed }, index) => `<th scope="col"><div class="compare-column-heading"><div><span class="compare-column-marker" data-color="${index}" aria-hidden="true"></span><button type="button" class="catalog-model-link" data-compare-open="${escapeHtml(option.build.run_id)}">${escapeHtml(option.model.display_name || option.model.model_id)}</button><small>${escapeHtml(failed ? "Could not load this build" : variantLabel(option.profile))}</small></div><button type="button" class="compare-remove" data-compare-toggle="${escapeHtml(option.key)}" aria-label="Remove ${escapeHtml(option.model.display_name || option.model.model_id)} ${escapeHtml(variantLabel(option.profile))} from comparison" title="Remove from comparison"><span aria-hidden="true">×</span></button></div></th>`).join("")}</tr></thead>
    <tbody>
      ${group("Model")}
      ${generative ? row("Parameters", columns.map(({option}) => text(parameterText(option.model)))) : ""}
      ${generative && columns.some(column => comparisonWorkload(column.performance)?.kind === "image_text")
        ? row("Image resolution", columns.map(column => text(comparisonWorkload(column.performance)?.image_size?.join(" × ")))) : ""}
      ${row("Task", columns.map(({ option }) => text((option.model.tasks || []).map((id) => taxonomyLabel("tasks", id)).join(", "))))}
      ${row("Components", columns.map(({ variant }) => text(variant ? String(variant.details.length) : "")))}
      ${group("Performance on Modalix")}
      ${performance}
      ${group("Accuracy")}
      ${row("Headline metric", columns.map(({ accuracy }) => text(accuracy && `${accuracy.label} ${accuracy.value}`)))}
      ${row("Published", columns.map(({ accuracy }) => text(accuracy?.reference)))}
    </tbody>
  </table></div>`;
}

async function loadPickedVariant(option) {
  const anchor = (await loadRunDetail(option.build.run_id)).summary.run;
  return loadVariant(anchor);
}

let compareRequestId = 0;
let loadedColumns = [];
let generative = false;

function selectControl(label, key, choices, selected) {
  return `<label><span>${escapeHtml(label)}</span><select data-compare-control="${key}">${choices.map(([id, name]) =>
    `<option value="${escapeHtml(id)}"${id === selected ? " selected" : ""}>${escapeHtml(name)}</option>`).join("")}</select></label>`;
}

function selectionColor(key) {
  return state.comparePicks.filter(Boolean).indexOf(key);
}

function parameterText(model) {
  const count = model.parameter_count_billion;
  return Number.isFinite(count) && count > 0 ? `${count} B` : "";
}

function workloadMeasurements(column) {
  return {...column, measurements: comparisonWorkload(column.performance)?.measurements || {}};
}

function comparisonWorkload(component) {
  return component?.workloads?.find(workload => workload.key === state.compareWorkload);
}

function comparisonLabel(workload) {
  return workload?.kind === "image_text" ? "Image + text at encoder resolution" : workload?.label || "";
}

function generativePlot(columns) {
  const image = state.compareTask === "visual_question_answering";
  const workloads = [...new Map(columns.flatMap(column => (column.performance?.workloads || []))
    .filter(workload => image ? ["image_text", "image_dataset"].includes(workload.kind)
      : ["text", "text_dataset"].includes(workload.kind)).map(workload => [workload.key, workload])).values()]
    .sort((a, b) => (a.inputTokens || Infinity) - (b.inputTokens || Infinity));
  if (!workloads.some(workload => workload.key === state.compareWorkload)) state.compareWorkload = workloads[0]?.key || "";
  const plotted = columns.flatMap(column => {
    const workload = comparisonWorkload(column.performance);
    const decode = measuredText(workload?.measurements.decode_tps, "decode_tps");
    const x = column.option.model.parameter_count_billion;
    if (!Number.isFinite(x) || x <= 0 || !decode) return [];
    const label = `${column.option.model.display_name} · ${variantLabel(column.option.profile)} · ${workload.label}: ${parameterText(column.option.model)}, ${decode.text}`;
    return [{column, x, y: decode.number, label}];
  });
  return measurementChart(columns, plotted, {
    title: "Parameters & generation speed", xLabel: "Parameters (billions)", yLabel: "Generation (tokens/s)",
    subtitle: comparisonLabel(workloads.find(workload => workload.key === state.compareWorkload)),
    controls: workloads.length > 1 ? selectControl("Workload", "workload", workloads.map(workload => [workload.key, comparisonLabel(workload)]), state.compareWorkload) : "",
    note: "Variants without matching measurements remain available in Add model.",
    empty: "No matching measurements yet. Use Add model to inspect a variant.",
  });
}

function comparisonPlot(columns) {
  if (generative) return generativePlot(columns);
  const points = columns.flatMap(column => {
    const readings = column.measurements;
    const throughput = measuredText(readings?.throughput_fps, "throughput_fps");
    return throughput ? [{column, x: throughput.number}] : [];
  });
  const groups = new Map();
  for (const {column} of points) {
    const accuracy = column.accuracy;
    if (accuracy?.comparisonKey && Number.isFinite(accuracy.numericValue)) {
      groups.set(accuracy.comparisonKey, `${datasetName(accuracy.dataset)} · ${accuracy.label}`);
    }
  }
  const axes = [...groups];
  if (!groups.has(state.compareAxis)) state.compareAxis = axes[0]?.[0] || "";
  const plotted = points.flatMap(point => {
    const y = point.column.accuracy?.comparisonKey === state.compareAxis ? point.column.accuracy.numericValue : null;
    return Number.isFinite(y) ? [{...point, y}] : [];
  });
  const accuracy = plotted[0]?.column.accuracy;
  const yLabel = `${accuracy?.label || "Accuracy"}${accuracy?.unit ? ` (${accuracy.unit}${accuracy.direction === "lower" ? ", lower is better" : ""})` : ""}`;
  return measurementChart(columns, plotted.map(point => ({...point,
    label: `${point.column.option.model.display_name || point.column.option.model.model_id} · ${variantLabel(point.column.option.profile)}: ${point.x} inferences/s, accuracy ${point.column.accuracy.value}`,
  })), {
    title: "Accuracy & speed", xLabel: "Throughput (inferences/s)", yLabel,
    subtitle: accuracy?.dataset ? `Dataset: ${datasetName(accuracy.dataset)}` : "",
    controls: axes.length > 1 ? selectControl("Dataset & metric", "axis", axes, state.compareAxis) : "",
    note: "Accuracy points share a dataset and metric. Use Add model for variants without matching measurements.",
    empty: "No matching measurements yet. Use Add model to inspect a variant.",
  });
}

function measurementChart(columns, plotted, {title, xLabel, yLabel, subtitle = "", controls = "", note, empty}) {
  const maxX = Math.max(.01, ...plotted.map(point => point.x)) * 1.12;
  const maxY = Math.max(1, ...plotted.map(point => point.y));
  const minY = Math.max(0, Math.min(maxY, ...plotted.map(point => point.y)) - (maxY * .08));
  const upperY = maxY * 1.05;
  const x = value => 88 + value / maxX * 830;
  const y = value => 335 - (value - minY) / (upperY - minY) * 285;
  const tick = (value, digits = 1) => Number(value.toFixed(digits)).toLocaleString();
  const grid = Array.from({length: 5}, (_, i) => {
    const horizontal = minY + (upperY - minY) * i / 4;
    const vertical = maxX * i / 4;
    return `<line x1="88" x2="918" y1="${y(horizontal)}" y2="${y(horizontal)}"/><text x="73" y="${y(horizontal) + 4}" text-anchor="end">${tick(horizontal)}</text>
      <line x1="${x(vertical)}" x2="${x(vertical)}" y1="50" y2="335"/><text x="${x(vertical)}" y="360" text-anchor="middle">${tick(vertical, generative ? 2 : 1)}</text>`;
  }).join("");
  const dots = plotted.map(({column, label, x: px, y: py}) => {
    return `<g class="compare-point" data-compare-toggle="${escapeHtml(column.option.key)}" data-color="${selectionColor(column.option.key)}" role="button" tabindex="0" aria-pressed="${selectionColor(column.option.key) >= 0}" aria-disabled="${selectionColor(column.option.key) < 0 && state.comparePicks.filter(Boolean).length >= MAX_COMPARED_BUILDS}" aria-label="${escapeHtml(label)}">
      <circle class="compare-point-hit" cx="${x(px)}" cy="${y(py)}" r="16"/><circle class="compare-point-dot" cx="${x(px)}" cy="${y(py)}" r="7"/></g>`;
  }).join("");
  const legend = plotted.filter(({column}) => selectionColor(column.option.key) >= 0).map(({column}) => {
    const {option} = column;
    return `<button type="button" data-compare-toggle="${escapeHtml(option.key)}" data-color="${selectionColor(option.key)}" aria-label="Remove ${escapeHtml(option.model.display_name)} ${escapeHtml(variantLabel(option.profile))} from comparison"><span class="compare-column-marker" aria-hidden="true"></span>${escapeHtml(option.model.display_name)} · ${escapeHtml(variantLabel(option.profile))}<span aria-hidden="true">×</span></button>`;
  }).join("");
  return `<section class="compare-chart-panel" aria-labelledby="compare-chart-title">
    <div class="compare-chart-header"><div><h3 id="compare-chart-title">${escapeHtml(title)}</h3>
      ${subtitle ? `<p class="compare-chart-dataset">${escapeHtml(subtitle)}</p>` : ""}
      <p>${plotted.length} of ${columns.length} variants shown · Select up to three to inspect below.</p></div>
      ${controls ? `<div class="compare-chart-controls">${controls}</div>` : ""}</div>
    <div class="compare-legend">${legend}</div>
    ${plotted.length ? `<div class="compare-chart-scroll" role="region" aria-label="Model measurement chart" tabindex="0"><svg class="compare-chart" viewBox="0 0 960 410" role="group" aria-label="Model measurements: ${escapeHtml(xLabel)} versus ${escapeHtml(yLabel)}">
      <g class="compare-chart-grid">${grid}</g><text class="compare-axis-title" x="503" y="398" text-anchor="middle">${escapeHtml(xLabel)}</text>
      <text class="compare-axis-title" transform="translate(23 195) rotate(-90)" text-anchor="middle">${escapeHtml(yLabel)}</text>${dots}</svg></div>` : `<p class="compare-empty">${escapeHtml(empty)}</p>`}
    <p class="compare-chart-note">${escapeHtml(note)}${plotted.some(({column}) => column.synthetic) ? " Includes synthetic preview data for visualization only." : ""}</p>
    <div class="compare-tooltip" id="compare-point-tooltip" role="tooltip" hidden></div>
  </section>`;
}

function modelPicker(columns) {
  const full = state.comparePicks.filter(Boolean).length >= MAX_COMPARED_BUILDS;
  const choices = columns.filter(column => selectionColor(column.option.key) < 0).map(({option}) =>
    `<option value="${escapeHtml(option.key)}">${escapeHtml(option.model.display_name || option.model.model_id)} · ${escapeHtml(variantLabel(option.profile))}</option>`).join("");
  return `<div class="compare-add compare-chart-controls"><label><span>Add model</span><select data-compare-add${full ? " disabled" : ""}><option value="">${full ? "3 models selected" : "Choose a model variant"}</option>${choices}</select></label></div>`;
}

function renderMeasurements() {
  hideComparisonTooltip();
  const results = elements["compare-pane"].querySelector(".compare-results");
  const selected = state.comparePicks.filter(Boolean).map(key => loadedColumns.find(column => column.option.key === key)).filter(Boolean);
  results.innerHTML = `${comparisonPlot(loadedColumns)}${modelPicker(loadedColumns)}${selected.length ? `<section class="compare-selected" aria-labelledby="compare-selected-title"><div class="compare-section-heading"><h3 id="compare-selected-title">Selected models</h3><span>${selected.length} / ${MAX_COMPARED_BUILDS}</span></div>${comparisonTable(generative ? selected.map(workloadMeasurements) : selected)}<p class="compare-chart-note">Bold marks the best value.</p></section>` : ""}`;
}

export function toggleComparison(key) {
  const picks = state.comparePicks.filter(Boolean);
  const index = picks.indexOf(key);
  if (index >= 0) picks.splice(index, 1);
  else if (picks.length < MAX_COMPARED_BUILDS && loadedColumns.some(column => column.option.key === key)) picks.push(key);
  state.comparePicks = Array.from({length: MAX_COMPARED_BUILDS}, (_, i) => picks[i] || "");
  renderMeasurements();
}

export function updateComparisonChart() {
  renderMeasurements();
}

let tooltipPoint = null;

export function hideComparisonTooltip() {
  tooltipPoint?.removeAttribute("aria-describedby");
  tooltipPoint = null;
  const tooltip = elements["compare-pane"].querySelector(".compare-tooltip");
  if (tooltip) tooltip.hidden = true;
}

export function showComparisonTooltip(point) {
  const tooltip = elements["compare-pane"].querySelector(".compare-tooltip");
  const column = loadedColumns.find(column => column.option.key === point.dataset.compareToggle);
  if (!tooltip || !column) return;
  hideComparisonTooltip();
  const throughput = measuredText(column.measurements.throughput_fps, "throughput_fps");
  const field = (label, value) => `<div><dt>${label}</dt><dd>${escapeHtml(value)}</dd></div>`;
  tooltip.innerHTML = `<strong>${escapeHtml(column.option.model.display_name || column.option.model.model_id)}</strong><dl>
    ${field("Variant", variantLabel(column.option.profile))}
    ${generative ? field("Parameters", parameterText(column.option.model)) +
      field("Workload", comparisonWorkload(column.performance)?.label || "Not measured") +
      field("Generation", measuredText(comparisonWorkload(column.performance)?.measurements.decode_tps, "decode_tps")?.text || "Not measured")
      : field("Throughput", throughput?.text || "Not measured") +
      field("Accuracy", column.accuracy ? `${column.accuracy.value} · ${column.accuracy.label}` : "Not measured")}</dl>`;
  tooltip.hidden = false;
  tooltipPoint = point;
  point.setAttribute("aria-describedby", tooltip.id);
  const anchor = point.querySelector(".compare-point-dot").getBoundingClientRect();
  const box = tooltip.getBoundingClientRect();
  const left = Math.max(12, Math.min(innerWidth - box.width - 12, anchor.left + anchor.width / 2 - box.width / 2));
  const above = anchor.top - box.height - 12;
  const top = above >= 72 ? above : Math.max(12, Math.min(innerHeight - box.height - 12, anchor.bottom + 12));
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

export async function renderComparison() {
  showView("compare");
  const requestId = ++compareRequestId;
  hideComparisonTooltip();
  loadedColumns = [];
  const intro = `<header class="catalog-intro"><h2 id="compare-title" tabindex="-1">Compare</h2><p>Explore models for your task. Find the balance between accuracy, speed and resource use.</p></header>`;
  if (!catalogMode()) {
    elements["compare-pane"].innerHTML = `${intro}<p class="muted-message">Choose a branch under Models and turn off build history to compare models.</p>`;
    return;
  }
  const all = catalogVariants();
  const tasks = [...new Set(all.flatMap(option => option.model.tasks || []))].sort((a, b) => taxonomyLabel("tasks", a).localeCompare(taxonomyLabel("tasks", b)));
  if (!tasks.includes(state.compareTask)) state.compareTask = all.find(option => state.comparePicks.includes(option.key))?.model.tasks?.[0] || tasks[0] || "";
  const taskOptions = all.filter(option => option.model.tasks?.includes(state.compareTask));
  const options = taskOptions;
  state.comparePicks = state.comparePicks.map(key => options.some(option => option.key === key) ? key : "");
  generative = ["text_generation", "visual_question_answering"].includes(state.compareTask);
  const pending = !generative && options.length > 0 && options.every(option => option.model.model_type !== "vision");
  elements["compare-pane"].innerHTML = `${intro}<div class="compare-filters">${selectControl("Task", "task", tasks.map(id => [id, taxonomyLabel("tasks", id)]), state.compareTask)}</div>
    <div class="compare-results" aria-live="polite">${pending ? '<div class="compare-empty"><span class="compare-tbd">TBD</span><h3>Model comparisons</h3></div>' : options.length ? '<p class="muted-message compare-loading" role="status"><span class="spinner"></span> Loading measurements…</p>' : '<p class="compare-empty">No models available for this task.</p>'}</div>`;
  if (pending || !options.length) return;
  const columns = await mapWithConcurrency(options, 3, async option => {
    if (requestId !== compareRequestId) return null;
    const variant = await loadPickedVariant(option).catch(() => null);
    const result = variant ? readPerformance(variant.performance) : null;
    return {option, variant, failed: !variant, performance: result?.kind === "supported" ? result.value : null,
      synthetic: result?.value?.generation?.synthetic, measurements: variant ? variantMeasurements(variant) : {},
      accuracy: variant ? accuracyHeadline(variant) : null};
  });
  if (requestId !== compareRequestId) return;
  loadedColumns = columns.filter(Boolean);
  renderMeasurements();
}
