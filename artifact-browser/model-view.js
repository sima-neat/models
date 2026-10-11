// Model page: performance, accuracy, model card, build details, and license.
import { adaptBenchmarkResult } from "./benchmark-adapters.js";
import { renderMarkdown, resolveDocumentationLink } from "./markdown.js";
import { state, elements, catalogMode, currentVariant, showView, taxonomyLabel } from "./state.js";
import { icons, escapeHtml, branchName, shortSha, profileLabel, showToast, disclosure } from "./ui.js";
import {
  metricDefinition, metricDisplay, formatDate, formatDuration,
  formatBytes,
} from "./metrics.js";
import {
  apiText, profileGroups, variantPackage, modelPackArtifacts, catalogModelForRun,
  compilerFilteredBuilds, documentationHeroUrl,
} from "./api.js";

function githubModelUrl(run, modelId) {
  const organization = String(state.repository.organization || "");
  const repository = String(state.repository.name || "");
  const configuredRepository = `${organization}/${repository}`.toLowerCase();
  const runRepository = String(run.repository || "").toLowerCase();
  const commit = String(run.commit_sha || "");
  if (!/^[A-Za-z0-9_.-]+$/.test(organization)
    || !/^[A-Za-z0-9_.-]+$/.test(repository)
    || runRepository !== configuredRepository
    || !/^[a-f0-9]{40}$/i.test(commit)
    || !/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test(modelId)) return null;
  return `https://github.com/${encodeURIComponent(organization)}/${encodeURIComponent(repository)}/tree/${commit}/models/${encodeURIComponent(modelId)}`;
}

function mpkValidationMark(tests = []) {
  const passed = tests.some((test) => test.name === "mpk-validation" && test.status === "passed");
  if (!passed) return "";
  const message = "MPK package integrity verified: archive structure, checksum, and build provenance match.";
  return `<span class="mpk-validation-mark" tabindex="0" role="img" aria-label="${message}" data-tooltip="${message}">✓</span>`;
}

function trustedHuggingFaceUrl(value = "") {
  try {
    const url = new URL(String(value));
    if (url.protocol !== "https:" || url.hostname !== "huggingface.co" || url.username || url.password) return null;
    return url.href;
  } catch {
    return null;
  }
}

function huggingFaceSourceLink(input = {}) {
  const sourceReference = String(input.source_reference || "");
  if (!sourceReference.startsWith("hf://") && !trustedHuggingFaceUrl(sourceReference)) return null;
  const declaredModelCard = trustedHuggingFaceUrl(input.metadata?.model_card);
  if (declaredModelCard) {
    return { href: declaredModelCard, label: declaredModelCard.replace(/^https:\/\//, "").replace(/\/$/, "") };
  }

  const match = /^hf:\/\/(models|datasets|spaces)\/([^/@?#]+)\/([^/@?#]+)@[a-f0-9]{40}\/.+$/i.exec(
    String(input.source_reference || ""),
  );
  if (!match) return null;
  const [, repositoryType, owner, repository] = match;
  const typePrefix = repositoryType === "models" ? "" : `${repositoryType}/`;
  const href = `https://huggingface.co/${typePrefix}${encodeURIComponent(owner)}/${encodeURIComponent(repository)}`;
  return { href, label: href.replace(/^https:\/\//, "") };
}

function sourceInputPanel(inputs) {
  if (!inputs.length) return "";
  const meta = inputs.map((input) => formatBytes(input.size_bytes)).join(" · ");
  return disclosure("source", inputs.length > 1 ? "Source models" : "Source model", inputs.map((input) => `<div class="source-block">
    <strong>${escapeHtml(input.name)}</strong>
    <p>${escapeHtml(input.source_reference || input.input_type)}</p>
    <div><span>${escapeHtml(formatBytes(input.size_bytes))}</span><code>${escapeHtml(input.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(input.sha256)}" aria-label="Copy source checksum">${icons.copy}</button></div>
  </div>`).join(""), meta);
}

export function detailSkeleton() {
  showView("model");
  elements["detail-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span><span></span></div>`;
}

export function showEmptyDetail() {
  elements["detail-pane"].innerHTML = "";
  showView("catalog");
}

function keyValue(label, value, { copy = false, mono = false } = {}) {
  const safeValue = escapeHtml(value ?? "—");
  return `<div class="key-value"><dt>${escapeHtml(label)}</dt><dd class="${mono ? "mono" : ""}"><span title="${safeValue}">${safeValue}</span>${copy && value ? `<button type="button" class="copy-button" data-copy="${safeValue}" aria-label="Copy ${escapeHtml(label)}">${icons.copy}</button>` : ""}</dd></div>`;
}

function detailTag(icon, value, kind) {
  return `<span class="detail-tag detail-tag-${kind}">${icon}<span>${escapeHtml(value)}</span></span>`;
}

// Benchmark details group the measurements by what they tell a reader. A metric this browser does not
// know yet still appears, under Other measurements.
const METRIC_GROUPS = [
  ["speed", "Speed", "gauge", ["latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "throughput_fps"]],
  ["heat", "Power & heat", "thermometer", ["avg_power_watts", "energy_per_inference_joules", "energy_joules"]],
  ["memory", "MLA memory", "memory", [
    "mla_memory_baseline_mib", "mla_memory_peak_mib", "mla_memory_peak_delta_mib", "mla_memory_final_mib",
  ]],
  ["startup", "Startup", "timer", ["model_load_ms", "time_to_first_inference_ms"]],
];

// Inside the MLA memory card the shared prefix is implied, so its rows use shorter names.
const GROUP_LABELS = {
  mla_memory_baseline_mib: "Before loading",
  mla_memory_peak_mib: "Peak",
  mla_memory_peak_delta_mib: "Peak increase",
  mla_memory_final_mib: "After the run",
};

function measuredValue(key, metric) {
  const definition = metricDefinition(key, metric);
  const display = metricDisplay(metric, definition);
  return display && { label: definition.label, text: display.value.toFixed(display.digits), unit: display.unit, value: display.value };
}

function metricRow({ label, text, unit }) {
  return `<div class="metric-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(text)}<small>${escapeHtml(unit)}</small></dd></div>`;
}

function latencyTrials(trials, measurements) {
  const definition = metricDefinition("latency_avg_ms", measurements.latency_avg_ms);
  const measured = trials.filter((trial) => Number.isFinite(trial.metrics.latency_avg_ms));
  if (measured.length < 2) return "";
  const values = measured.map((trial) => trial.metrics.latency_avg_ms * definition.scale);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const spread = mean ? ((Math.max(...values) - Math.min(...values)) / mean) * 100 : 0;
  const chips = measured.map((trial, index) => `<span>${trial.number} · <b>${values[index].toFixed(definition.digits)}</b></span>`).join("");
  return `<div class="trial-chips" role="list" aria-label="Average latency per trial">${chips}</div><p class="metric-caption">Latency per trial in ${escapeHtml(definition.unit)} · spread ${spread.toFixed(1)}%</p>`;
}

// The hottest temp1 sensor is the board's die temperature; other sensors are only used without one.
function temperatureSamples(series) {
  if (!series.length) return null;
  const hottest = (candidates) => [...candidates].sort((left, right) => (
    Math.max(...right.samples.map((sample) => sample.celsius))
    - Math.max(...left.samples.map((sample) => sample.celsius))
  ))[0];
  const temp1 = series.filter((item) => item.sensor.toLowerCase() === "temp1");
  return hottest(temp1.length ? temp1 : series);
}

function temperatureRow(selected) {
  const samples = selected.samples;
  return metricRow({
    label: "Temperature",
    text: `${samples[0].celsius.toFixed(1)} → ${samples.at(-1).celsius.toFixed(1)}`,
    unit: "°C",
  });
}

function temperatureSparkline(selected) {
  const samples = selected.samples;
  const temperatures = samples.map((sample) => sample.celsius);
  const startTime = Math.min(...samples.map((sample) => sample.timestamp));
  const elapsed = Math.max(Math.max(...samples.map((sample) => sample.timestamp)) - startTime, 1);
  const minimum = Math.min(...temperatures);
  const maximum = Math.max(...temperatures);
  const spread = maximum - minimum || 1;
  const width = 300;
  const height = 56;
  const x = (timestamp) => ((timestamp - startTime) / elapsed) * width;
  const y = (celsius) => 6 + ((maximum - celsius) / spread) * (height - 12);
  const line = samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.timestamp).toFixed(1)} ${y(sample.celsius).toFixed(1)}`).join(" ");
  const sampleLabel = selected.legacy && samples.length === 2 ? "start and end only" : `${samples.length} samples`;
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(selected.label)} temperature from ${minimum.toFixed(1)} to ${maximum.toFixed(1)} °C over ${samples.length} samples"><path class="sparkline-area" d="${line} L${width} ${height} L0 ${height} Z"/><path class="sparkline-line" d="${line}"/></svg><p class="metric-caption">${escapeHtml(selected.label)} · ${sampleLabel}</p>`;
}

// A bar from zero to `total`: the muted part runs to `first`, the accent part from there to the end.
function proportionBar(first, total, label, legend) {
  const split = Math.min(Math.max(first / total, 0), 1) * 100;
  return `<div class="metric-bar" role="img" aria-label="${escapeHtml(label)}"><svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true"><rect class="metric-bar-first" width="${split.toFixed(2)}" height="8"/><rect class="metric-bar-rest" x="${split.toFixed(2)}" width="${(100 - split).toFixed(2)}" height="8"/></svg></div><div class="metric-bar-legend">${legend.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function memoryBar(measurements) {
  const before = measuredValue("mla_memory_baseline_mib", measurements.mla_memory_baseline_mib);
  const peak = measuredValue("mla_memory_peak_mib", measurements.mla_memory_peak_mib);
  if (!before || !peak || peak.value <= 0) return "";
  return proportionBar(before.value, peak.value, `MLA memory rises from ${before.text} ${before.unit} before loading to a ${peak.text} ${peak.unit} peak`, [
    `Before ${before.text} ${before.unit}`, `Peak ${peak.text} ${peak.unit}`,
  ]);
}

function startupTimeline(measurements) {
  const load = measuredValue("model_load_ms", measurements.model_load_ms);
  const first = measuredValue("time_to_first_inference_ms", measurements.time_to_first_inference_ms);
  if (!load || !first || first.value <= 0) return "";
  return proportionBar(load.value, first.value, `Loading takes ${load.text} ${load.unit} of the ${first.text} ${first.unit} before the first inference`, [
    `Load ${load.text} ${load.unit}`, `First inference ${first.text} ${first.unit}`,
  ]);
}

function metricGroup(id, title, icon, rows, extra = "") {
  return `<div class="metric-group" data-group="${id}"><h4><i aria-hidden="true">${icons[icon]}</i>${escapeHtml(title)}</h4><dl>${rows.join("")}</dl>${extra}</div>`;
}

function metricGroups(benchmark) {
  const { measurements } = benchmark;
  const rowsFor = (keys) => keys.map((key) => {
    const value = measurements[key] && measuredValue(key, measurements[key]);
    return value && metricRow({ ...value, label: GROUP_LABELS[key] || value.label });
  }).filter(Boolean);
  const temperature = temperatureSamples(benchmark.temperatureSeries);
  const extras = {
    speed: latencyTrials(benchmark.trials, measurements),
    heat: temperature ? temperatureSparkline(temperature) : "",
    memory: memoryBar(measurements),
    startup: startupTimeline(measurements),
  };
  const groups = METRIC_GROUPS.map(([id, title, icon, keys]) => {
    const rows = [...rowsFor(keys), ...(id === "heat" && temperature ? [temperatureRow(temperature)] : [])];
    return rows.length ? { title, markup: metricGroup(id, title, icon, rows, extras[id]) } : null;
  }).filter(Boolean);
  const known = new Set(METRIC_GROUPS.flatMap(([, , , keys]) => keys));
  const otherRows = rowsFor(Object.keys(measurements).filter((key) => !known.has(key)).sort());
  if (otherRows.length) groups.push({ title: "Other", markup: metricGroup("other", "Other measurements", "benchmark", otherRows) });
  const unavailable = Object.entries(measurements)
    .filter(([key, metric]) => !measuredValue(key, metric))
    .map(([key, metric]) => metricDefinition(key, metric).label);
  return { groups, unavailable };
}

function benchmarkDetails(benchmark, payload) {
  const { groups, unavailable } = metricGroups(benchmark);
  const source = [`Result schema v${benchmark.schemaVersion}`, benchmark.tool].filter(Boolean).map(escapeHtml).join(" · ");
  const raw = payload.result_artifact_url
    ? ` · <a href="${escapeHtml(payload.result_artifact_url)}" target="_blank" rel="noreferrer">Raw result ${icons.external}</a>`
    : "";
  return `<div class="metric-groups">${groups.map((group) => group.markup).join("")}</div>
    ${unavailable.length ? `<p class="metric-unavailable">Not reported by this runtime: ${escapeHtml(unavailable.join(", "))}.</p>` : ""}
    <p class="metric-source">${source}${raw}</p>`;
}

// The performance tiles of each component, with the hint shown under each value.
const KPI_METRICS = [
  ["throughput_fps", "Inferences per second"],
  ["latency_avg_ms", "Average per inference"],
  ["avg_power_watts", "Whole board, while running"],
  ["energy_per_inference_joules", "Board power ÷ throughput"],
];

function kpiTile(key, metric, hint) {
  const definition = metricDefinition(key, metric);
  const display = metricDisplay(metric, definition);
  if (!display) return "";
  return `<div class="kpi"><span>${escapeHtml(definition.label)}</span><strong>${escapeHtml(display.value.toFixed(display.digits))}<small>${escapeHtml(display.unit.replace("/inference", ""))}</small></strong><small>${escapeHtml(hint)}</small></div>`;
}

function benchmarkContext(benchmark) {
  return [
    benchmark.workload,
    benchmark.platformVersion && `Platform ${benchmark.platformVersion}`,
    benchmark.neatVersion && `Neat ${benchmark.neatVersion}`,
    Number.isFinite(benchmark.samples) && `${benchmark.samples.toLocaleString()} samples per trial`,
    benchmark.timestamp && `Measured ${formatDate(benchmark.timestamp, false)}`,
  ].filter(Boolean).join(" · ");
}

function componentBenchmark(payload, runId) {
  if (!payload) return `<p class="muted-message">No completed benchmark is linked to this component yet.</p>`;
  const adapted = adaptBenchmarkResult(payload.result);
  if (adapted.kind === "unsupported") {
    return `<p class="muted-message">This benchmark schema (v${escapeHtml(adapted.schemaVersion)}) is newer than this browser can render.</p>${payload.result_artifact_url ? `<a class="external-link" href="${escapeHtml(payload.result_artifact_url)}" target="_blank" rel="noreferrer">Open raw result ${icons.external}</a>` : ""}`;
  }
  if (adapted.kind !== "supported") return "";
  const benchmark = adapted.value;
  const tiles = KPI_METRICS.map(([key, hint]) => kpiTile(key, benchmark.measurements[key], hint)).join("");
  const { groups } = metricGroups(benchmark);
  return `${tiles ? `<div class="kpis">${tiles}</div>` : ""}
    <p class="benchmark-context">${escapeHtml(benchmarkContext(benchmark))}</p>
    ${disclosure(`benchmark:${runId}`, "Benchmark details", benchmarkDetails(benchmark, payload), groups.map((group) => group.title).join(" · "))}`;
}

// A multi-component model shows each component's numbers under its name; they are never combined.
function componentPerformance(detail, named) {
  const run = detail.summary.run;
  return `<div class="component-performance">
    ${named ? `<h4>${escapeHtml(run.metadata?.component_id || "model")}</h4>` : ""}
    ${componentBenchmark(detail.benchmark, run.id)}
  </div>`;
}

function compileDuration(run) {
  return run.metadata?.compile_timing?.scope === "compile_recipe"
    ? formatDuration(run.metadata.compile_timing.started_at, run.metadata.compile_timing.finished_at)
    : "Not recorded";
}

function componentRow(detail, single) {
  const run = detail.summary.run;
  const component = run.metadata?.component_id || "model";
  const artifacts = modelPackArtifacts(detail.summary.artifacts);
  const tests = detail.tests.test_results || [];
  const parameters = detail.provenance.parameters?.parameters?.compile_parameters || {};
  const mpkBytes = artifacts.reduce((total, artifact) => total + (Number.isFinite(artifact.size_bytes) ? artifact.size_bytes : 0), 0);
  const artifactRows = artifacts.length
    ? artifacts.map((artifact) => keyValue("SHA-256", artifact.sha256, { copy: true, mono: true })).join("")
    : keyValue("SHA-256", "No artifact metadata was registered for this run.");
  const meta = [mpkBytes ? `MPK ${formatBytes(mpkBytes)}` : "", parameters.precision].filter(Boolean).join(" · ");
  const body = `<dl class="key-grid">
      ${artifactRows}
      ${keyValue("Run ID", run.id, { copy: true, mono: true })}
      ${keyValue("Fingerprint", run.run_fingerprint, { copy: true, mono: true })}
      ${keyValue("Compile recipe duration", compileDuration(run))}
      ${keyValue("Registry publication duration", formatDuration(run.started_at, run.finished_at))}
    </dl>
    ${fidelityPanel(tests, run.id)}
    ${Object.keys(parameters).length ? disclosure(`parameters:${run.id}`, "Compile parameters", `<pre>${escapeHtml(JSON.stringify(parameters, null, 2))}</pre>`, "JSON") : ""}`;
  return disclosure(`component:${run.id}`, single ? "Files and checksums" : component, body, meta);
}

function percentage(value, { signed = false } = {}) {
  if (!Number.isFinite(value)) return "—";
  const percent = value * 100;
  return `${signed && percent > 0 ? "+" : ""}${percent.toFixed(2)}%`;
}

// Display names for common accuracy metrics; any other metric shows its declared name.
const ACCURACY_METRIC_LABELS = {
  top1_accuracy: "Top-1",
  top5_accuracy: "Top-5",
  coco_map: "mAP 50–95",
  coco_ap50: "AP50",
  coco_ap75: "AP75",
  coco_ap_small: "AP small",
  coco_ap_medium: "AP medium",
  coco_ap_large: "AP large",
  coco_ar1: "AR@1",
  coco_ar10: "AR@10",
  coco_ar100: "AR@100",
  coco_ar_small: "AR small",
  coco_ar_medium: "AR medium",
  coco_ar_large: "AR large",
};
// The first metrics in display order are shown as tiles; any further metrics are listed in a table.
const ACCURACY_SUMMARY_TILES = 4;

// The registry does not preserve a result's metric order, so known metrics follow the label order and
// any others follow them in the order they arrive.
function orderedAccuracyMetrics(report) {
  const known = Object.keys(ACCURACY_METRIC_LABELS);
  const rank = (name) => (known.includes(name) ? known.indexOf(name) : known.length);
  return Object.entries(report.metrics || {}).sort(([left], [right]) => rank(left) - rank(right));
}

function accuracyReport(payload, tests = [], benchmarkTests = [], devkitPayload = null) {
  const fallback = tests.find((item) => item.suite === "accuracy" && item.summary?.schema_version === 1);
  const legacyDevkit = benchmarkTests.find((item) => item.suite === "accuracy" && item.summary?.schema_version === 2);
  return devkitPayload?.result || legacyDevkit?.summary || payload?.result || fallback?.summary;
}

function variantAccuracyReport(variant) {
  return variant.details.map((detail) => accuracyReport(
    detail.accuracy, detail.tests.test_results || [], detail.benchmarkTests?.test_results || [], variant.devkit,
  )).find((report) => [1, 2].includes(report?.schema_version)) || null;
}

// The first accuracy metric of a build and the source model's published score, for the Compare page.
export function accuracyHeadline(variant) {
  const report = variantAccuracyReport(variant);
  if (report?.schema_version === 1) return { label: "Top-1", value: percentage(report.metrics?.candidate?.top1_accuracy), published: "" };
  const [name, metric] = report ? orderedAccuracyMetrics(report)[0] || [] : [];
  if (!name) return null;
  return { label: ACCURACY_METRIC_LABELS[name] || name, value: accuracyMetricValue(metric), published: publishedAccuracyValue(metric) };
}

function accuracyPanel(report) {
  if (report?.schema_version === 2) return accuracySection(report, accuracyMetricTiles(report), accuracyContextV2(report), accuracyMetricTable(report));
  if (!report || report.schema_version !== 1) return "";
  const candidate = report.metrics?.candidate || {};
  const reference = report.metrics?.reference || {};
  const comparison = report.metrics?.comparison || {};
  const evaluation = report.evaluation || {};
  return accuracySection(report, `
      <div class="accuracy-metric primary"><span>Candidate top-1</span><strong>${percentage(candidate.top1_accuracy)}</strong><small>Reference ${percentage(reference.top1_accuracy)}</small></div>
      <div class="accuracy-metric primary"><span>Candidate top-5</span><strong>${percentage(candidate.top5_accuracy)}</strong><small>Reference ${percentage(reference.top5_accuracy)}</small></div>
      <div class="accuracy-metric"><span>Top-1 agreement</span><strong>${percentage(comparison.top1_agreement)}</strong><small>Candidate vs reference</small></div>
      <div class="accuracy-metric"><span>Top-1 delta</span><strong>${percentage(comparison.top1_delta, { signed: true })}</strong><small>Top-5 ${percentage(comparison.top5_delta, { signed: true })}</small></div>
    `, `${keyValue("Profile", evaluation.profile)}${keyValue("Scope", report.report_scope)}${keyValue("Workers", report.execution?.workers)}${keyValue("Throughput", Number.isFinite(report.execution?.samples_per_second) ? `${report.execution.samples_per_second.toFixed(2)} images/s` : null)}`);
}

function accuracyMetricValue(metric) {
  if (!Number.isFinite(metric?.value)) return "—";
  return metric.unit === "ratio" ? percentage(metric.value) : `${metric.value.toFixed(4)} ${metric.unit}`;
}

// A metric's reference is the source model's published score, recorded in the model's accuracy entry.
function publishedAccuracyValue(metric) {
  return Number.isFinite(metric?.reference) ? accuracyMetricValue({ ...metric, value: metric.reference }) : "";
}

function accuracyMetricTiles(report) {
  return orderedAccuracyMetrics(report).slice(0, ACCURACY_SUMMARY_TILES).map(([name, metric], index) => {
    const published = publishedAccuracyValue(metric);
    return `<div class="accuracy-metric${index === 0 ? " primary" : ""}"><span>${escapeHtml(ACCURACY_METRIC_LABELS[name] || name)}</span><strong>${escapeHtml(accuracyMetricValue(metric))}</strong><small>${escapeHtml(published ? `Published ${published}` : "")}</small></div>`;
  }).join("");
}

function accuracyMetricTable(report) {
  const rows = orderedAccuracyMetrics(report).slice(ACCURACY_SUMMARY_TILES).map(([name, metric]) => (
    `<tr><th scope="row">${escapeHtml(ACCURACY_METRIC_LABELS[name] || name)}</th><td>${escapeHtml(accuracyMetricValue(metric))}</td><td>${escapeHtml(publishedAccuracyValue(metric) || "—")}</td></tr>`
  )).join("");
  if (!rows) return "";
  return `<div class="benchmark-measurements"><h4>Additional metrics</h4><div class="benchmark-measurements-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">Value</th><th scope="col">Published</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function accuracyContextV2(report) {
  return `${keyValue("Validation", report.validation?.id)}${keyValue("Measured on", "DevKit")}${keyValue("Core", report.execution?.core_version)}`;
}

function accuracySection(report, metricsMarkup, contextMarkup, tableMarkup = "") {
  const dataset = report.dataset || {};
  const fullCoverage = Number.isInteger(dataset.evaluated_samples)
    && dataset.evaluated_samples === dataset.available_samples;
  // A full run still reads provisional when its validation entry declares it so.
  const qualified = fullCoverage && report.validation?.status !== "provisional";
  const status = qualified ? "Qualification" : "Provisional";
  const statusClass = qualified ? "qualification" : "provisional";
  const statusExplanation = qualified
    ? "This run evaluated the full configured validation split and is intended for final accuracy reporting."
    : fullCoverage
    ? "This run evaluated the full configured validation split, but its validation entry is still marked provisional."
    : "This run evaluated a sampled subset of the validation images. It is useful for routine regression checks; run the qualification profile for final accuracy reporting.";
  const coverageLabel = fullCoverage ? "Full coverage" : "Sampled coverage";
  const coverageExplanation = fullCoverage
    ? "Every image in the configured validation split was evaluated."
    : "A deterministic subset of the configured validation split was evaluated.";
  const sampleText = Number.isInteger(dataset.evaluated_samples) && Number.isInteger(dataset.available_samples)
    ? `${dataset.evaluated_samples.toLocaleString()} of ${dataset.available_samples.toLocaleString()} images`
    : "Sample count unavailable";
  const details = `<p class="disclosure-text"><strong>${status}.</strong> ${escapeHtml(statusExplanation)}</p>
      <p class="disclosure-text"><strong>${coverageLabel}.</strong> ${escapeHtml(coverageExplanation)}</p>
      <dl class="key-grid">${contextMarkup}</dl>
      ${tableMarkup}`;
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    <div class="detail-section-title"><h3>Accuracy</h3><span class="status-badge ${statusClass}">${status}</span></div>
    <p class="accuracy-scope"><strong>${escapeHtml(sampleText)}</strong> · ${escapeHtml(dataset.id || "Dataset unavailable")} ${escapeHtml(dataset.variant || "")} / ${escapeHtml(dataset.split || "")}</p>
    <div class="accuracy-summary">${metricsMarkup}</div>
    ${disclosure("accuracy", "Evaluation details", details, coverageLabel)}
  </section>`;
}

function fidelityPanel(tests, runId) {
  const result = tests.find((item) =>
    item.suite === "correctness"
    && item.name === "numerical-fidelity"
    && item.summary?.schema_version === 1
  );
  const report = result?.summary;
  if (!report) return "";
  const metrics = (report.metrics || []).map((metric) => `
    <div class="benchmark-metric"><span>${escapeHtml(metric.name)} · Cosine similarity</span><strong>${Number(metric.cosine_similarity).toFixed(4)}</strong></div>
    <div class="benchmark-metric"><span>${escapeHtml(metric.name)} · Normalized RMSE</span><strong>${Number(metric.normalized_rmse).toFixed(4)}</strong></div>
  `).join("");
  const scope = report.scope === "full_pipeline" ? "Full pipeline" : "Component";
  return `<div class="fidelity-panel">
    <div class="group-title"><h4>Numerical fidelity</h4><button class="action-button" type="button" data-download="numerical-fidelity" data-run-id="${escapeHtml(runId)}">${icons.download}<span>Download JSON</span></button></div>
    <p class="fidelity-scope"><strong>${escapeHtml(scope)}</strong> · ${escapeHtml(report.status)} · ${Number(report.samples).toLocaleString()} samples · ${escapeHtml(report.dataset?.id || "Dataset unavailable")} ${escapeHtml(report.dataset?.variant || "")}</p>
    <div class="benchmark-summary">${metrics}</div>
    <dl class="key-grid">${keyValue("Reference", report.reference)}${keyValue("Candidate", report.candidate)}</dl>
    <p class="muted-message">Output similarity, not task accuracy. Output reordering can affect these metrics.</p>
  </div>`;
}

function licensePanel(license) {
  if (!license?.spdx) return "";
  const redistribution = license.redistribution ? `Redistribution ${license.redistribution}` : "Redistribution not recorded";
  return `<section id="license-section" class="license-section" tabindex="-1"><div class="detail-section-title"><h3>License</h3></div>
    <p class="license-line"><strong>${escapeHtml(license.spdx)}</strong><span>${escapeHtml(redistribution)}</span></p>
    ${license.attribution ? `<p class="license-attribution">${escapeHtml(license.attribution)}</p>` : ""}
  </section>`;
}

function documentationKey(documentation) {
  return `${documentation.run_id}:${documentation.artifact_sha256}`;
}

// The README's title, hero, and Overview are already on the page, so the full card starts at the section
// after Overview. The registry rejects a README without that opening.
function modelCardRemainder(markdown) {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  const overview = lines.findIndex((line) => /^##\s+Overview\s*$/.test(line));
  const next = overview < 0 ? -1 : lines.findIndex((line, index) => index > overview && /^##\s/.test(line));
  return next < 0 ? null : lines.slice(next).join("\n");
}

function modelCardBody(documentation) {
  const markdown = state.documentationCache.get(documentationKey(documentation));
  if (markdown === undefined) return `<p class="muted-message" data-model-card-pending>Loading model card…</p>`;
  const remainder = modelCardRemainder(markdown);
  const heroUrl = documentationHeroUrl(documentation);
  return `<div class="model-documentation-body">${renderMarkdown(remainder ?? markdown, {
    stripTitle: remainder === null,
    resolveImage: (source) => (source === documentation.hero?.markdown_path ? heroUrl : null),
    resolveLink: (source) => resolveDocumentationLink(documentation, source),
  })}</div>`;
}

function overviewSection(model) {
  const documentation = model?.documentation;
  if (!documentation?.overview_markdown) return "";
  const heroUrl = documentationHeroUrl(documentation);
  const hero = heroUrl
    ? `<figure class="model-card-hero"><img src="${escapeHtml(heroUrl)}" alt="${escapeHtml(documentation.hero.alt_text || `${model.display_name || model.model_id} model hero`)}" loading="lazy" decoding="async" /></figure>`
    : "";
  const fullCard = documentation.run_id && documentation.artifact_sha256
    ? disclosure("model-card", "Full model card", modelCardBody(documentation), "README")
    : "";
  return `<section id="overview-section" class="overview-section" tabindex="-1" aria-labelledby="overview-title">
    <div class="detail-section-title"><h3 id="overview-title">Overview</h3></div>
    ${hero}
    <div class="model-documentation-body">${renderMarkdown(documentation.overview_markdown, { resolveLink: (source) => resolveDocumentationLink(documentation, source) })}</div>
    ${fullCard}
  </section>`;
}

// Fetches the README behind an open Full model card row, verifies it against the catalog checksum, and
// fills the row. A row that already shows the card is left alone.
export async function loadModelCard() {
  const variant = currentVariant();
  const documentation = variant ? catalogModelForRun(variant.anchor)?.documentation : null;
  const body = () => elements["detail-pane"].querySelector('[data-disclosure="model-card"] > .disclosure-body');
  if (!documentation || !body()?.querySelector("[data-model-card-pending]")) return;
  const requestId = ++state.documentationRequestId;
  try {
    const markdown = await apiText(`/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(documentation.artifact_name || "readme")}/content`);
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(markdown)))]
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (digest !== documentation.artifact_sha256) throw new Error("The model card checksum did not match the catalog.");
    state.documentationCache.set(documentationKey(documentation), markdown);
  } catch (error) {
    if (requestId !== state.documentationRequestId) return;
    const target = body();
    if (target) target.innerHTML = `<p class="muted-message" data-model-card-pending>The model card could not be loaded. Close and reopen this row to try again.</p>`;
    showToast(error.message, true);
    return;
  }
  if (requestId !== state.documentationRequestId || state.selectedVariantKey !== variant.key) return;
  const target = body();
  if (target) target.innerHTML = modelCardBody(documentation);
}

// Variant tabs switch between the latest builds in the current catalog, so a build reached from
// history shows none.
// The catalog model whose current build is this variant, or null for an older or history-only build.
function catalogModelOfVariant(variant) {
  const model = catalogMode() ? state.models.find((item) => item.model_id === variant.modelId) : null;
  return model?.builds.some((build) => variant.runs.some((run) => run.id === build.run_id)) ? model : null;
}

function detailBuildSwitcher(variant) {
  const model = catalogModelOfVariant(variant);
  const inCatalog = Boolean(model);
  const profiles = inCatalog ? profileGroups({ ...model, builds: compilerFilteredBuilds(model) }) : [];
  const tabs = profiles.map(([profile]) => {
    const selected = profile === variant.profile;
    return `<button class="detail-variant-tab" type="button" role="tab" data-detail-model-id="${escapeHtml(model.model_id)}" data-detail-profile-id="${escapeHtml(profile)}" aria-selected="${selected}" tabindex="${selected ? "0" : "-1"}">${escapeHtml(profile)}</button>`;
  }).join("");
  return `<section class="detail-build-switcher" aria-label="Variant selection">
    <strong>${profiles.length > 1 ? "Variants" : "Variant"}</strong>
    ${profiles.length ? `<div class="detail-variant-tabs" role="tablist" aria-label="${escapeHtml(model.display_name || model.model_id)} variants">${tabs}</div>` : `<span class="detail-variant-single">${escapeHtml(profileLabel(variant.anchor))}</span>`}
  </section>`;
}

function uniqueBy(items, key) {
  return [...new Map(items.map((item) => [key(item), item])).values()];
}

function targetName(run) {
  const target = run.target_platform || "";
  return target ? target.charAt(0).toUpperCase() + target.slice(1) : "the target";
}

function cliDownloadCommand(variant) {
  return `sima-cli models download --id ${variant.modelId} --variant ${variant.profile} --branch ${branchName(variant.anchor.git_ref)}`;
}

function sectionNavigation(sections) {
  return `<nav class="subnav" aria-label="Model page sections">${sections.map(([id, label]) => (
    `<button type="button" data-scroll-to="${id}">${label}</button>`
  )).join("")}</nav>`;
}

// The package is the glance view of Details: its size and checksum, and the two ways to download it.
function packageCard(variant, fullPackage, packageArtifact, hasArtifacts) {
  const run = variant.anchor;
  const components = hasArtifacts
    ? `<button class="action-button" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">${icons.download}<span>Download components</span></button>`
    : "";
  if (!fullPackage) return components ? `<div class="package-actions">${components}</div>` : "";
  return `<div class="package-card">
    <div class="package-card-copy"><strong>Full package</strong><span>All components and declared runtime resources in one verified ZIP.</span></div>
    ${packageArtifact ? `<dl class="package-facts"><div><dt>Size</dt><dd>${escapeHtml(formatBytes(packageArtifact.size_bytes))}</dd></div><div><dt>SHA-256</dt><dd class="artifact-hash"><code>${escapeHtml(packageArtifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(packageArtifact.sha256)}" aria-label="Copy package checksum">${icons.copy}</button></dd></div></dl>` : ""}
    <div class="package-actions"><button class="action-button" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>Download package</span></button>${components}</div>
  </div>`;
}

export function renderDetail(variant) {
  showView("model");
  const run = variant.anchor;
  const details = variant.details;
  const allTests = details.flatMap((detail) => detail.tests.test_results || []);
  const inputs = uniqueBy(details.flatMap((detail) => detail.provenance.inputs || []), (input) => input.sha256 || input.name);
  const model = variant.modelId || run.run_type;
  const catalogModel = catalogModelForRun(run);
  const displayName = catalogModel?.display_name || model;
  const precisions = uniqueBy(details.map((detail) => detail.provenance.parameters?.parameters?.compile_parameters?.precision)
    .filter(Boolean), (precision) => precision);
  const categories = run.model_categories || {};
  const license = run.metadata?.license;
  const sourceUrl = githubModelUrl(run, model);
  const sourceLabel = `View ${model} source on GitHub at commit ${shortSha(run.commit_sha)}`;
  const accuracyMarkup = accuracyPanel(variantAccuracyReport(variant));
  const overviewMarkup = overviewSection(catalogModel);
  const hasArtifacts = details.some((detail) => modelPackArtifacts(detail.summary.artifacts).length);
  // Every component's MPK must pass validation for the build to carry the mark.
  const validated = details.every((detail) => mpkValidationMark(detail.tests.test_results || []));
  const fullPackage = variantPackage(variant.runs);
  const packageArtifact = (variant.packageSummary?.artifacts || []).find((item) => item.artifact_type === "model-package");
  const huggingFaceSource = inputs.map(huggingFaceSourceLink).find(Boolean);
  const builtAt = Math.max(...variant.runs.map((item) => Date.parse(item.finished_at || item.created_at || "") || 0));
  const singleComponent = details.length === 1;
  const backLabel = elements["result-title"].textContent || "Models";
  const downloadAction = fullPackage
    ? `<button class="download-action" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>Download</span>${packageArtifact ? `<small>${escapeHtml(formatBytes(packageArtifact.size_bytes))}</small>` : ""}</button>`
    : hasArtifacts ? `<button class="download-action" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">${icons.download}<span>Download</span></button>` : "";
  const sections = [
    overviewMarkup && ["overview-section", "Overview"],
    ["components-section", "Performance"],
    accuracyMarkup && ["accuracy-section", "Accuracy"],
    ["details-section", "Details"],
    license?.spdx && ["license-section", "License"],
  ].filter(Boolean);

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">${escapeHtml(backLabel)}</button><span aria-hidden="true">/</span><span aria-current="page">${escapeHtml(displayName)}</span></nav>
    <header class="detail-title">
      <div class="detail-title-copy">
        <h2>${escapeHtml(displayName)}</h2>
        <div class="detail-tags">${(categories.tasks || []).map((task) => detailTag(icons.task, taxonomyLabel("tasks", task), "task")).join("")}${categories.architecture?.id ? detailTag(icons.architecture, taxonomyLabel("architectures", categories.architecture.id), "architecture") : ""}${categories.model_type ? detailTag(icons.modelType, taxonomyLabel("model_types", categories.model_type), "model-type") : ""}${precisions.length ? detailTag(icons.precision, precisions.join(" + "), "precision") : ""}${detailTag(icons.compiler, `${run.toolchain_name || "compiler"} ${run.toolchain_version || ""}`.trim(), "compiler")}${detailTag(icons.branch, branchName(run.git_ref), "branch")}${license?.spdx ? detailTag(icons.license, license.spdx, "license") : ""}</div>
        ${catalogModel?.description ? `<p class="lede">${escapeHtml(catalogModel.description)}</p>` : ""}
      </div>
      <div class="detail-title-actions">
        ${downloadAction}
        ${catalogModelOfVariant(variant) ? `<button class="cli-action" type="button" data-copy="${escapeHtml(cliDownloadCommand(variant))}">${icons.copy}<span>Copy CLI command</span></button>` : ""}
        <div class="detail-links">${validated ? mpkValidationMark(allTests) : ""}${huggingFaceSource ? `<a class="huggingface-source-link" href="${escapeHtml(huggingFaceSource.href)}" target="_blank" rel="noreferrer" aria-label="Open source ONNX model card on Hugging Face" data-tooltip="Open source ONNX model card on Hugging Face">${icons.huggingFace}</a>` : ""}${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" data-tooltip="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}</div>
      </div>
    </header>
    ${detailBuildSwitcher(variant)}
    ${sectionNavigation(sections)}

    ${overviewMarkup}

    <section id="components-section" class="component-section" tabindex="-1"><div class="detail-section-title"><h3>Performance on ${escapeHtml(targetName(run))}</h3></div>
      ${singleComponent ? "" : `<p class="section-lede">This model runs as ${details.length} components, each measured on its own.</p>`}
      ${details.map((detail) => componentPerformance(detail, !singleComponent)).join("")}
      ${singleComponent ? "" : `<p class="section-note">Each component is measured on its own. The CPU stage between components is not measured, so component latencies do not add up to end-to-end latency.</p>`}
    </section>

    ${accuracyMarkup}

    <section id="details-section" class="details-section" tabindex="-1"><div class="detail-section-title"><h3>Details</h3></div>
      ${packageCard(variant, fullPackage, packageArtifact, hasArtifacts)}
      <div class="disclosure-list">
        ${disclosure("provenance", "Build provenance", `<dl class="key-grid">
          ${keyValue("Commit", run.commit_sha, { copy: true, mono: true })}
          ${keyValue("Branch", branchName(run.git_ref))}
          ${keyValue("Built", formatDate(builtAt || run.created_at))}
          ${keyValue("Target", run.target_platform)}
          ${keyValue("Compiler", run.toolchain_version)}
        </dl>${run.ci_run_url ? `<a class="external-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">Open GitHub workflow ${icons.external}</a>` : ""}`, `${shortSha(run.commit_sha).slice(0, 7)} · ${branchName(run.git_ref)}`)}
        ${sourceInputPanel(inputs)}
        ${details.map((detail) => componentRow(detail, singleComponent)).join("")}
      </div>
    </section>

    ${licensePanel(license)}
  </div>`;
  if (state.openDisclosures.has("model-card")) loadModelCard();
}

export function errorPanel(message, title = "Registry unavailable", { closable = false } = {}) {
  return `<div class="error-state">${closable ? `<nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">Back to models</button></nav>` : ""}<span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}
