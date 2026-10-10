// Model detail panel: performance, accuracy, documentation, provenance, and license.
import { adaptBenchmarkResult } from "./benchmark-adapters.js";
import { renderMarkdown, resolveDocumentationLink } from "./markdown.js";
import { state, elements, catalogMode, currentVariant } from "./state.js";
import { icons, escapeHtml, branchName, shortSha, profileLabel, showToast } from "./ui.js";
import {
  METRIC_DEFINITIONS, SUMMARY_METRICS, metricDefinition, metricDisplay, formatDate, formatDuration,
  formatBytes,
} from "./metrics.js";
import {
  apiText, profileGroups, variantPackage, modelPackArtifacts, catalogModelForRun,
  compilerFilteredBuilds,
} from "./api.js";
import { compareCheckbox } from "./compare-view.js";

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
  return `<section><div class="detail-section-title"><h3>Source input</h3></div>${inputs.map((input) => `<div class="source-block">
    <strong>${escapeHtml(input.name)}</strong>
    <p>${escapeHtml(input.source_reference || input.input_type)}</p>
    <div><span>${escapeHtml(formatBytes(input.size_bytes))}</span><code>${escapeHtml(input.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(input.sha256)}" aria-label="Copy source checksum">${icons.copy}</button></div>
  </div>`).join("")}</section>`;
}

export function detailSkeleton() {
  elements["detail-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span><span></span></div>`;
}

export function showEmptyDetail() {
  elements["detail-pane"].innerHTML = `<div class="detail-empty"><h2>No model selected</h2><p>Select a model to choose a variant, inspect provenance, and download a verified build.</p></div>`;
}

function keyValue(label, value, { copy = false, mono = false } = {}) {
  const safeValue = escapeHtml(value ?? "—");
  return `<div class="key-value"><dt>${escapeHtml(label)}</dt><dd class="${mono ? "mono" : ""}"><span title="${safeValue}">${safeValue}</span>${copy && value ? `<button type="button" class="copy-button" data-copy="${safeValue}" aria-label="Copy ${escapeHtml(label)}">${icons.copy}</button>` : ""}</dd></div>`;
}

function detailTag(icon, value, kind) {
  return `<span class="detail-tag detail-tag-${kind}">${icon}<span>${escapeHtml(value)}</span></span>`;
}

function metricCard(key, metric) {
  const definition = metricDefinition(key, metric);
  const display = metricDisplay(metric, definition);
  const value = display ? `${display.value.toFixed(display.digits)} ${display.unit}` : "Not available";
  const reason = !display && metric?.reason ? ` title="${escapeHtml(metric.reason.replaceAll("_", " "))}"` : "";
  return `<div class="benchmark-metric"${reason}><span>${escapeHtml(definition.label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function availableTrialMetrics(trials) {
  const fields = [...new Set(trials.flatMap((trial) => Object.keys(trial.metrics)))];
  return fields.filter((field) => (
    trials.filter((trial) => Number.isFinite(trial.metrics[field])).length >= 2
  )).sort((left, right) => {
    const knownLeft = Object.keys(METRIC_DEFINITIONS).indexOf(left);
    const knownRight = Object.keys(METRIC_DEFINITIONS).indexOf(right);
    if (knownLeft >= 0 || knownRight >= 0) return (knownLeft < 0 ? 1000 : knownLeft) - (knownRight < 0 ? 1000 : knownRight);
    return left.localeCompare(right);
  });
}

export function trialChart(trials, measurements = {}) {
  const available = availableTrialMetrics(trials);
  if (!available.length) return "";
  const field = available.includes(state.trialMetric) ? state.trialMetric : available[0];
  const definition = metricDefinition(field, measurements[field]);
  const measuredTrials = trials
    .filter((trial) => Number.isFinite(trial.metrics[field]))
    .map((trial) => ({ ...trial, value: trial.metrics[field] * definition.scale }));
  const values = measuredTrials.map((trial) => trial.value);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const spread = maximum - minimum || Math.max(maximum * 0.05, 0.1);
  const low = Math.max(0, minimum - spread * 0.2);
  const high = maximum + spread * 0.2;
  const width = 640;
  const left = 70;
  const right = 28;
  const top = 28;
  const row = 34;
  const height = top + trials.length * row + 30;
  const x = (value) => left + ((value - low) / (high - low)) * (width - left - right);
  const grid = [0, .25, .5, .75, 1].map((fraction) => {
    const position = left + fraction * (width - left - right);
    const label = low + fraction * (high - low);
    return `<line x1="${position}" y1="${top - 8}" x2="${position}" y2="${height - 25}"/><text x="${position}" y="${height - 7}">${label.toFixed(2)}</text>`;
  }).join("");
  const points = measuredTrials.map((trial, index) => {
    const y = top + index * row;
    return `<text class="trial-label" x="0" y="${y + 4}">Trial ${trial.number}</text><circle cx="${x(trial.value)}" cy="${y}" r="6"/><text class="trial-value" x="${Math.min(x(trial.value) + 11, width - 55)}" y="${y + 4}">${trial.value.toFixed(definition.digits)}</text>`;
  }).join("");
  const options = available.map((metric) => (
    `<option value="${escapeHtml(metric)}"${metric === field ? " selected" : ""}>${escapeHtml(metricDefinition(metric, measurements[metric]).label)}</option>`
  )).join("");
  const direction = definition.direction === "neutral" ? "" : ` · ${definition.direction} is better`;
  return `<figure class="trial-chart benchmark-chart"><figcaption><span class="chart-title">Trial comparison</span><label class="chart-control"><span>Metric</span><select data-trial-metric>${options}</select></label></figcaption><p class="chart-note">${escapeHtml(definition.unit + direction)}</p><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(definition.label)} in ${escapeHtml(definition.unit)} for ${measuredTrials.length} benchmark trials"><g class="chart-grid">${grid}</g><g class="chart-points">${points}</g></svg><table class="sr-only"><caption>${escapeHtml(definition.label)} by benchmark trial</caption><thead><tr><th>Trial</th><th>${escapeHtml(definition.label)} in ${escapeHtml(definition.unit)}</th></tr></thead><tbody>${measuredTrials.map((trial) => `<tr><td>${trial.number}</td><td>${trial.value}</td></tr>`).join("")}</tbody></table></figure>`;
}

function measurementTable(measurements) {
  const entries = Object.entries(measurements)
    .filter(([key]) => !SUMMARY_METRICS.includes(key))
    .sort(([left], [right]) => {
      const knownLeft = Object.keys(METRIC_DEFINITIONS).indexOf(left);
      const knownRight = Object.keys(METRIC_DEFINITIONS).indexOf(right);
      if (knownLeft >= 0 || knownRight >= 0) return (knownLeft < 0 ? 1000 : knownLeft) - (knownRight < 0 ? 1000 : knownRight);
      return left.localeCompare(right);
    });
  if (!entries.length) return "";
  const rows = entries.map(([key, metric]) => {
    const definition = metricDefinition(key, metric);
    const display = metricDisplay(metric, definition);
    const reason = metric.reason ? metric.reason.replaceAll("_", " ") : metric.status;
    const value = display ? `${display.value.toFixed(display.digits)} ${display.unit}` : reason;
    return `<tr><th scope="row">${escapeHtml(definition.label)}</th><td>${escapeHtml(value)}</td><td>${escapeHtml(metric.aggregation)}</td><td>${metric.sampleCount}</td></tr>`;
  }).join("");
  return `<div class="benchmark-measurements"><h4>Additional measurements</h4><div class="benchmark-measurements-wrap"><table><thead><tr><th scope="col">Measurement</th><th scope="col">Value</th><th scope="col">Statistic</th><th scope="col">Samples</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function temperatureChart(series) {
  if (!series.length) return "";
  const hottest = (candidates) => [...candidates].sort((left, right) => (
    Math.max(...right.samples.map((sample) => sample.celsius))
    - Math.max(...left.samples.map((sample) => sample.celsius))
  ))[0];
  const temp1 = series.filter((item) => item.sensor.toLowerCase() === "temp1");
  const selected = hottest(temp1.length ? temp1 : series);
  const samples = selected.samples;
  const temperatures = samples.map((sample) => sample.celsius);
  const startTime = Math.min(...samples.map((sample) => sample.timestamp));
  const endTime = Math.max(...samples.map((sample) => sample.timestamp));
  const elapsed = Math.max(endTime - startTime, 1000);
  const minimum = Math.min(...temperatures);
  const maximum = Math.max(...temperatures);
  const spread = maximum - minimum || 1;
  const low = minimum - Math.max(spread * .25, .5);
  const high = maximum + Math.max(spread * .25, .5);
  const width = 640;
  const height = 190;
  const left = 48;
  const right = 24;
  const top = 18;
  const bottom = 35;
  const x = (timestamp) => left + ((timestamp - startTime) / elapsed) * (width - left - right);
  const y = (celsius) => top + ((high - celsius) / (high - low)) * (height - top - bottom);
  const grid = [0, .5, 1].map((fraction) => {
    const celsius = low + (1 - fraction) * (high - low);
    const position = top + fraction * (height - top - bottom);
    return `<line x1="${left}" y1="${position}" x2="${width - right}" y2="${position}"/><text class="temperature-axis-label" x="${left - 8}" y="${position + 4}">${celsius.toFixed(1)}°</text>`;
  }).join("");
  const path = samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.timestamp)} ${y(sample.celsius)}`).join(" ");
  const points = samples.map((sample) => {
    const label = sample.trial ? `Trial ${sample.trial}` : sample.phase;
    return `<circle cx="${x(sample.timestamp)}" cy="${y(sample.celsius)}" r="5"><title>${escapeHtml(label)}: ${sample.celsius.toFixed(2)} °C</title></circle>`;
  }).join("");
  const ticks = [0, .5, 1].map((fraction) => {
    const seconds = (elapsed * fraction) / 1000;
    const position = left + fraction * (width - left - right);
    return `<text x="${position}" y="${height - 8}">${seconds.toFixed(0)}s</text>`;
  }).join("");
  const legacy = selected.legacy && samples.length === 2;
  return `<figure class="temperature-chart benchmark-chart"><figcaption><span class="chart-title">Temperature ${legacy ? "change" : "over benchmark"}</span><span class="chart-series-label">${escapeHtml(selected.label)}</span></figcaption><p class="chart-note">${legacy ? "Start/end snapshots only" : `${samples.length} samples`} · ${minimum.toFixed(1)}–${maximum.toFixed(1)} °C</p><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Temperature in degrees Celsius for ${escapeHtml(selected.label)}"><g class="chart-grid">${grid}${ticks}</g><path class="temperature-line" d="${path}"/><g class="temperature-points">${points}</g></svg><table class="sr-only"><caption>Temperature samples for ${escapeHtml(selected.label)}</caption><thead><tr><th>Elapsed time</th><th>Temperature in degrees Celsius</th></tr></thead><tbody>${samples.map((sample) => `<tr><td>${((sample.timestamp - startTime) / 1000).toFixed(1)} seconds</td><td>${sample.celsius}</td></tr>`).join("")}</tbody></table></figure>`;
}

// The component row shows these measurements; its expanded body shows the rest.
const ROW_METRICS = ["latency_avg_ms", "throughput_fps", "avg_power_watts"];

function componentBenchmark(payload) {
  if (!payload) return { benchmark: null, body: `<p class="muted-message">No completed benchmark is linked to this component yet.</p>` };
  const adapted = adaptBenchmarkResult(payload.result);
  if (adapted.kind === "unsupported") {
    return { benchmark: null, body: `<p class="muted-message">This benchmark schema (v${escapeHtml(adapted.schemaVersion)}) is newer than this browser can render.</p>${payload.result_artifact_url ? `<a class="external-link" href="${escapeHtml(payload.result_artifact_url)}">Open raw result ${icons.external}</a>` : ""}` };
  }
  if (adapted.kind !== "supported") return { benchmark: null, body: "" };
  const benchmark = adapted.value;
  const summary = SUMMARY_METRICS
    .filter((key) => !ROW_METRICS.includes(key) && benchmark.measurements[key])
    .map((key) => metricCard(key, benchmark.measurements[key]))
    .join("");
  return { benchmark, body: `<div class="component-benchmark"><h4>Benchmark <span>Schema v${benchmark.schemaVersion}</span></h4>
    ${summary ? `<div class="benchmark-summary">${summary}</div>` : ""}
    ${trialChart(benchmark.trials, benchmark.measurements)}
    ${temperatureChart(benchmark.temperatureSeries)}
    ${measurementTable(benchmark.measurements)}
    <dl class="benchmark-context">${keyValue("Workload", benchmark.workload)}${keyValue("Samples / trial", benchmark.samples)}${keyValue("Platform", benchmark.platformVersion)}${keyValue("Neat", benchmark.neatVersion)}${keyValue("Measured", formatDate(benchmark.timestamp))}</dl>
  </div>` };
}

function componentStat(label, value) {
  return `<span class="component-stat"><small>${escapeHtml(label)}</small>${escapeHtml(value)}</span>`;
}

function rowMetric(benchmark, key) {
  const metric = benchmark?.measurements[key];
  const display = metricDisplay(metric, metricDefinition(key, metric));
  return display ? `${display.value.toFixed(display.digits)} ${display.unit}` : "—";
}

function compileDuration(run) {
  return run.metadata?.compile_timing?.scope === "compile_recipe"
    ? formatDuration(run.metadata.compile_timing.started_at, run.metadata.compile_timing.finished_at)
    : "Not recorded";
}

function componentRow(detail, open) {
  const run = detail.summary.run;
  const component = run.metadata?.component_id || "model";
  const artifacts = modelPackArtifacts(detail.summary.artifacts);
  const tests = detail.tests.test_results || [];
  const parameters = detail.provenance.parameters?.parameters?.compile_parameters || {};
  const { benchmark, body } = componentBenchmark(detail.benchmark);
  const mpkBytes = artifacts.reduce((total, artifact) => total + (Number.isFinite(artifact.size_bytes) ? artifact.size_bytes : 0), 0);
  const artifactRows = artifacts.length
    ? artifacts.map((artifact) => keyValue("SHA-256", artifact.sha256, { copy: true, mono: true })).join("")
    : keyValue("SHA-256", "No artifact metadata was registered for this run.");
  return `<details class="component-row" data-component-run-id="${escapeHtml(run.id)}"${open ? " open" : ""}>
    <summary><strong>${escapeHtml(component)}</strong>${componentStat("MPK", mpkBytes ? formatBytes(mpkBytes) : "—")}${ROW_METRICS.map((key) => componentStat(metricDefinition(key).label, rowMetric(benchmark, key))).join("")}</summary>
    <div class="component-body">
      <dl class="key-grid">
        ${artifactRows}
        ${keyValue("Run ID", run.id, { copy: true, mono: true })}
        ${keyValue("Fingerprint", run.run_fingerprint, { copy: true, mono: true })}
        ${keyValue("Compile recipe duration", compileDuration(run))}
        ${keyValue("Registry publication duration", formatDuration(run.started_at, run.finished_at))}
      </dl>
      ${body}
      ${fidelityPanel(tests, run.id)}
      ${Object.keys(parameters).length ? `<details><summary>Compile parameters <span>JSON</span></summary><pre>${escapeHtml(JSON.stringify(parameters, null, 2))}</pre></details>` : ""}
    </div>
  </details>`;
}

function percentage(value, { signed = false } = {}) {
  if (!Number.isFinite(value)) return "—";
  const percent = value * 100;
  return `${signed && percent > 0 ? "+" : ""}${percent.toFixed(2)}%`;
}

// Display names for common accuracy metrics; any other metric shows its declared name.
const ACCURACY_METRIC_LABELS = {
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

function accuracyPanel(payload, tests = [], benchmarkTests = [], devkitPayload = null) {
  const fallback = tests.find((item) => item.suite === "accuracy" && item.summary?.schema_version === 1);
  const legacyDevkit = benchmarkTests.find((item) => item.suite === "accuracy" && item.summary?.schema_version === 2);
  const report = devkitPayload?.result || legacyDevkit?.summary || payload?.result || fallback?.summary;
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
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    <div class="detail-section-title"><h3>Accuracy</h3><details class="accuracy-explainer"><summary class="accuracy-status ${statusClass}" aria-label="Explain ${status.toLowerCase()} accuracy status">${status}</summary><div class="accuracy-explainer-card"><strong>${status}</strong><p>${escapeHtml(statusExplanation)}</p><p><strong>${coverageLabel}.</strong> ${escapeHtml(coverageExplanation)}</p></div></details></div>
    <p class="accuracy-scope"><strong>${escapeHtml(sampleText)}</strong> · ${escapeHtml(dataset.id || "Dataset unavailable")} ${escapeHtml(dataset.variant || "")} / ${escapeHtml(dataset.split || "")}</p>
    <div class="accuracy-summary">${metricsMarkup}</div>
    ${tableMarkup}
    <dl class="accuracy-context">${contextMarkup}</dl>
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
  return `<section class="fidelity-section">
    <div class="detail-section-title"><h3>Numerical fidelity</h3><button class="text-button" type="button" data-download="numerical-fidelity" data-run-id="${escapeHtml(runId)}">Download JSON</button></div>
    <p class="fidelity-scope"><strong>${escapeHtml(scope)}</strong> · ${escapeHtml(report.status)} · ${Number(report.samples).toLocaleString()} samples · ${escapeHtml(report.dataset?.id || "Dataset unavailable")} ${escapeHtml(report.dataset?.variant || "")}</p>
    <div class="benchmark-summary">${metrics}</div>
    <p class="muted-message">Output similarity, not task accuracy. Output reordering can affect these metrics.</p>
    <details><summary>Evaluation details</summary><dl>${keyValue("Reference", report.reference)}${keyValue("Candidate", report.candidate)}</dl></details>
  </section>`;
}

function licensePanel(license) {
  if (!license?.spdx) return "";
  const redistribution = license.redistribution
    ? license.redistribution.charAt(0).toUpperCase() + license.redistribution.slice(1)
    : "Unknown";
  return `<section class="license-section"><div class="detail-section-title"><h3>License</h3><span>${escapeHtml(license.spdx)}</span></div><dl class="key-grid">
    ${keyValue("SPDX", license.spdx, { copy: true, mono: true })}
    ${keyValue("Redistribution", redistribution)}
  </dl>${license.attribution ? `<p class="license-attribution"><strong>Attribution</strong>${escapeHtml(license.attribution)}</p>` : ""}</section>`;
}

function documentationHeroUrl(documentation) {
  const hero = documentation?.hero;
  if (!hero?.artifact_name || !hero?.artifact_sha256 || !documentation?.run_id) return null;
  return `${state.apiBase}/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(hero.artifact_name)}/content`;
}

function documentationPanel(model) {
  const documentation = model?.documentation;
  if (!documentation?.overview_markdown || !documentation?.run_id || !documentation?.artifact_sha256) return "";
  const expanded = state.expandedDocumentationModelId === model.model_id;
  const cacheKey = `${documentation.run_id}:${documentation.artifact_sha256}`;
  const fullMarkdown = state.documentationCache.get(cacheKey);
  const heroUrl = documentationHeroUrl(documentation);
  const resolveImage = (source) => source === documentation.hero?.markdown_path ? heroUrl : null;
  const resolveLink = (source) => resolveDocumentationLink(documentation, source);
  const overviewHero = heroUrl
    ? `<figure class="model-card-hero"><img src="${escapeHtml(heroUrl)}" alt="${escapeHtml(documentation.hero.alt_text || `${model.display_name || model.model_id} model hero`)}" loading="lazy" decoding="async" /></figure>`
    : "";
  const body = expanded && fullMarkdown
    ? renderMarkdown(fullMarkdown, { stripTitle: true, resolveImage, resolveLink })
    : `${overviewHero}${renderMarkdown(documentation.overview_markdown, { resolveLink })}`;
  const label = expanded ? "Show overview only" : "Read full model card";
  return `<section class="model-documentation${expanded ? " expanded" : ""}" aria-labelledby="model-documentation-title">
    <div class="detail-section-title"><h3 id="model-documentation-title">Model card</h3>${expanded ? `<button class="documentation-collapse" type="button" data-documentation-toggle aria-expanded="true" aria-controls="model-documentation-body">${label}</button>` : ""}</div>
    <div id="model-documentation-body" class="model-documentation-body">${body}</div>
    <div class="documentation-disclosure${expanded ? " expanded" : ""}">
      <button type="button" data-documentation-toggle aria-expanded="${expanded}" aria-controls="model-documentation-body">${label}<span aria-hidden="true">${expanded ? "↑" : "↓"}</span></button>
    </div>
  </section>`;
}

export async function toggleDocumentation(button) {
  const variant = currentVariant();
  const model = variant ? catalogModelForRun(variant.anchor) : null;
  const documentation = model?.documentation;
  if (!model || !documentation) return;
  if (state.expandedDocumentationModelId === model.model_id) {
    state.expandedDocumentationModelId = null;
    renderDetail(variant);
    elements["detail-pane"].querySelector("[data-documentation-toggle]")?.focus();
    return;
  }
  const cacheKey = `${documentation.run_id}:${documentation.artifact_sha256}`;
  const requestId = ++state.documentationRequestId;
  button.disabled = true;
  button.textContent = "Loading model card…";
  try {
    if (!state.documentationCache.has(cacheKey)) {
      const markdown = await apiText(`/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(documentation.artifact_name || "readme")}/content`);
      const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(markdown)))]
        .map((byte) => byte.toString(16).padStart(2, "0")).join("");
      if (digest !== documentation.artifact_sha256) throw new Error("The model card checksum did not match the catalog.");
      state.documentationCache.set(cacheKey, markdown);
    }
    if (requestId !== state.documentationRequestId || state.selectedVariantKey !== variant.key) return;
    state.expandedDocumentationModelId = model.model_id;
    renderDetail(variant);
    elements["detail-pane"].querySelector(".documentation-collapse")?.focus();
  } catch (error) {
    if (requestId === state.documentationRequestId) {
      button.disabled = false;
      button.textContent = "Read full model card";
      showToast(error.message, true);
    }
  }
}

// Variant tabs switch between the latest builds in the current catalog, so a build reached from
// history shows none.
function detailBuildSwitcher(variant) {
  const model = catalogMode() ? state.models.find((item) => item.model_id === variant.modelId) : null;
  const inCatalog = model?.builds.some((build) => variant.runs.some((run) => run.id === build.run_id));
  const profiles = inCatalog ? profileGroups({ ...model, builds: compilerFilteredBuilds(model) }) : [];
  const compared = state.compareSelection.has(variant.key);
  const tabs = profiles.map(([profile]) => {
    const selected = profile === variant.profile;
    return `<button class="detail-variant-tab" type="button" role="tab" data-detail-model-id="${escapeHtml(model.model_id)}" data-detail-profile-id="${escapeHtml(profile)}" aria-selected="${selected}" tabindex="${selected ? "0" : "-1"}"><span>${escapeHtml(profile)}</span>${selected ? "<b>Current</b>" : ""}</button>`;
  }).join("");
  return `<section class="detail-build-switcher" aria-label="Variant selection">
    ${profiles.length ? `<div class="detail-switcher-heading"><strong>Variants</strong><span>${profiles.length} variant${profiles.length === 1 ? "" : "s"}</span></div>
    <div class="detail-variant-tabs" role="tablist" aria-label="${escapeHtml(model.display_name || model.model_id)} variants">${tabs}</div>` : `<div class="detail-switcher-heading"><strong>Variant</strong><span>${escapeHtml(profileLabel(variant.anchor))}</span></div>`}
    <label class="detail-compare-select compare-select" title="${compared ? "Remove from" : "Add to"} comparison">
      ${compareCheckbox(variant)}
      <span aria-hidden="true">✓</span><b>${compared ? "Selected for comparison" : "Add variant to comparison"}</b>
    </label>
  </section>`;
}

function uniqueBy(items, key) {
  return [...new Map(items.map((item) => [key(item), item])).values()];
}

export function renderDetail(variant) {
  state.variantGroups.set(variant.key, variant);
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
  const accuracyMarkup = details.map((detail) => accuracyPanel(
    detail.accuracy, detail.tests.test_results || [], detail.benchmarkTests?.test_results || [], variant.devkit,
  )).find(Boolean) || "";
  const hasBenchmark = details.some((detail) => detail.benchmark);
  const hasArtifacts = details.some((detail) => modelPackArtifacts(detail.summary.artifacts).length);
  // Every component's MPK must pass validation for the build to carry the mark.
  const validated = details.every((detail) => mpkValidationMark(detail.tests.test_results || []));
  const fullPackage = variantPackage(variant.runs);
  const packageArtifact = (variant.packageSummary?.artifacts || []).find((item) => item.artifact_type === "model-package");
  const huggingFaceSource = inputs.map(huggingFaceSourceLink).find(Boolean);
  const builtAt = Math.max(...variant.runs.map((item) => Date.parse(item.finished_at || item.created_at || "") || 0));
  const singleComponent = details.length === 1;

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <div class="detail-topline"><button class="detail-layout-toggle" type="button" aria-pressed="${state.detailLayout === "focused"}" title="Give the selected model more space">${state.detailLayout === "focused" ? "Show wider list" : "Focus model"}</button><button class="close-detail" type="button" aria-label="Close details">×</button></div>
    <div class="detail-title"><div class="detail-title-copy"><p class="kicker">${escapeHtml(run.target_platform || "Model artifact")}</p><h2>${escapeHtml(displayName)}</h2>${displayName !== model ? `<p>${escapeHtml(model)}</p>` : ""}</div><div class="detail-title-actions">${fullPackage ? `<button class="artifact-download-shortcut" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}" aria-label="Download full package" data-tooltip="Download full package">${icons.download}</button>` : hasArtifacts ? `<button class="artifact-download-shortcut" type="button" data-download-variant="${run.id}" data-run-id="${run.id}" aria-label="Download individual components" data-tooltip="Download individual components">${icons.download}</button>` : ""}${validated ? mpkValidationMark(allTests) : ""}${hasBenchmark ? `<button class="benchmark-shortcut" type="button" data-scroll-benchmark aria-label="Jump to component benchmarks" data-tooltip="Jump to component benchmarks">${icons.benchmark}</button>` : ""}${huggingFaceSource ? `<a class="huggingface-source-link" href="${escapeHtml(huggingFaceSource.href)}" target="_blank" rel="noreferrer" aria-label="Open source ONNX model card on Hugging Face" data-tooltip="Open source ONNX model card on Hugging Face">${icons.huggingFace}</a>` : ""}${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" data-tooltip="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}</div></div>
    ${detailBuildSwitcher(variant)}
    <div class="detail-tags">${precisions.length ? detailTag(icons.precision, precisions.join(" + "), "precision") : ""}${detailTag(icons.compiler, `${run.toolchain_name || "compiler"} ${run.toolchain_version || ""}`.trim(), "compiler")}${detailTag(icons.branch, branchName(run.git_ref), "branch")}${license?.spdx ? detailTag(icons.license, license.spdx, "license") : ""}${categories.model_type ? detailTag(icons.modelType, categories.model_type, "model-type") : ""}${categories.architecture?.id ? detailTag(icons.architecture, categories.architecture.id, "architecture") : ""}${(categories.tasks || []).map((task) => detailTag(icons.task, task, "task")).join("")}</div>

    ${documentationPanel(catalogModel)}

    ${fullPackage ? `<section class="artifact-section"><div class="detail-section-title"><h3>Full Package</h3><button class="text-button" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">Download ZIP</button></div>${packageArtifact ? `<dl class="artifact-metadata"><div><dt>File size</dt><dd>${escapeHtml(formatBytes(packageArtifact.size_bytes))}</dd></div><div><dt>SHA-256</dt><dd class="artifact-hash"><code>${escapeHtml(packageArtifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(packageArtifact.sha256)}" aria-label="Copy package checksum">${icons.copy}</button></dd></div></dl>` : ""}<p class="muted-message">Includes all components and declared runtime resources.</p><button class="text-button" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">Download individual components</button></section>` : ""}

    <section><div class="detail-section-title"><h3>Build provenance</h3></div><dl class="key-grid">
      ${keyValue("Commit", run.commit_sha, { copy: true, mono: true })}
      ${keyValue("Branch", branchName(run.git_ref))}
      ${keyValue("Built", formatDate(builtAt || run.created_at))}
      ${keyValue("Target", run.target_platform)}
      ${keyValue("Compiler", run.toolchain_version)}
    </dl>${run.ci_run_url ? `<a class="external-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">Open GitHub workflow ${icons.external}</a>` : ""}</section>

    ${sourceInputPanel(inputs)}

    ${licensePanel(license)}

    ${accuracyMarkup}

    <section id="components-section" class="component-section" tabindex="-1"><div class="detail-section-title"><h3>Components</h3><span>${details.length}</span></div>
      <div class="component-list">${details.map((detail) => componentRow(detail, singleComponent || state.expandedRuns.has(detail.summary.run.id))).join("")}</div>
      ${singleComponent ? "" : `<p class="muted-message">Each component is measured on its own. The CPU stage between components is not measured, so component latencies do not add up to end-to-end latency.</p>`}
    </section>
  </div>`;
  elements["detail-pane"].querySelector('.detail-variant-tab[aria-selected="true"]')
    ?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

export function errorPanel(message, title = "Registry unavailable", { closable = false } = {}) {
  return `<div class="error-state">${closable ? `<button class="close-detail" type="button" aria-label="Close details">×</button>` : ""}<span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}
