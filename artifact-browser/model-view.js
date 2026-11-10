// Model page: performance, accuracy, model card, build details, and license.
import {
  REFERENCE_ORIGINS, accuracyDisplay, accuracyMetricLabel, formatAccuracy, orderedMetricNames, referenceNotes, referenceOrigin,
  referenceScores,
} from "./accuracy-metrics.js";
import { selectLayers, layerTimeScale, layerFamily, LAYER_FAMILIES } from "./mla-layer-stats.js";
import { MLA_PEAK_INT8_TOPS, matchMlaStages, mlaPeakTops, mlaStageSimulation, mlaUtilizationMetrics, sustainedEffectiveTops } from "./mla-utilization.js";
import { readAccuracy, readPerformance } from "./model-report.js";
import { renderMarkdown, resolveDocumentationLink } from "./markdown.js";
import { customerView, state, elements, catalogMode, currentVariant, showView, taxonomyLabel } from "./state.js";
import { icons, escapeHtml, branchName, shortSha, profileLabel, variantLabel, precisionLabel, showToast, disclosure } from "./ui.js";
import {
  metricDefinition, metricDisplay, formatDate, formatDuration,
  formatBytes,
} from "./metrics.js";
import {
  apiText, profileGroups, variantPackage, modelPackArtifacts, catalogModelForRun,
  compilerFilteredBuilds, documentationHeroUrl, documentationHeroFallbackUrl,
} from "./api.js";
import { packageFormat, artifactDelivery, deploymentSize, downloadOffered, huggingFaceLocation } from "./downloads.js";

function githubModelUrl(run, modelId, sourceCommit = run.commit_sha) {
  const organization = String(state.repository.organization || "");
  const repository = String(state.repository.name || "");
  const configuredRepository = `${organization}/${repository}`.toLowerCase();
  const runRepository = String(run.repository || "").toLowerCase();
  const commit = String(sourceCommit || "");
  if (!/^[A-Za-z0-9_.-]+$/.test(organization)
    || !/^[A-Za-z0-9_.-]+$/.test(repository)
    || runRepository !== configuredRepository
    || !/^[a-f0-9]{40}$/i.test(commit)
    || !/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test(modelId)) return null;
  return `https://github.com/${encodeURIComponent(organization)}/${encodeURIComponent(repository)}/tree/${commit}/models/${encodeURIComponent(modelId)}`;
}

// A reused build is measured by later runs, so each result names the commit and workflow that measured
// it, appended to the result's context line.
function measurementSource(run, modelId) {
  const commit = String(run?.commit_sha || "");
  if (!/^[a-f0-9]{40}$/i.test(commit)) return "";
  const sourceUrl = githubModelUrl(run, modelId);
  const commitMarkup = sourceUrl
    ? `<a class="measurement-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="View the ${escapeHtml(modelId)} source that measured this result, at commit ${escapeHtml(shortSha(commit))}">${escapeHtml(commit.slice(0, 7))}${icons.external}</a>`
    : escapeHtml(commit.slice(0, 7));
  const workflow = run.ci_run_url
    ? ` · <a class="measurement-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">GitHub workflow${icons.external}</a>`
    : "";
  return ` · Commit ${commitMarkup}${workflow}`;
}

function artifactValidationMark(tests = []) {
  const passed = tests.some((test) => ["mpk-validation", "deployment-validation"].includes(test.name) && test.status === "passed");
  if (!passed) return "";
  const label = tests.some((test) => test.name === "deployment-validation") ? "Model deployment" : "MPK package";
  const message = `${label} integrity verified: file structure, checksum, and build provenance match.`;
  return `<span class="mpk-validation-mark" tabindex="0" role="img" aria-label="${message}" data-tooltip="${message}">✓</span>`;
}

function trustedHttpsUrl(value = "") {
  try {
    const url = new URL(String(value));
    return url.protocol === "https:" && !url.username && !url.password ? url : null;
  } catch {
    return null;
  }
}

function huggingFaceRepositoryUrl(sourceReference = "") {
  const match = /^hf:\/\/(models|datasets|spaces)\/([^/@?#]+)\/([^/@?#]+)@[a-f0-9]{40}\/.+$/i.exec(String(sourceReference));
  if (!match) return null;
  const [, repositoryType, owner, repository] = match;
  const typePrefix = repositoryType === "models" ? "" : `${repositoryType}/`;
  return new URL(`https://huggingface.co/${typePrefix}${encodeURIComponent(owner)}/${encodeURIComponent(repository)}`);
}

// The original model is the declared `upstream.model_card`, whatever ONNX was compiled; a pinned
// Hugging Face source without a card falls back to its repository.
function originalModelLink(input = {}) {
  const url = trustedHttpsUrl(input.metadata?.model_card) || huggingFaceRepositoryUrl(input.source_reference);
  if (!url) return null;
  const huggingFace = url.hostname === "huggingface.co";
  return {
    href: url.href,
    huggingFace,
    label: `Open the original model on ${huggingFace ? "Hugging Face" : url.hostname}`,
  };
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
  hideExecutionTooltip();
  showView("model");
  elements["detail-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span><span></span></div>`;
}

export function showEmptyDetail() {
  hideExecutionTooltip();
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

function measuredValue(key, metric) {
  const definition = metricDefinition(key);
  const display = metricDisplay(metric, definition);
  return display && { label: definition.label, text: display.value.toFixed(display.digits), unit: display.unit };
}

function metricRow({ label, text, unit }) {
  return `<div class="metric-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(text)}<small>${escapeHtml(unit)}</small></dd></div>`;
}

// The headline metrics, one number each.
const KPI_METRICS = ["throughput_fps", "latency_avg_ms", "avg_power_watts", "energy_per_inference_joules"];

// The SoC draws on two SoM rails: the MLA's own and the one for the rest of the SoC.
const SOC_RAILS = ["power_rail_mla_watts", "power_rail_soc_ddr_vdd_watts"];
// Names for the SoM power rails Sentinel reports, what each rail supplies, and its colour in the power bar.
// The SoC rails come first: the power bar outlines them as one block, and the SoC rows name them.
// Both are red, the SoC's colour; Non-MLA is striped so it stays distinct from the MLA.
const RAILS = {
  power_rail_mla_watts: ["MLA core (0.68 V)", "Supplies the MLA.", "var(--red)"],
  power_rail_soc_ddr_vdd_watts: ["Non-MLA SoC (VDD)", "Supplies the SoC logic outside the MLA, the memory controller, and DDR activity.", "repeating-linear-gradient(135deg, var(--red) 0 2px, color-mix(in srgb, var(--red) 25%, var(--surface-raised)) 2px 4px)"],
  power_rail_soc_ddr_vddq_watts: ["DRAM I/O (VDDQ)", "Supplies the SoC to DRAM interface.", "var(--amber)"],
  power_rail_soc_ddr_vdd2h_watts: ["DRAM (VDD2H)", "Supplies part of the DRAM.", "var(--violet)"],
  power_rail_pcie_eth_watts: ["PCIe and Ethernet PHY", "Supplies the PCIe and Ethernet PHYs.", "var(--green)"],
  power_rail_hdmi_watts: ["HDMI PHY (1.2 V)", "Supplies the HDMI PHY.", "var(--ink-soft)"],
  power_rail_platform_1v8_watts: ["SoM platform (1.8 V)", "Supplies other SoM circuits.", "color-mix(in srgb, var(--blue) 50%, var(--surface-raised))"],
  power_rail_platform_3v3_watts: ["SoM platform (3.3 V)", "Supplies other SoM circuits.", "color-mix(in srgb, var(--green) 50%, var(--surface-raised))"],
};
// Performance details: one row per reading, with its label, its metrics, and a short explanation shown
// behind an (i) mark, one line per value when a row has several. A row names a rail when its colour should match that rail in the power bar,
// and a depth when it is part of the row above it.
const DETAIL_CARDS = [
  ["performance", "Performance", "gauge", [], [
    ["Throughput", ["throughput_fps"], "Inferences completed per second, several inputs in flight."],
    ["Latency (mean)", ["latency_avg_ms"], "Mean time for one inference."],
    ["Latency (p50 / p95 / p99)", ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"], "p50: 50% of inferences are faster.\np95: 95% are faster.\np99: 99% are faster."],
  ]],
  ["heat", "Heat", "thermometer", ["Average", "Peak"], [
    ["SoC", ["avg_die_temperature_c", "peak_die_temperature_c"], "Hottest SoC die sensor."],
    ["SoM", ["avg_board_temperature_c", "peak_board_temperature_c"], "Hottest SoM board sensor."],
  ]],
  ["cpu", "CPU", "compiler", ["Average", "Peak"], [
    ["All cores", ["avg_cpu_usage_pct", "peak_cpu_usage_pct"], "Mean load across all cores."],
    ["Busiest core", ["avg_cpu_core_usage_pct", "peak_cpu_core_usage_pct"], "Load on the most loaded core."],
  ]],
  ["power", "Power", "power", ["Average", "Peak"], [
    ["SoM", ["avg_power_watts", "peak_power_watts"], "Sum of all SoM rails."],
    ["SoC", "soc", "MLA plus Non-MLA.\nPeak: sum of their peaks, an upper bound.", null, 1],
    ["MLA", ["avg_mla_power_watts", "peak_mla_power_watts"], "MLA core rail (0.68 V).", "power_rail_mla_watts", 2],
    ["Non-MLA", "power_rail_soc_ddr_vdd_watts", "SoC and DDR VDD rail: the SoC outside the MLA.", null, 2],
    ...Object.keys(RAILS).filter((key) => !SOC_RAILS.includes(key)).map((key) => [RAILS[key][0], key, RAILS[key][1], null, 1]),
  ]],
  ["memory", "Memory", "memory", ["Average", "Peak"], [
    ["MLA", ["mla_memory_avg_mib", "mla_memory_peak_mib"], "Buffers the MLA holds: weights, working memory, and inputs only the MLA reads."],
    ["EV74", ["ev74_memory_avg_mib", "ev74_memory_peak_mib"], "Buffers EV74 holds, such as the frames it preprocesses for the MLA.\nMLA plus EV74 is the model's accelerator memory."],
    ["Linux (board)", ["avg_linux_memory_mib", "peak_linux_memory_mib"], "Memory the whole board uses, every process included.\nIt includes EV74 buffers, so do not add it to the rows above."],
  ]],
];
// Performance, heat, and CPU fill the left column; power, memory, and any other measurements the right.
const RIGHT_COLUMN_CARDS = new Set(["power", "memory", "other"]);
// The tiles and cards place these metrics; any other measured metric is listed under Other measurements.
const PLACED_METRICS = new Set([
  ...KPI_METRICS,
  ...DETAIL_CARDS.flatMap(([, , , , rows]) => rows.flatMap(([, keys]) => (Array.isArray(keys) ? keys : []))),
]);

function kpiTile(key, metric) {
  const value = measuredValue(key, metric);
  return value ? `<div class="kpi"><span>${escapeHtml(value.label)}</span><strong>${escapeHtml(value.text)}<small>${escapeHtml(value.unit.replace("/inference", ""))}</small></strong></div>` : "";
}

function infoMark(text) {
  return text ? `<span class="info-mark" tabindex="0" role="img" aria-label="${escapeHtml(text)}" data-tooltip="${escapeHtml(text)}">i</span>` : "";
}

// A label's trailing "(...)" qualifier, such as "Latency (p50 / p95 / p99)", is set in quieter text after the name.
function readingLabel(label) {
  const [, name, qualifier] = /^(.*?)\s*\(([^()]+)\)$/.exec(label) || [null, label, ""];
  return `${escapeHtml(name)}${qualifier ? `<span class="qualifier">(${escapeHtml(qualifier)})</span>` : ""}`;
}

function readingCell(value) {
  return `<td>${value ? `${escapeHtml(value.text)}<small>${escapeHtml(value.unit)}</small>` : "—"}</td>`;
}

function powerRails(measurements) {
  return (measurements.peak_power_watts?.details?.breakdown || []).filter((rail) => Number.isFinite(rail.mean));
}

// The average and peak of one SoM rail, or of the SoC rails together, from the rail details of peak SoM power.
// Rails peak at different moments, so the sum of their peaks is only an upper bound on the SoC peak.
function railReadings(measurements, key) {
  const keys = key === "soc" ? SOC_RAILS : [key];
  const rails = powerRails(measurements).filter((rail) => keys.includes(rail.key));
  if (rails.length !== keys.length) return [];
  const sum = (field) => rails.reduce((total, rail) => total + rail[field], 0);
  const reading = (value) => Number.isFinite(value) && { text: value.toFixed(2), unit: "W" };
  return [reading(sum("mean")), reading(sum("maximum"))];
}

function railDot(key) {
  if (key === "soc") return `<i class="rail-dot rail-dot-soc" aria-hidden="true"></i>`;
  return RAILS[key] ? `<i class="rail-dot" style="background:${RAILS[key][2]}" aria-hidden="true"></i>` : "";
}

// A card without columns shows a row's readings in one cell, such as latency percentiles: "36.0 / 37.8 / 39.6 ms".
function joinedCell(values) {
  const unit = values.find(Boolean)?.unit || "";
  return `<td>${values.map((value) => escapeHtml(value ? value.text : "—")).join(" / ")}<small>${escapeHtml(unit)}</small></td>`;
}

// The SoM rails in RAILS order, with their names, explanations, and colours.
function namedRails(measurements) {
  const known = Object.keys(RAILS);
  return powerRails(measurements)
    .sort((a, b) => (known.indexOf(a.key) + 1 || 99) - (known.indexOf(b.key) + 1 || 99))
    .map((rail) => ({ ...rail, name: RAILS[rail.key]?.[0] || rail.label, info: RAILS[rail.key]?.[1], colour: RAILS[rail.key]?.[2] || "var(--ink-faint)" }));
}

// The SoM power each rail draws on average, as one bar split by rail. The SoC rails sit in one outlined block
// that matches the SoC row; the table rows below the bar name every rail.
function railSplit(measurements) {
  const rails = namedRails(measurements).filter((rail) => rail.mean > 0);
  const total = rails.reduce((sum, rail) => sum + rail.mean, 0);
  if (rails.length < 2 || total <= 0) return "";
  const segment = (rail) => `<span style="flex-grow:${rail.mean.toFixed(4)};background:${rail.colour}"></span>`;
  const soc = rails.filter((rail) => SOC_RAILS.includes(rail.key));
  const others = rails.filter((rail) => !SOC_RAILS.includes(rail.key));
  const socBlock = soc.length
    ? `<span class="rail-soc" style="flex-grow:${soc.reduce((sum, rail) => sum + rail.mean, 0).toFixed(4)}">${soc.map(segment).join("")}</span>`
    : "";
  const segments = socBlock + others.map(segment).join("");
  return `<div class="rail-split" role="img" title="Average SoM power by rail" aria-label="Average SoM power by rail: ${escapeHtml(rails.map((rail) => `${rail.name} ${rail.mean.toFixed(2)} W`).join(", "))}">${segments}</div>`;
}

// The load of each CPU core: a bar for the average and a tick for the peak, in two columns of cores.
function coreLoad(cores) {
  if (cores.length < 2) return "";
  const busiest = cores.reduce((top, core, index) => (core.mean > cores[top].mean ? index : top), 0);
  const percent = (value) => Math.min(100, Math.max(0, value)).toFixed(1);
  const rows = cores.map((core, index) => {
    const peak = Number.isFinite(core.maximum) ? `<i style="left:${percent(core.maximum)}%"></i>` : "";
    const summary = `Core ${index}: ${core.mean.toFixed(1)}% average${Number.isFinite(core.maximum) ? `, ${core.maximum.toFixed(1)}% peak` : ""}`;
    return `<li${index === busiest ? ' class="busiest"' : ""} title="${summary}" aria-label="${summary}"><span>C${index}</span><span class="core-track"><b style="width:${percent(core.mean)}%"></b>${peak}</span><span>${core.mean.toFixed(0)}%</span></li>`;
  }).join("");
  return `<p class="metric-caption core-legend"><span><b></b>Average</span><span><i></i>Peak</span></p><ul class="core-load" style="--core-rows:${Math.ceil(cores.length / 2)}">${rows}</ul>`;
}

function detailCard(measurements, [id, title, icon, columns, readings], { note, lead, extra } = {}) {
  const rows = readings
    .map(([label, keys, info, rail, depth]) => ({
      label, info, depth, rail: rail || (Array.isArray(keys) ? null : keys),
      values: Array.isArray(keys) ? keys.map((key) => measuredValue(key, measurements[key])) : railReadings(measurements, keys),
    }))
    .filter((row) => row.values.some(Boolean))
    .map((row) => `<tr${row.depth ? ` class="depth-${row.depth}"` : ""}><th scope="row">${railDot(row.rail)}${readingLabel(row.label)}${infoMark(row.info)}</th>${columns.length ? row.values.map(readingCell).join("") : joinedCell(row.values)}</tr>`);
  if (!rows.length) return null;
  const head = columns.length ? `<thead><tr><td></td>${columns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join("")}</tr></thead>` : "";
  return {
    id,
    title,
    markup: `<div class="detail-card"><h4><i aria-hidden="true">${icons[icon]}</i>${escapeHtml(title)}${note ? `<small>${escapeHtml(note)}</small>` : ""}</h4>${lead || ""}<table>${head}<tbody>${rows.join("")}</tbody></table>${extra || ""}</div>`,
  };
}

// Names the board telemetry source, or why the board readings are missing.
function telemetryText(telemetry) {
  if (!telemetry) return "";
  if (telemetry.status !== "ok") return `Unavailable: ${telemetry.status || "unknown reason"}`;
  const interval = telemetry.intervalMs ? ` every ${(telemetry.intervalMs / 1000).toFixed(0)} s` : "";
  return `Sentinel ${telemetry.sentinelVersion} · ${telemetry.samples} samples${interval}`;
}

function performanceDetails(performance, run) {
  const { measurements } = performance;
  const additions = {
    cpu: { note: performance.cpuCores.length > 0 && `${performance.cpuCores.length} cores`, extra: coreLoad(performance.cpuCores) },
    power: { lead: railSplit(measurements) },
  };
  const cards = DETAIL_CARDS.map((card) => detailCard(measurements, card, additions[card[0]])).filter(Boolean);
  const otherRows = Object.keys(measurements).filter((key) => !PLACED_METRICS.has(key)).sort()
    .map((key) => measuredValue(key, measurements[key])).filter(Boolean).map(metricRow);
  if (otherRows.length) {
    cards.push({ id: "other", title: "Other", markup: `<div class="detail-card"><h4><i aria-hidden="true">${icons.performance}</i>Other measurements</h4><dl>${otherRows.join("")}</dl></div>` });
  }
  const column = (right) => `<div class="detail-column">${cards
    .filter((card) => RIGHT_COLUMN_CARDS.has(card.id) === right)
    .map((card) => card.markup).join("")}</div>`;
  const provenance = [
    ["Telemetry", telemetryText(performance.telemetry)],
    ["Runtime", performance.tool],
  ].filter(([, text]) => text).map(([term, text]) => `<div><dt>${term}</dt><dd>${escapeHtml(text)}</dd></div>`).join("");
  const raw = `<button class="performance-action" type="button" data-download="model-report" data-run-id="${escapeHtml(run.id)}">${icons.download}Raw result</button>`;
  const csv = `<a class="performance-action" href="${escapeHtml(csvDataUrl(performanceCsv(performance)))}" download="${escapeHtml(csvFileName(performance))}">${icons.download}Download CSV</a>`;
  return {
    titles: cards.map((card) => card.title),
    markup: `<div class="detail-cards">${column(false)}${column(true)}</div>
    <footer class="performance-footer"><dl class="performance-provenance">${provenance}</dl><div class="performance-actions">${raw}${csv}</div></footer>`,
  };
}

// The CSV download: every reading of the model as one table. Context comes first, then each card's
// metrics, every SoM rail, every CPU core, the remaining measurements, and the MLA layers of each component.
const CSV_EXTRA_METRICS = { power: ["energy_per_inference_joules"] };

function performanceCsv(performance) {
  const rows = [["Section", "Metric", "Value", "Unit", "Key", "Note"]];
  const add = (section, metric, value, unit = "", key = "", note = "") => rows.push([section, metric, value ?? "", unit, key, note]);
  const telemetry = performance.telemetry;
  const seconds = (milliseconds) => Number.isFinite(milliseconds) && (milliseconds / 1000).toFixed(1);
  [
    ["Model", performance.modelId], ["Build", performance.buildProfileId], ["Workload", performance.workload],
    ["CPU cores", performance.cpuCores.length], ["Measured", performance.timestamp], ["Board telemetry", telemetry?.status],
    ["Sentinel", telemetry?.sentinelVersion], ["Telemetry samples", telemetry?.samples],
    ["Telemetry interval", seconds(telemetry?.intervalMs), "s"], ["Runtime", performance.tool],
  ].filter(([, value]) => value || value === 0).forEach(([name, value, unit]) => add("Context", name, value, unit));
  const metric = (section, key) => {
    const shown = measuredValue(key, performance.measurements[key]);
    if (shown) add(section, shown.label, shown.text, shown.unit, key);
  };
  const placed = new Set();
  DETAIL_CARDS.forEach(([id, title, , , cardRows]) => {
    [...cardRows.flatMap(([, keys]) => (Array.isArray(keys) ? keys : [])), ...(CSV_EXTRA_METRICS[id] || [])]
      .forEach((key) => { placed.add(key); metric(title, key); });
  });
  namedRails(performance.measurements).forEach((rail) => {
    add("Power rails", `${rail.name}, average`, rail.mean.toFixed(2), "W", rail.key);
    if (Number.isFinite(rail.maximum)) add("Power rails", `${rail.name}, peak`, rail.maximum.toFixed(2), "W", rail.key);
  });
  performance.cpuCores.forEach((core, index) => {
    add("CPU cores", `Core ${index}, average`, core.mean.toFixed(1), "%");
    if (Number.isFinite(core.maximum)) add("CPU cores", `Core ${index}, peak`, core.maximum.toFixed(1), "%");
  });
  Object.keys(performance.measurements).filter((key) => !placed.has(key)).sort().forEach((key) => metric("Other", key));
  const named = performance.components.length > 1;
  performance.components.forEach(({ id, layerAnalysis }) => {
    layerAnalysis?.layers.forEach((layer) => add(
      named ? `${humanize(id)} MLA layers` : "MLA layers", layer.name, layer.timeUs, "µs",
      `stage:${layer.stage}/layer:${layer.order}`, "Separate profiling pass; compiler order",
    ));
  });
  const cell = (value) => (/[",\r\n]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value));
  return rows.map((row) => row.map(cell).join(",")).join("\r\n");
}

// The byte order mark lets spreadsheet applications read the file as UTF-8, for units such as °C.
function csvDataUrl(csv) {
  return `data:text/csv;charset=utf-8,${encodeURIComponent(`\uFEFF${csv}\r\n`)}`;
}

function csvFileName(performance) {
  const name = [performance.modelId, performance.buildProfileId].filter(Boolean).join("-") || "model";
  return `${name.replace(/[^A-Za-z0-9._-]+/g, "_")}-performance.csv`;
}

function performanceContext(performance) {
  return [
    performance.workload,
    performance.timestamp && `Measured ${formatDate(performance.timestamp, false)}`,
  ].filter(Boolean).join(" · ");
}

const EXECUTION_STAGE_INFO = {
  Preproc: ["Prepare input", "Prepares input data for the model."],
  Cast: ["Convert input", "Changes the input data type."],
  Quant: ["Quantize input", "Converts input values to the model’s integer format."],
  Tess: ["Arrange input", "Rearranges input data for the accelerator."],
  CastTess: ["Convert & Arrange Input", "Changes the input data type and prepares its layout for the accelerator."],
  QuantTess: ["Quantize & Arrange Input", "Converts input values to integers and prepares their layout for the accelerator."],
  "Compiled Model": ["Model Inference", "Runs the compiled model."],
  Detess: ["Restore output layout", "Rearranges the model output for further processing."],
  Dequant: ["Dequantize output", "Converts integer output values back to floating point."],
  DetessCast: ["Convert & arrange output", "Restores the output layout and changes its data type."],
  DetessDequant: ["Prepare output", "Restores the output layout and converts values back to floating point."],
  SimaBoxDecode: ["Decode output", "Converts raw model outputs into usable results."],
};
const PROCESSOR_LABELS = { EV74: "Vector Processor", MLA: "Machine Learning Accelerator", A65: "Arm CPU" };

function executionStageInfo(step) {
  const [title, description] = EXECUTION_STAGE_INFO[step.name] || [step.name, "Runs this processing step."];
  const processor = [step.runsOn, PROCESSOR_LABELS[step.runsOn]].filter(Boolean).join(" · ");
  return `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p>
    ${step.runsOn ? `<span class="execution-tooltip-processor" data-runs-on="${escapeHtml(step.runsOn)}">${escapeHtml(processor)}</span>` : ""}`;
}

let executionTooltipStage = null;
let executionTooltipTimer;

export function hideExecutionTooltip() {
  clearTimeout(executionTooltipTimer);
  executionTooltipStage?.removeAttribute("aria-describedby");
  executionTooltipStage = null;
  elements["execution-tooltip"].hidden = true;
}

export function keepExecutionTooltip() {
  clearTimeout(executionTooltipTimer);
}

export function deferHideExecutionTooltip() {
  executionTooltipTimer = setTimeout(hideExecutionTooltip, 150);
}

export function showExecutionTooltip(stage) {
  hideExecutionTooltip();
  const tooltip = elements["execution-tooltip"];
  tooltip.innerHTML = stage.querySelector("template").innerHTML;
  tooltip.dataset.runsOn = stage.dataset.runsOn || "";
  tooltip.hidden = false;
  executionTooltipStage = stage;
  stage.setAttribute("aria-describedby", tooltip.id);
  const anchor = stage.getBoundingClientRect();
  const roomAbove = Math.max(0, anchor.top - 84);
  const roomBelow = Math.max(0, innerHeight - anchor.bottom - 24);
  tooltip.style.maxHeight = `${Math.max(roomAbove, roomBelow)}px`;
  const box = tooltip.getBoundingClientRect();
  tooltip.style.left = `${Math.max(12, Math.min(innerWidth - box.width - 12, anchor.left + anchor.width / 2 - box.width / 2))}px`;
  tooltip.style.top = `${box.height <= roomAbove + .5 ? anchor.top - box.height - 12 : anchor.bottom + 12}px`;
}

// Each step of the benchmarked Model and the processor it ran on. Details opens its tensor table.
function executionGraphDiagram(graph, runId) {
  if (!graph) return "";
  const steps = graph.steps.map((step) => step.runsOn
    ? `<li><button type="button" class="execution-step" data-execution-stage data-runs-on="${escapeHtml(step.runsOn)}">${escapeHtml(step.name)}<small>${escapeHtml(step.runsOn)}</small><template>${executionStageInfo(step)}</template></button></li>`
    : `<li class="execution-end"><span class="execution-step">${escapeHtml(step.name)}</span></li>`).join("");
  const rows = graph.tensors.map((tensor) => `<tr${tensor.connection ? "" : ' class="continued"'}><th scope="row">${escapeHtml(tensor.connection)}</th>${[tensor.dtype, tensor.shape, tensor.scale, tensor.zeroPoint].map((value) => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("");
  const table = `<div class="tensor-table"><table><thead><tr>${["Connection", "Type", "Shape", "Scale", "Zero point"].map((label) => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const key = `tensors:${runId}`;
  const open = state.openDisclosures.has(key);
  const details = graph.tensors.length
    ? `<button type="button" class="execution-details" data-details="${escapeHtml(key)}" aria-controls="${escapeHtml(key)}" aria-expanded="${open}">Details${icons.chevron}</button>`
    : "";
  return `<section class="execution-graph" aria-label="Execution graph"><div class="execution-graph-row"><ol>${steps}</ol>${details}</div>
    ${graph.tensors.length ? `<div id="${escapeHtml(key)}"${open ? "" : " hidden"}>${table}</div>` : ""}</section>`;
}

const layerTime = (value) => value.toLocaleString(undefined, { maximumFractionDigits: 2 });

function layerAnalysisResults(analysis, view) {
  const { layers, totalUs, count } = selectLayers(analysis, view);
  const scaleUs = layerTimeScale(analysis);
  const label = (layer) => `<th scope="row" class="layer-name" title="${escapeHtml(layer.name)}">${escapeHtml(layer.name.replace(/^MLA_\d+\//, ""))}<small>Stage ${layer.stage} · #${layer.order}</small></th>`;
  const rows = layers.map((layer) => `<tr class="layer-family-${layerFamily(layer.name)}" data-layer-family="${layerFamily(layer.name)}">
    ${label(layer)}
    <td><div class="layer-timing"><span class="layer-timing-number">${layerTime(layer.timeUs)}</span><span class="layer-bar-track" aria-hidden="true"><span style="width:${layer.timeUs / scaleUs * 100}%"></span></span></div></td>
    <td class="layer-share">${(totalUs > 0 ? layer.timeUs / totalUs * 100 : 0).toFixed(2)}%</td>
  </tr>`).join("");
  const families = new Set(analysis.layers.map((layer) => layerFamily(layer.name)));
  const legend = Object.entries(LAYER_FAMILIES).filter(([id]) => families.has(id))
    .map(([id, title]) => `<span class="layer-family-${id}" data-layer-family="${id}"><i aria-hidden="true"></i><span class="layer-legend-label" data-label="${escapeHtml(title)}"><span>${title}</span></span></span>`).join("");
  const axis = `<span class="layer-axis" aria-label="Scale: 0 to ${layerTime(scaleUs)} microseconds"><span>0</span><span>${layerTime(scaleUs / 2)}</span><span>${layerTime(scaleUs)}</span></span>`;
  return `<p class="layer-results" role="status">${layers.length.toLocaleString()} of ${count.toLocaleString()} layers · ${view.order === "compiler" ? "Runtime order" : "Slowest layers first"}</p>
    <div class="layer-legend" aria-label="Layer types inferred from compiled names"><span>Layer type (from name)</span>${legend}</div>
    ${layers.length ? `<div class="layer-table-scroll" tabindex="0" role="region" aria-label="Compiled layer timings"><table class="layer-table">
      <colgroup><col><col><col></colgroup>
      <thead><tr><th scope="col">Compiled layer</th><th scope="col">Execution time <span class="layer-time-unit">(µs)</span>${axis}</th><th scope="col">Share (%)</th></tr></thead>
      <tbody>${rows}</tbody></table></div>` : '<p class="muted-message">No layers match these filters.</p>'}`;
}

function layerAnalysisBody(analysis, view) {
  const { count, zeroCount, top10Share } = selectLayers(analysis);
  const scaleUs = layerTimeScale(analysis);
  return `<div class="layer-summary">
    <div><strong>${layerTime(analysis.totalUs)} <small>µs</small></strong><span>Profiled MLA time</span></div>
    <div><strong>${count.toLocaleString()}</strong><span>Compiled layers · ${zeroCount} zero-time</span></div>
    <div><strong>${top10Share.toFixed(1)}<small>%</small></strong><span>Time in 10 slowest layers</span></div>
  </div>
  <p class="layer-explanation">Measured execution times from a separate MLA profiling pass. Shares use the component’s profiled time, not end-to-end latency. Compiled layer names may differ from ONNX operators.</p>
  <div class="layer-controls">
    <label class="layer-search">Layer name<input type="search" data-layer-filter="query" value="${escapeHtml(view.query)}" placeholder="Search compiled layers" autocomplete="off"></label>
    <label class="layer-threshold"><span>Minimum time: <output data-layer-threshold-value>${layerTime(view.minUs)} µs</output></span><input type="range" data-layer-filter="minUs" min="0" max="${scaleUs}" step="any" value="${view.minUs}"></label>
    <div class="layer-order-toggle" role="group" aria-label="Layer order">${[["compiler", "Runtime order"], ["time", "Slowest layers first"]].map(([order, label]) => `<button type="button" data-layer-filter="order" value="${order}" aria-pressed="${view.order === order}">${label}</button>`).join("")}</div>
  </div>
  <div data-layer-results>${layerAnalysisResults(analysis, view)}</div>`;
}

// Buttons that pick one of a pipeline's components; `attribute` names the control for its handler.
function componentSwitch(labels, selected, attribute) {
  return `<div class="layer-order-toggle component-switch" role="group" aria-label="Component">${labels.map((label, index) => `<button type="button" ${attribute} value="${index}" aria-pressed="${index === selected}">${escapeHtml(label)}</button>`).join("")}</div>`;
}

// The layer analysis of one component, or of a pipeline's components with a switch between them.
function layerAnalysisPanel({ sources, selected, view }) {
  const switcher = sources.length > 1 ? componentSwitch(sources.map(({ label }) => label), selected, 'data-layer-filter="component"') : "";
  return `${switcher}${layerAnalysisBody(sources[selected].analysis, view)}`;
}

// The execution graph of one component, or of a pipeline's components with a switch between them.
function executionGraphSection(components, key) {
  const graphed = components.filter(({ graph }) => graph);
  if (!graphed.length) return "";
  const meta = graphed.length > 1
    ? graphed.map(({ label }) => label).join(" · ")
    : graphed[0].graph.steps.filter((step) => step.runsOn).map((step) => step.runsOn).join(" · ");
  return disclosure(`graphs:${key}`, "Execution Graph",
    componentPanels(graphed.map(({ label }) => label), graphed.map(({ id, graph }) => executionGraphDiagram(graph, id))), meta);
}

// Every component's panel, with a switch that shows one at a time when there are several.
function componentPanels(labels, panels) {
  const switcher = labels.length > 1 ? componentSwitch(labels, 0, "data-component-switch") : "";
  return `<div data-component-panels>${switcher}${panels.map((markup, index) =>
    `<div data-component-panel="${index}"${index ? " hidden" : ""}>${markup}</div>`).join("")}</div>`;
}

export function selectComponentPanel(event) {
  const button = event.target.closest("[data-component-switch]");
  if (!button) return;
  const panel = button.closest("[data-component-panels]");
  panel.querySelectorAll("[data-component-switch]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  panel.querySelectorAll("[data-component-panel]").forEach((item) => { item.hidden = item.dataset.componentPanel !== button.value; });
}

function layerAnalysisSection(sources, key) {
  const analyzed = sources.filter(({ analysis }) => analysis);
  if (!analyzed.length) return "";
  const previous = state.layerViews.get(key);
  const entry = { sources: analyzed, selected: previous?.selected || 0, view: previous?.view || { query: "", minUs: 0, order: "time" } };
  state.layerViews.set(key, entry);
  const { analysis } = analyzed[0];
  const meta = analyzed.length > 1
    ? analyzed.map(({ label }) => label).join(" · ")
    : `${analysis.layers.length.toLocaleString()} layers · ${layerTime(analysis.totalUs)} µs MLA`;
  return disclosure(`layers:${key}`, "Runtime Layer Analysis",
    `<div data-layer-analysis="${escapeHtml(key)}">${layerAnalysisPanel(entry)}</div>`, meta);
}

export function updateLayerAnalysis(event) {
  const control = event.target.closest("[data-layer-filter]");
  if (!control) return;
  const field = control.dataset.layerFilter;
  if (event.type !== (["order", "component"].includes(field) ? "click" : "input")) return;
  const panel = control.closest("[data-layer-analysis]");
  const entry = state.layerViews.get(panel.dataset.layerAnalysis);
  if (field === "component") {
    // Components have different time scales, so the minimum time starts over.
    entry.selected = Number(control.value);
    entry.view.minUs = 0;
    panel.innerHTML = layerAnalysisPanel(entry);
    return;
  }
  const { view } = entry;
  const { analysis } = entry.sources[entry.selected];
  view[field] = field === "minUs" ? Number(control.value) : control.value;
  panel.querySelector("[data-layer-results]").innerHTML = layerAnalysisResults(analysis, view);
  panel.querySelector("[data-layer-threshold-value]").textContent = `${layerTime(view.minUs)} µs`;
  panel.querySelectorAll('[data-layer-filter="order"]').forEach((button) => {
    button.setAttribute("aria-pressed", String(button.value === view.order));
  });
}

export function highlightLayerLegend(event) {
  const row = event.target.closest(".layer-table tbody tr[data-layer-family]");
  if (!row || row.contains(event.relatedTarget)) return;
  const panel = row.closest("[data-layer-analysis]");
  const family = row.dataset.layerFamily;
  panel.querySelectorAll(".layer-legend [data-layer-family]").forEach((item) => {
    item.classList.toggle("is-highlighted", event.type === "pointerover" && item.dataset.layerFamily === family);
  });
}

// The KPI tiles, the context line, and Performance details of one performance result.
function performanceSummary(performance, run, modelId) {
  const tiles = KPI_METRICS.map((key) => kpiTile(key, performance.measurements[key])).join("");
  const details = performanceDetails(performance, run);
  return `${tiles ? `<div class="kpis">${tiles}</div>` : ""}
    <p class="performance-context">${escapeHtml(performanceContext(performance))}${measurementSource(run, modelId)}</p>
    ${disclosure(`performance:${run.id}`, "Performance Details", details.markup, details.titles.join(" · "))}`;
}

// Component and role ids read as words: "feature_extractor" becomes "Feature extractor".
function humanize(id) {
  const words = String(id).replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

// The component ids in the order the pipeline runs them, when the pipeline names exactly the
// variant's components; a pipeline built from another branch's model.yaml may not.
function pipelineOrder(variant) {
  const order = (variant.pipeline || []).filter((stage) => "component" in stage).map((stage) => stage.component);
  const ids = variant.details.map((detail) => detail.summary.run.metadata.component_id);
  return order.length === ids.length && order.every((id) => ids.includes(id)) ? order : null;
}

// One line from input to output: numbered component chips, with any CPU step between them as a dashed chip.
function pipelineStrip(pipeline) {
  let number = 0;
  const stages = pipeline.map((stage) => ("component" in stage
    ? `<li><span class="pipeline-chip"><span class="component-number">${++number}</span>${escapeHtml(humanize(stage.component))}</span></li>`
    : `<li><span class="pipeline-chip cpu"><span class="pipeline-cpu">CPU</span>${escapeHtml(stage.cpu)}</span></li>`));
  return `<ol class="component-pipeline" aria-label="Order the components run in"><li class="pipeline-end">Input</li>${stages.join("")}<li class="pipeline-end">Output</li></ol>`;
}

// Each component's MLA stages set against the simulator's operation counts, in developer mode only: the rates rest
// on a simulator estimate.
function renderMlaUtilization(components, simulation, precisions, throughputFps, key) {
  if (customerView || !simulation && !components.some(({ mlaStageStats }) => mlaStageStats?.status === "ok")) return "";
  const results = components.map((component) => {
    const { stages, reason } = matchMlaStages(component.mlaStageStats, simulation?.components?.[component.id]);
    const peak = mlaPeakTops(precisions[component.id]);
    return { ...component, reason, peak, metrics: stages && mlaUtilizationMetrics(stages, { peakTops: peak?.tops }) };
  });
  const number = (value, digits) => value === null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
  const percent = (value) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;
  // Component numbers are never added up, so only a single-component model gets a model-level rate.
  const compared = results.length === 1 && results[0].metrics && Number.isFinite(throughputFps) && results[0].batch > 0;
  const sustainedTops = compared ? sustainedEffectiveTops(results[0].metrics.total, throughputFps, results[0].batch) : null;
  // A component's totals as tiles in the layer summary's style, each explained behind an (i) mark.
  const tile = (value, digits, unit, label, info) => `<div><strong>${number(value, digits)}${value === null ? "" : unit}</strong><span>${label}${infoMark(info)}</span></div>`;
  const summary = ({ peak, metrics: { stages, total } }) => `<div class="layer-summary mla-summary">
    ${total.utilization === null ? "" : tile(total.utilization * 100, 1, "<small>%</small>", "Utilization",
      `Effective TOPS as a share of the MLA's ${number(peak.tops, 1)} TOPS ${peak.datatype} peak.${peak.datatype === "BF16"
        ? `\nA BF16 multiply takes four times the cycles of an INT8 one, so the peak is a quarter of ${MLA_PEAK_INT8_TOPS} TOPS.` : ""}`)}
    ${tile(total.effectiveTops, 2, " <small>TOPS</small>", "Effective TOPS",
      `Useful operations per second: the Model Compiler simulator's count over the measured MLA time.${compared ? `\nSustained at the measured throughput: ${number(sustainedTops, 2)} TOPS.` : ""}`)}
    ${tile(total.packing === null ? null : total.packing * 100, 1, "<small>%</small>", "Packing",
      "Share of the operations the MLA executes that are useful. The rest is padding.")}
    ${tile(total.measuredUs, 1, " <small>µs</small>", "MLA time",
      `Measured on the board${stages.length > 1 ? ` across ${stages.length} MLA stages` : ` for MLA stage ${stages[0].index}`}.`)}
  </div>`;
  const row = (stage) => `<tr><th scope="row">MLA stage ${stage.index}</th>${[number(stage.measuredUs, 1), number(stage.simulatedUs, 1),
    number(stage.effectiveOps / 1e9, 2), percent(stage.packing), number(stage.effectiveTops, 2), percent(stage.utilization)]
    .map((value) => `<td>${value}</td>`).join("")}</tr>`;
  const headings = ["Stage", "Measured <small>(µs)</small>", "Simulated <small>(µs)</small>", "Effective GOP", "Packing",
    "Effective TOPS", "Utilization"];
  // Each stage's numbers stay behind a Details button, as the execution graph's tensors do.
  const stageDetails = ({ id, peak, metrics }) => {
    const detailsKey = `mla-stages:${key}:${id}`;
    const open = state.openDisclosures.has(detailsKey);
    return `<div class="mla-summary-row">${summary({ peak, metrics })}
      <button type="button" class="execution-details" data-details="${escapeHtml(detailsKey)}" aria-controls="${escapeHtml(detailsKey)}" aria-expanded="${open}">Details${icons.chevron}</button></div>
      <div id="${escapeHtml(detailsKey)}" class="tensor-table"${open ? "" : " hidden"}><table><thead><tr>${headings.map((heading) => `<th scope="col">${heading}</th>`).join("")}</tr></thead>
      <tbody>${metrics.stages.map(row).join("")}</tbody></table></div>`;
  };
  const panels = results.map((result) => (result.metrics
    ? stageDetails(result) : `<p class="muted-message">${escapeHtml(result.reason)}</p>`));
  const meta = compared ? `${number(sustainedTops, 2)} effective TOPS sustained`
    : results.length > 1 ? results.map(({ label }) => label).join(" · ") : "";
  return disclosure(`mla-utilization:${key}`, "MLA Utilization",
    `<div class="mla-utilization">${componentPanels(results.map(({ label }) => label), panels)}</div>`, meta);
}

// The build profile's numbers for the whole model, with each component's layer analysis, MLA utilization and
// execution graph.
function modelPerformance(variant, modelId) {
  const benchmarkRun = variant.performance;
  const result = readPerformance(benchmarkRun, pipelineOrder(variant) || []);
  if (result.kind === "missing") return `<p class="muted-message">Not measured at this revision.</p>`;
  if (result.kind === "unsupported") return `<p class="muted-message">This result format is not supported by this browser.</p>`;
  const { run } = benchmarkRun;
  const components = result.value.components.map((component) => ({ ...component, label: humanize(component.id) }));
  return `<div class="component-performance">
    ${performanceSummary(result.value, run, modelId)}
    ${layerAnalysisSection(components.map(({ label, layerAnalysis }) => ({ label, analysis: layerAnalysis })), run.id)}
    ${renderMlaUtilization(components, mlaStageSimulation(variant.packageTests?.test_results), componentPrecisions(variant),
      result.value.measurements.throughput_fps?.value, run.id)}
    ${executionGraphSection(components.map(({ id, label, executionGraph }) => ({ id: `${run.id}:${id}`, label, graph: executionGraph })), run.id)}
  </div>`;
}

// Each component's compile precision, such as w8a8.
function componentPrecisions(variant) {
  return Object.fromEntries(variant.details.map((detail) => [detail.summary.run.metadata.component_id,
    detail.provenance.parameters?.parameters?.compile_parameters?.precision]));
}

function compileDuration(run) {
  return formatDuration(run.metadata.compile_timing.started_at, run.metadata.compile_timing.finished_at);
}

function componentRow(detail, single) {
  const run = detail.summary.run;
  const component = run.metadata.component_id;
  const artifacts = modelPackArtifacts(detail.summary.artifacts);
  const parameters = detail.provenance.parameters?.parameters?.compile_parameters || {};
  const mpkBytes = artifacts.reduce((total, artifact) => total + (Number.isFinite(deploymentSize(artifact)) ? deploymentSize(artifact) : 0), 0);
  const artifactRows = artifacts.length
    ? artifacts.map((artifact) => keyValue(artifactDelivery(artifact) === "archive" ? "SHA-256" : "Manifest SHA-256", artifact.sha256, { copy: true, mono: true })).join("")
    : keyValue("SHA-256", "No artifact metadata was registered for this run.");
  const meta = [mpkBytes ? `${run.metadata?.artifact_kind === "llima" ? "Model" : "MPK"} ${formatBytes(mpkBytes)}` : "", parameters.precision && precisionLabel(parameters.precision)]
    .filter(Boolean).join(" · ");
  const body = `<dl class="key-grid">
      ${artifactRows}
      ${keyValue("Run ID", run.id, { copy: true, mono: true })}
      ${keyValue("Fingerprint", run.run_fingerprint, { copy: true, mono: true })}
      ${keyValue("Compile recipe duration", compileDuration(run))}
      ${keyValue("Registry publication duration", formatDuration(run.started_at, run.finished_at))}
    </dl>
    ${Object.keys(parameters).length ? disclosure(`parameters:${run.id}`, "Compile parameters", `<pre>${escapeHtml(JSON.stringify(parameters, null, 2))}</pre>`, "JSON") : ""}`;
  return disclosure(`component:${run.id}`, single ? "Files and checksums" : component, body, meta);
}

// The first metrics in display order are shown as tiles; any further metrics are listed in a table.
const ACCURACY_SUMMARY_TILES = 4;

// The registry does not preserve a result's metric order, so known metrics follow the label order and
// any others follow them in the order they arrive.
function orderedAccuracyMetrics(metrics) {
  return orderedMetricNames(Object.keys(metrics)).map((name) => [name, metrics[name]]);
}

// The first accuracy metric of a build's first dataset and the source model's reference score, for the Compare
// page and the catalog.
export function accuracyHeadline(variant) {
  const result = readAccuracy(variant.accuracy);
  if (result.kind !== "supported" || !result.value.datasets.length) return null;
  const { metrics, dataset } = result.value.datasets[0];
  const [name, value] = orderedAccuracyMetrics(metrics)[0] || [];
  if (!name) return null;
  const display = accuracyDisplay(name, value);
  return {
    label: accuracyMetricLabel(name), value: formatAccuracy(name, value),
    reference: formatAccuracy(name, referenceScores(variant.reference, dataset.id)[name], { blank: true }),
    numericValue: display.number,
    unit: display.unit,
    direction: display.direction,
    dataset,
    comparisonKey: dataset.id ? JSON.stringify([dataset.id, dataset.variant, dataset.split, dataset.sha256, name]) : null,
  };
}

// The build profile's accuracy run, with the published and FP32 baseline scores on the same dataset.
function accuracyPanel(variant, modelId) {
  const reference = variant.reference || [];
  const referenceMarkup = customerView ? "" : referenceDisclosures(reference);
  const result = readAccuracy(variant.accuracy);
  if (result.kind === "missing") return referenceAccuracySection(reference);
  if (result.kind === "unsupported") {
    return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
      <div class="detail-section-title"><h3>Accuracy</h3></div>
      <p class="muted-message">This result format is not supported by this browser.</p>
      ${referenceMarkup}
    </section>`;
  }
  const { datasets, tool } = result.value;
  const sourceMarkup = measurementSource(variant.accuracy.run, modelId);
  // One dataset keeps its coverage beside the section title; several each title their own result.
  const badge = datasets.length === 1 ? coverageBadge(datasets[0].dataset) : "";
  const results = datasets.map((entry) => accuracyResult(
    entry, tool, scoresByOrigin(reference, entry.dataset.id), sourceMarkup, datasets.length > 1,
  )).join("");
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    <div class="detail-section-title"><h3>Accuracy</h3>${badge}</div>
    ${results}
    ${referenceMarkup}
  </section>`;
}

// Each origin's reference scores on a dataset, keyed by origin.
function scoresByOrigin(reference, datasetId) {
  return Object.fromEntries(Object.keys(REFERENCE_ORIGINS).map((origin) => [origin, referenceScores(reference, datasetId, origin)]));
}

function accuracyMetricTiles(metrics, scores) {
  return orderedAccuracyMetrics(metrics).slice(0, ACCURACY_SUMMARY_TILES).map(([name, value], index) => {
    const notes = referenceNotes(name, scores).map((note) => `<small>${escapeHtml(note)}</small>`).join("");
    return `<div class="accuracy-metric${index === 0 ? " primary" : ""}"><span>${escapeHtml(accuracyMetricLabel(name))}</span><strong>${escapeHtml(formatAccuracy(name, value))}</strong>${notes || "<small></small>"}</div>`;
  }).join("");
}

// The Published column always shows; the FP32 baseline column only when the model has a baseline here.
function accuracyMetricTable(metrics, scores) {
  const origins = Object.keys(REFERENCE_ORIGINS).filter((origin) => origin === "published" || Object.keys(scores[origin]).length);
  const rows = orderedAccuracyMetrics(metrics).slice(ACCURACY_SUMMARY_TILES).map(([name, value]) => (
    `<tr><th scope="row">${escapeHtml(accuracyMetricLabel(name))}</th><td>${escapeHtml(formatAccuracy(name, value))}</td>${origins.map((origin) => `<td>${escapeHtml(formatAccuracy(name, scores[origin][name]))}</td>`).join("")}</tr>`
  )).join("");
  if (!rows) return "";
  const headers = origins.map((origin) => `<th scope="col">${escapeHtml(REFERENCE_ORIGINS[origin].label)}</th>`).join("");
  return `<div class="metric-table"><h4>Additional metrics</h4><div class="metric-table-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">Value</th>${headers}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

// Conditions are written as a phrase, such as "640x640, FP32"; notes may already end a sentence.
function sentence(text = "") {
  return !text || /[.!?]$/.test(text) ? text : `${text}.`;
}

// One row per origin the model has, listing that origin's entries.
function referenceDisclosures(reference) {
  return Object.entries(REFERENCE_ORIGINS).map(([origin, { title, id }]) => {
    const entries = reference.filter((entry) => referenceOrigin(entry) === origin);
    if (!entries.length) return "";
    return disclosure(id, title, referenceTables(entries), `${entries.length} ${entries.length === 1 ? "source" : "sources"}`);
  }).join("");
}

// Each reference entry with the dataset and conditions it was measured on, and where it came from.
function referenceTables(reference) {
  return reference.map((entry) => {
    const { label } = REFERENCE_ORIGINS[referenceOrigin(entry)] || REFERENCE_ORIGINS.published;
    const metrics = entry.metrics || {};
    const rows = orderedMetricNames(Object.keys(metrics)).map((name) => (
      `<tr><th scope="row">${escapeHtml(accuracyMetricLabel(name))}</th><td>${escapeHtml(formatAccuracy(name, metrics[name]))}</td></tr>`
    )).join("");
    const source = trustedHttpsUrl(entry.source);
    const link = source
      ? ` <a class="measurement-link" href="${escapeHtml(source.href)}" target="_blank" rel="noreferrer">Source${icons.external}</a>`
      : entry.script ? ` Measured by ${escapeHtml(entry.script)}.` : "";
    return `<div class="metric-table published-scores"><h4>${escapeHtml(`${entry.dataset?.id || ""} ${entry.dataset?.split || ""}`.trim())}</h4>
      <p class="disclosure-text">${escapeHtml(sentence(entry.conditions))}${entry.note ? ` ${escapeHtml(sentence(entry.note))}` : ""}${link}</p>
      <div class="metric-table-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">${escapeHtml(label)}</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }).join("");
}

// A build without an accuracy result still shows its reference scores, each origin under its own explanation.
const REFERENCE_SCOPES = {
  published: "The model's authors published these scores for the source model.",
  fp32_baseline: "The FP32 source model scores these on the package's own inputs.",
};

function referenceAccuracySection(reference) {
  if (!reference.length) return "";
  const groups = Object.keys(REFERENCE_ORIGINS).map((origin) => {
    const entries = reference.filter((entry) => referenceOrigin(entry) === origin);
    return entries.length ? `<p class="accuracy-scope">${escapeHtml(REFERENCE_SCOPES[origin])}</p>${referenceTables(entries)}` : "";
  }).join("");
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    <div class="detail-section-title"><h3>Accuracy</h3></div>
    <p class="accuracy-scope">No accuracy result for this build yet.</p>
    ${groups}
  </section>`;
}

// A run that evaluated every sample of its split has full coverage; any other run evaluated a sampled subset.
function fullCoverage(dataset) {
  return Number.isInteger(dataset.evaluated_samples) && dataset.evaluated_samples === dataset.available_samples;
}

function coverageBadge(dataset) {
  const full = fullCoverage(dataset);
  return `<span class="status-badge ${full ? "qualification" : "provisional"}">${full ? "Full coverage" : "Sampled coverage"}</span>`;
}

// One dataset's result. With several, each is titled by its model.yaml name and shows its own coverage.
function accuracyResult({ name, metrics, dataset, details }, tool, scores, sourceMarkup, titled) {
  const full = fullCoverage(dataset);
  const coverageLabel = full ? "Full coverage" : "Sampled coverage";
  const coverageExplanation = full
    ? "Every sample in the dataset split was evaluated."
    : "A deterministic subset of the dataset split was evaluated.";
  const sampleText = Number.isInteger(dataset.evaluated_samples) && Number.isInteger(dataset.available_samples)
    ? `${dataset.evaluated_samples.toLocaleString()} of ${dataset.available_samples.toLocaleString()} samples`
    : "Sample count unavailable";
  const context = `${keyValue("Measured on", "DevKit")}${keyValue("Runtime", tool)}${Object.entries(details)
    .map(([key, value]) => keyValue(humanize(key), typeof value === "object" ? JSON.stringify(value) : String(value))).join("")}`;
  const evaluation = `<p class="disclosure-text"><strong>${coverageLabel}.</strong> ${escapeHtml(coverageExplanation)}</p>
      <dl class="key-grid">${context}</dl>
      ${accuracyMetricTable(metrics, scores)}`;
  return `<div class="accuracy-dataset" data-accuracy-dataset="${escapeHtml(name)}">
    ${titled ? `<div class="accuracy-dataset-title"><h4>${escapeHtml(name)}</h4>${coverageBadge(dataset)}</div>` : ""}
    <p class="accuracy-scope"><strong>${escapeHtml(sampleText)}</strong> · ${escapeHtml(dataset.id || "Dataset unavailable")} ${escapeHtml(dataset.variant || "")} / ${escapeHtml(dataset.split || "")}${sourceMarkup}</p>
    <div class="accuracy-summary">${accuracyMetricTiles(metrics, scores)}</div>
    ${customerView ? "" : disclosure(`accuracy:${name}`, "Evaluation details", evaluation, coverageLabel)}
  </div>`;
}

// Numerical fidelity compares the whole model's outputs with the source model's, on the package run.
function fidelityPanel(tests, runId) {
  const result = tests.find((item) =>
    item.suite === "correctness"
    && item.name === "numerical-fidelity"
    && item.summary?.schema_version === 1
  );
  const report = result?.summary;
  if (!report) return "";
  const metrics = (report.metrics || []).map((metric) => `
    <div class="fidelity-metric"><span>${escapeHtml(metric.name)} · Cosine similarity</span><strong>${Number(metric.cosine_similarity).toFixed(4)}</strong></div>
    <div class="fidelity-metric"><span>${escapeHtml(metric.name)} · Normalized RMSE</span><strong>${Number(metric.normalized_rmse).toFixed(4)}</strong></div>
  `).join("");
  return `<div class="fidelity-panel">
    <div class="group-title"><h4>Numerical fidelity</h4><button class="action-button" type="button" data-download="numerical-fidelity" data-run-id="${escapeHtml(runId)}">${icons.download}<span>Download JSON</span></button></div>
    <p class="fidelity-scope">${escapeHtml(report.status)} · ${Number(report.samples).toLocaleString()} samples · ${escapeHtml(report.dataset?.id || "Dataset unavailable")} ${escapeHtml(report.dataset?.variant || "")}</p>
    <div class="fidelity-metrics">${metrics}</div>
    <dl class="key-grid">${keyValue("Reference", report.reference)}${keyValue("Candidate", report.candidate)}</dl>
    <p class="muted-message">Output similarity, not task accuracy. Output reordering can affect these metrics.</p>
  </div>`;
}

// `listed` is the license's entry in licenses.json: its name, and what it asks of someone who redistributes
// the model.
function licensePanel(license, listed) {
  if (!license?.spdx) return "";
  const redistribution = license.redistribution ? `Redistribution ${license.redistribution}` : "Redistribution not recorded";
  return `<section id="license-section" class="license-section" tabindex="-1"><div class="detail-section-title"><h3>License</h3></div>
    <p class="license-line"><strong>${escapeHtml(license.spdx)}</strong>${listed ? `<span>${escapeHtml(listed.name)}</span>` : ""}<span>${escapeHtml(redistribution)}</span></p>
    ${listed ? `<p class="license-summary">${escapeHtml(listed.summary)}</p>` : ""}
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
  if (markdown === undefined) return `<p class="muted-message" data-model-card-pending>Loading usage instructions…</p>`;
  const remainder = modelCardRemainder(markdown);
  const heroUrl = documentationHeroUrl(documentation);
  return `<div class="model-documentation-body">${renderMarkdown(remainder ?? markdown, {
    stripTitle: remainder === null,
    resolveImage: (source) => (source === documentation.hero?.markdown_path ? heroUrl : null),
    resolveLink: (source) => resolveDocumentationLink(documentation, source),
  })}</div>`;
}

function modelHero(model) {
  const documentation = model?.documentation;
  if (!documentation) return "";
  const heroUrl = documentationHeroUrl(documentation);
  return heroUrl
    ? `<figure class="model-card-hero"><img src="${escapeHtml(heroUrl)}" data-hero-fallback="${escapeHtml(documentationHeroFallbackUrl(documentation))}" alt="${escapeHtml(documentation.hero.alt_text || `${model.display_name || model.model_id} model hero`)}" decoding="async" /></figure>`
    : "";
}

function overviewSection(model) {
  const documentation = model?.documentation;
  if (!documentation?.overview_markdown) return "";
  const instructions = documentation.run_id && documentation.artifact_sha256
    ? `<div data-model-card-body>${modelCardBody(documentation)}</div>`
    : "";
  return `<section id="overview-section" class="overview-section" tabindex="-1" aria-labelledby="overview-title">
    <div class="detail-section-title"><h3 id="overview-title">Overview</h3></div>
    <div class="model-documentation-body">${renderMarkdown(documentation.overview_markdown, { resolveLink: (source) => resolveDocumentationLink(documentation, source) })}</div>
    ${instructions}
  </section>`;
}

// Loads the remaining README sections inline after checking the catalog checksum.
async function loadModelCard() {
  const variant = currentVariant();
  const documentation = variant ? catalogModelForRun(variant.anchor)?.documentation : null;
  const body = () => elements["detail-pane"].querySelector("[data-model-card-body]");
  if (!documentation || !body()?.querySelector("[data-model-card-pending]")) return;
  const requestId = ++state.documentationRequestId;
  try {
    const markdown = await apiText(`/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(documentation.artifact_name || "readme")}/content`);
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(markdown)))]
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (digest !== documentation.artifact_sha256) throw new Error("The model card checksum did not match the catalog.");
    state.documentationCache.set(documentationKey(documentation), markdown);
  } catch (error) {
    if (requestId !== state.documentationRequestId || state.selectedVariantKey !== variant.key) return;
    const target = body();
    if (target) target.innerHTML = `<p class="muted-message" data-model-card-pending>The usage instructions could not be loaded.</p><button class="text-button" type="button" data-action="retry">Retry</button>`;
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
    return `<button class="detail-variant-tab" type="button" role="tab" data-detail-model-id="${escapeHtml(model.model_id)}" data-detail-profile-id="${escapeHtml(profile)}" aria-selected="${selected}" tabindex="${selected ? "0" : "-1"}">${escapeHtml(variantLabel(profile))}</button>`;
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

function sectionNavigation(sections) {
  return `<nav class="subnav" aria-label="Model page sections">${sections.map(([id, label]) => (
    `<button type="button" data-scroll-to="${id}">${label}</button>`
  )).join("")}</nav>`;
}

// The package is the glance view of Details: its size and checksum, and the ways to download it.
// A single-component package is that component's MPK, so only multi-component builds offer components.
function packageCard(variant, fullPackage, packageArtifact, hasArtifacts, singleComponent) {
  const run = variant.anchor;
  const folder = packageArtifact && artifactDelivery(packageArtifact) !== "archive";
  const location = packageArtifact && huggingFaceLocation(packageArtifact);
  const components = hasArtifacts && !(fullPackage && singleComponent)
    ? `<button class="action-button" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">${icons.download}<span>Download components</span></button>`
    : "";
  if (!fullPackage) return components ? `<div class="package-actions">${components}</div>` : "";
  return `<div class="package-card">
    <div class="package-card-copy"><strong>Full package</strong><span>${folder ? "Download the model folder with sima-cli using its verified manifest." : packageFormat(packageArtifact) === "MPK" ? "The compiled model in one verified MPK." : "All components and declared runtime resources in one verified ZIP."}</span>${location ? `<a href="${escapeHtml(location.url)}" target="_blank" rel="noreferrer">Deployment on Hugging Face ↗</a>` : ""}</div>
    ${packageArtifact ? `<dl class="package-facts"><div><dt>Size</dt><dd>${escapeHtml(formatBytes(deploymentSize(packageArtifact)))}</dd></div><div><dt>${folder ? "Manifest SHA-256" : "SHA-256"}</dt><dd class="artifact-hash"><code>${escapeHtml(packageArtifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(packageArtifact.sha256)}" aria-label="Copy package checksum">${icons.copy}</button></dd></div></dl>` : ""}
    <div class="package-actions"><button class="action-button" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>Download package</span></button>${components}</div>
  </div>`;
}

export function renderDetail(variant) {
  hideExecutionTooltip();
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
  // A reused build keeps the categories it was compiled with; the catalog has the current ones.
  const categories = catalogModel?.categories || run.model_categories || {};
  const compilation = details[0]?.provenance.parameters?.parameters || {};
  const parameters = { ...compilation.compile_parameters, ...compilation.compiled_parameters };
  const hasImageResolution = [parameters.input_height, parameters.input_width].every((value) => Number.isInteger(value) && value > 0);
  const compiledLimits = ["llm", "vlm"].includes(categories.model_type) ? [
    Number.isInteger(parameters.max_num_tokens) && parameters.max_num_tokens > 0
      ? keyValue("Maximum context", `${parameters.max_num_tokens.toLocaleString("en-US")} tokens (input + output)`) : "",
    hasImageResolution ? keyValue("Image input resolution", `${parameters.input_height} × ${parameters.input_width} pixels`) : "",
  ].join("") : "";
  // The catalog carries the license model.yaml declares now. A build's own metadata keeps the license it
  // was compiled with.
  const license = catalogModel?.license || run.metadata?.license;
  const offered = downloadOffered(license?.spdx, variant.licenses);
  // CI reuses a build until its compile inputs change, so the build's commit can predate the scripts
  // that measured it. The source link opens the commit the model card comes from; Build provenance
  // keeps the build's own commit.
  const sourceCommit = catalogModel?.documentation?.source_commit || run.commit_sha;
  const sourceUrl = githubModelUrl(run, model, sourceCommit);
  const sourceLabel = `View ${model} source on GitHub at commit ${shortSha(sourceCommit)}`;
  const accuracyMarkup = accuracyPanel(variant, model);
  const overviewMarkup = overviewSection(catalogModel);
  const hasArtifacts = details.some((detail) => modelPackArtifacts(detail.summary.artifacts).length);
  // Every component's MPK must pass validation for the build to carry the mark.
  const validated = details.every((detail) => artifactValidationMark(detail.tests.test_results || []));
  const fullPackage = variantPackage(variant.runs);
  const packageArtifact = (variant.packageSummary?.artifacts || []).find((item) => item.artifact_type === "model-package");
  const downloadArtifact = packageArtifact || details.flatMap((detail) => modelPackArtifacts(detail.summary.artifacts))[0];
  const downloadLabel = downloadArtifact && artifactDelivery(downloadArtifact) !== "archive" ? "Download folder" : "Download";
  const originalModel = inputs.map(originalModelLink).find(Boolean);
  const builtAt = Math.max(...variant.runs.map((item) => Date.parse(item.finished_at || item.created_at || "") || 0));
  const singleComponent = details.length === 1;
  const order = pipelineOrder(variant);
  const backLabel = elements["result-title"].textContent || "Models";
  const downloadAction = !offered ? `<span class="download-unavailable">${icons.license}Not available for download</span>`
    : fullPackage
    ? `<button class="download-action" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>${downloadLabel}</span>${packageArtifact ? `<small>${escapeHtml(formatBytes(deploymentSize(packageArtifact)))}</small>` : ""}</button>`
    : hasArtifacts ? `<button class="download-action" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">${icons.download}<span>${downloadLabel}</span></button>` : "";
  const sections = [
    overviewMarkup && ["overview-section", "Overview"],
    ["components-section", "Performance"],
    accuracyMarkup && ["accuracy-section", "Accuracy"],
    !customerView && ["details-section", "Details"],
    license?.spdx && ["license-section", "License"],
  ].filter(Boolean);

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">${escapeHtml(backLabel)}</button><span aria-hidden="true">/</span><span aria-current="page">${escapeHtml(displayName)}</span></nav>
    <header class="detail-title">
      <div class="detail-title-copy">
        <h2>${escapeHtml(displayName)}</h2>
        <div class="detail-tags">${(categories.tasks || []).map((task) => detailTag(icons.task, taxonomyLabel("tasks", task), "task")).join("")}${categories.architecture?.id ? detailTag(icons.architecture, taxonomyLabel("architectures", categories.architecture.id), "architecture") : ""}${categories.model_type ? detailTag(icons.modelType, taxonomyLabel("model_types", categories.model_type), "model-type") : ""}${precisions.length ? detailTag(icons.precision, precisions.map(precisionLabel).join(" + "), "precision") : ""}${detailTag(icons.compiler, `${run.toolchain_name || "compiler"} ${run.toolchain_version || ""}`.trim(), "compiler")}${detailTag(icons.branch, branchName(run.git_ref), "branch")}${license?.spdx ? detailTag(icons.license, license.spdx, "license") : ""}</div>
        ${catalogModel?.description ? `<p class="lede">${escapeHtml(catalogModel.description)}</p>` : ""}
        <div class="detail-title-actions">
          ${downloadAction}
          <div class="detail-links">${validated ? artifactValidationMark(allTests) : ""}${originalModel ? `<a class="original-model-link${originalModel.huggingFace ? "" : " external-model-link"}" href="${escapeHtml(originalModel.href)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(originalModel.label)}" data-tooltip="${escapeHtml(originalModel.label)}">${originalModel.huggingFace ? icons.huggingFace : icons.external}</a>` : ""}${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" data-tooltip="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}</div>
        </div>
      </div>
      ${modelHero(catalogModel)}
    </header>
    ${detailBuildSwitcher(variant)}
    ${compiledLimits ? `<dl class="key-grid">${compiledLimits}</dl>` : ""}
    ${sectionNavigation(sections)}

    ${overviewMarkup}

    <section id="components-section" class="component-section" tabindex="-1"><div class="detail-section-title"><h3>Performance on ${escapeHtml(targetName(run))}</h3></div>
      ${singleComponent ? "" : `<p class="section-lede">This model runs as ${details.length} components, measured running together.</p>`}
      ${order ? pipelineStrip(variant.pipeline) : ""}
      ${modelPerformance(variant, model)}
    </section>

    ${accuracyMarkup}

    ${customerView ? "" : `<section id="details-section" class="details-section" tabindex="-1"><div class="detail-section-title"><h3>Details</h3></div>
      ${offered ? packageCard(variant, fullPackage, packageArtifact, hasArtifacts, singleComponent) : ""}
      ${fullPackage ? fidelityPanel(variant.packageTests?.test_results || [], fullPackage.run_id) : ""}
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
    </section>`}

    ${licensePanel(license, variant.licenses?.[license?.spdx])}
  </div>`;
  loadModelCard();
}

export function errorPanel(message, title = "Registry unavailable", { closable = false } = {}) {
  return `<div class="error-state">${closable ? `<nav class="crumbs" aria-label="Breadcrumb"><button class="close-detail" type="button">Back to models</button></nav>` : ""}<span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}
