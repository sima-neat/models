// Model page: performance, accuracy, model card, build details, and license.
import { accuracyMetricLabel, orderedMetricNames, publishedScores } from "./accuracy-metrics.js";
import { selectLayers, layerTimeScale, layerFamily, LAYER_FAMILIES } from "./layer-stats.js";
import { adaptBenchmarkResult } from "./benchmark-adapters.js";
import { renderMarkdown, resolveDocumentationLink } from "./markdown.js";
import { state, elements, catalogMode, currentVariant, showView, taxonomyLabel } from "./state.js";
import { icons, escapeHtml, branchName, shortSha, profileLabel, variantLabel, precisionLabel, showToast, disclosure } from "./ui.js";
import {
  metricDefinition, metricDisplay, formatDate, formatDuration,
  formatBytes,
} from "./metrics.js";
import {
  apiText, profileGroups, variantPackage, modelPackArtifacts, catalogModelForRun,
  compilerFilteredBuilds, documentationHeroUrl,
} from "./api.js";
import { packageFormat, modelDownloadCommand } from "./downloads.js";

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

function mpkValidationMark(tests = []) {
  const passed = tests.some((test) => test.name === "mpk-validation" && test.status === "passed");
  if (!passed) return "";
  const message = "MPK package integrity verified: archive structure, checksum, and build provenance match.";
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

function measuredValue(key, metric) {
  const definition = metricDefinition(key, metric);
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
// Benchmark details: one row per reading, with its label, its metrics, and a short explanation shown
// behind an (i) mark, one line per value when a row has several. A row names a rail when its colour should match that rail in the power bar,
// and a depth when it is part of the row above it.
const DETAIL_CARDS = [
  ["performance", "Performance", "gauge", [], [
    ["Throughput (async / sync)", ["throughput_fps", "sync_throughput_fps"], "Async: several inputs in flight.\nSync: one input at a time."],
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
    ["MLA", ["mla_memory_avg_mib", "mla_memory_peak_mib"], "Memory allocated to the MLA."],
    ["CMA pool", ["avg_cma_memory_mib", "peak_cma_memory_mib"], "Contiguous memory in use, shared with other drivers."],
    ["Linux", ["avg_linux_memory_mib", "peak_linux_memory_mib"], "Memory used by Linux and its processes."],
  ]],
];
// Performance, heat, and CPU fill the left column; power, memory, and any other measurements the right.
const RIGHT_COLUMN_CARDS = new Set(["power", "memory", "other"]);
// The tiles and cards place these metrics; any other measured metric is listed under Other measurements.
// Benchmark energy is energy per inference times the sample count, so it is not shown again.
const PLACED_METRICS = new Set([
  ...KPI_METRICS, "energy_joules",
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
  const window = telemetry.windowMs ? ` over ${(telemetry.windowMs / 1000).toFixed(1)} s` : "";
  return `Sentinel ${telemetry.sentinelVersion} · ${telemetry.samples} samples${interval}${window}`;
}

// Synchronous throughput: the latency loop runs one input at a time, so it completes one batch per mean latency.
function syncThroughput(benchmark) {
  const latency = benchmark.measurements.latency_avg_ms;
  if (latency?.status !== "measured" || !(latency.value > 0) || !(benchmark.batchSize > 0)) return null;
  return { status: "measured", value: (benchmark.batchSize * 1000) / latency.value, unit: benchmark.measurements.throughput_fps?.unit };
}

function benchmarkDetails(benchmark, payload) {
  const { measurements } = benchmark;
  const readings = { ...measurements, sync_throughput_fps: syncThroughput(benchmark) };
  const additions = {
    cpu: { note: benchmark.cpuCount > 0 && `${benchmark.cpuCount} cores`, extra: coreLoad(benchmark.cpuCores) },
    power: { lead: railSplit(measurements) },
  };
  const cards = DETAIL_CARDS.map((card) => detailCard(readings, card, additions[card[0]])).filter(Boolean);
  const otherRows = Object.keys(measurements).filter((key) => !PLACED_METRICS.has(key)).sort()
    .map((key) => measuredValue(key, measurements[key])).filter(Boolean).map(metricRow);
  if (otherRows.length) {
    cards.push({ id: "other", title: "Other", markup: `<div class="detail-card"><h4><i aria-hidden="true">${icons.benchmark}</i>Other measurements</h4><dl>${otherRows.join("")}</dl></div>` });
  }
  const column = (right) => `<div class="detail-column">${cards
    .filter((card) => RIGHT_COLUMN_CARDS.has(card.id) === right)
    .map((card) => card.markup).join("")}</div>`;
  const provenance = [
    ["Telemetry", telemetryText(benchmark.telemetry)],
    ["Runtime", benchmark.tool],
  ].filter(([, text]) => text).map(([term, text]) => `<div><dt>${term}</dt><dd>${escapeHtml(text)}</dd></div>`).join("");
  const raw = payload.result_artifact_url
    ? `<a class="benchmark-action" href="${escapeHtml(payload.result_artifact_url)}" target="_blank" rel="noreferrer">Raw result${icons.external}</a>`
    : "";
  const csv = `<a class="benchmark-action" href="${escapeHtml(csvDataUrl(benchmarkCsv(benchmark, readings)))}" download="${escapeHtml(csvFileName(benchmark))}">${icons.download}Download CSV</a>`;
  return {
    titles: cards.map((card) => card.title),
    markup: `<div class="detail-cards">${column(false)}${column(true)}</div>
    <footer class="benchmark-footer"><dl class="benchmark-provenance">${provenance}</dl><div class="benchmark-actions">${raw}${csv}</div></footer>`,
  };
}

// The CSV download: every reading of one component as one table. Context comes first, then each
// card's metrics, every SoM rail, every CPU core, and the remaining measurements.
const CSV_EXTRA_METRICS = { power: ["energy_per_inference_joules", "energy_joules"] };

function benchmarkCsv(benchmark, readings) {
  const rows = [["Section", "Metric", "Value", "Unit", "Key", "Note"]];
  const add = (section, metric, value, unit = "", key = "", note = "") => rows.push([section, metric, value ?? "", unit, key, note]);
  const telemetry = benchmark.telemetry;
  const seconds = (milliseconds) => Number.isFinite(milliseconds) && (milliseconds / 1000).toFixed(1);
  [
    ["Model", benchmark.modelId], ["Build", benchmark.buildProfileId], ["Component", benchmark.componentId],
    ["Workload", benchmark.workload], ["Platform", benchmark.platformVersion], ["Neat", benchmark.neatVersion],
    ["MLA clock", Number.isFinite(benchmark.mlaFrequencyHz) && benchmark.mlaFrequencyHz / 1e6, "MHz"],
    ["CPU clock", Number.isFinite(benchmark.cpuFrequencyKhz) && benchmark.cpuFrequencyKhz / 1e3, "MHz"],
    ["CPU cores", benchmark.cpuCount], ["Batch size", benchmark.batchSize], ["Samples per trial", benchmark.samples],
    ["Measured", benchmark.timestamp], ["Board telemetry", telemetry?.status], ["Sentinel", telemetry?.sentinelVersion],
    ["Telemetry samples", telemetry?.samples], ["Telemetry interval", seconds(telemetry?.intervalMs), "s"],
    ["Telemetry window", seconds(telemetry?.windowMs), "s"], ["Tool", benchmark.tool], ["Result schema", benchmark.schemaVersion],
  ].filter(([, value]) => value || value === 0).forEach(([name, value, unit]) => add("Context", name, value, unit));
  const metric = (section, key) => {
    const reading = readings[key];
    if (!reading) return;
    const definition = metricDefinition(key, reading);
    const shown = measuredValue(key, reading);
    const note = !shown ? `Not reported${reading.reason ? `: ${reading.reason}` : ""}` : key === "sync_throughput_fps" ? "From the mean latency" : "";
    add(section, definition.label, shown?.text, shown?.unit ?? definition.unit, key, note);
  };
  const placed = new Set();
  DETAIL_CARDS.forEach(([id, title, , , cardRows]) => {
    [...cardRows.flatMap(([, keys]) => (Array.isArray(keys) ? keys : [])), ...(CSV_EXTRA_METRICS[id] || [])]
      .forEach((key) => { placed.add(key); metric(title, key); });
  });
  namedRails(benchmark.measurements).forEach((rail) => {
    add("Power rails", `${rail.name}, average`, rail.mean.toFixed(2), "W", rail.key);
    if (Number.isFinite(rail.maximum)) add("Power rails", `${rail.name}, peak`, rail.maximum.toFixed(2), "W", rail.key);
  });
  benchmark.cpuCores.forEach((core, index) => {
    add("CPU cores", `Core ${index}, average`, core.mean.toFixed(1), "%");
    if (Number.isFinite(core.maximum)) add("CPU cores", `Core ${index}, peak`, core.maximum.toFixed(1), "%");
  });
  Object.keys(benchmark.measurements).filter((key) => !placed.has(key)).sort().forEach((key) => metric("Other", key));
  benchmark.layerAnalysis?.layers.forEach((layer) => add(
    "MLA layers", layer.name, layer.timeUs, "µs", `stage:${layer.stage}/layer:${layer.order}`,
    "Separate profiling pass; compiler order",
  ));
  const cell = (value) => (/[",\r\n]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value));
  return rows.map((row) => row.map(cell).join(",")).join("\r\n");
}

// The byte order mark lets spreadsheet applications read the file as UTF-8, for units such as °C.
function csvDataUrl(csv) {
  return `data:text/csv;charset=utf-8,${encodeURIComponent(`\uFEFF${csv}\r\n`)}`;
}

function csvFileName(benchmark) {
  const name = [benchmark.modelId, benchmark.buildProfileId, benchmark.componentId].filter(Boolean).join("-") || "model";
  return `${name.replace(/[^A-Za-z0-9._-]+/g, "_")}-benchmark.csv`;
}

// The MLA and CPU clocks bound the speed a result can reach, so the context names the clocks it ran at.
function benchmarkContext(benchmark) {
  const clock = (name, megahertz) => Number.isFinite(megahertz) && `${name} ${megahertz.toLocaleString()} MHz`;
  return [
    benchmark.workload,
    benchmark.platformVersion && `Platform ${benchmark.platformVersion}`,
    benchmark.neatVersion && `Neat ${benchmark.neatVersion}`,
    clock("MLA", benchmark.mlaFrequencyHz / 1e6),
    clock("CPU", benchmark.cpuFrequencyKhz / 1e3),
    Number.isFinite(benchmark.samples) && `${benchmark.samples.toLocaleString()} samples per trial`,
    benchmark.timestamp && `Measured ${formatDate(benchmark.timestamp, false)}`,
  ].filter(Boolean).join(" · ");
}

// Each step of the benchmarked Model and the processor it ran on. Details, at the right of the heading,
// opens the tensors between the steps under the graph.
function executionGraphSection(graph, runId) {
  if (!graph) return "";
  const steps = graph.steps.map((step) => (step.runsOn
    ? `<li><span class="execution-step" data-runs-on="${escapeHtml(step.runsOn)}">${escapeHtml(step.name)}<small>${escapeHtml(step.runsOn)}</small></span></li>`
    : `<li class="execution-end"><span class="execution-step">${escapeHtml(step.name)}</span></li>`)).join("");
  const rows = graph.tensors.map((tensor) => `<tr${tensor.connection ? "" : ' class="continued"'}><th scope="row">${escapeHtml(tensor.connection)}</th>${[tensor.dtype, tensor.shape, tensor.scale, tensor.zeroPoint].map((value) => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("");
  const table = `<div class="tensor-table"><table><thead><tr>${["Connection", "Type", "Shape", "Scale", "Zero point"].map((label) => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const key = `tensors:${runId}`;
  const open = state.openDisclosures.has(key);
  const details = graph.tensors.length
    ? `<button type="button" class="execution-details" data-details="${escapeHtml(key)}" aria-controls="${escapeHtml(key)}" aria-expanded="${open}">Details${icons.chevron}</button>`
    : "";
  return `<section class="execution-graph" aria-label="Execution graph"><div class="execution-head"><h5>Execution graph</h5>${details}</div><ol>${steps}</ol>
    ${graph.tensors.length ? `<div id="${escapeHtml(key)}"${open ? "" : " hidden"}>${table}</div>` : ""}</section>`;
}

const layerTime = (value) => value.toLocaleString(undefined, { maximumFractionDigits: 2 });

function layerAnalysisResults(analysis, view) {
  const table = view.mode === "table";
  const { layers, totalUs, count } = selectLayers(analysis, {
    ...view, order: table ? "compiler" : "time",
  });
  const scaleUs = layerTimeScale(analysis);
  const label = (layer) => `<th scope="row" class="layer-name" title="${escapeHtml(layer.name)}">${escapeHtml(layer.name.replace(/^MLA_\d+\//, ""))}<small>Stage ${layer.stage} · #${layer.order}</small></th>`;
  const rows = layers.map((layer) => `<tr class="layer-family-${layerFamily(layer.name)}">
    ${label(layer)}
    ${table
      ? `<td class="layer-timing-number">${layerTime(layer.timeUs)}</td><td class="layer-share">${(totalUs > 0 ? layer.timeUs / totalUs * 100 : 0).toFixed(2)}%</td>`
      : `<td><div class="layer-timing-value"><span class="layer-bar-track" aria-hidden="true" title="${layerTime(layer.timeUs)} µs"><span style="width:${layer.timeUs / scaleUs * 100}%"></span></span><span>${layerTime(layer.timeUs)} <small>µs</small></span></div></td>`}
  </tr>`).join("");
  const axis = `<span class="layer-axis" aria-label="Scale: 0 to ${layerTime(scaleUs)} microseconds"><span>0</span><span>${layerTime(scaleUs / 2)}</span><span>${layerTime(scaleUs)}</span></span>`;
  const families = new Set(layers.map((layer) => layerFamily(layer.name)));
  const legend = Object.entries(LAYER_FAMILIES).filter(([id]) => families.has(id))
    .map(([id, title]) => `<span class="layer-family-${id}"><i aria-hidden="true"></i>${title}</span>`).join("");
  return `<p class="layer-results" role="status">${layers.length.toLocaleString()} of ${count.toLocaleString()} layers · ${table ? "compiler order" : "slowest first"}</p>
    ${layers.length ? `${table ? "" : `<div class="layer-legend" aria-label="Layer families inferred from compiled names"><span>Family (from name)</span>${legend}</div>`}
    <div class="layer-table-scroll" tabindex="0" role="region" aria-label="${table ? "Compiled layer timings" : "Layer execution time chart"}"><table class="layer-table ${table ? "layer-numeric-table" : "layer-duration-table"}">
      <thead><tr><th scope="col">Compiled layer</th><th scope="col">Execution time <span class="layer-time-unit">(µs)</span>${table ? "" : axis}</th>${table ? '<th scope="col">Share (%)</th>' : ""}</tr></thead>
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
    <div class="layer-view-toggle" role="group" aria-label="Layer statistics view">${["duration", "table"].map((mode) => `<button type="button" data-layer-filter="mode" value="${mode}" aria-pressed="${view.mode === mode}">${mode === "duration" ? "Duration" : "Table"}</button>`).join("")}</div>
  </div>
  <div data-layer-results>${layerAnalysisResults(analysis, view)}</div>`;
}

function layerAnalysisSection(analysis, runId) {
  if (!analysis) return "";
  const view = state.layerViews.get(runId)?.view || { query: "", minUs: 0, mode: "duration" };
  state.layerViews.set(runId, { analysis, view });
  return disclosure(`layers:${runId}`, "Runtime Layer Analysis",
    `<div data-layer-analysis="${escapeHtml(runId)}">${layerAnalysisBody(analysis, view)}</div>`,
    `${analysis.layers.length.toLocaleString()} layers · ${layerTime(analysis.totalUs)} µs MLA`);
}

export function updateLayerAnalysis(event) {
  const control = event.target.closest("[data-layer-filter]");
  if (!control) return;
  const field = control.dataset.layerFilter;
  if (event.type !== (field === "mode" ? "click" : "input")) return;
  const panel = control.closest("[data-layer-analysis]");
  const { analysis, view } = state.layerViews.get(panel.dataset.layerAnalysis);
  view[field] = field === "minUs" ? Number(control.value) : control.value;
  panel.querySelector("[data-layer-results]").innerHTML = layerAnalysisResults(analysis, view);
  panel.querySelector("[data-layer-threshold-value]").textContent = `${layerTime(view.minUs)} µs`;
  panel.querySelectorAll('[data-layer-filter="mode"]').forEach((button) => {
    button.setAttribute("aria-pressed", String(button.value === view.mode));
  });
}

function componentBenchmark(payload, runId, benchmarkRun, modelId) {
  if (!payload) return `<p class="muted-message">No completed benchmark is linked to this component yet.</p>`;
  const adapted = adaptBenchmarkResult(payload.result);
  if (adapted.kind === "unsupported") {
    return `<p class="muted-message">This benchmark schema (v${escapeHtml(adapted.schemaVersion)}) is newer than this browser can render.</p>${payload.result_artifact_url ? `<a class="external-link" href="${escapeHtml(payload.result_artifact_url)}" target="_blank" rel="noreferrer">Open raw result ${icons.external}</a>` : ""}`;
  }
  if (adapted.kind !== "supported") return "";
  const benchmark = adapted.value;
  const tiles = KPI_METRICS.map((key) => kpiTile(key, benchmark.measurements[key])).join("");
  const details = benchmarkDetails(benchmark, payload);
  return `${tiles ? `<div class="kpis">${tiles}</div>` : ""}
    <p class="benchmark-context">${escapeHtml(benchmarkContext(benchmark))}${measurementSource(benchmarkRun, modelId)}</p>
    ${executionGraphSection(benchmark.executionGraph, runId)}
    ${layerAnalysisSection(benchmark.layerAnalysis, runId)}
    ${disclosure(`benchmark:${runId}`, "Benchmark details", details.markup, details.titles.join(" · "))}`;
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
  const ids = variant.details.map((detail) => detail.summary.run.metadata?.component_id || "model");
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

// A multi-component model shows each component's numbers under its name; they are never combined.
function componentPerformance(detail, named, number, modelId) {
  const run = detail.summary.run;
  const id = run.metadata?.component_id || "model";
  const heading = number
    ? `<h4 class="numbered"><span class="component-number">${number}</span>${escapeHtml(humanize(id))}</h4>`
    : `<h4>${escapeHtml(id)}</h4>`;
  return `<div class="component-performance">
    ${named ? heading : ""}
    ${componentBenchmark(detail.benchmark, run.id, detail.benchmarkRun, modelId)}
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
  const meta = [mpkBytes ? `MPK ${formatBytes(mpkBytes)}` : "", parameters.precision && precisionLabel(parameters.precision)]
    .filter(Boolean).join(" · ");
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

// The first metrics in display order are shown as tiles; any further metrics are listed in a table.
const ACCURACY_SUMMARY_TILES = 4;

// The registry does not preserve a result's metric order, so known metrics follow the label order and
// any others follow them in the order they arrive.
function orderedAccuracyMetrics(report) {
  const metrics = report.metrics || {};
  return orderedMetricNames(Object.keys(metrics)).map((name) => [name, metrics[name]]);
}

// The DevKit accuracy run wins over the legacy result on the benchmark run, which wins over compile-run
// accuracy. Each result carries the run that measured it.
function accuracyResult(detail, devkitPayload) {
  const fallback = (detail.tests.test_results || [])
    .find((item) => item.suite === "accuracy" && item.summary?.schema_version === 1);
  const legacyDevkit = (detail.benchmarkTests?.test_results || [])
    .find((item) => item.suite === "accuracy" && item.summary?.schema_version === 2);
  if (devkitPayload?.result) return { report: devkitPayload.result, run: devkitPayload.run };
  if (legacyDevkit) return { report: legacyDevkit.summary, run: detail.benchmarkRun };
  return { report: detail.accuracy?.result || fallback?.summary, run: detail.summary.run };
}

function variantAccuracy(variant) {
  return variant.details.map((detail) => accuracyResult(detail, variant.devkit))
    .find(({ report }) => [1, 2].includes(report?.schema_version)) || null;
}

// The first accuracy metric of a build and the source model's published score, for the Compare page.
export function accuracyHeadline(variant) {
  const report = variantAccuracy(variant)?.report;
  if (report?.schema_version === 1) return { label: "Top-1", value: percentage(report.metrics?.candidate?.top1_accuracy), published: "" };
  const [name, metric] = report ? orderedAccuracyMetrics(report)[0] || [] : [];
  if (!name) return null;
  const published = publishedScores(variant.published, report.dataset?.id, report.metrics)[name];
  return { label: accuracyMetricLabel(name), value: accuracyMetricValue(metric), published: publishedAccuracyValue(published) };
}

function accuracyPanel(result, modelId, published = []) {
  const report = result?.report;
  const source = measurementSource(result?.run, modelId);
  const publishedMarkup = published.length
    ? disclosure("published-scores", "Published scores", publishedTables(published), `${published.length} ${published.length === 1 ? "source" : "sources"}`)
    : "";
  if (report?.schema_version === 2) {
    const scores = publishedScores(published, report.dataset?.id, report.metrics);
    return accuracySection(report, accuracyMetricTiles(report, scores), accuracyContextV2(report), accuracyMetricTable(report, scores), source, publishedMarkup);
  }
  if (!report || report.schema_version !== 1) return publishedAccuracySection(published);
  const candidate = report.metrics?.candidate || {};
  const reference = report.metrics?.reference || {};
  const comparison = report.metrics?.comparison || {};
  const evaluation = report.evaluation || {};
  return accuracySection(report, `
      <div class="accuracy-metric primary"><span>Candidate top-1</span><strong>${percentage(candidate.top1_accuracy)}</strong><small>Reference ${percentage(reference.top1_accuracy)}</small></div>
      <div class="accuracy-metric primary"><span>Candidate top-5</span><strong>${percentage(candidate.top5_accuracy)}</strong><small>Reference ${percentage(reference.top5_accuracy)}</small></div>
      <div class="accuracy-metric"><span>Top-1 agreement</span><strong>${percentage(comparison.top1_agreement)}</strong><small>Candidate vs reference</small></div>
      <div class="accuracy-metric"><span>Top-1 delta</span><strong>${percentage(comparison.top1_delta, { signed: true })}</strong><small>Top-5 ${percentage(comparison.top5_delta, { signed: true })}</small></div>
    `, `${keyValue("Profile", evaluation.profile)}${keyValue("Scope", report.report_scope)}${keyValue("Workers", report.execution?.workers)}${keyValue("Throughput", Number.isFinite(report.execution?.samples_per_second) ? `${report.execution.samples_per_second.toFixed(2)} images/s` : null)}`, "", source, publishedMarkup);
}

function accuracyMetricValue(metric) {
  if (!Number.isFinite(metric?.value)) return "—";
  return metric.unit === "ratio" ? percentage(metric.value) : `${metric.value.toFixed(4)} ${metric.unit}`;
}

// Published scores are ratios, as taxonomies/accuracy-metrics.yaml requires.
function publishedAccuracyValue(value) {
  return Number.isFinite(value) ? percentage(value) : "";
}

function accuracyMetricTiles(report, scores) {
  return orderedAccuracyMetrics(report).slice(0, ACCURACY_SUMMARY_TILES).map(([name, metric], index) => {
    const published = publishedAccuracyValue(scores[name]);
    return `<div class="accuracy-metric${index === 0 ? " primary" : ""}"><span>${escapeHtml(accuracyMetricLabel(name))}</span><strong>${escapeHtml(accuracyMetricValue(metric))}</strong><small>${escapeHtml(published ? `Published ${published}` : "")}</small></div>`;
  }).join("");
}

function accuracyMetricTable(report, scores) {
  const rows = orderedAccuracyMetrics(report).slice(ACCURACY_SUMMARY_TILES).map(([name, metric]) => (
    `<tr><th scope="row">${escapeHtml(accuracyMetricLabel(name))}</th><td>${escapeHtml(accuracyMetricValue(metric))}</td><td>${escapeHtml(publishedAccuracyValue(scores[name]) || "—")}</td></tr>`
  )).join("");
  if (!rows) return "";
  return `<div class="benchmark-measurements"><h4>Additional metrics</h4><div class="benchmark-measurements-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">Value</th><th scope="col">Published</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

// Conditions are written as a phrase, such as "640x640, FP32"; notes may already end a sentence.
function sentence(text = "") {
  return !text || /[.!?]$/.test(text) ? text : `${text}.`;
}

// Each source the model's authors published scores in, with the dataset and conditions they measured on.
function publishedTables(published) {
  return published.map((entry) => {
    const metrics = entry.metrics || {};
    const rows = orderedMetricNames(Object.keys(metrics)).map((name) => (
      `<tr><th scope="row">${escapeHtml(accuracyMetricLabel(name))}</th><td>${escapeHtml(publishedAccuracyValue(metrics[name]) || "—")}</td></tr>`
    )).join("");
    const source = trustedHttpsUrl(entry.source);
    const link = source ? ` <a class="measurement-link" href="${escapeHtml(source.href)}" target="_blank" rel="noreferrer">Source${icons.external}</a>` : "";
    return `<div class="benchmark-measurements published-scores"><h4>${escapeHtml(`${entry.dataset?.id || ""} ${entry.dataset?.split || ""}`.trim())}</h4>
      <p class="disclosure-text">${escapeHtml(sentence(entry.conditions))}${entry.note ? ` ${escapeHtml(sentence(entry.note))}` : ""}${link}</p>
      <div class="benchmark-measurements-wrap"><table><thead><tr><th scope="col">Metric</th><th scope="col">Published</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }).join("");
}

// A build without an accuracy result still shows what the model's authors published.
function publishedAccuracySection(published) {
  if (!published.length) return "";
  return `<section id="accuracy-section" class="accuracy-section" tabindex="-1">
    <div class="detail-section-title"><h3>Accuracy</h3></div>
    <p class="accuracy-scope">No accuracy result for this build yet. The model's authors published these scores for the source model.</p>
    ${publishedTables(published)}
  </section>`;
}

function accuracyContextV2(report) {
  return `${keyValue("Validation", report.validation?.id)}${keyValue("Measured on", "DevKit")}${keyValue("Core", report.execution?.core_version)}`;
}

function accuracySection(report, metricsMarkup, contextMarkup, tableMarkup = "", sourceMarkup = "", publishedMarkup = "") {
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
    <p class="accuracy-scope"><strong>${escapeHtml(sampleText)}</strong> · ${escapeHtml(dataset.id || "Dataset unavailable")} ${escapeHtml(dataset.variant || "")} / ${escapeHtml(dataset.split || "")}${sourceMarkup}</p>
    <div class="accuracy-summary">${metricsMarkup}</div>
    ${disclosure("accuracy", "Evaluation details", details, coverageLabel)}
    ${publishedMarkup}
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

function cliDownloadCommand(variant) {
  return modelDownloadCommand(variant.modelId, variant.profile, elements["branch-filter"].value || branchName(variant.anchor.git_ref), state.apiBase.includes(".stg."));
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
  const components = hasArtifacts && !(fullPackage && singleComponent)
    ? `<button class="action-button" type="button" data-download-variant="${escapeHtml(run.id)}" data-run-id="${escapeHtml(run.id)}">${icons.download}<span>Download components</span></button>`
    : "";
  if (!fullPackage) return components ? `<div class="package-actions">${components}</div>` : "";
  return `<div class="package-card">
    <div class="package-card-copy"><strong>Full package</strong><span>${packageFormat(packageArtifact) === "Folder" ? "Download the model folder with sima-cli using its verified manifest." : packageFormat(packageArtifact) === "MPK" ? "The compiled model in one verified MPK." : "All components and declared runtime resources in one verified ZIP."}</span></div>
    ${packageArtifact ? `<dl class="package-facts"><div><dt>Size</dt><dd>${escapeHtml(formatBytes(packageArtifact.metadata?.deployment_size_bytes ?? packageArtifact.size_bytes))}</dd></div><div><dt>SHA-256</dt><dd class="artifact-hash"><code>${escapeHtml(packageArtifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(packageArtifact.sha256)}" aria-label="Copy package checksum">${icons.copy}</button></dd></div></dl>` : ""}
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
  // A reused build keeps the categories it was compiled with; the catalog has the current ones.
  const categories = catalogModel?.categories || run.model_categories || {};
  const license = run.metadata?.license;
  // CI reuses a build until its compile inputs change, so the build's commit can predate the scripts
  // that measured it. The source link opens the commit the model card comes from; Build provenance
  // keeps the build's own commit.
  const sourceCommit = catalogModel?.documentation?.source_commit || run.commit_sha;
  const sourceUrl = githubModelUrl(run, model, sourceCommit);
  const sourceLabel = `View ${model} source on GitHub at commit ${shortSha(sourceCommit)}`;
  const accuracyMarkup = accuracyPanel(variantAccuracy(variant), model, variant.published || []);
  const overviewMarkup = overviewSection(catalogModel);
  const hasArtifacts = details.some((detail) => modelPackArtifacts(detail.summary.artifacts).length);
  // Every component's MPK must pass validation for the build to carry the mark.
  const validated = details.every((detail) => mpkValidationMark(detail.tests.test_results || []));
  const fullPackage = variantPackage(variant.runs);
  const packageArtifact = (variant.packageSummary?.artifacts || []).find((item) => item.artifact_type === "model-package");
  const originalModel = inputs.map(originalModelLink).find(Boolean);
  const builtAt = Math.max(...variant.runs.map((item) => Date.parse(item.finished_at || item.created_at || "") || 0));
  const singleComponent = details.length === 1;
  const order = pipelineOrder(variant);
  const performanceDetails = order
    ? [...details].sort((a, b) => order.indexOf(a.summary.run.metadata?.component_id) - order.indexOf(b.summary.run.metadata?.component_id))
    : details;
  const backLabel = elements["result-title"].textContent || "Models";
  const downloadAction = fullPackage
    ? `<button class="download-action" type="button" data-download="package" data-run-id="${escapeHtml(fullPackage.run_id)}">${icons.download}<span>Download</span>${packageArtifact ? `<small>${escapeHtml(formatBytes(packageArtifact.metadata?.deployment_size_bytes ?? packageArtifact.size_bytes))}</small>` : ""}</button>`
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
        <div class="detail-tags">${(categories.tasks || []).map((task) => detailTag(icons.task, taxonomyLabel("tasks", task), "task")).join("")}${categories.architecture?.id ? detailTag(icons.architecture, taxonomyLabel("architectures", categories.architecture.id), "architecture") : ""}${categories.model_type ? detailTag(icons.modelType, taxonomyLabel("model_types", categories.model_type), "model-type") : ""}${precisions.length ? detailTag(icons.precision, precisions.map(precisionLabel).join(" + "), "precision") : ""}${detailTag(icons.compiler, `${run.toolchain_name || "compiler"} ${run.toolchain_version || ""}`.trim(), "compiler")}${detailTag(icons.branch, branchName(run.git_ref), "branch")}${license?.spdx ? detailTag(icons.license, license.spdx, "license") : ""}</div>
        ${catalogModel?.description ? `<p class="lede">${escapeHtml(catalogModel.description)}</p>` : ""}
      </div>
      <div class="detail-title-actions">
        ${downloadAction}
        ${catalogModelOfVariant(variant) ? `<button class="cli-action" type="button" data-copy="${escapeHtml(cliDownloadCommand(variant))}">${icons.copy}<span>Copy CLI command</span></button>` : ""}
        <div class="detail-links">${validated ? mpkValidationMark(allTests) : ""}${originalModel ? `<a class="original-model-link${originalModel.huggingFace ? "" : " external-model-link"}" href="${escapeHtml(originalModel.href)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(originalModel.label)}" data-tooltip="${escapeHtml(originalModel.label)}">${originalModel.huggingFace ? icons.huggingFace : icons.external}</a>` : ""}${sourceUrl ? `<a class="model-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHtml(sourceLabel)}" data-tooltip="${escapeHtml(sourceLabel)}">${icons.github}</a>` : ""}</div>
      </div>
    </header>
    ${detailBuildSwitcher(variant)}
    ${sectionNavigation(sections)}

    ${overviewMarkup}

    <section id="components-section" class="component-section" tabindex="-1"><div class="detail-section-title"><h3>Performance on ${escapeHtml(targetName(run))}</h3></div>
      ${singleComponent ? "" : `<p class="section-lede">This model runs as ${details.length} components, each measured on its own.</p>`}
      ${order ? pipelineStrip(variant.pipeline) : ""}
      ${performanceDetails.map((detail, index) => componentPerformance(detail, !singleComponent, order && index + 1, model)).join("")}
      ${singleComponent ? "" : `<p class="section-note">Each component is measured on its own. The CPU stage between components is not measured, so component latencies do not add up to end-to-end latency.</p>`}
    </section>

    ${accuracyMarkup}

    <section id="details-section" class="details-section" tabindex="-1"><div class="detail-section-title"><h3>Details</h3></div>
      ${packageCard(variant, fullPackage, packageArtifact, hasArtifacts, singleComponent)}
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
