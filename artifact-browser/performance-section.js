// Performance section of the model page: headline tiles, Performance Details cards, power rails, and the CSV.
import { shows } from "./state.js";
import { icons, escapeHtml, disclosure } from "./ui.js";
import { metricDefinition, metricDisplay, formatDate } from "./metrics.js";
import { humanize, infoMark, measurementSource, tile } from "./page-parts.js";

function measuredValue(key, metric) {
  const definition = metricDefinition(key);
  const display = metricDisplay(metric, definition);
  return display && { label: definition.label, text: display.value.toFixed(display.digits), unit: display.unit };
}

function metricRow({ label, text, unit }) {
  return `<div class="metric-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(text)}<small>${escapeHtml(unit)}</small></dd></div>`;
}

// The headline metrics, one number each. Two more tiles follow them: SoC power and inferences per watt.
const KPI_METRICS = ["throughput_fps", "latency_avg_ms"];

// Metrics the page shows in no tile or card and the CSV lists with a card, such as energy with power.
const CSV_EXTRA_METRICS = { power: ["energy_per_inference_joules"] };

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
    ["Time to first token", ["ttft_ms"], "First token, including vision."],
    ["Prefill", ["prefill_tps"], "Input tokens / TTFT."],
    ["Generation", ["decode_tps"], "Generated tokens per second, excluding the first token."],
    ["Draft contribution", ["draft_contribution_pct"], "Percentage of generated tokens accepted from the draft model."],
    ["Generation speedup", ["decode_speedup"], "Speculative generation rate / baseline generation rate.\nExcludes prompt processing and the first token."],
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
  ...Object.values(CSV_EXTRA_METRICS).flat(),
  ...DETAIL_CARDS.flatMap(([, , , , rows]) => rows.flatMap(([, keys]) => (Array.isArray(keys) ? keys : []))),
]);

function kpiTile(key, metric) {
  const value = measuredValue(key, metric);
  return value ? tile(value.label, value.text, value.unit) : "";
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

// Average SoC power, and the inferences per second each watt of it delivers. No metric holds either: both
// come from the SoC rails, so a report without rail readings shows neither tile.
function socPowerTiles(measurements) {
  const watts = powerRails(measurements).filter((rail) => SOC_RAILS.includes(rail.key));
  if (watts.length !== SOC_RAILS.length) return "";
  const soc = watts.reduce((total, rail) => total + rail.mean, 0);
  const throughput = metricDisplay(measurements.throughput_fps, metricDefinition("throughput_fps"))?.value;
  return tile("SoC power", soc.toFixed(2), "W")
    + (soc > 0 && Number.isFinite(throughput) ? tile("Inferences/Watt", (throughput / soc).toFixed(1), "", "SoC Power") : "");
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
  // The raw result is the report as the registry stores it, with the readings the customer view leaves out.
  const raw = shows.rawResult
    ? `<button class="performance-action" type="button" data-download="model-report" data-run-id="${escapeHtml(run.id)}">${icons.download}Raw result</button>`
    : "";
  const csv = `<a class="performance-action" href="${escapeHtml(csvDataUrl(performanceCsv(performance)))}" download="${escapeHtml(csvFileName(performance))}">${icons.download}Download CSV</a>`;
  return {
    titles: cards.map((card) => card.title),
    markup: `<div class="detail-cards">${column(false)}${column(true)}</div>
    <footer class="performance-footer"><dl class="performance-provenance">${provenance}</dl><div class="performance-actions">${raw}${csv}</div></footer>`,
  };
}

// The CSV download: every reading of the model as one table. Context comes first, then each card's
// metrics, every SoM rail, every CPU core, the remaining measurements, and the MLA layers of each component.

function performanceCsv(performance) {
  const rows = [["Section", "Metric", "Value", "Unit", "Key", "Note"]];
  const add = (section, metric, value, unit = "", key = "", note = "") => rows.push([section, metric, value ?? "", unit, key, note]);
  const telemetry = performance.telemetry;
  const seconds = (milliseconds) => Number.isFinite(milliseconds) && (milliseconds / 1000).toFixed(1);
  [
    ["Model", performance.modelId], ["Build", performance.buildProfileId],
    ...(performance.componentId ? [["Component", performance.componentId]] : []), ["Workload", performance.workload],
    ["Decoding", performance.decodingRole],
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
  const name = [performance.modelId, performance.buildProfileId, performance.componentId,
    performance.workloadId, performance.decodingRole].filter(Boolean).join("-") || "model";
  return `${name.replace(/[^A-Za-z0-9._-]+/g, "_")}-performance.csv`;
}

function performanceContext(performance) {
  return performance.timestamp ? `Measured ${formatDate(performance.timestamp, false)}` : "";
}

// The KPI tiles, the context line, and Performance details of one performance result.
export function performanceSummary(performance, run, modelId, componentId = "") {
  const tiles = KPI_METRICS.map((key) => kpiTile(key, performance.measurements[key])).join("") + socPowerTiles(performance.measurements);
  const details = performanceDetails(performance, run);
  return `${tiles ? `<div class="kpis">${tiles}</div>` : ""}
    <p class="performance-context">${escapeHtml(performanceContext(performance))}${measurementSource(run, modelId)}</p>
    ${disclosure(`performance:${run.id}${componentId ? `:${componentId}` : ""}`, "Performance Details", details.markup, details.titles.join(" · "))}`;
}

export function generativeBenchmark(performance, run, method) {
  const benchmark = {...performance.generation, workloads: performance.workloads};
  const speculative = benchmark.settings?.decoding === "speculative";
  const methodLabel = {gemma4_mtp: "MTP", dflash: "DFlash", eagle3: "EAGLE3"}[method] || "Speculative";
  const ttftHeading = `TTFT (ms)${infoMark("Time to first token: language prefill through the first output token. Image timings also include image preprocessing and vision encoding. Text timings exclude tokenization and input preparation.")}`;
  const label = workload => workload.label;
  const cells = (workload, keys) => keys.map(key => {
    const value = workload.metrics?.[key]?.value;
    return `<td>${Number.isFinite(value) ? value.toFixed(key === "decode_speedup" ? 2 : 1) : "—"}</td>`;
  }).join("");
  const rows = benchmark.workloads.filter(workload => workload.kind === "text").map(workload =>
    `<tr><th scope="row">${escapeHtml(label(workload))}</th>${cells(workload, ["ttft_ms", "prefill_tps", "decode_tps"])}</tr>`).join("");
  const images = benchmark.workloads.filter(workload => workload.kind === "image_text").map(workload =>
    `<div class="benchmark-measurements image-performance"><h4>Image + text</h4>
      <p class="benchmark-input">One ${escapeHtml(workload.image_size.join(" × "))} image with the prompt “${escapeHtml(workload.prompt)}”</p>
      <div class="benchmark-measurements-wrap"><table aria-label="Image and text performance"><thead><tr>
        <th scope="col">${ttftHeading}</th><th scope="col">Generation (tokens/s)</th>
      </tr></thead><tbody><tr>${cells(workload, ["ttft_ms", "decode_tps"])}</tr></tbody></table></div>
    </div>`).join("");
  const datasets = benchmark.workloads.filter(workload => workload.dataset);
  const measured = workload => workload.metrics.decode_tps.sample_count;
  const subset = datasets.some(workload => measured(workload) < workload.dataset.available_samples);
  const requests = workload => subset
    ? `${measured(workload).toLocaleString()} of ${workload.dataset.available_samples.toLocaleString()}` : measured(workload);
  const baselineRequests = Math.max(0, ...datasets.map(workload => measured(workload.baseline)));
  const datasetTable = (kind, title) => {
    const workloads = benchmark.workloads.filter(workload => workload.kind === kind);
    if (!workloads.length) return "";
    const rows = workloads.map(workload => `<tr><th scope="row">${escapeHtml(label(workload))}</th>
      <td>${escapeHtml(requests(workload))}</td>${cells(workload.baseline, ["ttft_ms"])}
      ${cells(workload, ["ttft_ms"])}${cells(workload.baseline, ["decode_tps"])}${cells(workload, ["decode_tps", "draft_contribution_pct", "decode_speedup"])}</tr>`).join("");
    return `<div class="benchmark-measurements"><h4>${title}</h4><div class="benchmark-measurements-wrap">
      <table aria-label="${title}"><thead><tr><th scope="col">Dataset</th><th scope="col">Requests</th>
      <th scope="col">Baseline ${ttftHeading}</th><th scope="col">${methodLabel} ${ttftHeading}</th>
      <th scope="col">Baseline (tokens/s)</th><th scope="col">${methodLabel} (tokens/s)</th>
      <th scope="col">Draft contribution (%)</th><th scope="col">Generation speedup (×)</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  };
  const options = benchmark.workloads.map(workload =>
    `<option value="${escapeHtml(workload.id)}">${escapeHtml(label(workload))}</option>`).join("");
  const panels = benchmark.workloads.map((workload, index) => {
    const details = (reading, baseline = false) => performanceDetails({...performance, ...reading, components: [],
      ...(baseline ? {modelId: benchmark.baseline?.model_id, buildProfileId: benchmark.baseline?.build_profile_id} : {}),
      workload: label(workload), workloadId: workload.id,
      decodingRole: baseline ? "baseline" : speculative ? methodLabel.toLowerCase() : "ordinary",
      samples: reading.metrics.decode_tps.sample_count}, run).markup;
    return `<div data-genai-details="${escapeHtml(workload.id)}"${index ? " hidden" : ""}>${workload.baseline ? `<h4>${methodLabel}</h4>` : ""}${details(workload)}
      ${workload.baseline ? `<h4>Baseline</h4>${details(workload.baseline, true)}` : ""}</div>`;
  }).join("");
  const details = `<p>Board telemetry covers measured requests, excluding warmup and model loading.</p>
    <label class="workload-picker">Workload <select data-genai-workload>${options}</select></label>${panels}
    <p>Generation rates exclude the first token. Image TTFT includes image preprocessing and vision encoding.
    Draft contribution is the percentage of generated tokens accepted from the draft model. Generation speedup compares aggregate generation rates on the same dataset and output budget, excluding prompt processing and the first token.</p>`;
  return `<div data-generative-benchmark>${benchmark.synthetic ? '<p class="benchmark-context">Synthetic preview data — for visualization only.</p>' : ""}${rows ? `<div class="benchmark-measurements"><h4>Text</h4>
    <div class="benchmark-measurements-wrap"><table aria-label="Text performance"><thead><tr>
    <th scope="col">Workload</th><th scope="col">${ttftHeading}</th><th scope="col">Prefill (tokens/s)</th><th scope="col">Generation (tokens/s)</th>
    </tr></thead><tbody>${rows}</tbody></table></div></div>` : ""}${images}
    ${speculative ? `<p>Baseline: ${escapeHtml(benchmark.baseline?.display_name || benchmark.baseline?.model_id)}, without speculative decoding. ${subset ? `It measures the first ${escapeHtml(baselineRequests)} prompts of each subset, with the same output budget.` : "Both use identical prompts and output budgets."}</p>` : ""}
    ${datasetTable("text_dataset", "Text datasets")}${datasetTable("image_dataset", "Image + text datasets")}
    <p class="benchmark-context">${speculative ? `${subset ? "Seeded random subset of each dataset" : "Full datasets"} · one measurement per prompt · one excluded warmup per dataset and deployment`
      : `1 warmup · ${escapeHtml(benchmark.settings?.repetitions)} measured runs`} · empty cache · up to 256 output tokens${speculative ? "" : " (text buckets: first token plus 256 decode steps)"}</p>
    ${disclosure("genai-method", "Measurement details", details, "Heat · CPU · Power · Memory")}</div>`;
}
