// Reads the whole-model reports a build profile's benchmark runs publish
// (schemas/model-report.schema.json, written by tools/model_report).
import { analyzeLayers } from "./mla-layer-stats.js";

// Each benchmark stage's registry run type, as tools/model_registry/fingerprints.py BENCHMARK_RUN_TYPES records it.
export const BENCHMARK_RUN_TYPES = {
  performance: "model_performance",
  accuracy: "model_accuracy",
};

// The Sentinel reading behind each telemetry metric, as tools/model_report/metrics.json names it, and the
// field whose entries break it down. A list field reports its busiest entry, one per CPU core or board sensor.
const TELEMETRY_METRICS = {
  avg_power_watts: ["power_som_watts", "mean"],
  peak_power_watts: ["power_som_watts", "maximum", "power_rails_watts"],
  avg_mla_power_watts: ["power_mla_watts", "mean"],
  peak_mla_power_watts: ["power_mla_watts", "maximum"],
  avg_die_temperature_c: ["die_temperature_max_c", "mean"],
  peak_die_temperature_c: ["die_temperature_max_c", "maximum"],
  avg_board_temperature_c: ["board_temperatures_c", "mean"],
  peak_board_temperature_c: ["board_temperatures_c", "maximum"],
  avg_cpu_usage_pct: ["cpu_usage_pct", "mean"],
  peak_cpu_usage_pct: ["cpu_usage_pct", "maximum"],
  avg_cpu_core_usage_pct: ["cpu_cores_usage_pct", "mean"],
  peak_cpu_core_usage_pct: ["cpu_cores_usage_pct", "maximum"],
  mla_memory_avg_mib: ["mla_memory_mib", "mean"],
  mla_memory_peak_mib: ["mla_memory_mib", "maximum"],
  ev74_memory_avg_mib: ["ev74_memory_mib", "mean"],
  ev74_memory_peak_mib: ["ev74_memory_mib", "maximum"],
  avg_linux_memory_mib: ["linux_memory_used_mib", "mean"],
  peak_linux_memory_mib: ["linux_memory_used_mib", "maximum"],
};

// The customer view leaves the board's Linux memory out of a performance report: every process on the board
// uses it, so it describes the board, not the model.
export const CUSTOMER_HIDDEN_METRICS = ["avg_linux_memory_mib", "peak_linux_memory_mib"];
// The Sentinel reading behind those metrics.
const CUSTOMER_HIDDEN_TELEMETRY = [...new Set(CUSTOMER_HIDDEN_METRICS.map((key) => TELEMETRY_METRICS[key][0]))];

function withoutKeys(record, keys) {
  return Object.fromEntries(Object.entries(record).filter(([key]) => !keys.includes(key)));
}

// One set of readings: a whole-model report, a component measured on its own, or a generation workload.
function customerReadings(readings) {
  return {
    ...readings,
    ...(readings.metrics ? { metrics: withoutKeys(readings.metrics, CUSTOMER_HIDDEN_METRICS) } : {}),
    ...(readings.telemetry ? { telemetry: withoutKeys(readings.telemetry, CUSTOMER_HIDDEN_TELEMETRY) } : {}),
  };
}

function customerWorkload(workload) {
  return { ...customerReadings(workload), ...(workload.baseline ? { baseline: customerWorkload(workload.baseline) } : {}) };
}

// A performance report as the customer view may show it. Every reader of the page receives this report, so
// the Memory card and the CSV leave the same readings out.
export function customerPerformanceReport(report) {
  const components = Object.entries(report.components || {}).map(([id, component]) => [id, customerReadings(component)]);
  return {
    ...customerReadings(report),
    ...(report.components ? { components: Object.fromEntries(components) } : {}),
    ...(report.generation?.workloads
      ? { generation: { ...report.generation, workloads: report.generation.workloads.map(customerWorkload) } } : {}),
  };
}

function finite(value) {
  return Number.isFinite(value) ? value : null;
}

function pyneatVersion(report) {
  if (typeof report.runtime?.llima === "string") return `LLiMa ${report.runtime.llima} · MoLE ${report.runtime.mole || ""}`;
  return typeof report.runtime?.pyneat === "string" ? `PyNeat ${report.runtime.pyneat}` : "";
}

// A performance report holds its metrics; an accuracy report holds them per dataset.
function supported(report, stage) {
  const results = stage === "accuracy" ? report?.datasets
    : report?.scope === "components" ? report.components : report?.metrics;
  return report?.schema === "sima.model-report" && report.schema_version === 1 && report.stage === stage
    && Boolean(results) && typeof results === "object";
}

// A benchmark run of one build profile: the registry run and its report. No benchmark run reads as
// missing, and another report format as unsupported.
function read(benchmarkRun, stage, adapt) {
  if (!benchmarkRun?.report) return { kind: "missing" };
  if (!supported(benchmarkRun.report, stage)) return { kind: "unsupported" };
  return { kind: "supported", value: adapt(benchmarkRun) };
}

// A report metric is a plain number in the unit metrics.js gives it. A telemetry metric also keeps the range and
// sample count of the Sentinel reading it came from, and peak SoM power the rails that make it up.
function measurement(key, value, telemetry) {
  const metric = { key, status: "measured", value, minimum: null, maximum: null, details: {} };
  const [field, statistic, breakdown] = TELEMETRY_METRICS[key] || [];
  if (!field) return metric;
  const entries = [telemetry[field]].flat().filter((entry) => entry?.samples > 0 && Number.isFinite(entry[statistic]));
  if (!entries.length) return metric;
  const selected = entries.reduce((busiest, entry) => (entry[statistic] > busiest[statistic] ? entry : busiest));
  return {
    ...metric,
    minimum: finite(selected.minimum),
    maximum: finite(selected.maximum),
    sampleCount: selected.samples,
    details: breakdown ? { breakdown: (telemetry[breakdown] || []).filter((entry) => entry?.samples > 0) } : {},
  };
}

// Where the board telemetry came from: the Sentinel daemon that sampled it, and how often.
function telemetrySource(telemetry) {
  if (!telemetry || typeof telemetry !== "object") return null;
  return {
    status: typeof telemetry.status === "string" ? telemetry.status : "",
    sentinelVersion: typeof telemetry.sentinel_version === "string" ? telemetry.sentinel_version : "",
    samples: Number.isInteger(telemetry.samples) ? telemetry.samples : 0,
    intervalMs: finite(telemetry.sample_interval_ms),
  };
}

// The average and peak load of each CPU core, in core order.
function cpuCores(telemetry) {
  const cores = telemetry?.cpu_cores_usage_pct;
  if (!Array.isArray(cores)) return [];
  return cores.filter((core) => Number.isFinite(core?.mean))
    .map((core) => ({ mean: core.mean, maximum: finite(core.maximum) }));
}

// Reports from before EV74 memory carry the whole CMA pool in use, which is not the model's memory.
const RETIRED_METRICS = new Set(["avg_cma_memory_mib", "peak_cma_memory_mib"]);

function generationWorkloads(generation) {
  const adapt = workload => {
    const telemetry = workload.telemetry || {};
    const values = Object.fromEntries(Object.entries(workload.metrics || {}).map(([key, metric]) => [key, metric.value]));
    for (const [key, [field, statistic]] of Object.entries(TELEMETRY_METRICS)) {
      const entries = [telemetry[field]].flat().filter(entry => entry?.samples > 0 && Number.isFinite(entry[statistic]));
      if (entries.length) values[key] = Math.max(...entries.map(entry => entry[statistic]));
    }
    return {
      ...workload,
      key: workload.kind === "text" ? `text_${workload.input_tokens}` : workload.id,
      label: workload.dataset ? ({gsm8k: "GSM8K", "bfcl-v3": "BFCL v3", alpacaeval: "AlpacaEval",
        realworldqa: "RealWorldQA", docvqa_val: "DocVQA", chartqa: "ChartQA"}[workload.dataset.name] || workload.dataset.name)
        : workload.kind === "text" ? `${workload.input_tokens} input tokens` : `Image ${workload.image_size.join(" × ")} + text`,
      baseline: workload.baseline ? adapt(workload.baseline) : null,
      inputTokens: workload.input_tokens,
      measurements: Object.fromEntries(Object.entries(values).filter(([, value]) => Number.isFinite(value))
        .map(([key, value]) => [key, measurement(key, value, telemetry)])),
      telemetry: telemetrySource(telemetry),
      cpuCores: cpuCores(telemetry),
    };
  };
  return (generation?.workloads || []).map(adapt);
}

function performanceReadings(report) {
  const telemetry = report.telemetry || {};
  return {
    measurements: Object.fromEntries(Object.entries(report.metrics || {})
      .filter(([key, value]) => Number.isFinite(value) && !RETIRED_METRICS.has(key))
      .map(([key, value]) => [key, measurement(key, value, telemetry)])),
    workload: typeof report.workload === "string" ? report.workload : "",
    telemetry: telemetrySource(report.telemetry),
    cpuCores: cpuCores(report.telemetry),
  };
}

// A performance report measures the whole model, however many components it runs, and records each
// component's layer timings and execution plan. The registry stores the report as PostgreSQL jsonb, which
// reorders its keys, so `order` gives the component IDs in the order the model runs them.
export function readPerformance(benchmarkRun, order = []) {
  return read(benchmarkRun, "performance", ({ run, report }) => {
    return {
      ...performanceReadings(report),
      scope: report.scope || "model",
      tool: pyneatVersion(report),
      generation: report.generation || null,
      workloads: generationWorkloads(report.generation),
      timestamp: run?.finished_at || run?.created_at || null,
      modelId: run?.metadata?.model_id,
      buildProfileId: run?.metadata?.build_profile_id,
      components: inRunOrder(Object.entries(report.components || {}), order).map(([id, component]) => ({
        id,
        ...(report.scope === "components" ? performanceReadings(component) : {}),
        layerAnalysis: analyzeLayers(component?.mla_layer_stats?.layers),
        mlaStageStats: component?.mla_stage_stats || null,
        // Inferences in each MLA run, which Core's execution plan records for every model.
        batch: component?.execution_plan?.model?.batch,
        executionGraph: executionGraph(component?.execution_plan),
      })),
    };
  });
}

// Report entries with the IDs `order` names first, in its order, and any others after them as stored.
function inRunOrder(entries, order) {
  const rank = ([id]) => (order.includes(id) ? order.indexOf(id) : order.length);
  return [...entries].sort((left, right) => rank(left) - rank(right));
}

// An accuracy report scores the whole model on each dataset model.yaml names, as ratios over the evaluated part
// of it. The run's parameters give each dataset's registry entry and model.yaml's order; the first is the headline.
export function readAccuracy(benchmarkRun) {
  return read(benchmarkRun, "accuracy", ({ run, parameters, report, datasetNames }) => {
    const declared = parameters?.accuracy?.datasets || {};
    const names = [...Object.keys(declared), ...Object.keys(report.datasets)]
      .filter((name, index, all) => all.indexOf(name) === index && report.datasets[name]);
    return {
      datasets: names.map((name) => {
        const { metrics, details, ...evaluated } = report.datasets[name];
        const dataset = { ...declared[name], ...evaluated };
        return {
          name,
          metrics: Object.fromEntries(Object.entries(metrics || {}).filter(([, value]) => Number.isFinite(value))),
          dataset: datasetNames?.[dataset.id] ? { ...dataset, name: datasetNames[dataset.id] } : dataset,
          details: details && typeof details === "object" ? details : {},
        };
      }),
      tool: pyneatVersion(report),
      timestamp: run?.finished_at || run?.created_at || null,
    };
  });
}

// A quantization parameter is one number, or one per channel for a per-axis tensor.
function quantText(value, format) {
  if (Array.isArray(value)) return "per channel";
  return Number.isFinite(value) ? format(value) : "—";
}

// The steps the benchmarked Model ran, in order, and the tensors on each connection, read from the
// Core execution plan the performance script records. Input and Output are the graph's ends; the MLA
// step is the compiled model. Another plan schema shows no graph.
export function executionGraph(plan) {
  if (plan?.schema !== "sima.neat.execution_plan" || plan.schema_version !== 1
    || !Array.isArray(plan.nodes) || !Array.isArray(plan.edges)) return null;
  const steps = plan.nodes.map((node) => (["Input", "Output"].includes(node.kind)
    ? { name: node.kind, runsOn: null }
    : { name: node.runs_on === "MLA" ? "Compiled Model" : String(node.kind), runsOn: String(node.runs_on) }));
  const names = new Map(plan.nodes.map((node, index) => [node.id, steps[index].name]));
  const tensors = plan.edges.flatMap((edge) => (Array.isArray(edge.tensors) ? edge.tensors : []).map((tensor, index) => ({
    connection: index === 0 ? `${names.get(edge.from) ?? edge.from} → ${names.get(edge.to) ?? edge.to}` : "",
    dtype: String(tensor.dtype),
    shape: Array.isArray(tensor.shape) ? tensor.shape.join("×") : "—",
    scale: quantText(tensor.quant?.scale, (value) => (value >= 1 ? value.toFixed(2) : String(Number(value.toPrecision(3))))),
    zeroPoint: quantText(tensor.quant?.zero_point, String),
  })));
  return { steps, tensors };
}
