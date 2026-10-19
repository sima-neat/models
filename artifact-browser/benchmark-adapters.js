import { analyzeLayers } from "./layer-stats.js";

function measurement(key, metric) {
  if (!metric || typeof metric !== "object") return null;
  const declaredStatus = ["measured", "unavailable", "unsupported"].includes(metric.status)
    ? metric.status
    : "unavailable";
  const invalidMeasuredValue = declaredStatus === "measured" && !Number.isFinite(metric.value);
  const status = invalidMeasuredValue ? "unavailable" : declaredStatus;
  return {
    key,
    status,
    reason: invalidMeasuredValue ? "invalid_value" : (typeof metric.reason === "string" ? metric.reason : null),
    value: status === "measured" && Number.isFinite(metric.value) ? metric.value : null,
    unit: typeof metric.unit === "string" ? metric.unit : "",
    aggregation: typeof metric.aggregation === "string" ? metric.aggregation : "unknown",
    minimum: finite(metric.minimum),
    maximum: finite(metric.maximum),
    standardDeviation: finite(metric.standard_deviation),
    sampleCount: Number.isInteger(metric.sample_count) ? metric.sample_count : 0,
    details: metric.details && typeof metric.details === "object" ? metric.details : {},
  };
}

function finite(value) {
  return Number.isFinite(value) ? value : null;
}

// Formats 1-5 stored each metric already converted. Format 6 stores the tool's saved report, so its
// metrics are read here, under the names and from the fields tools/model_registry/benchmark_metrics.py uses.
const REPORT_METRICS = {
  latency_avg_ms: ["latency_ms", "ms"],
  latency_p50_ms: ["latency_p50_ms", "ms"],
  latency_p95_ms: ["latency_p95_ms", "ms"],
  latency_p99_ms: ["latency_p99_ms", "ms"],
  throughput_fps: ["fps", "inferences/s"],
  avg_power_watts: ["avg_power_watts", "W"],
  energy_joules: ["energy_joules", "J"],
};
// Each Sentinel reading: its telemetry field, the statistic, and the field whose entries break it down.
// A list field reports its busiest entry, one per CPU core or board sensor.
const TELEMETRY_METRICS = {
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
  avg_cma_memory_mib: ["cma_memory_mib", "mean"],
  peak_cma_memory_mib: ["cma_memory_mib", "maximum"],
  avg_linux_memory_mib: ["linux_memory_used_mib", "mean"],
  peak_linux_memory_mib: ["linux_memory_used_mib", "maximum"],
};

function measured(value, unit, aggregation, sampleCount) {
  return { status: "measured", value, unit, aggregation, sample_count: sampleCount };
}

// The metrics a saved Core report measured. Core reports 0 for a headline value it did not measure, and
// 0 samples for a reading Sentinel did not provide; those stay out, as in the registry.
export function reportMetrics(result) {
  const report = result.benchmark_report || {};
  const metrics = {};
  for (const [key, metric] of Object.entries(report.metrics || {})) {
    metrics[key] = {...metric, status: "measured"};
  }
  for (const [key, [field, unit]] of Object.entries(REPORT_METRICS)) {
    if (report[field] > 0) metrics[key] = measured(report[field], unit, "raw", 1);
  }
  const inferences = (result.configuration?.samples ?? 0) * (result.configuration?.workload?.batch_size ?? 0);
  if (metrics.energy_joules && inferences > 0) {
    metrics.energy_per_inference_joules = measured(report.energy_joules / inferences, "J/inference", "raw", 1);
  }
  const telemetry = report.telemetry || {};
  for (const [key, [field, statistic, breakdown]] of Object.entries(TELEMETRY_METRICS)) {
    const entries = [telemetry[field]].flat().filter((entry) => entry?.samples > 0 && Number.isFinite(entry[statistic]));
    if (!entries.length) continue;
    const selected = entries.reduce((busiest, entry) => (entry[statistic] > busiest[statistic] ? entry : busiest));
    const unit = selected.unit === "C" ? "°C" : selected.unit;
    metrics[key] = {
      ...measured(selected[statistic], unit, statistic === "mean" ? "mean" : "max", selected.samples),
      minimum: selected.minimum,
      maximum: selected.maximum,
    };
    if (breakdown) metrics[key].details = { breakdown: (telemetry[breakdown] || []).filter((entry) => entry?.samples > 0) };
  }
  return metrics;
}

function common(result) {
  const report = result.schema_version >= 6 ? result.benchmark_report : result.core_report;
  const stored = result.schema_version >= 6 ? reportMetrics(result) : result.metrics || {};
  const measurements = Object.fromEntries(Object.entries(stored)
    .map(([key, metric]) => [key, measurement(key, metric)])
    .filter(([, metric]) => metric));
  return {
    schemaVersion: result.schema_version,
    synthetic: result.provenance?.synthetic === true,
    workload: result.schema_version >= 5
      ? `${result.configuration?.workload?.selection === "mpk_default" ? "Core default MPK" : result.configuration?.workload?.selection || "Unknown workload"} · ${result.configuration?.workload?.preprocess || "unknown"} preprocessing · ${result.configuration?.workload?.postprocess || "unknown"} postprocessing`
      : "Historical benchmark configuration",
    status: result.status,
    tool: [result.tool?.name, result.tool?.version].filter((part) => typeof part === "string" && part).join(" "),
    timestamp: result.timestamp_utc,
    workloads: report?.workloads,
    settings: result.configuration,
    timing: report?.timing,
    measurements,
    samples: result.configuration?.samples,
    platformVersion: result.compatibility?.actual_platform_version,
    neatVersion: result.compatibility?.actual_neat_version,
    mlaFrequencyHz: result.environment?.mla_frequency_hz,
    cpuFrequencyKhz: result.environment?.cpu_frequency_khz,
    modelId: result.model?.id,
    buildProfileId: result.model?.build_profile_id,
    componentId: result.model?.component_id,
    cpuCount: result.environment?.cpu_count,
    batchSize: result.configuration?.workload?.batch_size,
    executionGraph: executionGraph(result.execution_plan ?? report?.execution_plan),
    telemetry: telemetrySource(report),
    cpuCores: cpuCores(report),
    layerAnalysis: analyzeLayers(report?.layer_stats),
  };
}

// Where the board telemetry came from, read from the Core report: the Sentinel daemon that sampled
// it and how much of the throughput window it covered. Reports before Sentinel carry none.
function telemetrySource(report) {
  const telemetry = report?.telemetry;
  if (!telemetry || typeof telemetry !== "object") return null;
  return {
    status: typeof telemetry.status === "string" ? telemetry.status : "",
    sentinelVersion: typeof telemetry.sentinel_version === "string" ? telemetry.sentinel_version : "",
    samples: Number.isInteger(telemetry.samples) ? telemetry.samples : 0,
    intervalMs: finite(telemetry.sample_interval_ms),
    windowMs: finite(report.throughput_duration_ms),
  };
}

// The average and peak load of each CPU core, in core order, from the Core report's Sentinel telemetry.
function cpuCores(report) {
  const cores = report?.telemetry?.cpu_cores_usage_pct;
  if (!Array.isArray(cores)) return [];
  return cores.filter((core) => Number.isFinite(core?.mean))
    .map((core) => ({ mean: core.mean, maximum: finite(core.maximum) }));
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

function adaptResult(result) {
  return { ...common(result), orchestration: result.orchestration || null };
}

// Core and LLiMa use the same result envelope and reader.
export const benchmarkAdapters = new Map([1, 2, 3, 4, 5, 6].map((version) => [version, adaptResult]));

export function adaptBenchmarkResult(result) {
  if (!result || typeof result !== "object") return { kind: "missing" };
  const adapter = benchmarkAdapters.get(result.schema_version);
  if (!adapter) return { kind: "unsupported", schemaVersion: result.schema_version };
  return { kind: "supported", value: adapter(result) };
}

export function generativeWorkloads(result) {
  const adapted = adaptBenchmarkResult(result);
  if (adapted.kind !== "supported" || !Array.isArray(adapted.value.workloads)) return [];
  return adapted.value.workloads.map((workload) => ({
    ...adaptBenchmarkResult({...result, benchmark_report: workload}).value,
    key: workload.kind === "text" ? `text_${workload.input_tokens}` : `image_${workload.image_size.join("x")}`,
    kind: workload.kind,
    inputTokens: workload.input_tokens,
    label: workload.kind === "text" ? `${workload.input_tokens} input tokens` : `Image ${workload.image_size.join(" × ")} + text`,
  })).sort((a, b) => (a.inputTokens ?? Infinity) - (b.inputTokens ?? Infinity) || a.key.localeCompare(b.key));
}
