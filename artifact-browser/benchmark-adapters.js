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

function temperatureSeries(result) {
  const environment = result.environment || {};
  const observations = environment.observations || {};
  const explicit = Array.isArray(environment.temperature_samples)
    ? environment.temperature_samples
    : [];
  const snapshots = explicit.length
    ? explicit
    : [
      observations.start && { ...observations.start, phase: "start", trial: null },
      observations.end && { ...observations.end, phase: "end", trial: null },
    ].filter(Boolean);
  const series = new Map();
  snapshots.forEach((snapshot, sampleIndex) => {
    const timestamp = Date.parse(snapshot.timestamp_utc);
    if (!Number.isFinite(timestamp)) return;
    const duplicateCounts = new Map();
    (snapshot.temperatures_c || []).forEach((reading) => {
      if (!Number.isFinite(reading.celsius)) return;
      const sensor = String(reading.sensor || "Temperature");
      const source = String(reading.source || "");
      const duplicateKey = `${source}\u0000${sensor}`;
      const occurrence = duplicateCounts.get(duplicateKey) || 0;
      duplicateCounts.set(duplicateKey, occurrence + 1);
      const id = String(reading.sensor_id || `${duplicateKey}\u0000${occurrence}`);
      if (!series.has(id)) {
        series.set(id, {
          id,
          sensor,
          source,
          label: source ? `${sensor} · ${source}` : `${sensor}${occurrence ? ` #${occurrence + 1}` : ""}`,
          samples: [],
          legacy: !explicit.length,
        });
      }
      series.get(id).samples.push({
        timestamp,
        celsius: reading.celsius,
        phase: snapshot.phase || "sample",
        trial: Number.isInteger(snapshot.trial) ? snapshot.trial : null,
        sampleIndex,
      });
    });
  });
  return [...series.values()]
    .filter((item) => item.samples.length >= 2)
    .sort((left, right) => left.label.localeCompare(right.label));
}

function common(result) {
  const measurements = Object.fromEntries(Object.entries(result.metrics || {})
    .map(([key, metric]) => [key, measurement(key, metric)])
    .filter(([, metric]) => metric));
  return {
    schemaVersion: result.schema_version,
    workload: result.schema_version === 5
      ? `${result.configuration?.workload?.selection === "mpk_default" ? "Core default MPK" : result.configuration?.workload?.selection || "Unknown workload"} · ${result.configuration?.workload?.preprocess || "unknown"} preprocessing · ${result.configuration?.workload?.postprocess || "unknown"} postprocessing`
      : "Historical benchmark configuration",
    status: result.status,
    tool: [result.tool?.name, result.tool?.version].filter((part) => typeof part === "string" && part).join(" "),
    timestamp: result.timestamp_utc,
    measurements,
    trials: (result.trials || []).map((trial) => ({
      number: trial.trial,
      metrics: Object.fromEntries(Object.entries(trial)
        .filter(([field]) => field !== "trial")
        .map(([field, value]) => [field, finite(value)])),
    })).filter((trial) => Number.isInteger(trial.number)),
    temperatureSeries: temperatureSeries(result),
    samples: result.configuration?.samples,
    platformVersion: result.compatibility?.actual_platform_version,
    neatVersion: result.compatibility?.actual_neat_version,
    mlaFrequencyHz: result.environment?.mla_frequency_hz,
    cpuFrequencyKhz: result.environment?.cpu_frequency_khz,
    executionGraph: executionGraph(result.core_report?.execution_plan),
    telemetry: telemetrySource(result.core_report),
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

function adaptV1(result) {
  return { ...common(result), orchestration: null };
}

function adaptV2(result) {
  return { ...common(result), orchestration: result.orchestration || null };
}

function adaptV3(result) {
  return { ...common(result), orchestration: result.orchestration || null };
}

function adaptV4(result) {
  return { ...common(result), orchestration: result.orchestration || null };
}

export const benchmarkAdapters = new Map([[1, adaptV1], [2, adaptV2], [3, adaptV3], [4, adaptV4], [5, adaptV4]]);

export function adaptBenchmarkResult(result) {
  if (!result || typeof result !== "object") return { kind: "missing" };
  const adapter = benchmarkAdapters.get(result.schema_version);
  if (!adapter) return { kind: "unsupported", schemaVersion: result.schema_version };
  return { kind: "supported", value: adapter(result) };
}
