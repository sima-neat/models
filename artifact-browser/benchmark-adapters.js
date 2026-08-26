function measured(metric) {
  return metric?.status === "measured" && Number.isFinite(metric.value)
    ? { value: metric.value, unit: metric.unit, minimum: metric.minimum, maximum: metric.maximum }
    : null;
}

function common(result) {
  return {
    schemaVersion: result.schema_version,
    status: result.status,
    timestamp: result.timestamp_utc,
    metrics: {
      latency: measured(result.metrics?.latency_avg_ms),
      latencyP50: measured(result.metrics?.latency_p50_ms),
      latencyP95: measured(result.metrics?.latency_p95_ms),
      throughput: measured(result.metrics?.throughput_fps),
      power: measured(result.metrics?.avg_power_watts),
      energy: measured(result.metrics?.energy_per_inference_joules),
    },
    trials: (result.trials || []).map((trial) => ({
      number: trial.trial,
      latencyMs: trial.latency_avg_ms,
      throughputFps: trial.throughput_fps,
    })).filter((trial) => Number.isFinite(trial.latencyMs)),
    samples: result.configuration?.samples,
    platformVersion: result.compatibility?.actual_platform_version,
    neatVersion: result.compatibility?.actual_neat_version,
  };
}

function adaptV1(result) {
  return { ...common(result), orchestration: null };
}

function adaptV2(result) {
  return { ...common(result), orchestration: result.orchestration || null };
}

export const benchmarkAdapters = new Map([[1, adaptV1], [2, adaptV2]]);

export function adaptBenchmarkResult(result) {
  if (!result || typeof result !== "object") return { kind: "missing" };
  const adapter = benchmarkAdapters.get(result.schema_version);
  if (!adapter) return { kind: "unsupported", schemaVersion: result.schema_version };
  return { kind: "supported", value: adapter(result) };
}
