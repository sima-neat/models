// Metric definitions and value formatting.
export const METRIC_DEFINITIONS = {
  latency_avg_ms: { label: "Average latency", unit: "ms", digits: 2, scale: 1, direction: "lower", summary: true },
  latency_p50_ms: { label: "Latency p50", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  latency_p95_ms: { label: "Latency p95", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  throughput_fps: { label: "Throughput", unit: "fps", digits: 1, scale: 1, direction: "higher", summary: true },
  avg_power_watts: { label: "Board power", unit: "W", digits: 2, scale: 1, direction: "lower", summary: true },
  energy_joules: { label: "Benchmark energy", unit: "J", digits: 2, scale: 1, direction: "lower" },
  energy_per_inference_joules: {
    label: "Energy / inference", unit: "mJ/inference", digits: 2, scale: 1000, direction: "lower", summary: true,
  },
  model_load_ms: { label: "Model load", unit: "ms", digits: 1, scale: 1, direction: "lower" },
  time_to_first_inference_ms: {
    label: "Time to first inference", unit: "ms", digits: 1, scale: 1, direction: "lower", summary: true,
  },
  mla_memory_baseline_mib: { label: "MLA memory baseline", unit: "MiB", digits: 1, scale: 1, direction: "neutral" },
  mla_memory_peak_mib: { label: "MLA memory peak", unit: "MiB", digits: 1, scale: 1, direction: "lower" },
  mla_memory_peak_delta_mib: {
    label: "Peak MLA memory", unit: "MiB", digits: 1, scale: 1, direction: "lower", summary: true,
  },
  mla_memory_final_mib: { label: "MLA memory final", unit: "MiB", digits: 1, scale: 1, direction: "neutral" },
};
export const SUMMARY_METRICS = Object.entries(METRIC_DEFINITIONS)
  .filter(([, definition]) => definition.summary)
  .map(([key]) => key);

function fallbackMetricLabel(key) {
  const words = String(key).replace(/[._/-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Measurement";
}

export function metricDefinition(key, metric = null) {
  if (key === "throughput_fps" && metric?.unit) {
    return { ...METRIC_DEFINITIONS[key], unit: metric.unit };
  }
  return METRIC_DEFINITIONS[key] || {
    label: fallbackMetricLabel(key),
    unit: metric?.unit || "",
    digits: 2,
    scale: 1,
    direction: "neutral",
  };
}

export function metricDisplay(metric, definition = metricDefinition(metric?.key || "", metric)) {
  if (!metric || metric.status !== "measured" || !Number.isFinite(metric.value)) return null;
  return {
    value: metric.value * definition.scale,
    unit: definition.scale === 1 ? (metric.unit || definition.unit) : definition.unit,
    digits: definition.digits,
  };
}

export function formatDate(value, includeTime = true) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", year: date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    ...(includeTime ? { hour: "numeric", minute: "2-digit" } : {}),
  }).format(date);
}

export function formatDuration(start, end) {
  if (!start || !end) return "—";
  const seconds = Math.round((new Date(end) - new Date(start)) / 1000);
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds >= 3600) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m ${seconds % 60}s`;
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}
