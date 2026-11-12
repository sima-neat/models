// Metric definitions and value formatting. Each performance metric matches tools/model_report/metrics.json,
// whose `display` unit and scale it shows.
export const METRIC_DEFINITIONS = {
  ttft_ms: { label: "Time to first token", unit: "ms", digits: 1, scale: 1, direction: "lower" },
  prefill_tps: { label: "Prefill throughput", unit: "tokens/s", digits: 1, scale: 1, direction: "higher" },
  decode_tps: { label: "Decode throughput", unit: "tokens/s", digits: 1, scale: 1, direction: "higher" },
  draft_contribution_pct: { label: "Draft contribution", unit: "%", digits: 1, scale: 1, direction: "higher" },
  decode_speedup: { label: "Decode speedup", unit: "×", digits: 2, scale: 1, direction: "higher" },
  latency_avg_ms: { label: "Average latency", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  latency_p50_ms: { label: "Latency p50", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  latency_p95_ms: { label: "Latency p95", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  latency_p99_ms: { label: "Latency p99", unit: "ms", digits: 2, scale: 1, direction: "lower" },
  throughput_fps: { label: "Throughput", unit: "inferences/s", digits: 1, scale: 1, direction: "higher" },
  avg_power_watts: { label: "SoM power", unit: "W", digits: 2, scale: 1, direction: "lower" },
  peak_power_watts: { label: "Peak SoM power", unit: "W", digits: 2, scale: 1, direction: "lower" },
  avg_mla_power_watts: { label: "MLA power", unit: "W", digits: 2, scale: 1, direction: "lower" },
  peak_mla_power_watts: { label: "Peak MLA power", unit: "W", digits: 2, scale: 1, direction: "lower" },
  energy_per_inference_joules: {
    label: "Energy / inference", unit: "mJ/inference", digits: 2, scale: 1000, direction: "lower",
  },
  mla_memory_peak_mib: { label: "Peak MLA memory", unit: "MiB", digits: 1, scale: 1, direction: "lower" },
  mla_memory_avg_mib: { label: "MLA memory", unit: "MiB", digits: 1, scale: 1, direction: "lower" },
  ev74_memory_peak_mib: { label: "Peak EV74 memory", unit: "MiB", digits: 1, scale: 1, direction: "lower" },
  ev74_memory_avg_mib: { label: "EV74 memory", unit: "MiB", digits: 1, scale: 1, direction: "lower" },
  avg_linux_memory_mib: { label: "Board Linux memory used", unit: "MiB", digits: 0, scale: 1, direction: "neutral" },
  peak_linux_memory_mib: { label: "Peak board Linux memory used", unit: "MiB", digits: 0, scale: 1, direction: "neutral" },
  avg_die_temperature_c: { label: "SoC temperature", unit: "°C", digits: 1, scale: 1, direction: "lower" },
  peak_die_temperature_c: { label: "Peak SoC temperature", unit: "°C", digits: 1, scale: 1, direction: "lower" },
  avg_board_temperature_c: { label: "SoM temperature", unit: "°C", digits: 1, scale: 1, direction: "lower" },
  peak_board_temperature_c: { label: "Peak SoM temperature", unit: "°C", digits: 1, scale: 1, direction: "lower" },
  avg_cpu_usage_pct: { label: "CPU usage, all cores", unit: "%", digits: 1, scale: 1, direction: "lower" },
  peak_cpu_usage_pct: { label: "Peak CPU usage, all cores", unit: "%", digits: 1, scale: 1, direction: "lower" },
  avg_cpu_core_usage_pct: { label: "CPU usage, busiest core", unit: "%", digits: 1, scale: 1, direction: "lower" },
  peak_cpu_core_usage_pct: { label: "Peak CPU usage, busiest core", unit: "%", digits: 1, scale: 1, direction: "lower" },
};

function fallbackMetricLabel(key) {
  const words = String(key).replace(/[._/-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Measurement";
}

export function metricDefinition(key) {
  return METRIC_DEFINITIONS[key] || {
    label: fallbackMetricLabel(key),
    unit: "",
    digits: 2,
    scale: 1,
    direction: "neutral",
  };
}

export function metricDisplay(metric, definition = metricDefinition(metric?.key || "")) {
  if (!metric || metric.status !== "measured" || !Number.isFinite(metric.value)) return null;
  return {
    value: metric.value * definition.scale,
    unit: definition.unit,
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
