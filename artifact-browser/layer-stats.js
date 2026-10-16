// Compiled MLA timings from one component's profiling pass, separate from headline latency.
export function analyzeLayers(rows) {
  if (!Array.isArray(rows) || !rows.length) return null;
  const layers = [];
  const stages = new Map();
  let totalUs = 0;
  for (const [index, row] of rows.entries()) {
    if (!row || typeof row.name !== "string" || !row.name
      || !Number.isSafeInteger(row.stage_index) || row.stage_index < 0
      || !Number.isFinite(row.total_execution_time_us) || row.total_execution_time_us < 0) return null;
    const layer = { order: index + 1, name: row.name, stage: row.stage_index, timeUs: row.total_execution_time_us };
    layers.push(layer);
    totalUs += layer.timeUs;
    const stage = stages.get(layer.stage) || { index: layer.stage, timeUs: 0, count: 0 };
    stage.timeUs += layer.timeUs;
    stage.count += 1;
    stages.set(layer.stage, stage);
  }
  if (!Number.isFinite(totalUs)) return null;
  return { layers, totalUs, stages: [...stages.values()].sort((a, b) => a.index - b.index) };
}

export function selectLayers(analysis, { query = "", order = "time" } = {}) {
  const selected = analysis.layers;
  const totalUs = selected.reduce((sum, layer) => sum + layer.timeUs, 0);
  const ranked = [...selected].sort((a, b) => b.timeUs - a.timeUs || a.order - b.order);
  const top10Us = ranked.slice(0, 10).reduce((sum, layer) => sum + layer.timeUs, 0);
  const search = String(query).trim().toLowerCase();
  const layers = (order === "compiler" ? selected : ranked).filter((layer) => layer.name.toLowerCase().includes(search));
  return {
    layers, totalUs, count: selected.length,
    zeroCount: selected.filter((layer) => layer.timeUs === 0).length,
    top10Share: totalUs > 0 ? top10Us / totalUs * 100 : 0,
  };
}

// One fixed microsecond axis for the component, even when a search hides its slowest layer.
export function layerTimeScale(analysis) {
  const maximum = analysis.layers.reduce((value, layer) => Math.max(value, layer.timeUs), 0);
  if (!maximum) return 1;
  const step = 10 ** Math.floor(Math.log10(maximum));
  const rounded = Math.ceil(maximum / step) * step;
  return Number.isFinite(rounded) ? rounded : maximum;
}
