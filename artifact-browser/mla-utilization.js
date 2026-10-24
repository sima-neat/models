// MLA utilization: each MLA stage's latency on the board (Core's MLA stage stats in the model report) set against
// the operations the Model Compiler's simulator counts for it (mla-stage-simulation.json, which the package run
// publishes as a test result). Both number a component's stages as Core does, so the stage index pairs them.

// The MLA's peak rate at INT8. Its multiplier takes INT8 or BF16 operands, and a BF16 multiply takes four times the
// cycles of an INT8 one (the Gen2 ISA's MMUL latency), so a component with BF16 activations peaks at a quarter of it.
export const MLA_PEAK_INT8_TOPS = 50;
export const MLA_PEAK_BF16_TOPS = MLA_PEAK_INT8_TOPS / 4;

// The peak a component's precision runs at: its activations set the multiplier's operands, so w8a16 and w4a16 run
// at the BF16 rate too. Null when the precision is unknown.
export function mlaPeakTops(precision) {
  if (/a8$/.test(precision || "")) return { tops: MLA_PEAK_INT8_TOPS, datatype: "INT8" };
  if (/a16$/.test(precision || "")) return { tops: MLA_PEAK_BF16_TOPS, datatype: "BF16" };
  return null;
}

// The simulation document the package run's test results carry, or null.
export function mlaStageSimulation(tests = []) {
  const result = tests.find((item) =>
    item.suite === "performance" && item.name === "mla-stage-simulation" && item.summary?.schema_version === 1);
  return result?.summary || null;
}

// Each MLA stage with its measured latency and its simulation, or why the two cannot be compared.
export function matchMlaStages(measured, simulated) {
  if (measured?.status !== "ok") return { reason: `Not measured: ${measured?.status || "no MLA stage stats"}` };
  if (simulated?.status !== "ok") return { reason: `Not simulated: ${simulated?.status || "no MLA stage simulation"}` };
  const indices = (stages) => stages.map((stage) => stage.stage_index).join(",");
  if (!measured.stages.length || indices(measured.stages) !== indices(simulated.stages)) {
    return { reason: "The board and the simulator report different MLA stages." };
  }
  return {
    stages: measured.stages.map((stage, index) => ({
      index: stage.stage_index,
      measuredUs: stage.latency_us,
      simulatedUs: simulated.stages[index].simulated_latency_us,
      effectiveOps: simulated.stages[index].effective_ops,
      totalOps: simulated.stages[index].total_ops,
    })),
  };
}

// The effective TOPS a model sustains at its measured throughput. Throughput counts inferences, and each MLA run
// covers one batch of them.
export function sustainedEffectiveTops(total, throughputFps, batch) {
  return total.effectiveOps * throughputFps / batch / 1e12;
}

// Rates in TOPS from the measured latency: the effective rate counts only operations that contribute to the result,
// packing is the share of the MLA's executed operations that do, and utilization is the effective rate as a share
// of the peak.
export function mlaUtilizationMetrics(stages, { peakTops = null } = {}) {
  const metrics = (stage) => {
    const effectiveTops = stage.measuredUs > 0 ? stage.effectiveOps / stage.measuredUs / 1e6 : null;
    return {
      ...stage,
      effectiveTops,
      packing: stage.totalOps > 0 ? stage.effectiveOps / stage.totalOps : null,
      utilization: peakTops && effectiveTops !== null ? effectiveTops / peakTops : null,
    };
  };
  const sum = (field) => stages.reduce((total, stage) => total + stage[field], 0);
  return {
    stages: stages.map(metrics),
    total: metrics({
      measuredUs: sum("measuredUs"), simulatedUs: sum("simulatedUs"),
      effectiveOps: sum("effectiveOps"), totalOps: sum("totalOps"),
    }),
  };
}
