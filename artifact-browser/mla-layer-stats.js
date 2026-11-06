// Core's MLA layer stats: compiled MLA timings from one component's profiling pass, separate from headline latency.
export function analyzeLayers(rows) {
  if (!Array.isArray(rows) || !rows.length) return null;
  const layers = [];
  const stages = new Map();
  let totalUs = 0;
  for (const [index, row] of rows.entries()) {
    if (!row || typeof row.name !== "string" || !row.name
      || !Number.isSafeInteger(row.stage_index) || row.stage_index < 0
      || !Number.isFinite(row.latency_us) || row.latency_us < 0) return null;
    const layer = { order: index + 1, name: row.name, stage: row.stage_index, timeUs: row.latency_us };
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

export function selectLayers(analysis, { query = "", order = "time", minUs = 0 } = {}) {
  const selected = analysis.layers;
  const totalUs = selected.reduce((sum, layer) => sum + layer.timeUs, 0);
  const ranked = [...selected].sort((a, b) => b.timeUs - a.timeUs || a.order - b.order);
  const top10Us = ranked.slice(0, 10).reduce((sum, layer) => sum + layer.timeUs, 0);
  const search = String(query).trim().toLowerCase();
  const layers = (order === "compiler" ? selected : ranked).filter((layer) => layer.name.toLowerCase().includes(search) && layer.timeUs >= minUs);
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

export const LAYER_FAMILIES = {
  conv: "Convolution", matmul: "Matmul / Einsum", activation: "Activation",
  pool: "Pooling", elementwise: "Elementwise", layout: "Tensor layout",
  normalization: "Normalization", resize: "Resize", reduction: "Reduction",
  io: "Input / output", other: "Other",
};

// AFE names Relay/composite operations; N2A adds vertices for segmentation and outputs.
// Generated slice/concat names include producer/consumer names after the operation prefix.
// Keep those helpers separate from the referenced computation. These remain name-based
// display families: the profiling report does not carry the compiler's operator type.
export function layerFamily(name) {
  const text = name.toLowerCase().replace(/^mla_\d+\//, "");
  // large_tensor_helper.py creates these before the consumer and can reuse them.
  if (/^(?:slice_|concat_|batch_concat_)/.test(text) || text.includes("_batch_slice_from_")) return "layout";
  if (/^(?:placeholder|input|output)(?:_|$)/.test(text)
    || /_output(?:_(?:batch_)?segment_\d+)?$/.test(text)) return "io";
  if (/_output_reshape(?:_|\/|$)/.test(text)) return "layout";

  // Parent-first matching keeps decomposed reshapes together and supports
  // scoped names (block_0/conv2d, etc.).
  for (const component of text.split("/")) {
    const operation = component.replace(/^(?:nn\.|qnn[._])/, "");
    if (/^(?:group_)?conv(?:[123]d)?(?:transpose)?(?:_|$)/.test(operation)) return "conv";
    if (/^(?:matmul|batch_matmul|dense|einsum|einsum_with_repetition|tuple_einsum|single_input_tuple_einsum)(?:_|$)/.test(operation)) return "matmul";
    if (/^(?:pad_|global_)?(?:avg|max)?_?pool(?:[123]d)?(?:_|$)/.test(operation)) return "pool";
    if (/^(?:softmax|log_softmax|layer_norm|layernorm|rms_norm|rmsnorm|instance_norm|instancenorm|batch_norm|l2_norm|lrn)(?:_|$)/.test(operation)) return "normalization";
    // add_relu/add_clip are AddActivationOp, rather than standalone activations.
    if (/^(?:add|subtract|multiply|mul|divide|maximum|minimum|constant_multiply_add|power|exp|log|log2|log10|sqrt|rsqrt|reciprocal|sin|cos|erf)(?:_|$)/.test(operation)) return "elementwise";
    if (/^(?:relu|leaky_relu|prelu|swish|hard_swish|sigmoid|hard_sigmoid|gelu|quick_gelu|elu|selu|tanh|softplus|clip|activation)(?:_|$)/.test(operation)) return "activation";
    if (/^(?:reshape|transpose|strided_slice|slice|slice_concat|tuple_concat|concatenate|concat|split|squeeze|expand_dims|flatten|batch_flatten|broadcast_to|pad|depth_to_space|tile|take|gather|copy|channel_permute|tessellate|detessellate|layout_transform)(?:_|$)/.test(operation)) return "layout";
    if (/^(?:image\.resize[123]d|resize|upsampling)(?:_|$)/.test(operation)) return "resize";
    if (/^(?:reduce_(?:sum|max|min|mean|prod)|sum|max|min|mean|prod|variance|argmax|argmin|topk)(?:_|$)/.test(operation)) return "reduction";
  }
  return "other";
}
