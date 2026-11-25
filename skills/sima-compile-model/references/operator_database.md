# Operator Support Database

Read this when you interpret an `audit-model` finding or an entry that
`query-op` prints.

- `data/supported_operators.json` is the sole operator-support source the
  guard uses. `--supported-ops-json <path>` points the guard at another copy.
- Keys match canonical ONNX `node.op_type` names. Multi-node MLA patterns, such
  as RMSNorm through opset 22, are in [composite patterns](composite_patterns.md)
  instead.
- Each entry records hardware and software constraints in
  `sima_hw_sw_constraints`, 5D support in `fived`, and precision support in
  `int8` and `bfloat16`. The BF16 entries apply to Modalix.
- The guard counts an operator as supported for a dtype when that flag is `Y`.
  `--dtype any` accepts either flag.
- A `Y` flag does not settle the attribute, shape, opset, dtype,
  constant-input, and fallback conditions. Read every field `query-op` prints
  for the operator before you rely on the entry.

## Audit results

`audit-model` sorts each operator type in the graph into one of three groups:

| Group | Meaning |
| --- | --- |
| supported | The entry's flag for the requested dtype is `Y`. |
| unsupported | The entry exists, and its flag is not `Y`. The audit prints its constraints and notes. |
| unknown | The database has no entry for the operator. |

The audit prints the database's `release`, the operator counts, and the
unsupported and unknown operators. `--json` prints the full report, including
each supported operator's constraints. It exits with 2 when any operator is
unsupported or unknown, 0 when every operator is supported, and 1 on an error
such as a missing model or the missing `onnx` package.
