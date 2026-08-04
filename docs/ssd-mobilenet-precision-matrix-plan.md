# SSD-MobileNet precision and tessellation matrix

## Goal

Represent, compile, validate, and package the four meaningful Modalix contracts
for every SSD-MobileNet family member supported by the pinned Model Compiler:

| Precision | Input/output tessellation |
| --- | --- |
| INT8 | MLA |
| INT8 | EV74 (off MLA) |
| BF16 | MLA |
| BF16 | EV74 (off MLA) |

The matrix describes real compiler contracts, not requested labels. A cell is
publishable only after its precision, boundary transport, tessellation
placement, quantization structure, numeric quality, and MPK provenance have all
been audited. A compiler/model limitation remains visible as an unsupported
cell with reproducible evidence instead of being silently omitted or packaged.

## Reusable implementation

1. Declare the four variants in each model's `model.yaml`. Keep precision and
   tessellation orthogonal so adding a new model does not require workflow or
   packaging special cases.
2. Resolve the compiler version and runner label once from `deps/manifest.json`.
   The workflow builds its matrix from catalog metadata and accepts optional
   model and variant filters for focused qualification runs.
3. Keep model-specific graph surgery and compiler calls in the model directory.
   Shared discovery, selection, provenance, MPK verification, packaging, and
   publication stay in `tools/model_registry`.
4. Exclude known-unsupported cells from normal builds, but fail a targeted
   request with the stored reason when none of the selected models supports the
   requested variant. Reject stale compiled output for an unsupported cell.

## Quantization contract

- Use exactly the immutable, checksum-pinned calibration asset declared by the
  model.
- Use at least 128 representative COCO images for calibration. Validate the
  tensor dtype, range, shape, sample count, content digest, and image IDs before
  calling the compiler.
- Keep a disjoint held-out set of 32 representative images for numeric checks;
  calibration samples must never double as qualification samples.
- INT8 uses asymmetric per-tensor 8-bit activations, symmetric per-channel
  8-bit weights, and SiMa requantization. BF16 uses BF16 activations and weights.
- Audit the serialized graph for the expected convolution count, quantized
  status, scalar type, and weight representation. Treat compiler warnings that
  report zeroed weights as structural failures.
- Compare every raw SSD box and class head against the floating model on every
  held-out sample. Gate minimum cosine similarity and maximum normalized RMSE
  with thresholds derived from successful compiler measurements and record the
  complete report in the model audit.

## Tessellation and boundary contract

- `tess_mla` supplies explicit MLA input and output tessellation parameters;
  `tess_off_mla` leaves tessellation and detessellation on EV74.
- Public inputs and outputs remain FP32 because Model Compiler 2.1.3 does not
  expose the newer public BF16/FP32 boundary APIs. Internal compute precision is
  verified independently from public transport precision.
- The MPK audit must prove the expected MLA plugin count, boundary transforms,
  transport tensor types and sizes, tessellation stages, compiler version,
  source digest, and variant identity.

## Qualification and CI

1. Unit-test catalog/schema behavior, calibration validation, surgery, precision
   audits, tessellation placement, quality gates, and packaging rejection paths.
2. Run formatting/linting and the complete Python test suite locally.
3. Run each newly supported cell on the pinned Vulcan compiler runner. Retain
   its audit and validated MPK artifact; a compile-only log is insufficient.
4. Run the complete supported catalog matrix from the final commit, then verify
   packaging and publication jobs. Pull-request discovery remains on hosted
   runners so untrusted PR code never reaches a self-hosted compiler runner.
5. Keep decoded COCO accuracy and on-device throughput/latency as separate
   hardware qualification gates. Do not infer those results from raw-head
   compiler similarity or compilation success.

## Acceptance criteria

- Every model declares all four cells.
- Every supported cell uses at least 128 verified representative calibration
  images and passes its held-out numeric and structural audits.
- Unsupported cells state an exact compiler/model reason and cannot be selected
  or packaged as supported.
- The final PR commit passes the full local suite, PR sanitization, catalog
  discovery, all supported Vulcan matrix jobs, MPK verification, and packaging.
- Documentation distinguishes compiler qualification from decoded accuracy and
  hardware performance without claiming measurements that were not run.
