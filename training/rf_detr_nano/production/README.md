# RF-DETR Nano QAT: staged production handoff

This directory implements local qualification using the production Nano's
two-component and report conventions. **It is not an active model definition,
a deployable source export, or a production-ready release.** Stock
`models/rf_detr_nano` is unchanged. Source hosting, download wiring and catalog
registration are deferred at the user's request. No GPU training starts here.

## What is implemented

- `prepare_contract.py` verifies a caller-pinned artifact manifest, every listed
  file checksum and the source/output-grid binding, then writes a new local
  handoff. Historical absolute paths are not used to find files.
- `compile.py` accepts two independently hash-bound, trusted, already materialized
  native `.sima` checkpoints. It requires the matching compiler's numerical-seal
  verification, preserves the captured graph identities, performs no calibration
  or bias correction, compiles one component at a time and writes conventional
  `compile-outputs.json`. It rejects unsupported compilers instead of importing
  the diagnostic ONNX or overlaying compiler source. Local module versions are
  recorded; the production compiler pin still requires the standard runner.
- `runtime.py` requests explicit FLOAT32 rescale, antialiased resize, ordered
  normalization and divide/RNE INT8 quantization through public PyNeat. No legacy
  affine preprocessing, reciprocal substitution or FLOAT32 component handoff is
  accepted. Missing runtime support fails clearly. Actual hardware support and
  numerical behavior still need qualification.
- The boundary adapters preserve the captured INT8 input order, source grids,
  lower-index proposal ties and `MaskMinRepeatZero` exhaustion. The decoder
  consumes selected boxes before encoder memory and returns deltas, class logits
  and box references. The approved parameter-free CPU formula decodes final boxes.
- `accuracy.py` uses that image route, raw COCO category IDs, maxDets `[1,10,100]`
  and the pinned official annotations. All 5,000 images are the default; `--limit`
  is explicitly a local screen. Reports identify the source checkpoint and both
  packages. A screen is not full validation.
- `performance.py` follows the current Nano's seeded 720p image, bounded
  depth-four pipeline, 100-frame warmup, minimum 1,000-frame/60-second measurement,
  telemetry, per-component layer/stage statistics and execution-plan reports.
  Final box decoding is inside the timer, unlike stock's recorded throughput.

**The Python handoff and decoder are explicit NumPy CPU oracles. They are not the
optimized native TopK/Gather or BoxDecode kernels.** The feature handoff uses the
existing safe `.cvu()` path; zero-copy parity or 240 FPS is not claimed. Native
kernel integration remains required before a performance-ready release.

## Local commands

Use verified local files from one frozen candidate, not a mixture of source
weights, old qparams and newer packages. The manifest checksum must come from an
independent retained receipt. `SHA256SUMS` alone is not an authenticity source.

```bash
python prepare_contract.py --artifacts "$ARTIFACT_DIR" \
  --manifest-sha256 "$MANIFEST_SHA256" --output handoff.json
```

Save the printed `contract_sha256` independently. On an identity-checked DevKit
with the matching PyNeat/Core/Kernels stack and verified packages:

```bash
PYTHONPATH=/path/to/models/tools python accuracy.py \
  --mpk backbone="$BACKBONE" --mpk decoder="$DECODER" \
  --contract handoff.json --contract-sha256 "$CONTRACT_SHA256" \
  --dataset coco-2017-val="$COCO_VAL" --output accuracy-full5000

PYTHONPATH=/path/to/models/tools python performance.py \
  --mpk backbone="$BACKBONE" --mpk decoder="$DECODER" \
  --contract handoff.json --contract-sha256 "$CONTRACT_SHA256" \
  --dataset coco-2017-val="$COCO_VAL" --output performance
```

Install the same `requirements-accuracy.txt` as stock Nano for the scorer, plus
the matching public PyNeat runtime. Never run an ARM test executable on an x86
SDK host: inspect it with `file` and use the established guarded DevKit workflow.
These commands do not acquire device ownership or authorize firmware changes.

To compile *trusted, already materialized* native checkpoint files in a matching
compiler environment, use both independent file checksums:

```bash
python compile.py --model "$REFERENCE_ONNX" \
  --native-model backbone="$NATIVE_BACKBONE" --native-model decoder="$NATIVE_DECODER" \
  --native-sha256 backbone="$BACKBONE_NATIVE_SHA" --native-sha256 decoder="$DECODER_NATIVE_SHA" \
  --contract handoff.json --contract-sha256 "$CONTRACT_SHA256" \
  --build-profile w8a8_mixed --compile-invocation w8a8_mixed --output-dir new-compile
```

This is **not** the missing native-materialization step. A new compile needs a
new package binding; the old handoff's package hashes must not be reused. The
staged compiler script does not claim source parity, precision audit, hardware
accuracy or performance merely because compilation finishes.

## Why registration remains blocked

The available accurate ONNX explicitly declares
`NONDEPLOYABLE_ORDERED_NATIVE_ATTENTION_DIAGNOSTIC_REFERENCE` and
`sima_qat.compiler_eligible=false`; it also exposes a validity guard. The companion
QAT export-inspection utility preserves those declarations and rejects this graph
as a direct compiler input. Existing successful packages came from a separately
materialized native graph and a campaign-specific compiler/frontend cohort.

The following are still required, not bypassed by these scripts:

1. Supported, reproducible native-program materialization from the pinned trained
   checkpoint, or a faithful compiler-ready ONNX export. No arbitrary source
   patching or public `compiler_eligible` flag flip substitutes for this.
2. A verified compile in the production Model Compiler dependency, currently
   3.0.0.1, and the matching public typed-IO/ordered-preprocessing runtime.
3. Optimized native CPU bridge/decoder integration, buffer-lifetime qualification
   and fresh end-to-end timing. The existing packed-arena offsets are not copied
   into these customer-facing scripts.
4. New PyTorch/ONNX/native/packaged-ELF comparisons and complete board COCO/FPS
   results for the *same* candidate and image path.
5. Source publication/wiring when requested, followed by a normal authored
   `models/rf_detr_nano_qat` definition, conversion, surgery, stage tests, README
   and hero. Do not add an active entry with placeholder URLs or invented scores.

The recorded source AP 46.9819 and fused-image AP 46.6548 remain below the
one-point gap target against FP32 48.4077. `acceptance.py` retains that absolute
gate, plus a 0.1-point source/compiled budget. These budgets do not accumulate and
do not replace the registry's shared regression checks. Historical results are
not new results for this implementation.
