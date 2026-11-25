---
name: neat-compile-model
description: Compile an ONNX or PyTorch model into an MPK for a SiMa Modalix DevKit with the Model Compiler. Use to audit an ONNX graph's operators against MLA support, rewrite unsupported operators or detection outputs for BoxDecode, optimize a compatible graph when performance work is requested, quantize and compile with calibration, or lay out a reproducible model recipe the way the Models repository does. Not for compiling an LLM or VLM, which LLiMa compiles into a deployment folder rather than an MPK (use `sima-llima-compile-run`).
---

# Neat Compile Model

To compile an LLM or VLM, use the `sima-llima-compile-run` skill instead.

Run every command from this skill's folder, so the `scripts/` paths resolve,
and give model and build paths as absolute paths outside it. Take the steps in
order and stop where the request ends. An audit ends at step 2 and changes
nothing. A graph rewrite ends at step 3 or 4 with its step 6 evidence. A compile
runs to step 6.

To reproduce a model published in the Models repository, or to lay out your own
recipe its way, read [model recipe](references/model-recipe.md). It maps these
steps to `convert.py`, `surgery.py`, `calibration.py`, and `compile.py`.

## Steps

1. **Locate the Model Compiler environment.** In the Neat SDK, run
   `activate-model-compiler`. If it is unavailable, install the package for the
   host architecture with `sima-cli neat install model-compiler/<amd64|arm64>`.
   The unqualified target installs the latest published main artifact. Append
   `@develop` only to test the current develop compiler, or a release tag to pin
   a release. Done when
   `python3 -c 'import afe; print("Model Compiler Python APIs available")'`
   prints that sentence.

2. **Audit the operators.** This step is a required gate. Compile nothing
   until it is done.
   - Fetch an ONNX file that is not local with
     `python3 scripts/download_onnx.py --url <url> --output-dir <dir>`, or with
     `--hf-repo <org/repo> --hf-file <file>` for Hugging Face.
   - Inspect the source model, its static shapes, dtypes, opset, public tensor
     contract, and any compiler logs. A tensor contract is each input and
     output's name, shape, order, dtype, layout, and score domain.
   - Freeze symbolic dimensions before screening:

     ```bash
     python3 scripts/onnx_static_simplify.py \
       --input /abs/path/model.onnx \
       --output /abs/path/model.static.sim.onnx \
       --replace batch=1
     ```

   - Audit with the precision the build will use. If the target is unspecified,
     assume Modalix, and screen both `int8` and `bfloat16` when that difference
     matters to the request:

     ```bash
     python3 scripts/model_surgery_guard.py audit-model \
       --model /abs/path/model.static.sim.onnx \
       --dtype int8
     python3 scripts/model_surgery_guard.py query-op --op <op_type>
     python3 scripts/model_surgery_guard.py list-supported --dtype bfloat16
     ```

   - The audit checks operator names and precision flags. It cannot establish
     that attributes, shapes, opsets, composite fusion, fallback, or placement
     satisfy every MLA constraint. Treat it as preliminary screening, and claim
     no MLA placement from it.
   - For a PyTorch model without an exported ONNX graph, the audit does not
     apply. Claim no compatibility before AFE import and compiler placement have
     been inspected.
   - When outputs will feed SiMa BoxDecode, read
     [BoxDecode contracts](references/boxdecode.md) before you change the graph.

   [Operator database](references/operator_database.md) explains the audit's
   groups, exit codes, and entry fields. [Composite patterns](references/composite_patterns.md)
   covers multi-node patterns, such as RMSNorm, that the audit cannot see. Done
   when the audit has run on the static graph, every unsupported operator and
   every supported operator with constraints has its entry checked against the
   graph, and every unknown operator and open placement or fusion question is
   written down.

3. **Repair incompatibilities.** Run this step when the audit reports an
   unsupported or unknown operator, when an entry's constraint fails for the
   graph, or when outputs must match a BoxDecode contract.
   - Use a rewrite from [compatibility patterns](references/compatibility_patterns.md)
     only after the database shows that the original operator form cannot run on
     MLA. Keep a natively supported operator unless a separate, measured
     optimization justifies changing it.
   - Prefer a maintainable source-model change. Otherwise make the smallest safe
     ONNX edit.
   - Preserve public tensor names, shapes, ordering, dtype, layout, and score
     domain unless the requested contract changes them on purpose. A changed
     output is an API change, so report the new output contract.

   Done when the changed graph has been audited again and step 6 holds its
   equivalence evidence.

4. **Optimize a compatible graph.** Run this step only when performance work is
   requested or compiler evidence identifies a relevant bottleneck. A request to
   compile does not call for it.
   - Name the requested metric or the bottleneck. Keep a baseline when the
     request needs a measured improvement.
   - Read [optimization patterns](references/optimization_patterns.md), and
     apply only patterns whose match conditions and guards are proven for the
     graph. Make the change as step 3 does, and preserve the same contracts.
   - Full MLA placement does not prove that a graph is optimally represented.
   - Rely on the current compiler layout, tessellation, and any-shape defaults
     unless a reviewed pattern requires otherwise.
   - A mathematically proven graph-only rewrite needs no compile to apply.
     Confirming placement or a hardware-dependent benefit needs one.

   Done when each applied pattern names its proven match and guards, the graph
   has been audited again, and step 6 holds its equivalence evidence. A measured
   request also holds its baseline and resulting measurements.

5. **Quantize and compile.** For an ONNX model with FP32 4D image inputs, run:

   ```bash
   python3 scripts/quantize_compile.py \
     --model_path /abs/path/model.onnx \
     --model_format onnx \
     --build_dir /abs/path/build
   ```

   For an INT8 build, add
   `--real_data --dataset_images <folder> --num_calib_samples 20`, because
   without `--real_data` the script calibrates on random values, which only
   shows that the model compiles.

   [Quantize and compile](references/quantize_compile.md) covers PyTorch, dynamic
   shapes, calibration images, precision, QAT QDQ exports, memory errors, and the
   import API for other inputs. Done when `<build_dir>/<model_basename>/` holds
   the compiled `*_mpk.tar.gz` archive and `quantization_manifest.json`.

6. **Verify.** Run each check that applies:
   - For a rewrite, run the ONNX checker and shape inference, then compare the
     original and changed model on representative inputs. Report the numerical
     tolerance for an equivalent rewrite, or a task-level metric for an
     intentional output or math change.
   - For a compile, `--verify` compares the quantized and floating-point model on
     one random input. That shows numerical agreement on that input, not task
     accuracy, and calibration data proves nothing about quality either.
   - For BoxDecode outputs, treat only the compiled model pack's `*_mpk.json` as
     authoritative for tensor order, layout, dtype, quantization, and resize
     metadata. Then run the compiled model through BoxDecode as
     [BoxDecode contracts](references/boxdecode.md) check 6 describes.
   - Measure accuracy and speed on a DevKit with the `neat-benchmark-model`
     skill.

   Done when the report gives each check that ran with its result, and labels as
   unconfirmed every unverified accuracy, unresolved compatibility constraint,
   fallback placement, and unmeasured performance claim. A latency, throughput,
   or memory gain that depends on compiler scheduling or hardware behavior stays
   unconfirmed until it is measured, whatever the graph structure suggests.
