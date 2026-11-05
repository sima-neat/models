# LeWAM PushT — staged registration

This is a source implementation under review, **not a published or qualified model**.
There is no active catalog entry, approved deployment, or transferred performance
score. Four exported graphs require four immutable source pins; the registry's
existing file-set source and grouped compilation handle them without a new
publication or promotion lifecycle.

## Prepare source publication

From the repository, with its Python dependencies available:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/publication.py prepare \
  --graph front=/absolute/path/front.onnx \
  --graph body=/absolute/path/body.onnx \
  --graph head=/absolute/path/head.onnx \
  --graph policy=/absolute/path/policy.onnx \
  --output /absolute/path/new-publication-input
```

The new directory contains `<component>/model.onnx` for each component and one
`publication-input.json`. Preparation validates the self-contained ONNX files
and fixed interfaces, hashes them, and copies their original bytes. The receipt's
checkpoint/source fields identify the intended source; they do **not** prove
weight lineage or numerical equivalence. Keep these large files outside Git.
Use a new output directory; existing frozen campaign artifacts remain unchanged.

### Trusted publisher handoff — still required

The existing `.github/workflows/vulcan-model-convert.yml` requires an active
model's `convert.py`, runs conversion tests, and publishes one `model.onnx`.
It cannot consume this prepared directory through its current dispatch inputs.
These helpers neither upload files nor implement that missing CI handoff.

A maintainer must arrange a reviewed, trusted CI invocation of the existing
**Publish ONNX and print source pin** step once per component:

1. Admit the prepared files through the trusted CI artifact handoff; verify their
   hashes and sizes against `publication-input.json` before granting publication.
2. For each component, set `MODEL_ID=lewam_pusht` and point `EXPORT_DIR` at that
   component's directory, so the unchanged step reads `$EXPORT_DIR/model.onnx`.
   Use the actual CI run/attempt and reviewed source/config provenance. Preserve
   the existing publisher role, conditional S3 write, KMS encryption, checksum,
   and download verification. Do not execute contributor-controlled code in a
   privileged PR job or bypass normal review.
3. Retain each returned `$EXPORT_DIR/source.yaml` separately. The content hash in
   the existing object key distinguishes the components.

No privileged workflow changes or publication are included here. After the
trusted publisher has returned all four real pins:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/publication.py bind \
  --prepared /absolute/path/new-publication-input \
  --pin front=/absolute/path/front-source.yaml \
  --pin body=/absolute/path/body-source.yaml \
  --pin head=/absolute/path/head-source.yaml \
  --pin policy=/absolute/path/policy-source.yaml \
  --cache /absolute/path/cache \
  --output /absolute/path/compilation-source.yaml
```

Binding revalidates the prepared graphs, checks each publisher pin against its
original bytes, and uses the shared registry Fetcher to verify all four downloads.
It writes an ordinary `compilation.source` YAML fragment with
`main_file: front.onnx` and four named files. A missing, mismatched, or failed download leaves
no fragment; existing output is never replaced. There is no second published
receipt or automatic catalog promotion. Merge the verified fragment into the
model template as a reviewed `model.yaml` only after the release gates below.

## Data preparation

The original 150-row PushT source fixture pack has a separate helper:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/stage_data.py \
  --evaluation-pack /absolute/path/original-evaluation-pack \
  --output /absolute/path/new-evaluation-input
```

It verifies the original files and produces a deterministic source archive and
`evaluation-stage.json`. This is **not** held-out data, calibration, or a complete
runtime evaluation bundle. Publish and adapt it through Dataset Registry; do not
register it as `pusht/cal`.

The real calibration archive for `pusht/cal` requires `calibration.json`, the
original `selection.json`, and four `<component>.npz` files. Its schema is
`lewam.calibration.v1`, source is `source_trajectories`, and checkpoint hash is the
pinned original checkpoint. It records disjoint nonempty `episode_ids` and
`evaluation_episode_ids`, plus each component file's hash. The loader checks the
evaluation exclusion against the hash-pinned original selection. Each NPZ has
finite FP32 NCHW `inputs` and one integer episode ID per input. Do not substitute
synthetic samples, repeated rows, guessed episode IDs, or evaluation data. Add
the real Dataset Registry pin to the reviewed definition; do not invent one.

## Implementation and remaining release gates

The staged implementation includes the four-component model template, graph
validation, grouped compilation, BF16 activations/weights with both FP32
normalization options disabled, hash-checked calibration loading, and a
public-PyNeat baseline.
The baseline uses four runners, full19 packing, eight own-feedback updates, all
ten outputs, separate FP32 **host** arithmetic, bounded requests, and explicit
runner cleanup. Accuracy and whole-replan scripts remain undeclared until their
data and measurement protocols are qualified.

The following gates remain open:

1. **Compiler and precision.** Reproduce the front's proven normalization lowering
   through the standard compiler and compile all four graphs using
   `deps/manifest.json`, not the old local compiler. Audit the actual lowered
   operators before a heavy build. BF16 configuration is not final-ELF precision
   proof; retain real calibration until the pinned compiler's requirements are
   established. Verify source equivalence and absence of FP32 MLA work in the
   exact final executables. Archive/interface checks and the shared diagnostic
   fidelity metrics cannot replace this proof. The candidate-only compile stop
   remains until an implemented, independently checked qualification gate exists;
   a successful host test is not authorization to publish compile outputs.
2. **Native execution and dependencies.** Freshly admit a DevKit and qualify the
   public-Neat interfaces, preprocessing, completion, layout, numerical results,
   and cleanup. Resolve the complete ARM dependency set from the recorded
   environment, including torch, stable-worldmodel, pygame, and pymunk where used.
   Four top-level pins are insufficient: stable-worldmodel eagerly imports more
   dependencies, and the shared benchmark installer uses
   `--no-deps --only-binary=:all:`. The qualified ARM evaluator dependency closure
   and bootstrap are still missing. Host availability does not demonstrate target
   installation or compatibility.
3. **Accuracy data and protocol.** Publish the registered evaluation/request
   bundles. The `lewam.episodes.v1` loader requires recorded reset seeds, package
   versions, arrays, and the original byte-pinned 150-row non-held-out selection.
   Its `initial_images` array must contain the recorded RGB224 uint8 observations,
   not post-reset renders; the evaluator also retains them for step-five history.
   Add `requirements-accuracy.txt` only from the verified dependency closure, not
   guessed package pins. Extend the environment receipt beyond its current four
   packages to bind the rendering stack, including OpenCV and NumPy, then validate
   reset/render/RNG scheduling against the upstream direct-policy evaluator.
   Staging the original source pack does not supply these missing fields.
4. **Performance protocol and baselines.** The script measures RGB224 to ten
   actions, including Neat normalization—not the registry's canonical 720p image
   protocol. Agree and implement the action-model protocol in the shared
   documentation/runner/workflow/tests before enabling performance. Set finite
   expectations only from actual published develop reports; preserve the shared
   10% throughput/5% accuracy limits. If an enabled stage has no baseline, use the
   normal explicit `run_tests: false` and `skip_reason`, never guessed numbers.
5. **Application.** Finish and qualify the separate Apps example. Do not embed a
   private runner or runtime DSO as a package resource.

The historical 15.393 image-ready replans/s and 12/14 numerical result describe a
different runtime, not this public-Neat implementation. No new FPS, accuracy, or
release PASS is claimed.

## Host checks

```bash
PYTHONPATH=tools python -m pytest tests/test_lewam_*.py
PYTHONPATH=tools python -m model_registry validate-models --repository .
```

Host tests exercise graph serialization, source binding, archive/interface checks,
and mathematical code; native/compiler/publication test doubles are not native
qualification. The shared metric vocabulary affects benchmark fingerprints:
adding the PushT metric can schedule existing models' benchmark stages, without
changing their ONNX compile fingerprints. Do not bypass that policy or transfer
historical scores to the new runtime.
