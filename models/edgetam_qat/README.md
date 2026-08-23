# EdgeTAM pure-W8A8 QAT

This directory stages the qualified EdgeTAM prompt, recurrent, and memory
components for registry onboarding. All three components use INT8 weights and
activations with INT32 accumulation. The compiled artifacts contain no BF16,
FP16, FP32, INT16/A16, or A65 fallback. Each component invocation contains one
MLA stage and one MLA ELF.

The qualification meets the requested 1.5-point accuracy bound:

- source-QAT J&F drops 1.3781 points from matched BF16 across 49 observations;
- Modalix prompt selected J drops 1.2755 points from matched BF16 across seven
  labelled records;
- prompt and recurrent board outputs are byte-exact to AFE.

Download the complete, content-bound reproduction bundle from the
[`edgetam-pure-w8a8-r160-v1` prerelease](https://github.com/sima-neat/models/releases/tag/edgetam-pure-w8a8-r160-v1).
Because this build required Model Compiler changes, the same release provides
the exact
[`edgetam_model_compiler_overlay_r160_20260823.tar.gz`](https://github.com/sima-neat/models/releases/download/edgetam-pure-w8a8-r160-v1/edgetam_model_compiler_overlay_r160_20260823.tar.gz)
overlay separately. Its SHA-256 is
`c2c5f17e587a5e85e402433e6829f7824d5f70f7dc5ee037d3e239ac26e12a44`.
It is an internal reproduction workaround, not a supported general-purpose
Model Compiler release.

The release also attaches the three MPKs and standalone MLA ELFs separately.
Exact asset URLs, hashes, sizes, board metrics, and evidence limitations are
recorded in `qualification.json`, `artifacts/COMPILER_OVERLAY.json`, and
`artifacts/TERMINAL_PURE_W8A8_ACCEPTANCE.json`.

A `model.yaml` is intentionally not present yet. Standard registry compilation
cannot reproduce this three-component artifact until the packaged Model
Compiler overlay is upstreamed and the registry has a supported multi-component
recipe. Use the qualified MPKs now; complete the blockers in
`qualification.json` before treating this directory as a standard buildable
registry entry.
