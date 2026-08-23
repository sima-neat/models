# OSTrack ViT-B no-CE QAT

This directory stages the accepted Modalix `r57` qualification for registry
onboarding. The deployment is **mixed precision**, not full INT8: representable
backbone QAT paths remain quantized, while semantic attention and the 15
QAT-dequantized score/size/offset head affines execute in BF16.

The content-bound artifact and terminal silicon receipt are listed in
`artifacts/SHA256SUMS`; qualification metrics and the single-MLA-ELF contract are
recorded in `qualification.json`.

The qualified MPK is published as the prerelease asset
[`ostrack_r57_bf16io_fp32cast_mpk.tar.gz`](https://api.github.com/repos/sima-neat/models/releases/assets/523827871).

A `model.yaml` is intentionally not present yet. Publishing the qualified MPK
makes the accepted board artifact available, but the immutable compiler input
still needs its own registry-accessible source URI. Do not substitute the MPK
for the ONNX source or invent a placeholder URL. Complete the remaining
publication blockers in `qualification.json`, then add and validate the normal
registry entry and build recipe.
