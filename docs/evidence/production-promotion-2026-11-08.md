# Production promotion acceptance — November 8–9, 2026

This records Models #574 separately from the completed Vulcan deployment.
Initial acceptance uses Models Develop `c136509f790f19cb998f5e53a81d969d3201c160`
and Vulcan deployment `0be72e6fa2cf94aeae7cad45893b997fe193e746`.

## Environment and deployed gate

[Promotion run 42573200689](https://github.com/sima-neat/models/actions/runs/42573200689)
resolved `VULCAN_ENV: production`. Its saved `promotion-plan` also pins:

- writer: `https://artifacts-registry.neat.sima.ai`;
- reader: `https://models-registry.neat.sima.ai`;
- target: `refs/heads/develop`.

The configure and publisher-credential prepare jobs succeeded. The protected
`production-promote-develop` environment has owners-models review, self-review
prevention, no administrator bypass and a Develop-only branch rule. The deployed
role and variable are recorded in the
[deployment comment](https://github.com/sima-neat/models/issues/574#issuecomment-6733634981).

## MPK acceptance attempt

The run reused finalized production candidate
`0dcd86c6-222d-4dca-a67f-939f87348d87`, YOLO26x Pose `w8a16`, with manifest digest
`41986555db358d3e9a76a9fa631307c245eced9e00f3432853e088c6cee26678`.
The request failed with HTTP 422 `candidate_incompatible`:
`Target must declare this on-demand model`. Approval and reporting were skipped.
This did not establish a pending promotion or successful promotion. The
[correlated preparation issue](https://github.com/sima-neat/models/issues/593)
is a plan, not a promoted receipt.

YOLO26x Pose is now an automatic build on Develop and has no on-demand target
requirements. This eligibility rejection precedes package validation and is not
the old ZIP-validation failure. Do not change its permanent source policy merely
to pass acceptance. The scoped operator fixture in
[`tests/fixtures/production_promotion`](https://github.com/sima-neat/models/tree/develop/tests/fixtures/production_promotion)
retains the candidate's genuine compile identities and requires guarded restoration.
A same-revision CAS attempt to stage that metadata was rejected with HTTP 409
`catalog_plan_conflict: Model metadata cannot change within a revision`.
No target metadata changed and no fixture approval exists. Activation needs a
separate reviewed fixture revision with matching documentation provenance;
ordinary progress publication cannot alter source metadata.

## Positive MPK validator and artifact-download fixture

The read-only `tests/fixtures/production_promotion/verify_mpk.py` fixture downloaded
the deployed production writer ZIP and verified its AWS code hash:
`Xv60uirpiRD7jcRTNMEHPDhxbSIpMk5WE4EzWD+Jfxo=`. It loaded the deployed Python
validator source with local native dependencies, using the genuine frozen
candidate and original target requirements against actual production S3 objects.
No validation checks were mocked. Both compile and package roots selected as
`layout: archive, format: mpk`; their exact registered S3 versions were verified.

The production run-artifact API for package run
`93d636bc-8c76-4e7e-98fc-c86c25484651` returned an exact-version download URL.
The complete downloaded package matched **60,864,167 bytes** and SHA-256
`680d659c6ef6776a26bd29ddbb4b823c901d9e4d823bab7dc97f2d3ca0ce9fa5`.
The compile root on run `8a5f273f-ce64-4f11-9b33-c4f9ea4980a6` has the same
content identity, as required for a single-MPK package. The downloaded gzip/tar
layout was also parsed successfully (4 members), without extracting executable
content.

This is a local replay of verified deployed source and a real production artifact
download. It did not invoke the writer handler, create a pending request, approve
anything or change a catalog. Full protected MPK promotion remains distinct from
this successful package-format regression check.

## Negative authorization and selection checks

A publisher-role request to the approval route using a nonexistent test UUID
returned HTTP 403. It could not exercise approver permissions. After the failed
MPK workflow and rejected metadata fixture attempt, the Develop snapshot remained
`af84e29f1dc195d921e376f1a90fa8a803246ccd1c43ca4934f223477844bc5e`,
YOLO26x Pose still had no approved selection, and its candidate retained only the
original `latest_successful` hold, with no `promotion_pending` hold.

## LLiMa acceptance

Production inventory initially contained two candidate upload prefixes: the
YOLO26x Pose candidate above and RAFT Stereo
`303036ab-895a-4139-bb77-2db5b8ac5991`. Signed candidate inspection reported both
finalized. Neither was an eligible Develop promotion target. The production
compile-run searches for Gemma, Qwen and Whisper returned no compiled candidates;
LLiMa metadata in the catalog did not establish a deployable candidate.

With owner authorization, started
[Whisper Small production build 42573456789](https://github.com/sima-neat/models/actions/runs/42573456789)
on Develop, selecting only `whisper_small/w8a16`, with `force_rebuild_all=false`.
Its declared source, compiler, runner class and profile are unchanged. The run
completed successfully: 50 compiler units and the shared deployment-manifest
verifier passed. Model-specific compile tests were explicitly skipped by
`tests.yaml`; no runtime transcription qualification is claimed.

The finalized production candidate is `10309d98-4a81-4ed3-b374-65628c2e1ec9`,
manifest digest `b6810a5d2a542e21ba8a08724638f54d62565fffcc2dafffb2117822e4bf1250`.
It binds compile run `be3f8569-f813-4612-82b5-7dc38bf375f1` and package run
`d688eb30-2f68-4c2e-9797-ac9de047b766`. Each contains the same 57 runtime
files totaling 944,513,603 bytes. The compile manifest SHA-256 is
`43927105ef0ed032254ad120bf9a6e3e543cc9d09ff639b70fb519c5d0773e7e`;
the package manifest SHA-256 is
`cee10932ac054e3fe8be02bb9e760d629f306e64d3a042ee6e7e70c9c7557aa3`.

### Protected review and stale-snapshot recovery

[Original promotion run 42575819430](https://github.com/sima-neat/models/actions/runs/42575819430)
created request `ec23b2ae-7352-43f2-803d-aad48604070d` against Develop snapshot
`af84e29f1dc195d921e376f1a90fa8a803246ccd1c43ca4934f223477844bc5e`.
Owner `florianvoss-commit` approved the protected environment and the gated job
successfully assumed `NeatProductionModelsRegistryPromotionRole`. The decision
then correctly returned HTTP 409 `catalog_snapshot_conflict`: the catalog had
advanced after merging the workflow implementation. No selection changed.
[Cancellation run 42585602178](https://github.com/sima-neat/models/actions/runs/42585602178)
later confirmed the stale request as `cancelled`, with pending protection released,
after a replacement request had acquired its own protection.

### Successful production Develop promotion — November 9

[Refreshed promotion run 42585435494](https://github.com/sima-neat/models/actions/runs/42585435494)
completed successfully using trusted Models Develop revision
`18e6e9e612515567648e85bf8d055abea152e5c4` and deployed Vulcan revision
`022fc9f940e77e4e0396483169ba4614b26f2691`. Its saved plan selects production
and Develop. It reused the same finalized candidate, manifest digest, compile
run and package run listed above; promotion did not compile or copy payloads.

The replacement request `6026e9a9-6962-4f1d-a516-bec60aa8ef89` pinned snapshot
`e3c7a15f8df65465d6444b9144a339420f1abd7b0d028268b415f205f0769424`.
GitHub records independent protected approval by `florianvoss-commit` for
`production-promote-develop`. The backend decision artifact reports:

- state: `promoted`;
- decision time: `2026-11-09 08:26:46.939512 UTC`;
- actor: `arn:aws:sts::069171140053:assumed-role/NeatProductionModelsRegistryPromotionRole/models-promotion-approve-42585435494-1`;
- result snapshot: `4da71bbfe9405280669d12341a5cb5f0432617e515a483a6c6283f82625aa53a`.

The public production Develop catalog returned that exact snapshot and selected
candidate `10309d98-4a81-4ed3-b374-65628c2e1ec9` with its unchanged manifest digest.
Both the compile root and default package project as `layout: directory,
format: llima`, bound to the original runs above. No review or credential bypass
was used.

### Selected-catalog download

The normal installed customer CLI downloaded through the production Develop
selection into a separate fresh destination:

```sh
SIMA_MODELS_REGISTRY_BASE_URL=https://models-registry.neat.sima.ai \
  sima-cli models download --branch develop --id whisper_small --variant w8a16 \
  --output /tmp/models-574-native-selected-download --json
```

The command succeeded and reported `downloaded_files: 57`, `reused_files: 0`,
`file_count: 57`, `format: directory`, `size_bytes: 944513603`, package run
`d688eb30-2f68-4c2e-9797-ac9de047b766`, and manifest SHA-256
`cee10932ac054e3fe8be02bb9e760d629f306e64d3a042ee6e7e70c9c7557aa3`.
An independent pass hashed every downloaded runtime file against the immutable
candidate's package artifact metadata and verified all 57 relative paths, sizes
and SHA-256 values, totaling 944,513,603 bytes. The native deployment layout,
including the `elf_files/` files and adjacent reproduction scripts, is preserved;
there is no deployment ZIP.

This completes production Develop LLiMa promotion and selected-download
acceptance. It does not establish runtime transcription quality, DevKit
qualification, Main approval or Hugging Face publication.

## Completion criteria

LLiMa acceptance is complete. MPK acceptance remains incomplete until an eligible
production candidate completes protected promotion, a backend `promoted` receipt,
result snapshot, correct catalog selection and verified download. The MPK
read-only fixture does not substitute for that path. Retain the original
stale-snapshot failure as negative evidence and its explicit cancellation.
Cleanup remains disabled in production; cancel/reject abandoned pending requests
before expiry and handle expired holds through controlled cleanup.
