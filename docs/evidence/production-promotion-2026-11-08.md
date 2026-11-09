# Production promotion acceptance — November 8, 2026

This records Models #574 separately from the completed Vulcan deployment.
Acceptance uses Models Develop `c136509f790f19cb998f5e53a81d969d3201c160`
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
content identity, as required for a single-MPK package.

This is a local replay of verified deployed source and a real production artifact
download. It did not invoke the writer handler, create a pending request, approve
anything or change a catalog. Full protected MPK promotion remains distinct from
this successful package-format regression check.

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
Its declared source, compiler, runner class and profile are unchanged. Build,
protected promotion, catalog selection and checksum-verified download results
will be recorded here as they complete. No runtime transcription qualification
is claimed by this registry acceptance test.

## Completion criteria

Acceptance remains incomplete until both package layouts have a backend
`promoted` receipt, result snapshot, correct catalog selection and verified
downloads. Record the reviewer gate and role assumption. Preserve the failed
request as negative evidence and verify it did not create an approved selection.
Cleanup remains disabled in production; cancel/reject abandoned pending requests
before expiry and handle expired holds through controlled cleanup.
