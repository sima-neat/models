# Production promotion acceptance fixture

`mpk.json` pins the genuine production YOLO26x Pose candidate from the original
failed promotion. The artifact and compile provenance remain unchanged. The
proposed target fixture would temporarily restore only that profile's original on-demand target
requirements; it does not change model definitions, compile artifacts, or the
source-controlled automatic-build policy.

This is an operator fixture, not a model qualification or a general publication
workflow. Apply its guards in order. Do not publish while another Develop catalog
publication is in flight. Save the baseline outside Git; it may contain unrelated
catalog data. Keep the temporary policy in place only for the reviewed test window.

A promoted selection is server-owned. The ordinary publication API cannot
withdraw it or disable its on-demand model. Cleanup therefore requires the
repository lock and a database transaction scoped to the exact fixture promotion
UUID, candidate, target, model and profile. Require exactly one matching approval,
preserve the promotion history, audit the withdrawal, and release only the matching
`approved_selection` hold. Then restore the captured policy through the CAS snapshot
API. Do not delete candidates, runs or artifact objects. If the required operator
cleanup path or reviewer is unavailable, prepare the fixture without activating it.

The same-revision snapshot API rejects this metadata patch with
`catalog_plan_conflict: Model metadata cannot change within a revision`.
Activation requires a separate reviewed fixture Git revision and documentation
runs bound to that revision. Do not bypass that guard with a direct catalog
edit. The manifest is prepared; it has not activated an on-demand target.

## Read-only deployed-validator replay

`verify_mpk.py` is the executed read-only fixture. It downloads the deployed
writer package, verifies its AWS code hash, loads only its Python source with
local native dependencies, and validates the frozen candidate against the real
production S3 versions. It then downloads the package through the production
run-artifact API and checks every byte against the registered hash and size.
It neither invokes the writer handler nor accesses a database, changes a catalog,
approves a selection, or recompiles a model.

Use an operator Python environment with boto3, psycopg and Models' dependencies,
export a valid production `saml2aws` session, and save the candidate JSON from a
signed production candidate-inspection request. From the repository root:

```sh
AWS_REGION=us-west-2 ARTIFACT_BUCKET=<production artifact bucket> \
  PRODUCTION_AWS_ACCOUNT_ID=<account> PYTHONPATH=tools \
  python tests/fixtures/production_promotion/verify_mpk.py \
  --candidate /absolute/path/candidate.json --output-dir /absolute/path/verification
```

The JSON result explicitly records `live_protected_promotion: false`. Do not
report this source replay and artifact download as a live handler invocation,
protected approval or Develop catalog download acceptance.
